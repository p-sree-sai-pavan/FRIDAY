"""
core/orchestrator.py
=====================
FRIDAY Orchestrator — Unified Tool-Call Pipeline

All prompts follow one path:
  1. Classify prompt → pick model
  2. First call with tools available
     - Plain text response → faithfulness check → save to memory, return
     - Tool call(s) detected → switch to TOOLS_MODEL for execution loop
  3. Tool execution loop (max 12 iterations):
     - safety check each tool call
     - execute approved tools (parallel if multiple)
     - feed results back to LLM
     - repeat until plain text final response
  4. Save final response + extract facts to memory
  5. Gemini fallback (online mode only) if Groq fails entirely

The orchestrator is mode-agnostic — it always calls get_client() from
model_client.py which returns the correct provider for the current mode.
"""

import asyncio
import json
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

import tools  # noqa: F401 — side-effect import triggers tool registration

from memory import read as memory_read, write as memory_write, compress_results, write_semantic
from core.brain import _load_system_prompt, _load_history, _save_history, _parse_response, _faithfulness_check
from core.model_client import get_client, get_fallback_client

log = logging.getLogger("orchestrator")

MAX_TOOL_ITERATIONS = 12

MEMORY_TRIGGERS = (
    "what do you know", "do you remember", "what did i",
    "who am i", "tell me about me", "what have i", "my details"
)

_background_tasks: set = set()


# ========================
# MODEL SELECTOR
# ========================

async def _select_model(prompt: str, has_tools: bool) -> str:
    """
    Pick primary vs fast model based on prompt complexity.
    Rules:
      1. Tools registered → always PRIMARY (FAST can't call tools reliably)
      2. Memory/personal query → always PRIMARY (small models ignore injected context)
      3. Otherwise → ask FAST to classify, use result
    """
    if has_tools:
        log.info("[Orchestrator] Tools present — using PRIMARY_MODEL")
        return config.PRIMARY_MODEL

    if any(t in prompt.lower() for t in MEMORY_TRIGGERS):
        log.info("[Orchestrator] Memory query — using PRIMARY_MODEL")
        return config.PRIMARY_MODEL

    try:
        client = get_client()
        resp = await client.chat.completions.create(
            model=config.FAST_MODEL,
            temperature=0,
            max_tokens=5,
            messages=[{
                "role": "user",
                "content": (
                    "Classify this request. Reply with ONE word only:\n"
                    "simple   → greetings, time, basic facts\n"
                    "complex  → coding, research, analysis, multi-step tasks\n\n"
                    f"Request: {prompt}"
                )
            }]
        )
        word   = resp.choices[0].message.content.strip().lower()
        chosen = config.FAST_MODEL if "simple" in word else config.PRIMARY_MODEL
        log.info(f"[Orchestrator] Model selected: {chosen} (classified: {word})")
        return chosen
    except Exception as e:
        log.warning(f"[Orchestrator] Model selection failed: {e} — using PRIMARY_MODEL")
        return config.PRIMARY_MODEL


# ========================
# BACKGROUND FACT EXTRACTION
# ========================

async def _extract_facts(prompt: str, response: str):
    """Extract personal facts about Pavan and persist them to facts.db."""
    try:
        client = get_client()
        # response_format=json_object is Groq-native; Ollama supports it on some
        # models but not all. We include it only in online mode and always wrap
        # json.loads in a try/except so a plain-text reply doesn't crash anything.
        extra = {"response_format": {"type": "json_object"}} if config.MODEL_MODE == "online" else {}
        resp = await client.chat.completions.create(
            model=config.FAST_MODEL,
            temperature=0,
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": (
                    "Extract factual information about Pavan from this conversation exchange. "
                    "Only extract clear explicit facts stated by the user — not assumptions. "
                    "Return JSON: {\"facts\": [{\"category\": \"...\", \"key\": \"...\", \"value\": \"...\"}]} "
                    "or {\"facts\": []} if nothing factual was stated.\n\n"
                    f"User: {prompt}\nFRIDAY: {response}"
                )
            }],
            **extra
        )
        raw  = resp.choices[0].message.content or ""
        # Strip markdown fences if model wrapped response anyway
        raw  = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        data  = json.loads(raw)
        facts = data.get("facts", [])
        for fact in facts:
            if all(k in fact for k in ("category", "key", "value")):
                await write_semantic(fact["category"], fact["key"], fact["value"])
    except Exception as e:
        log.warning(f"[Orchestrator] Fact extraction failed: {e}")


def _launch_background(coro):
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


# ========================
# CONFIRMATION PROMPT
# ========================

async def _ask_confirmation(tool_name: str, arguments: dict, session) -> bool:
    args_display = ", ".join(f"{k}={v}" for k, v in arguments.items())
    answer = await session.prompt_async(
        f"\n[FRIDAY] About to run: {tool_name}({args_display})\nAllow? (y/n): "
    )
    return answer.strip().lower() in ("y", "yes")


# ========================
# TOOL EXECUTION
# ========================

async def _execute_tool(tool_name: str, arguments: dict, session) -> str:
    from tools.registry import registry, execute_tool_call
    from core.safety import check as safety_check, Decision

    tool = registry.get(tool_name)
    if tool is None:
        return f"Error: tool '{tool_name}' not found."

    decision = await safety_check(tool, arguments)

    if decision == Decision.DENY:
        log.warning(f"[Orchestrator] DENIED: {tool_name}")
        return f"Action '{tool_name}' was denied by safety policy."

    if decision == Decision.ASK:
        confirmed = await _ask_confirmation(tool_name, arguments, session)
        if not confirmed:
            return f"Pavan chose not to run: {tool_name}."

    try:
        result = await execute_tool_call(tool_name, arguments)
        return str(result)
    except Exception as e:
        log.error(f"[Orchestrator] Tool error ({tool_name}): {e}")
        return f"Error running {tool_name}: {e}"


# ========================
# TOOL-CALL LOOP
# ========================

async def _tool_loop(messages: list, tool_schemas: list, session) -> str:
    """
    Continue calling the LLM + executing tools until it returns a plain text response.
    Caps at MAX_TOOL_ITERATIONS to prevent infinite loops.
    """
    client = get_client()

    for iteration in range(MAX_TOOL_ITERATIONS):
        try:
            resp = await client.chat.completions.create(
                model=config.TOOLS_MODEL,
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS,
                messages=messages,
                tools=tool_schemas,
                tool_choice="auto",
                # parallel_tool_calls is Groq-specific — Ollama's OpenAI-compatible
                # endpoint does not support it and returns a 400. Only send it online.
                **({ "parallel_tool_calls": True } if config.MODEL_MODE == "online" else {})
            )
        except Exception as e:
            log.error(f"[Orchestrator] Tool loop API error (iteration {iteration}): {e}")
            return f"An error occurred during tool execution: {e}"

        message = resp.choices[0].message

        if not message.tool_calls:
            log.info(f"[Orchestrator] Tool loop done after {iteration + 1} iteration(s).")
            return message.content or ""

        messages.append({
            "role":       "assistant",
            "content":    message.content or "",
            "tool_calls": [
                {
                    "id":       tc.id,
                    "type":     "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                }
                for tc in message.tool_calls
            ]
        })

        tasks = []
        for tc in message.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            tasks.append(_execute_tool(tc.function.name, args, session))

        results = await asyncio.gather(*tasks)

        for tc, result in zip(message.tool_calls, results):
            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result
            })

    log.warning("[Orchestrator] Max tool iterations reached.")
    return "I ran into a problem completing that — it required too many steps. Try breaking it into smaller requests."


# ========================
# MAIN ENTRY POINT
# ========================

async def orchestrate(prompt: str, session) -> dict:
    from tools.registry import registry

    system_prompt  = _load_system_prompt()
    history        = _load_history()
    memory_results = await memory_read(prompt)
    context        = compress_results(memory_results)

    full_system = system_prompt
    if context:
        full_system = (
            f"{system_prompt}\n\n"
            f"--- RELEVANT MEMORY ---\n{context}\n--- END MEMORY ---"
        )

    messages = [{"role": "system", "content": full_system}]
    messages += history
    messages.append({"role": "user", "content": prompt})

    tool_schemas = registry.to_groq_tools()
    final_text   = ""
    client       = get_client()

    model = await _select_model(prompt, has_tools=bool(tool_schemas))

    try:
        resp = await client.chat.completions.create(
            model=model,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
            messages=messages,
            tools=tool_schemas if tool_schemas else None,
            tool_choice="auto" if tool_schemas else None
        )

        message = resp.choices[0].message

        if not message.tool_calls:
            log.info("[Orchestrator] Direct response — no tools.")
            final_text = _parse_response(message.content or "").get("response", "")

        else:
            log.info(f"[Orchestrator] {len(message.tool_calls)} tool call(s) — entering tool loop.")

            messages.append({
                "role":       "assistant",
                "content":    message.content or "",
                "tool_calls": [
                    {
                        "id":       tc.id,
                        "type":     "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                    }
                    for tc in message.tool_calls
                ]
            })

            tasks = []
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tasks.append(_execute_tool(tc.function.name, args, session))

            results = await asyncio.gather(*tasks)

            for tc, result in zip(message.tool_calls, results):
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      result
                })

            final_text = await _tool_loop(messages, tool_schemas, session)

    except Exception as e:
        log.error(f"[Orchestrator] Primary client error: {e}")

        # Gemini fallback — online mode only
        fallback = get_fallback_client()
        if fallback:
            try:
                resp       = fallback.models.generate_content(model=config.FALLBACK_MODEL, contents=prompt)
                final_text = _parse_response(resp.text).get("response", "")
                log.info("[Orchestrator] Gemini fallback succeeded.")
            except Exception as e2:
                log.error(f"[Orchestrator] Gemini fallback error: {e2}")
                final_text = "Both primary and fallback models failed."
        else:
            if config.MODEL_MODE == "offline":
                final_text = "Ollama is not responding. Make sure it is running: ollama serve"
            else:
                final_text = "Primary model failed and no fallback is configured."

    # Save conversation history
    history.append({"role": "user",      "content": prompt})
    history.append({"role": "assistant", "content": final_text})
    _save_history(history)

    # Faithfulness-gated episodic memory write
    if final_text:
        if context:
            score = await _faithfulness_check(prompt, final_text, context)
            if score >= 0.5:
                await memory_write(prompt, final_text)
            else:
                log.warning(f"[Orchestrator] Faithfulness {score:.2f} — skipping episodic write.")
        else:
            await memory_write(prompt, final_text)

    if final_text:
        _launch_background(_extract_facts(prompt, final_text))

    log.info(f"[Orchestrator] Done | Q: {prompt[:50]} | A: {final_text[:50]}")
    return {"response": final_text, "command": None}
