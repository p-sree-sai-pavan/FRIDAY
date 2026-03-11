"""
core/orchestrator.py
=====================
FRIDAY Orchestrator — Unified Tool-Call Pipeline

All prompts follow one path:
  1. Classify prompt → pick model (never FAST_MODEL when tools exist)
  2. First call with tools available (tool_choice="auto")
     - Plain text response → faithfulness check → save to memory, return
     - Tool call(s) detected → switch to TOOLS_MODEL for execution loop
  3. Tool execution loop (TOOLS_MODEL, max 6 iterations):
     - safety check each tool call
     - execute approved tools (parallel if multiple)
     - feed results back to LLM
     - repeat until plain text final response
  4. Save final response + extract facts to memory
  5. Gemini fallback if Groq fails entirely
"""

import asyncio
import json
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

# Import tools package to trigger tool registration (tools/__init__.py auto-discovers tools)
import tools  # noqa: F401 — side-effect import

from memory import groq_client, read as memory_read, write as memory_write, compress_results, write_semantic
from core.brain import _load_system_prompt, _load_history, _save_history, _parse_response, _faithfulness_check

log = logging.getLogger("orchestrator")

MAX_TOOL_ITERATIONS = 12

# Queries that must always use PRIMARY_MODEL — small models ignore injected context
MEMORY_TRIGGERS = (
    "what do you know", "do you remember", "what did i",
    "who am i", "tell me about me", "what have i", "my details"
)

# Track background tasks to prevent garbage collection
_background_tasks: set = set()


# ========================
# MODEL SELECTOR
# Rules:
#   1. Tools registered → always PRIMARY_MODEL (FAST_MODEL cannot call tools)
#   2. Memory/personal query → always PRIMARY_MODEL (small models ignore context)
#   3. Otherwise → classify with FAST_MODEL, use result
# ========================

async def _select_model(prompt: str, has_tools: bool) -> str:
    # Rule 1 — tools need a capable model
    if has_tools:
        log.info("[Orchestrator] Tools registered — using PRIMARY_MODEL")
        return config.PRIMARY_MODEL

    # Rule 2 — memory queries need a model that follows injected context
    if any(t in prompt.lower() for t in MEMORY_TRIGGERS):
        log.info("[Orchestrator] Memory query — using PRIMARY_MODEL")
        return config.PRIMARY_MODEL

    # Rule 3 — safe to classify for tool-free simple queries
    try:
        resp = await groq_client.chat.completions.create(
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
# Runs silently after every response.
# Extracts personal facts about Pavan and writes them to facts.db.
# ========================

async def _extract_facts(prompt: str, response: str):
    try:
        resp = await groq_client.chat.completions.create(
            model=config.FAST_MODEL,
            temperature=0,
            max_tokens=200,
            response_format={"type": "json_object"},
            messages=[{
                "role": "user",
                "content": (
                    "Extract factual information about Pavan from this conversation exchange. "
                    "Only extract clear explicit facts stated by the user — not assumptions. "
                    "Return JSON: {\"facts\": [{\"category\": \"...\", \"key\": \"...\", \"value\": \"...\"}]} "
                    "or {\"facts\": []} if nothing factual was stated.\n\n"
                    f"User: {prompt}\nFRIDAY: {response}"
                )
            }]
        )
        data  = json.loads(resp.choices[0].message.content)
        facts = data.get("facts", [])
        for fact in facts:
            if all(k in fact for k in ("category", "key", "value")):
                await write_semantic(fact["category"], fact["key"], fact["value"])
    except Exception as e:
        log.warning(f"[Orchestrator] Fact extraction failed: {e}")


def _launch_background(coro):
    """Launch a background coroutine safely — stores task ref to prevent GC."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


# ========================
# CONFIRMATION PROMPT
# ========================

async def _ask_confirmation(tool_name: str, arguments: dict, session) -> bool:
    args_display = ", ".join(f"{k}={v}" for k, v in arguments.items())
    prompt_text  = f"\n[FRIDAY] About to run: {tool_name}({args_display})\nAllow? (y/n): "
    answer = await session.prompt_async(prompt_text)
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

async def _tool_loop(messages: list, tools: list, session) -> str:
    for iteration in range(MAX_TOOL_ITERATIONS):
        try:
            resp = await groq_client.chat.completions.create(
                model=config.TOOLS_MODEL,
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                parallel_tool_calls=True
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
    return "I ran into a problem completing that — it required too many steps. Please try breaking it into smaller requests."


# ========================
# MAIN ENTRY POINT
# ========================

async def orchestrate(prompt: str, session) -> dict:
    from tools.registry import registry

    system_prompt = _load_system_prompt()
    history       = _load_history()

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

    tools      = registry.to_groq_tools()
    final_text = ""

    # Select model — never FAST_MODEL when tools are registered
    model = await _select_model(prompt, has_tools=bool(tools))

    try:
        resp = await groq_client.chat.completions.create(
            model=model,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
            messages=messages,
            tools=tools if tools else None,
            tool_choice="auto" if tools else None
        )

        message = resp.choices[0].message

        if not message.tool_calls:
            log.info("[Orchestrator] Direct response — no tools.")
            result     = _parse_response(message.content or "")
            final_text = result.get("response", "")

        else:
            log.info(f"[Orchestrator] {len(message.tool_calls)} tool call(s) — switching to TOOLS_MODEL.")

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

            final_text = await _tool_loop(messages, tools, session)

    except Exception as e:
        log.error(f"[Orchestrator] Groq error: {e}")
        try:
            from google import genai
            client = genai.Client(api_key=config.GEMINI_API_KEY)
            resp   = client.models.generate_content(model=config.FALLBACK_MODEL, contents=prompt)
            result = _parse_response(resp.text)
            final_text = result.get("response", "")
            log.info("[Orchestrator] Gemini fallback succeeded.")
        except Exception as e2:
            log.error(f"[Orchestrator] Gemini fallback error: {e2}")
            final_text = "Both primary and fallback models failed."

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

    # Background — extract personal facts (safe: task ref is stored)
    if final_text:
        _launch_background(_extract_facts(prompt, final_text))

    log.info(f"[Orchestrator] Done | Q: {prompt[:50]} | A: {final_text[:50]}")
    return {"response": final_text, "command": None}