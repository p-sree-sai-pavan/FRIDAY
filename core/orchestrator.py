"""
core/orchestrator.py
=====================
FRIDAY Orchestrator — Unified Tool-Call Pipeline

All prompts follow one path:
  1. Load system prompt + memory context
  2. First call → PRIMARY_MODEL with tools available (tool_choice="auto")
     - Plain text response → save to memory, return
     - Tool call(s) detected → switch to TOOLS_MODEL for execution loop
  3. Tool execution loop (TOOLS_MODEL, max 6 iterations):
     - safety check each tool call
     - execute approved tools (parallel if multiple)
     - feed results back to LLM
     - repeat until plain text final response
  4. Save final response to memory, return to Pavan
  5. Gemini fallback if Groq fails entirely
"""

import asyncio
import json
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from memory.manager import groq_client, read as memory_read, write as memory_write, compress_results
from core.brain import _load_system_prompt, _load_history, _save_history, _parse_response

log = logging.getLogger("orchestrator")

MAX_TOOL_ITERATIONS = 6


# ========================
# CONFIRMATION PROMPT
# ========================

async def _ask_confirmation(tool_name: str, arguments: dict) -> bool:
    """Print what FRIDAY is about to do and wait for Pavan's y/n."""
    args_display = ", ".join(f"{k}={v}" for k, v in arguments.items())
    prompt_text  = f"\n[FRIDAY] About to run: {tool_name}({args_display})\nAllow? (y/n): "
    answer = await asyncio.to_thread(lambda: input(prompt_text).strip().lower())
    return answer in ("y", "yes")


# ========================
# TOOL EXECUTION
# ========================

async def _execute_tool(tool_name: str, arguments: dict) -> str:
    """Safety check → execute → return result string."""
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
        confirmed = await _ask_confirmation(tool_name, arguments)
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

async def _tool_loop(messages: list, tools: list) -> str:
    """
    Runs on TOOLS_MODEL (llama-3-groq-70b-tool-use).
    Loops until LLM returns plain text or MAX_TOOL_ITERATIONS reached.
    Executes multiple tool calls in parallel when LLM requests them.
    """
    for iteration in range(MAX_TOOL_ITERATIONS):
        resp = await groq_client.chat.completions.create(
            model=config.TOOLS_MODEL,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            parallel_tool_calls=True
        )

        message = resp.choices[0].message

        # No tool calls — LLM gave final text response
        if not message.tool_calls:
            log.info(f"[Orchestrator] Tool loop done after {iteration + 1} iteration(s).")
            return message.content or ""

        # Append assistant message with tool calls to history
        messages.append({
            "role":       "assistant",
            "content":    message.content or "",
            "tool_calls": [
                {
                    "id":       tc.id,
                    "type":     "function",
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in message.tool_calls
            ]
        })

        # Execute all tool calls in parallel
        tasks = []
        for tc in message.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            tasks.append(_execute_tool(tc.function.name, args))

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

async def orchestrate(prompt: str) -> dict:
    """
    Single unified path for all prompts.
    Returns {"response": str, "command": None}
    """
    from tools.registry import registry

    # Load system prompt and conversation history
    system_prompt = _load_system_prompt()
    history       = _load_history()

    # Load memory context
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

    try:
        # First call — PRIMARY_MODEL decides whether tools are needed
        resp = await groq_client.chat.completions.create(
            model=config.PRIMARY_MODEL,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
            messages=messages,
            tools=tools if tools else None,
            tool_choice="auto" if tools else None
        )

        message = resp.choices[0].message

        if not message.tool_calls:
            # Plain text — no tools needed
            log.info("[Orchestrator] Direct response — no tools.")
            result     = _parse_response(message.content or "")
            final_text = result.get("response", "")

        else:
            # Tool calls detected — switch to TOOLS_MODEL
            log.info(f"[Orchestrator] {len(message.tool_calls)} tool call(s) — switching to TOOLS_MODEL.")

            messages.append({
                "role":       "assistant",
                "content":    message.content or "",
                "tool_calls": [
                    {
                        "id":       tc.id,
                        "type":     "function",
                        "function": {
                            "name":      tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            })

            # Execute first batch in parallel
            tasks = []
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tasks.append(_execute_tool(tc.function.name, args))

            results = await asyncio.gather(*tasks)

            for tc, result in zip(message.tool_calls, results):
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      result
                })

            # Continue loop on TOOLS_MODEL
            final_text = await _tool_loop(messages, tools)

    except Exception as e:
        log.error(f"[Orchestrator] Groq error: {e}")

        # Gemini fallback — plain prompt, no tools
        try:
            from google import genai
            client = genai.Client(api_key=config.GEMINI_API_KEY)
            resp = client.models.generate_content(
                model=config.FALLBACK_MODEL,
                contents=prompt
            )
            result = _parse_response(resp.text)
            final_text = result.get("response", "")
            log.info("[Orchestrator] Gemini fallback succeeded.")
        except Exception as e2:
            log.error(f"[Orchestrator] Gemini fallback error: {e2}")
            final_text = "Both primary and fallback models failed."

    # Save to conversation history + episodic memory
    history.append({"role": "user",      "content": prompt})
    history.append({"role": "assistant", "content": final_text})
    _save_history(history)

    if final_text:
        await memory_write(prompt, final_text)

    log.info(f"[Orchestrator] Done | Q: {prompt[:50]} | A: {final_text[:50]}")
    return {"response": final_text, "command": None}