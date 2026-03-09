import json
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from memory.manager import groq_client
from core.brain import ask as brain_ask
from models.router import route as get_model_for_task

log = logging.getLogger("orchestrator")

# We use the fast model in JSON mode to cleanly categorize user intent, avoiding injection.
CLASSIFY_PROMPT = """You are FRIDAY's Supervisor Agent. 
Determine if the user's input requires complex multi-step execution.

Return ONLY a JSON object with a single key "route" and exactly one of these string values:
- "SIMPLE": greetings, generic questions, quick facts, basic coding help, single web searches.
- "COMPLEX": deep multi-step web research, building applications, OS actions, complex multi-file coding.

User Input:
{user_input}
"""

async def _classify_intent(prompt: str) -> str:
    """Uses LLM with JSON mode to fast-classify if request is Simple or Complex."""
    try:
        resp = await groq_client.chat.completions.create(
            model=config.FAST_MODEL,
            temperature=0,
            max_tokens=64,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(user_input=prompt)}]
        )
        content = resp.choices[0].message.content
        route = json.loads(content).get("route", "SIMPLE").upper()
        if "COMPLEX" in route:
            return "COMPLEX"
        return "SIMPLE"
    except Exception as e:
        log.warning(f"Orchestrator classify error: {e}")
        # Default to simple on error to keep system responsive
        return "SIMPLE"

async def orchestrate(prompt: str) -> dict:
    """
    The main Supervisors logic. 
    Routes to brain.py (Simple) or agents (Complex).
    Returns a dict with {"response": "...", "command": ...}
    """
    decision = await _classify_intent(prompt)
    log.info(f"[Orchestrator] Classified input as: {decision}")

    if decision == "SIMPLE":
        # Pass directly to the core brain (which has memory and CRAG search built-in)
        log.info("[Orchestrator] Routing to core/brain.py")
        return await brain_ask(prompt)
    
    else:
        # COMPLEX route: dispatch to multi-layer agent system
        log.info("[Orchestrator] Routing to COMPLEX agent flow")
        
        try:
            from agents import dispatch
            agent_result = await dispatch(prompt)
            return {"response": agent_result, "command": None}
        except Exception as e:
            log.error(f"[Orchestrator] Agent dispatch failed: {e}, falling back to brain")
            result = await brain_ask(prompt)
            fallback_msg = (
                "🧠 [Supervisor Note] Agent subsystem encountered an error. "
                "Falling back to conversational core.\n\n"
            )
            result["response"] = fallback_msg + result.get("response", "")
            return result
