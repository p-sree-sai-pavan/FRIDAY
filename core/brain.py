"""
core/brain.py
==============
FRIDAY Conversational Core — Helper Functions

Owns conversation history (memory.json) and system prompt loading.
These helpers are used by orchestrator.py, which owns the full pipeline.

Responsibilities:
  - Load/save system prompt
  - Load/save conversation history
  - Parse LLM responses
  - Faithfulness check (guards memory writes)
"""

import json
import os
import sys
import logging
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

log = logging.getLogger("brain")

# ========================
# PATHS
# brain.py owns conversation history (memory.json)
# manager.py owns episodic/semantic memory (qdrant + facts.db)
# ========================
HISTORY_FILE       = os.path.join(config.MEMORY_PATH, "memory.json")
SYSTEM_PROMPT_FILE = os.path.join(config.PROMPTS_PATH, "system_prompt.txt")

os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)


# ========================
# SYSTEM PROMPT
# ========================

def _load_system_prompt() -> str:
    """Load the system prompt from disk and append current timestamp."""
    try:
        with open(SYSTEM_PROMPT_FILE, encoding="utf-8") as f:
            content = f.read().strip()
        now = datetime.now().strftime("%A, %d %B %Y %I:%M %p")
        log.info(f"[Brain] System prompt loaded. Length: {len(content)} chars")
        return f"{content}\n\nCurrent date and time: {now}"
    except FileNotFoundError:
        log.error(f"[Brain] system_prompt.txt NOT FOUND at: {SYSTEM_PROMPT_FILE}")
        return "You are FRIDAY, an advanced AI assistant for Pavan."


# ========================
# CONVERSATION HISTORY
# ========================

def _load_history() -> list:
    """Load conversation history from memory.json. Returns [] on any error."""
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
    except (json.JSONDecodeError, OSError) as e:
        log.warning(f"[Brain] History load error: {e}")
    return []


def _save_history(history: list):
    """Save conversation history, keeping only the last MAX_HISTORY entries."""
    try:
        trimmed = history[-config.MAX_HISTORY:]
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(trimmed, f, indent=2, ensure_ascii=False)
    except OSError as e:
        log.warning(f"[Brain] History save error: {e}")


# ========================
# RESPONSE PARSING
# ========================

def _parse_response(text: str) -> dict:
    """Plain text response — wraps in standard format."""
    return {"response": text.strip(), "command": None}


# ========================
# FAITHFULNESS CHECK
# Runs only when memory context was used.
# Checks if the answer faithfully reflects the retrieved context.
# Prevents bad/hallucinated answers from poisoning the memory store.
# ========================

async def _faithfulness_check(prompt: str, answer: str, context: str) -> float:
    """
    Returns a score 0.0–1.0.
    < 0.5 → answer is unfaithful or off-topic → don't save to memory.
    >= 0.5 → save normally.
    Uses get_client() so it works in both online and offline modes.
    """
    from core.model_client import get_client

    try:
        client = get_client()
        extra  = {"response_format": {"type": "json_object"}} if config.MODEL_MODE == "online" else {}
        resp = await client.chat.completions.create(
            model=config.FAST_MODEL,
            temperature=0,
            max_tokens=10,
            messages=[{
                "role": "user",
                "content": (
                    f"Rate how faithfully this answer reflects the context and addresses the query.\n"
                    f"Query: {prompt}\n"
                    f"Context: {context[:500]}\n"
                    f"Answer: {answer[:300]}\n"
                    f"Reply ONLY with JSON: {{\"score\": 0.0-1.0}}"
                )
            }],
            **extra
        )
        raw   = (resp.choices[0].message.content or "").strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        score = json.loads(raw).get("score", 0.5)
        return float(score)
    except Exception as e:
        log.warning(f"[Brain] Faithfulness check error: {e} — defaulting to 0.5")
        return 0.5
