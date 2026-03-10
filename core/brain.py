"""
core/brain.py
==============
FRIDAY Conversational Core

Handles all non-tool responses:
  - Loads system prompt + conversation history
  - Retrieves memory context (HyDE + hybrid CRAG)
  - Calls PRIMARY_MODEL with full context
  - Lightweight faithfulness check when memory was used
  - Saves response to episodic memory
  - Gemini fallback on Groq failure
"""

import json
import os
import sys
import logging
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

from memory.manager import (
    groq_client,
    read as memory_read,
    write as memory_write,
    compress_results,
)

# ========================
# LOGGING
# ========================
os.makedirs(config.LOGS_PATH, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(config.LOGS_PATH, "friday.log")),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("brain")

# ========================
# PATHS
# brain.py owns conversation history (memory.json)
# manager.py owns episodic/semantic memory (qdrant + facts.db)
# ========================
BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE       = os.path.join(config.MEMORY_PATH, "memory.json")
SYSTEM_PROMPT_FILE = os.path.join(config.PROMPTS_PATH, "system_prompt.txt")

os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)


# ========================
# HELPERS
# ========================

def _load_system_prompt() -> str:
    from datetime import datetime
    try:
        with open(SYSTEM_PROMPT_FILE) as f:
            content = f.read().strip()
        now = datetime.now().strftime("%A, %d %B %Y %I:%M %p")
        log.info(f"[Brain] System prompt loaded. Length: {len(content)} chars")
        return f"{content}\n\nCurrent date and time: {now}"
    except FileNotFoundError:
        log.error(f"[Brain] system_prompt.txt NOT FOUND at: {SYSTEM_PROMPT_FILE}")
        return "You are FRIDAY, an advanced AI assistant for Pavan."

def _load_history() -> list:
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE) as f:
                return json.load(f)
    except Exception as e:
        log.warning(f"History load error: {e}")
    return []


def _save_history(history: list):
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history[-config.MAX_HISTORY:], f, indent=2)
    except Exception as e:
        log.warning(f"History save error: {e}")


def _parse_response(text: str) -> dict:
    """Plain text response — no JSON parsing needed."""
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
    ≥ 0.5 → save normally.
    """
    try:
        resp = await groq_client.chat.completions.create(
            model=config.FAST_MODEL,
            temperature=0,
            max_tokens=10,
            response_format={"type": "json_object"},
            messages=[{
                "role": "user",
                "content": (
                    f"Rate how faithfully this answer reflects the context and addresses the query.\n"
                    f"Query: {prompt}\n"
                    f"Context: {context[:500]}\n"
                    f"Answer: {answer[:300]}\n"
                    f"Reply ONLY with JSON: {{\"score\": 0.0-1.0}}"
                )
            }]
        )
        raw   = resp.choices[0].message.content
        score = json.loads(raw).get("score", 0.5)
        return float(score)
    except Exception as e:
        log.warning(f"Faithfulness check error: {e} — defaulting to 0.5")
        return 0.5


# ========================
# MAIN ASK
# ========================

async def ask(prompt: str) -> dict:
    system_prompt  = _load_system_prompt()
    history        = _load_history()

    # Retrieve memory context
    memory_results = await memory_read(prompt)
    context        = compress_results(memory_results)

    # Inject memory into system prompt if relevant context exists
    full_system = system_prompt
    if context:
        full_system = (
            f"{system_prompt}\n\n"
            f"--- RELEVANT MEMORY ---\n{context}\n--- END MEMORY ---"
        )

    # Build messages
    messages = [{"role": "system", "content": full_system}]
    messages += history
    messages.append({"role": "user", "content": prompt})

    # Primary call — Groq
    result = None
    try:
        resp = await groq_client.chat.completions.create(
            model=config.PRIMARY_MODEL,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
            messages=messages
        )
        result = _parse_response(resp.choices[0].message.content)

    except Exception as e:
        log.error(f"Groq error: {e}")

        # Gemini fallback
        try:
            from google import genai
            client = genai.Client(api_key=config.GEMINI_API_KEY)
            resp = client.models.generate_content(
                model=config.FALLBACK_MODEL,
                contents=prompt
            )
            result = _parse_response(resp.text)
            log.info("Gemini fallback succeeded.")
        except Exception as e2:
            log.error(f"Gemini fallback error: {e2}")
            result = {"response": "Both primary and fallback models failed.", "command": None}

    answer = result.get("response", "")

    # Faithfulness check — only when memory context was used
    save_to_memory = True
    if answer and context:
        score = await _faithfulness_check(prompt, answer, context)
        if score < 0.5:
            log.warning(
                f"[Brain] Faithfulness score {score:.2f} — "
                f"skipping memory write for: {prompt[:60]}"
            )
            save_to_memory = False
        else:
            log.debug(f"[Brain] Faithfulness score {score:.2f} — saving to memory.")

    # Save conversation history
    history.append({"role": "user",      "content": prompt})
    history.append({"role": "assistant", "content": answer})
    _save_history(history)

    # Save to episodic memory only if faithful
    if save_to_memory and answer:
        await memory_write(prompt, answer)

    log.info(f"Q: {prompt[:60]} | A: {answer[:60]}")
    return result