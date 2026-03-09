import json
import os
import sys
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

# Fix 15: import shared groq client from manager — no second client
from memory.manager import (
    groq_client,
    read as memory_read,
    write as memory_write,
    compress_results,
    self_rag_reflect,
    save_feedback
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
# Fix 13: brain.py no longer manages memory.json directly.
# memory.json is owned by manager.py (search_working reads it).
# brain.py only writes conversation history here.
# ========================
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.abspath(os.path.join(BASE_DIR, "..", config.MEMORY_PATH, "memory.json"))
SYSTEM_PROMPT_FILE = os.path.abspath(os.path.join(config.PROMPTS_PATH, "system_prompt.txt"))

os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)


def _load_system_prompt() -> str:
    try:
        with open(SYSTEM_PROMPT_FILE) as f:
            return f.read().strip()
    except FileNotFoundError:
        return "You are FRIDAY, an advanced AI assistant. Respond in JSON: {\"response\": \"...\", \"command\": null}"


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
    """Extract JSON from model output."""
    try:
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start != -1 and end > start:
            data = json.loads(text[start:end])
            if data.get("command") == "null":
                data["command"] = None
            return data
    except Exception:
        pass
    return {"response": text.strip(), "command": None}


# ========================
# MAIN ASK — Fix 14: now uses full memory pipeline
# ========================

async def ask(prompt: str) -> dict:
    system_prompt = _load_system_prompt()
    history       = _load_history()

    # Fix 14: retrieve memory context before generating
    memory_results = await memory_read(prompt)
    context        = compress_results(memory_results)

    # Inject context into system prompt if we have relevant memory
    full_system = system_prompt
    if context:
        full_system = f"{system_prompt}\n\n--- RELEVANT MEMORY ---\n{context}\n--- END MEMORY ---"

    # Build messages
    messages = [{"role": "system", "content": full_system}]
    messages += history
    messages.append({"role": "user", "content": prompt})

    try:
        resp = await groq_client.chat.completions.create(
            model=config.PRIMARY_MODEL,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
            messages=messages
        )
        raw    = resp.choices[0].message.content
        result = _parse_response(raw)

    except Exception as e:
        log.error(f"Groq error: {e}")
        # Gemini fallback
        try:
            import google.generativeai as genai
            genai.configure(api_key=config.GEMINI_API_KEY)
            model  = genai.GenerativeModel(config.FALLBACK_MODEL)
            resp   = model.generate_content(prompt)
            result = _parse_response(resp.text)
        except Exception as e2:
            log.error(f"Gemini fallback error: {e2}")
            result = {"response": "Both primary and fallback models failed.", "command": None}

    answer = result.get("response", "")

    # Fix 14: Self-RAG reflection + feedback
    if answer and context:
        was_good = await self_rag_reflect(prompt, answer, context)
        await save_feedback(prompt, context, answer, was_good)
        if not was_good:
            log.warning(f"Self-RAG flagged low quality answer for: {prompt}")

    # Save to history (Fix 13: brain owns history, manager owns episodic/semantic)
    history.append({"role": "user", "content": prompt})
    history.append({"role": "assistant", "content": answer})
    _save_history(history)

    # Fix 14: write to episodic memory
    await memory_write(prompt, answer)

    log.info(f"Q: {prompt[:60]} | A: {answer[:60]}")
    return result