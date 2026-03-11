"""
core/model_client.py
====================
Unified async LLM client for FRIDAY.

Returns the correct async client based on config.MODEL_MODE:
  - "online"  → AsyncGroq  (Groq API, requires GROQ_API_KEY)
  - "offline" → AsyncOpenAI pointed at Ollama's local endpoint

The orchestrator and all other callers just call get_client() and
get_fallback_client() — they never need to know which provider is active.

Why a separate module instead of putting this in config.py?
  - config.py is imported at module level everywhere (even before mode is set)
  - Instantiating async clients at import time causes issues if the event
    loop isn't ready yet. This module is imported lazily, only when needed.
"""

import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

log = logging.getLogger("model_client")

# Module-level cache — clients are created once and reused
_online_client  = None
_offline_client = None


def get_client():
    """
    Return the async LLM client for the active mode.
    Created once and cached. Safe to call multiple times.

    Online  → groq.AsyncGroq
    Offline → openai.AsyncOpenAI (pointed at Ollama localhost:11434)
    """
    global _online_client, _offline_client

    if config.MODEL_MODE == "offline":
        if _offline_client is None:
            try:
                from openai import AsyncOpenAI
                _offline_client = AsyncOpenAI(
                    base_url=config.OFFLINE_BASE_URL,
                    api_key="ollama",   # Ollama doesn't validate the key, but the field is required
                )
                log.info(f"[ModelClient] Offline client ready → {config.OFFLINE_BASE_URL}")
            except ImportError:
                log.error("[ModelClient] 'openai' package not installed. Run: pip install openai")
                raise
        return _offline_client

    else:  # online
        if _online_client is None:
            from groq import AsyncGroq
            _online_client = AsyncGroq(api_key=config.GROQ_API_KEY or "")
            log.info("[ModelClient] Online client ready → Groq API")
        return _online_client


def get_fallback_client():
    """
    Return the Gemini fallback client (online mode only).
    Returns None in offline mode — there is no cloud fallback.
    """
    if config.MODEL_MODE == "offline":
        return None

    if not config.GEMINI_API_KEY:
        return None

    try:
        from google import genai
        return genai.Client(api_key=config.GEMINI_API_KEY)
    except Exception as e:
        log.warning(f"[ModelClient] Gemini fallback unavailable: {e}")
        return None


async def check_offline_available() -> tuple[bool, str]:
    """
    Ping Ollama to check if it's running and has the required models.
    Returns (available: bool, message: str).
    Called during startup mode selection so the user gets instant feedback.
    """
    try:
        import httpx
        async with httpx.AsyncClient(timeout=3.0) as http:
            resp = await http.get(f"{config.OFFLINE_BASE_URL.replace('/v1', '')}/api/tags")
            if resp.status_code != 200:
                return False, "Ollama is not running. Start it with: ollama serve"

            data        = resp.json()
            model_names = [m["name"].split(":")[0] for m in data.get("models", [])]
            needed      = {
                config.OFFLINE_PRIMARY_MODEL.split(":")[0],
                config.OFFLINE_FAST_MODEL.split(":")[0],
            }
            missing = needed - set(model_names)

            if missing:
                pull_cmds = "  ".join(f"ollama pull {m}" for m in missing)
                return False, f"Missing models: {', '.join(missing)}\nPull them with: {pull_cmds}"

            return True, f"Ollama running | {len(data.get('models', []))} model(s) available"

    except Exception as e:
        return False, f"Cannot reach Ollama at {config.OFFLINE_BASE_URL} — is it running? ({e})"
