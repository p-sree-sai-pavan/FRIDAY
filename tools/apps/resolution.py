"""
tools/apps/resolution.py
========================
Resolves application names to executable commands using a multi-step pipeline:
Alias -> Start Menu -> PATH -> LLM Fallback -> Direct
"""

import difflib
import logging
import shutil

from .constants import ALIASES
from .discovery import scan_start_menu, resolve_lnk

log = logging.getLogger("apps.resolution")


# ========================
# FUZZY MATCHING
# ========================

def _fuzzy_find(name: str, candidates: list[str], threshold: float = 0.6) -> str | None:
    """Find the best fuzzy match for name in candidates."""
    matches = difflib.get_close_matches(name.lower(), candidates, n=1, cutoff=threshold)
    return matches[0] if matches else None


# ========================
# LLM COMMAND GENERATION
# ========================

_llm_cache: dict[str, str | None] = {}

def _llm_resolve(name: str) -> str | None:
    """
    Ask the LLM to generate a Windows command to open the given app.
    Returns the command string, or None if the LLM can't figure it out.
    """
    if name in _llm_cache:
        return _llm_cache[name]

    try:
        from groq import Groq
        import config as cfg

        client = Groq(api_key=cfg.GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=cfg.FAST_MODEL,
            temperature=0,
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": (
                    "You are a Windows command expert. Generate the exact command to open "
                    f"'{name}' on Windows 10/11. Rules:\n"
                    "- Reply with ONLY the command, nothing else\n"
                    "- Use 'start' for URLs (e.g. start https://example.com)\n"
                    "- Use exe name or full path for apps\n"
                    "- If the app doesn't exist or you're unsure, reply with just: UNKNOWN\n"
                    "- Never use PowerShell unless absolutely necessary\n"
                    "- No explanations, no markdown, just the raw command"
                )
            }]
        )
        cmd = resp.choices[0].message.content.strip()

        # Safety: reject anything that looks dangerous
        dangerous = ["del ", "rm ", "format ", "rmdir", "rd ", "::{", "reg delete", "shutdown"]
        if any(d in cmd.lower() for d in dangerous):
            log.warning(f"[Apps] LLM returned dangerous command for '{name}': {cmd}")
            _llm_cache[name] = None
            return None

        if cmd.upper() == "UNKNOWN" or not cmd or len(cmd) > 500:
            _llm_cache[name] = None
            return None

        log.info(f"[Apps] LLM generated command for '{name}': {cmd}")
        _llm_cache[name] = cmd
        return cmd

    except Exception as e:
        log.warning(f"[Apps] LLM resolve failed for '{name}': {e}")
        _llm_cache[name] = None
        return None


# ========================
# APP RESOLUTION PIPELINE
# ========================

def resolve_app(name: str) -> tuple[str, str]:
    """
    Resolve an app name to a launchable command.
    Returns (command, source) where source describes how it was found.
    """
    key = name.lower().strip()

    # Step 1: Direct alias match
    if key in ALIASES:
        return ALIASES[key], "alias"

    # Step 2: Fuzzy alias match (catches typos)
    alias_match = _fuzzy_find(key, list(ALIASES.keys()))
    if alias_match:
        return ALIASES[alias_match], f"alias (fuzzy: '{alias_match}')"

    # Step 3: Start Menu discovery
    start_menu = scan_start_menu()

    # Direct Start Menu match
    if key in start_menu:
        lnk_path = start_menu[key]
        target = resolve_lnk(lnk_path)
        if target:
            return f'"{target}"', "start menu"
        return f'"{lnk_path}"', "start menu (shortcut)"

    # Fuzzy Start Menu match
    sm_match = _fuzzy_find(key, list(start_menu.keys()))
    if sm_match:
        lnk_path = start_menu[sm_match]
        target = resolve_lnk(lnk_path)
        if target:
            return f'"{target}"', f"start menu (fuzzy: '{sm_match}')"
        return f'"{lnk_path}"', f"start menu shortcut (fuzzy: '{sm_match}')"

    # Step 4: System PATH
    which_result = shutil.which(key)
    if which_result:
        return f'"{which_result}"', "system PATH"

    # Step 5: LLM generates command
    llm_cmd = _llm_resolve(key)
    if llm_cmd:
        return llm_cmd, "LLM-generated"

    # Step 6: Last resort — try the name directly
    return key, "direct (unresolved)"
