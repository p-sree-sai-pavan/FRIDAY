"""
tools/apps/resolution.py
========================
Resolves application names to executable commands.

ARCHITECTURE:
  Previously a fully synchronous module. This caused the async event loop
  to block whenever the LLM fallback was needed, because the sync Groq
  client blocks the calling thread (and in an asyncio program, that IS
  the event loop thread).

  Now: fully async throughout. The LLM step uses AsyncGroq directly —
  no event loop blocking. Filesystem steps (Start Menu scan, .lnk resolve)
  are sync but fast, wrapped in asyncio.to_thread where appropriate.

Resolution pipeline (in order):
  1. Direct alias       — instant dict lookup (ALIASES constant)
  2. Fuzzy alias        — catches typos like 'spottify', 'vsc'
  3. Start Menu direct  — scan installed .lnk shortcuts
  4. Start Menu fuzzy   — best fuzzy match in installed apps
  5. System PATH        — shutil.which (fast env-var lookup, not threaded)
  6. Async LLM fallback — AsyncGroq generates the Windows command
  7. Direct passthrough — last resort, pass name directly to OS
"""

import asyncio
import difflib
import logging
import shutil

from .constants import ALIASES
from .discovery import scan_start_menu, resolve_lnk

log = logging.getLogger("apps.resolution")

# Commands the LLM must never be allowed to return
_DANGEROUS_PATTERNS = [
    "del ",    # cmd delete (note: also catches 'del/f', 'del/s')
    "del/",
    "rm ",
    "format ",
    "rmdir",
    "rd ",
    "::{",     # dangerous CLSID shell tricks
    "reg delete",
    "reg add",
    "shutdown",
    "cipher /w",
    "bcdedit",
    "net user",
    "diskpart",
    "attrib -r",
]


# ========================
# FUZZY MATCHING
# ========================

def _fuzzy_find(name: str, candidates: list[str], threshold: float = 0.6) -> str | None:
    """Return the best fuzzy match from candidates, or None if below threshold."""
    matches = difflib.get_close_matches(name.lower(), candidates, n=1, cutoff=threshold)
    return matches[0] if matches else None


# ========================
# ASYNC LLM FALLBACK
# ========================

_llm_cache: dict[str, str | None] = {}


async def _llm_resolve(name: str) -> str | None:
    """
    Ask Groq to generate the Windows command to open the given app.

    Uses AsyncGroq natively — never blocks the event loop.
    Results are cached in-process so the same app name never hits
    the API twice per session.

    Returns the command string, or None if the LLM can't resolve it.
    """
    if name in _llm_cache:
        return _llm_cache[name]

    try:
        from groq import AsyncGroq
        import config as cfg

        client = AsyncGroq(api_key=cfg.GROQ_API_KEY)
        resp = await client.chat.completions.create(
            model=cfg.FAST_MODEL,
            temperature=0,
            max_tokens=80,
            messages=[{
                "role": "user",
                "content": (
                    "You are a Windows command expert. Generate the exact command to open "
                    f"'{name}' on Windows 10/11.\n"
                    "Rules:\n"
                    "- Reply with ONLY the command — no explanation, no markdown\n"
                    "- Use 'start' for URLs (e.g. start https://example.com)\n"
                    "- Use the .exe name or full path for desktop apps\n"
                    "- If the app doesn't exist or you're unsure, reply with exactly: UNKNOWN\n"
                    "- Avoid PowerShell unless there is no other option"
                )
            }]
        )
        cmd = resp.choices[0].message.content.strip()

        # Safety gate — reject any command that looks destructive
        if any(pattern in cmd.lower() for pattern in _DANGEROUS_PATTERNS):
            log.warning(f"[Apps] LLM returned dangerous command for '{name}': {cmd!r}")
            _llm_cache[name] = None
            return None

        if not cmd or cmd.upper() == "UNKNOWN" or len(cmd) > 500:
            _llm_cache[name] = None
            return None

        log.info(f"[Apps] LLM resolved '{name}' → '{cmd}'")
        _llm_cache[name] = cmd
        return cmd

    except Exception as e:
        log.warning(f"[Apps] LLM resolve failed for '{name}': {e}")
        _llm_cache[name] = None
        return None


# ========================
# MAIN ASYNC RESOLUTION PIPELINE
# ========================

async def resolve_app(name: str) -> tuple[str, str]:
    """
    Resolve an app name to a launchable Windows command.

    Returns:
        (command, source) — command is the string to execute,
        source is a human-readable label showing how it was resolved.

    All steps are async-safe. Blocking I/O (filesystem scans) runs
    in asyncio.to_thread. The LLM step uses AsyncGroq natively.
    """
    key = name.lower().strip()

    # Step 1: Direct alias — O(1) dict lookup, instant
    if key in ALIASES:
        return ALIASES[key], "alias"

    # Step 2: Fuzzy alias — catches typos and slang
    alias_match = _fuzzy_find(key, list(ALIASES.keys()))
    if alias_match:
        return ALIASES[alias_match], f"alias (fuzzy: '{alias_match}')"

    # Step 3 & 4: Start Menu — filesystem scan, run in thread
    start_menu = await asyncio.to_thread(scan_start_menu)

    if key in start_menu:
        lnk_path = start_menu[key]
        target = await asyncio.to_thread(resolve_lnk, lnk_path)
        if target:
            return f'"{target}"', "start menu"
        return f'"{lnk_path}"', "start menu (shortcut)"

    sm_match = _fuzzy_find(key, list(start_menu.keys()))
    if sm_match:
        lnk_path = start_menu[sm_match]
        target = await asyncio.to_thread(resolve_lnk, lnk_path)
        if target:
            return f'"{target}"', f"start menu (fuzzy: '{sm_match}')"
        return f'"{lnk_path}"', f"start menu shortcut (fuzzy: '{sm_match}')"

    # Step 5: System PATH — shutil.which is a pure env-var lookup, fast enough inline
    which_result = shutil.which(key)
    if which_result:
        return f'"{which_result}"', "system PATH"

    # Step 6: Async LLM fallback — AsyncGroq, never blocks the event loop
    llm_cmd = await _llm_resolve(key)
    if llm_cmd:
        return llm_cmd, "LLM-generated"

    # Step 7: Last resort — pass name directly and let the OS figure it out
    return key, "direct (unresolved)"
