"""
main.py
========
FRIDAY Entry Point

Startup flow:
  1. Print banner
  2. Mode selection menu  ← NEW: Online (Groq) or Offline (Ollama)
  3. Pre-flight checks    (adapts based on selected mode)
  4. Initialize resources (embedding models + Qdrant + SQLite)
  5. Load orchestrator + tool registry
  6. REPL loop
  7. Graceful shutdown
"""

import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import config

os.makedirs(config.LOGS_PATH, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(config.LOGS_PATH, "friday_main.log"), encoding="utf-8"),
    ]
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class Color:
    CYAN    = '\033[96m'
    GREEN   = '\033[92m'
    YELLOW  = '\033[93m'
    RED     = '\033[91m'
    MAGENTA = '\033[95m'
    BLUE    = '\033[94m'
    WHITE   = '\033[97m'
    DIM     = '\033[2m'
    RESET   = '\033[0m'
    BOLD    = '\033[1m'


def print_banner():
    print(f"{Color.CYAN}{Color.BOLD}")
    print("========================================")
    try:
        print("███████╗██████╗ ██╗██████╗  █████╗ ██╗   ██╗")
        print("██╔════╝██╔══██╗██║██╔══██╗██╔══██╗╚██╗ ██╔╝")
        print("█████╗  ██████╔╝██║██║  ██║███████║ ╚████╔╝ ")
        print("██╔══╝  ██╔══██╗██║██║  ██║██╔══██║  ╚██╔╝  ")
        print("██║     ██║  ██║██║██████╔╝██║  ██║   ██║   ")
        print("╚═╝     ╚═╝  ╚═╝╚═╝╚═════╝ ╚═╝  ╚═╝   ╚═╝   ")
    except UnicodeEncodeError:
        print("      F R I D A Y  -  A G E N T")
    print(f"       Phase 2 Engine — v{config.VERSION}")
    print("========================================")
    print(f"{Color.RESET}")


# ========================
# MODE SELECTION
# ========================

async def select_mode() -> str:
    """
    Interactive startup menu for selecting Online vs Offline mode.

    Online  → Groq API (llama-3.3-70b, fast, needs internet + API key)
    Offline → Ollama   (local models on Pavan's RTX 4060, no internet)

    Returns the selected mode string: "online" or "offline".
    """
    from core.model_client import check_offline_available

    print(f"{Color.BOLD}{Color.WHITE}  Select AI Mode{Color.RESET}")
    print(f"  {Color.DIM}{'─' * 38}{Color.RESET}")
    print()

    # Show online option with model info
    online_label = f"{Color.GREEN}● Online{Color.RESET}   — Groq API"
    print(f"  {Color.BOLD}[1]{Color.RESET}  {online_label}")
    print(f"       {Color.DIM}Primary : {config.ONLINE_PRIMARY_MODEL}{Color.RESET}")
    print(f"       {Color.DIM}Fast    : {config.ONLINE_FAST_MODEL}{Color.RESET}")
    if config.GROQ_API_KEY:
        print(f"       {Color.GREEN}✓ API key found{Color.RESET}")
    else:
        print(f"       {Color.RED}✗ GROQ_API_KEY missing{Color.RESET}")
    print()

    # Show offline option — ping Ollama to get real status
    print(f"  {Color.BOLD}[2]{Color.RESET}  {Color.BLUE}● Offline{Color.RESET}  — Ollama (local)")
    print(f"       {Color.DIM}Primary : {config.OFFLINE_PRIMARY_MODEL}{Color.RESET}")
    print(f"       {Color.DIM}Fast    : {config.OFFLINE_FAST_MODEL}{Color.RESET}")
    print(f"       {Color.DIM}Checking Ollama...{Color.RESET}", end="\r")
    offline_ok, offline_msg = await check_offline_available()
    status_color  = Color.GREEN if offline_ok else Color.YELLOW
    status_symbol = "✓" if offline_ok else "⚠"
    print(f"       {status_color}{status_symbol} {offline_msg}{Color.RESET}       ")
    print()

    print(f"  {Color.DIM}{'─' * 38}{Color.RESET}")

    # Default: online if key present, offline if not
    default = "1" if config.GROQ_API_KEY else "2"
    default_label = "Online" if default == "1" else "Offline"
    print(f"  {Color.DIM}Default [{default}] {default_label} — press Enter to confirm{Color.RESET}")
    print()

    while True:
        try:
            raw = (await asyncio.to_thread(input, f"  {Color.CYAN}>{Color.RESET} ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)

        choice = raw if raw else default

        if choice == "1":
            if not config.GROQ_API_KEY:
                print(f"  {Color.RED}[ERROR] GROQ_API_KEY is missing from .env — cannot use Online mode.{Color.RESET}")
                continue
            print(f"\n  {Color.GREEN}[OK] Online mode selected → Groq API{Color.RESET}\n")
            return "online"

        elif choice == "2":
            if not offline_ok:
                print(f"  {Color.YELLOW}[WARN] Ollama is not ready: {offline_msg}{Color.RESET}")
                confirm = (await asyncio.to_thread(input, "  Continue anyway? (y/n): ")).strip().lower()
                if confirm != "y":
                    continue
            print(f"\n  {Color.BLUE}[OK] Offline mode selected → Ollama local{Color.RESET}\n")
            return "offline"

        else:
            print(f"  {Color.YELLOW}Enter 1 for Online or 2 for Offline.{Color.RESET}")


# ========================
# PRE-FLIGHT CHECKS
# ========================

async def pre_flight_checks():
    """Verify the selected mode's requirements. Adapts to online vs offline."""
    print(f"{Color.YELLOW}Running Pre-Flight Checks...{Color.RESET}")

    if config.MODEL_MODE == "online":
        # Already validated key in select_mode, but double-check
        if not config.GROQ_API_KEY:
            print(f"{Color.RED}[ERROR] GROQ_API_KEY is missing from .env{Color.RESET}")
            sys.exit(1)
        print(f"{Color.GREEN}[OK] Groq API key found{Color.RESET}")

        if config.GEMINI_API_KEY:
            print(f"{Color.GREEN}[OK] Gemini fallback available{Color.RESET}")
        else:
            print(f"{Color.YELLOW}[WARN] Gemini key missing — no fallback if Groq fails{Color.RESET}")

        tavily = os.getenv("TAVILY_API_KEY", "").strip()
        if tavily:
            print(f"{Color.GREEN}[OK] Tavily Search active{Color.RESET}")
        else:
            print(f"{Color.YELLOW}[WARN] Tavily key missing — using DuckDuckGo fallback{Color.RESET}")

    else:  # offline
        print(f"{Color.BLUE}[OK] Offline mode — no API keys required{Color.RESET}")
        print(f"{Color.YELLOW}[WARN] Web search disabled in offline mode{Color.RESET}")
        print(f"{Color.DIM}       Ollama: {config.OFFLINE_BASE_URL}{Color.RESET}")

    print(f"{Color.GREEN}[OK] Pre-Flight Complete.{Color.RESET}\n")


# ========================
# RESOURCE INIT
# ========================

async def initialize_resources():
    """Initialize embedding models, Qdrant, and SQLite."""
    from memory.core.resources import _res

    print(f"{Color.YELLOW}Loading memory resources...{Color.RESET}")
    print(f"  {Color.CYAN}•{Color.RESET} Embedding models ({config.EMBEDDING_MODEL})")
    print(f"  {Color.CYAN}•{Color.RESET} Qdrant vector store")
    print(f"  {Color.CYAN}•{Color.RESET} SQLite facts database")

    await _res.initialize()

    if _res.qdrant_available:
        print(f"{Color.GREEN}[OK] All memory systems ready{Color.RESET}")
    else:
        print(f"{Color.YELLOW}[WARN] Qdrant unavailable — episodic memory disabled{Color.RESET}")
        print(f"{Color.YELLOW}       Facts and recent history will still work{Color.RESET}")
    print()


async def select_input_mode() -> bool:
    """
    Ask whether Pavan wants to use voice input or text input.
    Returns True for voice, False for text.
    Checks mic + TTS availability before enabling voice.
    """
    from io_friday.ears import check_mic_available
    from io_friday.mouth import check_tts_available

    print(f"{Color.BOLD}{Color.WHITE}  Select Input Mode{Color.RESET}")
    print(f"  {Color.DIM}{'─' * 38}{Color.RESET}\n")

    # Check voice dependencies live
    mic_ok,  mic_msg  = await check_mic_available()
    tts_ok,  tts_msg  = await check_tts_available()
    voice_ok = mic_ok and tts_ok

    mic_icon  = f"{Color.GREEN}✓{Color.RESET}" if mic_ok  else f"{Color.RED}✗{Color.RESET}"
    tts_icon  = f"{Color.GREEN}✓{Color.RESET}" if tts_ok  else f"{Color.RED}✗{Color.RESET}"

    print(f"  {Color.BOLD}[1]{Color.RESET}  {Color.CYAN}● Voice{Color.RESET}  — speak to FRIDAY")
    print(f"       {mic_icon} {mic_msg}")
    print(f"       {tts_icon} {tts_msg}")
    if not voice_ok:
        print(f"       {Color.YELLOW}⚠ Voice unavailable — missing dependencies above{Color.RESET}")
    print()

    print(f"  {Color.BOLD}[2]{Color.RESET}  {Color.WHITE}● Text{Color.RESET}   — type to FRIDAY  {Color.DIM}(default){Color.RESET}")
    print()
    print(f"  {Color.DIM}{'─' * 38}{Color.RESET}")
    print(f"  {Color.DIM}Default [2] Text — press Enter to confirm{Color.RESET}\n")

    while True:
        try:
            raw = (await asyncio.to_thread(input, f"  {Color.CYAN}>{Color.RESET} ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)

        choice = raw if raw else "2"

        if choice == "1":
            if not voice_ok:
                print(f"  {Color.YELLOW}[WARN] Voice dependencies missing — please fix above then retry.{Color.RESET}")
                continue
            print(f"\n  {Color.CYAN}[OK] Voice mode enabled — speak after the prompt{Color.RESET}\n")
            return True
        elif choice == "2":
            print(f"\n  {Color.WHITE}[OK] Text mode enabled{Color.RESET}\n")
            return False
        else:
            print(f"  {Color.YELLOW}Enter 1 for Voice or 2 for Text.{Color.RESET}")


# ========================
# MAIN
# ========================

async def main():
    os.system("")   # Enable ANSI escape codes on Windows

    print_banner()

    # ── Step 1: Model mode selection ────────────────────────────────────────
    mode = await select_mode()
    config.apply_mode(mode)

    # ── Step 2: Pre-flight checks ────────────────────────────────────────────
    await pre_flight_checks()

    # ── Step 3: Initialize memory resources ─────────────────────────────────
    await initialize_resources()

    # ── Step 4: Input mode selection (text or voice) ─────────────────────────
    voice_mode = await select_input_mode()

    # ── Step 5: Load orchestrator + tools ────────────────────────────────────
    print(f"{Color.YELLOW}Loading FRIDAY core...{Color.RESET}")
    from core.orchestrator import orchestrate
    from tools.registry import registry

    tool_count = len(registry.list_all())
    if tool_count > 0:
        print(f"{Color.GREEN}[OK] {tool_count} tool(s) registered:{Color.RESET}")
        for t in registry.list_all():
            print(f"  {Color.CYAN}•{Color.RESET} {t.name} [{t.risk.value}]")
    else:
        print(f"{Color.YELLOW}[WARN] No tools registered{Color.RESET}")

    mode_tag = (
        f"{Color.GREEN}[ONLINE — Groq]{Color.RESET}"
        if config.MODEL_MODE == "online"
        else f"{Color.BLUE}[OFFLINE — Ollama]{Color.RESET}"
    )
    input_tag = f"{Color.CYAN}[VOICE]{Color.RESET}" if voice_mode else f"{Color.WHITE}[TEXT]{Color.RESET}"
    print(f"\n  Mode    : {mode_tag}  {input_tag}")
    print(f"  Primary : {Color.DIM}{config.PRIMARY_MODEL}{Color.RESET}")

    if voice_mode:
        print(f"\nSay '{Color.YELLOW}exit{Color.RESET}' or '{Color.YELLOW}quit{Color.RESET}' to shutdown. "
              f"Say '{Color.YELLOW}switch{Color.RESET}' to change AI mode.\n")
    else:
        print(f"\nType '{Color.YELLOW}exit{Color.RESET}' to shutdown. "
              f"Type '{Color.YELLOW}switch{Color.RESET}' to change AI mode. "
              f"Type '{Color.YELLOW}voice{Color.RESET}' to toggle voice.\n")

    # ── Step 6: REPL ─────────────────────────────────────────────────────────
    from prompt_toolkit import PromptSession
    from prompt_toolkit.styles import Style

    session = PromptSession()
    style   = Style.from_dict({'prompt': 'ansicyan bold'})

    while True:
        try:
            # ── Get input ──────────────────────────────────────────────────
            if voice_mode:
                from io_friday.ears import listen
                print(f"{Color.DIM}  🎙  Listening...{Color.RESET}", end="\r")
                user_input = await listen()
                print(" " * 30, end="\r")
                if not user_input:
                    continue
                print(f"{Color.CYAN}You  > {Color.RESET}{user_input}")
            else:
                user_input = await session.prompt_async(
                    [('class:prompt', '\nUser > ')],
                    style=style
                )
                user_input = user_input.strip()
                if not user_input:
                    continue

            # ── Built-in commands ──────────────────────────────────────────
            if user_input.lower() in ("exit", "quit", "shutdown"):
                farewell = "Goodbye, Pavan."
                print(f"\n{Color.MAGENTA}FRIDAY: {farewell}{Color.RESET}")
                if voice_mode:
                    from io_friday.mouth import speak
                    await speak(farewell)
                break

            if user_input.lower() == "switch":
                new_mode = await select_mode()
                config.apply_mode(new_mode)
                from core import model_client as _mc
                _mc._online_client  = None
                _mc._offline_client = None
                print(f"{Color.GREEN}[OK] Switched to {config.MODEL_MODE} mode.{Color.RESET}")
                continue

            if user_input.lower() == "voice":
                voice_mode = not voice_mode
                state = "enabled" if voice_mode else "disabled"
                print(f"{Color.CYAN}[OK] Voice {state}.{Color.RESET}")
                continue

            # ── Orchestrate ────────────────────────────────────────────────
            print(f"{Color.MAGENTA}FRIDAY > {Color.RESET}...", end="\r")
            result        = await orchestrate(user_input, session)
            response_text = result.get("response", "No response generated.")
            command       = result.get("command")

            print(" " * 50, end="\r")
            print(f"{Color.MAGENTA}FRIDAY > {Color.RESET}{response_text}")

            if command:
                print(f"\n{Color.YELLOW}[Command Requested]{Color.RESET} {command}")

            # Speak the response if voice mode is active
            if voice_mode and response_text:
                from io_friday.mouth import speak
                await speak(response_text)

        except KeyboardInterrupt:
            print(f"\n{Color.MAGENTA}FRIDAY: Shutting down.{Color.RESET}")
            break
        except EOFError:
            print(f"\n{Color.MAGENTA}FRIDAY: Goodbye, Pavan.{Color.RESET}")
            break
        except Exception as e:
            logging.error(f"Main loop error: {e}", exc_info=True)
            print(f"\n{Color.RED}An internal error occurred: {e}{Color.RESET}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
