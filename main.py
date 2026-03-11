"""
main.py
========
FRIDAY Entry Point

Startup flow:
  1. Print banner + pre-flight checks (fast — no ML models loaded yet)
  2. Lazy-import orchestrator (triggers model loading on first call)
  3. REPL loop with prompt_toolkit (history, up-arrow, clean UX)
  4. Graceful shutdown on exit/quit/Ctrl+C
"""

import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import config

# Initialize logging early — before any other imports
os.makedirs(config.LOGS_PATH, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(config.LOGS_PATH, "friday_main.log"), encoding="utf-8"),
        # Stream handler suppressed for cleaner CLI
    ]
)
# Silence chatty libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class Color:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_banner():
    """Prints the startup banner."""
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


async def pre_flight_checks():
    """Verifies critical API keys before starting."""
    print(f"{Color.YELLOW}Running Pre-Flight Checks...{Color.RESET}")

    if not config.GROQ_API_KEY:
        print(f"{Color.RED}[ERROR] GROQ_API_KEY is missing from .env{Color.RESET}")
        sys.exit(1)

    print(f"{Color.GREEN}[OK] Groq API key found{Color.RESET}")

    if config.GEMINI_API_KEY:
        print(f"{Color.GREEN}[OK] Gemini fallback available{Color.RESET}")
    else:
        print(f"{Color.YELLOW}[WARN] Gemini key missing — no fallback if Groq fails{Color.RESET}")

    tavily = os.getenv("TAVILY_API_KEY")
    if tavily and tavily.strip():
        print(f"{Color.GREEN}[OK] Tavily Search active (Agentic RAG enabled){Color.RESET}")
    else:
        print(f"{Color.YELLOW}[WARN] Tavily key missing — using DuckDuckGo fallback tier{Color.RESET}")

    print(f"{Color.GREEN}[OK] Pre-Flight Complete.{Color.RESET}\n")


async def main():
    os.system("")  # Enable ANSI colors on Windows

    print_banner()
    await pre_flight_checks()

    # Import orchestrator here — this triggers lazy model loading chain
    # Pre-flight has passed, so API keys are validated
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

    print(f"\nType '{Color.YELLOW}exit{Color.RESET}' or '{Color.YELLOW}quit{Color.RESET}' to shutdown.\n")

    # Initialize prompt_toolkit session
    from prompt_toolkit import PromptSession
    from prompt_toolkit.styles import Style

    session = PromptSession()
    style = Style.from_dict({
        'prompt': 'ansicyan bold',
    })

    while True:
        try:
            user_input = await session.prompt_async(
                [('class:prompt', '\nUser > ')],
                style=style
            )
            user_input = user_input.strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "shutdown"]:
                print(f"\n{Color.MAGENTA}FRIDAY: Goodbye, Pavan.{Color.RESET}")
                break

            # Send to Orchestrator
            print(f"{Color.MAGENTA}FRIDAY > {Color.RESET}...", end="\r")

            result = await orchestrate(user_input, session)

            response_text = result.get("response", "No response generated.")
            command = result.get("command")

            # Clear loading text and print response
            print(" " * 50, end="\r")
            print(f"{Color.MAGENTA}FRIDAY > {Color.RESET}{response_text}")

            if command:
                print(f"\n{Color.YELLOW}[Command Requested]{Color.RESET} {command}")

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
    # Windows: use default ProactorEventLoop (supports prompt_toolkit properly)
    # Do NOT set WindowsSelectorEventLoopPolicy — it breaks prompt_toolkit's async I/O
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
