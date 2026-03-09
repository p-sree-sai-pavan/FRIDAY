import asyncio
import logging
import sys
import os
import io

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import config

# IMPORTANT: Initialize logging early
os.makedirs(config.LOGS_PATH, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(config.LOGS_PATH, "friday_main.log")),
        # Suppress noisy stream output for a cleaner CLI
    ]
)
# Silence chatty libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from core.orchestrator import orchestrate
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style

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

# Removed simple async_input in favor of prompt_toolkit

async def pre_flight_checks():
    """Verifies critical API keys before starting."""
    print(f"{Color.YELLOW}Running Pre-Flight Checks...{Color.RESET}")
    if not config.GROQ_API_KEY:
        print(f"{Color.RED}[ERROR] GROQ_API_KEY is missing from .env{Color.RESET}")
        sys.exit(1)
        
    tavily = os.getenv("TAVILY_API_KEY")
    if tavily:
        print(f"{Color.GREEN}[OK] Tavily Search active (Agentic RAG enabled){Color.RESET}")
    else:
        print(f"{Color.YELLOW}[WARN] Tavily key missing. Using DuckDuckGo fallback tier.{Color.RESET}")
        
    print(f"{Color.GREEN}[OK] Pre-Flight Complete.{Color.RESET}\n")

async def main():
    os.system("") # Enable ANSI colors on Windows
    print_banner()
    await pre_flight_checks()

    print(f"Type '{Color.YELLOW}exit{Color.RESET}' or '{Color.YELLOW}quit{Color.RESET}' to shutdown.\n")

    # Initialize prompt_toolkit session for perfect Agentic UX (history, cursors, etc)
    session = PromptSession()
    style = Style.from_dict({
        'prompt': 'ansicyan bold',
    })

    while True:
        try:
            # Get user input asynchronously with persistent history & up-arrow support
            user_input = await session.prompt_async([('class:prompt', '\nUser > ')], style=style)
            user_input = user_input.strip()

            if not user_input:
                continue
                
            if user_input.lower() in ["exit", "quit", "shutdown"]:
                print(f"\n{Color.MAGENTA}FRIDAY: Goodbye, sir.{Color.RESET}")
                break

            # Send to Orchestrator (Supervisor)
            print(f"{Color.MAGENTA}FRIDAY > {Color.RESET}...", end="\r")
            
            result = await orchestrate(user_input)
            
            response_text = result.get("response", "No response generated.")
            command = result.get("command")

            # Print final response
            # Clear the loading text cleanly
            print(" " * 50, end="\r")
            print(f"{Color.MAGENTA}FRIDAY > {Color.RESET}{response_text}")

            # Temporary handling for tools (until core/agent.py is fully wired)
            if command:
                print(f"\n{Color.YELLOW}[Command Requested]{Color.RESET} {command}")
                print(f"{Color.RED}(Command execution is disabled until Phase 2 tools are built){Color.RESET}")

        except KeyboardInterrupt:
            print(f"\n{Color.MAGENTA}FRIDAY: Shutting down.{Color.RESET}")
            break
        except Exception as e:
            logging.error(f"Main loop error: {e}", exc_info=True)
            print(f"\n{Color.RED}An internal error occurred: {e}{Color.RESET}")

if __name__ == "__main__":
    # Ensure Windows asyncio works cleanly
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
