import os
from dotenv import load_dotenv
load_dotenv()

# ========================
# APP
# ========================
APP_NAME = "FRIDAY"
VERSION  = "2.0.0"

# ========================
# RUNTIME MODE
# Set at startup by the user selection menu in main.py.
# Everything in the app reads from here — never hardcoded.
# Values: "online" | "offline"
# ========================
MODEL_MODE = "online"   # default; overwritten by select_mode() before first use

# ========================
# ONLINE MODE — Groq API
# Fast, high-quality, requires internet + GROQ_API_KEY
# ========================
ONLINE_PRIMARY_MODEL  = "llama-3.3-70b-versatile"
ONLINE_FAST_MODEL     = "llama-3.1-8b-instant"
ONLINE_TOOLS_MODEL    = "llama-3.3-70b-versatile"
ONLINE_VISION_MODEL   = "llama4-scout-17b-16e"
ONLINE_FALLBACK_MODEL = "gemini-2.5-flash"   # used if Groq fails entirely

# ========================
# OFFLINE MODE — Ollama (local)
# Runs on Pavan's RTX 4060 / 32GB RAM, no internet needed.
# Change these to whatever models you have pulled in Ollama.
# Check available models with: ollama list
# ========================
OFFLINE_PRIMARY_MODEL = "llama3.1:8b"     # best balance for your GPU
OFFLINE_FAST_MODEL    = "llama3.2:3b"     # tiny + instant for classification
OFFLINE_TOOLS_MODEL   = "llama3.1:8b"     # tool calling needs a capable model
OFFLINE_VISION_MODEL  = "llava:7b"        # multimodal (if pulled)
OFFLINE_BASE_URL      = "http://localhost:11434/v1"   # Ollama OpenAI-compatible endpoint

# ========================
# ACTIVE MODEL ACCESSORS
# The orchestrator always reads from these — never from ONLINE_* / OFFLINE_* directly.
# These are updated by apply_mode() when the user picks a mode at startup.
# ========================
PRIMARY_MODEL  = ONLINE_PRIMARY_MODEL
FAST_MODEL     = ONLINE_FAST_MODEL
TOOLS_MODEL    = ONLINE_TOOLS_MODEL
VISION_MODEL   = ONLINE_VISION_MODEL
FALLBACK_MODEL = ONLINE_FALLBACK_MODEL

def apply_mode(mode: str) -> None:
    """
    Switch the active model set to 'online' or 'offline'.
    Call this once from main.py after the user makes their selection.
    All other modules read PRIMARY_MODEL, FAST_MODEL etc. which are
    updated here — they never need to know about the mode themselves.
    """
    global MODEL_MODE, PRIMARY_MODEL, FAST_MODEL, TOOLS_MODEL, VISION_MODEL, FALLBACK_MODEL

    mode = mode.lower().strip()
    if mode not in ("online", "offline"):
        raise ValueError(f"Unknown mode '{mode}'. Must be 'online' or 'offline'.")

    MODEL_MODE = mode

    if mode == "online":
        PRIMARY_MODEL  = ONLINE_PRIMARY_MODEL
        FAST_MODEL     = ONLINE_FAST_MODEL
        TOOLS_MODEL    = ONLINE_TOOLS_MODEL
        VISION_MODEL   = ONLINE_VISION_MODEL
        FALLBACK_MODEL = ONLINE_FALLBACK_MODEL
    else:
        PRIMARY_MODEL  = OFFLINE_PRIMARY_MODEL
        FAST_MODEL     = OFFLINE_FAST_MODEL
        TOOLS_MODEL    = OFFLINE_TOOLS_MODEL
        VISION_MODEL   = OFFLINE_VISION_MODEL
        FALLBACK_MODEL = None   # no cloud fallback in offline mode

# ========================
# AI SETTINGS
# ========================
TEMPERATURE = 0.45
MAX_TOKENS  = 1000

# ========================
# FILE PATHS
# ========================
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MEMORY_PATH    = os.path.join(BASE_DIR, "data", "memory")
LOGS_PATH      = os.path.join(BASE_DIR, "data", "logs")
PROMPTS_PATH   = os.path.join(BASE_DIR, "data", "prompts")
WORKSPACE_PATH = os.path.join(BASE_DIR, "workspace")

# ========================
# MEMORY SETTINGS
# ========================
MAX_HISTORY               = 10
MEMORY_COLLECTION         = "friday_memory"
SIMILARITY_THRESHOLD      = 0.95
MIN_RELEVANCE             = 0.2
EMBEDDING_MODEL           = "all-MiniLM-L6-v2"
SPARSE_MODEL              = "Qdrant/bm25"
CRAG_CORRECT_THRESHOLD    = 0.7
CRAG_AMBIGUOUS_THRESHOLD  = 0.4
MAX_RAG_RETRIES           = 2

# ========================
# SAFETY
# ========================
AUTO_EXECUTE_CONFIDENCE = 80
ASK_FIRST_CONFIDENCE    = 60
DENY_CONFIDENCE         = 40

# ========================
# VOICE
# ========================
SAMPLE_RATE     = 16000
VAD_SENSITIVITY = 0.5
SILENCE_CHUNKS  = 50

# ========================
# API KEYS
# ========================
GROQ_API_KEY   = (os.getenv("GROQ_API_KEY")   or "").strip() or None
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or "").strip() or None

# ========================
# GENERAL
# ========================
LOG_LEVEL = "INFO"
DEBUG     = False
