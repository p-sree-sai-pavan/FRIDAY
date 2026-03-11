import os
from dotenv import load_dotenv
load_dotenv()

# ========================
# APP
# ========================
APP_NAME = "FRIDAY"
VERSION = "2.0.0"

# ========================
# AI SETTINGS
# ========================
PRIMARY_MODEL  = "llama-3.3-70b-versatile"
FAST_MODEL     = "llama-3.1-8b-instant"
TOOLS_MODEL    = "llama-3.3-70b-versatile"
VISION_MODEL   = "llama4-scout-17b-16e"
FALLBACK_MODEL = "gemini-2.5-flash"
TEMPERATURE = 0.45
MAX_TOKENS = 1000

# ========================
# FILE PATHS
# ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_PATH = os.path.join(BASE_DIR, "data", "memory")
LOGS_PATH = os.path.join(BASE_DIR, "data", "logs")
PROMPTS_PATH = os.path.join(BASE_DIR, "data", "prompts")
WORKSPACE_PATH = os.path.join(BASE_DIR, "workspace")

# ========================
# MEMORY SETTINGS
# ========================
MAX_HISTORY = 10
MEMORY_COLLECTION = "friday_memory"
SIMILARITY_THRESHOLD = 0.95
MIN_RELEVANCE = 0.2
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SPARSE_MODEL = "Qdrant/bm25"
CRAG_CORRECT_THRESHOLD = 0.7
CRAG_AMBIGUOUS_THRESHOLD = 0.4
MAX_RAG_RETRIES = 2

# ========================
# SAFETY
# ========================
AUTO_EXECUTE_CONFIDENCE = 80
ASK_FIRST_CONFIDENCE = 60
DENY_CONFIDENCE = 40

# ========================
# VOICE
# ========================
SAMPLE_RATE = 16000
VAD_SENSITIVITY = 0.5
SILENCE_CHUNKS = 50

# ========================
# API KEYS (stripped to prevent whitespace issues)
# ========================
GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or "").strip() or None
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or "").strip() or None

# ========================
# GENERAL
# ========================
LOG_LEVEL = "INFO"
DEBUG = False