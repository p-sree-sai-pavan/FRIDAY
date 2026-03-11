"""
memory/core/resources.py
========================
Manages all heavy resources: ML models, Qdrant, and SQLite.

ARCHITECTURE:
  Previously used lazy-property loading with a shared threading.Lock.
  This caused a deadlock on first run: _init_qdrant() held the lock and
  then called self.vector_size → _init_models() → tried to re-acquire
  the same non-reentrant lock → hung forever.

  Now: a single explicit `await _res.initialize()` is called once from
  main.py after pre-flight checks. Steps run sequentially in threads
  with a deterministic order — no shared lock is needed:
    1. Load ML models   (must come first — Qdrant needs vector_size)
    2. Setup Qdrant     (depends on vector_size from step 1)
    3. Setup SQLite     (independent, fast)

  After initialize() completes, all attributes are plain values.
  No properties, no locks, safe to read from any async or sync context.
"""

import asyncio
import atexit
import logging
import os
import sqlite3
import sys
import threading

import psutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config

log = logging.getLogger("memory.resources")

# ========================
# PATHS
# ========================
QDRANT_PATH  = os.path.join(config.MEMORY_PATH, "qdrant")
HISTORY_FILE = os.path.join(config.MEMORY_PATH, "memory.json")
SEMANTIC_DB  = os.path.join(config.MEMORY_PATH, "facts.db")

os.makedirs(config.MEMORY_PATH, exist_ok=True)
os.makedirs(QDRANT_PATH, exist_ok=True)

# Shared Groq async client — no network call until first use


class Resources:
    """
    Holds all initialized resources as plain attributes.

    Usage:
        from memory.core.resources import _res
        await _res.initialize()     # call once in main.py
        _res.embedder.encode(...)   # use anywhere after that
        _res.qdrant.search(...)
        _res.db.execute(...)
    """

    def __init__(self):
        # ML models
        self.embedder:         object = None
        self.sparse_embedder:  object = None
        self.vector_size:      int    = None

        # Qdrant
        self.qdrant:           object = None
        self.qdrant_available: bool   = False

        # SQLite
        self.db:      sqlite3.Connection = None
        self.db_lock: threading.Lock     = threading.Lock()

        self._initialized = False

    # ========================
    # PUBLIC ENTRY POINT
    # ========================

    async def initialize(self) -> None:
        """
        Initialize all resources in the correct dependency order.
        Safe to call multiple times — no-ops after the first call.
        Each step runs inside asyncio.to_thread so the event loop
        is never blocked. Steps are sequential by design — Qdrant
        depends on vector_size which comes from the model step.
        """
        if self._initialized:
            return

        log.info("[Resources] Starting initialization...")

        await asyncio.to_thread(self._load_models)    # Step 1: models (~2-3s)
        await asyncio.to_thread(self._setup_qdrant)   # Step 2: vector store
        await asyncio.to_thread(self._setup_sqlite)   # Step 3: fact DB

        self._initialized = True
        log.info("[Resources] All resources ready.")

    # ========================
    # STEP 1 — ML Models
    # ========================

    def _load_models(self) -> None:
        """Load dense and sparse embedding models. Runs in a thread."""
        log.info("[Resources] Loading embedding models...")
        from sentence_transformers import SentenceTransformer
        from fastembed import SparseTextEmbedding

        self.embedder        = SentenceTransformer(config.EMBEDDING_MODEL)
        self.sparse_embedder = SparseTextEmbedding(model_name=config.SPARSE_MODEL)
        self.vector_size     = self.embedder.get_sentence_embedding_dimension()
        log.info(f"[Resources] Models loaded. Dense dim={self.vector_size}")

    # ========================
    # STEP 2 — Qdrant
    # ========================

    def _clear_stale_qdrant_lock(self) -> None:
        """
        Delete Qdrant's .lock file only if no other FRIDAY process holds it.
        A stale lock is left behind when the process crashes unexpectedly.
        """
        lock_file = os.path.join(QDRANT_PATH, ".lock")
        if not os.path.exists(lock_file):
            return

        current_pid = os.getpid()
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                if proc.info["pid"] == current_pid:
                    continue
                pname = (proc.info["name"] or "").lower()
                if "python" in pname:
                    cmd_str = " ".join(proc.info.get("cmdline") or []).lower()
                    if "friday" in cmd_str or "main.py" in cmd_str:
                        log.warning("[Resources] Another FRIDAY instance running — skipping lock clear")
                        return
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        try:
            os.remove(lock_file)
            log.info("[Resources] Removed stale Qdrant lock file")
        except OSError as e:
            log.warning(f"[Resources] Could not remove lock file: {e}")

    def _setup_qdrant(self) -> None:
        """
        Connect to local Qdrant and ensure the memory collection exists.
        Requires self.vector_size to already be set (from _load_models).
        Runs in a thread — safe to call blocking Qdrant client methods.
        """
        from qdrant_client import QdrantClient
        from qdrant_client.models import (
            Distance, VectorParams, SparseVectorParams, SparseIndexParams
        )
        log.info("[Resources] Initializing Qdrant vector store...")

        for attempt in range(2):
            try:
                self.qdrant = QdrantClient(path=QDRANT_PATH)
                break
            except Exception as e:
                if attempt == 0 and "already running" in str(e).lower():
                    log.warning("[Resources] Qdrant locked — attempting stale lock cleanup")
                    self._clear_stale_qdrant_lock()
                    continue
                log.error(f"[Resources] Qdrant init failed: {e}")
                log.warning("[Resources] Episodic memory DISABLED — FRIDAY will still work without it")
                return

        if self.qdrant is None:
            return

        try:
            existing = [c.name for c in self.qdrant.get_collections().collections]
            if config.MEMORY_COLLECTION not in existing:
                self.qdrant.create_collection(
                    collection_name=config.MEMORY_COLLECTION,
                    vectors_config={
                        "dense": VectorParams(
                            size=self.vector_size,
                            distance=Distance.COSINE
                        )
                    },
                    sparse_vectors_config={
                        "sparse": SparseVectorParams(
                            index=SparseIndexParams(on_disk=False)
                        )
                    }
                )
                log.info(f"[Resources] Created collection: {config.MEMORY_COLLECTION}")
            else:
                log.info(f"[Resources] Using existing collection: {config.MEMORY_COLLECTION}")

            self.qdrant_available = True

        except Exception as e:
            log.error(f"[Resources] Collection setup failed: {e}")
            self.qdrant = None
            self.qdrant_available = False

    # ========================
    # STEP 3 — SQLite
    # ========================

    def _setup_sqlite(self) -> None:
        """Create facts.db and ensure all required tables exist. Runs in a thread."""
        log.info("[Resources] Initializing SQLite facts database...")
        self.db = sqlite3.connect(SEMANTIC_DB, check_same_thread=False)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id         INTEGER PRIMARY KEY,
                category   TEXT,
                key        TEXT UNIQUE,
                value      TEXT,
                updated_at INTEGER
            )
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id         INTEGER PRIMARY KEY,
                query      TEXT,
                context    TEXT,
                answer     TEXT,
                was_useful INTEGER,
                timestamp  INTEGER
            )
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS user_profile (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        self.db.commit()
        log.info("[Resources] SQLite ready.")

    # ========================
    # SHUTDOWN
    # ========================

    def shutdown(self) -> None:
        """Close all connections cleanly. Called automatically on exit via atexit."""
        if self.qdrant:
            self.qdrant.close()
            self.qdrant = None
        if self.db:
            self.db.close()
            self.db = None
        log.info("[Resources] Shutdown complete.")


# ========================
# GLOBAL INSTANCE
# ========================
# Import this wherever resources are needed:
#   from memory.core.resources import _res, groq_client
#
# IMPORTANT: call `await _res.initialize()` once from main.py
# before any memory operations are performed.

_res = Resources()
atexit.register(_res.shutdown)
