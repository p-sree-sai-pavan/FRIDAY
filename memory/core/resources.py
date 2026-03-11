"""
memory/core/resources.py
========================
Manages the lazy initialization of all heavy resources: ML models, Qdrant, and SQLite.
Loading is deferred until first access to ensure instant startup.
"""

import atexit
import logging
import os
import sqlite3
import sys
import threading

import psutil
from groq import AsyncGroq

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config

log = logging.getLogger("memory.resources")

# Constants
QDRANT_PATH  = os.path.join(config.MEMORY_PATH, "qdrant")
HISTORY_FILE = os.path.join(config.MEMORY_PATH, "memory.json")
SEMANTIC_DB  = os.path.join(config.MEMORY_PATH, "facts.db")

os.makedirs(config.MEMORY_PATH, exist_ok=True)
os.makedirs(QDRANT_PATH, exist_ok=True)

# Shared Groq Client (no network call until first use)
groq_client = AsyncGroq(api_key=config.GROQ_API_KEY or "")


class _LazyResources:
    """Thread-safe lazy loader for ML models, Qdrant, and SQLite."""

    def __init__(self):
        self._lock = threading.Lock()
        self._embedder = None
        self._sparse_embedder = None
        self._vector_size = None
        self._qdrant = None
        self._db = None
        self._db_lock = threading.Lock()   # serializes SQLite writes
        self._initialized = False

    # --- ML Models ---

    @property
    def embedder(self):
        if self._embedder is None:
            self._init_models()
        return self._embedder

    @property
    def sparse_embedder(self):
        if self._sparse_embedder is None:
            self._init_models()
        return self._sparse_embedder

    @property
    def vector_size(self):
        if self._vector_size is None:
            self._init_models()
        return self._vector_size

    def _init_models(self):
        with self._lock:
            if self._embedder is not None:
                return
            log.info("[Memory] Loading embedding models (first-time only)...")
            from sentence_transformers import SentenceTransformer
            from fastembed import SparseTextEmbedding
            self._embedder = SentenceTransformer(config.EMBEDDING_MODEL)
            self._sparse_embedder = SparseTextEmbedding(model_name=config.SPARSE_MODEL)
            self._vector_size = self._embedder.get_sentence_embedding_dimension()
            log.info(f"[Memory] Models loaded. Dense dim={self._vector_size}")

    # --- Qdrant ---

    @property
    def qdrant(self):
        if self._qdrant is None:
            self._init_qdrant()
        return self._qdrant

    @property
    def qdrant_available(self) -> bool:
        """Check if Qdrant is available without triggering init."""
        if self._qdrant is not None:
            return True
        try:
            self._init_qdrant()
            return self._qdrant is not None
        except Exception:
            return False

    def _clear_stale_lock(self):
        """Remove Qdrant's .lock file if no other FRIDAY/python process holds it."""
        lock_file = os.path.join(QDRANT_PATH, ".lock")
        if not os.path.exists(lock_file):
            return

        current_pid = os.getpid()
        other_python_running = False
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                if proc.info["pid"] == current_pid:
                    continue
                pname = (proc.info["name"] or "").lower()
                if "python" in pname:
                    cmdline = proc.info.get("cmdline") or []
                    cmd_str = " ".join(cmdline).lower()
                    if "friday" in cmd_str or "main.py" in cmd_str:
                        other_python_running = True
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if other_python_running:
            log.warning("[Memory] Another FRIDAY instance is running — cannot clear Qdrant lock")
            return

        try:
            os.remove(lock_file)
            log.info("[Memory] Removed stale Qdrant lock file")
        except OSError as e:
            log.warning(f"[Memory] Could not remove lock file: {e}")

    def _init_qdrant(self):
        with self._lock:
            if self._qdrant is not None:
                return
            from qdrant_client import QdrantClient
            from qdrant_client.models import (
                Distance, VectorParams, SparseVectorParams, SparseIndexParams
            )
            log.info("[Memory] Initializing Qdrant vector store...")

            for attempt in range(2):
                try:
                    self._qdrant = QdrantClient(path=QDRANT_PATH)
                    break
                except Exception as e:
                    if attempt == 0 and "already running" in str(e).lower():
                        log.warning(f"[Memory] Qdrant locked — attempting stale lock cleanup")
                        self._clear_stale_lock()
                        continue
                    log.error(f"[Memory] Qdrant init failed: {e}")
                    log.warning("[Memory] Episodic memory DISABLED — FRIDAY will still work without it")
                    return

            if self._qdrant is None:
                return

            try:
                existing = [c.name for c in self._qdrant.get_collections().collections]
                if config.MEMORY_COLLECTION not in existing:
                    self._qdrant.create_collection(
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
                    log.info(f"[Memory] Created collection: {config.MEMORY_COLLECTION}")
                else:
                    log.info(f"[Memory] Using existing collection: {config.MEMORY_COLLECTION}")
            except Exception as e:
                log.error(f"[Memory] Collection setup failed: {e}")
                self._qdrant = None

    # --- SQLite ---

    @property
    def db(self):
        if self._db is None:
            self._init_db()
        return self._db

    @property
    def db_lock(self):
        return self._db_lock

    def _init_db(self):
        with self._lock:
            if self._db is not None:
                return
            log.info("[Memory] Initializing SQLite facts database...")
            self._db = sqlite3.connect(SEMANTIC_DB, check_same_thread=False)
            self._db.execute("""CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY, category TEXT, key TEXT UNIQUE,
                value TEXT, updated_at INTEGER
            )""")
            self._db.execute("""CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY, query TEXT, context TEXT,
                answer TEXT, was_useful INTEGER, timestamp INTEGER
            )""")
            self._db.execute("""CREATE TABLE IF NOT EXISTS user_profile (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )""")
            self._db.commit()
            log.info("[Memory] SQLite ready.")

    def shutdown(self):
        """Clean up resources on exit."""
        with self._lock:
            if self._qdrant:
                self._qdrant.close()
                self._qdrant = None
            if self._db:
                self._db.close()
                self._db = None

# Global instance
_res = _LazyResources()
atexit.register(_res.shutdown)
