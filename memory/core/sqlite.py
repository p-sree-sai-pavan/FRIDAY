"""
memory/core/sqlite.py
=====================
Handles SQLite-based semantic memory (facts, user profiles, feedback).
"""

import asyncio
import logging
import sqlite3
import time

from .resources import _res
from .utils import STOPWORDS

log = logging.getLogger("memory.sqlite")


async def db_execute(sql: str, params: tuple = ()):
    """Write SQLite query in thread — serialized by lock to prevent concurrent write errors."""
    def _run():
        with _res.db_lock:
            cur = _res.db.cursor()
            cur.execute(sql, params)
            _res.db.commit()
    return await asyncio.to_thread(_run)


async def db_read(sql: str, params: tuple = ()):
    """Read-only SQLite query in thread — no lock needed for reads."""
    def _run():
        cur = _res.db.cursor()
        cur.row_factory = sqlite3.Row
        return cur.execute(sql, params).fetchall()
    return await asyncio.to_thread(_run)


async def get_user_profile() -> dict:
    try:
        rows = await db_read("SELECT key, value FROM user_profile")
        return {row["key"]: row["value"] for row in rows}
    except Exception as e:
        log.warning(f"[Memory] User profile read error: {e}")
        return {}


async def save_user_profile(key: str, value: str):
    try:
        await db_execute(
            "INSERT OR REPLACE INTO user_profile (key, value) VALUES (?, ?)",
            (key, value)
        )
        log.info(f"[Memory] Saved profile: {key} = {value}")
    except Exception as e:
        log.error(f"[Memory] Profile save error: {e}")


async def search_semantic(query: str) -> dict:
    """Keyword search across facts table."""
    try:
        words = [w for w in query.lower().split() if len(w) >= 3 and w not in STOPWORDS]
        if not words:
            return {"source": "semantic", "content": [], "relevance": 0}

        conditions, params = [], []
        for word in words:
            conditions.append("(LOWER(category) LIKE ? OR LOWER(key) LIKE ? OR LOWER(value) LIKE ?)")
            params.extend([f"%{word}%", f"%{word}%", f"%{word}%"])

        rows = await db_read(
            f"SELECT DISTINCT category, key, value FROM facts WHERE {' OR '.join(conditions)}",
            tuple(params)
        )

        if not rows:
            return {"source": "semantic", "content": [], "relevance": 0}

        return {
            "source":    "semantic",
            "content":   [{"category": r["category"], "key": r["key"], "value": r["value"]} for r in rows],
            "relevance": 1.0
        }
    except Exception as e:
        log.warning(f"[Memory] Semantic search error: {e}")
        return {"source": "semantic", "content": [], "relevance": 0}


async def write_semantic(category: str, key: str, value: str):
    """Upsert a fact into semantic DB."""
    try:
        await db_execute(
            """INSERT INTO facts (category, key, value, updated_at) 
               VALUES (?, ?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
            (category, key, value, int(time.time()))
        )
        log.info(f"[Memory] Saved fact: [{category}] {key} = {value}")
    except Exception as e:
        log.error(f"[Memory] Semantic write error: {e}")


async def save_feedback(query: str, context: str, answer: str, was_useful: bool):
    try:
        await db_execute(
            "INSERT INTO feedback (query, context, answer, was_useful, timestamp) VALUES (?, ?, ?, ?, ?)",
            (query, context, answer, int(was_useful), int(time.time()))
        )
    except Exception as e:
        log.warning(f"[Memory] Feedback save error: {e}")
