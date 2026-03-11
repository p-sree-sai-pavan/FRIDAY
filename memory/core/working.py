"""
memory/core/working.py
======================
Handles Working Memory (recent conversation history from JSON).
Owned by brain.py, but read by manager.py for context.
"""

import asyncio
import json
import logging
import os

from .utils import cosine_similarity, encode_dense
from .resources import HISTORY_FILE
import config

log = logging.getLogger("memory.working")

_working_cache: dict = {}
_CACHE_MAX  = 500
_CACHE_TRIM = 100


async def search_working(query: str) -> dict:
    try:
        if not os.path.exists(HISTORY_FILE):
            return {"source": "working", "content": [], "relevance": 0}

        def _load():
            with open(HISTORY_FILE, encoding="utf-8") as f:
                return json.load(f)
        history = await asyncio.to_thread(_load)

        if not history:
            return {"source": "working", "content": [], "relevance": 0}

        recent = history[-config.MAX_HISTORY:]
        q_vec = await encode_dense(query)

        scored = []
        for msg in recent:
            if msg.get("role") == "system":
                continue
            content = msg.get("content", "")
            if not content:
                continue

            if content not in _working_cache:
                if len(_working_cache) >= _CACHE_MAX:
                    for k in list(_working_cache.keys())[:_CACHE_TRIM]:
                        del _working_cache[k]
                _working_cache[content] = await encode_dense(content)
            
            score = cosine_similarity(q_vec, _working_cache[content])
            if score >= config.MIN_RELEVANCE:
                scored.append({"content": content, "role": msg.get("role", ""), "score": score})

        scored.sort(key=lambda x: x["score"], reverse=True)
        top = scored[:5]
        avg = sum(r["score"] for r in top) / len(top) if top else 0
        return {"source": "working", "content": top, "relevance": avg}

    except Exception as e:
        log.warning(f"[Memory] Working memory search error: {e}")
        return {"source": "working", "content": [], "relevance": 0}
