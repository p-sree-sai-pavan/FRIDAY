"""
memory/api.py
=============
The public API for the FRIDAY Memory Manager.
"""

import asyncio

from .core.resources import _res
from .core.sqlite import write_semantic, search_semantic, get_user_profile, save_user_profile, save_feedback
from .core.qdrant import write_episodic, search_episodic
from .core.working import search_working
from .core.pipeline import hyde_transform, crag_grade, crag_web_search, compress_results
import config


async def read(query: str) -> list:
    """Full pipeline: HyDE -> Hybrid search -> CRAG grade -> web fallback."""
    count = 0
    if _res.qdrant_available:
        try:
            def _count():
                return _res.qdrant.count(collection_name=config.MEMORY_COLLECTION).count
            count = await asyncio.to_thread(_count)
        except Exception:
            count = 0

    expanded = await hyde_transform(query) if count > 0 else query

    results = await asyncio.gather(
        search_working(expanded),
        search_episodic(expanded),
        search_semantic(query)
    )
    results = [r for r in results if r["content"]]
    results.sort(key=lambda x: x["relevance"], reverse=True)

    if not results:
        return await crag_web_search(query)

    grade = await crag_grade(query, results)

    if grade["correct"]:
        return grade["correct"]
    if grade["ambiguous"]:
        web = await crag_web_search(query)
        return grade["ambiguous"] + web

    return await crag_web_search(query)


async def write(prompt: str, response: str):
    """Write a conversation exchange to episodic memory."""
    await write_episodic(prompt, response)


# Expose explicitly for imports
__all__ = [
    "read",
    "write",
    "write_semantic",
    "get_user_profile",
    "save_user_profile",
    "save_feedback",
    "compress_results",
]
