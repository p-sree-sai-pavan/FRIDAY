"""
memory/core/qdrant.py
=====================
Handles Qdrant-based episodic memory (vector search).
"""

import asyncio
import logging
import uuid
import time

from .resources import _res
from .utils import encode_dense, encode_sparse
import config

log = logging.getLogger("memory.qdrant")


async def search_episodic(query: str) -> dict:
    if not _res.qdrant_available:
        return {"source": "episodic", "content": [], "relevance": 0}
    try:
        from qdrant_client.models import Prefetch, FusionQuery, Fusion

        d_vec = (await encode_dense(query)).tolist()
        s_vec = await encode_sparse(query)

        def _query():
            return _res.qdrant.query_points(
                collection_name=config.MEMORY_COLLECTION,
                prefetch=[
                    Prefetch(query=d_vec, using="dense", limit=10),
                    Prefetch(query=s_vec, using="sparse", limit=10)
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=5
            ).points
        results = await asyncio.to_thread(_query)

        if not results:
            return {"source": "episodic", "content": [], "relevance": 0}

        return {
            "source": "episodic",
            "content": [{"text": r.payload.get("text", "")} for r in results],
            "relevance": sum(r.score for r in results) / len(results)
        }
    except Exception as e:
        log.warning(f"[Memory] Episodic search error: {e}")
        return {"source": "episodic", "content": [], "relevance": 0}


async def write_episodic(prompt: str, response: str):
    if not _res.qdrant_available:
        log.debug("[Memory] Qdrant unavailable — skipping episodic write")
        return
    try:
        from qdrant_client.models import (
            PointStruct, Prefetch, FusionQuery, Fusion
        )

        text  = f"User: {prompt}\nFRIDAY: {response}"
        d_vec = (await encode_dense(text)).tolist()
        s_vec = await encode_sparse(text)

        def _write():
            _res.qdrant.upsert(
                collection_name=config.MEMORY_COLLECTION,
                points=[
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector={
                            "dense": d_vec,
                            "sparse": s_vec
                        },
                        payload={
                            "text": text,
                            "timestamp": int(time.time()),
                            "type": "conversation"
                        }
                    )
                ]
            )
        await asyncio.to_thread(_write)
    except Exception as e:
        log.warning(f"[Memory] Episodic write error: {e}")
