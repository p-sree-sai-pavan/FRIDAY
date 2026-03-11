"""
memory/core/utils.py
====================
Helper functions for memory operations: cosine similarity, dense encoding, sparse encoding, and common stopwords.
"""

import asyncio

import numpy as np

from .resources import _res

STOPWORDS = {
    "the","and","for","are","but","not","you","all","can","had","her",
    "was","one","our","out","day","get","has","him","his","how","its",
    "may","new","now","old","see","two","who","did","have","what","when",
    "with","from","that","this","will","your","been","does","just","into",
    "over","such","than","then","they","them","also","pavan","tell",
    "show","give","find","make","like","know","want","need","help","about"
}


def cosine_similarity(a, b) -> float:
    a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


async def encode_dense(text: str) -> np.ndarray:
    return await asyncio.to_thread(
        lambda: _res.embedder.encode(text, show_progress_bar=False)
    )


async def encode_sparse(text: str):
    from qdrant_client.models import SparseVector
    def _run():
        r = list(_res.sparse_embedder.embed([text]))[0]
        return SparseVector(indices=r.indices.tolist(), values=r.values.tolist())
    return await asyncio.to_thread(_run)
