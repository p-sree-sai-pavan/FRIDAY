"""
memory/core/pipeline.py
=======================
The LLM pipeline for memory: HyDE (query expansion) and CRAG (grading).
"""

import asyncio
import logging
from core.model_client import get_client
import config

log = logging.getLogger("memory.pipeline")


def compress_results(results: list) -> str:
    """Flatten multiple result dictionaries into a searchable context string."""
    text = ""
    for r in results:
        if r["source"] == "working":
            sub = [f"{msg['role']}: {msg['content']}" for msg in r["content"]]
            text += "\n".join(sub) + "\n"
        elif r["source"] == "episodic":
            sub = [msg["text"] for msg in r["content"]]
            text += "\n".join(sub) + "\n"
        elif r["source"] == "semantic":
            sub = [f"FACT -> {msg['category']} | {msg['key']}: {msg['value']}" for msg in r["content"]]
            text += "\n".join(sub) + "\n"
        elif r["source"] == "web_fallback":
            # FIX: was silently dropped before — web search results now included
            sub = [msg["text"] for msg in r["content"]]
            text += "\n".join(sub) + "\n"
    return text.strip()


async def hyde_transform(query: str) -> str:
    """Expand query with hypothetical document embeddings (keywords)."""
    try:
        resp = await get_client().chat.completions.create(
            model=config.FAST_MODEL, temperature=0.3, max_tokens=60,
            messages=[{"role": "user", "content":
                f"Expand into search keywords only. No answer. Query: {query}\nSearch terms:"}]
        )
        return f"{query} {resp.choices[0].message.content.strip()}"
    except Exception as e:
        log.warning(f"[Memory] HyDE transform failed: {e} — using raw query")
        return query


async def _crag_grade_single(query: str, result: dict) -> str:
    """Grade a single retrieved document as CORRECT, AMBIGUOUS, or INCORRECT."""
    avg_score = result.get("relevance", 0)
    if avg_score >= config.CRAG_CORRECT_THRESHOLD:
        return "CORRECT"
    if avg_score < config.CRAG_AMBIGUOUS_THRESHOLD:
        return "INCORRECT"
    try:
        compressed = compress_results([result])
        resp = await get_client().chat.completions.create(
            model=config.FAST_MODEL, temperature=0, max_tokens=10,
            messages=[{"role": "user", "content":
                f"Grade: CORRECT, AMBIGUOUS, or INCORRECT.\nQuery: {query}\nContext: {compressed}\nGrade:"}]
        )
        return resp.choices[0].message.content.strip().upper()
    except Exception as e:
        log.warning(f"[Memory] CRAG grade failed: {e} — defaulting to AMBIGUOUS")
        return "AMBIGUOUS"


async def crag_grade(query: str, results: list) -> dict:
    """Grade all results independently."""
    if not results:
        return {"correct": [], "ambiguous": [], "incorrect": True}
    grades = await asyncio.gather(*[_crag_grade_single(query, r) for r in results])
    correct, ambiguous = [], []
    for result, grade in zip(results, grades):
        if "CORRECT" in grade and "INCORRECT" not in grade:
            correct.append(result)
        elif "AMBIGUOUS" in grade:
            ambiguous.append(result)
    return {
        "correct":   correct,
        "ambiguous": ambiguous,
        "incorrect": len(correct) == 0 and len(ambiguous) == 0
    }


async def crag_web_search(query: str) -> list:
    """Execute a web search if memory lacks the requested info."""
    try:
        from tools.search import search_for_crag
        return await search_for_crag(query)
    except Exception as e:
        log.warning(f"[Memory] CRAG Web Search fallback error: {e}")
        return []
