"""
tools/search.py — Real web search for FRIDAY

Tier 1: Tavily  — 1000 req/month free, needs TAVILY_API_KEY (best quality for RAG)
Tier 2: DuckDuckGo via `ddgs`  — free, no API key, sync wrapped in thread (always works)

Priority: Tavily if key present, else DuckDuckGo.
Wire-in: manager.py calls search_for_crag(query) → returns list[dict]
         compatible with the existing crag_web_search() format.
"""

import asyncio
import logging
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

log = logging.getLogger("search")

# ========================
# OPTIONAL KEYS — from .env
# ========================
_TAVILY_KEY = os.getenv("TAVILY_API_KEY", "").strip()

MAX_RESULTS  = 5    # how many results to return
DDGS_TIMEOUT = 10   # seconds for ddgs sync call


# ========================
# TIER 2 — DuckDuckGo (free fallback, no key)
# DDGS is sync, so we wrap in asyncio.to_thread
# ========================

async def _search_ddgs(query: str) -> list[dict]:
    """DuckDuckGo text search — free, no API key required."""
    def _run():
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
            with DDGS(timeout=DDGS_TIMEOUT) as ddgs:
                results = ddgs.text(
                    query,
                    max_results=MAX_RESULTS,
                    safesearch="off"
                )
                if not results:
                    return []
                return [
                    {
                        "source":    "ddgs",
                        "title":     r.get("title", ""),
                        "url":       r.get("href", ""),
                        "content":   r.get("body", ""),
                        "relevance": 0.75,
                    }
                    for r in results
                    if r.get("body", "").strip()
                ]
        except Exception as e:
            log.warning(f"[DuckDuckGo] error: {e}")
            return []

    return await asyncio.to_thread(_run)


# ========================
# TIER 1 — Tavily (best quality for RAG, 1000 req/month free)
# Purpose-built for LLM/RAG — returns clean pre-extracted content
# ========================

async def _search_tavily(query: str) -> list[dict]:
    """Tavily AI search — best quality for RAG, needs TAVILY_API_KEY in .env."""
    if not _TAVILY_KEY:
        return []
    try:
        from tavily import AsyncTavilyClient
        client = AsyncTavilyClient(api_key=_TAVILY_KEY)
        resp   = await client.search(
            query,
            search_depth="advanced",
            max_results=MAX_RESULTS,
            include_raw_content=True,
            include_answer=False,
        )
        results = resp.get("results", [])
        return [
            {
                "source":    "tavily",
                "title":     r.get("title", ""),
                "url":       r.get("url", ""),
                "content":   r.get("raw_content", r.get("content", "")),
                "relevance": r.get("score", 0.8),
            }
            for r in results
            if r.get("raw_content", r.get("content", "")).strip()
        ]
    except Exception as e:
        log.warning(f"[Tavily] error: {e}")
        return []


# ========================
# ORCHESTRATOR
# Best available tier runs first. Falls back automatically.
# ========================

async def search(query: str) -> list[dict]:
    """
    Run the best available search tier, fall back if it fails or returns empty.
    Returns list of dicts: {source, title, url, content, relevance}
    """
    if not query or not query.strip():
        return []

    # Try Tavily first if key exists
    if _TAVILY_KEY:
        results = await _search_tavily(query)
        if results:
            log.info(f"[search] Tavily returned {len(results)} results")
            return results

    # Fallback to DuckDuckGo
    results = await _search_ddgs(query)
    if results:
        log.info(f"[search] DuckDuckGo returned {len(results)} results")
        return results

    log.error(f"[search] ALL tiers failed for: {query[:50]}")
    return []


# ========================
# CONTEXT STRING — for direct injection into brain.py / manager.py
# ========================

def results_to_context(results: list[dict]) -> str:
    """
    Convert search results to a plain string for LLM context injection.
    Compatible with manager.py's compress_results() format.
    """
    if not results:
        return ""
    parts = []
    for r in results:
        title   = r.get("title", "").strip()
        content = r.get("content", "").strip()
        if content:
            parts.append(f"[WEB: {title}]\n{content}" if title else f"[WEB]\n{content}")
    return "\n\n".join(parts)


# ========================
# CRAG-COMPATIBLE WRAPPER — plug this into manager.py
# Returns same shape as crag_web_search() currently does
# ========================

async def search_for_crag(query: str) -> list[dict]:
    """
    Drop-in replacement for manager.py's crag_web_search().
    Returns CRAG-compatible result dicts with source/content/relevance.
    """
    raw = await search(query)
    if not raw:
        return []

    # Group all results into one CRAG-compatible result dict
    return [{
        "source": "web_fallback",  # best source label
        "content":   [{"text": r["content"], "score": r["relevance"]} for r in raw],
        "relevance": max(r["relevance"] for r in raw),
    }]


# ========================
# SELF-TEST
# ========================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    # suppress noisy httpx logs from ddgs internals
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    async def _test():
        queries = [
            "latest developments in AI agents 2025",
            "current Bitcoin price today",
            "Python 3.13 new features what changed",
        ]
        for q in queries:
            print(f"\n{'='*60}")
            print(f"Query: {q}")
            results = await search(q)
            if results:
                print(f"✅ Source: {results[0]['source']} | Count: {len(results)}")
                print(f"   Title: {results[0]['title']}")
                print(f"   Snippet: {results[0]['content'][:200]}")
            else:
                print("❌ NO RESULTS — all tiers failed")

        print(f"\n{'='*60}")
        print("CRAG wrapper test:")
        crag = await search_for_crag("what is groq")
        if crag:
            top = crag[0]
            print(f"✅ Source: {top['source']} | Relevance: {top['relevance']}")
            print(f"   Items: {len(top['content'])}")
            print(f"   First text: {top['content'][0]['text'][:200]}")
        else:
            print("❌ CRAG wrapper returned nothing")

    asyncio.run(_test())
