"""
agents/tools/research_agent.py
================================
FRIDAY ResearchAgent

Performs verified, multi-source research using Tavily (primary) or
DuckDuckGo (fallback), then applies CRAG-style grading before surfacing
any content to the user.

Pipeline:
    query → search (Tavily/DDG) → fetch top pages → grade each source →
    filter to CORRECT/AMBIGUOUS → synthesise answer → return AgentResult

FRIDAY principle: If grading scores are all INCORRECT, the agent says
"I could not find verified information" rather than hallucinating.

CRAG Thresholds (mirrors memory/manager.py):
    CORRECT   : relevance ≥ 0.7
    AMBIGUOUS : relevance ≥ 0.4
    INCORRECT : relevance <  0.4  → discarded
"""

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from .models import AgentResult, TaskIntent

logger = logging.getLogger("FRIDAY.ResearchAgent")

# ── Optional deps ────────────────────────────────────────────────────────────
try:
    from tavily import AsyncTavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

try:
    from duckduckgo_search import AsyncDDGS
    DDG_AVAILABLE = True
except ImportError:
    DDG_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


# ─────────────────────────────────────────────
#  CRAG Thresholds (keep in sync with memory/manager.py)
# ─────────────────────────────────────────────
CRAG_CORRECT_THRESHOLD   = 0.70
CRAG_AMBIGUOUS_THRESHOLD = 0.40

# ─────────────────────────────────────────────
#  Data Classes
# ─────────────────────────────────────────────

@dataclass
class SourceResult:
    url: str
    title: str
    snippet: str
    full_text: str = ""
    relevance_score: float = 0.0
    crag_grade: str = "UNGRADED"   # CORRECT | AMBIGUOUS | INCORRECT


@dataclass
class ResearchReport:
    query: str
    answer: str
    sources: list[SourceResult]
    confidence: str              # HIGH | MEDIUM | LOW | NONE
    search_engine: str
    elapsed_ms: int


# ─────────────────────────────────────────────
#  ResearchAgent
# ─────────────────────────────────────────────

class ResearchAgent:
    """
    Multi-source research with CRAG grading.

    Usage:
        agent  = ResearchAgent()
        result = await agent.deep_research("latest Python 3.13 features")
    """

    def __init__(
        self,
        max_sources: int = 5,
        max_full_pages: int = 3,
        browser=None,
    ):
        self.max_sources   = max_sources
        self.max_full_pages = max_full_pages
        self._browser      = browser     # Optional BrowserAgent for full-page fetch
        self._embedder     = None        # Lazy-loaded SentenceTransformer

    # ── Public Entry Point ──────────────────────────────────────────────

    async def deep_research(
        self,
        query: str,
        seed_url: Optional[str] = None,
        num_sources: Optional[int] = None,
    ) -> AgentResult:
        if not query:
            return AgentResult(
                success=False, intent=TaskIntent.RESEARCH,
                summary="No query provided.", error="EMPTY_QUERY",
            )

        t_start = time.monotonic()
        sources, engine = await self._fetch_sources(query, seed_url, num_sources or self.max_sources)

        if not sources:
            return AgentResult(
                success=False, intent=TaskIntent.RESEARCH,
                summary="No search results found. Both Tavily and DuckDuckGo returned empty.",
                error="NO_RESULTS",
            )

        # ── Grade sources ────────────────────────────────────────────────
        graded = await self._grade_sources(query, sources)
        kept   = [s for s in graded if s.crag_grade in ("CORRECT", "AMBIGUOUS")]

        if not kept:
            return AgentResult(
                success=False, intent=TaskIntent.RESEARCH,
                summary=(
                    "Research complete but all sources were graded INCORRECT "
                    "for this query. I cannot provide a verified answer.\n\n"
                    f"Raw search results ({len(sources)}) are attached for manual review."
                ),
                data={"raw_sources": [s.__dict__ for s in graded]},
            )

        # ── Synthesise ───────────────────────────────────────────────────
        answer, confidence = self._synthesise(query, kept)
        elapsed = int((time.monotonic() - t_start) * 1000)

        report = ResearchReport(
            query=query,
            answer=answer,
            sources=kept,
            confidence=confidence,
            search_engine=engine,
            elapsed_ms=elapsed,
        )

        return AgentResult(
            success=True,
            intent=TaskIntent.RESEARCH,
            summary=self._format_report(report),
            data={
                "report": report.__dict__,
                "sources": [s.__dict__ for s in kept],
                "discarded_sources": len(graded) - len(kept),
            },
        )

    # ── Search ──────────────────────────────────────────────────────────

    async def _fetch_sources(
        self, query: str, seed_url: Optional[str], limit: int
    ) -> tuple[list[SourceResult], str]:
        """Try Tavily → DuckDuckGo → BrowserAgent seed URL."""
        sources = []
        engine  = "none"

        # ── Tavily ───────────────────────────────────────────────────────
        if TAVILY_AVAILABLE and os.getenv("TAVILY_API_KEY"):
            try:
                sources = await self._search_tavily(query, limit)
                engine  = "Tavily"
                logger.info(f"[ResearchAgent] Tavily returned {len(sources)} results.")
            except Exception as exc:
                logger.warning(f"[ResearchAgent] Tavily failed: {exc}")

        # ── DuckDuckGo fallback ──────────────────────────────────────────
        if not sources and DDG_AVAILABLE:
            try:
                sources = await self._search_ddg(query, limit)
                engine  = "DuckDuckGo"
                logger.info(f"[ResearchAgent] DDG returned {len(sources)} results.")
            except Exception as exc:
                logger.warning(f"[ResearchAgent] DDG failed: {exc}")

        # ── Seed URL supplement ──────────────────────────────────────────
        if seed_url:
            seed = await self._fetch_url_as_source(seed_url, query)
            if seed:
                sources.insert(0, seed)  # Prioritise user-supplied URL
                engine = f"{engine}+seed"

        return sources[:limit], engine

    @staticmethod
    async def _search_tavily(query: str, limit: int) -> list[SourceResult]:
        client  = AsyncTavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        resp    = await client.search(
            query=query,
            max_results=limit,
            search_depth="advanced",
            include_raw_content=True,
        )
        results = []
        for r in resp.get("results", []):
            results.append(SourceResult(
                url=r.get("url", ""),
                title=r.get("title", ""),
                snippet=r.get("content", "")[:400],
                full_text=(r.get("raw_content") or r.get("content") or "")[:6000],
            ))
        return results

    @staticmethod
    async def _search_ddg(query: str, limit: int) -> list[SourceResult]:
        results = []
        async with AsyncDDGS() as ddgs:
            async for r in ddgs.atext(query, max_results=limit):
                results.append(SourceResult(
                    url=r.get("href", ""),
                    title=r.get("title", ""),
                    snippet=r.get("body", "")[:400],
                    full_text=r.get("body", "")[:2000],
                ))
        return results

    async def _fetch_url_as_source(self, url: str, query: str) -> Optional[SourceResult]:
        if not self._browser:
            return None
        try:
            result = await self._browser.navigate_and_extract(url)
            if result.success:
                snap = result.data["snapshot"]
                return SourceResult(
                    url=snap.url,
                    title=snap.title,
                    snippet=snap.text_content[:400],
                    full_text=snap.text_content[:6000],
                )
        except Exception as exc:
            logger.warning(f"[ResearchAgent] Seed URL fetch failed: {exc}")
        return None

    # ── CRAG Grading ────────────────────────────────────────────────────

    async def _grade_sources(self, query: str, sources: list[SourceResult]) -> list[SourceResult]:
        """
        Score each source for relevance to the query.
        Uses sentence-transformers cosine similarity if available,
        otherwise falls back to keyword overlap.
        """
        if ST_AVAILABLE:
            return await self._grade_semantic(query, sources)
        return self._grade_keyword(query, sources)

    async def _grade_semantic(self, query: str, sources: list[SourceResult]) -> list[SourceResult]:
        """Cosine similarity between query embedding and source text embeddings."""
        if self._embedder is None:
            logger.info("[ResearchAgent] Loading sentence-transformer model…")
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")

        q_emb = self._embedder.encode(query, convert_to_tensor=True)
        for s in sources:
            text  = f"{s.title} {s.snippet}"
            s_emb = self._embedder.encode(text, convert_to_tensor=True)
            score = float(util.cos_sim(q_emb, s_emb))
            s.relevance_score = round(score, 4)
            s.crag_grade      = self._score_to_grade(score)

        sources.sort(key=lambda x: x.relevance_score, reverse=True)
        return sources

    def _grade_keyword(self, query: str, sources: list[SourceResult]) -> list[SourceResult]:
        """Fallback: keyword overlap ratio."""
        q_tokens = set(re.findall(r"\w+", query.lower()))
        for s in sources:
            text     = f"{s.title} {s.snippet}".lower()
            s_tokens = set(re.findall(r"\w+", text))
            if not q_tokens:
                score = 0.0
            else:
                overlap = len(q_tokens & s_tokens)
                score   = overlap / len(q_tokens)
            s.relevance_score = round(min(score * 1.2, 1.0), 4)  # gentle scaling
            s.crag_grade      = self._score_to_grade(s.relevance_score)
        sources.sort(key=lambda x: x.relevance_score, reverse=True)
        return sources

    @staticmethod
    def _score_to_grade(score: float) -> str:
        if score >= CRAG_CORRECT_THRESHOLD:
            return "CORRECT"
        if score >= CRAG_AMBIGUOUS_THRESHOLD:
            return "AMBIGUOUS"
        return "INCORRECT"

    # ── Synthesis ───────────────────────────────────────────────────────

    def _synthesise(self, query: str, sources: list[SourceResult]) -> tuple[str, str]:
        """
        Concatenate top-graded source text for LLM synthesis (or direct return).
        The calling Orchestrator should pass this to the LLM for final wording.
        Here we return the raw passages + confidence.
        """
        correct   = [s for s in sources if s.crag_grade == "CORRECT"]
        ambiguous = [s for s in sources if s.crag_grade == "AMBIGUOUS"]

        if len(correct) >= 2:
            confidence = "HIGH"
        elif len(correct) == 1:
            confidence = "MEDIUM"
        elif ambiguous:
            confidence = "LOW"
        else:
            confidence = "NONE"

        passages = []
        for s in (correct + ambiguous)[:self.max_full_pages]:
            passages.append(
                f"[Source: {s.title} | {s.url} | Grade: {s.crag_grade} | Score: {s.relevance_score}]\n"
                f"{s.full_text[:1500]}\n"
            )

        answer = "\n---\n".join(passages) if passages else "No verified content available."
        return answer, confidence

    @staticmethod
    def _format_report(report: ResearchReport) -> str:
        lines = [
            f"🔍 Research complete | Query: '{report.query}'",
            f"   Engine    : {report.search_engine}",
            f"   Confidence: {report.confidence}",
            f"   Sources   : {len(report.sources)} verified | {report.elapsed_ms}ms",
            "",
        ]
        for i, s in enumerate(report.sources[:5], 1):
            grade_icon = {"CORRECT": "✅", "AMBIGUOUS": "⚠️", "INCORRECT": "❌"}.get(s.crag_grade, "?")
            lines.append(f"  {i}. {grade_icon} [{s.relevance_score:.2f}] {s.title}")
            lines.append(f"     {s.url}")
            lines.append(f"     {s.snippet[:120]}…")
        return "\n".join(lines)