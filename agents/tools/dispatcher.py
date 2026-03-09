"""
agents/dispatcher.py
====================
FRIDAY Agent Dispatcher

Receives COMPLEX task payloads from the Orchestrator and routes them
to the correct specialist agent. Acts as the sole entry point for all
Phase 2 agentic work.

Task Intent Classification:
    FORM_FILL   → FormAgent   (fill + submit a web form)
    WEB_BROWSE  → BrowserAgent (navigate, click, extract)
    RESEARCH    → ResearchAgent (multi-page deep research)
    UNKNOWN     → Returns a structured error, never hallucinates
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .models import AgentResult, TaskIntent
from .browser_agent import BrowserAgent
from .form_agent import FormAgent, FormPayload
from .research_agent import ResearchAgent

logger = logging.getLogger("FRIDAY.Dispatcher")


# ─────────────────────────────────────────────
#  Task Definition
# ─────────────────────────────────────────────

@dataclass
class AgentTask:
    """Structured task payload passed from the Orchestrator."""
    raw_instruction: str
    intent: TaskIntent = TaskIntent.UNKNOWN
    url: Optional[str] = None
    query: Optional[str] = None
    form_data: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────
#  Intent Classifier
# ─────────────────────────────────────────────

_FORM_KEYWORDS    = re.compile(r"\b(fill|submit|complete|sign up|register|enter.*form|form.*fill)\b", re.I)
_BROWSE_KEYWORDS  = re.compile(r"\b(open|go to|visit|navigate|click|browse|find on|scrape)\b", re.I)
_RESEARCH_KEYWORDS= re.compile(r"\b(research|find out|what is|summarize|compare|look up|latest|news about|deep dive)\b", re.I)
_URL_PATTERN      = re.compile(r"https?://[^\s]+|www\.[^\s]+", re.I)


def classify_intent(instruction: str) -> tuple[TaskIntent, Optional[str]]:
    """
    Lightweight regex classifier. Returns (intent, extracted_url | None).
    The Orchestrator's LLM call should pre-classify when possible; this is
    a fallback / validator.
    """
    url_match = _URL_PATTERN.search(instruction)
    url = url_match.group(0) if url_match else None

    if _FORM_KEYWORDS.search(instruction):
        return TaskIntent.FORM_FILL, url
    if _BROWSE_KEYWORDS.search(instruction):
        return TaskIntent.WEB_BROWSE, url
    if _RESEARCH_KEYWORDS.search(instruction):
        return TaskIntent.RESEARCH, url

    # Fallback: if a URL was given but intent unclear, browse it
    if url:
        return TaskIntent.WEB_BROWSE, url

    return TaskIntent.UNKNOWN, None


# ─────────────────────────────────────────────
#  Dispatcher
# ─────────────────────────────────────────────

class AgentDispatcher:
    """
    Central dispatcher for FRIDAY's agent layer.

    Usage (from Orchestrator):
        dispatcher = AgentDispatcher(confirm_callback=self._ask_user)
        result = await dispatcher.dispatch(task)
    """

    def __init__(self, confirm_callback=None):
        """
        Args:
            confirm_callback: An async callable(prompt: str) -> bool that asks
                              the user for yes/no confirmation. If None, any
                              action requiring confirmation is auto-declined
                              for safety.
        """
        self._confirm = confirm_callback or self._safe_decline
        self.browser  = BrowserAgent()
        self.form     = FormAgent(browser=self.browser, confirm_callback=self._confirm)
        self.research = ResearchAgent()

    # ── Public Entry Point ──────────────────────────────────────────────

    async def dispatch(self, raw_instruction: str, context: dict = None) -> AgentResult:
        """
        Parse the instruction, build an AgentTask, route to specialist.

        Args:
            raw_instruction : Natural-language command from the user/orchestrator.
            context         : Optional dict from memory manager (episodic snippets, etc.)

        Returns:
            AgentResult with success flag, summary, and structured data.
        """
        context = context or {}
        intent, url = classify_intent(raw_instruction)

        task = AgentTask(
            raw_instruction=raw_instruction,
            intent=intent,
            url=url,
            query=raw_instruction,
            context=context,
        )

        logger.info(f"[Dispatcher] Intent={intent.name} | URL={url}")

        try:
            if intent == TaskIntent.FORM_FILL:
                return await self._handle_form(task)
            elif intent == TaskIntent.WEB_BROWSE:
                return await self._handle_browse(task)
            elif intent == TaskIntent.RESEARCH:
                return await self._handle_research(task)
            else:
                return AgentResult(
                    success=False,
                    intent=intent,
                    summary="I couldn't determine what kind of task this is. "
                            "Please rephrase with a URL or clearer action verb.",
                    error="UNKNOWN_INTENT",
                )
        except Exception as exc:
            logger.exception(f"[Dispatcher] Agent failed: {exc}")
            return AgentResult(
                success=False,
                intent=intent,
                summary=f"Agent encountered an error: {exc}",
                error=str(exc),
            )

    # ── Route Handlers ──────────────────────────────────────────────────

    async def _handle_form(self, task: AgentTask) -> AgentResult:
        if not task.url:
            return AgentResult(
                success=False,
                intent=task.intent,
                summary="A URL is required to fill a form. Please provide one.",
                error="MISSING_URL",
            )

        payload = FormPayload(
            url=task.url,
            instruction=task.raw_instruction,
            prefill_data=task.context.get("user_profile", {}),
        )
        return await self.form.execute(payload)

    async def _handle_browse(self, task: AgentTask) -> AgentResult:
        if not task.url:
            # Try to extract from context or ask to search first
            return AgentResult(
                success=False,
                intent=task.intent,
                summary="Please provide a URL to browse.",
                error="MISSING_URL",
            )
        return await self.browser.navigate_and_extract(task.url, task.raw_instruction)

    async def _handle_research(self, task: AgentTask) -> AgentResult:
        return await self.research.deep_research(task.query, seed_url=task.url)

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    async def _safe_decline(prompt: str) -> bool:
        """Default confirm callback — always declines for safety."""
        logger.warning(f"[Dispatcher] Auto-declining confirmation (no callback set): {prompt}")
        return False