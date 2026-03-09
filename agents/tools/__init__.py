"""
FRIDAY Agent Subsystem — Phase 2
"""
from .models import AgentResult, TaskIntent
from .dispatcher import AgentDispatcher
from .browser_agent import BrowserAgent
from .form_agent import FormAgent, FormPayload
from .research_agent import ResearchAgent

__all__ = [
    "AgentDispatcher",
    "AgentResult",
    "TaskIntent",
    "BrowserAgent",
    "FormAgent",
    "FormPayload",
    "ResearchAgent",
]