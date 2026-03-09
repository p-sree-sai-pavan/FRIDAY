"""
agents/models.py
=================
Shared data structures for the FRIDAY agent layer.

Kept in their own file so dispatcher.py and all tool agents
can import from here without circular dependencies.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional


class TaskIntent(Enum):
    FORM_FILL  = auto()
    WEB_BROWSE = auto()
    RESEARCH   = auto()
    UNKNOWN    = auto()


@dataclass
class AgentResult:
    success: bool
    intent: TaskIntent
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    requires_confirmation: bool = False
    confirmation_prompt: Optional[str] = None
    error: Optional[str] = None