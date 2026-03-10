"""
core/safety.py
===============
FRIDAY Safety Layer — STUB

Full implementation coming after tools are built.
Currently auto-approves all actions so the system runs.
"""

from enum import Enum


class Decision(Enum):
    AUTO = "auto"
    ASK  = "ask"
    DENY = "deny"


async def check(tool, arguments: dict) -> Decision:
    """Stub — approves everything. Replace with real logic later."""
    return Decision.AUTO