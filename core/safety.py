"""
core/safety.py
===============
FRIDAY Safety Layer

Gates every tool execution by its RiskLevel.
READ/WRITE → auto-execute
SYSTEM     → ask Pavan first
IRREVERSIBLE → always ask, no exceptions
"""

from enum import Enum
import logging

log = logging.getLogger("safety")


class Decision(Enum):
    AUTO = "auto"
    ASK  = "ask"
    DENY = "deny"


async def check(tool, arguments: dict) -> Decision:
    from tools.registry import RiskLevel

    if tool.risk == RiskLevel.READ:
        return Decision.AUTO

    if tool.risk == RiskLevel.WRITE:
        log.info(f"[Safety] Auto-executing WRITE tool: {tool.name}")
        return Decision.AUTO

    if tool.risk == RiskLevel.SYSTEM:
        log.warning(f"[Safety] SYSTEM tool requires confirmation: {tool.name}")
        return Decision.ASK

    if tool.risk == RiskLevel.IRREVERSIBLE:
        log.warning(f"[Safety] IRREVERSIBLE tool requires confirmation: {tool.name}")
        return Decision.ASK

    # Unknown risk level — deny by default
    log.error(f"[Safety] Unknown risk level for {tool.name} — denying.")
    return Decision.DENY