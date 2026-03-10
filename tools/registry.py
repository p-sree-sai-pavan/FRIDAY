"""
tools/registry.py
==================
FRIDAY Tool Registry

Central catalog of every action FRIDAY can take.
Tools register themselves at import. Orchestrator feeds
them to Groq's function calling API. Safety layer gates
every execution by risk level.

Risk Levels (maps to config thresholds):
    READ         → auto-execute always         (search, read file, screenshot)
    WRITE        → auto-execute + inform Pavan (open app, write file)
    SYSTEM       → ask first                   (kill process, run terminal cmd)
    IRREVERSIBLE → always ask, show preview    (delete file, send message)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

log = logging.getLogger("registry")


# ========================
# RISK LEVELS
# ========================

class RiskLevel(Enum):
    READ         = "read"
    WRITE        = "write"
    SYSTEM       = "system"
    IRREVERSIBLE = "irreversible"


# ========================
# TOOL DEFINITION
# ========================

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict          # JSON Schema — what Groq expects
    risk: RiskLevel
    handler: Callable         # async function that runs the tool
    requires_confirmation: Optional[bool] = None  # override risk default if needed

    def __post_init__(self):
        # Default confirmation requirement derived from risk
        if self.requires_confirmation is None:
            self.requires_confirmation = self.risk in (
                RiskLevel.SYSTEM,
                RiskLevel.IRREVERSIBLE
            )

    def to_groq(self) -> dict:
        """Export in Groq's exact function calling schema format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


# ========================
# REGISTRY
# ========================

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        """Register a tool. Called by each tool module at import."""
        if tool.name in self._tools:
            log.warning(f"[Registry] Tool '{tool.name}' already registered — overwriting.")
        self._tools[tool.name] = tool
        log.debug(f"[Registry] Registered tool: {tool.name} | risk={tool.risk.value}")

    def get(self, name: str) -> Optional[Tool]:
        """Look up a tool by name. Returns None if not found."""
        return self._tools.get(name)

    def list_all(self) -> list[Tool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def to_groq_tools(self) -> list[dict]:
        """
        Export all tools in Groq's function calling format.
        Pass this directly to groq_client.chat.completions.create(tools=...)
        """
        return [tool.to_groq() for tool in self._tools.values()]

    def summary(self) -> str:
        """Human-readable list of registered tools — for debug/logging."""
        if not self._tools:
            return "No tools registered."
        lines = [f"  [{t.risk.value.upper()}] {t.name} — {t.description}"
                 for t in self._tools.values()]
        return "\n".join(lines)


# ========================
# GLOBAL INSTANCE
# ========================
# Single registry used across the entire app.
# Import this instance everywhere:
#   from tools.registry import registry

registry = ToolRegistry()


# ========================
# HELPER — execute a tool call from Groq response
# ========================

async def execute_tool_call(tool_name: str, arguments: dict) -> Any:
    """
    Look up and execute a tool by name with given arguments.
    Returns the tool's result (any type).
    Raises KeyError if tool not found.
    Raises Exception if handler fails — caller handles this.
    """
    tool = registry.get(tool_name)
    if tool is None:
        raise KeyError(f"Tool '{tool_name}' not found in registry.")

    log.info(f"[Registry] Executing: {tool_name}({arguments})")
    result = await tool.handler(**arguments)
    log.info(f"[Registry] Result from {tool_name}: {str(result)[:120]}")
    return result