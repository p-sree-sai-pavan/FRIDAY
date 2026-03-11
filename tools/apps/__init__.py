"""
tools/apps/__init__.py
======================
Exposes the 10 app-control tools to the FRIDAY registry.
"""

import logging

# We import registry this way because it's in the parent directory
from tools.registry import registry, Tool, RiskLevel

# Import handlers from our internal module
from .handlers import (
    open_app_handler,
    close_app_handler,
    list_running_apps_handler,
    list_installed_apps_handler,
    switch_to_app_handler,
    minimize_app_handler,
    maximize_app_handler,
    snap_window_handler,
    is_app_running_handler,
    restart_app_handler,
)

log = logging.getLogger("apps")

registry.register(Tool(
    name="open_app",
    description=(
        "Open one or more applications on Pavan's computer. "
        "Pass a single name like 'spotify' or multiple comma-separated names like 'chrome, spotify, notepad'. "
        "Handles common names, slang (vsc, yt, wp, gpt, tg), "
        "typos, and auto-discovers installed apps from the Start Menu."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "App name(s) to open. Single: 'spotify'. Multiple: 'chrome, spotify, notepad, discord'"
            }
        },
        "required": ["name"]
    },
    risk=RiskLevel.WRITE,
    handler=open_app_handler
))

registry.register(Tool(
    name="close_app",
    description=(
        "Close/kill a running application by name. "
        "Handles fuzzy matching (e.g. 'chrome' matches 'chrome.exe')."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name of the app to close (e.g. 'chrome', 'notepad', 'spotify')"
            }
        },
        "required": ["name"]
    },
    risk=RiskLevel.SYSTEM,
    handler=close_app_handler
))

registry.register(Tool(
    name="list_running_apps",
    description=(
        "List all currently running GUI applications on the computer, "
        "with their names, PIDs, and memory usage."
    ),
    parameters={
        "type": "object",
        "properties": {},
        "required": []
    },
    risk=RiskLevel.READ,
    handler=list_running_apps_handler
))

registry.register(Tool(
    name="list_installed_apps",
    description=(
        "List all installed applications on the computer, "
        "discovered from Start Menu shortcuts. "
        "Use this to know what apps are available before opening them."
    ),
    parameters={
        "type": "object",
        "properties": {},
        "required": []
    },
    risk=RiskLevel.READ,
    handler=list_installed_apps_handler
))

registry.register(Tool(
    name="switch_to_app",
    description=(
        "Bring a running application window to the foreground. "
        "Matches by window title or process name."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name of the app to switch to"
            }
        },
        "required": ["name"]
    },
    risk=RiskLevel.WRITE,
    handler=switch_to_app_handler
))

registry.register(Tool(
    name="minimize_app",
    description="Minimize an application window.",
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name of the app to minimize"
            }
        },
        "required": ["name"]
    },
    risk=RiskLevel.WRITE,
    handler=minimize_app_handler
))

registry.register(Tool(
    name="maximize_app",
    description="Maximize an application window.",
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name of the app to maximize"
            }
        },
        "required": ["name"]
    },
    risk=RiskLevel.WRITE,
    handler=maximize_app_handler
))

registry.register(Tool(
    name="snap_window",
    description=(
        "Snap an application window to a position on screen. "
        "Positions: left, right, top, bottom, top-left, top-right, bottom-left, bottom-right, center, fullscreen."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name of the app to snap"
            },
            "position": {
                "type": "string",
                "description": "Position to snap to: left, right, top, bottom, top-left, top-right, bottom-left, bottom-right, center, fullscreen",
                "enum": ["left", "right", "top", "bottom", "top-left", "top-right", "bottom-left", "bottom-right", "center", "fullscreen"]
            }
        },
        "required": ["name", "position"]
    },
    risk=RiskLevel.WRITE,
    handler=snap_window_handler
))

registry.register(Tool(
    name="is_app_running",
    description="Check if a specific application is currently running.",
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name of the app to check"
            }
        },
        "required": ["name"]
    },
    risk=RiskLevel.READ,
    handler=is_app_running_handler
))

registry.register(Tool(
    name="restart_app",
    description="Close and reopen an application. Useful when an app is frozen or needs a fresh start.",
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name of the app to restart"
            }
        },
        "required": ["name"]
    },
    risk=RiskLevel.SYSTEM,
    handler=restart_app_handler
))

log.info("[Apps] 10 tools registered: open_app, close_app, list_running_apps, list_installed_apps, switch_to_app, minimize_app, maximize_app, snap_window, is_app_running, restart_app")
