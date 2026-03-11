"""
tools/apps/handlers.py
======================
The actual tool handler functions executed by the AI agent.

All handlers are async. resolve_app is now a native async function,
so open_app_handler calls it with a plain `await` — no asyncio.to_thread
wrapper needed.
"""

import asyncio
import ctypes
import difflib
import logging
import os
import subprocess
import webbrowser

import psutil

from .constants import SYSTEM_PROCESSES
from .discovery import scan_start_menu
from .resolution import resolve_app
from .win32_utils import _find_window, _get_screen_size

log = logging.getLogger("apps.handlers")


async def open_app_handler(name: str) -> str:
    """Open one or more applications by name with smart async resolution."""
    names = [n.strip() for n in name.split(",") if n.strip()]
    if not names:
        return "No app name provided."

    results = []

    for app_name in names:
        # resolve_app is now fully async — direct await, no thread wrapper needed
        command, source = await resolve_app(app_name)
        log.info(f"[Apps] Opening '{app_name}' → '{command}' (source: {source})")

        try:
            expanded_cmd = os.path.expandvars(command)

            # Case 1: Explicit web or Windows URI (start http..., start ms-settings:, etc.)
            if expanded_cmd.lower().startswith("start "):
                target = expanded_cmd[6:].strip()
                if target.startswith("http"):
                    webbrowser.open(target)
                else:
                    os.startfile(target)
                results.append(f"Opened {app_name}")
                continue

            # Case 2: Simple app name / path — ShellExecute via os.startfile.
            # Windows resolves bare names like 'chrome', 'spotify' through App Paths registry.
            # Only works without arguments.
            try:
                clean_target = expanded_cmd.strip('"')
                os.startfile(clean_target)
                results.append(f"Opened {app_name}")
                continue
            except (OSError, FileNotFoundError):
                pass  # Fall through to shell launch if it has arguments

            # Case 3: Complex commands with arguments — use cmd.exe's start.
            # `start ""` prevents the first quoted string from being treated as the window title.
            subprocess.Popen(f'start "" {expanded_cmd}', shell=True)
            results.append(f"Opened {app_name}")

        except Exception as e:
            log.error(f"[Apps] Failed to open '{app_name}': {e}")
            results.append(f"Failed to open {app_name}: {e}")

    return ". ".join(results) + "."


async def close_app_handler(name: str) -> str:
    """Close/kill an app by name, with system process protection."""
    key = name.lower().strip()
    key_no_exe = key.replace(".exe", "")

    # Guard: never kill known system processes
    if key in SYSTEM_PROCESSES or (key_no_exe + ".exe") in SYSTEM_PROCESSES:
        return f"Refusing to close protected system process: '{name}'."

    closed    = []
    not_found = True

    for proc in psutil.process_iter(["pid", "name"]):
        try:
            pname       = proc.info["name"] or ""
            pname_lower = pname.lower()
            pname_bare  = pname_lower.replace(".exe", "")

            if key in (pname_lower, pname_bare) or (
                difflib.SequenceMatcher(None, key, pname_bare).ratio() > 0.7
            ):
                not_found = False
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except psutil.TimeoutExpired:
                    proc.kill()
                closed.append(f"{pname} (PID {proc.info['pid']})")

        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    if not_found:
        return f"No running process found matching '{name}'."
    if closed:
        return f"Closed: {', '.join(closed)}"
    return f"Found '{name}' but could not close it (access denied)."


async def list_running_apps_handler() -> str:
    """List all running user applications, excluding system processes."""
    apps = []

    for proc in psutil.process_iter(["pid", "name", "memory_info", "status"]):
        try:
            pname = proc.info["name"] or ""
            if not pname:
                continue
            if pname.lower() in SYSTEM_PROCESSES:
                continue
            # On Windows, apps waiting for input report 'sleeping' not 'running'.
            # Only skip genuinely dead/zombie processes.
            if proc.info["status"] in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
                continue

            mem    = proc.info.get("memory_info")
            mem_mb = round(mem.rss / (1024 * 1024), 1) if mem else 0

            apps.append({
                "name":      pname.replace(".exe", ""),
                "pid":       proc.info["pid"],
                "memory_mb": mem_mb
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    if not apps:
        return "No visible applications are currently running."

    # Deduplicate by name, keeping the highest-memory instance
    seen: dict[str, dict] = {}
    for app in apps:
        k = app["name"].lower()
        if k not in seen or app["memory_mb"] > seen[k]["memory_mb"]:
            seen[k] = app

    sorted_apps = sorted(seen.values(), key=lambda x: x["memory_mb"], reverse=True)

    lines = [f"Running apps ({len(sorted_apps)}):"]
    for app in sorted_apps[:25]:
        lines.append(f"  - {app['name']} — {app['memory_mb']} MB (PID {app['pid']})")

    return "\n".join(lines)


async def list_installed_apps_handler() -> str:
    """List all installed applications discovered from Start Menu shortcuts."""
    start_menu = await asyncio.to_thread(scan_start_menu)

    if not start_menu:
        return "Could not find any installed apps from Start Menu."

    skip_words = {"uninstall", "readme", "help", "documentation", "license", "release notes"}
    apps = [
        name.title()
        for name in sorted(start_menu.keys())
        if not any(s in name.lower() for s in skip_words)
    ]

    if not apps:
        return "No installed apps found."

    lines = [f"Installed apps ({len(apps)}):"]
    for app in apps:
        lines.append(f"  - {app}")

    return "\n".join(lines)


async def switch_to_app_handler(name: str) -> str:
    """Bring an app window to the foreground."""
    try:
        user32     = ctypes.windll.user32
        hwnd, title = _find_window(name)

        if hwnd is None:
            return f"No window found for '{name}'. The app might not be running."

        user32.ShowWindow(hwnd, 9)       # SW_RESTORE
        user32.SetForegroundWindow(hwnd)
        return f"Switched to: {title}"

    except Exception as e:
        log.error(f"[Apps] Switch to app failed: {e}")
        return f"Failed to switch to {name}: {e}"


async def minimize_app_handler(name: str) -> str:
    """Minimize an app window."""
    try:
        user32      = ctypes.windll.user32
        hwnd, title = _find_window(name)

        if hwnd is None:
            return f"No window found for '{name}'. The app might not be running."

        user32.ShowWindow(hwnd, 6)  # SW_MINIMIZE
        return f"Minimized: {title}"

    except Exception as e:
        log.error(f"[Apps] Minimize failed: {e}")
        return f"Failed to minimize {name}: {e}"


async def maximize_app_handler(name: str) -> str:
    """Maximize an app window."""
    try:
        user32      = ctypes.windll.user32
        hwnd, title = _find_window(name)

        if hwnd is None:
            return f"No window found for '{name}'. The app might not be running."

        user32.ShowWindow(hwnd, 3)       # SW_MAXIMIZE
        user32.SetForegroundWindow(hwnd)
        return f"Maximized: {title}"

    except Exception as e:
        log.error(f"[Apps] Maximize failed: {e}")
        return f"Failed to maximize {name}: {e}"


async def snap_window_handler(name: str, position: str = "left") -> str:
    """Snap a window to a position on screen."""
    try:
        user32      = ctypes.windll.user32
        hwnd, title = _find_window(name)

        if hwnd is None:
            return f"No window found for '{name}'. The app might not be running."

        screen_w, screen_h = _get_screen_size()
        user32.ShowWindow(hwnd, 9)   # SW_RESTORE first
        SWP_NOZORDER = 0x0004
        pos = position.lower().strip()

        positions = {
            ("left",   "l"):                     (0,              0,             screen_w // 2, screen_h),
            ("right",  "r"):                     (screen_w // 2,  0,             screen_w // 2, screen_h),
            ("top",    "top-half"):               (0,              0,             screen_w,      screen_h // 2),
            ("bottom", "bottom-half"):            (0,              screen_h // 2, screen_w,      screen_h // 2),
            ("top-left",     "tl"):              (0,              0,             screen_w // 2, screen_h // 2),
            ("top-right",    "tr"):              (screen_w // 2,  0,             screen_w // 2, screen_h // 2),
            ("bottom-left",  "bl"):              (0,              screen_h // 2, screen_w // 2, screen_h // 2),
            ("bottom-right", "br"):              (screen_w // 2,  screen_h // 2, screen_w // 2, screen_h // 2),
            ("center", "c"):                     ((screen_w - screen_w // 2) // 2,
                                                  (screen_h - screen_h // 2) // 2,
                                                  screen_w // 2, screen_h // 2),
        }

        if pos in ("fullscreen", "full", "fill"):
            user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE
        else:
            rect = None
            for keys, coords in positions.items():
                if pos in keys:
                    rect = coords
                    break

            if rect is None:
                valid = "left, right, top, bottom, top-left, top-right, bottom-left, bottom-right, center, fullscreen"
                return f"Unknown position '{position}'. Valid options: {valid}"

            x, y, w, h = rect
            user32.SetWindowPos(hwnd, 0, x, y, w, h, SWP_NOZORDER)

        user32.SetForegroundWindow(hwnd)
        return f"Snapped '{title}' to {position}."

    except Exception as e:
        log.error(f"[Apps] Snap window failed: {e}")
        return f"Failed to snap {name}: {e}"


async def is_app_running_handler(name: str) -> str:
    """Check if a specific app is currently running."""
    key = name.lower().strip()

    for proc in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            pname      = proc.info["name"] or ""
            pname_bare = pname.lower().replace(".exe", "")

            if key in (pname.lower(), pname_bare) or (
                difflib.SequenceMatcher(None, key, pname_bare).ratio() > 0.7
            ):
                mem    = proc.info.get("memory_info")
                mem_mb = round(mem.rss / (1024 * 1024), 1) if mem else 0
                return f"Yes, {pname} is running (PID {proc.info['pid']}, {mem_mb} MB)."

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return f"No, {name} is not currently running."


async def restart_app_handler(name: str) -> str:
    """Close an app and reopen it. Useful when an app is frozen."""
    close_result = await close_app_handler(name)
    log.info(f"[Apps] Restart — close phase: {close_result}")

    await asyncio.sleep(1.5)   # give the OS time to release file handles

    open_result = await open_app_handler(name)
    log.info(f"[Apps] Restart — open phase: {open_result}")

    return f"Restarted {name}. {close_result} → {open_result}"
