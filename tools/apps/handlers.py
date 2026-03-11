"""
tools/apps/handlers.py
======================
The actual tool handler functions executed by the AI agent.
"""

import asyncio
import ctypes
import difflib
import logging
import subprocess

import psutil

from .constants import SYSTEM_PROCESSES
from .discovery import scan_start_menu
from .resolution import resolve_app
from .win32_utils import _find_window, _get_screen_size

log = logging.getLogger("apps.handlers")


async def open_app_handler(name: str) -> str:
    """Open one or more applications by name with smart resolution."""
    names = [n.strip() for n in name.split(",") if n.strip()]
    if not names:
        return "No app name provided."

    results = []
    import os
    import webbrowser

    for app_name in names:
        command, source = resolve_app(app_name)
        log.info(f"[Apps] Opening '{app_name}' -> command='{command}' (source: {source})")

        try:
            # Expand environment variables (e.g., %ProgramFiles%)
            expanded_cmd = os.path.expandvars(command)

            # Case 1: Explicit Web URIs or native Windows URIs
            if expanded_cmd.lower().startswith("start "):
                target = expanded_cmd[6:].strip() # remove 'start '
                if target.startswith("http"):
                    webbrowser.open(target)
                else:
                    os.startfile(target)
                results.append(f"Opened {app_name}")
                continue

            # Case 2: Native App Path Resolution
            # os.startfile uses native Windows ShellExecute, which natively
            # understands 'chrome', 'spotify', etc., from the App Paths registry.
            # It only accepts strings without arguments, so we try this first.
            try:
                clean_target = expanded_cmd.strip('"')
                os.startfile(clean_target)
                results.append(f"Opened {app_name}")
                continue
            except (OSError, FileNotFoundError):
                pass # Proceed to Case 3 if it has arguments or requires a shell

            # Case 3: Complex commands with arguments
            # Use cmd.exe's start command, but safely:
            # `start ""` prevents the bug where first quoted string becomes the window title.
            safe_shell_cmd = f'start "" {expanded_cmd}'
            subprocess.Popen(safe_shell_cmd, shell=True)
            results.append(f"Opened {app_name}")

        except Exception as e:
            log.error(f"[Apps] Failed to open '{app_name}': {e}")
            results.append(f"Failed to open {app_name}: {e}")

    return ". ".join(results) + "."


async def close_app_handler(name: str) -> str:
    """Close/kill an app by name."""
    key = name.lower().strip()
    closed = []
    not_found = True

    for proc in psutil.process_iter(["pid", "name"]):
        try:
            pname = proc.info["name"] or ""
            pname_lower = pname.lower()
            pname_bare = pname_lower.replace(".exe", "")

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
    """List all running GUI applications (not system processes)."""
    apps = []

    for proc in psutil.process_iter(["pid", "name", "memory_info", "status"]):
        try:
            pname = proc.info["name"] or ""
            if not pname:
                continue
            if pname.lower() in SYSTEM_PROCESSES:
                continue
            if proc.info["status"] != psutil.STATUS_RUNNING:
                continue

            mem = proc.info.get("memory_info")
            mem_mb = round(mem.rss / (1024 * 1024), 1) if mem else 0

            apps.append({
                "name": pname.replace(".exe", ""),
                "pid": proc.info["pid"],
                "memory_mb": mem_mb
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    if not apps:
        return "No visible applications are currently running."

    seen: dict[str, dict] = {}
    for app in apps:
        key = app["name"].lower()
        if key not in seen or app["memory_mb"] > seen[key]["memory_mb"]:
            seen[key] = app

    sorted_apps = sorted(seen.values(), key=lambda x: x["memory_mb"], reverse=True)

    lines = [f"Running apps ({len(sorted_apps)}):"]
    for app in sorted_apps[:25]:
        lines.append(f"  - {app['name']} -- {app['memory_mb']} MB (PID {app['pid']})")

    return "\n".join(lines)


async def list_installed_apps_handler() -> str:
    """List all installed applications discovered from Start Menu shortcuts."""
    start_menu = scan_start_menu()

    if not start_menu:
        return "Could not find any installed apps from Start Menu."

    skip_words = {"uninstall", "readme", "help", "documentation", "license", "release notes"}
    apps = []
    for name in sorted(start_menu.keys()):
        if any(s in name.lower() for s in skip_words):
            continue
        apps.append(name.title())

    if not apps:
        return "No installed apps found."

    lines = [f"Installed apps ({len(apps)}):"]
    for app in apps:
        lines.append(f"  - {app}")

    return "\n".join(lines)


async def switch_to_app_handler(name: str) -> str:
    """Bring an app window to the foreground."""
    try:
        user32 = ctypes.windll.user32
        hwnd, title = _find_window(name)

        if hwnd is None:
            return f"No window found for '{name}'. The app might not be running."

        SW_RESTORE = 9
        user32.ShowWindow(hwnd, SW_RESTORE)
        user32.SetForegroundWindow(hwnd)

        return f"Switched to: {title}"

    except Exception as e:
        log.error(f"[Apps] Switch to app failed: {e}")
        return f"Failed to switch to {name}: {e}"


async def minimize_app_handler(name: str) -> str:
    """Minimize an app window."""
    try:
        user32 = ctypes.windll.user32
        hwnd, title = _find_window(name)

        if hwnd is None:
            return f"No window found for '{name}'. The app might not be running."

        SW_MINIMIZE = 6
        user32.ShowWindow(hwnd, SW_MINIMIZE)

        return f"Minimized: {title}"

    except Exception as e:
        log.error(f"[Apps] Minimize failed: {e}")
        return f"Failed to minimize {name}: {e}"


async def maximize_app_handler(name: str) -> str:
    """Maximize an app window."""
    try:
        user32 = ctypes.windll.user32
        hwnd, title = _find_window(name)

        if hwnd is None:
            return f"No window found for '{name}'. The app might not be running."

        SW_MAXIMIZE = 3
        user32.ShowWindow(hwnd, SW_MAXIMIZE)
        user32.SetForegroundWindow(hwnd)

        return f"Maximized: {title}"

    except Exception as e:
        log.error(f"[Apps] Maximize failed: {e}")
        return f"Failed to maximize {name}: {e}"


async def snap_window_handler(name: str, position: str = "left") -> str:
    """Snap a window to left or right half of the screen."""
    try:
        user32 = ctypes.windll.user32
        hwnd, title = _find_window(name)

        if hwnd is None:
            return f"No window found for '{name}'. The app might not be running."

        screen_w, screen_h = _get_screen_size()
        SW_RESTORE = 9
        user32.ShowWindow(hwnd, SW_RESTORE)

        pos = position.lower().strip()
        SWP_NOZORDER = 0x0004

        if pos in ("left", "l"):
            user32.SetWindowPos(hwnd, 0, 0, 0, screen_w // 2, screen_h, SWP_NOZORDER)
        elif pos in ("right", "r"):
            user32.SetWindowPos(hwnd, 0, screen_w // 2, 0, screen_w // 2, screen_h, SWP_NOZORDER)
        elif pos in ("top", "top-half"):
            user32.SetWindowPos(hwnd, 0, 0, 0, screen_w, screen_h // 2, SWP_NOZORDER)
        elif pos in ("bottom", "bottom-half"):
            user32.SetWindowPos(hwnd, 0, 0, screen_h // 2, screen_w, screen_h // 2, SWP_NOZORDER)
        elif pos in ("top-left", "tl"):
            user32.SetWindowPos(hwnd, 0, 0, 0, screen_w // 2, screen_h // 2, SWP_NOZORDER)
        elif pos in ("top-right", "tr"):
            user32.SetWindowPos(hwnd, 0, screen_w // 2, 0, screen_w // 2, screen_h // 2, SWP_NOZORDER)
        elif pos in ("bottom-left", "bl"):
            user32.SetWindowPos(hwnd, 0, 0, screen_h // 2, screen_w // 2, screen_h // 2, SWP_NOZORDER)
        elif pos in ("bottom-right", "br"):
            user32.SetWindowPos(hwnd, 0, screen_w // 2, screen_h // 2, screen_w // 2, screen_h // 2, SWP_NOZORDER)
        elif pos in ("center", "c"):
            w, h = screen_w // 2, screen_h // 2
            user32.SetWindowPos(hwnd, 0, (screen_w - w) // 2, (screen_h - h) // 2, w, h, SWP_NOZORDER)
        elif pos in ("fullscreen", "full", "fill"):
            SW_MAXIMIZE = 3
            user32.ShowWindow(hwnd, SW_MAXIMIZE)
        else:
            return f"Unknown position '{position}'. Use: left, right, top, bottom, top-left, top-right, bottom-left, bottom-right, center, fullscreen."

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
            pname = proc.info["name"] or ""
            pname_bare = pname.lower().replace(".exe", "")

            if key in (pname.lower(), pname_bare) or (
                difflib.SequenceMatcher(None, key, pname_bare).ratio() > 0.7
            ):
                mem = proc.info.get("memory_info")
                mem_mb = round(mem.rss / (1024 * 1024), 1) if mem else 0
                return f"Yes, {pname} is running (PID {proc.info['pid']}, {mem_mb} MB)."
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return f"No, {name} is not currently running."


async def restart_app_handler(name: str) -> str:
    """Close and reopen an app."""
    close_result = await close_app_handler(name)
    log.info(f"[Apps] Restart - close phase: {close_result}")

    await asyncio.sleep(1.5)

    open_result = await open_app_handler(name)
    log.info(f"[Apps] Restart - open phase: {open_result}")

    return f"Restarted {name}. {close_result} {open_result}"
