"""
tools/apps/discovery.py
=======================
Handles auto-discovering installed applications from Windows Start Menu links.
"""

import logging
import os
import subprocess
import threading

log = logging.getLogger("apps.discovery")

_start_menu_cache: dict[str, str] = {}
_start_menu_lock = threading.Lock()
_start_menu_scanned = False


def scan_start_menu() -> dict[str, str]:
    """
    Walk Start Menu folders, find .lnk files, extract names.
    Returns {lowercase_name: full_lnk_path}.
    Cached after first run.
    """
    global _start_menu_scanned, _start_menu_cache

    with _start_menu_lock:
        if _start_menu_scanned:
            return _start_menu_cache

        folders = []

        # Current user Start Menu
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            folders.append(os.path.join(appdata, "Microsoft", "Windows", "Start Menu", "Programs"))

        # All users Start Menu
        programdata = os.environ.get("ProgramData", r"C:\ProgramData")
        folders.append(os.path.join(programdata, "Microsoft", "Windows", "Start Menu", "Programs"))

        found: dict[str, str] = {}

        for folder in folders:
            if not os.path.isdir(folder):
                continue
            for root, dirs, files in os.walk(folder):
                for fname in files:
                    if fname.lower().endswith(".lnk"):
                        name = fname[:-4]  # strip .lnk
                        full_path = os.path.join(root, fname)
                        key = name.lower().strip()
                        if key and key not in found:
                            found[key] = full_path

        _start_menu_cache = found
        _start_menu_scanned = True
        log.info(f"[Apps] Start Menu scan: found {len(found)} shortcuts")
        return found


def resolve_lnk(lnk_path: str) -> str | None:
    """
    Resolve a .lnk shortcut to its target executable path.
    Uses PowerShell — works without pywin32.
    """
    try:
        ps_cmd = (
            f'(New-Object -ComObject WScript.Shell)'
            f'.CreateShortcut("{lnk_path}").TargetPath'
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            capture_output=True, text=True, timeout=5
        )
        target = result.stdout.strip()
        if target and os.path.exists(target):
            return target
    except Exception as e:
        log.debug(f"[Apps] LNK resolve failed for {lnk_path}: {e}")
    return None
