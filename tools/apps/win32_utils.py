"""
tools/apps/win32_utils.py
=========================
Win32 API wrappers via ctypes for window management (find, snap, resize).
"""

import ctypes
import ctypes.wintypes
import difflib

import psutil


def _find_window(name: str) -> tuple[int | None, str]:
    """
    Find a window by title or process name.
    Returns (hwnd, window_title) or (None, "").
    """
    user32 = ctypes.windll.user32
    target_hwnd = None
    target_title = ""
    key = name.lower().strip()

    WNDENUMPROC = ctypes.WINFUNCTYPE(
        ctypes.wintypes.BOOL,
        ctypes.wintypes.HWND,
        ctypes.wintypes.LPARAM
    )

    def _enum_by_title(hwnd, lparam):
        nonlocal target_hwnd, target_title
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value
        if title and key in title.lower():
            target_hwnd = hwnd
            target_title = title
            return False
        return True

    user32.EnumWindows(WNDENUMPROC(_enum_by_title), 0)

    if target_hwnd is not None:
        return target_hwnd, target_title

    # Fallback: match process names, then find their windows
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            pname = (proc.info["name"] or "").lower().replace(".exe", "")
            if key in pname or difflib.SequenceMatcher(None, key, pname).ratio() > 0.7:
                pid = proc.info["pid"]

                def _enum_by_pid(hwnd, lparam):
                    nonlocal target_hwnd, target_title
                    if not user32.IsWindowVisible(hwnd):
                        return True
                    proc_id = ctypes.wintypes.DWORD()
                    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(proc_id))
                    if proc_id.value == pid:
                        length = user32.GetWindowTextLengthW(hwnd)
                        if length > 0:
                            buf = ctypes.create_unicode_buffer(length + 1)
                            user32.GetWindowTextW(hwnd, buf, length + 1)
                            target_hwnd = hwnd
                            target_title = buf.value
                            return False
                    return True

                user32.EnumWindows(WNDENUMPROC(_enum_by_pid), 0)
                if target_hwnd:
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return target_hwnd, target_title


def _get_screen_size() -> tuple[int, int]:
    """Get primary screen resolution."""
    user32 = ctypes.windll.user32
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
