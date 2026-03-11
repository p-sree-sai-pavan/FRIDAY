"""
memory/__init__.py
==================
Exposes the public API for the FRIDAY Memory Manager.
"""

from .api import (
    read,
    write,
    write_semantic,
    get_user_profile,
    save_user_profile,
    save_feedback,
    compress_results,
)

__all__ = [
    "read",
    "write",
    "write_semantic",
    "get_user_profile",
    "save_user_profile",
    "save_feedback",
    "compress_results",
]
