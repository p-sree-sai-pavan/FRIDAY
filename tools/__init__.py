"""
tools/__init__.py
==================
FRIDAY Tool Auto-Discovery

When this package is imported, it automatically discovers and imports
all tool modules in the tools/ directory. Each tool module registers
itself with the global registry at import time.

Skips: __init__.py, registry.py, and empty files.
"""

import importlib
import logging
import os
import pkgutil

log = logging.getLogger("tools")

# Import registry first — tool modules need it
from tools.registry import registry  # noqa: F401

# Auto-discover and import all tool modules in this package
_package_dir = os.path.dirname(__file__)

for _finder, _name, _ispkg in pkgutil.iter_modules([_package_dir]):
    if _name in ("registry",):
        continue  # skip the registry itself

    _module_path = os.path.join(_package_dir, f"{_name}.py")

    # Skip empty files (0 bytes or just whitespace)
    try:
        if os.path.isfile(_module_path):
            size = os.path.getsize(_module_path)
            if size <= 1:  # empty or just a newline
                log.debug(f"[Tools] Skipping empty module: {_name}")
                continue
    except OSError:
        continue

    try:
        importlib.import_module(f"tools.{_name}")
        log.debug(f"[Tools] Loaded module: {_name}")
    except Exception as e:
        log.warning(f"[Tools] Failed to load module '{_name}': {e}")

log.info(f"[Tools] {len(registry.list_all())} tool(s) registered")