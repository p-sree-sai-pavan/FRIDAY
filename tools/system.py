"""
tools/system.py
================
FRIDAY System Control — General-Purpose Commands & System Info

Tools:
  run_command     — Execute any shell command (SYSTEM risk, asks confirmation)
  get_system_info — Battery, CPU, RAM, disk, network info (READ, no confirmation)

This is the real "body" — the LLM can generate and execute ANY command,
not just hardcoded ones. The safety layer ensures destructive commands
always ask for confirmation before running.
"""

import logging
import os
import platform
import subprocess
import sys

import psutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.registry import registry, Tool, RiskLevel

log = logging.getLogger("system")


# ========================
# run_command
# The LLM generates and executes any shell command.
# Risk: SYSTEM — always asks user confirmation.
# Timeout: 30 seconds to prevent hanging.
# ========================

async def _run_command(command: str) -> str:
    """Execute a shell command and return the output."""
    log.info(f"[System] Executing: {command}")

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            encoding="utf-8",
            errors="replace"
        )

        output = ""
        if result.stdout and result.stdout.strip():
            output += result.stdout.strip()
        if result.stderr and result.stderr.strip():
            if output:
                output += "\n"
            output += f"[stderr] {result.stderr.strip()}"

        if not output:
            if result.returncode == 0:
                output = "Command executed successfully (no output)."
            else:
                output = f"Command failed with exit code {result.returncode}."

        # Cap output to prevent overwhelming the LLM context
        if len(output) > 3000:
            output = output[:3000] + "\n... (output truncated at 3000 chars)"

        log.info(f"[System] Command returned (exit={result.returncode}, {len(output)} chars)")
        return output

    except subprocess.TimeoutExpired:
        log.warning(f"[System] Command timed out after 30s: {command}")
        return "Command timed out after 30 seconds."
    except Exception as e:
        log.error(f"[System] Command error: {e}")
        return f"Error executing command: {e}"


# ========================
# get_system_info
# Gathers system information without running any commands.
# Risk: READ — no confirmation needed.
# ========================

async def _get_system_info() -> str:
    """Get comprehensive system information."""
    info = []

    # OS
    info.append(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
    info.append(f"Machine: {platform.machine()}")
    info.append(f"Hostname: {platform.node()}")
    info.append(f"User: {os.getlogin()}")

    # CPU
    cpu_count = psutil.cpu_count(logical=True)
    cpu_phys = psutil.cpu_count(logical=False)
    cpu_percent = psutil.cpu_percent(interval=0.5)
    info.append(f"CPU: {cpu_phys} cores ({cpu_count} logical) | Usage: {cpu_percent}%")

    try:
        freq = psutil.cpu_freq()
        if freq:
            info.append(f"CPU Freq: {freq.current:.0f} MHz (max: {freq.max:.0f} MHz)")
    except Exception:
        pass

    # RAM
    mem = psutil.virtual_memory()
    info.append(
        f"RAM: {mem.used / (1024**3):.1f} GB / {mem.total / (1024**3):.1f} GB "
        f"({mem.percent}% used)"
    )

    # Disk
    for part in psutil.disk_partitions():
        try:
            usage = psutil.disk_usage(part.mountpoint)
            info.append(
                f"Disk {part.device}: {usage.used / (1024**3):.1f} GB / "
                f"{usage.total / (1024**3):.1f} GB ({usage.percent}% used)"
            )
        except PermissionError:
            continue

    # Battery
    try:
        battery = psutil.sensors_battery()
        if battery:
            plug = "Plugged in" if battery.power_plugged else "On battery"
            secs = battery.secsleft
            if secs == psutil.POWER_TIME_UNLIMITED:
                time_left = "charging"
            elif secs == psutil.POWER_TIME_UNKNOWN:
                time_left = "unknown"
            else:
                hrs = secs // 3600
                mins = (secs % 3600) // 60
                time_left = f"{hrs}h {mins}m remaining"
            info.append(f"Battery: {battery.percent}% | {plug} | {time_left}")
    except Exception:
        info.append("Battery: Not available (desktop)")

    # Network
    try:
        addrs = psutil.net_if_addrs()
        for iface, addr_list in addrs.items():
            for addr in addr_list:
                if addr.family.name == "AF_INET" and not addr.address.startswith("127."):
                    info.append(f"Network: {iface} -> {addr.address}")
    except Exception:
        pass

    # Uptime
    try:
        import time
        boot = psutil.boot_time()
        uptime_secs = time.time() - boot
        hrs = int(uptime_secs // 3600)
        mins = int((uptime_secs % 3600) // 60)
        info.append(f"Uptime: {hrs}h {mins}m")
    except Exception:
        pass

    return "\n".join(info)


# ========================
# REGISTER TOOLS
# ========================

registry.register(Tool(
    name="run_command",
    description=(
        "Execute any shell command on Pavan's Windows computer and return the output. "
        "Use this for ANY system task that doesn't have a dedicated tool: "
        "checking disk space, network info, process management, file operations, "
        "installing packages, running scripts, system configuration, and anything else. "
        "The command runs in cmd.exe with a 30-second timeout. "
        "Output is capped at 3000 characters."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": (
                    "The shell command to execute. Examples: "
                    "'ipconfig', 'wmic logicaldisk get size,freespace', "
                    "'powershell Get-Process | Sort-Object CPU -Desc | Select -First 10', "
                    "'ping google.com -n 4', 'dir C:\\Users'"
                )
            }
        },
        "required": ["command"]
    },
    risk=RiskLevel.SYSTEM,
    handler=_run_command
))

registry.register(Tool(
    name="get_system_info",
    description=(
        "Get system information: OS, CPU, RAM, disk usage, battery status, "
        "network interfaces, and uptime. No arguments needed."
    ),
    parameters={
        "type": "object",
        "properties": {},
        "required": []
    },
    risk=RiskLevel.READ,
    handler=_get_system_info
))

log.info("[System] 2 tools registered: run_command, get_system_info")
