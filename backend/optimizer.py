import psutil
import os
import time
import platform

# Auto-adjust UID threshold based on OS
if platform.system() == "Darwin":  # macOS
    USER_UID_THRESHOLD = 500
else:  # Linux or other
    USER_UID_THRESHOLD = 1000

IDLE_CPU_THRESHOLD = 1.0  # % CPU to be considered idle
IDLE_TIME_THRESHOLD = 60  # seconds of runtime
MEMORY_USAGE_THRESHOLD_MB = 20  # skip killing if using more than 20MB
DRY_RUN = True  # Toggle to True to log instead of killing

# Expanded whitelist substrings (case-insensitive match)
WHITELIST = [
    'python', 'code', 'terminal', 'qt', 'obsidian', 'spotify', 'discord', 'safari',
    'webkit', 'mdworker', 'helper', 'plugin-container', 'finder', 'dock', 'systemuiserver',
    'notificationcenter', 'windowmanager', 'loginwindow', 'cloud', 'raycast', 'noteful',
    'bitwarden', 'core', 'service', 'extension', 'widget', 'agent', 'render', 'appstore'
]

def is_idle(proc, now):
    try:
        cpu = proc.cpu_percent(interval=None)
        create_time = proc.create_time()
        uid = proc.uids().real if hasattr(proc, "uids") else os.getuid()
        name = proc.name().lower()
        mem = proc.memory_info().rss / (1024 * 1024)  # MB

        if uid < USER_UID_THRESHOLD:
            print(f"SKIP {name} (PID {proc.pid}): system UID {uid}")
            return False
        if any(w in name for w in WHITELIST):
            print(f"SKIP {name} (PID {proc.pid}): whitelisted")
            return False
        if cpu > IDLE_CPU_THRESHOLD:
            print(f"SKIP {name} (PID {proc.pid}): CPU {cpu:.2f}%")
            return False
        if mem > MEMORY_USAGE_THRESHOLD_MB:
            print(f"SKIP {name} (PID {proc.pid}): using {mem:.1f}MB RAM")
            return False
        if now - create_time < IDLE_TIME_THRESHOLD:
            print(f"SKIP {name} (PID {proc.pid}): too recent ({now - create_time:.1f}s)")
            return False
        return True
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False

def clean_memory():
    """
    Scans and logs (or terminates) idle user processes to free up RAM.
    """
    now = time.time()
    affected = []
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            print(f"Checking: {proc.pid} {proc.name()}")
        except Exception:
            pass
        if is_idle(proc, now):
            try:
                name = proc.name()
                pid = proc.pid
                if DRY_RUN:
                    affected.append((pid, name))
                else:
                    proc.terminate()
                    affected.append((pid, name))
            except Exception:
                continue

    if DRY_RUN:
        print("[Dry-Run] These processes would be terminated:")
    else:
        print("[Optimizer] Terminated processes:")

    for pid, name in affected:
        print(f" - {name} (PID {pid})")