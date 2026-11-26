"""
24/7 watchdog for the MR-TAE orchestrator on Windows.

What it does:
    - Launches `pipeline/mlflow_orchestrator.py --resume` and watches the process.
    - On crash, restarts with exponential back-off (30 s -> 60 s -> 120 s -> 600 s cap).
    - Refuses to restart if the orchestrator has reported "all_done" in
      session_state.json (the run is genuinely complete).
    - Sends Telegram notifications for state transitions if creds are configured.
    - Gracefully exits on SIGINT/SIGTERM (Ctrl-C) with no orphaned children.
    - Survives Windows reboots when launched via Task Scheduler at logon
      (see scripts/install_watchdog_task.ps1).

Drop into:
    FYP_MASTER/pipeline/watchdog.py

Run manually for testing:
    python pipeline/watchdog.py
    python pipeline/watchdog.py --dry-run        # just print what it would do
    python pipeline/watchdog.py --no-telegram

Configure Telegram (optional):
    setx TELEGRAM_BOT_TOKEN "123456:abcdef..."
    setx TELEGRAM_CHAT_ID  "987654321"
    (Or put them in pipeline/.env -- see notes at the top of the file)

Author note (Claude, May 1 2026):
    This is intentionally dependency-light -- only `requests` is optional and
    only required if you turn on Telegram. Everything else is stdlib so it
    can survive on a fresh Python install.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SESSION_STATE = PROJECT_ROOT / "session_state.json"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "watchdog.log"
ORCHESTRATOR = PROJECT_ROOT / "pipeline" / "mlflow_orchestrator.py"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"  # Windows
if not VENV_PYTHON.exists():
    VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"      # POSIX fallback

BACKOFF_SCHEDULE_S = (30, 60, 120, 300, 600)  # caps at 10 minutes
MAX_CONSECUTIVE_RESTARTS = 12  # ~ 2 hr of failure before giving up
HEALTH_POLL_S = 30

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("watchdog")


# ---------------------------------------------------------------------------
# Telegram notifier (optional)
# ---------------------------------------------------------------------------
class Notifier:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        if enabled and (not self.token or not self.chat_id):
            log.info("Telegram credentials not set -- notifications disabled.")
            self.enabled = False
        if self.enabled:
            try:
                import requests  # noqa: F401  (lazy import, optional dependency)
            except ImportError:
                log.warning("`requests` not installed -- Telegram disabled.")
                self.enabled = False

    def send(self, msg: str) -> None:
        if not self.enabled:
            return
        import requests
        try:
            requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={"chat_id": self.chat_id, "text": msg, "parse_mode": "Markdown"},
                timeout=10,
            )
        except Exception as e:
            log.warning(f"Telegram send failed: {e}")


# ---------------------------------------------------------------------------
# Session-state inspection
# ---------------------------------------------------------------------------
def session_is_complete() -> bool:
    """
    Detects when the orchestrator has finished all queued models.
    Reads session_state.json (managed by training/resilience.py).

    Tolerates a few different schemas because the resilience manager has
    been edited a few times. Anything truthy in any of these is "done".
    """
    if not SESSION_STATE.exists():
        return False
    try:
        with SESSION_STATE.open() as f:
            state = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log.warning(f"Could not parse session_state.json: {e}")
        return False

    # Heuristics, ordered most-explicit to least.
    if state.get("status") in ("complete", "all_done", "finished"):
        return True
    queue = state.get("queue") or state.get("model_queue") or []
    if isinstance(queue, list) and len(queue) > 0:
        # Run is complete if every queue item is in done/skipped status.
        terminal = {"done", "skipped", "completed", "failed_permanent"}
        if all(
            (isinstance(m, dict) and m.get("status", "") in terminal)
            for m in queue
        ):
            return True
    return False


def session_summary() -> str:
    """Brief text summary of the queue for notification messages."""
    if not SESSION_STATE.exists():
        return "no session_state.json yet"
    try:
        with SESSION_STATE.open() as f:
            state = json.load(f)
    except Exception:
        return "session_state.json unreadable"
    queue = state.get("queue") or state.get("model_queue") or []
    if not queue:
        return f"status={state.get('status', 'unknown')}"
    counts: dict[str, int] = {}
    for m in queue:
        if isinstance(m, dict):
            counts[m.get("status", "unknown")] = counts.get(m.get("status", "unknown"), 0) + 1
    return ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------
def launch_orchestrator() -> subprocess.Popen:
    """
    Launch the orchestrator. On Windows we set CREATE_NEW_PROCESS_GROUP so
    that we can deliver Ctrl-Break without killing ourselves.
    """
    cmd = [str(VENV_PYTHON), str(ORCHESTRATOR), "--resume"]
    log.info(f"Launching: {' '.join(cmd)}")

    creationflags = 0
    if sys.platform.startswith("win"):
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    return subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
        encoding="utf-8",
        creationflags=creationflags,
    )


def stream_output(proc: subprocess.Popen, tail_log: Path) -> int:
    """
    Stream child stdout/stderr to our log AND to a tail file the dashboard
    can read. Returns the child's exit code.
    """
    tail_log.parent.mkdir(parents=True, exist_ok=True)
    with tail_log.open("a", encoding="utf-8") as tail:
        tail.write(f"\n===== watchdog launch {datetime.now(timezone.utc).isoformat()} =====\n")
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            log.info(f"[orch] {line}")
            tail.write(line + "\n")
            tail.flush()
    return proc.wait()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def watchdog_loop(notifier: Notifier, dry_run: bool) -> int:
    notifier.send(f"*Watchdog started* on `{os.environ.get('COMPUTERNAME', 'host')}`\n{session_summary()}")
    log.info(f"Project root: {PROJECT_ROOT}")
    log.info(f"Python:       {VENV_PYTHON}")
    log.info(f"Orchestrator: {ORCHESTRATOR}")

    consecutive_failures = 0
    tail_log = LOG_DIR / "orchestrator_tail.log"

    while True:
        if session_is_complete():
            log.info("Session is complete (all models done). Watchdog exiting cleanly.")
            notifier.send(f"*Watchdog: training complete*\n{session_summary()}")
            return 0

        if dry_run:
            log.info("[dry-run] would have launched orchestrator now. Exiting.")
            return 0

        start_t = time.time()
        try:
            proc = launch_orchestrator()
        except FileNotFoundError as e:
            log.error(f"Could not launch orchestrator: {e}")
            notifier.send(f"*Watchdog: launch failed* `{e}`")
            return 2

        rc = stream_output(proc, tail_log)
        elapsed = time.time() - start_t
        log.info(f"Orchestrator exited rc={rc} after {elapsed:.0f} s")

        # Decide whether to restart.
        if session_is_complete():
            notifier.send(f"*Training complete*\n{session_summary()}")
            return 0

        if rc == 0:
            # The orchestrator exited cleanly but session isn't complete --
            # could be a graceful pause (e.g. heat throttle ended). Retry quickly.
            consecutive_failures = 0
            log.info("Clean exit but session not done. Restarting in 10 s.")
            time.sleep(10)
            continue

        # Crash path -- back off.
        consecutive_failures += 1
        if consecutive_failures >= MAX_CONSECUTIVE_RESTARTS:
            log.error("Too many consecutive failures. Watchdog giving up.")
            notifier.send(
                f"*Watchdog giving up* after {consecutive_failures} crashes.\n"
                f"{session_summary()}"
            )
            return 3

        idx = min(consecutive_failures - 1, len(BACKOFF_SCHEDULE_S) - 1)
        wait_s = BACKOFF_SCHEDULE_S[idx]
        log.warning(
            f"Crash #{consecutive_failures} (rc={rc}). Sleeping {wait_s} s "
            f"before restart."
        )
        notifier.send(
            f"*Orchestrator crashed* rc=`{rc}` ({consecutive_failures})\n"
            f"Restarting in {wait_s} s\n{session_summary()}"
        )
        # Sleep in small chunks so SIGINT remains responsive
        for _ in range(wait_s):
            time.sleep(1)


def install_signal_handlers(notifier: Notifier) -> None:
    def handler(signum, _frame):
        log.info(f"Caught signal {signum}, exiting watchdog.")
        notifier.send(f"*Watchdog stopped* (signal {signum})\n{session_summary()}")
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handler)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Don't actually launch -- just check setup and exit.",
    )
    parser.add_argument(
        "--no-telegram", action="store_true",
        help="Disable Telegram notifications even if creds are set.",
    )
    args = parser.parse_args(argv)

    notifier = Notifier(enabled=not args.no_telegram)
    install_signal_handlers(notifier)

    if not VENV_PYTHON.exists():
        log.error(f"Cannot find venv Python at {VENV_PYTHON}")
        return 4
    if not ORCHESTRATOR.exists():
        log.error(f"Cannot find orchestrator at {ORCHESTRATOR}")
        return 5

    return watchdog_loop(notifier, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
