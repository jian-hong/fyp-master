"""
pipeline/autonomous_loop.py
15-Day Autonomous Campaign Controller — MR-TAE FYP

Replaces the direct invocation of mlflow_orchestrator.py as the top-level
entry point. The watchdog.py should watch THIS process; this process
orchestrates the seven campaign phases described in config/campaign_15day.yaml.

Usage:
    python pipeline/autonomous_loop.py --start           # begin fresh campaign
    python pipeline/autonomous_loop.py --resume          # resume from campaign_state.json
    python pipeline/autonomous_loop.py --status          # print current state
    python pipeline/autonomous_loop.py --dry-run         # validate config, no execution
    python pipeline/autonomous_loop.py --force-phase X   # skip to specific phase (dangerous)
    python pipeline/autonomous_loop.py --eval-only       # run evaluation only, no training

    # After human reviews a failed phase and wants to retry:
    python pipeline/autonomous_loop.py --resume --retry-phase phase_A_train

Drop into: FYP_MASTER/pipeline/autonomous_loop.py
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
CAMPAIGN_STATE_FILE = PROJECT_ROOT / "campaign_state.json"
_cfg_primary = PROJECT_ROOT / "config" / "campaign_15day.yaml"
CAMPAIGN_CFG_FILE = (
    _cfg_primary if _cfg_primary.is_file()
    else PROJECT_ROOT / "campaign_15day.yaml"
)
LOGS_DIR = PROJECT_ROOT / "logs"
DIGESTS_DIR = LOGS_DIR / "digests"

_IS_WIN = platform.system() == "Windows"
_VENV_PYTHON = (
    PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if _IS_WIN
    else PROJECT_ROOT / ".venv" / "bin" / "python"
)

LOGS_DIR.mkdir(parents=True, exist_ok=True)
DIGESTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "campaign.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("campaign")


# ── State dataclasses ─────────────────────────────────────────────────────────
@dataclass
class PhaseState:
    status: str = "pending"          # pending | running | done | failed | skipped
    start_iso: str | None = None
    end_iso: str | None = None
    attempts: int = 0
    metrics: dict = field(default_factory=dict)
    output_files: list = field(default_factory=list)
    notes: str = ""


@dataclass
class CampaignState:
    campaign_version: str = "15day_v2"
    start_iso: str = ""
    target_end_iso: str = ""
    current_phase_id: str = ""
    phases: dict = field(default_factory=dict)
    last_eval_iso: str | None = None
    last_digest_iso: str | None = None
    best_model_id: str | None = None
    best_real_ncc: float | None = None
    figure_manifest: list = field(default_factory=list)
    error_log: list = field(default_factory=list)


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def save_state(state: CampaignState) -> None:
    tmp = CAMPAIGN_STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(state), indent=2, default=str), encoding="utf-8")
    tmp.replace(CAMPAIGN_STATE_FILE)


def load_state() -> CampaignState | None:
    if not CAMPAIGN_STATE_FILE.exists():
        return None
    try:
        d = json.loads(CAMPAIGN_STATE_FILE.read_text(encoding="utf-8"))
        state = CampaignState(**{k: v for k, v in d.items() if k in CampaignState.__dataclass_fields__})
        state.phases = {k: PhaseState(**v) if isinstance(v, dict) else v for k, v in d.get("phases", {}).items()}
        return state
    except Exception as e:
        log.warning(f"Could not parse campaign_state.json: {e}. Starting fresh.")
        return None


def load_cfg() -> dict:
    with CAMPAIGN_CFG_FILE.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Notifications ─────────────────────────────────────────────────────────────
class Notifier:
    def __init__(self, cfg: dict):
        self.enabled = cfg.get("campaign", {}).get("notifications", {}).get("enabled", False)
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        if self.enabled and not (self.token and self.chat_id):
            log.info("Telegram not configured — notifications disabled.")
            self.enabled = False

    def send(self, msg: str) -> None:
        if not self.enabled:
            return
        try:
            import requests
            requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={"chat_id": self.chat_id, "text": msg, "parse_mode": "Markdown"},
                timeout=10,
            )
        except Exception as e:
            log.debug(f"Telegram send failed: {e}")


# ── Subprocess runner ─────────────────────────────────────────────────────────
def run_subprocess(
    script: str,
    args: list[str],
    timeout_hours: float,
    state: CampaignState,
    phase_id: str,
    label: str = "",
) -> int:
    """
    Launch a project script as a subprocess. Streams output to both the log
    and a phase-specific tail file. Returns exit code.
    On Windows, uses CREATE_NEW_PROCESS_GROUP so the child keeps running
    if the parent receives a Ctrl-C during interactive testing.
    """
    python = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable
    script_path = PROJECT_ROOT / script
    cmd = [python, str(script_path), *[str(a) for a in args]]

    log.info(f"[{phase_id}] launching: {' '.join(cmd)}")

    creationflags = 0
    if _IS_WIN:
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    tail_path = LOGS_DIR / f"{phase_id}_tail.log"
    with tail_path.open("a", encoding="utf-8") as tail:
        tail.write(f"\n=== {label or phase_id} {_now()} ===\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            encoding="utf-8",
            creationflags=creationflags,
        )
        assert proc.stdout is not None
        deadline = time.time() + timeout_hours * 3600
        for line in proc.stdout:
            line = line.rstrip()
            log.info(f"  [{phase_id}] {line}")
            tail.write(line + "\n")
            tail.flush()
            if time.time() > deadline:
                log.error(f"[{phase_id}] timeout after {timeout_hours}h, killing process")
                proc.kill()
                state.error_log.append({"phase": phase_id, "error": "timeout", "ts": _now()})
                save_state(state)
                return -1

    return proc.wait()


# ── Evaluation helper ─────────────────────────────────────────────────────────
def run_evaluation(state: CampaignState, phase_id: str, cfg: dict) -> dict:
    """
    Run auto_evaluator.py after a phase. Updates state with best NCC.
    Returns dict of metrics.
    """
    eval_script = PROJECT_ROOT / "evaluation" / "auto_evaluator.py"
    if not eval_script.exists():
        log.warning("auto_evaluator.py not found — skipping eval")
        return {}
    rc = run_subprocess(
        "evaluation/auto_evaluator.py",
        ["--phase", phase_id, "--verify-hashes"],
        timeout_hours=2,
        state=state,
        phase_id=phase_id,
        label="evaluation",
    )
    # Load eval output
    eval_out = PROJECT_ROOT / "results" / f"eval_{phase_id}_summary.json"
    if eval_out.exists():
        try:
            metrics = json.loads(eval_out.read_text())
            if "best_real_ncc" in metrics:
                ncc = float(metrics["best_real_ncc"])
                if state.best_real_ncc is None or ncc > state.best_real_ncc:
                    state.best_real_ncc = ncc
                    state.best_model_id = metrics.get("best_model_id")
                    log.info(f"New best real NCC: {ncc:.4f} ({state.best_model_id})")
            state.last_eval_iso = _now()
            save_state(state)
            return metrics
        except Exception as e:
            log.warning(f"Could not parse eval summary: {e}")
    return {}


def run_figures(state: CampaignState, phase_id: str) -> None:
    """Run the figure generator and update state manifest."""
    fig_script = PROJECT_ROOT / "evaluation" / "generate_method_comparison_figures.py"
    if not fig_script.exists():
        log.warning("generate_method_comparison_figures.py not found — skipping")
        return
    run_subprocess(
        "evaluation/generate_method_comparison_figures.py",
        ["--snr-levels", "-15", "-5", "5"],
        timeout_hours=1,
        state=state,
        phase_id=phase_id,
        label="figures",
    )
    # Collect generated PNG paths
    figs = sorted((PROJECT_ROOT / "results").rglob("signal_overlay_*.png"))
    state.figure_manifest = [str(f.relative_to(PROJECT_ROOT)) for f in figs]
    save_state(state)


# ── Daily digest ──────────────────────────────────────────────────────────────
def generate_digest(state: CampaignState, cfg: dict, notifier: Notifier) -> None:
    """Build a markdown digest from current campaign state and write to logs/digests/."""
    digest_script = PROJECT_ROOT / "pipeline" / "daily_digest.py"
    if digest_script.exists():
        run_subprocess(
            "pipeline/daily_digest.py",
            ["--state", str(CAMPAIGN_STATE_FILE)],
            timeout_hours=0.25,
            state=state,
            phase_id="digest",
        )
    else:
        # Fallback: build a minimal digest inline
        day_n = 1
        if state.start_iso:
            try:
                start = datetime.fromisoformat(state.start_iso)
                now = datetime.now(timezone.utc).astimezone()
                day_n = max(1, (now - start).days + 1)
            except Exception:
                pass

        lines = [
            f"# Campaign Digest — Day {day_n} — {_now()}",
            "",
            f"**Current phase:** `{state.current_phase_id}`",
            f"**Best real NCC so far:** `{state.best_real_ncc:.4f}`" if state.best_real_ncc else "**Best real NCC:** not yet measured",
            f"**Best model:** `{state.best_model_id or 'TBD'}`",
            "",
            "## Phase Summary",
        ]
        for pid, ps in state.phases.items():
            p = ps if isinstance(ps, PhaseState) else PhaseState(**ps)
            ncc_str = f"NCC={p.metrics.get('best_real_ncc','?')}" if p.metrics else ""
            lines.append(f"- `{pid}` → **{p.status}** {ncc_str}")

        lines += [
            "",
            "## Figures Generated",
            f"Total: {len(state.figure_manifest)}",
        ]
        for f in state.figure_manifest[:10]:
            lines.append(f"  - `{f}`")
        if len(state.figure_manifest) > 10:
            lines.append(f"  ... and {len(state.figure_manifest)-10} more")

        if state.error_log:
            lines += ["", "## Errors", ""]
            for e in state.error_log[-5:]:
                lines.append(f"- [{e.get('ts','')}] `{e.get('phase','?')}`: {e.get('error','')}")

        digest_text = "\n".join(lines)
        digest_date = datetime.now().strftime("%Y-%m-%d_%H%M")
        digest_file = DIGESTS_DIR / f"digest_{digest_date}.md"
        digest_file.write_text(digest_text, encoding="utf-8")
        log.info(f"Digest written: {digest_file}")
        notifier.send(f"*Campaign Digest — Day {day_n}*\n{state.current_phase_id}\nBest NCC: {state.best_real_ncc or 'N/A'}\nFigures: {len(state.figure_manifest)}")

    state.last_digest_iso = _now()
    save_state(state)


# ── Phase runner ──────────────────────────────────────────────────────────────
def run_phase(phase_cfg: dict, state: CampaignState, cfg: dict, notifier: Notifier) -> bool:
    """
    Execute a single phase. Returns True if success criteria are met.
    Updates state throughout.
    """
    pid = phase_cfg["id"]
    phase = state.phases.get(pid)
    if phase is None:
        phase = PhaseState()
        state.phases[pid] = phase
    if isinstance(phase, dict):
        phase = PhaseState(**phase)
        state.phases[pid] = phase

    # Skip if dependency failed and skip_if_phase_failed is set
    skip_dep = phase_cfg.get("skip_if_phase_failed")
    if skip_dep:
        dep = state.phases.get(skip_dep)
        if dep and (dep.status if isinstance(dep, PhaseState) else dep.get("status")) == "failed":
            log.info(f"[{pid}] Skipping because dependency {skip_dep} failed.")
            phase.status = "skipped"
            phase.notes = f"Dependency {skip_dep} failed"
            save_state(state)
            return True  # Don't block the campaign

    # Already done
    if phase.status == "done":
        log.info(f"[{pid}] Already complete — skipping.")
        return True

    notifier.send(f"*Phase started:* `{pid}`\n{phase_cfg.get('name','')}")
    phase.status = "running"
    phase.start_iso = _now()
    state.current_phase_id = pid
    save_state(state)

    max_retries = phase_cfg.get("on_failure", {}).get("max_retries", 1)

    for attempt in range(1, max_retries + 2):
        phase.attempts = attempt
        save_state(state)

        subproc_cfg = phase_cfg.get("subprocess", {})
        rc = run_subprocess(
            subproc_cfg["script"],
            subproc_cfg.get("args", []),
            subproc_cfg.get("timeout_hours", 24),
            state=state,
            phase_id=pid,
            label=f"attempt {attempt}",
        )

        # Post-subprocess (e.g. retrain after HPO)
        post_cfg = phase_cfg.get("post_subprocess")
        if post_cfg and rc == 0:
            rc_post = run_subprocess(
                post_cfg["script"],
                post_cfg.get("args", []),
                post_cfg.get("timeout_hours", 24),
                state=state,
                phase_id=pid,
                label="post_subprocess",
            )
            if rc_post != 0:
                log.warning(f"[{pid}] post_subprocess returned rc={rc_post} — continuing anyway")

        # Run evaluation
        metrics = {}
        if phase_cfg.get("eval_after", True):
            metrics = run_evaluation(state, pid, cfg)
            phase.metrics = metrics
            save_state(state)

        # Check success criteria
        success = _check_criteria(phase_cfg.get("success_criteria", {}), metrics, state)

        if success:
            phase.status = "done"
            phase.end_iso = _now()
            save_state(state)
            notifier.send(
                f"*Phase complete:* `{pid}`\n"
                f"Real NCC: {metrics.get('best_real_ncc', 'N/A')}\n"
                f"Attempt: {attempt}"
            )
            if phase_cfg.get("generate_figures_after"):
                run_figures(state, pid)
            if phase_cfg.get("digest_after"):
                generate_digest(state, cfg, notifier)
            return True

        failure_action = phase_cfg.get("on_failure", {}).get("action", "fail")
        log.warning(f"[{pid}] attempt {attempt} did not meet criteria. Action: {failure_action}")

        if attempt >= max_retries + 1:
            break

        if failure_action == "extend_phase3_epochs":
            extend_by = phase_cfg.get("on_failure", {}).get("extend_by_epochs", 20)
            _extend_phase3_epochs(extend_by)
            log.info(f"Extended Phase 3 training by {extend_by} epochs. Retrying.")
        elif failure_action == "skip_and_use_phase_a_best":
            log.info(f"[{pid}] Using Phase A best as fallback.")
            break
        else:
            break

    # Phase failed all attempts
    failure_action = phase_cfg.get("on_failure", {}).get("action", "fail")
    if failure_action == "skip":
        phase.status = "skipped"
        phase.notes = phase_cfg.get("on_failure", {}).get("note", "Skipped on failure")
        log.info(f"[{pid}] marked as skipped (non-blocking)")
        state.error_log.append({"phase": pid, "error": "failed_criteria", "ts": _now()})
        save_state(state)
        if phase_cfg.get("digest_after"):
            generate_digest(state, cfg, notifier)
        return True  # Skip = don't block campaign

    phase.status = "failed"
    phase.end_iso = _now()
    state.error_log.append({"phase": pid, "error": "all_attempts_failed", "ts": _now()})
    save_state(state)
    notifier.send(f"*Phase FAILED:* `{pid}` after {phase.attempts} attempts")
    if phase_cfg.get("digest_after"):
        generate_digest(state, cfg, notifier)
    return False


def _check_criteria(criteria: dict, metrics: dict, state: CampaignState) -> bool:
    """Return True only if all specified criteria are met. Unknown metrics pass."""
    if not criteria:
        return True

    # models_complete_count
    if "models_complete_count" in criteria:
        try:
            ss = json.loads((PROJECT_ROOT / "session_state.json").read_text())
            models = ss.get("models", {})
            done = sum(1 for v in models.values() if v.get("status") == "done")
            required = criteria["models_complete_count"]
            if done < required:
                log.info(f"Criteria: models_complete_count {done}/{required} not met")
                return False
        except Exception as e:
            log.warning(f"Could not read session_state for model count: {e}")

    # best_real_ncc_min
    if "best_real_ncc_min" in criteria:
        actual = state.best_real_ncc or metrics.get("best_real_ncc", 0)
        if actual is None or float(actual) < criteria["best_real_ncc_min"]:
            log.info(f"Criteria: best_real_ncc {actual} < {criteria['best_real_ncc_min']}")
            return False

    # vs traditional baselines
    if "neural_vs_savgol_snr_minus5_delta" in criteria:
        savgol_ref = 0.819
        neural_at_m5 = metrics.get("ncc_at_snr_minus5", 0) or 0
        delta = float(neural_at_m5) - savgol_ref
        required = float(criteria["neural_vs_savgol_snr_minus5_delta"])
        if delta < required:
            log.info(f"Criteria: neural NCC at -5dB ({neural_at_m5:.3f}) - savgol ({savgol_ref:.3f}) = {delta:.3f} < {required}")
            return False

    # ncc_improvement_over_phase_a
    if "ncc_improvement_over_phase_a" in criteria:
        phase_a = state.phases.get("phase_A_train")
        if phase_a:
            a_ncc = (phase_a.metrics if isinstance(phase_a, PhaseState) else phase_a.get("metrics", {})).get("best_real_ncc", 0) or 0
            current = state.best_real_ncc or 0
            improvement = float(current) - float(a_ncc)
            if improvement < criteria["ncc_improvement_over_phase_a"]:
                log.info(f"Criteria: NCC improvement {improvement:.4f} < {criteria['ncc_improvement_over_phase_a']}")
                return False

    # figures_generated
    if "figures_generated" in criteria:
        actual = len(state.figure_manifest)
        if actual < criteria["figures_generated"]:
            log.info(f"Criteria: figures_generated {actual} < {criteria['figures_generated']}")
            return False

    # csv_tables_generated
    if "csv_tables_generated" in criteria:
        csvs = list((PROJECT_ROOT / "results").glob("thesis_table_*.csv"))
        if len(csvs) < criteria["csv_tables_generated"]:
            log.info(f"Criteria: csv tables {len(csvs)} < {criteria['csv_tables_generated']}")
            return False

    return True


def _extend_phase3_epochs(n: int) -> None:
    """Increment phase3_epochs in training_config.yaml."""
    cfg_path = PROJECT_ROOT / "config" / "training_config.yaml"
    if not cfg_path.exists():
        log.warning("training_config.yaml not found — cannot extend Phase 3")
        return
    import re
    text = cfg_path.read_text()
    def _inc(m):
        return m.group(0).replace(m.group(1), str(int(m.group(1)) + n))
    new_text = re.sub(r"phase3_epochs:\s*(\d+)", _inc, text)
    cfg_path.write_text(new_text)
    log.info(f"Extended phase3_epochs by {n}")


# ── Status check helper ───────────────────────────────────────────────────────
def check_previous_phase_dependency(phase_cfg: dict, state: CampaignState) -> bool:
    dep = phase_cfg.get("depends_on")
    if not dep:
        return True
    dep_state = state.phases.get(dep)
    if dep_state is None:
        return False
    status = dep_state.status if isinstance(dep_state, PhaseState) else dep_state.get("status")
    return status in ("done", "skipped")


# ── Periodic digest scheduler ─────────────────────────────────────────────────
def _should_generate_digest(state: CampaignState, cfg: dict) -> bool:
    """Return True if 12+ hours have passed since last digest."""
    if not state.last_digest_iso:
        return True
    try:
        last = datetime.fromisoformat(state.last_digest_iso)
        delta_h = (datetime.now(timezone.utc).astimezone() - last).total_seconds() / 3600
        return delta_h >= 12
    except Exception:
        return True


# ── Signal handling ───────────────────────────────────────────────────────────
_SHUTDOWN_REQUESTED = False

def _handle_signal(signum, _frame):
    global _SHUTDOWN_REQUESTED
    log.info(f"Signal {signum} received — requesting graceful shutdown after current phase")
    _SHUTDOWN_REQUESTED = True


signal.signal(signal.SIGINT, _handle_signal)
if hasattr(signal, "SIGTERM"):
    signal.signal(signal.SIGTERM, _handle_signal)


# ── Main campaign loop ────────────────────────────────────────────────────────
def run_campaign(state: CampaignState, cfg: dict, notifier: Notifier, dry_run: bool, retry_phase: str | None) -> int:
    phases = cfg.get("phases", [])
    log.info(f"Campaign has {len(phases)} phases.")

    if dry_run:
        log.info("DRY RUN — validating config, printing plan, no execution")
        for ph in phases:
            pid = ph["id"]
            ps = state.phases.get(pid, PhaseState())
            log.info(f"  Phase {pid}: status={ps.status if isinstance(ps, PhaseState) else ps.get('status','pending')}")
        return 0

    # Reset a specific phase for retry
    if retry_phase:
        if retry_phase in state.phases:
            state.phases[retry_phase] = PhaseState()
            save_state(state)
            log.info(f"Reset phase {retry_phase} for retry")

    notifier.send(f"*Campaign started/resumed*\nPhases: {len(phases)}\nBest NCC so far: {state.best_real_ncc or 'N/A'}")

    for phase_cfg in phases:
        if _SHUTDOWN_REQUESTED:
            log.info("Graceful shutdown requested — pausing between phases.")
            notifier.send("*Campaign paused* — will resume on next start.")
            return 0

        pid = phase_cfg["id"]
        ps = state.phases.get(pid, PhaseState())
        if isinstance(ps, dict):
            ps = PhaseState(**ps)

        # Skip completed phases
        if ps.status == "done":
            log.info(f"Phase {pid} already done — skipping.")
            continue

        # Skip phases blocked by incomplete dependencies
        if not check_previous_phase_dependency(phase_cfg, state):
            dep = phase_cfg.get("depends_on")
            dep_state = state.phases.get(dep, PhaseState())
            dep_status = dep_state.status if isinstance(dep_state, PhaseState) else dep_state.get("status")
            if dep_status == "failed":
                log.error(f"Phase {pid} blocked: dependency {dep} failed. Stopping campaign.")
                return 2
            log.info(f"Phase {pid} waiting for dependency {dep} (status={dep_status}) — this shouldn't happen")
            continue

        # Periodic digest before starting phase
        if _should_generate_digest(state, cfg):
            generate_digest(state, cfg, notifier)

        success = run_phase(phase_cfg, state, cfg, notifier)

        if not success:
            log.error(f"Phase {pid} failed and is not skippable. Campaign halted.")
            notifier.send(f"*CAMPAIGN HALTED* at phase `{pid}`\nRun with --retry-phase {pid} after manual fix.")
            return 1

    # All phases done
    log.info("All phases complete. Generating final digest.")
    generate_digest(state, cfg, notifier)
    notifier.send(
        f"*Campaign COMPLETE*\nBest model: `{state.best_model_id}`\n"
        f"Best real NCC: {state.best_real_ncc:.4f}\n"
        f"Figures: {len(state.figure_manifest)}"
    )
    return 0


# ── Entry point ───────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", action="store_true", help="Start a new campaign (will overwrite existing state)")
    parser.add_argument("--resume", action="store_true", help="Resume from campaign_state.json")
    parser.add_argument("--status", action="store_true", help="Print status and exit")
    parser.add_argument("--dry-run", action="store_true", help="Validate config, print plan, no execution")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation and generate figures only")
    parser.add_argument("--force-phase", default=None, help="Skip all phases up to this one")
    parser.add_argument("--retry-phase", default=None, help="Reset and retry a specific phase")
    parser.add_argument("--config", default=str(CAMPAIGN_CFG_FILE), help="Path to campaign YAML")
    args = parser.parse_args(argv)

    # Load config
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        log.error(f"Campaign config not found: {cfg_path}")
        return 3
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    notifier = Notifier(cfg)

    # Load or initialize state
    if args.start:
        log.info("Starting fresh campaign.")
        state = CampaignState()
        state.start_iso = _now()
        state.phases = {ph["id"]: PhaseState() for ph in cfg.get("phases", [])}
        save_state(state)
    elif args.resume or args.dry_run or args.eval_only:
        state = load_state()
        if state is None:
            log.info("No campaign state found — initializing new state.")
            state = CampaignState()
            state.start_iso = _now()
            state.phases = {ph["id"]: PhaseState() for ph in cfg.get("phases", [])}
            save_state(state)
    else:
        parser.print_help()
        return 0

    # Status display
    if args.status:
        print(json.dumps(asdict(state), indent=2, default=str))
        return 0

    # Eval only
    if args.eval_only:
        metrics = run_evaluation(state, "manual_eval", cfg)
        run_figures(state, "manual_eval")
        generate_digest(state, cfg, notifier)
        print(json.dumps(metrics, indent=2))
        return 0

    # Force to a specific phase
    if args.force_phase:
        phases = cfg.get("phases", [])
        phase_ids = [p["id"] for p in phases]
        if args.force_phase not in phase_ids:
            log.error(f"Unknown phase: {args.force_phase}")
            return 4
        for pid in phase_ids:
            if pid == args.force_phase:
                break
            if pid not in state.phases or state.phases[pid].status != "done":
                state.phases[pid] = PhaseState(status="skipped", notes="force-skipped by --force-phase")
        save_state(state)
        log.info(f"Force-advanced to {args.force_phase}")

    return run_campaign(state, cfg, notifier, args.dry_run, args.retry_phase)


if __name__ == "__main__":
    raise SystemExit(main())
