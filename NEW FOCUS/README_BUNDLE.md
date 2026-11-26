# Drop-in Bundle — May 1, 2026 Update
**For:** Oo Jian Hong (22004678) | MR-TAE FYP

This folder contains four artifacts for the next two weeks' work, plus a bonus
test-rendered demonstration figure. Each file is a drop-in for your existing
`FYP_MASTER/` tree at the path indicated.

---

## 1. `TWO_WEEK_PROGRESS_REPORTS.md`
**Where:** `FYP_MASTER/docs/progress/` (or wherever you keep prior reports).

Two reports drafted in your supervisor's exact template, covering the gap
since the Apr 13–19 update:
- Apr 20–26: NCC reconciliation, signal-overlay figures (lecturer's ask),
  flat-SNR baseline, CleanUNet-1D integration begun, noise-narrative fix.
- Apr 27–May 3: CleanUNet-1D benchmark, kernel optimizations, Noise2Void
  fine-tuning, 24/7 watchdog, knowledge distillation prep.

**Action:** edit the bracketed `[INSERT_VALUE]` placeholders with your actual
numbers as you run the work, then submit.

## 2. `PROJECT_IMPROVEMENT_PLAN.md`
**Where:** `FYP_MASTER/docs/`

Scoped improvement plan for the remaining 4 weeks. Tagged **🔴 DO** (thesis
blockers), **🟡 SHOULD** (high-value polish), **🟢 FUTURE WORK** (Chapter 6
name-drops only — *do not implement now*). Includes:
- A 4-week calendar
- A "what to stop doing" section (no Mamba, no Lightning migration, no more
  ablations — protect your time)
- Concrete library upgrade matrix with time costs
- Verified citations for the SOTA section in Chapter 6

## 3. `generate_method_comparison_figures.py`
**Where:** `FYP_MASTER/evaluation/generate_method_comparison_figures.py`

Drop-in Python script that addresses the supervisor's Apr 18 feedback
verbatim. Run after every benchmark refresh:

```bash
python evaluation/generate_method_comparison_figures.py
```

Produces three sets of outputs:
- `results/{model_name}/signal_overlay_SNR{XX}dB.png` — 4-panel figure per
  model per SNR level. Panel (b) is the overlay the lecturer asked for.
- `results/method_comparison_grid_SNR{XX}dB.png` — one-page grid summary.
- `results/method_comparison_residuals_SNR{XX}dB.png` — residual grid.
- `results/method_comparison_summary.json` — machine-readable summary for
  downstream documents (RESULTS_COMPARISON.md auto-fill, etc.).

A demo render with synthetic data (`demo_signal_overlay.png`) is included
in this folder so you can see exactly what your supervisor will receive.

**Adjust if needed:** the `load_model()` function near the bottom of the
script tries three different checkpoint formats. If your trainer saves
checkpoints in yet another shape, edit just that one function.

## 4. `watchdog.py`
**Where:** `FYP_MASTER/pipeline/watchdog.py`

Python watchdog for 24/7 fault-tolerant orchestration:
- Launches `mlflow_orchestrator.py --resume` and watches the process.
- Restarts on crash with exponential back-off (30 s → 600 s cap).
- Reads `session_state.json` to detect "all models done" and exits cleanly.
- Optional Telegram notifications (set `TELEGRAM_BOT_TOKEN` and
  `TELEGRAM_CHAT_ID` env vars; if not set, just logs to file).
- Stdlib-only by default — no Lightning, Hydra, or DAG-orchestrator
  dependencies. The `requests` library is the only optional dependency
  (Telegram only).

Test before installing:
```bash
python pipeline/watchdog.py --dry-run --no-telegram
```

## 5. `install_watchdog_task.ps1`
**Where:** `FYP_MASTER/scripts/install_watchdog_task.ps1`

Right-click → Run as Administrator. Registers the watchdog with Windows Task
Scheduler so it auto-starts on logon, on system boot, and retries hourly:

```powershell
.\scripts\install_watchdog_task.ps1               # install
.\scripts\install_watchdog_task.ps1 -ShowStatus   # check it
.\scripts\install_watchdog_task.ps1 -Uninstall    # remove
```

Also prints the four `powercfg` commands you need to run once to disable
sleep/standby on the training machine.

---

## Order of operations (today's actions)

1. Drop the five files into your project at the indicated paths.
2. Run `python evaluation/generate_method_comparison_figures.py` — gives you
   the figures the supervisor asked for *today*.
3. Reconcile the NCC discrepancy (Section 1.1 of the improvement plan).
4. Edit the placeholders in the two progress reports as the actual values
   come in.
5. Once you're happy with the watchdog dry-run output, run the PowerShell
   installer.
6. Email the reports to the supervisor with the three best signal-overlay
   PNGs attached as evidence.

---

## Ground rules I followed when drafting this

- **No scope creep.** Mamba, diffusion models, Lightning, Hydra, and Ray are
  all explicitly *not* recommended for the next 4 weeks. They go in
  Chapter 6 as Future Work, not in your code.
- **No fabricated metrics.** Every number that needs filling in is a clearly
  marked `[INSERT_VALUE]`. Don't ship the reports with placeholders intact.
- **Copyright-clean.** All recommended SOTA papers are referenced by author
  + venue + year — no verbatim quoting. Verify each before citing.
- **Stdlib-friendly.** The watchdog and figure script have minimal
  dependencies so they don't break your environment.
- **Windows-native.** The Task Scheduler installer is the right level of
  ceremony for a single-machine 24/7 setup. NSSM, systemd, Kubernetes,
  and Airflow are all overkill at this stage.
