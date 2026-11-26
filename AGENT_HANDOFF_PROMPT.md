# AGENT HANDOFF — MR-TAE FYP 15-Day Autonomous Campaign
## Paste this entire document as the first message in your new agent window.

---

## WHO YOU ARE

You are a senior ML engineering agent working on an active final-year project.
You have FULL agentic access: file read/write, bash execution, Python execution.
The human will relay your output back to Claude if needed. Treat every task as
production-grade — no stubs, no placeholders, no "TODO" in delivered code.

---

## PROJECT IDENTITY

- **Student:** Oo Jian Hong (22004678), University of Malaya, 2026
- **Project:** Deep-learning AE-PD signal denoising for high-voltage transformer monitoring
- **Model:** MR-TAE (Multi-Resolution Temporal Autoencoder) — MWCNN encoder +
  BiGRU+Swin bottleneck + attention-gated skip connections + dual-head MTL decoder
- **GPU:** RTX 4070 (12 GB VRAM), Windows 11
- **Python env:** `.venv` at `D:\New folder (2)\Data\FYP_MASTER\.venv\Scripts\python.exe`
- **Project root:** `D:\New folder (2)\Data\FYP_MASTER\`

---

## CURRENT STATE (as of May 9, 2026)

### What is done:
- 13 ablation model variants defined in `config/model_registry.yaml`
- 5 of 13 models are partially trained (MR-TAE-FULL, noBiGRU, noSwin, noAttn, noMTL)
- Optuna HPO (30 trials) complete; best config in `config/best_hparams.yaml`
- SHAP/LIME explainability analysis complete on 200 Q.Lin windows
- Chapters 4 and 5 of thesis substantially drafted
- `pipeline/mlflow_orchestrator.py` handles sequential model training, crash-safe
- `training/resilience.py` manages `session_state.json` for epoch-level resume
- `pipeline/watchdog.py` handles process-level crash restart
- `app.py` is a Flask+React dashboard on port 1001

### CRITICAL FINDING from last agent run (DO NOT IGNORE):
Savitzky-Golay traditional baseline is BEATING neural models at low SNR:
  - SNR −15 dB: savgol NCC=0.265 vs best neural NCC=0.203 → neural LOSES by 0.062
  - SNR  −5 dB: savgol NCC=0.819 vs best neural NCC=0.618 → neural LOSES by 0.201
  - SNR  +5 dB: wavelet NCC=0.904 vs best neural NCC=0.924 → neural wins by 0.020
Root cause: models have NOT completed Phase 3 (hard SNR: -20 to -5 dB) training.
Fix: complete Phase 3 training first, then verify neural models exceed baselines at ALL SNRs.
The 15-day campaign must address this as its FIRST priority.

### What is needed:
1. A 15-day autonomous campaign controller (`pipeline/autonomous_loop.py`)
2. Auto-evaluation + figure generation after every phase
3. Daily digest reports (markdown)
4. Metric-gated phase transitions (don't advance until thresholds are met)
5. Integration of existing watchdog.py (watchdog watches autonomous_loop)
6. Updated dashboard integration (app.py API points to autonomous_loop.py)
7. Comprehensive final figures: per-method overlays, NCC comparison tables,
   PSD plots, ablation heatmap, training curves, SHAP visualization

---

## FILES TO READ FIRST (in this order)

Before writing any code, you MUST read these files to understand the interfaces:

1. `pipeline/mlflow_orchestrator.py`       — orchestrator CLI args, exit codes, session_state schema
2. `training/resilience.py`                — session_state.json schema and API
3. `config/model_registry.yaml`            — all 13 model names and ablation flags
4. `config/training_config.yaml`           — hyperparameter defaults
5. `session_state.json`                    — CURRENT training state (what's actually trained)
6. `evaluation/benchmark_runner.py`        — benchmark CLI args and output format
7. `hpo/optuna_study.py`                   — Optuna HPO CLI args
8. `app.py`                                — Flask API endpoints to modify
9. `mr_tae_fusion/models/__init__.py` OR `models/base.py` — model builder interface

Also read these files that were created in the previous session:
10. `pipeline/watchdog.py`                 — existing crash recovery (must integrate, not replace)
11. `evaluation/generate_method_comparison_figures.py` — figure generator (already working)

---

## ARCHITECTURE OF WHAT YOU WILL BUILD

```
Windows Task Scheduler (auto-start on boot)
  └─► start_15day.ps1
        └─► pipeline/autonomous_loop.py  [YOU BUILD THIS — the brain]
              ├─► campaign_state.json    [persistent state, survives crashes]
              ├─► Phase A: train_all     → calls mlflow_orchestrator.py --resume
              ├─► Phase B: hpo_extended  → calls hpo/optuna_study.py
              ├─► Phase C: noise2void    → calls pipeline/n2v_finetune.py [YOU BUILD]
              ├─► Phase D: wgan_train    → calls wgan_augmentation/train_wgan.py
              ├─► Phase E: wgan_retrain  → calls mlflow_orchestrator.py (best model + WGAN)
              ├─► Phase F: distill       → calls pipeline/distill.py [YOU BUILD]
              └─► Phase G: final_report  → calls evaluation/generate_thesis_report.py [YOU BUILD]
              
              After EVERY phase:
              ├─► evaluation/auto_evaluator.py  [YOU BUILD — wraps benchmark_runner.py]
              ├─► evaluation/generate_method_comparison_figures.py  [ALREADY EXISTS]
              └─► pipeline/daily_digest.py      [YOU BUILD — markdown report]
```

---

## PHASE DEFINITIONS AND SUCCESS CRITERIA

### Phase A — Complete Training (Days 1–4, max 96h)
**Task:** Train all 13 model variants to 100 epochs with full 3-phase curriculum.
**Subprocess:** `python pipeline/mlflow_orchestrator.py --resume`
**Success criteria (ALL must be met):**
  - All 13 models have `status=done` in `session_state.json`
  - MR-TAE-FULL-LossB achieves real-set NCC ≥ 0.75 on `data/test_real_qlin.pt`
  - Best model achieves synthetic NCC ≥ 0.65 across all 5 SNR levels
  - Neural best BEATS Savitzky-Golay at SNR −5 dB (NCC delta > 0)  ← MOST IMPORTANT
**On failure:** Extend Phase 3 epochs by 20 (patch training_config.yaml phase3_epochs += 20), retry once.

### Phase B — Extended HPO (Days 5–6, max 48h)
**Task:** Run 50 more Optuna trials on best Phase A model, focusing on Phase 3 performance.
**Subprocess:** `python hpo/optuna_study.py --n-trials 50 --focus-snr -5 -15`
**Success criteria:**
  - Best trial improves real-set NCC by ≥ 0.01 over Phase A best
  - Retrain best config and verify metric improvement
**On failure:** Keep Phase A best, document in digest.

### Phase C — Noise2Void Fine-tuning (Days 7–8, max 48h)
**Task:** Self-supervised fine-tuning on unlabelled Q.Lin real data.
**Script to build:** `pipeline/n2v_finetune.py`
  - Load best checkpoint from Phase A/B
  - Blind-spot mask: structured 5-sample exclusion zone
  - 50 epochs, replay buffer (20% synthetic to prevent catastrophic forgetting)
  - Save as `results/MR-TAE-N2V/best.pt`
**Success criteria:**
  - Real-set NCC improves ≥ 0.01 over Phase B
  - Synthetic NCC degrades ≤ 0.03 (replay buffer constraint)

### Phase D — WGAN-GP Noise Training (Days 9–10, max 48h)
**Task:** Train WGAN-GP on real noise segments extracted from Q.Lin silent gaps.
**Check existing:** `wgan_augmentation/` — may already have stubs.
**Build if missing:** `wgan_augmentation/wgan_noise_trainer.py`
  - Extract inter-pulse gaps via amplitude threshold on Q.Lin `.mat` files
  - Train WGAN-GP: generator + critic, GP weight=10, n_critic=5
  - Stop when FID (or MMD on held-out noise) plateaus for 50 epochs
  - Save: `wgan_augmentation/noise_generator.pth`, `wgan_augmentation/wgan_samples.pt`
**Success criteria:** Generated noise MMD vs real noise < 0.1

### Phase E — WGAN-Augmented Retraining (Days 11–12, max 48h)
**Task:** Retrain best model (Phase B HPO config) with 50% WGAN noise augmentation.
**Subprocess:** `python pipeline/mlflow_orchestrator.py --model MR-TAE-FULL-LossB --wgan-aug 0.5`
**Success criteria:** Real-set NCC improvement ≥ 0.01 over Phase B (should close -15 dB gap)

### Phase F — Knowledge Distillation (Day 13, max 24h)
**Task:** Distil best model → EdgePD-Net (MWCNN-BiGRU, ~1.31M params) for PD-CLIP.
**Script to build:** `pipeline/distill.py`
  - Teacher: best model from Phase E
  - Student: MWCNN-BiGRU architecture (already in registry)
  - Loss: 0.5 * Charbonnier(student, clean) + 0.3 * MSE(student, teacher) + 0.2 * KL(segs)
  - 50 epochs with early stopping
  - Export: `results/EdgePD-Net/best.pt`, `results/EdgePD-Net/edgepd.onnx` (ONNX opset 17)
  - INT8 PTQ with 200-window calibration set
**Success criteria:** Student real NCC ≥ 0.70, latency ≤ 8ms batch=1

### Phase G — Final Benchmarking + Report (Days 14–15, max 48h)
**Task:** Comprehensive evaluation and figure generation for thesis.
**Script to build:** `evaluation/generate_thesis_report.py`
Required outputs (all at 300 DPI):
  - `results/figures/fig_01_method_comparison_grid.png`   — per-model 4-panel at 3 SNRs
  - `results/figures/fig_02_ncc_vs_snr_all_models.png`    — line chart NCC vs SNR {-15,-10,-5,0,5} dB
  - `results/figures/fig_03_ablation_heatmap.png`         — NCC heatmap: rows=models, cols=SNR levels
  - `results/figures/fig_04_training_curves.png`          — NCC/RMSE vs epoch for all models
  - `results/figures/fig_05_efficiency_scatter.png`       — latency vs NCC, bubble=params
  - `results/figures/fig_06_shap_overlay.png`             — SHAP attribution on 3 signal windows
  - `results/figures/fig_07_real_vs_synth_comparison.png` — Q.Lin real signal denoised vs clean
  - `results/figures/fig_08_psd_comparison.png`           — PSD overlay all 13 models
  - `results/figures/fig_09_vs_traditional_baselines.png` — neural best vs savgol/wavelet/bayesshrink
  - `results/figures/fig_10_edgepd_comparison.png`        — EdgePD-Net vs teacher
  - `results/thesis_table_01_all_models.csv`              — full metrics table
  - `results/thesis_table_02_ablation.csv`                — ablation delta table
  - `results/thesis_table_03_vs_sota.csv`                 — vs external SOTA
  - `results/FINAL_REPORT.md`                             — human-readable summary

---

## CAMPAIGN STATE SCHEMA

`campaign_state.json` must follow this exact schema (your code must enforce it):

```json
{
  "campaign_version": "15day_v2",
  "start_iso": "2026-05-09T08:00:00+08:00",
  "target_end_iso": "2026-05-24T08:00:00+08:00",
  "current_phase_id": "phase_A_train",
  "phases": {
    "phase_A_train": {
      "status": "running|done|failed|skipped",
      "start_iso": "...",
      "end_iso": null,
      "attempts": 1,
      "metrics": {
        "best_real_ncc": null,
        "best_synth_ncc": null,
        "models_complete": 0,
        "vs_savgol_delta_snr_minus5": null
      },
      "output_files": [],
      "notes": ""
    }
  },
  "last_eval_iso": null,
  "last_digest_iso": null,
  "best_model_id": null,
  "best_real_ncc": null,
  "figure_manifest": [],
  "error_log": []
}
```

---

## FILE DROP LOCATIONS

All new files you create go here:

| File | Location in project |
|------|---------------------|
| `autonomous_loop.py` | `pipeline/autonomous_loop.py` |
| `campaign_15day.yaml` | `config/campaign_15day.yaml` |
| `campaign_state.json` | `campaign_state.json` (project root) |
| `n2v_finetune.py` | `pipeline/n2v_finetune.py` |
| `distill.py` | `pipeline/distill.py` |
| `auto_evaluator.py` | `evaluation/auto_evaluator.py` |
| `daily_digest.py` | `pipeline/daily_digest.py` |
| `generate_thesis_report.py` | `evaluation/generate_thesis_report.py` |
| `start_15day.ps1` | `scripts/start_15day.ps1` |
| `campaign_state.json` | `campaign_state.json` (root) |

---

## INVARIANTS — NEVER VIOLATE THESE

1. **Test sets are FROZEN.** Never pass `data/test_synthetic_ae.pt` or
   `data/test_real_qlin.pt` to a training DataLoader. Verify SHA256 before every eval.
   Expected hashes are in `docs/TEST_SET_HASHES.md`.

2. **Real Q.Lin data is NEVER used for supervised training.** Only Noise2Void
   (self-supervised, Phase C) and inference is permitted on Q.Lin.

3. **campaign_state.json is written AFTER every subprocess call** (even if it failed).
   Never lose state.

4. **Every subprocess is launched with CREATE_NEW_PROCESS_GROUP on Windows** so
   Ctrl-C in the parent doesn't kill the child's checkpoint.

5. **No DAG orchestration tools** (Airflow, Dagster, Prefect). The campaign controller
   IS the orchestration. It's a single Python process.

6. **All figures saved at 300 DPI.** Thesis requires it.

7. **NCC is the primary metric.** SNR_out and RMSE are secondary. Phase transitions
   are gated on real-set NCC thresholds.

---

## VERIFICATION CHECKLIST (run before declaring any phase complete)

### After Phase A:
```bash
python -c "
import json, pathlib
ss = json.loads(pathlib.Path('session_state.json').read_text())
models = ss.get('models', {})
done = [k for k,v in models.items() if v.get('status')=='done']
print(f'{len(done)}/13 models complete: {done}')
"
python evaluation/auto_evaluator.py --phase A --verify-hashes
```

### After every phase:
```bash
python evaluation/auto_evaluator.py --phase [X]
# Must produce: results/eval_phase_[X]_summary.json
# Must show: best_real_ncc, all 13 models benchmarked, no NaN values
```

### Final figure verification:
```bash
python -c "
import pathlib
figs = list(pathlib.Path('results/figures').glob('fig_*.png'))
print(f'{len(figs)}/10 figures present')
for f in sorted(figs):
    import PIL.Image
    img = PIL.Image.open(f)
    print(f'  {f.name}: {img.size[0]}x{img.size[1]} px')
"
```

---

## COMMUNICATION PROTOCOL

After completing each task, output a structured status block:

```
=== TASK COMPLETE: [task name] ===
Files modified/created: [list]
Key metrics: [NCC values, file sizes, etc.]
Next action: [what you're doing next]
Blockers: [anything the human needs to decide]
=== END STATUS ===
```

If you encounter an error you cannot resolve (missing file, import error,
schema mismatch), output:

```
=== BLOCKED: [error description] ===
File needed: [path]
What the human should do: [specific action]
=== END BLOCKED ===
```

---

## FIRST ACTIONS (execute in this order)

1. `cat pipeline/mlflow_orchestrator.py`           → understand CLI and session schema
2. `cat training/resilience.py`                    → understand session_state.json
3. `cat config/model_registry.yaml`                → get all 13 model names
4. `cat session_state.json 2>/dev/null || echo MISSING` → see current training state
5. `cat evaluation/benchmark_runner.py | head -80` → understand benchmark CLI
6. `cat app.py | grep -A5 "def.*resume\|def.*start\|def.*stop"` → dashboard API
7. Build `pipeline/autonomous_loop.py` (provided as reference in this handoff)
8. Build `config/campaign_15day.yaml`
9. Build `pipeline/daily_digest.py`
10. Build `evaluation/auto_evaluator.py`
11. Build `pipeline/n2v_finetune.py` (Phase C)
12. Build `pipeline/distill.py` (Phase F)
13. Build `evaluation/generate_thesis_report.py` (Phase G)
14. Update `app.py` to point Resume/Stop to autonomous_loop.py
15. Build `scripts/start_15day.ps1`
16. Run `python pipeline/autonomous_loop.py --dry-run` to verify state machine
17. Run `python pipeline/autonomous_loop.py --start` to begin the campaign

---

## REFERENCE CODE

The following files were built in the previous Claude session and are
already in the project under `NEW FOCUS/`:
- `generate_method_comparison_figures.py` — complete, tested, 300 DPI figures
- `watchdog.py` — process crash recovery
- `install_watchdog_task.ps1` — Windows Task Scheduler registration

The autonomous_loop.py you build should INTEGRATE with watchdog.py like this:
  watchdog.py watches autonomous_loop.py (not mlflow_orchestrator.py directly)
  When autonomous_loop.py crashes mid-phase, watchdog restarts it
  autonomous_loop.py reads campaign_state.json and resumes the correct phase

---
## Good luck. The student defends in ~3 weeks. Ship it.
