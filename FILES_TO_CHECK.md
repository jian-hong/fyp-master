# FILES THE NEW AGENT (CURSOR / CLAUDE CODE) MUST READ FIRST
# Send these to the new agent window before it starts coding.
# The agent MUST cat/read each one before writing any new code.

## MANDATORY READS (agent cannot proceed without these)

| Priority | File | Why |
|----------|------|-----|
| 🔴 1 | `pipeline/mlflow_orchestrator.py` | CLI args (`--resume`, `--model`, `--hparams`, `--wgan-aug`), exit codes, subprocess interface |
| 🔴 2 | `training/resilience.py` | `session_state.json` exact schema — which fields, what values for status |
| 🔴 3 | `config/model_registry.yaml` | All 13 model IDs exactly as they appear in the registry |
| 🔴 4 | `session_state.json` | Current training state — which models are actually done |
| 🔴 5 | `evaluation/benchmark_runner.py` | CLI args, output format, what keys are in the JSON output |
| 🟡 6 | `config/training_config.yaml` | phase1/2/3 epoch counts, LR, batch size |
| 🟡 7 | `app.py` | Existing Flask routes: `/api/training/resume`, `/api/training/stop`, `/api/training/start` |
| 🟡 8 | `hpo/optuna_study.py` | `--n-trials`, `--study-name`, `--objective` args |
| 🟡 9 | `models/base.py` OR `mr_tae_fusion/models/__init__.py` | `build_model_from_registry(name, registry, cfg)` signature |
| 🟢 10 | `wgan_augmentation/` (whole dir) | What already exists vs what must be built |
| 🟢 11 | `evaluation/shap_analysis.py` | For Phase G figure generation |
| 🟢 12 | `training/trainer.py` | To build n2v_finetune.py and distill.py correctly |

## HOW TO SEND THESE

In your terminal:
```bash
# Dump all mandatory reads in one shot:
cat pipeline/mlflow_orchestrator.py
cat training/resilience.py
cat config/model_registry.yaml
cat session_state.json
cat evaluation/benchmark_runner.py | head -150
```

Or upload individual files to the new agent window.

## WHAT TO TELL THE NEW AGENT ABOUT CURRENT STATE

Copy this block verbatim into the new agent:

```
CURRENT STATE AS OF MAY 9, 2026:
- 5/13 models have some training: MR-TAE-FULL, noBiGRU, noSwin, noAttn, noMTL
- CRITICAL: traditional baselines (Savitzky-Golay) beat neural models at SNR -15 and -5 dB
  - SNR -5: savgol NCC=0.819, best neural NCC=0.618 → neural LOSES by 0.201
  - SNR +5: wavelet NCC=0.904, best neural NCC=0.924 → neural barely wins
- Models are underperforming because Phase 3 (hard SNR) training is incomplete
- Fix: ensure all models complete their full 3-phase curriculum before Phase A is declared done
- Files from previous Claude session are in "NEW FOCUS/" folder:
  - generate_method_comparison_figures.py (working, tested)
  - watchdog.py (process crash recovery)
  - install_watchdog_task.ps1 (Task Scheduler)
- New files to drop in (from this handoff):
  - pipeline/autonomous_loop.py
  - config/campaign_15day.yaml
  - pipeline/daily_digest.py
  - evaluation/auto_evaluator.py
  - scripts/start_15day.ps1
```

## INTERFACE CONTRACTS THE NEW AGENT MUST HONOUR

### session_state.json (read by autonomous_loop.py)
The existing `training/resilience.py` manages this file. The autonomous_loop
reads two things from it:
1. `state["models"][model_id]["status"]` — must be `"done"` for a model to count as complete
2. The full model list must match `config/model_registry.yaml`

### eval_{phase_id}_summary.json (written by auto_evaluator.py)
The autonomous_loop reads these keys:
- `best_real_ncc` — float
- `best_model_id` — string matching a key in model_registry
- `ncc_at_snr_minus5` — float (NCC of best model at SNR = -5 dB)
- `model_rankings` — list of dicts with keys: model, real_ncc, synth_ncc, latency_ms, params_m

### campaign_state.json (written/read by autonomous_loop.py)
Schema is defined in the handoff prompt. Agent must NOT change field names.

## RED FLAGS TO WATCH FOR

1. If `session_state.json` schema doesn't match what `resilience.py` writes:
   → Fix the read logic in `autonomous_loop._check_criteria`, NOT resilience.py

2. If `benchmark_runner.py` doesn't accept `--phase` or doesn't write JSON:
   → Wrap it in `auto_evaluator.py` which does have that interface

3. If `app.py` dashboard has a different resume route than `/api/training/resume`:
   → Update `app.py` to add the campaign controller route, don't rename existing ones

4. If WGAN scripts don't exist in `wgan_augmentation/`:
   → Build `wgan_augmentation/wgan_noise_trainer.py` from scratch per Phase D spec in AGENT_HANDOFF_PROMPT.md

5. If the model builder (`build_model_from_registry`) doesn't exist:
   → Check `mr_tae_fusion/models/__init__.py` for an equivalent function
