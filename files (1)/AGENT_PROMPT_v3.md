# AGENT PROMPT v3 — Next Phase Implementation
## For: Claude Opus 4.6 + Subagents
## Status: Training is LIVE (PID 21528). Phase 3 picks up from here.

---

## ⚠️ MANDATORY — ACTIVATE .venv BEFORE ANYTHING ELSE

Every subagent, every shell command, every `python` call must run inside `.venv`.

```bash
# Activate first — always
source .venv/bin/activate

# Verify you are inside the venv (prompt should show (.venv))
which python   # must show: .../project_root/.venv/bin/python
python -c "import torch; print(torch.cuda.get_device_name(0))"

# If you need to install anything missing:
pip install <package>          # no --break-system-packages needed inside venv
# PyTorch (if ever reinstalling):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

**Never use `python3` or the system pip outside the venv.** If a subagent opens a new shell, the first line must be `source .venv/bin/activate`. This applies to every bash block in every task below.

---

## CONTEXT SUMMARY (Read Before Acting)

Training is running. 6 model variants are queued. Here is where everything stands:

**Done:** Signal physics corrected (fs=1MHz, AE kHz range), all 6 models registered, trainer with AMP/thermal/OOM handling, MLflow orchestrator running, GELU activations, linear output layer.

**CRITICAL gap right now:** Fixed test sets do not exist yet. Both test sets must be created BEFORE any model finishes training. This is Task 0 — do it IMMEDIATELY as the first action.

**Your tasks in priority order:**
1. Build test sets NOW (Task 0)
2. Q.Lin loader + signal validation figure (Task 1)
3. Benchmark runner (Task 2)
4. Dashboard Flask backend (Task 3)
5. Optuna HPO (Task 4)
6. SHAP/LIME explainability (Task 5)
7. WGAN noise augmentation (Task 6)
8. Auto-retrain Noise2Void (Task 7)
9. Final documentation (Task 8)

---

## TASK -1 — WIRE IN RESILIENCE SYSTEM (Integrate before running anything)

Three files have been provided: `training/resilience.py`, `pipeline/mlflow_orchestrator.py`, and `resume.sh`. Wire them into the existing codebase:

**Step 1 — Replace old orchestrator** with the new `mlflow_orchestrator.py`. It has `--resume` flag, session state tracking, dynamic early stopping, and SIGINT handling built in.

**Step 2 — Inside `training/trainer.py`**, add the resilience callback calls:
```python
# At trainer init:
from training.resilience import ResilienceManager
# Pass resilience into trainer, call from train loop:
decision = self.resilience.on_epoch_end(epoch, phase, metrics)
if decision["action"] == "skip_model":
    raise EarlyStopException(decision["reason"])
```

**Step 3 — Make `resume.sh` executable:**
```bash
chmod +x resume.sh
```

**Step 4 — Test the resume loop:**
```bash
source .venv/bin/activate
# Start training, then Ctrl+C after 3 epochs
python pipeline/mlflow_orchestrator.py --run-all
# Then immediately resume:
python pipeline/mlflow_orchestrator.py --resume
# Verify it continues from epoch 3, not epoch 0
```

**Step 5 — Verify `RESUME_PROMPT.md` is written** after every epoch. Open it and confirm it has the correct model, epoch, and checkpoint path. This file is your safety net when WiFi drops or Cursor crashes.

---

## TASK 0 — BUILD FIXED TEST SETS (DO THIS FIRST, RIGHT NOW)

Create `data/build_test_sets.py`. This is BLOCKING everything downstream.

```python
"""
data/build_test_sets.py

Builds and saves both fixed evaluation test sets.
Must be run BEFORE any model finishes training.
Test sets are NEVER modified after creation.
"""
```

### Synthetic AE test set (10,000 windows)
- 2048 samples per window at 1 MHz
- SNR levels: -15, -10, -5, 0, +5 dB — 2000 windows per level
- All PD types represented (surface, corona, internal)
- Apply acoustic transfer function to all pulses
- Apply all noise types in composite
- Save both `clean_signals` and `noisy_signals` tensors
- Save `segmentation_masks` for each window
- Output: `data/test_synthetic_ae.pt`

### Real Q.Lin test set (all 90% of Lin et al. data)
- Load all `.mat` files from `data/raw/qlin/`
- Segment into 2048-sample windows (NOT 1-second = 1M sample windows — use sliding window)
- Apply z-score normalization per window (same normalization used in training)
- Save with class labels: 0=1.0mm, 1=1.8mm, 2=2.0mm, 3=2.5mm
- Output: `data/test_real_qlin.pt`

### After saving, print and log:
```python
import hashlib
for path in ['data/test_synthetic_ae.pt', 'data/test_real_qlin.pt']:
    sha = hashlib.sha256(open(path,'rb').read()).hexdigest()
    print(f"SHA256 {path}: {sha}")
    # Write to docs/TEST_SET_HASHES.md — include in thesis
```

---

## TASK 1 — Q.LIN LOADER + SIGNAL VALIDATION

### data/qlin_loader.py
```python
"""
Loads Lin et al. (2022) .mat files.
Handles both scipy.io.loadmat (v5) and h5py (v7.3) formats.
Segments into 2048-sample windows with configurable overlap.
Applies same normalization as training pipeline.
"""
```

Key function signature:
```python
def load_qlin_dataset(
    data_dir: str,
    window_size: int = 2048,
    overlap: float = 0.5,       # 50% overlap between windows
    normalize: bool = True,
    split: float = 0.9,         # 0.9 = 90% test, 10% for WGAN
) -> dict:
    """Returns {'test': (X, y), 'wgan_pool': (X, y)}"""
```

### docs/signal_validation.png
After loading Q.Lin data, generate a 2×3 comparison figure:
- Row 1: 3 synthetic AE pulses (corrected generator, different SNR levels)
- Row 2: 3 real Q.Lin windows of matching class
- Both rows must show similar pulse morphology: sharp onset + 500-1500µs oscillatory tail
- Both rows must show PSD peak in [50kHz, 200kHz] range
- If the synthetic pulses do NOT look similar to real: STOP and flag to researcher before proceeding

---

## TASK 2 — BENCHMARK RUNNER

### evaluation/benchmark_runner.py

```python
"""
Master evaluation script. Runs after all models are trained.
Evaluates every model on BOTH fixed test sets.
Never touches training data.
"""
```

Evaluation protocol:
```python
MODELS_TO_EVALUATE = [
    # Traditional baselines (no checkpoint needed)
    "wavelet_db4_soft", "wavelet_bayesshrink", "savitzky_golay",
    # Deep learning (load from checkpoints)
    "pure_cnn", "unet_1d", "mwcnn",
    "MR-TAE-noSwin", "MR-TAE-noAttn", "MR-TAE-noBiGRU",
    "MR-TAE-noMTL", "MR-TAE-noWavelet", "MR-TAE-FULL",
]

METRICS = {
    "synthetic": ["snr_improvement", "rmse", "ncc", "dice", "false_positive_rate"],
    "real_qlin": ["rmse", "ncc"],  # No segmentation GT for real data
    "efficiency": ["param_count_M", "inference_ms_b1", "inference_ms_b32", "train_time_min"],
}
```

Generated figures (save all to `results/benchmark/`):
1. `comparison_bar.png` — grouped bars: NCC-Synth, NCC-Real, RMSE-Real per model
2. `ablation_heatmap.png` — NCC on 3×3 encoder×bottleneck grid (matplotlib imshow)
3. `efficiency_scatter.png` — inference latency (x) vs. NCC-Real (y), bubble = param count
4. `snr_sweep.png` — NCC vs. input SNR at {-15,-10,-5,0,+5} dB, line per model
5. `comparison_table.json` — machine-readable output for dashboard API

Auto-update `docs/RESULTS_COMPARISON.md` with the full table.

---

## TASK 3 — DASHBOARD FLASK BACKEND (app.py provided)

The Flask backend is already written in `dashboard/app.py`. Your job:

### 3a — Build the React frontend

Structure: `dashboard/frontend/src/`
- `App.jsx` — router (react-router-dom)
- `pages/Overview.jsx` — live metrics grid + model queue
- `pages/Training.jsx` — live SSE stream chart (Chart.js) + log box
- `pages/GPU.jsx` — temperature ring + stats
- `pages/Benchmark.jsx` — interactive comparison table + charts
- `pages/Files.jsx` — file tree browser
- `pages/Logs.jsx` — scrollable full log

Design: Use the dark theme from the preview widget (--c0 background, --accent cyan, monospace font for values). Keep it clean, no excessive animation.

### 3b — Install script

Create `dashboard/setup_dashboard.sh`:
```bash
#!/bin/bash
set -e
cd dashboard/frontend
npm ci
npm run build
echo "Frontend built. Starting server..."
cd ../..
python dashboard/app.py
```

### 3c — Reverse proxy (optional, for single port)

Add to `app.py`: serve the React build from `/` and proxy `/api/` routes. Already done in the template — just wire it up after `npm run build` produces `frontend/build/`.

---

## TASK 4 — OPTUNA HPO

### hpo/optuna_study.py

```python
"""
Hyperparameter optimization for MR-TAE-FULL.
Uses Optuna with Hyperband pruner for efficiency.
Stores study in SQLite (resumable after crash).
Logs each trial to MLflow.
Target: maximize test_real_ncc.
"""

import optuna
from optuna.integration import MLflowCallback
from optuna.pruners import HyperbandPruner

def objective(trial):
    config = {
        "swin_num_heads":    trial.suggest_categorical("swin_heads", [4, 8, 16]),
        "swin_window_size":  trial.suggest_categorical("swin_window", [16, 32, 64]),
        "bigru_hidden_dim":  trial.suggest_categorical("bigru_hidden", [128, 256, 512]),
        "encoder_base_ch":   trial.suggest_categorical("encoder_ch", [16, 32, 64]),
        "lr":                trial.suggest_float("lr", 1e-5, 3e-3, log=True),
        "weight_decay":      trial.suggest_float("wd", 1e-6, 1e-3, log=True),
        "dropout":           trial.suggest_float("dropout", 0.0, 0.3),
        "loss_config":       trial.suggest_categorical("loss", ["A", "B", "C", "D"]),
        "charb_eps":         trial.suggest_float("charb_eps", 1e-4, 1e-2, log=True),
    }
    
    # Train Phase 1 only (35 epochs) — pruner kills bad trials early
    model = build_model("MR-TAE-FULL", config)
    trainer = UnifiedTrainer(model, config, trial=trial)  # pass trial for pruning callbacks
    val_ncc = trainer.train_phase(phase=1, epochs=35)
    return val_ncc

study = optuna.create_study(
    study_name="mr-tae-full-hpo",
    direction="maximize",
    storage="sqlite:///hpo/optuna.db",
    load_if_exists=True,            # Resume after crash
    pruner=HyperbandPruner(min_resource=5, max_resource=35, reduction_factor=3),
)
```

After HPO completes:
- Save best params to `config/best_hparams.yaml`
- Print top 5 trials sorted by val NCC
- Generate `results/hpo/param_importance.png` using `optuna.visualization`

---

## TASK 5 — SHAP + LIME EXPLAINABILITY

### evaluation/explainability/shap_analysis.py

Run ONLY on the best model (highest NCC on real Q.Lin test set from benchmark results).

```python
"""
SHAP analysis: which time steps in a 2048-sample window contribute most
to the model's denoising decision?

Uses DeepExplainer on the encoder (the feature extractor part).
Background: 50 clean noise-only windows (no PD pulses).
Test: 200 windows from real Q.Lin data.
"""

import shap
import torch

def run_shap_analysis(model, test_loader, n_background=50, n_test=200):
    # Use only the encoder for explanation
    encoder_wrapper = EncoderWrapper(model)
    
    background = next(iter(noise_only_loader))[0][:n_background]
    test_samples = next(iter(test_loader))[0][:n_test]
    
    explainer = shap.DeepExplainer(encoder_wrapper, background)
    shap_values = explainer.shap_values(test_samples)
    
    # Plot 1: Summary — distribution of SHAP values across time steps
    # Plot 2: Individual example — overlay SHAP on top of raw signal
    # Plot 3: Mean absolute SHAP per time window (shows which region matters most)
```

Save: `results/explainability/shap_summary.png`, `results/explainability/shap_examples.png`

### evaluation/explainability/lime_analysis.py

```python
"""
LIME: segment the 2048-sample signal into 32 super-pixels.
Perturb each segment, observe NCC change.
Identifies which input regions the model relies on for reconstruction.
"""
```

---

## TASK 6 — WGAN NOISE AUGMENTATION

### data/wgan/extract_noise_segments.py

```python
"""
Extracts inter-pulse "blank gap" segments from Q.Lin data.
These are pure real-world AE noise with no PD events.
Used as training data for WGAN-noise.

Method:
1. Load Q.Lin windows
2. Compute rolling RMS with 50-sample window
3. Mark segments where RMS < threshold as "noise only"
4. Extract 2048-sample segments from these regions
5. Save to data/real_noise_segments.pt
"""
```

### data/wgan/wgan_noise.py

Architecture: DCGAN-style 1D generator/discriminator for 2048-sample noise windows.

Anti-overfitting measures (mandatory):
- Gradient penalty (WGAN-GP): `λ=10`
- Spectral normalization on ALL discriminator layers
- Feature matching loss (L2 on intermediate D features)
- Early stopping: compute MMD on held-out noise set every 50 epochs; stop when MMD plateau detected (< 1% change over 3 checks)
- Training data: only the real noise segments extracted above (NOT full Q.Lin windows)

---

## TASK 7 — AUTO-RETRAIN (NOISE2VOID)

### pipeline/auto_retrain.py

```python
"""
Self-supervised fine-tuning on unlabelled real AE data.
Uses Noise2Void masking strategy.

Trigger conditions:
1. New .mat files placed in data/incoming/
2. Rolling NCC on canary set drops > 0.05 from deployment baseline
3. Weekly scheduled run (check mtime of data/incoming/)

N2V masking:
- Randomly mask 10% of samples per window
- Replace masked values with random nearby values (structurally different, so model can't shortcut)
- Train to predict masked values from unmasked context
- Loss: MSE on masked positions only

Safety gate:
- New model must achieve NCC >= baseline - 0.01 on test_real_qlin.pt
- If not: reject weights, log failure, keep old model
- Alert logged to dashboard and MLflow
"""
```

---

## TASK 8 — FINAL DOCUMENTATION

Generate these files AFTER all training and benchmarking is complete:

### docs/ARCHITECTURE_SUMMARY.md
For each model in the registry, include:
- ASCII architecture diagram (encoder → bottleneck → decoder, annotated with dims)
- Parameter count
- What makes it different from MR-TAE-FULL
- Best use case (speed vs. accuracy trade-off)

### docs/EXAMINER_QA.md
Pre-written answers to 15 examiner questions covering:
- AE physics (propagation, impedance mismatch, frequency band)
- Field sensor installation (couplant, waveguide, magnetic holddown)
- AE vs. HFCT/UHF comparison table
- WGAN overfitting mitigation (explain the noise-segment extraction strategy)
- Source location ambiguity (how the model handles/doesn't handle it)
- Mechanical vs. PD noise discrimination (model's learned discrimination)
- The "constant deviation" problem with wavelet thresholding (key thesis contribution)
- Why NCC is better than SNR as a diagnostic metric

### docs/FINAL_VERDICT.md
Written after HPO + all benchmarks. Must answer:
1. Which model is most robust (NCC-Real)?
2. Which model is most efficient (NCC/latency ratio)?
3. Which ablation component contributes most?
4. Does WGAN augmentation demonstrably improve NCC-Real?
5. Does Loss Config B (Dice) or C (SSIM) outperform Config A (Focal)?
6. Recommendation for future deployment on edge hardware

---

## RTX 4070 CONSTRAINTS REMINDER (active during all tasks)
- AMP enabled for all model inference too (not just training)
- Benchmark inference: batch=1 AND batch=32, measure latency with `torch.cuda.synchronize()`
- SHAP: run in eval mode + `torch.no_grad()`, batch=32 max
- Always call `torch.cuda.empty_cache()` between model evaluations in benchmark runner

---

## SUBAGENT ALLOCATION

- **Subagent A** (IMMEDIATE): Task 0 — build test sets right now
- **Subagent B**: Tasks 1 + 2 — Q.Lin loader, validation figure, benchmark runner
- **Subagent C**: Task 3 — dashboard React frontend
- **Subagent D**: Task 4 + 5 — Optuna HPO + SHAP/LIME
- **Subagent E**: Tasks 6 + 7 + 8 — WGAN, auto-retrain, documentation

**A must complete before B. B must complete before D. C is independent.**

---

*Previous prompts: AGENT_PROMPT.md (architecture spec) | AGENT_PROMPT_v2.md (signal corrections) | FULL_RESEARCH_PLAN.md (MLOps plan)*
