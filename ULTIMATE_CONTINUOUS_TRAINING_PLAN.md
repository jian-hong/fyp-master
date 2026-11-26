# ULTIMATE CONTINUOUS TRAINING PLAN
## AE-PD Denoising Research — Multi-Month Automated Pipeline
### For: Oo Jian Hong, University of Malaya, 2026
### Target: RTX 4070 · .venv-isolated · Crash-safe · Resume-first · Runs months unattended

---

## PART 0 — WHY IT STOPS AFTER EACH MODEL (Root Cause)

The pipeline halts between models for **three distinct reasons**:

### Reason 1 — MLflow Ghost Run (the one described in your session)
`Trainer.__init__` calls `mlflow.start_run()`. When the previous run was never
explicitly ended (e.g., due to an exception that bypassed the `finally:` block),
MLflow enters an inconsistent state. **Fix:** Wrap every `run_model()` call in:
```python
import mlflow
mlflow.end_run()          # Always kill ghost runs first
try:
    history = run_model(...)
finally:
    mlflow.end_run()      # Always close cleanly after
```

### Reason 2 — The Orchestrator Has No Infinite Loop
`mlflow_orchestrator.py --resume` iterates once through the model queue and then
**exits the Python process**. It is designed as a single-pass sweep, not a daemon.
Once all 13 models are marked `done` in `session_state.json`, it returns `0` and
PowerShell considers the job finished.

### Reason 3 — No Phase-Chaining Logic
The orchestrator only calls `run_model()`. It has no code to detect "all supervised
models done → begin WGAN phase → begin N2V finetune → loop again." That chaining
must be built explicitly (see Part 2 below).

---

## PART 1 — IMMEDIATE VENV ENFORCEMENT FIX

Every script must self-enforce `.venv` isolation. Add this shim to the top of
`pipeline/mlflow_orchestrator.py`, `pipeline/run_continuous_dag.py`, and any new
entrypoint:

```python
# ── .venv self-enforcement ──────────────────────────────────────────────────
import sys, os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_VENV_PYTHON  = _PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"   # Windows
if _VENV_PYTHON.exists() and Path(sys.executable).resolve() != _VENV_PYTHON.resolve():
    import subprocess
    result = subprocess.run([str(_VENV_PYTHON)] + sys.argv, check=False)
    sys.exit(result.returncode)
# ────────────────────────────────────────────────────────────────────────────
```

This makes ANY `python pipeline/mlflow_orchestrator.py` call re-launch itself
inside `.venv` automatically, even if you forget to activate it first.

---

## PART 2 — THE INFINITE CONTINUOUS DAG LOOP

### Architecture Overview

```
pipeline/
├── run_dag.py                 ← NEW: single eternal entrypoint
├── dag_phases/
│   ├── phase_supervised.py    ← runs mlflow_orchestrator queue
│   ├── phase_hpo.py           ← Optuna 30-trial sweep
│   ├── phase_wgan.py          ← WGAN-GP noise/pulse training
│   ├── phase_n2v.py           ← Noise2Void self-supervised finetune
│   ├── phase_rl_finetune.py   ← RL reward-shaped refinement (NEW)
│   └── phase_benchmark.py     ← full evaluation sweep
└── dag_state.json             ← crash-safe DAG state (like session_state.json)
```

### `pipeline/run_dag.py` — The Eternal Loop

```python
"""
run_dag.py — Crash-safe infinite DAG executor.

Usage:
    python pipeline/run_dag.py                  # start/resume
    python pipeline/run_dag.py --once           # run one full cycle then exit
    python pipeline/run_dag.py --phase wgan     # jump to specific phase
    python pipeline/run_dag.py --cooldown 300   # GPU cooldown seconds between phases

Phases run in order:
    1. supervised   → train all model registry entries
    2. hpo          → Optuna search on best model
    3. wgan         → WGAN-GP noise + pulse augmentation
    4. n2v          → Noise2Void semi-supervised finetune
    5. rl_finetune  → RL-shaped reward refinement
    6. benchmark    → full evaluation on frozen test sets
    7. [loop back to 1 with augmented data]
"""

import json, time, logging, sys, subprocess
from pathlib import Path
from datetime import datetime

# ── .venv self-enforcement ───────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
_VENV = _ROOT / ".venv" / "Scripts" / "python.exe"
if _VENV.exists() and Path(sys.executable).resolve() != _VENV.resolve():
    sys.exit(subprocess.run([str(_VENV)] + sys.argv, check=False).returncode)
# ─────────────────────────────────────────────────────────────────────────────

PHASES = [
    "supervised",
    "hpo",
    "wgan",
    "n2v",
    "rl_finetune",
    "benchmark",
]

DAG_STATE_FILE = _ROOT / "dag_state.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [DAG]  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_ROOT / "logs" / "dag.log", mode="a"),
    ],
)
log = logging.getLogger("dag")


def load_dag_state() -> dict:
    if DAG_STATE_FILE.exists():
        with open(DAG_STATE_FILE) as f:
            return json.load(f)
    return {"cycle": 0, "current_phase": "supervised",
            "completed_phases": [], "last_updated": ""}


def save_dag_state(state: dict):
    state["last_updated"] = datetime.now().isoformat()
    with open(DAG_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def run_phase(phase: str, extra_args: list = None) -> bool:
    """Run a phase script. Returns True if successful."""
    script = _ROOT / "pipeline" / "dag_phases" / f"phase_{phase}.py"
    cmd = [str(_VENV), str(script)] + (extra_args or [])
    log.info("Starting phase: %s", phase)
    log.info("CMD: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(_ROOT))
    success = result.returncode == 0
    log.info("Phase %s → %s", phase, "OK" if success else f"FAILED (rc={result.returncode})")
    return success


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--phase", type=str, default=None)
    parser.add_argument("--cooldown", type=int, default=60)
    args = parser.parse_args()

    state = load_dag_state()
    cycle = state.get("cycle", 0)
    log.info("DAG Executor starting. Cycle=%d  Resume phase=%s",
             cycle, state["current_phase"])

    while True:
        for phase in PHASES:
            if args.phase and phase != args.phase:
                continue
            if phase in state.get("completed_phases", []) and not args.once:
                log.info("Phase '%s' already done this cycle. Skipping.", phase)
                continue

            state["current_phase"] = phase
            save_dag_state(state)

            success = run_phase(phase)

            if success:
                if phase not in state["completed_phases"]:
                    state["completed_phases"].append(phase)
                save_dag_state(state)
                log.info("GPU cooldown: %d seconds...", args.cooldown)
                time.sleep(args.cooldown)
            else:
                log.warning("Phase %s failed. Will retry next wake.", phase)
                time.sleep(120)   # short retry pause

        if args.once or args.phase:
            break

        # Full cycle complete — reset for next cycle
        cycle += 1
        state["cycle"] = cycle
        state["completed_phases"] = []
        state["current_phase"] = "supervised"
        save_dag_state(state)
        log.info("=" * 60)
        log.info("CYCLE %d COMPLETE. Starting next cycle.", cycle)
        log.info("=" * 60)
        time.sleep(args.cooldown * 5)    # longer rest between full cycles


if __name__ == "__main__":
    main()
```

**Launch command (all platforms, forever, in .venv):**
```bash
python pipeline/run_dag.py
```

**Resume after crash:**
```bash
python pipeline/run_dag.py          # reads dag_state.json, continues from last phase
```

**Skip to a specific phase:**
```bash
python pipeline/run_dag.py --phase wgan
```

---

## PART 3 — TRAINING METHOD COMPARISON MATRIX

This is the core scientific contribution. The agent must run ALL combinations below
and record NCC, SNR_improvement, Dice (segmentation), Latency_ms on BOTH test sets.

### 3A — Curricular Schedule Types

| Schedule ID | Phase 1 (easy) | Phase 2 (medium) | Phase 3 (hard) | Phase 4 (ultra-hard) |
|---|---|---|---|---|
| `CURR-STD` | [5,15] dB, 35ep | [-5,5] dB, 35ep | [-20,-5] dB, 30ep | — |
| `CURR-STEEP` | [5,15] dB, 20ep | [-5,5] dB, 25ep | [-20,-5] dB, 35ep | [-30,-20] dB, 20ep |
| `CURR-FLAT` | [-10,15] dB, 100ep | — | — | — |
| `CURR-REVERSE` | [-20,-5] dB, 35ep | [-5,5] dB, 35ep | [5,15] dB, 30ep | — (control experiment) |
| `NOCURR` | Full range [-20,15] dB from epoch 1, 100ep | — | — | — |

**Expected finding:** `CURR-STEEP` should outperform `NOCURR` on hard-SNR test set.
`CURR-REVERSE` is the negative control — should perform worst, confirming curricular
ordering matters.

### 3B — SNR Range Experiments

| SNR Config ID | Training SNR Range | Test SNR Points | Purpose |
|---|---|---|---|
| `SNR-STANDARD` | [-20,15] dB via curriculum | -15,-10,-5,0,5 dB | Current baseline |
| `SNR-EXTENDED` | [-30,15] dB | -25,-20,-15,-10,-5,0,5 dB | Stress test |
| `SNR-LOWONLY` | [-30,-10] dB | -25,-20,-15,-10 dB | AE field conditions |
| `SNR-HIGHONLY` | [-5,15] dB | -5,0,5,10,15 dB | Lab conditions |
| `SNR-ADVERSARIAL` | Step-by-step hard-mining | Hardest 10% of val set | Curriculum on-demand |

### 3C — Noise Type Comparison

| Noise Config ID | Noise Mixture | Ratio |
|---|---|---|
| `NOISE-AWGN` | Additive White Gaussian only | 100% AWGN |
| `NOISE-MIX-STD` | AWGN + NBI + Impulse | 50/25/25 |
| `NOISE-MIX-HARD` | AWGN + NBI + Impulse + Powerline 50Hz | 40/20/20/20 |
| `NOISE-WGAN` | AWGN + WGAN-generated real substation noise | 50/50 |
| `NOISE-MECH` | AWGN + Magnetostriction (100/120 Hz harmonics) | 50/50 |
| `NOISE-ACOUSTIC` | All above combined | 25/20/20/20/15 |

`NOISE-ACOUSTIC` is the most realistic. If MR-TAE still generalizes here, it is
directly deployable in real substation field conditions.

### 3D — Frequency Band Experiments

AE signals span 20 kHz – 500 kHz. These frequency-band configs probe robustness:

| Band Config ID | Sampling Rate | Signal Bandwidth | Target |
|---|---|---|---|
| `BAND-FULL` | 5 MHz (Nyquist for 2.5 MHz) | 20 kHz – 500 kHz | Current config |
| `BAND-LOW` | 1 MHz | 20 kHz – 100 kHz | Attenuation-heavy environments |
| `BAND-HIGH` | 5 MHz | 100 kHz – 500 kHz | High-frequency corona |
| `BAND-ULTRA` | 10 MHz | 500 kHz – 2 MHz | HFCT-range crossover |
| `BAND-AE-REAL` | Per Q.Lin spec | Match Q.Lin test data | Direct real-data alignment |

### 3E — Signal Length / Context Window

| Length Config | Samples | Duration @5MHz | Memory (batch=128) |
|---|---|---|---|
| `LEN-512` | 512 | 102 µs | ~50 MB |
| `LEN-1024` | 1024 | 205 µs | ~100 MB |
| `LEN-2048` | 2048 | 410 µs | ~200 MB (current) |
| `LEN-4096` | 4096 | 819 µs | ~400 MB |
| `LEN-8192` | 8192 | 1.6 ms | ~800 MB |

RTX 4070 12GB can handle up to `LEN-8192` at batch=32 with AMP. Longer context
window should benefit BiGRU and Swin Transformer more than plain CNN.

### 3F — Dataset Scale Experiments

| Dataset Config | Synthetic N_train | Augmentation | Real Q.Lin usage |
|---|---|---|---|
| `DATA-SMALL` | 2,000 | None | 0% |
| `DATA-STD` | 4,000 | None | 0% (current) |
| `DATA-LARGE` | 20,000 | None | 0% |
| `DATA-XL` | 100,000 | None | 0% |
| `DATA-AUG-WGAN` | 20,000 | 50% WGAN noise aug | 10% for WGAN training |
| `DATA-AUG-MIXUP` | 20,000 | Mixup α=0.2 | 0% |
| `DATA-AUG-FULL` | 50,000 | WGAN + Mixup + RandShift | 10% for WGAN |

**Scaling law analysis:** Plot NCC vs log(N_train) for each model. Find the
saturation point — this is publishable as a dataset sizing recommendation.

---

## PART 4 — EXTENDED MODEL ZOO (Big Models)

These are the architectures the agent must build and add to `model_registry.yaml`:

### Tier 1 — Extended Ablations (already designed, need implementation)
```
MR-TAE-FULL-L    = MR-TAE-FULL with 2× hidden dims everywhere (~130M params)
MR-TAE-FULL-XL   = MR-TAE-FULL with 4× hidden dims (~260M params) — fits RTX 4070 @batch=16
```

### Tier 2 — Transformer-Heavy Variants
```
MR-TAE-MAMBA     = Replace BiGRU with Mamba SSM (State Space Model) — 2024 SOTA for sequences
MR-TAE-SWIN2     = Replace Swin-v1 with Swin Transformer v2 (shifted window, log-spaced RPB)
MR-TAE-CROSSATTN = Cross-attention between BiGRU temporal features and Swin spatial features
MR-TAE-RETNET    = Replace Swin with RetNet (retention mechanism, parallelizable during training)
```

### Tier 3 — Diffusion / Generative Denoisers
```
DDPM-1D          = Denoising Diffusion Probabilistic Model adapted to 1D AE signals
ScoreSDE-1D      = Score-based SDE denoiser (Song et al. 2021) for 1D signals
DiffusionMedAE   = DDPM conditioned on SNR estimate (classifier-free guidance)
```

### Tier 4 — SOTA External Baselines (for fair comparison)
```
CleanUNet-1D     = NVIDIA CleanUNet adapted to 1D (already cloned, need integration)
DeepFilterNet-1D = DeepFilterNet frequency-domain adaptation
WaveUNet-1D      = WaveNet-style U-Net for 1D signals
```

### Model Registry YAML Format
```yaml
# config/model_registry.yaml — add these entries:
MR-TAE-FULL-L:
  use_bigru: true
  use_swin: true
  use_wavelet_pooling: true
  use_attention_gates: true
  use_mtl: true
  hidden_dim_multiplier: 2.0          # ← new flag
  n_params_estimate_M: 130

MR-TAE-MAMBA:
  use_bigru: false
  use_swin: false
  use_mamba: true                     # ← new flag
  use_wavelet_pooling: true
  use_attention_gates: true
  use_mtl: true
  n_params_estimate_M: 45

DDPM-1D:
  model_class: "DiffusionDenoiser1D"  # ← separate class loader
  diffusion_steps: 1000
  schedule: "cosine"
  conditioning: "snr_estimate"
  n_params_estimate_M: 55
```

---

## PART 5 — REINFORCEMENT LEARNING INTEGRATION

### Why RL for Signal Denoising?

Standard supervised denoising minimizes a per-sample loss (Charbonnier, MSE, etc.)
which optimizes the **average** quality. RL allows optimization for **downstream
task performance** — in your case: "did the denoised signal allow correct PD
type classification?" This is a non-differentiable objective that RL can reach but
backprop cannot.

### Recommended RL Architecture: PPO-RLHF Finetuning

This is the same paradigm as InstructGPT / RLHF for LLMs, applied to denoising:

```
Step 1 (Supervised Pre-train):   Train MR-TAE-FULL via curricular supervision → best.pt
Step 2 (Reward Model):           Train a frozen PD classifier on clean signals → reward_model.pt
Step 3 (RL Finetune):            Use PPO to finetune MR-TAE denoiser where reward =
                                 classifier_confidence(denoised_signal, true_label)
```

**Why PPO over SAC for this task:**
- PPO is on-policy → stable, well-understood convergence
- SAC is off-policy → better sample efficiency but requires replay buffer of full signals
  (expensive in VRAM for 2048-sample windows at batch=128)
- For 1D signal denoising, PPO is the standard choice per ICLR 2025 literature

### RL Reward Function Design

```python
def compute_reward(denoised_signal, clean_signal, pd_label, classifier, snr_in):
    """Multi-objective reward for PD signal denoiser PPO finetuning."""

    # R1: Classification accuracy reward (main RL signal)
    with torch.no_grad():
        logits = classifier(denoised_signal)
        probs  = torch.softmax(logits, dim=-1)
        class_reward = probs[:, pd_label].mean()   # confidence in correct class

    # R2: Reconstruction quality (keep model from collapsing)
    ncc = normalized_cross_correlation(denoised_signal, clean_signal)
    recon_reward = ncc.clamp(0, 1)

    # R3: SNR improvement bonus
    snr_out   = compute_snr(denoised_signal, clean_signal)
    snr_bonus = torch.sigmoid(snr_out - snr_in) * 0.2   # bonus for improvement

    # R4: Perceptual penalty (don't introduce artifacts)
    spectral_loss = spectral_l1(denoised_signal, clean_signal)
    artifact_penalty = -spectral_loss.clamp(0) * 0.3

    # Composite reward
    reward = (
        0.5 * class_reward
      + 0.3 * recon_reward
      + 0.1 * snr_bonus
      + 0.1 * (1 + artifact_penalty)
    )
    return reward.detach()
```

### `pipeline/dag_phases/phase_rl_finetune.py` Skeleton

```python
"""
Phase 5: PPO RL finetuning of best supervised model.

Requires:
  - results/{best_model}/best.pt           (supervised checkpoint)
  - results/reward_model/classifier.pt     (frozen PD classifier — train once)

Outputs:
  - results/{best_model}-RL/best_rl.pt
  - mlflow run: {best_model}-RL-PPOv1
"""
import torch, mlflow
from pathlib import Path
from training.ppo_trainer import PPOTrainer
from evaluation.reward_model import load_reward_classifier
from mr_tae_fusion.models.ablation_config import AblationConfig

def main():
    mlflow.end_run()
    project_root = Path(__file__).resolve().parents[2]

    # Load best supervised checkpoint
    best_ckpt = project_root / "results" / "MR-TAE-FULL" / "best.pt"
    assert best_ckpt.exists(), "Run supervised phase first"

    # Load frozen classifier as reward model
    reward_classifier = load_reward_classifier(
        project_root / "results" / "reward_model" / "classifier.pt"
    )
    reward_classifier.eval().requires_grad_(False)

    # PPO finetune
    trainer = PPOTrainer(
        model_checkpoint=str(best_ckpt),
        reward_fn=reward_classifier,
        n_epochs=20,
        ppo_clip=0.2,
        lr=1e-5,           # much lower than supervised lr
        kl_coeff=0.1,      # KL divergence penalty to prevent forgetting
        save_dir=str(project_root / "results" / "MR-TAE-FULL-RL"),
    )
    trainer.train()

if __name__ == "__main__":
    main()
```

### Alternative RL Methods Worth Researching

| Method | Best For | Complexity |
|---|---|---|
| **PPO** (recommended) | Stable finetuning, well-understood | Medium |
| **GRPO** (DeepSeek-style) | Group-relative rewards, no value network needed | Low |
| **SAC** | Off-policy, sample efficient, handles continuous action spaces | High |
| **DPO** (Direct Preference Opt.) | If you have pairwise signal quality rankings | Low |
| **Online DPO** | Iterative preference optimization without reward model | Medium |
| **REINFORCE++** | Simplified PPO variant, less memory overhead | Low |

**For thesis purposes, implement PPO first.** If time permits, compare PPO vs GRPO
(GRPO is much simpler to implement: no critic network needed, just compare rewards
within a batch and normalize).

---

## PART 6 — MULTI-MONTH SCHEDULED TRAINING PLAN

### Month 1 — Foundation (Weeks 1–4)

```
Week 1:  Implement .venv enforcement + run_dag.py + dag_state.json
         Build frozen test sets (data/test_synthetic_ae.pt, data/test_real_qlin.pt)
         Run supervised phase: all 13 current models (CURR-STD, NOISE-MIX-STD, DATA-STD)
         Expected GPU hours: ~18 hours

Week 2:  Run HPO phase (30 Optuna trials on MR-TAE-FULL)
         Build reward_model/classifier.pt (frozen PD 4-class classifier)
         Implement Phase RL skeleton + PPO trainer
         Expected GPU hours: ~8 hours (with Hyperband pruning)

Week 3:  Run extended model zoo: MR-TAE-FULL-L, MWCNN-BiGRU, UNet-BiGRU-Swin
         Run DATA-SCALE experiments: DATA-SMALL → DATA-XL (scaling law)
         Expected GPU hours: ~24 hours

Week 4:  WGAN-GP training on Q.Lin blank-gap noise segments
         Run NOISE comparison: NOISE-AWGN vs NOISE-MIX-STD vs NOISE-WGAN
         Benchmark everything on frozen test sets
         Expected GPU hours: ~20 hours
```

### Month 2 — Depth (Weeks 5–8)

```
Week 5:  Curricular schedule comparison: CURR-STD, CURR-STEEP, CURR-FLAT, NOCURR
         SNR range experiments: SNR-STANDARD vs SNR-EXTENDED vs SNR-LOWONLY
         Expected GPU hours: ~30 hours (4 curricula × ~8 hours per model)

Week 6:  Frequency band experiments: BAND-FULL vs BAND-LOW vs BAND-HIGH
         Signal length experiments: LEN-2048 vs LEN-4096 vs LEN-8192
         Expected GPU hours: ~20 hours

Week 7:  PPO RL finetuning on best model from Month 1
         Compare RL-finetuned vs supervised: NCC, Classification F1, artifact score
         Implement GRPO variant for comparison
         Expected GPU hours: ~15 hours

Week 8:  Full benchmark run across ALL experiments
         Generate thesis figures via scripts/generate_thesis_figures.py
         SHAP + LIME explainability on top-3 models
         Expected GPU hours: ~5 hours
```

### Month 3 — Advanced Research (Weeks 9–12)

```
Week 9:  Implement MR-TAE-MAMBA (replace BiGRU with Mamba SSM)
         Implement DDPM-1D diffusion denoiser
         Expected GPU hours: ~30 hours

Week 10: Self-supervised Noise2Void finetuning on unlabelled real segments
         Implement online drift detection (rolling NCC canary set)
         Expected GPU hours: ~10 hours

Week 11: DVC data versioning, Docker containerisation, W&B sweep visualization
         Final documentation pass: ARCHITECTURE_SUMMARY, FINAL_VERDICT, EXAMINER_QA
         Expected GPU hours: ~5 hours (mostly CPU/documentation)

Week 12: Full thesis figure regeneration at 300 DPI
         Viva preparation: run live demo, verify all SHA256 test set hashes
         Buffer for re-runs if any model needs more epochs
         Expected GPU hours: ~10 hours
```

### Monthly GPU Hour Budget Summary

| Month | GPU Hours | Key Output |
|---|---|---|
| Month 1 | ~70 hrs | All 13 ablations trained, HPO done, WGAN trained |
| Month 2 | ~70 hrs | Full curriculum/SNR/frequency/RL comparison matrix |
| Month 3 | ~55 hrs | Mamba, Diffusion, N2V, Docker, final thesis figures |
| **Total** | **~195 hrs** | **Complete research system** |

RTX 4070 running 16h/day = ~480 GPU hours/month. Plenty of headroom.

---

## PART 7 — COMPLETE AI CODING AGENT PROMPT

Copy this entire block and paste it directly to Claude Code, Cursor Agent, or
any agentic coding system. It is self-contained.

---

```
=================================================================================
AI CODING AGENT PROMPT — AE-PD DENOISING RESEARCH PIPELINE
University of Malaya FYP · RTX 4070 · Python 3.11 · PyTorch 2.x · Windows 11
=================================================================================

OVERVIEW
--------
You are implementing a production-grade, crash-safe, multi-month automated ML
research pipeline for acoustic emission (AE) partial discharge (PD) denoising.
The project already has 13 trained model ablations (MR-TAE variants) tracked in
MLflow. Your job is to extend it into a continuous, self-resuming, month-scale
automated research system.

CONSTRAINTS (non-negotiable)
-----------------------------
1. ALL Python execution MUST happen inside .venv (/.venv/Scripts/python.exe).
   Every new entrypoint script must include the .venv self-enforcement shim at top.
2. ALL training/evaluation scripts must be resumable. Use session_state.json /
   dag_state.json patterns already established in the repo.
3. Never duplicate model code. Use AblationConfig flags and the model registry.
4. All random seeds fixed at 42. DataLoaders created with seed=42, generator=torch.Generator().manual_seed(42).
5. NEVER touch data/test_synthetic_ae.pt or data/test_real_qlin.pt during training.
   These are the frozen evaluation test sets.
6. MLflow.end_run() must be called before AND after every mlflow.start_run() context.
7. All new model checkpoints saved to results/{model_id}/best.pt and epoch_{N}.pt.
8. All logs written to logs/{phase}_{timestamp}.log.

TASK 1 — .VENV ENFORCEMENT + INFINITE DAG LOOP
------------------------------------------------
Files to create/modify:
  pipeline/run_dag.py                        ← NEW eternal DAG entrypoint
  pipeline/dag_phases/phase_supervised.py    ← wraps mlflow_orchestrator
  pipeline/dag_phases/phase_hpo.py           ← runs hpo/optuna_study.py
  pipeline/dag_phases/phase_wgan.py          ← trains WGAN-GP
  pipeline/dag_phases/phase_n2v.py           ← Noise2Void finetune
  pipeline/dag_phases/phase_rl_finetune.py   ← PPO RL finetuning
  pipeline/dag_phases/phase_benchmark.py     ← full evaluation
  dag_state.json                             ← crash-safe DAG state

run_dag.py behavior:
  - On first run: starts from phase "supervised"
  - On re-run after crash: reads dag_state.json and resumes from last incomplete phase
  - On completion of all 6 phases: increments cycle counter and loops back to "supervised"
  - --once flag: runs one full cycle and exits
  - --phase {name} flag: jumps to a specific phase
  - --cooldown N flag: GPU cooldown seconds between phases (default 60)
  - Logs every phase start/end with timestamp to logs/dag.log

Each phase script:
  - Must be a standalone Python script runnable as: python pipeline/dag_phases/phase_X.py
  - Must call mlflow.end_run() as first line
  - Must write its completion status to dag_state.json on exit (success or failure)

TASK 2 — EXTENDED MODEL REGISTRY
----------------------------------
Add these entries to config/model_registry.yaml:

Tier 1 (scale-up ablations):
  MR-TAE-FULL-L    → MR-TAE-FULL with hidden_dim_multiplier=2.0
  MR-TAE-FULL-XL   → MR-TAE-FULL with hidden_dim_multiplier=4.0, train_batch=16

Tier 2 (transformer variants):
  MR-TAE-MAMBA     → Replace BiGRU with Mamba SSM. Use mamba-ssm library if available,
                     else implement a pure-PyTorch selective SSM approximation.
  MR-TAE-SWIN2     → Replace Swin v1 with Swin Transformer v2 (log-spaced RPB).
  MR-TAE-CROSSATTN → Cross-attention between BiGRU hidden states and Swin tokens.

Tier 3 (diffusion):
  DDPM-1D          → 1D DDPM. T=1000 steps, cosine schedule, UNet backbone.
                     Inference: DDIM sampler for fast 50-step evaluation.

Tier 4 (external baselines):
  CleanUNet-1D     → Already in Try Model/baselines/external/cleanunet_1d.py
  WaveUNet-1D      → WaveNet-style U-Net for 1D signals, 8 levels

In AblationConfig, add:
  mamba: bool = False
  hidden_dim_multiplier: float = 1.0
  diffusion: bool = False
  diffusion_steps: int = 1000
  cross_attn: bool = False

In build_model(), dispatch to the correct class based on these new flags.

TASK 3 — TRAINING COMPARISON MATRIX
-------------------------------------
Create: pipeline/run_comparison_matrix.py

This script runs the following experiment matrix (all via mlflow_orchestrator internals,
all on MR-TAE-FULL architecture):

Dimension 1 — Curricular Schedule:
  CURR-STD:     phase1=[5,15]dB 35ep, phase2=[-5,5]dB 35ep, phase3=[-20,-5]dB 30ep
  CURR-STEEP:   phase1=[5,15]dB 20ep, phase2=[-5,5]dB 25ep, phase3=[-20,-5]dB 35ep,
                phase4=[-30,-20]dB 20ep
  CURR-FLAT:    full range [-10,15]dB from epoch 1, 100ep
  CURR-REVERSE: phase1=[-20,-5]dB 35ep, phase2=[-5,5]dB 35ep, phase3=[5,15]dB 30ep
  NOCURR:       full range [-20,15]dB, 100ep

Dimension 2 — SNR Range:
  SNR-STANDARD: [-20,15] dB training
  SNR-EXTENDED: [-30,15] dB training
  SNR-LOWONLY:  [-30,-10] dB training (field conditions)
  SNR-ADVERSARIAL: hard-negative mining: always sample hardest 20% of val-loss examples

Dimension 3 — Noise Type Mixture:
  NOISE-AWGN:     100% AWGN
  NOISE-MIX-STD:  AWGN 50% + NBI 25% + Impulse 25%
  NOISE-MIX-HARD: AWGN 40% + NBI 20% + Impulse 20% + Powerline 20%
  NOISE-WGAN:     AWGN 50% + WGAN-generated real noise 50%

Dimension 4 — Dataset Scale:
  DATA-STD:   4,000 training samples
  DATA-LARGE: 20,000 training samples
  DATA-XL:    100,000 training samples

Dimension 5 — Signal Length:
  LEN-2048: 2048 samples (current)
  LEN-4096: 4096 samples
  LEN-8192: 8192 samples

The matrix script:
  - Reads dag_state.json to know which cells are done
  - Spawns each run as: python pipeline/mlflow_orchestrator.py --model MR-TAE-FULL
    with a config override .yaml generated per cell
  - Tags each MLflow run with the cell ID (e.g., CURR-STEEP/SNR-EXTENDED/NOISE-WGAN)
  - Saves matrix results to results/comparison_matrix/matrix_results.csv
  - Can be interrupted and resumed at any time

TASK 4 — PPO RL FINETUNING MODULE
------------------------------------
Files to create:
  training/ppo_trainer.py
  evaluation/reward_model.py
  pipeline/dag_phases/phase_rl_finetune.py

PPO Trainer requirements:
  - Load MR-TAE-FULL best.pt as the actor (policy network)
  - Value network: a 2-layer MLP on top of encoder features
  - Actor-Critic update: standard PPO clip loss (ε=0.2) + value loss + entropy bonus
  - KL divergence penalty against the supervised reference model (prevent catastrophic forgetting)
  - Reward = 0.5*classifier_confidence + 0.3*NCC + 0.1*SNR_improvement + 0.1*(1-spectral_L1)
  - Reward normalization: running mean/std with momentum=0.99
  - 20 epochs of PPO, lr=1e-5, batch=64
  - Save checkpoint every epoch to results/MR-TAE-FULL-RL/epoch_{N}.pt
  - Log all reward components separately to MLflow

Reward Model (evaluation/reward_model.py):
  - A 4-class PD classifier (Corona/Internal/Surface/Non-PD)
  - Architecture: 3-layer 1D ResNet, 64 channels
  - Train it ONCE on clean PD signals (denoised output of best supervised model)
  - Freeze it before PPO starts
  - Must be serialized separately to results/reward_model/classifier.pt

TASK 5 — NOISE2VOID SELF-SUPERVISED FINETUNING
------------------------------------------------
File: pipeline/dag_phases/phase_n2v.py

N2V adapted to 1D AE signals:
  - Input: unlabelled real Q.Lin signal segments (from test set blank-gap regions
    that do NOT contain PD events — safe to use for finetune without label leakage)
  - Method: mask 2% of samples per window, train to predict masked values from context
  - Use the MR-TAE encoder as the backbone, freeze decoder for first 5 epochs
  - Loss: MSE on masked positions only
  - 30 epochs, lr=5e-5
  - Save to results/MR-TAE-FULL-N2V/best.pt

TASK 6 — WGAN-GP DATA AUGMENTATION
-------------------------------------
Files:
  data/wgan/extract_noise_segments.py
  data/wgan/wgan_noise.py
  data/wgan/validate_wgan.py

WGAN-GP architecture for 1D noise signals:
  Generator:
    Input: z ~ N(0,1) of dim 128
    Architecture: 5× (Linear + Upsample + Conv1d + LeakyReLU)
    Output: 1D signal of length 2048
    Final activation: Tanh, then scale to match real noise amplitude

  Critic:
    Input: 1D signal of length 2048
    Architecture: 5× (Conv1d stride=2 + LeakyReLU)
    Output: scalar score (no sigmoid — Wasserstein)
    Spectral normalization on all Conv layers

  Training:
    Gradient penalty λ=10 (WGAN-GP standard)
    Critic steps per generator step: 5
    Optimizer: Adam lr=1e-4, β1=0.5, β2=0.9
    Early stopping: MMD on held-out noise set, patience=50 epochs
    Stop if MMD plateaus or starts increasing (overfitting signal)
    Save every 50 epochs to data/wgan/checkpoints/

Validation (data/wgan/validate_wgan.py):
  - Plot PSD of real vs generated noise (must match within 3 dB across full band)
  - Compute MMD(real, generated) — must be < 0.01 for usable quality
  - Visual inspection plots saved to results/wgan_validation/

TASK 7 — FULL BENCHMARK RUNNER
--------------------------------
File: evaluation/benchmark_runner.py (extend existing)

For every model in results/, run full evaluation on BOTH frozen test sets.
Report per model:
  Reconstruction: NCC, MSE, RMSE, SNR_in→SNR_out, SNR_improvement_dB
  Segmentation:   Dice coefficient, IoU, F1 per PD class
  Classification: Accuracy, macro-F1, confusion matrix (4 classes)
  Efficiency:     #params_M, inference_latency_ms (batch=1 and batch=32), VRAM_MB
  Frequency:      per-octave-band SNR improvement (4 bands: 20-50k, 50-100k, 100-250k, 250-500k Hz)
  RL delta:       NCC_RL - NCC_supervised (how much RL helped)

Output format:
  results/benchmark/master_table.csv     (all models × all metrics)
  results/benchmark/latex_table.tex      (ready for thesis copy-paste)
  results/benchmark/comparison_bar.png   (300 DPI, grouped bar chart)
  results/benchmark/snr_sweep.png        (NCC vs SNR for top-5 models)
  results/benchmark/efficiency_scatter.png (params vs NCC, bubble=latency)
  results/benchmark/ablation_heatmap.png (NCC across encoder×bottleneck grid)

TASK 8 — RESUME & CRASH SAFETY EVERYWHERE
-------------------------------------------
Every script that runs for > 5 minutes must:
  1. Write a heartbeat to logs/{script_name}_heartbeat.json every 60 seconds
     with {pid, phase, epoch, timestamp}
  2. Save a checkpoint at least every 10 epochs (not just best.pt)
  3. Read checkpoint on restart if epoch_{N}.pt exists in save_dir
  4. Write final status (success/failure/exception_traceback) to
     results/{model_id}/run_status.json on exit

Add a watchdog script: pipeline/watchdog.py
  - Run as background process: python pipeline/watchdog.py
  - Every 5 minutes, checks all heartbeat files
  - If heartbeat is stale > 10 minutes, logs an alert to logs/watchdog_alerts.log
  - On Windows: uses wmi or psutil to verify PID is alive
  - Does NOT auto-restart (to avoid corrupting GPU state) — just alerts

TASK 9 — DOCUMENTATION AUTO-GENERATION
-----------------------------------------
Script: scripts/generate_all_docs.py

Reads results/benchmark/master_table.csv and generates:
  docs/RESULTS_COMPARISON.md   (extended Table 4.1 with all models)
  docs/FINAL_VERDICT.md        (auto-written verdict with best model per criterion)
  docs/ARCHITECTURE_SUMMARY.md (ASCII diagrams + param counts from model introspection)
  RESUME_PROMPT.md             (current dag_state.json status as human-readable summary)

TASK 10 — WINDOWS POWERSHELL LAUNCHER (run_dag.ps1)
-----------------------------------------------------
Replace existing resume.ps1 with a full DAG launcher:

  .\run_dag.ps1                    → start/resume full DAG
  .\run_dag.ps1 -Phase wgan        → jump to wgan phase
  .\run_dag.ps1 -Once              → run one full cycle and exit
  .\run_dag.ps1 -Status            → print dag_state.json summary, don't train

The script must:
  1. Activate .venv
  2. Print GPU name + VRAM available
  3. Print dag_state.json status table (cycle, current phase, completed phases)
  4. Start MLflow UI in background on port 5000 (if not already running)
  5. Launch: python pipeline/run_dag.py [forwarded args]
  6. On Ctrl+C: print "Checkpoint saved. Run .\run_dag.ps1 to resume."

=================================================================================
EXECUTION ORDER FOR AGENT
=================================================================================

Do not start training until Tasks 1-4 are complete and tested.
Then execute in this order:

1. Task 1 (DAG loop + .venv enforcement)      — verify with: python pipeline/run_dag.py --once --phase supervised
2. Task 2 (model registry extensions)         — verify with: python training/trainer.py --smoke-test
3. Task 4 (PPO RL skeleton)                   — verify with: python training/ppo_trainer.py --test
4. Task 6 (WGAN-GP)                           — verify with: python data/wgan/wgan_noise.py --epochs 5 --test
5. Task 3 (comparison matrix runner)          — verify with: python pipeline/run_comparison_matrix.py --dry-run
6. Task 7 (benchmark runner)                  — verify with: python evaluation/benchmark_runner.py --quick-test
7. Task 8 (crash safety + watchdog)           — verify with: python pipeline/watchdog.py --check
8. Task 9 (documentation generator)           — verify with: python scripts/generate_all_docs.py --dry-run
9. Task 10 (PowerShell launcher)              — verify with: .\run_dag.ps1 -Status
10. LAUNCH: .\run_dag.ps1

=================================================================================
WHAT SUCCESS LOOKS LIKE
=================================================================================

After Month 1:
  - results/benchmark/master_table.csv has rows for all 13+ models
  - MLflow UI shows all runs with clean start/end timestamps (no orphan runs)
  - dag_state.json shows cycle=1, completed_phases=[all 6]
  - WGAN PSD validation passes (MMD < 0.01)

After Month 2:
  - results/comparison_matrix/matrix_results.csv filled for all 40+ experiments
  - results/MR-TAE-FULL-RL/best_rl.pt outperforms best.pt on classification F1
  - SNR sweep plot shows MR-TAE-FULL dominates below -10 dB (Swin+BiGRU contribution)

After Month 3:
  - MR-TAE-MAMBA results available (expected to match or exceed BiGRU on long windows)
  - DDPM-1D results available (expected high NCC but slow inference: >100ms/sample)
  - docs/FINAL_VERDICT.md auto-generated with complete empirical conclusion
  - All thesis figures at 300 DPI, ready for submission

=================================================================================
END OF AGENT PROMPT
=================================================================================
```

---

## PART 8 — QUICK-START COMMAND REFERENCE

```bash
# ── Start/resume the infinite DAG (recommended default) ──────────────────────
python pipeline/run_dag.py

# ── PowerShell (Windows) ─────────────────────────────────────────────────────
.\run_dag.ps1

# ── Check status without training ────────────────────────────────────────────
.\run_dag.ps1 -Status

# ── Jump to a specific phase ──────────────────────────────────────────────────
python pipeline/run_dag.py --phase wgan
python pipeline/run_dag.py --phase rl_finetune

# ── Run one full cycle then exit ──────────────────────────────────────────────
python pipeline/run_dag.py --once

# ── Run comparison matrix experiments ────────────────────────────────────────
python pipeline/run_comparison_matrix.py --resume

# ── Manual single-model override ─────────────────────────────────────────────
python pipeline/mlflow_orchestrator.py --model MR-TAE-MAMBA

# ── Full benchmark ────────────────────────────────────────────────────────────
python evaluation/benchmark_runner.py --compare-all

# ── Generate all docs ────────────────────────────────────────────────────────
python scripts/generate_all_docs.py

# ── Watchdog (run in separate terminal) ──────────────────────────────────────
python pipeline/watchdog.py
```

---

*Generated: April 2026 · Researcher: Oo Jian Hong, UM · GPU: RTX 4070 12GB*
