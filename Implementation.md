# IMPLEMENTATION.md — AE-PD Denoising FYP, Full Build Plan

> **Audience:** a coding agent (Claude Code / Cursor) executing tasks autonomously, *and* the human researcher reviewing each checkpoint.
> **Scope:** builds on top of the existing MR-TAE-Fusion codebase. Does **not** discard the MLflow orchestrator, resilience manager, curriculum scheduler, or ablation registry — extends them.
> **Non-goals:** cloud deployment, Kubernetes, production serving. This is a thesis, not a product.

---

## 0. Guiding principles

1. **Reproducibility beats novelty.** Every result must come from a seeded, logged, hash-verified run.
2. **Baselines before brilliance.** A plain 1D U-Net on flat-SNR data with MSE must exist before any claim about curriculum or attention is made.
3. **One change at a time.** Controlled experiments only. If two variables change between two runs, the comparison is worthless.
4. **Fail loudly.** Any silent fallback (e.g. CPU when CUDA fails, MSE when Charbonnier errors) must raise.
5. **Document while coding.** Every module gets a docstring that answers: *what, why, and what's the alternative we rejected.*

---

## 1. Pre-flight: diagnose and fix what's broken (Day 1, ~4 hours)

### 1.1 Fix the resume button (localhost:1001)
**Symptom:** clicking "Resume Training" in the dashboard does nothing; nothing new appears in MLflow UI.

**Root cause (most likely):** `app.py` spawns the orchestrator via `os.system` or a string `subprocess.call` that (a) doesn't use the venv's Python, (b) blocks the Flask request thread, or (c) dies silently on Windows because stdout isn't redirected.

**Fix template** (in `app.py`, whichever endpoint handles the resume button):
```python
import subprocess, sys, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
TRAINING_PROC = None  # module-level handle

@app.route("/api/training/start", methods=["POST"])
def start_training():
    global TRAINING_PROC
    if TRAINING_PROC and TRAINING_PROC.poll() is None:
        return {"status": "already_running", "pid": TRAINING_PROC.pid}, 409

    log_path = PROJECT_ROOT / "logs" / "training_live.log"
    log_path.parent.mkdir(exist_ok=True)
    log_fh = open(log_path, "a", buffering=1)

    # Windows-safe detached subprocess with explicit venv python
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

    TRAINING_PROC = subprocess.Popen(
        [sys.executable, str(PROJECT_ROOT / "pipeline" / "mlflow_orchestrator.py"), "--resume"],
        cwd=str(PROJECT_ROOT),
        stdout=log_fh, stderr=subprocess.STDOUT,
        creationflags=creationflags,
    )
    return {"status": "started", "pid": TRAINING_PROC.pid, "log": str(log_path)}
```

**Verification:**
- `POST /api/training/start` returns a PID.
- `tail -f logs/training_live.log` shows orchestrator output within ~5 seconds.
- MLflow UI at :5000 shows a new run appearing within one epoch.

**Clarifications for you:**
- MLflow on :5000 is a **read-only viewer**. It never drives training. If it looks "stuck," training is the culprit, not MLflow.
- Linux vs Windows is **not** the issue. The same code runs on both if `subprocess` is called correctly. No plugins needed. Do **not** install WSL for this.

### 1.2 Audit output activation and NCC computation
Before any new training, run a 1-minute sanity check:
1. In `mr_tae_fusion/models/mr_tae_fusion.py`, find the final output layer. If it's `nn.Sigmoid` or `nn.ReLU`, that is **wrong** for bipolar signals. Replace with `nn.Identity()` (residual prediction) or `nn.Tanh()` (if inputs are normalised to [−1, 1]).
2. In `evaluation/metrics.py`, confirm NCC is computed as
   `sum(x*y) / (sqrt(sum(x²)) * sqrt(sum(y²)))`
   over the **PD-pulse region only** (use the segmentation mask), *not* the full 2048 window. Add a second metric `ncc_global` for transparency but report `ncc_pulse` as primary.
3. Re-run `test_forward.py` on one checkpoint and verify outputs are bipolar (both positive and negative values present).

**Deliverable:** `docs/SANITY_CHECK.md` — one page documenting what you checked and fixed.

### 1.3 Reconcile thesis-reported NCC (0.5145) vs current (0.9480)
- Write `docs/METRIC_RECONCILIATION.md` explaining the discrepancy in one paragraph (e.g., "Thesis reported global NCC on a harder composite test set; current dashboard reports pulse-region NCC on synthetic-only test. Both are reported henceforth.").
- Do this now. In viva an examiner *will* notice.

---

## 2. Build the missing flat-SNR baseline (Days 2–3)

This is the single highest-leverage scientific fix.

### 2.1 Create `baselines/` package
```
baselines/
├── __init__.py
├── unet_flat.py           # plain 1D U-Net, 4 down / 4 up, no attention, no Swin, no BiGRU
├── pure_cnn_flat.py       # 8-layer 1D CNN, residual
└── train_baseline.py      # flat-SNR trainer, MSE+L1, no curriculum, no MTL
```

### 2.2 Training protocol for baselines
| Setting | Value | Reason |
|---|---|---|
| SNR distribution | Uniform over [−20, +15] dB | Flat, no curriculum |
| Noise type | WGN+Impulsive+NBI mixed | Match final-phase distribution of curriculum models |
| Loss | 0.5·MSE + 0.5·L1 | Simplest reasonable choice |
| Optimizer | AdamW, lr=3e-4, wd=1e-4 | Same as MR-TAE |
| Epochs | 100 | Same budget |
| Data | Same synthetic generator, seed=42 | Identical data |
| MTL | Off | Single-task regression |
| Test sets | Same frozen `.pt` files | Identical evaluation |

### 2.3 Controlled comparisons to run afterwards
| Model | Curriculum | MTL | Attention | Purpose |
|---|---|---|---|---|
| U-Net-flat | ❌ | ❌ | ❌ | Absolute floor |
| U-Net-curr | ✅ | ❌ | ❌ | Isolate curriculum effect |
| U-Net-curr-MTL | ✅ | ✅ | ❌ | Isolate MTL effect |
| MR-TAE-FULL-flat | ❌ | ✅ | ✅ | Isolate curriculum on your architecture |
| MR-TAE-FULL-curr (exists) | ✅ | ✅ | ✅ | Your proposed model |

These five runs answer: *does curriculum actually help, or is it the architecture?*

---

## 3. Add external SOTA anchor (Day 4)

Choose **one** of:
- **CleanUNet-1D** (NVIDIA 2022) — recommended, simplest to port
- **Mamba-1D / S4-denoise** (2023–2024) — strongest but harder to implement

Add under `baselines/external/cleanunet_1d.py`. Train with same protocol as Section 2. This gives you a published-SOTA comparison point for the thesis discussion chapter.

**Do not reimplement from scratch.** Clone the official repo, strip to the 1D core, wrap in your `BaseDenoiser` interface. ~200 LOC.

---

## 4. Optuna HPO (Days 5–6, runs overnight)

### 4.1 Scope
- **Only MR-TAE-FULL.** Not all 13 variants.
- **Phase-1 only, 30 epochs per trial.** Use Phase-1 NCC as the Optuna objective; the assumption is that Phase-1 leaderboard ≈ final leaderboard (validated by Hyperband).
- **30 trials, Hyperband pruner** (kills bottom 50% at epoch 10, 20).
- **Storage:** SQLite at `hpo/optuna.db` (survives crashes).

### 4.2 `config/search_space.yaml`
```yaml
learning_rate:    {type: loguniform, low: 1.0e-5, high: 5.0e-3}
weight_decay:     {type: loguniform, low: 1.0e-6, high: 1.0e-3}
batch_size:       {type: categorical, choices: [32, 64, 128]}
bigru_hidden:     {type: categorical, choices: [64, 128, 256]}
bigru_layers:     {type: int, low: 1, high: 3}
swin_window:      {type: categorical, choices: [8, 16, 32]}
swin_heads:       {type: categorical, choices: [2, 4, 8]}
focal_gamma:      {type: float, low: 1.0, high: 3.0}
charbonnier_eps:  {type: loguniform, low: 1.0e-4, high: 1.0e-2}
grad_clip:        {type: float, low: 0.5, high: 5.0}
```

### 4.3 After HPO
- Save winner to `config/best_hparams.yaml`.
- Retrain MR-TAE-FULL end-to-end (all 3 phases, 100 epochs) with those hparams.
- Log as `MR-TAE-FULL-HPO-BEST` in MLflow.
- This is the model that goes in your thesis Table 4.1.

---

## 5. Kernel-level optimization (Day 7, optional but easy wins)

Add to the trainer init:
```python
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")          # TF32 on Ampere+
torch.backends.cuda.matmul.allow_tf32 = True
model = model.to(memory_format=torch.channels_last) # for conv-heavy blocks
model = torch.compile(model, mode="reduce-overhead")# PyTorch 2.x
```

Additional:
- **Fused AdamW:** `torch.optim.AdamW(..., fused=True)` on CUDA.
- **Flash-Attention for Swin:** replace `scaled_dot_product_attention` with `torch.nn.functional.scaled_dot_product_attention` (already uses FlashAttn-2 under the hood on PyTorch 2.2+).
- **AMP bf16 on Ampere+:** bf16 is more stable than fp16 for gradients, same speed. Replace `torch.cuda.amp.autocast(dtype=torch.float16)` with `dtype=torch.bfloat16`.
- **Pinned memory + `num_workers=4`** on the DataLoader.

Expected: 20–40% faster epoch time on RTX 4070. Benchmark before/after in `docs/KERNEL_OPT_BENCH.md`.

---

## 6. WGAN-GP for real-noise synthesis (Days 8–10)

### 6.1 Extract real-noise segments
```
data/wgan/extract_noise_segments.py
  input:  data/raw/qlin/*.mat
  output: data/real_noise_segments.pt  (shape [N, 2048])
  method: amplitude threshold → find gaps with no PD event → slice 2048-sample chunks
```

### 6.2 Train WGAN-GP
- Architecture: 1D DCGAN-style generator (latent 128 → 2048), 1D critic with spectral normalization.
- Loss: WGAN-GP with λ=10.
- Early stop: Maximum Mean Discrepancy (MMD) on a held-out set of real gaps. Stop when MMD plateaus for 50 epochs.
- **Do not train to convergence on a generator loss alone** — GAN loss is not a quality indicator.

### 6.3 Augmented retraining
Retrain MR-TAE-FULL-HPO-BEST with noise sampled:
- 50% from mathematical generator (current pipeline)
- 50% from trained WGAN

Call this run `MR-TAE-FULL-WGAN-AUG`. Compare to `MR-TAE-FULL-HPO-BEST` on the **real Q.Lin test set**. If NCC on real data improves ≥ 0.02, WGAN earned its place.

**If it doesn't improve:** report honestly as a negative result. Negative results are thesis-worthy.

### 6.4 What about WGAN for pulse generation?
**Skip for the main thesis.** Add as "future work" only. Mode-collapse risk is high; your physics-informed pulse generator already produces realistic morphology. One paragraph of future-work justification is enough.

---

## 7. Self-supervised fine-tuning with Noise2Void (Days 11–12)

### 7.1 Rationale
Your Q.Lin real dataset has no clean ground truth. **Noise2Noise is physically impossible** (can't capture the same microsecond transient twice). **Noise2Void** (blind-spot masking) works on single noisy observations and is the correct paradigm here.

### 7.2 Implementation
```
pipeline/noise2void_finetune.py
  input:  checkpoints/MR-TAE-FULL-WGAN-AUG/best.pt
         data/raw/qlin/*.mat (unlabelled)
  output: checkpoints/MR-TAE-FULL-N2V/best.pt
```

- Mask random 0.5% of samples per window with neighbourhood averaging.
- Loss computed only at masked positions.
- 20 epochs fine-tune, lr=1e-5 (10× lower than training).
- Freeze the Swin block, fine-tune only BiGRU + decoder.

### 7.3 Validation
- Before/after on Q.Lin: measure (a) pulse-region NCC if clean proxy available, (b) downstream classification accuracy (primary metric — does denoising make classification easier?).

---

## 8. Benchmarking and reporting (Days 13–14)

### 8.1 `evaluation/benchmark_runner.py` — required outputs
For every model in `results/*/best.pt`, produce:
- `benchmark_results.json` with: `{synthetic: {ncc_pulse, ncc_global, rmse, snr_improvement, dice, fpr}, real_qlin: {...}, efficiency: {params_M, latency_ms_b1, vram_MB}, per_snr: {−15: ..., −10: ..., ...}}`
- Figures:
  - `results/comparison_bar_all_models.png`
  - `results/ablation_heatmap.png`
  - `results/efficiency_scatter.png` (latency vs NCC, bubble = params)
  - `results/snr_sweep_all.png`
  - `results/signal_comparison_{−15,0,+5}dB.png` (noisy | denoised | clean for each model)

### 8.2 Downstream classification test (critical)
Denoise → feed to a frozen classifier → measure accuracy. If the "best denoiser" doesn't improve downstream classification, the NCC is lying.

```
evaluation/classification_downstream.py
  classifier: 1D ResNet-18 trained on CLEAN signals only
  test: denoised outputs from each model
  metric: per-class accuracy + macro-F1
```

### 8.3 Explainability
- SHAP DeepExplainer on `MR-TAE-FULL-HPO-BEST`, 200 samples.
- LIME on 20 windows (qualitative).
- Output: `results/shap_summary.png`, `results/lime_examples.png`.

---

## 9. Documentation outputs (ongoing, finalise Day 15)

Every one of these must exist before the viva:

| File | Purpose |
|---|---|
| `docs/ARCHITECTURE_SUMMARY.md` | All models, ASCII diagrams, param counts, design rationale |
| `docs/EXAMINER_QA.md` | 20 Q&A covering AE physics, metric choice, curriculum rationale, MTL math, WGAN validation, N2V choice |
| `docs/TEST_SET_HASHES.md` | SHA256 of both test sets |
| `docs/METRIC_RECONCILIATION.md` | Why thesis NCC (0.5145) differs from dashboard NCC (0.9480) |
| `docs/AE_VS_EMI_NOISE.md` | Clarifies AE-specific noise model vs EMI-specific (the "wrong address" fix) |
| `docs/KERNEL_OPT_BENCH.md` | Before/after torch.compile + bf16 + channels_last |
| `docs/RESULTS_COMPARISON.md` | Auto-generated master results table |
| `docs/FINAL_VERDICT.md` | Empirical answers: most robust, most efficient, best ablation, is WGAN worth it, is curriculum worth it |
| `docs/LIMITATIONS.md` | Honest list: synthetic-to-real gap, single-GPU budget, no independent real test labels |

---

## 10. What we are deliberately *not* doing

Include this section in the thesis "Future Work" chapter with honest justification.

| Not doing | Why not |
|---|---|
| Airflow / Dagster / Prefect DAGs | Solo FYP, single machine. Added infra complexity with no scientific gain. Mentioned as future work for multi-node deployment. |
| Kubernetes | Same reason. One paragraph of future work. |
| WGAN for pulse generation | Mode collapse risk > benefit given existing physics generator. |
| Diffusion-model denoiser from scratch | Time. One external SOTA baseline (CleanUNet) is sufficient. |
| Full Mamba/S4 re-architecture | Beyond scope. Cite as future direction. |
| Multi-GPU distributed training | One RTX 4070. DDP adds complexity for no current benefit. |

---

## 11. Execution order summary (for the agent)

```
Day 1:  Section 1 — fix resume button, audit activation, audit NCC, reconcile metrics
Day 2:  Section 2.1–2.2 — build U-Net-flat baseline package
Day 3:  Section 2.2–2.3 — train U-Net-flat, U-Net-curr, U-Net-curr-MTL
Day 4:  Section 3     — port CleanUNet-1D, train
Day 5:  Section 4.1–4.2 — set up Optuna study
Day 6:  Section 4     — run HPO overnight, retrain with best
Day 7:  Section 5     — kernel optimizations, benchmark
Day 8:  Section 6.1–6.2 — extract noise gaps, start WGAN training
Day 9:  Section 6.2   — WGAN continues (runs overnight)
Day 10: Section 6.3   — WGAN-augmented retraining
Day 11: Section 7     — N2V fine-tuning
Day 12: Section 7.3   — validate N2V on real data
Day 13: Section 8.1–8.2 — benchmark runner + downstream classification
Day 14: Section 8.3   — SHAP/LIME
Day 15: Section 9     — finalise all documentation
```

Total: **~15 working days** assuming overnight GPU runs run unattended.

---

## 12. Agent execution rules

1. **Never modify `data/test_*.pt` files.** Verify SHA256 before every benchmark run.
2. **Never overwrite `results/{model_id}/best.pt` without moving the old one to `results/{model_id}/archive/`.**
3. **Every new run gets a unique MLflow run name** of the form `{model_id}_{yyyy-mm-dd_HHMM}`.
4. **Every script starts with `torch.manual_seed(42); np.random.seed(42); random.seed(42)`.**
5. **Every Python file imports from one `PROJECT_ROOT` derived from `__file__`** — no hardcoded paths.
6. **Every failure writes a stacktrace to `logs/errors/{timestamp}.log`** and raises. No silent fallbacks.
7. **Before marking a section "done," run the smoke tests in `tests/`** and update `PROJECT_STATUS.md`.

---

## 13. Known risks and mitigations

| Risk | Probability | Mitigation |
|---|---|---|
| HPO finds unstable hparams that work on Phase-1 but diverge in Phase-3 | Medium | Retrain HPO winner full 100 epochs; if diverges, fall back to current hparams and document |
| WGAN noise doesn't improve real-data NCC | Medium | Report honestly as negative result; thesis gains scientific credibility |
| torch.compile breaks on Swin block | Low-Medium | Wrap in try/except, fall back to eager mode, document in KERNEL_OPT_BENCH |
| Q.Lin real data format not as expected | Medium | Build `data/qlin_loader.py` first with a visual sanity-check figure before any training |
| Dashboard resume still broken after fix | Low | Document manual resume via `python pipeline/mlflow_orchestrator.py --resume` as fallback; include in README |