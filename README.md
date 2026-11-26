# Intelligent PD Denoising — AE Signal Denoising with MR-TAE

> Start here if you are taking this over: [JUNIOR_HANDOFF.md](JUNIOR_HANDOFF.md).
> Paste-ready brief for a new architecture: [JUNIOR_REBUILD_PROMPT.md](JUNIOR_REBUILD_PROMPT.md).
> The package is `Try Model/mr_tae_fusion`, not a top-level `mr_tae_fusion` folder.
> Headline numbers and which files to ignore are in the handoff. This README's hash for `test_synthetic_ae.pt` is stale. Use `docs/TEST_SET_HASHES.md`.

**Multi-Resolution Temporal Autoencoder for Partial Discharge signal denoising in power transformers.**

Thesis project — Oo Jian Hong, University of Malaya, 2026.

MR-TAE is a deep-learning pipeline that fuses **MWCNN** (Multi-level Wavelet CNN) encoding,
a **BiGRU + Swin Transformer** bottleneck, attention-gated skip connections, and a
**dual-head** decoder (signal reconstruction + PD-event segmentation) trained with
homoscedastic-uncertainty multi-task learning and a 3-phase SNR curriculum.

---

## Quick Start

```bash
# 1. Activate the virtual environment
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux / macOS

# 2. Train all 13 ablation variants (resumes automatically on crash)
python pipeline/mlflow_orchestrator.py --run-all

# 3. Open the research dashboard
python app.py                     # http://localhost:1001
```

> **Prerequisites:** Python 3.11+, NVIDIA GPU with CUDA (tested on RTX 4070 12 GB).
> Install dependencies with `pip install -r requirements.txt`.

---

## Project Structure

```
FYP_MASTER/
├── config/
│   ├── model_registry.yaml       # 13 model variants (ablation flags + loss config)
│   └── training_config.yaml      # LR, curriculum epochs, dataset sizing
├── mr_tae_fusion/                # Core model package
│   ├── models/                   #   MR-TAE architecture + ablation configs
│   ├── data/                     #   Synthetic PD dataset generator
│   └── training/                 #   Loss functions (Charbonnier, Focal, Dice, SSIM)
├── training/
│   ├── trainer.py                # Unified trainer (AMP, OOM recovery, snapshots)
│   ├── curricular_scheduler.py   # 3-phase SNR curriculum (easy → hard)
│   ├── resilience.py             # Crash-proof session persistence + dynamic early stop
│   ├── heat_monitor.py           # GPU thermal gating (pauses if too hot)
│   └── memory_manager.py         # RTX 4070 VRAM manager + batch-size selection
├── pipeline/
│   └── mlflow_orchestrator.py    # Resilient experiment orchestrator (single entry)
├── evaluation/
│   ├── benchmark_runner.py       # Full benchmark: NCC, RMSE, Dice, SNR-sweep, baselines
│   ├── metrics.py                # Standalone metric utilities
│   └── explainability/           # SHAP / LIME explanations
├── hpo/
│   └── optuna_study.py           # Optuna HPO (TPE + Hyperband, 30 trials)
├── wgan_augmentation/            # WGAN-GP for synthetic PD augmentation
├── dashboard/                    # Dashboard assets (if separated from app.py)
├── tests/
│   ├── test_forward.py           # Forward-pass smoke tests for all 13 variants
│   ├── test_integration.py       # End-to-end integration tests
│   └── test_sota.py              # SOTA comparison tests
├── matlab/                       # MATLAB feature extraction & classification scripts
├── docs/
│   ├── ARCHITECTURE_SUMMARY.md   # Model architecture diagrams + ablation methodology
│   ├── EXAMINER_QA.md            # 15 viva Q&A with thesis-defense answers
│   ├── TEST_SET_HASHES.md        # SHA256 hashes for test-set reproducibility
│   └── RESULTS_COMPARISON.md     # Auto-generated benchmark results table
├── data/                         # Test sets (test_synthetic_ae.pt, test_real_qlin.pt)
├── results/                      # Per-model checkpoints, histories, benchmark JSONs
├── mlruns/                       # MLflow tracking store
├── app.py                        # Research Command Centre (Flask + inline frontend)
├── session_state.json            # Crash-proof session state (auto-managed)
├── RESUME_PROMPT.md              # Auto-generated resume instructions for agent/human
├── Dockerfile                    # Containerised training environment
├── docker-compose.yml            # Docker Compose for training + dashboard
└── requirements.txt              # Python dependencies
```

---

## Training All Models

The orchestrator trains all 13 variants from `config/model_registry.yaml` in order of
increasing complexity. It persists state to `session_state.json` after every epoch, so
training resumes exactly where it left off after any crash, power loss, or manual stop.

```bash
# Train all models (queue is sorted by complexity, simplest first)
python pipeline/mlflow_orchestrator.py --run-all

# Resume from last checkpoint (reads session_state.json)
python pipeline/mlflow_orchestrator.py --resume

# Train a single model
python pipeline/mlflow_orchestrator.py --model MR-TAE-FULL

# Train with custom hyperparameters
python pipeline/mlflow_orchestrator.py --resume --hparams config/best_hparams.yaml
```

Each model trains for **100 epochs** across three curriculum phases:

| Phase | Epochs | SNR Range      | Noise Type    |
|-------|--------|----------------|---------------|
| 1     | 35     | +5 to +15 dB   | WGN           |
| 2     | 35     | −5 to +5 dB    | WGN+Impulsive |
| 3     | 30     | −20 to −5 dB   | Composite     |

---

## Running Hyperparameter Optimisation

```bash
# 30 Optuna trials, Phase 1 only (30 epochs each), Hyperband pruning
python hpo/optuna_study.py --n-trials 30 --study-name ae-pd-hpo

# Best config is saved to config/best_hparams.yaml
# Then re-train with the discovered hparams:
python pipeline/mlflow_orchestrator.py --resume --hparams config/best_hparams.yaml
```

The search space includes learning rate, weight decay, batch size, number of GRU layers,
Swin window size, and loss weighting coefficients.

---

## Reproducing Benchmark Results

```bash
# Run full benchmark on all trained checkpoints + classical baselines
python evaluation/benchmark_runner.py --compare-all

# Benchmark a single model
python evaluation/benchmark_runner.py \
    --model-id MR-TAE-FULL \
    --checkpoint results/MR-TAE-FULL/checkpoints/best.pt

# Regenerate figures only (no re-inference)
python evaluation/benchmark_runner.py --figures-only
```

**Test set integrity** — verify SHA256 hashes before benchmarking to ensure test data has
not been modified. Expected hashes are recorded in [`docs/TEST_SET_HASHES.md`](docs/TEST_SET_HASHES.md):

| Test Set               | SHA256 (first 16 chars)  |
|------------------------|--------------------------|
| `test_synthetic_ae.pt` | `ff49e064d1e4ad7b`       |
| `test_real_qlin.pt`    | `de22e1f513c56dce`       |

Full hashes:
- **test_synthetic_ae.pt:** `ff49e064d1e4ad7b1d2f44ac5967b3a4e6b4407828b7aeddc19bd9bb17ebff9b`
- **test_real_qlin.pt:** `de22e1f513c56dce125ce58e54ed9ba86fc8f6804a07a9bceb5005ed4f3d8896`

Verify with:
```bash
python -c "import hashlib, pathlib; print(hashlib.sha256(pathlib.Path('data/test_synthetic_ae.pt').read_bytes()).hexdigest())"
```

---

## References

1. **Q. Lin et al.** — Experimental dataset: 4 XLPE cable samples, 200 AE traces × 1 M samples each,
   gap distances 12/15/18/21 mm, 2 MHz sampling rate.
2. **Liu et al. (2018)** — MWCNN: Multi-level Wavelet CNN for Image Restoration.
   *IEEE TPAMI, 40(10)*.
3. **Liu et al. (2021)** — Swin Transformer: Hierarchical Vision Transformer using
   Shifted Windows. *ICCV 2021*.
4. **Kendall et al. (2018)** — Multi-task learning using uncertainty to weigh losses.
   *CVPR 2018*.
5. **Gulrajani et al. (2017)** — Improved Training of Wasserstein GANs. *NeurIPS 2017*.
6. **IEC 62478:2016** — High-voltage test techniques — Measurement of partial discharges
   by electromagnetic and acoustic methods.

---

*Dashboard: `python app.py` → [http://localhost:1001](http://localhost:1001)*
*Experiment tracking: `mlflow ui --backend-store-uri mlruns`*
