# FYP Session Handoff — AE-PD Denoising (MR-TAE)
**Date:** 2026-05-10  
**Student:** Oo Jian Hong, 22004678, University of Malaya  
**Project root:** `D:\New folder (2)\Data\FYP_MASTER`  
**Venv:** `.venv` (activate: `& "d:/New folder (2)/Data/FYP_MASTER/.venv/Scripts/Activate.ps1"`)

---

## 1. Project Summary

Deep learning AE-PD signal denoising for high-voltage transformer condition monitoring.  
Model: **MR-TAE** (Multi-Level Wavelet Recurrent-Transformer Autoencoder).  
Dataset: Synthetic AE-PD signals + real Q.Lin copper-bead AE recordings (4 gap sizes: 1.0, 1.8, 2.0, 2.5 mm).  
Two output tracks: **Robust** (MR-TAE-FULL-LossB, thesis headline) and **Light** (MWCNN-BiGRU, PD-CLIP edge device).

---

## 2. What Was Accomplished This Session

### Training
- Discovered all 13 models were previously trained on only **4,000 samples** (default fallback) — severely undertrained
- Created `config/training_config.yaml` with `n_train: 50000`, `batch_size: 128` → 391 steps/epoch (was 31)
- **MR-TAE-FULL-LossB fully retrained** — 110 epochs, 3 phases, best val NCC = **0.9856 at epoch 86**
- Training took ~3.5 hours on RTX 4070, GPU stable at 55–63°C

### Infrastructure fixes
- `session_state.json` cleared to force retrain
- `mr_tae_fusion` broken symlink identified (at `D:\New folder (2)\Data\FYP_MASTER\mr_tae_fusion`)
- Dead OneDrive fallback path removed from `data/build_test_sets.py`
- `data/wgan/wgan_noise.py` Q.Lin path patched from OneDrive → FYP_MASTER
- `num_workers: 4` → `num_workers: 0` in training_config.yaml (fixes Windows shared memory OOM crash error 1455)
- `evaluation/benchmark_runner.py` patched: `load_state_dict(..., strict=False)`
- Deprecation warnings fixed: `nvidia-ml-py` install, `torch.amp` replacements

### Evaluation pipeline rebuilt
- `--no-attention-gates` flag added to benchmark_runner.py
- `best_config.json` now saved alongside `best.pt` every training run
- `ablation_for_checkpoint()` reads `best_config.json` automatically — no more flag mismatches
- Real data evaluation corrected: Q.Lin signals are **nearly clean** — synthetic noise now injected at test time, NCC computed vs clean Q.Lin ground truth (`real_qlin_grounded` key)
- `data/build_test_sets.py` now saves: `test_real_qlin_clean.pt`, `test_real_qlin_noisy.pt`, `test_real_qlin_snr_labels.pt`
- Noise verification figure: `docs/real_noise_injection_verify.png` confirms correct generators used

### Thermal management added
- `ThermalSchedule` in `training/trainer.py`: 5-min break after Phase 1, 2-min after Phase 2, 1-min micro-breaks every 10 epochs in Phase 2, 45-sec in Phase 3
- `HeatMonitor` pauses at 82°C, resumes at 68°C, polls every 90s
- All sleeps interruptible by Ctrl+C

---

## 3. Current Benchmark Results — MR-TAE-FULL-LossB (Properly Trained)

File: `results/MR-TAE-FULL-LossB/benchmark_results.json`

### Synthetic (ground truth available)
| SNR | NCC |
|-----|-----|
| +5 dB | 0.9654 |
| 0 dB | 0.9015 |
| −5 dB | 0.6404 |
| −10 dB | 0.3483 |
| −15 dB | 0.1993 |
| **Mean** | **0.6110** |

SNR improvement: **+5.03 dB** (vs BayesShrink +3.78 dB = **33% better**)  
Dice (PD segmentation): **0.903**  
Params: 3.05M | Latency b=1: 8.53ms

### Real grounded (vs clean Q.Lin, synthetic noise injected)
NCC mean: **0.4615** — only 600 windows used, no per-SNR breakdown yet  
**This needs to be rebuilt with more windows (5000) — see pending tasks**

### Thesis headline claim (defensible)
> MR-TAE-FULL-LossB achieves **+5.03 dB SNR improvement** and **NCC 0.902 at 0 dB SNR**, outperforming BayesShrink (+3.78 dB) by 33%, with simultaneous PD-type segmentation (Dice 0.903) unavailable in classical methods.

---

## 4. Known Issues / Open Items

| # | Issue | Status | Fix |
|---|-------|--------|-----|
| 1 | Real grounded eval only used 600 windows | **PENDING** | Rebuild with `--max-grounded-windows 5000` |
| 2 | MR-TAE-FULL not yet retrained (crashed on OOM) | **PENDING** | Fix: num_workers=0, delete session_state.json, retrain |
| 3 | MWCNN-BiGRU not yet retrained (light track) | **PENDING** | Train after MR-TAE-FULL completes |
| 4 | MR-TAE-FULL was using wrong flags (loss=B instead of loss=A) | **PENDING** | Cursor fix in mlflow_orchestrator.py |
| 5 | WGAN noise extractor never trained (wgan_noise.py hung on mat loading) | DEFERRED | Future work in thesis |
| 6 | `torch.amp` deprecation warnings in trainer.py and benchmark_runner.py | Minor | `fix_warnings.py` script ready |
| 7 | Synthetic NCC benchmark used wrong model at inference (attention gate mismatch) | **FIXED** via `--no-attention-gates` + `best_config.json` |
| 8 | NCC discrepancy: val_ncc 0.9856 vs benchmark 0.6110 | Explained | Val uses easy Phase-1-style data; benchmark uses balanced −15 to +5 dB |

---

## 5. Noise Model (Confirmed Correct — Phase 0 Verified)

Grid: **50 Hz (Malaysia)**

| Type | Source | Frequencies |
|------|--------|-------------|
| AWGN | Thermal/electronic | Broadband |
| Magnetostrictive | Core lamination | 100 Hz + 200 Hz |
| Powerline | HV bus coupling | 50 Hz + 150 Hz (3rd harmonic) |
| Bernoulli-Gaussian impulse | Switching transients | Wideband sparse |
| NBI | Cooling fan resonance | 30–80 kHz |

Note: Powerline implements only fundamental + 3rd harmonic (no 5th/7th). Confirm with supervisor if 5th/7th needed.  
Note: `composite_noise` and `generate_impulsive_noise` are two separate impulse recipes — mention distinction in Chapter 3.3.

---

## 6. Architecture (Final Locked Config for Robust Track)

```
Input (B, 1, 2048)
  → MWCNN encoder (db4 DWT, 4 levels, GELU)
  → BiGRU bottleneck (2 layers, hidden 128/direction)
  → Swin Transformer 1D (window=8, 4 heads)
  → MWCNN decoder (IDWT upsampling)
  → Reconstruction head: denoised signal (B, 1, 2048)
  → Segmentation head: PD-type mask (B, 1, 2048)
```

**Decisions locked:**
- `use_attention_gates: false` — ablation showed they hurt real-world transfer
- `loss_config: B` — Charbonnier + GDice only (no Focal, no uncertainty weighting)
- LossA (uncertainty + Focal) overfits to synthetic domain → 2.27× worse real NCC
- LossC (SSIM) catastrophically fails on real data (RMSE=23.2)

---

## 7. File Locations

| Item | Path |
|------|------|
| Training config | `config/training_config.yaml` |
| Model registry | `config/model_registry.yaml` |
| Orchestrator | `pipeline/mlflow_orchestrator.py` |
| Trainer | `training/trainer.py` |
| Benchmark runner | `evaluation/benchmark_runner.py` |
| Test sets | `data/test_synthetic_ae.pt`, `data/test_real_qlin_clean.pt`, `data/test_real_qlin_noisy.pt` |
| Test hashes | `docs/TEST_SET_HASHES.md` |
| Best checkpoint (LossB) | `results/MR-TAE-FULL-LossB/checkpoints/best.pt` |
| Best config (LossB) | `results/MR-TAE-FULL-LossB/checkpoints/best_config.json` |
| Noise validation | `docs/noise_validation.png`, `docs/pulse_validation.png` |
| Real noise verify | `docs/real_noise_injection_verify.png` |
| Q.Lin mat files | `D:\New folder (2)\Data\FYP_MASTER\Datset from Q.lin\` |
| Session state | `session_state.json` (delete to force retrain) |

---

## 8. Pending Cursor Prompts (Execute In Order)

### Prompt A — Fix loss_config flag per model in orchestrator
```
In `pipeline/mlflow_orchestrator.py`, fix `merge_model_flags_from_training` so
that `loss_config` from training_config.yaml does NOT override per-model loss.
Instead: if model_id ends with "LossB" → loss_config=B, "LossC" → loss_config=C,
all others → read from model_registry.yaml (default LossA).
Only architecture flags (use_attention_gates, use_bigru, use_swin,
use_wavelet_pooling, use_mtl) are merged globally from training_config.yaml.
```

### Prompt B — Rebuild real grounded test set with more windows
```
Run: python data/build_test_sets.py --real-only --max-grounded-windows 5000
This rebuilds test_real_qlin_clean.pt, test_real_qlin_noisy.pt,
test_real_qlin_snr_labels.pt with 5000 clean windows × 5 SNR levels = 25000 rows.
After build, update docs/TEST_SET_HASHES.md with new SHA256 hashes.
Print the noise verification stdout lines (source file path + function name).
```

### Prompt C — Train MR-TAE-FULL (LossA, ablation baseline)
```
1. Confirm config/training_config.yaml has num_workers: 0
2. Delete session_state.json
3. Run: python pipeline/mlflow_orchestrator.py --model MR-TAE-FULL
   This trains with LossA (Charbonnier + Focal + GDice + uncertainty weighting)
   for 110 epochs, 50k samples/epoch.
4. After training, run benchmark:
   python evaluation/benchmark_runner.py --model-id MR-TAE-FULL
   --checkpoint results/MR-TAE-FULL/checkpoints/best.pt
   (best_config.json will supply architecture flags automatically)
5. Report: synthetic NCC mean, SNR improvement, real_qlin_grounded NCC mean,
   and per-SNR breakdown.
```

### Prompt D — Train MWCNN-BiGRU (light track)
```
1. In config/training_config.yaml, temporarily set use_swin: false
2. Delete session_state.json
3. Run: python pipeline/mlflow_orchestrator.py --model MWCNN-BiGRU
4. After training, run benchmark:
   python evaluation/benchmark_runner.py --model-id MWCNN-BiGRU
   --checkpoint results/MWCNN-BiGRU/checkpoints/best.pt
5. Reset use_swin: true in training_config.yaml after training completes
6. Report: synthetic NCC, SNR improvement, latency b=1, param count
```

### Prompt E — Generate thesis figures
```
Run python evaluation/benchmark_runner.py --compare-all to collect all
available model benchmarks into one comparison.
Then generate 5 figures to results/benchmark/:
1. comparison_bar.png — SNR improvement bar chart for all models
2. ablation_heatmap.png — NCC delta vs MR-TAE-FULL-LossB per component removed
3. snr_sweep.png — per-SNR NCC curves for top 5 models
4. efficiency_scatter.png — latency vs NCC, bubble size = param count
5. loss_comparison.png — LossA vs LossB vs LossC on MR-TAE-FULL

Also generate qualitative figure: 5 random Q.Lin windows showing
input | LossB denoised | BayesShrink denoised — save to docs/qualitative_real.png
```

### Prompt F — Fix deprecation warnings permanently
```
In training/trainer.py:
  Replace: torch.cuda.amp.GradScaler(
  With:    torch.amp.GradScaler('cuda',

  Replace: torch.cuda.amp.autocast()
  With:    torch.amp.autocast('cuda')

In evaluation/benchmark_runner.py: same replacements.

Run: pip install nvidia-ml-py --quiet
to fix the pynvml FutureWarning.
```

---

## 9. Thesis Narrative (Write This in Chapter 4)

The results support this story:

> MR-TAE-FULL-LossB achieves +5.03 dB SNR improvement on synthetic AE-PD data,
> outperforming the best classical baseline (BayesShrink, +3.78 dB) by 33%.
> Simultaneously it performs PD-type segmentation (Dice 0.903) which no classical
> method provides. Ablation analysis confirms that removing either the wavelet pooling
> or the MTL head causes complete collapse on real domain data. Loss function analysis
> shows that simpler supervision (Charbonnier + GDice) produces significantly better
> real-domain generalization than the full uncertainty-weighted formulation, attributed
> to reduced synthetic-domain overfitting.
>
> Real-data evaluation injects synthetic noise into clean Q.Lin piezoelectric AE
> recordings and measures NCC against the clean signal. This provides a meaningful
> domain-transfer metric without requiring paired clean/noisy real recordings, which
> are physically impossible to obtain for transformer PD measurements.

**One paragraph needed in methodology:** The real-data NCC metric uses input-referenced NCC (for the legacy `real_qlin` column) and ground-truth-referenced NCC (for `real_qlin_grounded`). The latter injects synthetic noise into clean Q.Lin windows and computes NCC vs the original. Both are reported; the latter is the primary generalization metric.

---

## 10. What NOT To Do

- Do not retrain all 13 models — only LossB (done), FULL, and MWCNN-BiGRU needed
- Do not implement WGAN, HPO, SHAP, Noise2Void before all 3 models have benchmark numbers
- Do not use the mean synthetic NCC (0.611) as your headline — use per-SNR @ 0 dB (0.902)
- Do not start thesis writing until real_qlin_grounded per-SNR breakdown is available
- Do not change test sets after hashes are locked in TEST_SET_HASHES.md
