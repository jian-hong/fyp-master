# Final Verdict — AE-PD Denoising Benchmarks

**Purpose:** Single place to summarize conclusions after all experiments are complete. Replace every `{{PLACEHOLDER}}` with measured values, or paste tables from `docs/RESULTS_COMPARISON.md` / MLflow.

**Last updated:** {{DATE}}

---

## 1. Executive summary

In one paragraph:

> {{ONE_PARAGRAPH_SUMMARY — e.g., MR-TAE-FULL achieved the best real-set NCC ({{NCC_REAL_FULL}}) while {{MODEL_EFF}} offered the best accuracy per million parameters. Wavelet pooling and {{COMPONENT}} had the largest ablation impact.}}

---

## 2. Best overall model

| Criterion | Value |
|-----------|--------|
| **Model name** | {{e.g., MR-TAE-FULL}} |
| **Primary metric (real Q.Lin NCC, mean)** | {{VALUE ± STD}} |
| **Synthetic test NCC (10K windows, pooled)** | {{VALUE}} |
| **RMSE (real, if reported)** | {{VALUE}} |
| **Segmentation Dice / FPR (if applicable)** | {{VALUE}} |
| **Checkpoint path** | `{{results/.../best.pt}}` |
| **Notes** | {{e.g., best trade-off after full 100-epoch curriculum}} |

---

## 3. Most efficient model

| Criterion | Value |
|-----------|--------|
| **Model name** | {{e.g., MR-TAE-noSwin-noBiGRU or MWCNN-BiGRU}} |
| **Parameters (M)** | {{X.XX}} |
| **Real NCC** | {{VALUE}} |
| **NCC per million parameters** | {{VALUE}} |
| **Inference latency (ms, batch 1, GPU)** | {{FROM benchmark_results.json}} |
| **Verdict** | {{Why this is “best bang for buck” for deployment}} |

---

## 4. Ablation insights

Fill from the 13-variant study (parameter counts in `ARCHITECTURE_SUMMARY.md`).

### 4.1 Single-component removals (base six)

| Variant | Params (M) | Real NCC | Δ vs FULL | Interpretation |
|---------|------------|----------|------------|----------------|
| MR-TAE-FULL | 3.05 | {{}} | — | Baseline |
| noBiGRU | 2.39 | {{}} | {{}} | {{}} |
| noSwin | 1.41 | {{}} | {{}} | {{}} |
| noAttn | 3.01 | {{}} | {{}} | {{}} |
| noMTL | 3.00 | {{}} | {{}} | {{}} |
| noWavelet | 2.99 | {{}} | {{}} | {{}} |

**Short conclusion:** {{e.g., Removing {{X}} hurt NCC most on real data, supporting {{design choice}}.}}

### 4.2 Two-component removals

| Variant | Params (M) | Real NCC | Note |
|---------|------------|----------|------|
| noSwin-noBiGRU | 1.01 | {{}} | {{}} |
| noSwin-noAttn | 1.36 | {{}} | {{}} |
| noBiGRU-noAttn | 2.35 | {{}} | {{}} |

### 4.3 Cross combinations

| Variant | Params (M) | Real NCC | Role in thesis |
|---------|------------|----------|----------------|
| MWCNN-BiGRU | 1.31 | {{}} | Minimal wavelet + temporal |
| UNet-BiGRU-Swin | 2.90 | {{}} | Non-wavelet encoder baseline |

---

## 5. Loss function findings

Configs: **A** = Charbonnier + Focal + GDice + uncertainty; **B** = Charbonnier + GDice; **C** = Charbonnier + SSIM (no seg); **D** = SmoothL1 + CE + GDice + uncertainty.

| Experiment | Loss config | Real NCC | Synthetic NCC | Comment |
|------------|---------------|----------|---------------|---------|
| MR-TAE-FULL | A | {{}} | {{}} | Default |
| MR-TAE-FULL-LossB | B | {{}} | {{}} | {{}} |
| MR-TAE-FULL-LossC | C | {{}} | {{}} | {{}} |
| {{Optional D run}} | D | {{}} | {{}} | {{}} |

**Takeaway:** {{e.g., Uncertainty weighting + Focal (A) improved {{metric}} vs B/C because {{reason}}.}}

---

## 6. Generalization: synthetic vs real

| Test set | Windows | Observation |
|----------|---------|----------------|
| Synthetic (5 SNR levels, 10K total) | 10,000 | {{e.g., ranking consistent with real / or divergence explained by domain gap}} |
| Real Q.Lin | 780,000 | {{e.g., primary deployment metric}} |

**Domain gap note:** {{e.g., models that win on synthetic but lose on real suggest overfitting to WGN-like artifacts.}}

---

## 7. Recommendations

### 7.1 For best accuracy

- **Choose:** {{MODEL}}
- **Reason:** {{}}

### 7.2 For edge / embedded deployment

- **Choose:** {{MODEL}}
- **Reason:** {{params / latency / memory}}

### 7.3 For future work

1. {{e.g., semi-supervised fine-tuning on plant recordings}}
2. {{e.g., uncertainty-aware alerting}}
3. {{e.g., multi-sensor fusion}}

---

## 8. Reproducibility checklist

- [ ] Test files match hashes in `docs/TEST_SET_HASHES.md`
- [ ] `config/training_config.yaml` and curriculum seeds documented in thesis
- [ ] Checkpoint IDs and MLflow run IDs archived
- [ ] `benchmark_runner.py` command line logged for final table

---

*After benchmarks, you may auto-fill §2–§5 from `results/*/benchmark_results.json` and `docs/RESULTS_COMPARISON.md`.*
