# Final report / arXiv draft outline (aligned with your latest results)

Use this skeleton when you send your draft for review. Figures are in this folder; regenerate with:

```text
.venv\Scripts\python.exe "NEW FOCUS\figure_arxiv_report.py"
```

---

## 1. Abstract (150–250 words)

- Problem: AE partial-discharge (PD) signals buried in noise at low SNR.
- Method: MR-TAE fusion (wavelet + BiGRU + Swin + attention + MTL); Loss-B variant best on synthetic test.
- Results: Mean NCC and ΔSNR vs classical wavelet / Wiener / SavGol at −15, −5, +5 dB; burst-zoom qualitative recovery.
- Claim: Proposed model preserves PD pulse shape and AE-band energy better than top classical baselines at **negative** input SNR.

---

## 2. Introduction

- PD monitoring motivation; why denoising before feature extraction / classification.
- Gap: classical methods vs learned denoisers on 1 MHz AE data.
- Contributions (bullet list):
  1. MR-TAE ablation study (13 variants).
  2. Multi-objective losses (A / B / C).
  3. Benchmark vs UNet-BiGRU-Swin and MWCNN-BiGRU.
  4. Synthetic + (if available) experimental validation narrative.

**Figure:** none, or system block diagram from thesis architecture chapter.

---

## 3. Methods

### 3.1 Signal model

- Window length 2048 @ 1 MHz; SNR definition; AWGN / WGAN noise (cite data pipeline).

### 3.2 MR-TAE architecture

- Encoder–decoder; ablation flags (W, B, S, A, M); Loss-B configuration for best model.

### 3.3 Baselines

- Soft-Wavelet (db4), SavGol+Wavelet, Wiener, SVD(95%) — one paragraph each.

### 3.4 Metrics

- NCC (shape fidelity), ΔSNR (noise reduction vs clean), optional AE-band PSD ratio, kurtosis retention.

---

## 4. Experiments

### 4.1 Dataset

- `test_synthetic_ae.pt`: 10k windows; SNR labels −15, −5, +5 dB (2000 each); +10 dB via re-corruption for high-SNR stress test.

### 4.2 Training

- Phases, checkpoints `best_final.pt`, curriculum SNR (reference `diagnostic_table.txt`).

### 4.3 Evaluation protocol

- 25 windows per SNR for **mean** metrics (`method_comparison_summary.json`).
- **Fixed single window** per SNR for **fair** qualitative figures (all methods see identical noisy/clean).

---

## 5. Results (map figures here)

| Subsection | Recommended figure(s) | Key numbers to cite |
|------------|----------------------|---------------------|
| 5.1 Quantitative ranking | `metrics_ncc_bars.png` | LossB NCC mean: +5 → 0.953, −5 → 0.760, −15 → 0.247 |
| 5.2 Multi-SNR denoising | `denoise_multisnr_FULL-LossB.png` | ΔSNR positive at −5 / −15; caution at +10 |
| 5.3 Low SNR (−5 dB) | `denoise_zoom_snr-5dB_win4000.png`, `before_after_FULL-LossB_snr-5dB.png` | Fixed-window table in eval log |
| 5.4 Very low SNR (−15 dB) | `denoise_zoom_snr-15dB_win0.png`, `spectrum_snr-15dB_win0.png` | SavGol competitive on NCC at single window |
| 5.5 Temporal diversity | `denoise_timediversity_snr-5dB_FULL-LossB.png` | Same model, windows 0 / 2000 / 3999 band |
| 5.6 Frequency domain | `spectrum_snr-5dB_win4000.png` | AE band 20–500 kHz preservation |
| 5.7 Ablation (optional) | `ranked_heatmap.png` from `evaluate_and_visualise.py` | 13-model heatmap |

**Writing tip for +10 dB:** Models trained for low-SNR denoising can **over-smooth** at high input SNR (negative ΔSNR). State this honestly; focus claims on −5 and −15 dB.

---

## 6. Discussion

- Why Loss-B wins (loss weighting / perceptual terms).
- When classical SavGol+Wavelet is strong (high SNR or single-window luck).
- Limitations: synthetic test set; no labelled +10 dB in test file; generalisation to field data.

---

## 7. Conclusion

- Restate best model: **MR-TAE-FULL-LossB**.
- Future work: real-lab AE captures, online inference, PD classification downstream.

---

## 8. LaTeX snippet (example)

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/denoise_zoom_snr-5dB_win4000.png}
  \caption{Zoomed PD burst at $ \mathrm{SNR}_{\mathrm{in}} = -5\,\mathrm{dB} $:
    noisy input, three best deep models (MR-TAE-FULL-LossB, FULL, UNet-BiGRU-Swin),
    and two classical baselines versus clean ground truth (green).}
  \label{fig:denoise_m5}
\end{figure}
```

---

## 9. Checklist before submission

- [ ] All figure paths exist in `arxiv_report_figures/`
- [ ] Captions match **same-window** protocol
- [ ] Table: mean ± std NCC from JSON (not single-window only)
- [ ] State sampling rate 1 MHz and zoom duration ($280/10^6$ s ≈ 280 µs)
- [ ] Ethics / data availability statement
- [ ] Reproducibility: commit hash + `figure_arxiv_report.py` command

---

## 10. When you send your draft back

Include:

1. PDF or Word section for Results (we align numbers to figures).
2. Which SNR levels you want to **emphasise** (−5 only vs full range).
3. Whether you have **real** AE recordings to add a second dataset subsection.

We will check: caption accuracy, metric consistency, over-claiming at +10 dB, and figure resolution (300 dpi PNGs provided).
