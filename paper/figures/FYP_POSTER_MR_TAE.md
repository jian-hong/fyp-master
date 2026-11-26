# MR-TAE FYP Poster — UM iFEST 2026
<!-- Layout: A0 portrait, 5-column grid. Sections map to columns as noted. -->
<!-- CURSOR: Replace all [IMAGE: filename] with actual figures from paper/figures/ -->

---

## HEADER (full width)

**UM iFEST 2026**

# Intelligent Partial Discharge Denoising using a Multi-Level Wavelet Recurrent-Transformer Autoencoder (MR-TAE)

**Oo Jian Hong** (22004678) · Supervisor: **Ir. Dr. Wong Jee Keen Raymond**
Department of Electrical Engineering, Faculty of Engineering, Universiti Malaya, Kuala Lumpur, Malaysia
`wjkraymond@um.edu.my`

---

## COLUMN 1 — INTRODUCTION & PROBLEM

### Background
Partial discharge (PD) is the earliest detectable sign of insulation breakdown in high-voltage (HV) transformers. On-site acoustic-emission (AE) sensors capture these diagnostic pulses but are overwhelmed by broadband noise at SNR as low as **−15 dB**.

Classical denoisers (wavelet thresholding, Wiener filter, Savitzky-Golay) share three fatal limitations:
- Require manual tuning per noise type
- Cannot separate impulsive interference from real PD pulses
- Destroy the morphological signatures (rise time, oscillation) needed for fault classification

### Clinical Significance
> A method that suppresses noise by blurring the signal destroys the very features that distinguish corona from internal discharge.

[IMAGE: fig1_system_overview.png — Pipeline from AE sensor → noise → MR-TAE → denoised + discharge-type label]
**Fig. 1.** End-to-end PD monitoring pipeline.

### Noise Model (5-component composite)
| Component | Source |
|---|---|
| AWGN | Thermal / electronic |
| Broadband structural | Magnetostrictive core hum |
| Powerline harmonics | 50 Hz + 3rd harmonic |
| Narrowband 80 kHz | Comms coupling / fan resonance |
| Bernoulli-Gaussian impulse | Switching transients |

Target SNR: −15 to +15 dB in 5 dB steps across **14,000 frozen test windows**.

---

## COLUMN 2 — ARCHITECTURE

### MR-TAE Architecture
Single unified multi-task autoencoder — denoising + discharge-type segmentation in one inference pass.

[IMAGE: fig2_architecture.png — Full MR-TAE diagram from PPTX (your verified diagram)]
**Fig. 2.** MR-TAE architecture. Encoder: Conv-first then db4-DWT. Bottleneck: BiGRU→Swin sequential. Decoder: Upsample + AG-gated skip + Conv. Dual output heads.

#### Encoder — MWCNN (Conv-first, DWT doubles channels)
```
Input (B,1,2048)
→ Input Proj Conv7 1→32ch       (B,32,2048)
→ Conv Block L1 (2×Conv+BN+GELU) skip₁(B,32,2048)
→ DWT L1: 32→64ch, 2048→1025   (B,64,1025)
→ Conv Block L2                  skip₂(B,64,1025)
→ DWT L2: 64→128ch, 1025→513   (B,128,513)
→ Conv Block L3                  skip₃(B,128,513)
→ DWT L3: 128→256ch, 513→257   (B,256,257)  ← bottleneck
```
Each db4-DWT stacks approximation + detail sub-bands → **channels ×2, length ÷2**.

#### Bottleneck — Sequential BiGRU → Swin-1D
- **BiGRU**: 2-layer, 128 hidden/direction → captures local temporal dynamics
- **Swin-1D**: embed=256, depth=2, 8 heads, window=32, shifted-window attn (shift=16)
- Residual: `y = Proj(Swin(BiGRU(x))) + Residual(x)`

#### Attention-Gated Skip Connections
Skip carries **noisy** pre-DWT features. Attention gate filters noise before decoder concat:
`Out = σ(ψ(ReLU(Wg·g + Wx·x))) × x`
where g = decoder gating signal (clean context), x = encoder skip (noisy).
→ noAttn ablation: gNCC collapses from 0.4366 → **0.1014**

#### Decoder (Linear default / IDWT variant)
Per level: `Upsample×2 → ½Conv+BN+GELU → Interpolate → Concat(AG_skip) → 2×Conv+BN+GELU`

#### Dual Output Heads
- **Signal**: Conv 32→16→1, linear activation → denoised waveform (B,1,2048)
- **Segmentation**: dual-stream fusion → 5-class logits (B,5,2048)
  - BG · Corona · Surface · Internal · Treeing

---

## COLUMN 3 — METHODOLOGY & TRAINING

### Dataset
| Split | Content | Size |
|---|---|---|
| Synthetic train | 6 DOP archetypes (Types A–F) | 80k windows |
| Synthetic test (frozen) | 7 SNR levels × 2000 windows | 14,000 |
| Real Q.Lin test (grounded) | 4 gap sizes, 7 SNR levels | 35,000 |

SHA-256 frozen test sets — no leakage possible.

### Multi-Phase Curriculum Training (NoMix — Best Config)

| Phase | Noise | SNR Range | Epochs |
|---|---|---|---|
| 1 | Gaussian only | +5 to +15 dB | 20 |
| 2 | +Impulsive | −5 to +5 dB | 30 |
| 3 | Full composite | −20 to −5 dB | 50 |

**NoMix** = phased curriculum without mixed-type augmentation windows, forcing purer discharge-type signature learning. This is the **overall best** configuration.

[IMAGE: fig09_curriculum.png — Curriculum training loss curves across 3 phases]
**Fig. 3.** Three-phase curriculum training loss curves.

### Loss Configuration A (primary — best)
`L = w₁·L_Charb + w₂·L_Focal + w₃·L_GenDice`
Weights w₁,w₂,w₃ = learned homoscedastic uncertainty weighting (Kendall 2018).
- Charbonnier: robust to spiky PD pulses vs. MSE
- Focal: handles severe PD vs. background class imbalance
- Generalized Dice: end-to-end segmentation quality

Optimiser: AdamW (lr=3×10⁻⁴, wd=10⁻⁴), cosine annealing, grad clip norm=1.0, AMP.

---

## COLUMN 4 — RESULTS

### Best Model: MR-TAE (NoMix Curriculum)

| Metric | MR-TAE NoMix | BayesShrink | Improvement |
|---|---|---|---|
| NCC (mean) | **0.8282** | 0.6578 | +25.9% |
| NCC @ −15 dB | **0.3777** | 0.1575 | +2.39× |
| SNR improvement | **9.64 dB** | 2.73 dB | +6.91 dB |
| Dice (5 classes) | **0.9192** | N/A | — |
| gNCC (real, no FT) | **0.4535** | 0.3844 | **+17.9%** |

[IMAGE: fig_gncc_comparison.png — gNCC bar chart: NoMix 0.4535 vs BayesShrink 0.3844 (+17.9%)]
**Fig. 4.** Real-domain gNCC (no fine-tuning): MR-TAE NoMix vs classical baselines.

[IMAGE: fig02_snr_bar.png — Per-SNR NCC bar chart comparing NoMix vs baselines]
**Fig. 5.** Per-SNR NCC: MR-TAE NoMix vs classical and DL baselines.

[IMAGE: fig03_snr_sweep.png — NCC vs SNR sweep −15 to +15 dB]
**Fig. 6.** NCC vs input SNR. Learned models cross above all classical methods below 0 dB.

### Component Ablation (key findings)

| Removed | NCC drop | gNCC drop | Insight |
|---|---|---|---|
| Wavelet encoder | −0.1688 | −0.5703→neg | Most critical: lossless freq. decomposition |
| MTL seg head | −0.1509 | −0.5971→neg | Forces morphology preservation |
| BiGRU | −0.1427 | −0.2059 | Temporal dynamics anchor |
| Swin-1D | −0.1359 | −0.1079 | Long-range context |
| Attention gate | −0.1223 | −0.3352 | Skip noise suppression |

[IMAGE: fig04_ablation_heatmap.png — Ablation NCC heatmap across SNR levels]
**Fig. 6.** Ablation NCC heatmap. Red = worst, green = best.

**Key finding:** Removing both BiGRU+Swin together (row 7) costs only −0.0017 NCC — either component alone provides sufficient sequence-modelling capacity. For latency-constrained deployment, MR-TAE-noSwin-noAttn achieves NCC 0.7998 at **8.75 ms**.

### Qualitative Waveform Recovery

[IMAGE: fig10_qualitative_waveforms.png — Noisy / MR-TAE / BayesShrink / Clean at −15 dB]
**Fig. 7.** Waveform comparison at −15 dB SNR. MR-TAE recovers pulse morphology; BayesShrink blurs it.

---

## COLUMN 5 — EXPLAINABILITY, DISCUSSION & CONCLUSIONS

### Explainability (LIME + SHAP)

[IMAGE: fig11_ablation_grid.png — LIME attribution panels for MR-TAE-FULL]
**Fig. 8.** LIME attributions at −15 dB. High attribution concentrates on PD transient samples, not background.

- LIME peak importance: **0.2342** (peak-to-mean 1.13×) — pulse-focused, not diffuse
- SHAP mean absolute: **0.001405** — concentrated on mid-window PD transients
- Component importance rank (LIME) = ablation NCC rank: wavelet → MTL → BiGRU → Swin → AG

### Efficiency Trade-off

[IMAGE: fig06_efficiency_scatter.png — Latency vs NCC scatter, bubble = params]
**Fig. 9.** Efficiency scatter. Pareto frontier: MWCNN-BiGRU (lightest) → noSwin-noAttn (fastest full-quality).

### Discharge-Type Robustness
Dice **0.9192** across 5 classes at −15 dB SNR confirms morphological feature preservation.
Practitioners get denoising + fault-type classification in a single forward pass — no separate classifier needed.

### Real-Domain Performance (No Fine-Tuning)
MR-TAE NoMix gNCC **0.4535** exceeds BayesShrink **0.3844** by **+17.9%** on the grounded Q.Lin test set.
Classical baselines (finalized): Wiener 0.3641, Savitzky-Golay 0.2915.

### Conclusions
✅ MR-TAE (NoMix) achieves **NCC 0.8282**, SNR improvement **9.64 dB**, Dice **0.9192** — best on all synthetic metrics.
✅ Real gNCC **0.4535** (+17.9% over BayesShrink 0.3844) without fine-tuning.
✅ Wavelet lossless downsampling + MTL segmentation are the two pillars of robustness.
✅ Joint denoising + segmentation: removing seg head collapses real-world gNCC to **−0.1605**.
✅ IDWT decoder variant: equivalent quality at 2.92M params (vs 3.05M linear).

### Future Work
- WGAN-GP augmentation for real noise distribution matching
- Noise2Void self-supervised adaptation on unlabelled field data
- INT8 quantisation for NXP i.MX 8M Plus edge deployment (<5 ms target)
- Multi-sensor fusion for PD source localisation

---

## REFERENCES (abbreviated)
[1] IEC 60270 · [3] Q. Lin et al., IEEE Trans. Dielectr. 2022 · [4] Donoho & Johnstone 1994 · [8] Zhang et al., DnCNN 2017 · [9] Ronneberger et al., U-Net 2015 · [10] Liu et al., MWCNN 2018 · [13] Liu et al., Swin 2021 · [15] Kendall et al., Uncertainty 2018 · [16] Ribeiro et al., LIME 2016 · [17] Lundberg & Lee, SHAP 2017

---
<!-- END POSTER CONTENT -->
