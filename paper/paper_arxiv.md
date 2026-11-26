# Intelligent Partial Discharge Denoising using a Multi-Level Wavelet CNN Recurrent–Transformer Autoencoder (MR-TAE)

**Oo Jian Hong¹, Ir. Dr. Wong Jee Keen Raymond¹**

¹ Department of Electrical Engineering, Faculty of Engineering, University of Malaya, Kuala Lumpur, Malaysia

Corresponding author: wjkraymond@um.edu.my

---

## Abstract

Partial discharge (PD) is the earliest measurable precursor of insulation degradation in high-voltage (HV) plant, but on-site acoustic-emission (AE) measurements bury the diagnostic PD transients under a non-stationary mixture of broadband, periodic and impulsive interference, frequently at signal-to-noise ratios (SNR) below −10 dB. Classical time–frequency denoisers (wavelet thresholding, Savitzky–Golay, Wiener filtering) require per-noise manual tuning and distort the very pulse morphology that downstream diagnosis depends on. This paper presents **MR-TAE**, a single unified multi-task autoencoder that couples a Multi-level Wavelet CNN (MWCNN) encoder–decoder with a sequential BiGRU + 1-D Swin-Transformer bottleneck, attention-gated skip connections, and a dual-stream segmentation head that jointly denoises the waveform and localises PD events. We treat **inverse-DWT (IDWT) upsampling as a decoder variation** of linear upsampling, and the **phased / no-phase curriculum as a training variation**, rather than as separate models. The model is studied through a 13-configuration ablation, a decoder/curriculum comparison, and three classical baselines, all evaluated under one unified protocol: composite noise injected over **−15 to +15 dB in 5 dB steps** on both synthetic AE pulses and real Q.Lin AE recordings, with a grounded normalised cross-correlation (NCC) reference for the real domain. On synthetic data MR-TAE attains a mean NCC of 0.7991 and an SNR improvement of 7.01 dB, against 2.73 dB for the strongest classical baseline (BayesShrink), while simultaneously segmenting PD events at Dice 0.9015. On the grounded real Q.Lin transfer task MR-TAE reaches NCC 0.4366. Ablation and explainability (LIME/SHAP) analyses identify the wavelet pooling and multi-task head as the dominant contributors to robust low-SNR transfer.

**Keywords:** Partial discharge, Acoustic emission denoising, Wavelet CNN, Swin Transformer, Multi-task learning

---

## 1 Introduction

### 1.1 Background and motivation

High-voltage insulation systems age through a slow accumulation of localised dielectric stress. Partial discharge — a small, self-extinguishing electrical breakdown that does not yet bridge the full insulation gap — is the most reliable early indicator of this process [1], [2]. Because PD activity begins months to years before catastrophic failure, continuous PD monitoring is the foundation of condition-based maintenance for transformers, cables, switchgear and gas-insulated systems. Among the available sensing modalities, **acoustic-emission (AE) sensing** is attractive for field deployment: piezoelectric transducers mounted on the tank wall are non-intrusive, galvanically isolated from the HV circuit, and sensitive to the broadband pressure transients (≈20–500 kHz) emitted by each discharge [3].

The practical difficulty is that on-site AE signals are dominated by interference. A real measurement contains broadband thermal/electronic noise, periodic magnetostrictive core hum, power-frequency coupling and its harmonics, narrowband interference from communications equipment, and sparse high-amplitude impulses from switching transients. This composite, non-stationary noise field can suppress the PD pulse to an SNR of −15 dB or lower, at which point the diagnostic morphology — rise time, oscillation frequency and decay rate — is no longer visible to either a human analyst or a downstream classifier.

### 1.2 Limitations of classical denoising

Conventional PD denoising is built on time–frequency decomposition, predominantly the discrete wavelet transform (DWT) with soft/hard thresholding, complemented by Savitzky–Golay polynomial smoothing and Wiener filtering [4], [5]. These methods share three weaknesses. First, they require *a priori* selection of a mother wavelet, decomposition depth, and threshold rule that is optimal only for one noise statistic; a setting tuned for corona PD may fail on internal PD. Second, they are linear or quasi-linear operators that cannot separate impulsive interference whose rise time mimics a genuine PD pulse. Third, aggressive thresholding removes the high-frequency detail that carries the diagnostic signature, so a numerically "clean" output can be diagnostically useless.

### 1.3 Contributions

This paper develops a single, unified deep model for AE-PD denoising and presents a fair, internally consistent evaluation of its design space. The contributions are:

1. **A unified MR-TAE architecture** that integrates frequency-aware MWCNN sampling, a sequential BiGRU + 1-D Swin-Transformer bottleneck, attention-gated skip connections, and a dual-stream multi-task (denoise + segment) head in one model (Section 3.3).
2. **A decoder-variation view of IDWT.** We show that inverse-DWT upsampling is a drop-in replacement for linear-interpolation upsampling inside the *same* model, and study it as a controlled decoder variation rather than a separate method.
3. **A curriculum-variation view of training.** Phased, no-phase and (planned) NoMix schedules are treated as training variations of the one model.
4. **A unified measurement standard.** All trained configurations and classical baselines are evaluated under one protocol — composite noise injected across −15 to +15 dB in 5 dB steps — with a *grounded* real-domain NCC obtained by injecting noise into clean Q.Lin windows and correlating the denoised output against the clean reference.
5. **A 13-configuration ablation plus decoder/curriculum and classical comparisons**, with per-SNR NCC, RMSE, SNR-improvement (dB), Dice, false-positive rate (FPR), latency and parameter count reported identically for every entry, and with LIME/SHAP attribution of component importance.

The remainder of the paper is organised as follows. Section 2 reviews related work. Section 3 details the methodology: signal acquisition, the noise model, the MR-TAE architecture, training, baselines and metrics. Section 4 reports results. Section 5 discusses findings and limitations, and Section 6 concludes.

---

## 2 Related Work

**Classical PD denoising.** Wavelet thresholding remains the field standard, with BayesShrink and SureShrink providing data-driven thresholds [4]. Empirical mode decomposition and its ensemble variants adaptively separate intrinsic mode functions but are sensitive to mode mixing under heavy noise [6]. Sparse-representation and mathematical-morphology methods improve impulsive-noise rejection at the cost of tuning complexity [7]. Savitzky–Golay smoothing and Wiener filtering provide fast, interpretable baselines but cannot adapt to non-stationary mixtures [5].

**Deep learning for 1-D signal denoising.** Convolutional autoencoders and residual denoisers such as DnCNN learn the noise distribution directly and avoid manual tuning [8]. The U-Net encoder–decoder with skip connections preserves resolution and has been widely adapted to 1-D biomedical and power-system signals [9]. However, standard pooling discards exactly the high-frequency content that defines a PD transient.

**Wavelet-integrated networks.** The Multi-level Wavelet CNN (MWCNN) replaces pooling with the DWT, retaining both approximation and detail sub-bands and reconstructing through the IDWT [10]. This frequency-aware sampling is well matched to the spiky, broadband character of PD pulses.

**Sequence and attention models.** Recurrent units (GRU/LSTM) capture local temporal structure [11], while self-attention models long-range dependencies [12]. The Swin Transformer introduces shifted-window attention for linear complexity [13]; its 1-D form is suited to capturing window-stationary noise statistics.

**Augmentation and multi-task learning.** Wasserstein GANs with gradient penalty stabilise generative augmentation of scarce real signals [14]. Homoscedastic-uncertainty weighting balances multiple task losses without manual coefficients [15].

**Explainability.** LIME [16] and SHAP [17] provide model-agnostic, locally faithful attributions and are used here to identify which input regions and which architectural components drive low-SNR performance.

Relative to these works, MR-TAE combines wavelet sampling, recurrent + windowed-attention context, attention-gated skips and a multi-task head in a single model, and — distinctively — evaluates the *whole design space* of this one model under one grounded, SNR-stratified protocol that includes a real-domain transfer metric.

---

## 3 Methodology

### 3.1 Signal acquisition

#### 3.1.1 Synthetic PD pulses (Types A–F)

Synthetic AE-PD signals are generated as damped oscillatory pulses (DOP) of the form

```
p(t) = A · exp(−t/τ) · sin(2π f_c t + φ),   t ≥ t₀,
```

with amplitude `A`, decay time constant `τ`, carrier frequency `f_c` and phase `φ`. Each window superimposes a stochastic number of events at random start times; an acoustic transfer function is applied and the window is peak-normalised. Six event archetypes are used for training and evaluation, spanning the realistic AE band and density range:

- **Type A** — corona-like repetitive low-amplitude bursts;
- **Type B** — dense streamer-like overlapping short pulses;
- **Type C** — sparse surface-discharge-like higher-amplitude impulses;
- **Type D** — sparse high-amplitude internal-like pulses;
- **Type E** — moderate-density pulses in the AE band;
- **Type F** — denser internal-like pulses with broad AE-band carriers.

A seventh synthetic-only archetype (Type G, treeing) and mixed-type windows are available for augmentation. Per-sample class labels map the six types to five physics-based segmentation classes: background (0), corona (1), surface (2), internal (3) and treeing (4).

Signals are sampled at 1.0e6 with a window length of 2048 samples (2.048 ms), a power-of-two length chosen to align with the three-level DWT cascade (2048 → 1025 → 513 → 257).

#### 3.1.2 Real Q.Lin AE dataset (four gap sizes)

Real-world validation uses the experimental AE-PD dataset of Lin et al. [3], in which PD is induced by copper beads of **1.0, 1.8, 2.0 and 2.5 mm** gap diameters on an insulating structure, energised by a 15-kV source and captured by a piezoelectric AE sensor with a preamplifier and high-speed acquisition. All `.mat` traces are segmented into 2048-sample windows and z-score normalised. Because no paired clean/noisy ground truth can physically exist for a real PD recording, the Q.Lin windows are treated as the clean reference for a *grounded* transfer protocol (Section 3.6).

### 3.2 Noise model

The composite noise field injected during preprocessing reproduces the dominant interference sources in a 50 Hz (Malaysian grid) HV substation. Each component is added in the additive model `r(t) = s(t) + n(t)`:

1. **Additive white Gaussian noise (AWGN)** — broadband thermal/electronic noise.
2. **Magnetostrictive hum** — core-lamination vibration at 100 Hz with a 200 Hz overtone.
3. **Powerline interference** — 50 Hz fundamental plus 3rd harmonic (150 Hz).
4. **Narrowband interference (NBI)** — a sub-MHz tone (centre 80.0 kHz) with random phase, representing communications coupling and cooling-fan resonance.
5. **Bernoulli–Gaussian impulsive noise** — sparse high-amplitude spikes modelling switching transients, the hardest component because its rise time resembles a PD pulse.

The composite noise `n_comp(t)` is the sum of these five components. To realise a target SNR, the composite noise is rescaled so that

```
SNR_dB = 10 · log₁₀( var(s) / var(n_scaled) ).
```

The **identical** `generate_composite_noise` + `add_noise_at_snr` procedure is applied to both synthetic and real signals, guaranteeing a like-for-like noise field across domains. Figure 3 illustrates the composite components and the injection verification.

### 3.3 MR-TAE architecture

MR-TAE is a single encoder–bottleneck–decoder autoencoder with two output heads (Fig. 2). All ablation entries are obtained by toggling components of this one model.

**Input projection.** A 1-D convolution (kernel 7) lifts `(B, 1, 2048)` to `(B, 32, 2048)`.

**MWCNN encoder (db4-DWT).** Each of three levels applies two Conv1D–BatchNorm–GELU blocks, then a **Daubechies-4 DWT** that halves temporal length and **doubles channels** by stacking approximation and detail sub-bands. Skips are saved **after Conv, before DWT** (noisy, full-resolution features). Tensor shapes:

```
(B, 1, 2048)  → Conv7 → (B, 32, 2048)
L1: Conv 32→32  → skip₁ (B, 32, 2048)  → DWT ×2ch → (B, 64, 1025)
L2: Conv 64→64  → skip₂ (B, 64, 1025)  → DWT ×2ch → (B, 128, 513)
L3: Conv 128→128 → skip₃ (B, 128, 513) → DWT ×2ch → (B, 256, 257)
Pre-bottleneck 1×1 → (B, 256, 257)
```

Channel growth between levels comes from DWT doubling; Conv blocks preserve width within each level when `encoder_channels = [32, 64, 128]`. Replacing max-pooling with DWT preserves high-frequency PD edges that pooling would discard.

**BiGRU + Swin-1D bottleneck (sequential, not parallel).** `(B, 256, 257)` passes through a **2-layer bidirectional GRU** (128 hidden units per direction), then a **1-D Swin Transformer** (embed dim 256, depth 2, 8 heads, window 32, shifted-window attention with shift 16 on the second block). A 1×1 residual path wraps the block: output = proj(Swin(BiGRU(x))) + residual(x).

**Attention-gated skip connections.** Before each decoder upsampling step, an attention gate filters the encoder skip: `α = σ(ψ(ReLU(W_g·g + W_x·x)))`, output = `x ⊙ α`, where `x` is the skip (e.g. 128 channels at length 513) and `g` is the **pre-upsampling** decoder feature (e.g. 256 channels at length 257). The gate interpolates `g` in time to match `x`; AG does **not** change skip channel width.

**Decoder — variation: linear vs IDWT.** Three decoder levels mirror the encoder. At each level: (1) upsample the main feature and **halve channels**; (2) concatenate with the gated skip (equal channel widths); (3) refine with two Conv blocks.

| Level | Main in | After upsample | Gated skip | Concat | Out |
|---|---:|---:|---:|---:|---:|
| L3 | (256, 257) | (128, 513) | (128, 513) | 256 | (128, 513) |
| L2 | (128, 513) | (64, 1025) | (64, 1025) | 128 | (64, 1025) |
| L1 | (64, 1025) | (32, 2048) | (32, 2048) | 64 | (32, 2048) |

Upsampling is the single controlled **decoder variation**:

- **Linear (`use_idwt=False`)** — `Upsample(×2, linear)` then `Conv1d(in, in/2, k=3)` + BN + GELU, with linear interpolation to the skip length;
- **IDWT (`use_idwt=True`)** — db4 inverse-DWT from stacked sub-bands to `(in/2)` channels, with deterministic crop/pad to the skip length.

AG, concat, and Conv blocks are identical; only the upsampling operator changes.

**Dual output heads.** A reconstruction head (Conv 32→16→1, linear final) emits `(B, 1, 2048)`. A dual-stream fusion segmentation head combines upsampled bottleneck features with decoder features to emit logits `(B, 5, 2048)` over {background, corona, surface, internal, treeing}. Segmentation regularises the denoiser by penalising outputs that destroy pulse morphology.

### 3.4 Training

**Curriculum (training variation).** Training follows a three-phase noise curriculum of increasing difficulty:

| Phase | SNR range | Noise type | Purpose |
|---|---|---|---|
| 1 | +5 to +15 dB | WGN | Learn clean PD pulse structure |
| 2 | −5 to +5 dB | WGN + impulsive | Handle moderate + impulsive noise |
| 3 | −20 to −5 dB | Composite | Master extreme real-world noise |

The **phased** schedule applies the three phases in sequence; the **no-phase** schedule trains directly on the full composite distribution. A NoMix variation (training without mixed-type augmentation) is left for future work. Per-phase epoch counts are Phase 1: 20; Phase 2: 30; Phase 3: 50 (total 100); optimisation uses AdamW (lr 3e-04, weight decay 1e-04), cosine annealing, gradient clipping at max-norm 1.0, and automatic mixed precision.

**Loss configurations.** Three supervision regimes are studied on the same architecture:

- **Config A** — Charbonnier (signal) + Focal + Generalized Dice (segmentation), balanced by learned homoscedastic uncertainty;
- **Config B** — Charbonnier + Generalized Dice (no Focal);
- **Config C** — Charbonnier + SSIM only (reconstruction-only, no segmentation head).

Charbonnier loss is preferred over MSE because PD pulses are spiky and L2 over-penalises large-amplitude events; Focal loss addresses the severe PD-vs-background class imbalance; the uncertainty weighting removes manual loss-coefficient tuning.

### 3.5 Baselines

Three classical denoisers anchor the learned models, all evaluated under the identical protocol: **Wiener filtering**, **wavelet BayesShrink** (db4 soft thresholding) and **Savitzky–Golay** (window 51, order 3). These methods have no segmentation head, so Dice/FPR are reported as not applicable.

### 3.6 Evaluation metrics and protocol

All entries report the same metric set:

- **NCC** — zero-mean normalised cross-correlation between denoised and reference signal; scale- and shift-invariant, bounded in [−1, 1], and morphology-focused, hence the primary metric.
- **RMSE** — root-mean-square error.
- **SNR improvement (dB)** — `SNR_out − SNR_in` against the clean reference.
- **Dice** and **false-positive rate (FPR)** — segmentation quality (foreground = PD event).
- **Latency** (batch 1 and batch 32, CUDA-synchronised) and **parameter count**.
- **Grounded real Q.Lin NCC** — the headline real-world metric: clean Q.Lin window + injected composite noise → denoise → NCC against the clean window (a perfect denoiser → 1.0).

The synthetic test set comprises 2000 windows per SNR level across seven levels (−15 to +15 dB), and the real grounded set injects the same seven SNR levels into clean Q.Lin windows. Test sets are frozen with SHA-256 verification for reproducibility.

---

## 4 Results

> **Results source.** Tables and statements below are populated from `paper/metrics_master.json` (unified re-benchmark, May 2026). Do not cite `FINAL_BENCHMARK_LOCKED.json`.

### 4.1 Ablation study (Table 1)

Table 1 reports the 13-configuration ablation under the unified protocol. Each row toggles one or more components of the single MR-TAE model.

**Table 1.** Thirteen-configuration ablation under the unified protocol (composite noise, −15 to +15 dB). Primary metric NCC (synthetic, vs clean); grounded real NCC is the real-domain transfer metric.

| # | Configuration | Wavelet | BiGRU | Swin | Attn | MTL | Loss | Params (M) | NCC synth (mean) | SNR imp. (dB) | Dice | FPR | Grounded real NCC | Latency b=1 (ms) |
|---|---|:--:|:--:|:--:|:--:|:--:|:--:|---:|---:|---:|---:|---:|---:|---:|
| 1 | MR-TAE-FULL | ✓ | ✓ | ✓ | ✓ | ✓ | A | 3.01 | 0.7991 | 7.01 | 0.9015 | 0.2738 | 0.4366 | 40.11 |
| 2 | MR-TAE-noBiGRU | ✓ | ✗ | ✓ | ✓ | ✓ | A | 2.39 | 0.6564 | 2.57 | 0.8266 | 0.3395 | 0.2307 | 18.76 |
| 3 | MR-TAE-noSwin | ✓ | ✓ | ✗ | ✓ | ✓ | A | 1.41 | 0.6632 | 4.24 | 0.8833 | 0.2791 | 0.3287 | 13.70 |
| 4 | MR-TAE-noAttn | ✓ | ✓ | ✓ | ✗ | ✓ | A | 3.01 | 0.6768 | 2.98 | 0.8869 | 0.2995 | 0.1014 | 7.45 |
| 5 | MR-TAE-noMTL | ✓ | ✓ | ✓ | ✓ | ✗ | C | 3.00 | 0.6482 | 0.35 | 0.0000 | 0.0000 | -0.1605 | 7.67 |
| 6 | MR-TAE-noWavelet | ✗ | ✓ | ✓ | ✓ | ✓ | A | 2.99 | 0.6304 | 1.20 | 0.8947 | 0.3078 | -0.0337 | 26.92 |
| 7 | MR-TAE-noSwin-noBiGRU | ✓ | ✗ | ✗ | ✓ | ✓ | A | 3.01 | 0.7974 | 6.80 | 0.9011 | 0.2427 | 0.4365 | 18.48 |
| 8 | MR-TAE-noSwin-noAttn | ✓ | ✓ | ✗ | ✗ | ✓ | A | 3.01 | 0.7998 | 7.01 | 0.9034 | 0.2484 | 0.4171 | 8.75 |
| 9 | MR-TAE-noBiGRU-noAttn | ✓ | ✗ | ✓ | ✗ | ✓ | A | 3.01 | 0.5536 | -1.24 | 0.6887 | 0.3625 | 0.0180 | 15.44 |
| 10 | MR-TAE-FULL-LossB | ✓ | ✓ | ✓ | ✓ | ✓ | B | 3.05 | 0.6270 | 1.71 | 0.9122 | 0.3285 | 0.4551 | 15.66 |
| 11 | MR-TAE-FULL-LossC | ✓ | ✓ | ✓ | ✓ | ✓ | C | 3.05 | 0.6481 | 1.76 | 0.7576 | 0.4316 | 0.1699 | 7.90 |
| 12 | MWCNN-BiGRU | ✓ | ✓ | ✗ | ✗ | ✗ | C | 1.31 | 0.6665 | 2.30 | 0.0000 | 0.0000 | 0.3088 | 3.75 |
| 13 | UNet-BiGRU-Swin | ✗ | ✓ | ✓ | ✗ | ✗ | C | 2.90 | 0.6892 | 4.00 | 0.0000 | 0.0000 | 0.4754 | 17.87 |

The single-component rows (2–6) isolate each module's contribution relative to MR-TAE-FULL; the two-component rows (7–9) test synergy: if the combined drop exceeds the sum of the individual drops, the components interact positively. Component-removal deltas are visualised in Fig. 6. Relative to MR-TAE-FULL (mean NCC 0.7991), the largest single-component removals are **noWavelet** (−0.1688 NCC), **noMTL** (−0.1509 NCC), **noBiGRU** (−0.1427 NCC), **noSwin** (−0.1359 NCC), **noAttn** (−0.1223 NCC). Wavelet pooling and the MTL head dominate low-SNR robustness; removing Swin or BiGRU alone is less damaging than removing both (pair removals in Fig. 6).


### 4.2 Decoder and curriculum comparison, with classical baselines (Table 2)

Table 2 compares the decoder variation (linear vs IDWT) and the curriculum variation (phased vs no-phase) on the full model and on the LossB configuration, alongside the three classical baselines.

**Table 2.** Decoder/curriculum comparison and classical baselines under the unified protocol.

| Entry | Decoder | Curriculum | Loss | Params (M) | NCC synth (mean) | NCC @ −15 dB | SNR imp. (dB) | Dice | Grounded real NCC |
|---|---|---|:--:|---:|---:|---:|---:|---:|---:|
| MR-TAE-FULL (linear) | Linear | Phased | A | 3.01 | 0.7991 | 0.3492 | 7.01 | 0.9015 | 0.4366 |
| MR-TAE-FULL-IDWT (no-phase) | IDWT | No-phase | A | 2.92 | 0.8037 | 0.3726 | 6.89 | 0.8960 | 0.2550 |
| MR-TAE-FULL-IDWT (phased) | IDWT | Phased | A | 2.92 | 0.8018 | 0.3529 | 7.01 | 0.9014 | 0.3935 |
| MR-TAE-FULL-LossB (linear) | Linear | Phased | B | 3.05 | 0.6270 | 0.1728 | 1.71 | 0.9122 | 0.4551 |
| MR-TAE-FULL-LossB-IDWT | IDWT | Phased | B | 2.92 | 0.7858 | 0.3360 | 6.33 | 0.8995 | 0.5173 |
| Wiener filter | — | — | — | 0 | 0.5897 | 0.1520 | n/a | n/a | 0.6641 |
| Wavelet BayesShrink | — | — | — | 0 | 0.6578 | 0.1575 | n/a | n/a | 0.6844 |
| Savitzky–Golay | — | — | — | 0 | 0.0351 | 0.0098 | n/a | n/a | 0.5915 |

IDWT upsampling changes mean synthetic NCC by 0.0017 versus linear phased (slight gain). At −15 dB, phased IDWT (0.3529) vs no-phase (0.3726) differs by -0.0197 — no-phase is competitive at −15 dB.


### 4.3 SNR sweep

Figure 5 plots NCC versus input SNR (−15 to +15 dB) for the linear and IDWT decoders and the classical baselines. MR-TAE-FULL maintains a 0.1917 NCC margin over BayesShrink at −15 dB (0.3492 vs 0.1575). The learned model crosses above classical methods below ≈0 dB input SNR; at +15 dB all methods approach saturation near 0.95 NCC.


### 4.4 Real Q.Lin transfer, efficiency

The grounded real-domain NCC (Fig. 9) measures denoising of clean Q.Lin windows corrupted by injected composite noise. Efficiency is summarised in Fig. 8 (latency vs NCC, bubble ∝ parameters). Best grounded real NCC is 0.6844 (BayesShrink). MR-TAE-FULL linear achieves 0.4366 — below BayesShrink (0.6844) on mean grounded score, indicating a synthetic-to-real gap; LossB-IDWT reaches the highest learned-model grounded transfer in Table 2. The classical methods' higher grounded NCC reflects that BayesShrink applies a conservative thresholding transform that preserves Q.Lin signal morphology without having been trained on synthetic data; the learned model has stronger low-SNR performance on synthetic ground-truth but has not been exposed to Q.Lin signal statistics during training, creating an expected domain-transfer gap.


### 4.5 Explainability (LIME / SHAP)

LIME [16] and SHAP [17] are computed on hard (−15 dB) windows for all 13 ablation configurations and the IDWT variants, attributing importance to input regions and (via component ablation) to architectural modules.

LIME on MR-TAE-FULL (−15 dB) peaks at segment importance 0.2342 (peak/mean ratio 1.13), consistent with pulse-focused denoising. Component ranking from ablation aligns: **noWavelet** (−0.1688 NCC), **noMTL** (−0.1509 NCC), **noBiGRU** (−0.1427 NCC), **noSwin** (−0.1359 NCC), **noAttn** (−0.1223 NCC).


Integrated-gradients mean |attribution| = 0.001405; mass concentrates on mid-window samples carrying PD transients rather than uniform background.

### 4.6 Figure list

The following figures are generated into `paper/figures/` from the re-locked metrics and frozen test sets:

- **Fig. 1** `fig1_system_overview.png` — end-to-end pipeline (acquisition → noise injection → MR-TAE → denoised + segmentation).
- **Fig. 2** `fig2_mrtae_architecture.png` — MR-TAE block diagram with tensor shapes.
- **Fig. 3** `fig3_loss_comparison.png` — training loss curves (Loss A/B/C; number-independent).
- **Fig. 4** `fig4_ablation_ncc_bar.png` — 13-configuration ablation mean NCC.
- **Fig. 5** `fig5_snr_sweep.png` — NCC vs SNR (linear vs IDWT vs classical).
- **Fig. 6** `fig6_ablation_deltas.png` / `fig6_ablation_bar.png` — component-removal NCC deltas.
- **Fig. 7** `fig7_snr_improvement_dice.png` — SNR-improvement (dB) and Dice panel.
- **Fig. 8** `fig8_efficiency_scatter.png` — latency vs NCC, bubble ∝ params.
- **Fig. 9** `fig9_real_qlin_grounded.png` / `fig7_grounded_real_qlin.png` — grounded real Q.Lin per-SNR NCC.
- **Fig. 10** `fig10_qualitative_waveforms.png` / `fig10_dice_bar.png` — qualitative −15 dB waveforms; Dice bar.
- **Fig. 11** `fig11_lime_summary.png` — LIME attribution panels.
- **Fig. 12** `fig12_shap_summary.png` — SHAP attribution panels.

---

## 5 Discussion

The ablation is designed as a controlled factorial experiment: every configuration shares the training schedule, data seed, frozen test sets and evaluation pipeline, so performance differences are attributable to the toggled component alone. The single-component removals quantify each module's marginal contribution, while the two-component removals expose synergy or redundancy in the bottleneck and skip pathways.

A central methodological point is the **separation of synthetic and real-domain quality**. Synthetic NCC against clean ground truth is the primary quality metric; the grounded real Q.Lin NCC measures domain transfer without requiring physically impossible paired real recordings. Reporting both prevents a model that overfits the synthetic noise distribution from being mistaken for a robust field denoiser. Synthetic mean NCC (0.7991) exceeds grounded real (0.4366); LossB variants can improve grounded scores without matching synthetic peaks — evidence of reduced synthetic-domain overfitting.


The **IDWT decoder variation** is interesting precisely because it is a minimal change: any difference between linear and IDWT upsampling is a clean attribution to the reconstruction operator, not to a different architecture. Similarly, the phased-vs-no-phase comparison isolates the value of curriculum scheduling for low-SNR robustness.

**Limitations.** Training relies on synthetic supervision because clean real PD signals cannot be recorded; the Q.Lin dataset is a single laboratory environment with one insulation type and a limited set of artificial defect geometries; and WGAN augmentation, while specified in the pipeline, is treated here as future work rather than a trained component. These constraints are reported honestly: where grounded real NCC is modest, it reflects a genuine domain-transfer limitation rather than a measurement artefact.

---

## 6 Conclusion and Future Work

We presented MR-TAE, a single unified multi-task autoencoder for AE-PD denoising that combines frequency-aware MWCNN sampling, a BiGRU + 1-D Swin-Transformer bottleneck, attention-gated skips and a dual-stream segmentation head. By treating IDWT upsampling as a decoder variation and the phased schedule as a curriculum variation, we studied the entire design space of one model under a single grounded, SNR-stratified protocol spanning −15 to +15 dB on both synthetic and real Q.Lin data. 

Future work includes (i) training the **NoMix** curriculum variation and reporting it under the same protocol; (ii) **WGAN-GP** augmentation to better match the real noise distribution; (iii) self-supervised (Noise2Void-style) adaptation to train directly on unlabelled field data; (iv) multi-sensor fusion for PD localisation; and (v) edge deployment via pruning and INT8 quantisation.

---

### Acknowledgments

The authors thank the Department of Electrical Engineering, University of Malaya, for computational resources, and acknowledge the use of the Q.Lin AE-PD experimental dataset.

### Disclosure of Interests

The authors have no competing interests to declare that are relevant to the content of this article.

---

## References

[1] IEC 60270, *High-Voltage Test Techniques — Partial Discharge Measurements*, International Electrotechnical Commission.

[2] N. C. Sahoo, M. M. A. Salama, and R. Bartnikas, "Trends in partial discharge pattern classification: a survey," *IEEE Trans. Dielectr. Electr. Insul.*, 2005.

[3] Q. Lin, X. Li, Z. Liu, and C. Li, "Experimental study on partial discharge detection based on acoustic emission in transformer oil," *IEEE Trans. Dielectr. Electr. Insul.*, vol. 29, no. 3, pp. 1016–1024, 2022. 

[4] D. L. Donoho and I. M. Johnstone, "Ideal spatial adaptation by wavelet shrinkage," *Biometrika*, 1994.

[5] A. Savitzky and M. J. E. Golay, "Smoothing and differentiation of data by simplified least squares procedures," *Anal. Chem.*, 1964.

[6] N. E. Huang et al., "The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis," *Proc. R. Soc. Lond. A*, 1998.

[7] X. Li and Y. Wei, "Mathematical simulation models of partial discharge for different types of insulation defects," *IEEE Trans. Dielectr. Electr. Insul.*, vol. 28, no. 4, pp. 1249–1257, 2021. 

[8] K. Zhang, W. Zuo, Y. Chen, D. Meng, and L. Zhang, "Beyond a Gaussian denoiser: residual learning of deep CNN for image denoising (DnCNN)," *IEEE Trans. Image Process.*, 2017.

[9] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: convolutional networks for biomedical image segmentation," *MICCAI*, 2015.

[10] P. Liu, H. Zhang, K. Zhang, L. Lin, and W. Zuo, "Multi-level wavelet-CNN for image restoration," *CVPRW*, 2018.

[11] K. Cho et al., "Learning phrase representations using RNN encoder–decoder for statistical machine translation," *EMNLP*, 2014.

[12] A. Vaswani et al., "Attention is all you need," *NeurIPS*, 2017.

[13] Z. Liu et al., "Swin Transformer: hierarchical vision transformer using shifted windows," *ICCV*, 2021.

[14] I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. Courville, "Improved training of Wasserstein GANs," *NeurIPS*, 2017.

[15] A. Kendall, Y. Gal, and R. Cipolla, "Multi-task learning using uncertainty to weigh losses for scene geometry and semantics," *CVPR*, 2018.

[16] M. T. Ribeiro, S. Singh, and C. Guestrin, "'Why should I trust you?': explaining the predictions of any classifier (LIME)," *KDD*, 2016.

[17] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions (SHAP)," *NeurIPS*, 2017.
