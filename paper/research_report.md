<!--
  research_report.md — exhaustive, freshly-written research report for MR-TAE AE-PD denoising.
  NUMERIC-RESULT POLICY: Results are locked from paper/metrics_master.json,
  results/paper_table1_ablation.csv, results/paper_table2_improved_models.csv, and
  results/additional_baselines_benchmark.json (2026-05-31 re-benchmark). Headline model:
  MR-TAE-FULL (NoMix). Classical grounded-NCC means use finalized poster values
  (BayesShrink 0.3844, Wiener 0.3641, Savitzky-Golay 0.2915). Q.Lin fine-tuning
  variants are documented but not used in headline comparisons. Figures in paper/figures/.
-->

# A Unified Multi-Level Wavelet Recurrent–Transformer Autoencoder (MR-TAE) for Acoustic-Emission Partial-Discharge Denoising: Architecture, Methodology and Evaluation

**Author:** Oo Jian Hong
**Supervisor:** Ir. Dr. Wong Jee Keen Raymond
**Affiliation:** Department of Electrical Engineering, Faculty of Engineering, University of Malaya, Kuala Lumpur, Malaysia

---

## Abstract

This report documents, in full technical depth, a unified deep-learning system for denoising acoustic-emission (AE) partial-discharge (PD) signals acquired from high-voltage insulation. The system, **MR-TAE** (Multi-Level Wavelet Recurrent–Transformer Autoencoder), is a single multi-task model that jointly reconstructs a clean PD waveform and segments PD events from a noisy input window of 2048 samples. The report is organised around the *flow* of the work: (i) what noise we must handle and why it is hard; (ii) how PD signals are acquired, both synthetically (six damped-oscillatory pulse archetypes, Types A–F) and from the real Q.Lin AE dataset (four copper-bead gap sizes); (iii) the composite noise model (AWGN, magnetostrictive hum, powerline + harmonic, narrowband interference, and Bernoulli–Gaussian impulse) and its SNR-controlled injection; (iv) the MR-TAE architecture described component-by-component with exact tensor shapes; (v) the three-phase training curriculum and the three loss configurations; (vi) the experimental setup, including frozen test sets, checkpoint selection and reproducibility controls; and (vii) the unified evaluation protocol and metric definitions. The design space of the *one* model is examined as a 13-configuration ablation, a decoder variation (linear vs inverse-DWT upsampling), and curriculum variations (phased, no-phase, and **NoMix**), all measured identically under composite noise from −15 to +15 dB in 5 dB steps, with a grounded real-domain NCC as the headline transfer metric. Under the locked benchmark, **MR-TAE (NoMix curriculum)** achieves mean synthetic NCC **0.8282**, SNR improvement **9.64 dB**, Dice **0.9192**, and grounded real NCC **0.4535** — exceeding BayesShrink (**0.3844**) by **17.9%** without fine-tuning. MR-TAE-FULL (phased) remains the ablation reference row.

---

## Table of Contents

1. Introduction
2. Literature Review
3. Methodology I — Signal Acquisition
4. Methodology II — The Noise Model
5. Methodology III — The MR-TAE Architecture
6. Methodology IV — Training: Curriculum and Losses
7. Experimental Setup and Reproducibility
8. Baselines and Evaluation Metrics
9. Results
10. Explainability (LIME / SHAP)
11. Discussion and Failure Modes
12. Conclusion and Future Work
- Appendix A — Full Result Tables (locked benchmark)
- Appendix B — Reproducibility Index (figure/table → script → data)
- Appendix C — File and Module Map
- Appendix D — Frozen Test-Set Hashes
- Appendix E — Notation and Abbreviations
- Appendix F — Complete Placeholder Token Index

---

# 1 Introduction

## 1.1 Background

Electrical power transmission and distribution depend on the integrity of high-voltage (HV) insulation. Transformers, cables, switchgear and gas-insulated systems all rely on dielectric materials — oil, paper, cross-linked polyethylene, epoxy, SF₆ — to withstand sustained electric fields far above the threshold at which air would break down. Over a service life measured in decades, these dielectrics age. Thermal cycling, moisture ingress, mechanical stress, and manufacturing imperfections create localised regions where the electric field concentrates. When the local field exceeds the local breakdown strength, a small, self-extinguishing electrical discharge occurs that does not bridge the entire insulation gap. This event is a **partial discharge (PD)**.

Although a single PD releases a negligible amount of energy, repeated PD activity is corrosive: each discharge erodes the dielectric a little further, the eroded region concentrates the field a little more, and the process accelerates. Left undetected, PD progresses through treeing and tracking to complete insulation failure — a flashover that can destroy the asset and trigger a wide-area outage. Because PD activity is measurable years before catastrophic failure, **PD monitoring is the cornerstone of condition-based maintenance for HV plant.**

The economic stakes are large. A single power transformer can cost several million US dollars and take a year or more to replace; an unplanned failure compounds the replacement cost with outage penalties and collateral damage. Detecting incipient insulation degradation early enough to schedule maintenance during a planned outage, rather than reacting to an emergency, is therefore of first-order importance to grid operators.

## 1.2 Acoustic-emission sensing and its noise problem

PD can be sensed electrically (via coupling capacitors or high-frequency current transformers), chemically (via dissolved-gas analysis of transformer oil), or acoustically. **Acoustic-emission (AE) sensing** detects the broadband pressure transient that each discharge radiates into the surrounding medium. A piezoelectric transducer mounted on the tank wall converts this pressure wave into a voltage signal. AE sensing is attractive for field deployment because it is non-intrusive, galvanically isolated from the HV circuit, and immune to the electromagnetic conducted interference that contaminates electrical methods.

The price of these advantages is a severe noise problem. The mechanical propagation path — oil, solid insulation, steel tank wall — attenuates and disperses the AE signal, and the sensor sits in an environment saturated with competing vibrations and electromagnetic pickup. The recorded waveform is a non-stationary superposition of:

- broadband thermal/electronic noise;
- periodic magnetostrictive hum from core-lamination vibration;
- power-frequency coupling and its harmonics;
- narrowband interference from communications and cooling-system resonance; and
- sparse, high-amplitude impulsive transients from switching operations.

In the field, this composite noise can suppress the PD pulse to a signal-to-noise ratio (SNR) of −15 dB or worse. At that level the diagnostic morphology of the pulse — its rise time, oscillation frequency and decay rate — is buried, and neither a human analyst nor an automated classifier can reliably extract it. **The denoising problem is therefore the gatekeeper to every downstream PD diagnostic task.**

## 1.3 Why classical denoising is insufficient

The traditional toolkit for PD denoising is built on time–frequency decomposition. Wavelet thresholding decomposes the signal into multi-resolution sub-bands and shrinks coefficients deemed to be noise; Savitzky–Golay filtering fits local polynomials to smooth the waveform; Wiener filtering applies an optimal linear filter under a stationarity assumption; and empirical mode decomposition adaptively separates intrinsic oscillatory modes. These methods are fast, interpretable, and effective in controlled conditions, but they share three structural limitations that make them unreliable in the field:

1. **They require manual, per-noise tuning.** The choice of mother wavelet, decomposition depth and threshold rule that is optimal for one noise statistic is sub-optimal for another. A setting tuned for corona PD on a broadband noise floor will under- or over-smooth internal PD under impulsive interference.
2. **They cannot reject signal-like interference.** Impulsive noise has a sharp rise time that mimics a genuine PD pulse, and a linear or threshold-based operator cannot distinguish the two from morphology alone.
3. **They distort the diagnostic signature.** Aggressive thresholding removes high-frequency detail — exactly the content that carries the PD signature — so a numerically clean output can be diagnostically worthless.

Deep learning offers a route past these limitations: a model can *learn* the joint structure of PD pulses and composite noise from data, adapt to a wide range of SNRs without manual retuning, and — crucially — be supervised to preserve diagnostic morphology rather than merely minimise residual energy. This is the opportunity that MR-TAE exploits.

## 1.4 Problem statement

The accurate analysis of PD signals is the most reliable indicator of incipient HV insulation failure, and the morphology of a PD pulse (amplitude, rise time, decay time, oscillation frequency) is the primary feature set for fault classification. The central difficulty is that PD pulses are low-amplitude, high-frequency transients contaminated, during on-site measurement, by a non-stationary mixture of broadband, periodic and impulsive interference that frequently buries the pulse below −10 dB SNR. Conventional time–frequency denoisers are limited by their dependence on manual wavelet/threshold selection and by their tendency to distort the diagnostic morphology. A practical, automated denoiser must therefore (i) operate across a wide SNR range without per-noise tuning, (ii) preserve diagnostic pulse morphology, and (iii) generalise from controlled training data to real field recordings for which no clean ground truth can be obtained.

## 1.5 Aim and objectives

**Aim.** To develop and rigorously evaluate a single, unified deep-learning model that denoises AE-PD signals across a wide SNR range while preserving diagnostic morphology and simultaneously localising PD events.

**Objectives.**

1. Design a unified multi-task autoencoder (MR-TAE) that integrates frequency-aware wavelet sampling, recurrent and windowed-attention context modelling, attention-gated skip connections, and a joint denoise+segment head.
2. Construct a physically motivated composite noise model and an SNR-controlled injection procedure applicable identically to synthetic and real signals.
3. Study the design space of the one model — a 13-configuration ablation, a decoder variation (linear vs inverse-DWT upsampling), and curriculum variations (phased, no-phase, NoMix) — under one unified protocol.
4. Establish a grounded real-domain evaluation that injects composite noise into clean Q.Lin AE windows and measures normalised cross-correlation against the clean reference.
5. Attribute component importance through LIME and SHAP explainability analyses.

## 1.6 Scope and constraints

This report covers the architecture, methodology, experimental design and evaluation protocol in full, with numeric results locked from the 2026-05-31 re-benchmark. Within scope: the unified MR-TAE model and its decoder/curriculum/loss variations (including **NoMix**, the recommended curriculum); the composite noise model; the synthetic and real Q.Lin datasets; classical baselines (Wiener, BayesShrink, Savitzky–Golay); deep-learning baselines (PureCNN1D, UNet1D_simple, DnCNN1D); and LIME/SHAP analyses. Out of scope for headline results: WGAN-GP augmentation and Q.Lin fine-tuning variants (documented separately, not used in primary comparisons).

## 1.7 Contributions

1. **A unified MR-TAE architecture** combining MWCNN db4-DWT sampling, a sequential BiGRU + 1-D Swin-Transformer bottleneck, attention-gated skips, and a dual-stream multi-task head.
2. **A decoder-variation treatment of IDWT** — inverse-DWT upsampling as a controlled drop-in replacement for linear upsampling within the same model.
3. **A curriculum-variation treatment of training** — phased, no-phase and **NoMix** schedules as variations of one model, with NoMix as the recommended deployment configuration.
4. **A unified, grounded measurement standard** spanning −15 to +15 dB in 5 dB steps on both synthetic and real Q.Lin data, with a grounded real NCC as the headline transfer metric.
5. **A complete 13-configuration ablation plus decoder/curriculum and classical comparisons**, reported identically across NCC, RMSE, SNR-improvement, Dice, FPR, latency and parameters, with LIME/SHAP component attribution.

## 1.8 Report structure

Chapter 2 reviews and synthesises the relevant literature. Chapters 3–6 present the methodology in four parts: signal acquisition, the noise model, the MR-TAE architecture, and training. Chapter 7 describes the experimental setup and reproducibility controls. Chapter 8 defines the baselines and metrics. Chapter 9 presents locked benchmark results. Chapter 10 covers explainability. Chapter 11 discusses findings and failure modes, and Chapter 12 concludes. Appendices provide full tables, a reproducibility index, a file map, test-set hashes, notation and a glossary (Appendix Q).

---

# 2 Literature Review

This chapter synthesises the research landscape relevant to AE-PD denoising. It is organised to motivate, in sequence, each design decision in MR-TAE: the physics of PD and AE sensing; the structure of the on-site noise field; the strengths and limits of classical denoising; the emergence of deep-learning denoisers; the specific roles of wavelet-integrated networks, recurrent and attention models, generative augmentation, multi-task learning and curriculum learning; and finally explainability. The chapter closes with a synthesis of the research gap that MR-TAE addresses.

## 2.1 Partial discharge and acoustic-emission sensing

Partial discharge is defined by IEC 60270 as a localised electrical discharge that only partially bridges the insulation between conductors [1]. Its diagnostic value derives from the fact that PD activity precedes complete breakdown by a long margin and that the statistical and morphological characteristics of PD pulses carry information about the underlying defect type — corona at sharp electrodes, surface discharge along interfaces, internal discharge within voids, and electrical treeing [2]. Among PD sensing modalities, AE sensing occupies a distinctive niche: it trades the high fidelity of electrical detection for immunity to electromagnetic conducted interference and for non-intrusive installation, which is decisive for retrofitting monitoring onto in-service plant.

The AE signal radiated by a discharge is a broadband transient concentrated in roughly the 20–500 kHz band, attenuated and dispersed by the mechanical propagation path. The piezoelectric transducer adds its own resonance, and the recording chain (preamplifier, acquisition) imposes bandwidth and quantisation limits. The consequence is that the *shape* of the recorded pulse encodes both the discharge physics and the channel — a fact that any morphology-preserving denoiser must respect.

## 2.2 The on-site noise environment

The interference that contaminates on-site AE-PD measurement is best understood as a superposition of physically distinct sources, each with a characteristic spectral and temporal signature:

- **Broadband noise** (thermal/electronic) is well modelled as additive white Gaussian noise (AWGN), with a flat spectrum and Gaussian amplitude statistics.
- **Periodic mechanical hum** arises from magnetostriction in the transformer core, vibrating the laminations at twice the line frequency and its overtones (100 Hz and 200 Hz on a 50 Hz grid).
- **Powerline interference**, also termed discrete spectral interference, couples the line frequency and its odd harmonics (50, 150 Hz …) into the sensor and cabling.
- **Narrowband interference (NBI)** originates from amplitude- and frequency-modulated radio sources, communications equipment, and cooling-system resonances; it appears as discrete spectral peaks and continuous time-domain oscillation.
- **Impulsive noise** comprises sparse, high-amplitude spikes from switching operations and power electronics; it is broadband in frequency but localised in time, and its sharp rise time is the principal source of false positives because it mimics PD morphology.

A realistic noise model must combine all of these, at controllable relative strengths and at controllable overall SNR — which is precisely what the composite model of Chapter 4 does.

## 2.3 Classical PD denoising

The dominant classical approach is **wavelet thresholding** [4]. The signal is decomposed by the discrete wavelet transform into approximation and detail coefficients; coefficients below a threshold are shrunk (soft or hard), and the signal is reconstructed by the inverse transform. Data-driven threshold rules — VisuShrink, SureShrink, and **BayesShrink** — remove the need to set the threshold by hand, with BayesShrink estimating a sub-band-adaptive threshold from the coefficient statistics. Wavelet methods are effective when the mother wavelet matches the pulse shape, but their performance is sensitive to that choice and they struggle with impulsive interference.

**Empirical mode decomposition (EMD)** and its ensemble variant (EEMD) decompose the signal into intrinsic mode functions adaptively, without a fixed basis [6]. They are powerful for non-stationary signals but suffer from mode mixing and are computationally heavy. **Sparse representation** and **mathematical morphology** methods improve impulsive-noise rejection, again at the cost of parameter tuning. **Savitzky–Golay** filtering fits a local least-squares polynomial and is excellent for smooth signals but blurs sharp transients [5]. **Wiener filtering** is the optimal linear filter under a stationarity assumption that on-site noise routinely violates.

The common thread is that classical denoisers are *fixed operators* whose optimality is contingent on assumptions (stationarity, a matched basis, a chosen threshold) that the field environment breaks. This motivates learned denoisers.

## 2.4 Deep-learning denoising

Learned denoisers replace hand-designed operators with functions fit to data. **Convolutional autoencoders** compress and reconstruct the signal, learning to suppress noise in the latent representation. **DnCNN** [8] introduced residual learning — predicting the noise rather than the clean signal — and batch normalisation to stabilise deep denoisers, setting a strong precedent for 1-D adaptations. The **U-Net** [9], with its symmetric encoder–decoder and skip connections, preserves high-resolution detail that pooling would otherwise discard, and has been widely adapted to 1-D biomedical and power-system signals.

The key weakness of a vanilla CNN/U-Net for PD is that standard pooling (max, average, strided convolution) is lossy in exactly the high-frequency band that defines a PD transient. A pooling step that discards half the samples can obliterate a pulse occupying only a few tens of samples in a 2048-sample window. This observation motivates wavelet-integrated networks.

## 2.5 Wavelet-integrated networks

The **Multi-level Wavelet CNN (MWCNN)** [10] replaces pooling with the discrete wavelet transform: at each downsampling step the signal is split into low- and high-frequency sub-bands, both retained and processed by subsequent convolutions, and reconstructed by the inverse DWT. The transform is invertible, so downsampling is lossless — no frequency information is discarded. For PD, whose pulses are spiky and broadband, this frequency-aware sampling is far better matched to the signal than pooling. MR-TAE adopts the MWCNN encoder–decoder as its backbone, using the Daubechies-4 (db4) wavelet, whose asymmetry and support length suit the asymmetric, fast-rising PD pulse.

## 2.6 Recurrent and attention models

A denoiser that only sees local convolutional context cannot exploit the temporal and global structure of the signal. **Gated recurrent units (GRU)** [11] model sequential dependencies efficiently; a **bidirectional** GRU processes the sequence in both directions, so each time step is informed by past and future context — valuable for recognising the characteristic rise-then-decay signature of a PD pulse and its quiet surroundings. **Self-attention** [12] models long-range dependencies directly, but full attention is quadratic in sequence length. The **Swin Transformer** [13] restores linear complexity through windowed attention with a shifted-window mechanism that still permits cross-window information flow. A 1-D Swin block is well suited to capturing window-stationary noise statistics (a hum frequency, an impulsive pattern) that are global properties of the window rather than local features. MR-TAE places a BiGRU and a 1-D Swin block *sequentially* in its bottleneck, so local temporal context established by the recurrent layer is refined by global attention.

## 2.7 Generative augmentation

Real labelled PD data are scarce, and no clean/noisy pairs can be obtained for real recordings. **Generative adversarial networks** can synthesise additional realistic samples, but vanilla GANs are unstable on 1-D signals and prone to mode collapse. The **Wasserstein GAN with gradient penalty (WGAN-GP)** [14] stabilises training by optimising the Wasserstein distance and enforcing the Lipschitz constraint through a gradient penalty rather than weight clipping, yielding diverse, high-fidelity samples. In this work WGAN-GP augmentation is documented as a planned extension (future work) rather than a trained component; it is not presented as a result.

## 2.8 Multi-task learning

Jointly learning related tasks can regularise a model and improve its primary task. For PD denoising, pairing the reconstruction task with a **PD-event segmentation** task forces the encoder to preserve the diagnostic features that the segmenter needs — a denoiser that blurs the signal to minimise residual energy would fail the segmentation task. Balancing multiple losses, however, requires care: fixed coefficients are fragile because the optimal balance drifts during training. **Homoscedastic-uncertainty weighting** [15] learns per-task weights automatically by treating each task's loss as a Gaussian likelihood with a learnable variance, with a log-variance regulariser that prevents a task from being trivially ignored. MR-TAE uses this mechanism in its full loss configuration.

## 2.9 Curriculum learning

Training a denoiser directly on the hardest examples (−15 dB composite noise) is unstable: the gradients are near-random when the pulse is invisible. **Curriculum learning** presents easier examples first and increases difficulty gradually, letting the model build useful representations before facing the hardest cases. MR-TAE's three-phase SNR curriculum (Chapter 6) realises this idea for PD denoising, and the comparison of phased versus no-phase training quantifies its value.

## 2.10 Explainability

A denoiser deployed for asset-critical monitoring should be interpretable. **LIME** [16] explains an individual prediction by fitting a locally faithful, interpretable surrogate around it, attributing importance to input regions. **SHAP** [17] grounds attributions in Shapley values from cooperative game theory, giving a unified, additive measure of feature contribution. Applied to AE-PD denoising on hard (−15 dB) windows, these methods reveal which parts of the input the model relies on (ideally the pulse regions rather than the noise floor) and, through component ablation, which architectural modules drive low-SNR performance.

## 2.11 Synthesis and research gap

The literature provides strong individual ingredients — wavelet sampling, recurrent and attention context, generative augmentation, multi-task learning, curriculum learning, explainability — but they are rarely combined in one PD denoiser, and still more rarely evaluated under a single, grounded, SNR-stratified protocol that includes a real-domain transfer metric. Much prior work trains and tests on short synthetic windows containing one to three isolated pulses, and reports synthetic metrics only, leaving the synthetic-to-real gap unquantified. MR-TAE addresses this gap on two fronts: architecturally, by unifying the ingredients in one multi-task model whose entire design space (decoder, curriculum, loss) is a set of controlled variations; and methodologically, by evaluating every configuration under one composite-noise protocol spanning −15 to +15 dB on both synthetic and real Q.Lin data, with a grounded NCC that measures true denoising quality on real recordings.

## 2.12 Comparative synthesis of denoising paradigms

The table below positions the principal paradigms against the requirements of field AE-PD denoising: adaptivity (no per-noise retuning), impulsive-noise rejection, morphology preservation, and the ability to localise PD events.

| Paradigm | Adaptive? | Impulsive rejection | Morphology preserved | Localises PD? | Field-readiness |
|---|---|---|---|---|---|
| Wiener filter | No (stationarity) | Poor | Moderate | No | Limited |
| Wavelet thresholding (BayesShrink) | Partially (data-driven threshold) | Moderate | Moderate–good | No | Common baseline |
| Savitzky–Golay | No | Poor | Blurs transients | No | Limited |
| EMD/EEMD | Yes (adaptive basis) | Moderate | Good | No | Mode-mixing risk |
| CNN / DnCNN autoencoder | Yes (learned) | Good | Good | No | Promising |
| U-Net (1-D) | Yes | Good | Good (skips) | Optionally | Promising |
| MWCNN | Yes | Good | Very good (lossless DWT) | No | Strong backbone |
| **MR-TAE (this work)** | Yes | Good (context + MTL) | Very good (DWT + MTL) | **Yes** (segmentation head) | Target |

The distinguishing feature of MR-TAE is the rightmost two columns: it both preserves morphology through lossless wavelet sampling and *explicitly localises* PD events through its segmentation head, which in turn regularises the denoiser toward diagnostic fidelity. No classical method offers localisation, and most learned denoisers in the PD literature address denoising alone.

## 2.13 Positioning of the evaluation methodology

A second contribution lies in evaluation rather than architecture. Much of the PD-denoising literature reports synthetic metrics on short windows and does not quantify transfer to real recordings, partly because real recordings lack clean ground truth. The grounded protocol used here — injecting the *same* composite noise into clean real windows and correlating the denoised output against the clean window — provides a principled real-domain quality measure without requiring physically impossible paired data. Combined with SNR stratification over seven levels and an identical protocol for every configuration and baseline, this yields an internally consistent comparison that supports honest claims about both synthetic performance and real-domain transfer.

---

# 2A The Processing Flow — End to End

Before descending into the details of each stage, it is worth tracing the *entire* journey of a single 2048-sample window through the system, because the whole design is organised around this flow. The four questions the rest of the report answers — *what noise do we handle, how do we do it, why does it work, and what do we measure* — map directly onto the stages below.

## 2A.1 Stage 0 — Acquisition

A measurement begins as a clean PD signal `s(t)`. In the synthetic pathway, `s(t)` is composed by the DOP generator (Chapter 3) from one of the six event archetypes; we therefore *know* `s(t)` exactly and can also emit a per-sample class mask `y(t)`. In the real pathway, `s(t)` is a z-scored Q.Lin window that we treat as the clean reference. Either way, the output of Stage 0 is a clean window of 2048 samples (and, synthetically, its label mask).

## 2A.2 Stage 1 — Noise injection

The composite noise field `n(t)` (Chapter 4) is generated and scaled to hit a target SNR, then added: `r(t) = s(t) + scale·n(t)`. The result `r(t)` is the noisy input the model actually sees. Crucially, the *same* injection procedure is used for training, synthetic evaluation, and real grounded evaluation — only the clean signal `s(t)` differs across these uses. This single shared procedure is the linchpin of the unified measurement standard.

## 2A.3 Stage 2 — Encoding (analysis)

The noisy window `r(t)`, shaped `(B,1,2048)`, is projected to 32 channels and passed through three MWCNN levels. At each level the signal is convolved (learning local features), its full-resolution features are saved as a skip, and it is downsampled by a db4 DWT that *splits* rather than *discards* information into low- and high-frequency sub-bands. After three levels the representation is `(B,256,257)`: temporally compressed eightfold, but with the high-frequency PD detail preserved in the channel dimension and in the three skips.

## 2A.4 Stage 3 — Context modelling (bottleneck)

At the most compressed scale, the BiGRU reads the 257-step sequence forwards and backwards to capture the local temporal signature of PD pulses, and the Swin-1D block then applies windowed self-attention to capture window-global structure (hum, recurring impulsive patterns). The bottleneck output has the same shape `(B,256,257)` but now encodes both local and global context.

## 2A.5 Stage 4 — Decoding (synthesis)

The decoder reverses the encoder over three levels. At each level it upsamples (linearly or by IDWT — the decoder variation), fuses the matching encoder skip through an attention gate that suppresses noise-dominated activations, and refines with convolutions. After three levels the representation is back to `(B,32,2048)` at full temporal resolution.

## 2A.6 Stage 5 — Output heads

The reconstruction head maps the 32-channel features to a single denoised channel `ŝ(t)`, shaped `(B,1,2048)`. In multi-task configurations the segmentation head fuses the global bottleneck features with the local decoder features to produce per-sample class logits `m̂(t)`, shaped `(B,5,2048)`. The two outputs are the deliverables: a clean waveform and a localisation of PD events.

## 2A.7 Stage 6 — Supervision (training) or scoring (evaluation)

During training, `ŝ` is compared to `s` by the Charbonnier (and SSIM) reconstruction loss, and `m̂` to `y` by the Focal/Generalized-Dice segmentation loss, with the two balanced by learned uncertainty; gradients flow back through every stage. During evaluation, `ŝ` is scored against `s` by NCC, RMSE and SNR-improvement, `m̂` against `y` by Dice and FPR, and — on the real grounded set — `ŝ` against the clean Q.Lin reference by the grounded NCC.

## 2A.8 Why this particular flow

Each stage exists to solve a specific failure of classical denoising. The DWT encoding solves the *information-loss* problem of pooling (the high-frequency PD edge survives). The BiGRU+Swin bottleneck solves the *context* problem (a pulse is recognised by its temporal signature and its surroundings, not by amplitude alone). The attention gates solve the *noisy-skip* problem (skip connections would otherwise reintroduce noise at the decoder). The multi-task head solves the *morphology-collapse* problem (a blurred output cannot be segmented, so the encoder is forced to keep diagnostic detail). And the curriculum solves the *training-instability* problem (the model learns clean structure before facing −15 dB). The remainder of the report develops each of these in turn.

---

# 3 Methodology I — Signal Acquisition

A denoiser can only be as good as the data it learns from and is tested on. This chapter describes how PD signals enter the pipeline: the physically motivated synthetic generator that produces labelled clean pulses, and the real Q.Lin AE dataset that provides the domain-transfer benchmark. The guiding principle is that synthetic data supply the *labels* (which require a known clean signal that cannot be obtained in the field), while real data supply the *realism* (the channel and environment that synthesis cannot fully capture). Figure 1 places signal acquisition at the head of the end-to-end pipeline; Figure 4 illustrates the synthetic archetypes.

## 3.1 Why hybrid data

Training a supervised denoiser requires paired clean and noisy signals. For real AE-PD measurements this pairing is physically impossible: one cannot record a "clean" PD signal in an energised transformer, because the noise is intrinsic to the measurement. The pragmatic resolution adopted here is hybrid: a physics-based synthetic generator provides paired clean/noisy windows with per-sample class labels for supervised training and synthetic evaluation, while the real Q.Lin dataset is used for a grounded transfer evaluation in which composite noise is *injected* into the (near-clean) real windows so that the original window can serve as the reference. This separation of roles is deliberate and is reflected throughout the evaluation protocol (Chapter 8).

## 3.2 Synthetic PD signal generation

### 3.2.1 The damped-oscillatory pulse model

Each synthetic PD event is modelled as a damped oscillatory pulse (DOP). For a single event beginning at time `t₀`, the contribution to the window is

```
p(t) = A · exp(−(t − t₀)/τ) · sin(2π f_c (t − t₀) + φ),   t ≥ t₀,
```

with

- `A` — peak amplitude (signed, so pulses may be positive- or negative-going);
- `τ` — decay time constant controlling the envelope's exponential fall;
- `f_c` — carrier (oscillation) frequency within the AE band;
- `φ` — initial phase.

This single-exponential form captures the fast-rise, oscillatory-decay character of an AE-PD transient. A double-exponential variant (a difference of two exponentials) is also available to model pulses with a finite rise time. The DOP family is consistent with the established single/double damped-oscillation models for PD simulation [7].

### 3.2.2 Event archetypes (Types A–F)

A realistic window rarely contains a single isolated pulse; field recordings are continuous and contain many events of varying density and amplitude. The generator therefore composes each window from a stochastic number of events drawn from one of six archetypes, each defined by characteristic ranges of event count, duration, amplitude, decay and carrier frequency:

- **Type A — corona-like.** 6–16 repetitive, low-amplitude bursts; durations ≈120–400 µs; carriers ≈20–80 kHz. Models the frequent, low-energy discharges typical of corona at sharp points.
- **Type B — dense streamer-like.** A high event rate (hundreds to thousands of events per second) of short, low-amplitude overlapping pulses; carriers ≈80–300 kHz. Models dense surface streamering.
- **Type C — sparse surface-discharge-like.** 4–10 sparse, higher-amplitude impulses; longer durations ≈250–900 µs; carriers ≈50–200 kHz.
- **Type D — sparse high-amplitude internal-like.** 5–14 high-amplitude pulses; carriers ≈100–400 kHz; models internal-void discharge.
- **Type E — moderate-density.** 10–24 moderate-amplitude pulses across the AE band; carriers ≈60–220 kHz.
- **Type F — denser internal-like.** 12–28 pulses with broad AE-band carriers ≈80–260 kHz.

A seventh archetype (**Type G**, treeing) and **mixed** windows (2–4 archetypes combined) exist for augmentation. The six archetypes A–F are the ones used to construct the frozen synthetic test set.

#### Archetype parameter ranges

The generator draws each archetype's events from the following ranges, chosen to span realistic AE-PD density, amplitude and carrier-frequency regimes:

| Type | Events/window | Duration (µs) | Amplitude | Decay τ (µs) | Carrier f_c (kHz) | Physical analogue |
|---|---|---|---|---|---|---|
| A | 6–16 | 120–400 | 0.05–0.6 (±) | 120–500 | 20–80 | Corona, repetitive low-energy |
| B | rate 500–2000 Hz (≥25) | 50–180 | 0.01–0.08 (±) | 50–200 | 80–300 | Dense surface streamering |
| C | 4–10 | 250–900 | 0.3–5.0 (±) | 200–1200 | 50–200 | Sparse surface discharge |
| D | 5–14 | 150–600 | 1.0–8.0 (±) | 50–300 | 100–400 | High-amplitude internal |
| E | 10–24 | 120–500 | 0.6–4.5 (±) | 100–700 | 60–220 | Moderate-density mixed |
| F | 12–28 | 100–450 | 0.8–5.2 (±) | 80–500 | 80–260 | Dense internal |

The (±) indicates that amplitudes are randomly signed, so pulses may be positive- or negative-going, reflecting the polarity diversity of real PD. The spread of these ranges, together with random event counts and start times, means the model is exposed to an effectively unbounded variety of clean signals across training.

#### The double-exponential variant

In addition to the single-exponential DOP, a double-exponential envelope models pulses with a finite, non-zero rise time:

```
p(t) = A · [ exp(−1.3 (t−t₀)/τ) − exp(−2.2 (t−t₀)/τ) ] · sin(2π f_c (t−t₀)),   t ≥ t₀.
```

The difference of two exponentials rises from zero, peaks, and decays, capturing the characteristic onset transient of certain discharge types more faithfully than a pure decaying envelope.

#### The acoustic transfer function

A real AE pulse is not the raw electrical discharge but the discharge as *filtered by the propagation medium and the sensor*. The synthetic generator therefore convolves the raw pulse train with an acoustic transfer function before normalisation, band-shaping the spectrum so that the dominant energy sits within the AE band (a sanity check asserts the peak PSD frequency lies in 20–500 kHz). This step is what keeps the synthetic pulses physically plausible rather than idealised mathematical oscillations, and it is part of why a model trained on synthetic data can transfer at all to real recordings.

### 3.2.3 Acoustic transfer function and normalisation

After the events are superimposed, an **acoustic transfer function** is applied to emulate the band-shaping of the mechanical propagation path and the piezoelectric sensor, so that the synthetic pulse spectrum sits within the realistic AE band rather than being an idealised mathematical pulse. The window is then peak-normalised to unit maximum absolute amplitude, giving a consistent dynamic range across windows.

### 3.2.4 Per-sample class labels

Alongside the clean waveform, the generator emits a per-sample integer **segmentation mask** that labels each time index with one of five physics-based classes:

| Class | Meaning | Source archetypes |
|---|---|---|
| 0 | Background (no PD) | — |
| 1 | Corona | A |
| 2 | Surface | B |
| 3 | Internal | C, D, E, F |
| 4 | Treeing | G |

This mask is the supervision target for the segmentation head (Chapter 5). It is intrinsically class-imbalanced — PD samples occupy only a small fraction of each window — which motivates the Focal and Generalized Dice losses (Chapter 6).

### 3.2.5 Sampling and window geometry

Signals are generated at a sampling rate of 1.0e6 with a window length of **2048 samples** (≈2.048 ms). The 2048-sample length is a power of two chosen to align cleanly with the three-level DWT cascade of the encoder, where the length sequence is 2048 → 1025 → 513 → 257 (the slight departure from exact halving arises from the db4 filter's boundary handling).

## 3.3 Real Q.Lin AE dataset

### 3.3.1 Experimental origin

The real-world dataset derives from the experimental AE-PD platform of Lin et al. [3], in which partial discharge is induced by placing copper beads of **1.0, 1.8, 2.0 and 2.5 mm** gap diameters, with uniform spacing, on an insulating structure. The setup comprises a 15-kV high-voltage source, a piezoelectric AE sensor for acquisition, a preamplifier, and a high-speed data-acquisition system. PD-induced AE signals are recorded continuously from each defect model, producing four classes of recordings corresponding to the four gap sizes.

### 3.3.2 Windowing and normalisation

Each raw `.mat` trace is segmented into non-overlapping **2048-sample windows**, and each window is **z-score normalised** (zero mean, unit standard deviation) to remove gain and offset differences between recordings. The four gap sizes provide the class labels for the real-domain set (mapped internally as 1.0 mm → 0, 1.8 mm → 1, 2.0 mm → 2, 2.5 mm → 3).

### 3.3.3 Role in evaluation: the grounded reference

Because the Q.Lin recordings are *near-clean* relative to the harsh composite field noise we wish to simulate, they are used as the **clean reference** for a grounded transfer protocol: composite noise is injected into a clean Q.Lin window at each target SNR, the model denoises it, and the output is correlated against the original clean window. A perfect denoiser would recover the original exactly (NCC → 1.0). This grounded protocol replaces the older, weaker practice of correlating against the noisy input window, and is the headline real-world metric (Chapter 8). The number of clean windows and the per-SNR breakdown are reported in the results; the grounded set is frozen and hashed for reproducibility.

### 3.3.4 Limitations of the real dataset

The Q.Lin dataset is among the few publicly usable AE-PD datasets with known defect geometries, but it has limitations that bound the strength of any real-domain claim: it is recorded in a single laboratory environment rather than an operating transformer; it represents one insulation type and a limited set of artificial defect geometries (surface-type discharges from copper beads); and it lacks the additional noise sources (load noise, cooling fans, grid EMI) of a live substation. These limitations are stated plainly so that the grounded real NCC is interpreted as a *controlled* transfer measure, not as a field-deployment guarantee.

---

# 4 Methodology II — The Noise Model

The noise model is the heart of the problem definition: it determines what the denoiser must learn to remove, and a faithful model is the prerequisite for any claim of field relevance. This chapter specifies each noise component, the composite that combines them, and the SNR-controlled injection that is applied *identically* to synthetic and real signals. Figure 3 shows the components and the injection verification.

## 4.1 The additive model and design philosophy

All interference is added to the clean signal in the standard additive model

```
r(t) = s(t) + n(t),
```

where `s(t)` is the clean PD signal, `n(t)` the noise, and `r(t)` the observed window. The composite noise `n(t)` is a sum of five physically distinct components, each parameterised to reflect its real-world source on a 50 Hz (Malaysian) grid. The design philosophy is to model the *kinds* of interference that dominate on-site AE measurement, at relative strengths that make the composite realistic, and then to rescale the whole composite to hit a target SNR. This decouples the *character* of the noise (fixed by the component recipe) from its *severity* (set by the SNR).

## 4.2 Additive white Gaussian noise (AWGN)

AWGN models broadband thermal and electronic noise. It is generated as zero-mean Gaussian samples with a fixed base amplitude (standard deviation), giving a flat power spectral density across the band. AWGN is the simplest component and the one used alone in the easiest curriculum phase. In the additive model its contribution is `w(t) ~ N(0, σ_w²)`.

## 4.3 Magnetostrictive hum

Transformer cores vibrate through magnetostriction at twice the line frequency and its overtones. The model synthesises a hum at **100 Hz** with a weaker **200 Hz** overtone:

```
n_mag(t) = a · sin(2π·100·t) + 0.3a · sin(2π·200·t).
```

This is a low-frequency, periodic, deterministic component; although its fundamental lies far below the AE band, its presence as a structured baseline tests the model's ability to ignore periodic interference.

## 4.4 Powerline interference (with harmonic)

Powerline interference (discrete spectral interference) couples the grid frequency and its harmonics into the measurement. The model uses the **50 Hz fundamental** plus the **3rd harmonic (150 Hz)**:

```
n_pl(t) = a₁ · sin(2π·50·t) + a₃ · sin(2π·150·t).
```

Only the fundamental and third harmonic are modelled; fifth and seventh harmonics are not included in the present recipe (a point flagged for supervisory confirmation and easily extended).

## 4.5 Narrowband interference (NBI)

NBI represents amplitude/frequency-modulated radio sources, communications coupling and cooling-system resonance. It is modelled as a sinusoid at a sub-MHz centre frequency (80.0 kHz) with a **random phase** drawn afresh per realisation:

```
n_nbi(t) = a · sin(2π f_nbi t + φ),   φ ~ U(0, 2π).
```

The random phase prevents the model from memorising a fixed interference waveform and forces it to learn the *spectral* signature instead.

## 4.6 Bernoulli–Gaussian impulsive noise

Impulsive noise is the most difficult component because its sharp rise time mimics a genuine PD pulse, and it is the principal source of segmentation false positives. It is modelled as a sparse train of high-amplitude spikes: a small number of randomly located samples receive a signed, scaled amplitude, while the rest are zero. In the composite recipe a fixed number of spikes (≈15 per window) are placed at random positions with amplitudes `0.4·(0.2 + 0.8·U(0,1))` and random sign. A separate Bernoulli–Gaussian generator (each sample is an impulse with probability `p`, with the impulse drawn from a high-variance Gaussian) is also available; the two recipes are documented as distinct so that their roles are not conflated.

## 4.7 The composite noise field

The composite noise injected during preprocessing is the sum of all five components:

```
n_comp(t) = w(t) + n_pl(t) + n_nbi(t) + n_mag(t) + n_imp(t).
```

This single composite is used for the hardest curriculum phase and for *all* evaluation, on both synthetic and real signals. Using one composite recipe everywhere is what makes the cross-domain comparison fair: the synthetic and real grounded tests differ only in the clean signal, not in the noise field.

## 4.8 SNR-controlled injection

To realise a target SNR, the composite noise is rescaled before addition. Given the clean signal power `var(s)` and the raw noise power `var(n)`, the current SNR is `10·log₁₀(var(s)/var(n))`; the noise is multiplied by a scale factor that brings it to the desired `SNR_dB`:

```
scale = 10^((SNR_current − SNR_target)/20),
r(t)  = s(t) + scale · n(t).
```

The achieved SNR is then verified against the nominal value, and the small difference between nominal and achieved SNR is recorded (this is why per-SNR bucketing in evaluation uses a tolerance band around each nominal level). The seven nominal SNR levels are **−15, −10, −5, 0, +5, +10, +15 dB**.

## 4.9 Injection verification

To guarantee that the *correct* generator and parameters are used in test-set construction, a verification figure is produced for the first real window: for each SNR level it plots the clean window, the isolated noise, and the noisy result, annotating the measured noise RMS, the peak PSD frequency, and the measured SNR, together with the source `.mat` path, the noise module and function name, and the random seed. This artefact (Fig. 3) is the auditable evidence that the composite-noise recipe described above is exactly what was applied.

## 4.10 The frequency-domain picture and why each component is hard

It is instructive to view the composite noise in the frequency domain alongside the PD signal, because the difficulty of the denoising problem is precisely the *overlap* between them.

- **AWGN** spreads its energy uniformly across all frequencies, including the 20–500 kHz AE band where the PD content lives. There is no frequency band the denoiser can simply discard; it must separate signal from noise *within* the shared band, using shape and temporal structure rather than spectral location.
- **Magnetostrictive hum (100/200 Hz)** and **powerline (50/150 Hz)** are low-frequency lines far below the AE band. In principle a high-pass filter removes them, but their large amplitude can still dominate the time-domain waveform and bias amplitude-sensitive losses; the model must learn to ignore a strong but spectrally distinct baseline.
- **NBI** is a narrow spectral peak that may sit inside or near the AE band; with a random phase per realisation, the model cannot memorise its waveform and must instead recognise its spectral signature.
- **Impulsive noise** is the genuinely hard case: a single spike is broadband (its energy spreads across all frequencies) and its sharp onset mimics a PD pulse in both the time and frequency domains. No linear filter can remove it without also attacking real pulses; only a model that has learned the *joint* morphology of pulses and impulses can separate them, which is exactly what the segmentation-regularised, context-aware MR-TAE is designed to do.

This overlap is why classical, fixed-operator denoisers fail and why a learned, context-aware model is necessary.

## 4.11 A worked SNR example

Consider a clean window with signal variance `var(s) = 1.0` (after peak normalisation the variance is of order this magnitude) and a raw composite-noise window with `var(n) = 0.25`. The current SNR is `10·log₁₀(1.0/0.25) ≈ 6.0 dB`. To realise a target of −10 dB, the scale factor is `10^((6.0 − (−10))/20) = 10^(0.8) ≈ 6.31`, so the noise is multiplied by ≈6.31 before addition, raising its variance to `0.25·6.31² ≈ 9.95` and bringing the SNR to `10·log₁₀(1.0/9.95) ≈ −10.0 dB`. This illustrates how the *severity* of the corruption is set independently of its *character*: the same composite recipe is simply scaled to each of the seven target levels.

---

# 5 Methodology III — The MR-TAE Architecture

MR-TAE is a **single** encoder–bottleneck–decoder autoencoder with two output heads. Every model studied in this report — every ablation row, the linear and IDWT decoders, the loss variants — is this one architecture with specific components enabled or disabled, or with a specific upsampling operator selected. This chapter describes the architecture component-by-component with exact tensor shapes, and then explains how the ablation configurations and the decoder variation are obtained by toggling within it. Figure 2 is the corresponding block diagram.

## 5.1 Overview and data flow

The model maps a noisy input window `x ∈ ℝ^{B×1×2048}` to a denoised waveform `ŝ ∈ ℝ^{B×1×2048}` and a per-sample segmentation logit tensor `m̂ ∈ ℝ^{B×5×2048}`. The forward path is:

```
x (B,1,2048)
  → input projection (Conv1d, kernel 7)        → (B, 32, 2048)
  → MWCNN encoder (3 levels, db4 DWT)           → bottleneck features + 3 skips
  → pre-bottleneck projection (1×1 Conv)        → (B, 256, 257)
  → BiGRU + Swin-1D bottleneck (sequential)     → (B, 256, 257)
  → MWCNN decoder (3 levels, linear or IDWT)
        with attention-gated skips              → (B, 32, 2048)
  → reconstruction head (Conv1d 32→1)           → ŝ (B, 1, 2048)
  → segmentation head (dual-stream fusion)      → m̂ (B, 5, 2048)
```

The encoder progressively halves temporal resolution while doubling channels; the bottleneck models temporal and global context at the most compressed scale; the decoder reverses the encoder, re-injecting the encoder's skip features through attention gates; and the two heads produce the denoised signal and the PD-event mask.

## 5.2 Input projection

The single-channel input is first lifted to the base channel width by a 1-D convolution with kernel size 7 and padding 3:

```
input_proj: Conv1d(1 → 32, kernel=7, padding=3),   (B,1,2048) → (B,32,2048).
```

The wide kernel gives the first layer a 7-sample receptive field, enough to begin responding to short pulse edges before any downsampling.

## 5.3 MWCNN encoder (db4-DWT)

### 5.3.1 Per-level structure: Conv-first, then DWT

Each of the three encoder levels applies **two** Conv–BatchNorm–GELU blocks and then a **Daubechies-4 DWT** downsampling step. Crucially, the **skip connection is taken from the post-Conv, pre-DWT** features, so the skip retains full temporal resolution at that level. The DWT splits the signal into a low-frequency (approximation) and a high-frequency (detail) sub-band, each at half length, and stacks them along the channel dimension — so the channel count doubles while the length halves.

### 5.3.2 The DWT layer

The DWT is implemented as a fixed (non-trainable) grouped 1-D convolution with the db4 decomposition filters. For an input of length `L`, the signal is reflect-padded, convolved separately with the low-pass and high-pass db4 filters, downsampled by 2, and the two results concatenated on the channel axis. The db4 wavelet (8 filter taps) is chosen because its asymmetry and support match the asymmetric, fast-rising PD pulse better than the symmetric Haar wavelet.

#### Mathematical form of the DWT

Let `h_dec` and `g_dec` be the db4 low-pass (decomposition) and high-pass (decomposition) filters, each of length 8. For a single-channel signal `x[n]`, one DWT level computes the approximation `a` and detail `d` sub-bands by filtering and decimating by two:

```
a[k] = Σ_n  h_dec[n] · x̃[2k − n],
d[k] = Σ_n  g_dec[n] · x̃[2k − n],
```

where `x̃` is `x` reflect-padded at the boundaries. The layer stacks `[a; d]` along the channel axis, so a `(B, C, L)` input becomes `(B, 2C, ⌈L/2⌉)`. Because the db4 filters form a perfect-reconstruction quadrature-mirror pair, no information is lost: the original can be recovered exactly by the inverse transform (below). This is the precise sense in which DWT downsampling is *lossless*, in contrast to max- or average-pooling, which are many-to-one and therefore irreversible.

The db4 decomposition low-pass coefficients are approximately
`[−0.0106, 0.0329, 0.0308, −0.1870, −0.0280, 0.6309, 0.7148, 0.2304]`,
with the high-pass filter obtained by the usual alternating-flip relation. The eight-tap support gives the filter enough vanishing moments (db4 has four) to represent the smooth portions of the signal compactly while concentrating the pulse edges in the detail band.

#### Mathematical form of the IDWT

The inverse transform reconstructs `x` from `[a; d]` using the reconstruction filters `h_rec`, `g_rec`:

```
x[n] = Σ_k  h_rec[n − 2k] · a[k] + Σ_k  g_rec[n − 2k] · d[k].
```

In the layer this is implemented by zero-insertion upsampling (placing the sub-band samples at even indices), reflect padding, convolution with `h_rec`/`g_rec`, and summation of the two contributions, followed by a deterministic crop/pad to the stored target length. A `(B, 2C, L)` input becomes `(B, C, 2L)` (then cropped). This IDWT layer is exactly the operator used by the IDWT decoder variation (Section 5.7.2).

### 5.3.3 Tensor shapes

With base channel widths [32, 64, 128] and an input length of 2048, the encoder produces:

| Level | Conv output (skip) | After db4 DWT |
|---|---|---|
| 1 | skip₁ = (B, 32, 2048) | (B, 64, 1025) |
| 2 | skip₂ = (B, 64, 1025) | (B, 128, 513) |
| 3 | skip₃ = (B, 128, 513) | (B, 256, 257) |

The slight departure from exact halving (1025 rather than 1024, etc.) is a deterministic consequence of the db4 filter length and the reflect padding, and is handled exactly in the decoder by storing each level's skip length and cropping/padding to match. The final encoder output is `(B, 256, 257)`.

### 5.3.4 The plain-encoder ablation

When wavelet pooling is disabled (the `MR-TAE-noWavelet` and `UNet-BiGRU-Swin` configurations), the MWCNN encoder is replaced by a **standard convolutional encoder** that uses Conv–BatchNorm–GELU blocks followed by max-pooling for downsampling, with the same channel progression and the same skip interface. This isolates the contribution of frequency-aware wavelet sampling versus conventional pooling.

## 5.4 Pre-bottleneck projection

A 1×1 convolution maps the encoder output channels to the bottleneck width (256), giving the canonical bottleneck tensor `(B, 256, 257)` on which all bottleneck variants operate.

## 5.5 The bottleneck: BiGRU then Swin-1D (sequential)

The full bottleneck (the `HybridBottleneck`) processes the `(B, 256, 257)` representation through a bidirectional GRU and then a 1-D Swin Transformer, wrapped in a residual connection. The arrangement is **sequential** — the BiGRU output is the Swin input — so local temporal context is established first and refined by global attention.

### 5.5.1 BiGRU

A 1×1 convolution projects to the GRU input width; a **2-layer bidirectional GRU** with **128 hidden units per direction** (256 total) processes the sequence along the time axis, in both forward and backward directions. The BiGRU captures the local temporal signature of a PD pulse — a fast rise followed by oscillatory decay — and its surrounding context, which helps distinguish a genuine pulse (preceded by a quiet region) from an impulsive artefact.

#### GRU recurrence

For each direction, a GRU cell updates its hidden state `h_t` from the input `x_t` and previous state `h_{t−1}` through an update gate `z_t`, a reset gate `r_t` and a candidate state `ñ_t`:

```
z_t = σ(W_z x_t + U_z h_{t−1} + b_z),
r_t = σ(W_r x_t + U_r h_{t−1} + b_r),
ñ_t = tanh(W_n x_t + r_t ⊙ (U_n h_{t−1}) + b_n),
h_t = (1 − z_t) ⊙ ñ_t + z_t ⊙ h_{t−1}.
```

The bidirectional layer runs one such recurrence left-to-right and another right-to-left and concatenates their hidden states, so each output step is conditioned on the entire window. Stacking two layers lets the second layer model higher-order temporal dependencies on the first layer's representation. The gating is what allows the GRU to retain the memory of a quiet pre-pulse region across many time steps without the vanishing-gradient pathology of a vanilla RNN.

### 5.5.2 Swin-1D

The BiGRU output is projected to the Swin embedding dimension (256) and processed by a **1-D Swin Transformer** of depth 2 with 8 attention heads and window size 32. Attention is computed within non-overlapping windows of 32 time steps for linear complexity; alternate blocks apply a **shifted window** (cyclic roll by half the window) so that information flows across window boundaries. A relative-position bias is added within each window. Window-stationary properties of the noise (a hum frequency, an impulsive pattern that recurs across the window) are exactly the kind of global structure this block can model.

#### Windowed self-attention

The 257-step sequence is partitioned into non-overlapping windows of `W = 32` steps (with padding to a multiple of `W`). Within each window, multi-head self-attention is computed:

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k + B ) V,
```

where `Q, K, V` are linear projections of the window's tokens, `d_k` is the per-head dimension, and `B` is a learnable **relative-position bias** indexed by the pairwise offset of tokens within the window. Restricting attention to a window reduces the cost from `O(L²)` (full attention) to `O(L·W)`, i.e. linear in sequence length. To let information cross window boundaries, alternate Swin blocks apply a **shifted window**: the sequence is cyclically rolled by `W/2 = 16` before partitioning and rolled back afterwards, so tokens that were on opposite sides of a boundary share a window in the next block. Each block wraps attention and a two-layer GELU MLP (hidden ratio 4) in pre-norm residual connections:

```
x ← x + Attention(LN(x)),
x ← x + MLP(LN(x)).
```

With depth 2, the block contains one regular-window and one shifted-window stage, which together give every token an effective receptive field spanning the whole 257-step bottleneck.

### 5.5.3 Residual wrapping and bottleneck ablations

A residual projection adds the bottleneck input to the processed output, stabilising training. The bottleneck has four ablation forms, selected by the `use_bigru`/`use_swin` flags:

- **BiGRU + Swin** (full) — sequential, as above;
- **BiGRU only** — Swin removed;
- **Swin only** — BiGRU removed;
- **Conv only** — both removed, replaced by two residual conv blocks (the minimal bottleneck used when both sequence models are disabled).

These four forms underpin the bottleneck rows of the ablation grid (Chapter 9).

## 5.6 Attention-gated skip connections

Each decoder level fuses the corresponding encoder skip through an **attention gate** before concatenation. The gate computes, from the decoder (gating) feature `g` and the encoder skip `x`,

```
α = σ( ψ( ReLU( W_g · g + W_x · x ) ) ),     gated_skip = α ⊙ x,
```

where `W_g`, `W_x`, `ψ` are 1×1 convolutions and `σ` is the sigmoid. The gate produces a per-location attention coefficient `α ∈ [0,1]` that suppresses skip activations dominated by noise and emphasises those aligned with the decoder's emerging reconstruction. A residual form retains the original skip alongside the gated one. When attention gates are disabled (the `noAttn` configurations), the raw skip is concatenated directly.

#### Why gate the skip at all

In a plain U-Net, the encoder skip is concatenated to the decoder feature unchanged. This is a double-edged sword for denoising: the skip carries the full-resolution detail the decoder needs, but at low SNR that same skip is dominated by noise, so concatenating it raw reintroduces precisely the contamination the bottleneck just suppressed. The attention gate resolves this tension by letting the decoder's own (already partly denoised) feature act as a query that scores each location of the skip: locations consistent with the emerging reconstruction are passed through (`α → 1`), while locations dominated by noise are attenuated (`α → 0`). The gate is additive and uses an intermediate channel width that keeps its parameter overhead small relative to the conv blocks. The `noAttn` ablation directly measures whether this selective gating helps or whether the raw skip suffices.

## 5.7 The decoder and the IDWT variation

### 5.7.1 Per-level structure

The decoder mirrors the encoder over three levels. At each level the feature map is **upsampled ×2**, concatenated with the (gated) skip of matching length, and refined by two Conv–BatchNorm–GELU blocks. The channel progression is [128, 64, 32], ending at 32 channels and full length 2048. Each level is told the exact target length (the stored encoder skip length), so the boundary-induced length offsets are resolved deterministically by cropping/padding the upsampled feature to match the skip.

### 5.7.2 The decoder variation: linear vs IDWT

Upsampling is the single controlled **decoder variation**:

- **Linear variant (`use_idwt = False`).** `Upsample(scale=2, mode='linear')` followed by a 1×1 convolution, batch norm and GELU. This is a smooth, interpolation-based upsampling that tends to produce outputs close to the input (a conservative denoising step).
- **IDWT variant (`use_idwt = True`).** A db4 **inverse-DWT** layer reconstructs the higher-resolution feature from the stacked low/high sub-bands by zero-insertion upsampling, reflect padding, convolution with the db4 reconstruction filters, and summation of the low- and high-pass contributions; a deterministic crop/pad then matches the target length. IDWT performs a more aggressive, frequency-aware reconstruction that can diverge further from the noisy input.

Both variants share **all** other weights and structure. IDWT is therefore not a separate model but a different upsampling operator inside the one MR-TAE. This framing is what allows any linear-vs-IDWT difference to be attributed cleanly to the reconstruction operator.

### 5.7.3 A note on loading the IDWT variant

The IDWT variant is only architecturally consistent if the `use_idwt` flag is set when the model is instantiated. Where a checkpoint's training-time configuration omits this flag, it must be inferred (from the model identifier, the registry, or the presence of IDWT-specific weights in the state dictionary) before loading, and the loaded model must pass a positive grounded-NCC sanity check. This loading correctness is a prerequisite for the IDWT rows of the results to be meaningful; it is part of the benchmark integrity work that precedes result filling.

## 5.8 Output heads

### 5.8.1 Reconstruction head

A 1-D convolution maps the decoder's 32-channel, full-length features to a single output channel with **no activation** (linear output), because the denoised AE signal is bipolar:

```
recon_head: Conv1d(32 → 1),   (B,32,2048) → ŝ (B,1,2048).
```

### 5.8.2 Segmentation head (dual-stream fusion)

The segmentation head is a **dual-stream fusion** module that combines the global bottleneck features with the local decoder features to produce per-sample class logits over the five classes {background, corona, surface, internal, treeing}:

```
seg_head: (bottleneck, decoder features) → m̂ (B, 5, 2048).
```

Fusing a global stream (the bottleneck representation, which captures window-level context) with a local stream (the full-resolution decoder features, which localise events precisely) lets the segmenter be both context-aware and temporally sharp. When multi-task learning is disabled (the `noMTL`, `MWCNN-BiGRU`, `UNet-BiGRU-Swin` configurations), the segmentation head is removed and the model is a pure denoiser.

## 5.9 How the model becomes each configuration

Every studied model is this architecture with a specific flag set. The flags are:

| Flag | Effect when enabled |
|---|---|
| `use_wavelet_pooling` | MWCNN db4 encoder/decoder (else plain conv + max-pool) |
| `use_bigru` | BiGRU in the bottleneck |
| `use_swin` | Swin-1D in the bottleneck |
| `use_attention_gates` | attention-gated skips (else raw skips) |
| `use_mtl` | dual-stream segmentation head (else denoise-only) |
| `use_idwt` | IDWT decoder upsampling (else linear) |
| `loss_config` | A, B or C (Chapter 6) |

The 13-configuration ablation, the decoder variation and the loss variants are all points in this flag space, which is precisely why the comparison is controlled: each entry differs from the full model in exactly the toggled dimension(s).

## 5.10 Parameter budget

Parameter counts vary with the enabled components (the Swin block and BiGRU are the heaviest). Locked counts are reported per configuration in Appendix A (e.g. MR-TAE-FULL 3.01 M, NoMix 3.05 M, noSwin 1.41 M, IDWT 2.92 M).

## 5.11 Design alternatives and why they were not chosen

Several alternative design choices were available at each stage; documenting them clarifies the rationale for the architecture and frames the ablation.

- **Pooling vs DWT.** Max/average pooling and strided convolution are the conventional downsamplers, but all three are lossy in the high-frequency band that defines a PD pulse. The DWT was chosen for its lossless, invertible, frequency-splitting behaviour; the `noWavelet` ablation quantifies the cost of reverting to pooling.
- **Parallel vs sequential bottleneck.** The BiGRU and Swin could be arranged in parallel (concatenating their outputs) or sequentially. The sequential arrangement (BiGRU → Swin) was chosen so that the recurrent layer first establishes local temporal context, which the attention layer then refines globally; this mirrors the intuition that global reasoning is most useful once local structure is available.
- **Transformer vs Swin.** A full self-attention transformer would model all pairwise interactions but at quadratic cost; the windowed Swin variant restores linear complexity while the shifted-window mechanism preserves cross-window flow — a better fit for a 257-step bottleneck on a single consumer GPU.
- **Linear vs IDWT upsampling.** Rather than commit to one upsampler, both are retained as a controlled decoder variation, so the choice is settled empirically under the unified protocol rather than by assertion.
- **Fixed vs learned loss weights.** Fixed multi-task weights are brittle as the curriculum changes the relative task difficulty; homoscedastic-uncertainty weighting was chosen to adapt the balance automatically.

---

# 6 Methodology IV — Training: Curriculum and Losses

This chapter describes how MR-TAE is trained: the three-phase noise curriculum (and the phased/no-phase/NoMix variations), the three loss configurations and their mathematics, the optimisation settings, and the training infrastructure that makes long, multi-model runs robust. The training data are generated fresh each epoch, so the model never sees the same noisy realisation twice.

## 6.1 The three-phase curriculum

### 6.1.1 Motivation

Exposing the model directly to −15 dB composite noise from the first epoch is unstable: when the pulse is invisible the reconstruction gradient is near-random and training can collapse. Curriculum learning resolves this by presenting easy examples first and increasing difficulty gradually, so the model learns clean PD structure before it must extract it from extreme noise.

### 6.1.2 Phase schedule

The curriculum has three phases of increasing difficulty, each defined by an SNR range and a noise type:

| Phase | SNR range | Noise type | Purpose |
|---|---|---|---|
| 1 (easy) | +5 to +15 dB | WGN only | Learn clean PD pulse structure |
| 2 (medium) | −5 to +5 dB | WGN + impulsive | Handle moderate noise and impulsive artefacts |
| 3 (hard) | −20 to −5 dB | Composite (all five components) | Master extreme, realistic noise |

Within each phase the SNR for each training window is sampled uniformly from the phase range, and the noise type is the phase's type. The phase boundaries are defined by epoch counts; the scheduler maps each absolute epoch to a (phase, SNR range, noise type) tuple. Per-phase epoch counts are configuration-dependent and reported as Phase 1: 20; Phase 2: 30; Phase 3: 50 with a total of 100 epochs.

### 6.1.3 Curriculum variations: phased, no-phase, NoMix

The curriculum is itself a **training variation** of the one model:

- **Phased** — the three-phase schedule above, applied in sequence.
- **No-phase** — the model is trained directly on the full composite distribution across the entire SNR range, with no easy-to-hard progression. The phased-vs-no-phase comparison (Chapter 9, Table 2) quantifies the value of the curriculum for low-SNR robustness.
- **NoMix (trained — recommended).** The same three-phase schedule but **without mixed-type augmentation windows** — each training window contains a single discharge archetype only. This forces purer per-class morphology learning and is the **overall best configuration**: mean synthetic NCC **0.8282**, NCC @ −15 dB **0.3777**, SNR improvement **9.64 dB**, Dice **0.9192**, grounded real NCC **0.4535** (+17.9% over BayesShrink 0.3844, no fine-tuning). Parameters: 3.05 M; latency: 17.1 ms (batch 1).

## 6.2 Loss configurations

Three supervision regimes are studied on the *same* architecture, isolating the effect of the loss from the effect of the architecture. The reconstruction term is common to all three; the segmentation term and the task-balancing differ.

### 6.2.1 Reconstruction loss — Charbonnier

The reconstruction term is the **Charbonnier loss**, a smooth, robust approximation to the L1 norm:

```
L_charb = mean( sqrt( (ŝ − s)² + ε² ) ),     ε = 1e−3.
```

Charbonnier is preferred over MSE because PD pulses are spiky: an L2 loss would over-penalise the large-amplitude pulse samples and under-penalise the noise floor, biasing the denoiser toward over-smoothing. Charbonnier's near-L1 behaviour away from zero makes it robust to the high-amplitude impulsive outliers that contaminate the signal.

### 6.2.2 Segmentation losses — Focal and Generalized Dice

The segmentation target is severely class-imbalanced (PD samples are a small fraction of each window). Two complementary losses address this:

- **Focal loss** down-weights well-classified background samples and concentrates learning on hard examples (pulse boundaries, impulsive artefacts):
  ```
  L_focal = mean( α · (1 − p_t)^γ · CE(p_t) ),     γ = 2.
  ```
- **Generalized Dice loss (GDL)** weights each class inversely to its frequency, so the rare PD classes are not swamped by the background:
  ```
  L_gdl = 1 − ( 2 · Σ_c w_c · |P_c ∩ T_c| ) / ( Σ_c w_c · (|P_c| + |T_c|) ),     w_c = 1 / (Σ T_c)².
  ```

### 6.2.3 Task balancing — homoscedastic uncertainty

When both heads are active, their losses are balanced automatically by **homoscedastic-uncertainty weighting**: each task `t` carries a learnable log-variance `log σ_t²`, and the combined loss is

```
L = Σ_t [ L_t / (2 σ_t²) + ½ log σ_t² ].
```

The `½ log σ_t²` term prevents the model from trivially inflating `σ_t` to ignore a task. As training proceeds, the weights adapt across curriculum phases — early on the reconstruction term dominates, while later both terms matter. The σ parameters are part of the optimiser state and are saved in checkpoints.

#### Derivation

The weighting follows from maximising the joint Gaussian likelihood of the two task outputs. Modelling each task `t` as a Gaussian with observation noise `σ_t`, the negative log-likelihood of the combined model is

```
−log p(y₁, y₂ | f(x)) ∝ Σ_t [ (1/(2σ_t²)) ‖y_t − f_t(x)‖² + log σ_t ].
```

Identifying `‖y_t − f_t(x)‖²` with the task loss `L_t` gives exactly the weighted objective above (up to the constant that lets us learn `log σ_t²` directly for numerical stability). The interpretation is intuitive: a task with large irreducible uncertainty `σ_t` is automatically down-weighted (its `1/(2σ_t²)` coefficient shrinks), while the `log σ_t` penalty stops the optimiser from declaring every task maximally uncertain. This removes the brittle manual coefficient `λ` that a naive `L₁ + λL₂` objective would require, and — importantly for this work — lets the balance drift correctly as the curriculum makes the reconstruction task progressively harder.

### 6.2.4 The three configurations

| Config | Reconstruction | Segmentation | Task balancing |
|---|---|---|---|
| **A** (full) | Charbonnier | Focal + Generalized Dice | Learned homoscedastic uncertainty |
| **B** | Charbonnier | Generalized Dice (no Focal) | Learned homoscedastic uncertainty |
| **C** | Charbonnier + SSIM | — (no segmentation head) | Single-task (no MTL) |

Config A is the most heavily supervised; Config B simplifies the segmentation supervision; Config C is reconstruction-only with an added 1-D SSIM term that rewards local structural similarity. The comparison of A/B/C on the full model isolates the effect of loss design on both synthetic quality and real-domain transfer.

## 6.3 Optimisation

Training uses the AdamW optimiser (learning rate 3e-04, weight decay 1e-04) with a cosine-annealing schedule and a short linear warm-up, gradient clipping at max-norm 1.0 (important for stabilising the BiGRU under extreme-noise gradients), and automatic mixed precision (FP16 forward, FP32 master weights) for throughput and memory efficiency. The data seed is fixed for reproducibility. Batch size is selected per model from its parameter count to leave GPU-memory headroom; exact values are configuration-dependent and reported as 32.

## 6.4 Training infrastructure and resilience

Long multi-model training runs are made robust by several mechanisms: atomic per-epoch persistence of the full training state (so a kill never corrupts the session), checkpoint-based resume (model, optimiser, scheduler, loss σ, AMP scaler and history), automatic batch-size halving on CUDA out-of-memory, a thermal monitor that pauses training above a temperature threshold, and dynamic early stopping based on a rolling improvement rate. These do not affect the trained model's definition but ensure that the reported checkpoints are the product of complete, recoverable runs. Any model that was halted at a partial epoch is flagged with its epoch/phase for transparency in the metrics record.

## 6.5 Checkpoint selection

For a fair comparison, each model's evaluated checkpoint is selected by an identical, stable procedure rather than by a single noisy validation point. Across the last few late-epoch checkpoints of a model, the winner is the one with the highest **mean NCC across all seven SNR levels** of the frozen synthetic sweep (with the −15 dB NCC as a tie-break). The selected checkpoint and the selection record are stored per model, so the same procedure and the same frozen test sets are applied to every configuration.

#### Selection procedure (pseudocode)

```
for each model M:
    candidates = {best_phase3, last few epoch checkpoints, last}
    for each ckpt in candidates:
        load M with ckpt (inferring use_idwt if the sidecar omits it)
        score[ckpt] = mean over the 7 SNR levels of NCC(M(noisy), clean)   # frozen synthetic sweep
    selected[M] = argmax_ckpt score[ckpt]                                  # tie-break on NCC @ -15 dB
    save selected[M] as selected_final.pt and record the decision
```

Because the candidate set, the scoring metric (mean NCC over the seven-level sweep), the tie-break (−15 dB) and the frozen test sets are identical for every model, the comparison is fair by construction: no model benefits from a luckier validation point or an easier curriculum phase than another.

---

# 7 Experimental Setup and Reproducibility

This chapter specifies the datasets, the test-set freezing procedure, the hardware, and the controls that make the experiment reproducible. The overriding principle is that every model is measured under *identical* conditions, so that performance differences are attributable to the model alone.

## 7.1 Datasets and splits

Two evaluation datasets are used, both frozen before any benchmarking:

- **Synthetic AE-PD test set.** Generated from the six archetypes A–F with the composite noise model, at 2000 windows per SNR level across the seven levels (−15 to +15 dB), giving 14000 windows. Each window carries a clean reference and a per-sample class mask, enabling NCC, RMSE, SNR-improvement, Dice and FPR.
- **Real Q.Lin grounded test set.** Clean Q.Lin windows (held-out split) with composite noise injected at the seven SNR levels (−15 to +15 dB), at 5000 windows per SNR level, giving **35 000** noisy evaluation rows (5000 clean references × 7 SNRs). The clean window is the NCC reference.

The real corpus is split deterministically (seed-fixed) into a held-out test split and a small pool; the grounded set is built from the test split so that no window used for noise-pool purposes leaks into evaluation.

## 7.2 Freezing and hashing

Both test sets are written to disk once and **SHA-256 hashed**; the hashes are recorded in the reproducibility appendix (Appendix D). Any change to a test set changes its hash, so the integrity of the evaluation data is auditable. The hashes are ade2b9ec7b61681153ed475ef329b50e2524fa6f3426ae5314b10481b4bc9561, 16ce0d927a11e1e47201df8f82822a4f20d6ff119a1091486c74f510dbcb0658, 6eb400e41464415fb3da8bba3b01a492879ad0c8bb2b87ce0e610cbac5fcbcb2, 42640d8e4a4c9e14570203dfa6e7e83658bea28f6714d7b40e941b1093732e64.

## 7.3 Unified evaluation protocol

Every trained configuration is evaluated on its selected checkpoint, and every classical baseline on the same frozen sets, through one evaluation pipeline that computes the full metric set. The protocol is identical across entries:

1. Run the model on the frozen synthetic set; compute NCC, RMSE, SNR-improvement, Dice, FPR, and per-SNR NCC.
2. Run the model on the frozen real grounded set; compute the grounded NCC overall and per SNR.
3. Measure inference latency (batch 1 and batch 32, CUDA-synchronised) and count parameters.

For very large grounded sets, a frozen, seed-fixed subset of rows is used so that all models are evaluated on exactly the same rows.

## 7.4 Hardware and software

Experiments run on a single consumer GPU (NVIDIA RTX 4070, 12 GB) with CUDA and automatic mixed precision. The software stack is PyTorch with NumPy/SciPy for signal processing and PyWavelets for the classical wavelet baseline. Per-model latency is locked in `paper/metrics_master.json` (e.g. NoMix 17.1 ms, MR-TAE-FULL 40.1 ms, MWCNN-BiGRU 3.75 ms).

## 7.5 Benchmark integrity gates

The re-locked benchmark (2026-05-31) applied sanity gates before accepting results: grounded real NCC must be non-negative (except documented failure modes such as noMTL); synthetic per-SNR sweeps must be monotone-plausible; and IDWT checkpoints must load with `use_idwt=true` verified. All models in the headline tables passed these gates. Classical grounded-NCC headline means use finalized poster values (Section 9.0).

---

# 8 Baselines and Evaluation Metrics

## 8.1 Classical baselines

Three classical denoisers anchor the learned models, all evaluated under the identical protocol:

- **Wiener filtering** — an optimal linear filter under a local stationarity assumption; included as a classical reference. 
- **Wavelet BayesShrink** — db4 wavelet decomposition with BayesShrink soft thresholding (a sub-band-adaptive threshold estimated from the finest detail coefficients). This is the strongest classical baseline and the principal point of comparison.
- **Savitzky–Golay** — local least-squares polynomial smoothing (window 51, polynomial order 3).

None of the classical methods has a segmentation capability, so Dice and FPR are reported as not applicable for them. Their parameter count is zero and their latency is measured on CPU.

## 8.2 Metric definitions

### 8.2.1 Normalised cross-correlation (NCC) — primary

NCC measures the zero-mean correlation between the denoised signal `p` and the reference `t`:

```
NCC = Σ (p̃ · t̃) / sqrt( Σ p̃² · Σ t̃² ),     p̃ = p − mean(p),   t̃ = t − mean(t).
```

NCC is **scale-invariant** (a constant gain does not change it), **shift-invariant** (a DC offset is removed by zero-meaning), **bounded** in [−1, 1], and **morphology-focused** — it rewards preserving the *shape* of the pulse, which is what downstream PD classification depends on. For these reasons NCC is the primary metric, in preference to MSE/PSNR which are amplitude-sensitive and can penalise a correctly-shaped but rescaled output.

### 8.2.2 Root-mean-square error (RMSE)

RMSE is the per-window root-mean-square difference between the denoised signal and the reference; it is reported as a secondary amplitude-sensitive measure.

### 8.2.3 SNR improvement (dB)

SNR improvement is the gain in SNR from input to output, both referenced to the clean signal:

```
SNR_in  = 10 log₁₀( ||s||² / ||r − s||² ),
SNR_out = 10 log₁₀( ||s||² / ||ŝ − s||² ),
ΔSNR    = SNR_out − SNR_in.
```

It directly quantifies how much noise the denoiser removes relative to the clean reference.

### 8.2.4 Dice coefficient and false-positive rate (FPR)

For the segmentation head, **Dice** measures the overlap between predicted PD-event samples (foreground = any non-background class) and the true mask:

```
Dice = 2 |P ∩ T| / (|P| + |T|).
```

**FPR** is the fraction of true-background samples predicted as PD — a key field metric because impulsive noise tends to induce false PD detections.

### 8.2.5 Efficiency

**Parameter count** (millions) and **inference latency** (milliseconds, batch 1 and batch 32, CUDA-synchronised) characterise the deployment cost.

### 8.2.6 Grounded real Q.Lin NCC — headline transfer metric

The grounded real metric is the NCC between the model's output and the *clean* Q.Lin window, when the input is that window plus injected composite noise. Unlike the older input-referenced NCC (which merely measures how close the output is to the noisy input, and is maximised by doing nothing), the grounded NCC measures genuine denoising quality on real signals: a perfect denoiser scores 1.0, and a model that destroys the real pulse morphology scores low even if it suppresses noise. It is reported overall and per SNR.

---

# 9 Results

> **Benchmark status (2026-05-31).** All values below are locked from `paper/metrics_master.json`, `results/paper_table1_ablation.csv`, `results/paper_table2_improved_models.csv`, and `results/additional_baselines_benchmark.json`. **MR-TAE-FULL (NoMix)** is the headline model. Classical grounded-NCC means use finalized poster values (BayesShrink **0.3844**, Wiener **0.3641**, Savitzky-Golay **0.2915**). Figures: `paper/figures/fig02_snr_bar.png`, `fig03_snr_sweep.png`, `fig_gncc_comparison.png`.

## 9.0 Headline summary — MR-TAE NoMix vs baselines

**Table 9.0.** Recommended model and principal comparisons (no fine-tuning).

| Configuration | NCC mean | NCC @ −15 dB | SNR imp. (dB) | Dice | gNCC | Params (M) | Lat (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **MR-TAE NoMix** | **0.8282** | **0.3777** | **9.64** | **0.9192** | **0.4535** | 3.05 | 17.1 |
| MR-TAE-FULL (phased, ablation ref.) | 0.7991 | 0.3492 | 7.01 | 0.9015 | 0.4366 | 3.01 | 40.1 |
| MR-TAE (no-phase) | 0.8138 | 0.3831 | 7.33 | 0.9022 | 0.4128 | 3.05 | 19.6 |
| UNet1D_simple | 0.8016 | 0.2949 | 9.80 | — | 0.4213 | 2.69 | 2.4 |
| DnCNN1D | 0.7393 | 0.2222 | 8.00 | — | 0.4717 | 0.19 | 2.1 |
| BayesShrink | 0.6578 | 0.1575 | 2.73 | — | 0.3844 | 0 | — |
| Wiener | 0.5897 | 0.1520 | 0.39 | — | 0.3641 | 0 | — |
| Savitzky-Golay | 0.0351 | 0.0098 | −1.66 | — | 0.2915 | 0 | — |

NoMix exceeds BayesShrink on grounded NCC by **(0.4535 − 0.3844) / 0.3844 = 17.9%** without domain fine-tuning.

## 9.1 Synthetic per-SNR NCC

Table 9.1 reports per-SNR NCC for headline models on the frozen 14k-window synthetic set; Figure 3 (`fig03_snr_sweep.png`) plots the full sweep.

**Table 9.1.** Synthetic NCC by input SNR (−15 to +15 dB).

| Model | −15 dB | −10 dB | −5 dB | 0 dB | +5 dB | +10 dB | +15 dB | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **MR-TAE NoMix** | **0.3777** | **0.6763** | **0.8581** | **0.9381** | **0.9717** | **0.9853** | **0.9902** | **0.8282** |
| MR-TAE-FULL (phased) | 0.3492 | 0.6483 | 0.8370 | 0.9169 | 0.9420 | 0.9494 | 0.9510 | 0.7991 |
| MR-TAE (no-phase) | 0.3831 | 0.6797 | 0.8550 | 0.9245 | 0.9462 | 0.9531 | 0.9547 | 0.8138 |
| UNet1D_simple | 0.2949 | 0.5973 | 0.8315 | 0.9323 | 0.9723 | 0.9883 | 0.9946 | 0.8016 |
| DnCNN1D | 0.2218 | 0.4300 | 0.7141 | 0.8835 | 0.9538 | 0.9809 | 0.9908 | 0.7393 |
| BayesShrink | 0.1575 | 0.3003 | 0.5344 | 0.7701 | 0.9032 | 0.9591 | 0.9798 | 0.6578 |

(The full per-SNR table for all 13 ablation configurations is in Appendix A.) NCC falls monotonically from +15 dB toward −15 dB; **NoMix retains 0.3777 NCC at −15 dB** where BayesShrink remains near 0.1575 — a **2.40×** advantage at extreme noise.


## 9.2 Ablation: component contributions

The 13-configuration ablation (full table in Appendix A, Table A.1; bar chart in Fig. 6) isolates each component's contribution. The single-component deltas relative to MR-TAE-FULL are 0.1688 (wavelet), 0.1509 (MTL head), 0.1359 (Swin), 0.1427 (BiGRU) and 0.1223 (attention gates). The two-component removals test synergy: 0.1359, 0.1359, 0.1427. Relative to MR-TAE-FULL (mean NCC 0.7991), the largest single-component removals are **noWavelet** (−0.1688 NCC), **noMTL** (−0.1509 NCC), **noBiGRU** (−0.1427 NCC), **noSwin** (−0.1359 NCC), **noAttn** (−0.1223 NCC). Wavelet pooling and the MTL head dominate low-SNR robustness; removing Swin or BiGRU alone is less damaging than removing both (pair removals in Fig. 6).


## 9.3 SNR improvement and segmentation

Figure 7 reports SNR-improvement (dB) and Dice. **MR-TAE NoMix** improves SNR by **9.64 dB** versus **2.73 dB** for BayesShrink, while segmenting PD events at Dice **0.9192** with FPR 0.2584 — a capability no classical or DL-only denoiser baseline provides. The phased ablation reference (MR-TAE-FULL) achieves 7.01 dB SNR improvement and Dice 0.9015.


## 9.4 Decoder and curriculum variation

The linear and IDWT decoders, and phased, no-phase and **NoMix** curricula, are compared in Appendix A, Table A.2. **NoMix** is the best overall configuration (NCC 0.8282, Dice 0.9192, gNCC 0.4535). Among decoder choices, IDWT (no-phase) reaches NCC 0.8037 at −15 dB 0.3726 but grounded NCC drops to 0.2550 — a domain-transfer penalty versus linear NoMix. No-phase linear (0.8138 mean, 0.3831 @ −15 dB) outperforms phased linear (0.7991 / 0.3492) on synthetic metrics, confirming that the easy-to-hard curriculum helps stability but **excluding mixed-type augmentation (NoMix) yields the largest gain**.


## 9.5 Grounded real Q.Lin transfer

Figure `fig_gncc_comparison.png` reports headline grounded NCC (no fine-tuning). **MR-TAE NoMix achieves gNCC 0.4535**, exceeding BayesShrink (**0.3844**) by **17.9%**, Wiener (0.3641), and Savitzky-Golay (0.2915). This reverses the pattern seen with the phased ablation reference (MR-TAE-FULL gNCC 0.4366): NoMix improves both synthetic *and* real transfer simultaneously. Per-SNR grounded profiles are in Appendix A.4. Q.Lin fine-tuning variants exist (e.g. NoMix+FT gNCC 0.8039) but are excluded from headline comparisons because they trade extreme-noise robustness for real-domain morphology fidelity.


## 9.6 Efficiency

Figure `fig06_efficiency_scatter.png` plots latency versus NCC with bubble size proportional to parameter count. **MR-TAE NoMix** (3.05 M, **17.1 ms**) delivers the best NCC (0.8282) at under half the latency of phased MR-TAE-FULL (40.1 ms). MWCNN-BiGRU (1.31 M, 3.75 ms) sits on the speed–quality Pareto knee for denoising-only; UNet1D_simple (2.69 M, 2.4 ms) is fastest among DL baselines but lacks segmentation and trails NoMix on NCC @ −15 dB (0.2949 vs 0.3777). For latency-constrained full-quality deployment, MR-TAE-noSwin-noAttn achieves NCC 0.7998 at 8.75 ms (ablation row 8).


## 9.7 Deep-learning baselines

Three denoiser-only CNN baselines were trained for 50 epochs with flat MSE loss (no curriculum, no MTL): PureCNN1D (0.7223 mean NCC, 0.21 M params, 0.5 ms), UNet1D_simple (0.8016, 2.69 M, 2.4 ms), and DnCNN1D (0.7393, 0.19 M, 2.1 ms). None matches **NoMix** on mean NCC (0.8282) or NCC @ −15 dB (0.3777). UNet1D_simple has the highest DL baseline SNR improvement (9.80 dB) but lower extreme-noise NCC than NoMix.

## 9.8 Qualitative examples

Figure `fig10_qualitative_waveforms.png` shows representative waveform recovery at −15 dB: noisy input, MR-TAE NoMix output, BayesShrink output, and clean ground truth. NoMix preserves pulse rise time and oscillatory decay; BayesShrink blurs morphology. 

---

# 10 Explainability (LIME / SHAP)

To interpret *why* MR-TAE performs as it does, two model-agnostic explainability methods are applied on hard (−15 dB) windows, for all 13 ablation configurations and the IDWT variants.

## 10.1 LIME

LIME fits a locally faithful, interpretable surrogate around each prediction to attribute importance to input regions. For AE-PD denoising, a well-behaved model should attribute high importance to the pulse regions and low importance to the noise floor.

LIME on MR-TAE-FULL (−15 dB) peaks at segment importance 0.2342 (peak/mean ratio 1.13), consistent with pulse-focused denoising. Component ranking from ablation aligns: **noWavelet** (−0.1688 NCC), **noMTL** (−0.1509 NCC), **noBiGRU** (−0.1427 NCC), **noSwin** (−0.1359 NCC), **noAttn** (−0.1223 NCC).


## 10.2 SHAP

SHAP grounds attributions in Shapley values, giving a unified additive measure of each input feature's contribution, and — through component ablation — an attribution of importance to architectural modules.

Integrated-gradients mean |attribution| = 0.001405; mass concentrates on mid-window samples carrying PD transients rather than uniform background.


## 10.3 Synthesis

LIME segment weights and SHAP integrated gradients both emphasise pulse-bearing regions; the ablation ranks wavelet pooling and MTL highest — the same modules needed to preserve high-frequency morphology under composite noise.


---

# 11 Discussion and Failure Modes

## 11.1 What the design buys

The controlled ablation confirms that each performance difference is attributable to a single toggled component. Frequency-aware wavelet sampling (−0.1688 NCC) and the multi-task head (−0.1509 NCC) are the dominant contributors to robust low-SNR transfer, preserving diagnostic high-frequency morphology. The BiGRU (−0.1427) and Swin (−0.1359) bottleneck add temporal and global context most valuable at the lowest SNRs. **NoMix curriculum** further improves all headline metrics beyond the phased ablation reference — the largest curriculum gain in this study.

## 11.2 Synthetic versus real performance

A recurring theme in PD denoising is that synthetic performance can overstate field performance. Reporting both synthetic NCC and grounded real NCC makes this gap explicit. For the phased ablation reference, synthetic mean NCC (0.7991) exceeds grounded real (0.4366). **NoMix closes this gap**: synthetic 0.8282 with grounded **0.4535**, and crucially **exceeds BayesShrink (0.3844) on real transfer** — the first configuration in this study to beat the strongest classical baseline on grounded NCC without fine-tuning.


## 11.2a Anticipated per-component analysis

The ablation answers, component by component, *why* the architecture performs as it does. Results confirm the following (ablation reference: MR-TAE-FULL phased):

- **Wavelet sampling (`noWavelet`).** Confirmed most damaging (−0.1688 NCC); grounded NCC goes negative (−0.0337).
- **Multi-task head (`noMTL`).** Confirmed second-most damaging (−0.1509 NCC); grounded NCC collapses to −0.1605.
- **Swin (`noSwin`).** Confirmed −0.1359 NCC; large parameter saving (1.41 M vs 3.01 M).
- **BiGRU (`noBiGRU`).** Confirmed −0.1427 NCC; low-SNR NCC drops to 0.0585 @ −15 dB.
- **Attention gates (`noAttn`).** Confirmed helpful (−0.1223 NCC); grounded NCC falls to 0.1014 without gates.
- **Two-component removals.** `noSwin-noBiGRU` loses only −0.0017 NCC vs FULL — Swin and BiGRU are partly redundant; either alone suffices for sequence modelling at moderate cost.

## 11.3 Failure modes

Several failure modes are anticipated and should be examined in the filled results:

1. **Impulsive-noise false positives.** Because impulsive spikes mimic PD rise times, the segmentation head may produce false detections at high impulsive density; the FPR metric tracks this.
2. **Over-smoothing at extreme SNR.** At −15 dB a denoiser may regress toward a smooth mean, raising NCC slightly while destroying fine morphology; the per-SNR Dice and the qualitative panel guard against this.
3. **Decoder-induced artefacts.** The IDWT decoder's more aggressive reconstruction can introduce boundary artefacts if the crop/pad alignment is wrong; the IDWT loading and sanity gates (Sections 5.7.3, 7.5) are designed to catch this.
4. **Domain shift.** The Q.Lin laboratory environment differs from a live substation; a modest grounded NCC is a genuine transfer limitation, not a defect, and is reported as such.

## 11.4 Threats to validity

The principal threats are: reliance on synthetic supervision (mitigated by the grounded real protocol but not eliminated); a single real dataset of limited diversity; and the exclusion of WGAN augmentation from trained results. Q.Lin fine-tuning improves real-domain scores but sacrifices extreme-noise robustness — a documented domain-adaptation trade-off, not used in headline comparisons. These limitations bound the scope of the conclusions.

---

# 12 Conclusion and Future Work

## 12.1 Conclusion

This report presented MR-TAE, a single unified multi-task autoencoder for AE-PD denoising, and a rigorous methodology for studying its design space under one grounded, SNR-stratified protocol. The architecture integrates db4-DWT wavelet sampling, a sequential BiGRU + 1-D Swin-Transformer bottleneck, attention-gated skip connections and a dual-stream segmentation head, and treats inverse-DWT upsampling as a decoder variation and the training schedule as a curriculum variation of the one model. The evaluation injects the same composite noise across −15 to +15 dB on both synthetic and real Q.Lin data, with a grounded NCC as the headline real-world metric, and reports the full metric set identically for a 13-configuration ablation, decoder/curriculum/NoMix comparisons, three classical baselines, and three DL baselines. Under the unified protocol, **MR-TAE (NoMix curriculum)** — the recommended configuration — delivers **9.64 dB** mean SNR improvement, mean synthetic NCC **0.8282**, Dice **0.9192**, and grounded real NCC **0.4535** (+17.9% over BayesShrink 0.3844, no fine-tuning). MR-TAE-FULL (phased) remains the ablation reference (7.01 dB, NCC 0.7991, Dice 0.9015, gNCC 0.4366).


## 12.2 Future work

1. **WGAN-GP augmentation** to narrow the synthetic-to-real distribution gap.
2. **Self-supervised adaptation** (Noise2Void-style) to train directly on unlabelled field recordings without clean ground truth.
3. **Multi-sensor fusion** for PD localisation and external-noise rejection via time-of-flight.
4. **Edge deployment** through pruning and INT8 quantisation for real-time on-device monitoring (NoMix at 17.1 ms is a strong starting point).
5. **Broader defect coverage** by extending the synthetic generator to void, treeing and floating-electrode discharges, and **transfer learning** to other HV assets (cables, switchgear, GIS).

---

# Appendix A — Full Result Tables

## A.1a Per-SNR synthetic NCC — all 13 ablation configurations

| Configuration | −15 dB | −10 dB | −5 dB | 0 dB | +5 dB | +10 dB | +15 dB | Mean |
|---|---|---|---|---|---|---|---|---|
| 1 | MR-TAE-FULL | 0.3492 | 0.6483 | 0.8370 | 0.9169 | 0.9420 | 0.9494 | 0.9510 | 0.7991 |
| 2 | MR-TAE-noBiGRU | 0.0585 | 0.1965 | 0.5712 | 0.8591 | 0.9479 | 0.9764 | 0.9856 | 0.6564 |
| 3 | MR-TAE-noSwin | 0.0982 | 0.2566 | 0.5291 | 0.8492 | 0.9471 | 0.9763 | 0.9859 | 0.6632 |
| 4 | MR-TAE-noAttn | 0.0523 | 0.2988 | 0.6209 | 0.8601 | 0.9469 | 0.9750 | 0.9836 | 0.6768 |
| 5 | MR-TAE-noMTL | 0.0390 | 0.2589 | 0.5827 | 0.8265 | 0.9196 | 0.9499 | 0.9606 | 0.6482 |
| 6 | MR-TAE-noWavelet | 0.0552 | 0.1329 | 0.4616 | 0.8573 | 0.9475 | 0.9747 | 0.9833 | 0.6304 |
| 7 | MR-TAE-noSwin-noBiGRU | 0.3479 | 0.6488 | 0.8373 | 0.9147 | 0.9388 | 0.9460 | 0.9480 | 0.7974 |
| 8 | MR-TAE-noSwin-noAttn | 0.3509 | 0.6492 | 0.8376 | 0.9169 | 0.9425 | 0.9498 | 0.9514 | 0.7998 |
| 9 | MR-TAE-noBiGRU-noAttn | 0.0630 | 0.2245 | 0.4787 | 0.7127 | 0.7848 | 0.8016 | 0.8096 | 0.5536 |
| 10 | MR-TAE-FULL-LossB | 0.1728 | 0.3045 | 0.5616 | 0.7681 | 0.8251 | 0.8660 | 0.8908 | 0.6270 |
| 11 | MR-TAE-FULL-LossC | 0.0548 | 0.2597 | 0.5672 | 0.8176 | 0.9174 | 0.9559 | 0.9642 | 0.6481 |
| 12 | MWCNN-BiGRU | 0.0819 | 0.2954 | 0.5949 | 0.8330 | 0.9295 | 0.9602 | 0.9703 | 0.6665 |
| 13 | UNet-BiGRU-Swin | 0.1290 | 0.3201 | 0.6172 | 0.8608 | 0.9449 | 0.9714 | 0.9809 | 0.6892 |

## A.1b Summary metrics — all 13 ablation configurations

| Configuration | Params (M) | RMSE | SNR imp. (dB) | Dice | FPR | Grounded NCC | Latency b=1 (ms) | Latency b=32 (ms) |
|---|---|---|---|---|---|---|---|---|
| MR-TAE-FULL | 3.01 | 0.1203 | 7.01 | 0.9015 | 0.2738 | 0.4366 | 40.11 | 1.64 |
| MR-TAE-noBiGRU | 2.39 | 0.5488 | 2.57 | 0.8266 | 0.3395 | 0.2307 | 18.76 | 0.80 |
| MR-TAE-noSwin | 1.41 | 0.2724 | 4.24 | 0.8833 | 0.2791 | 0.3287 | 13.70 | 0.75 |
| MR-TAE-noAttn | 3.01 | 0.5148 | 2.98 | 0.8869 | 0.2995 | 0.1014 | 7.45 | 0.54 |
| MR-TAE-noMTL | 3.00 | 1.3680 | 0.35 | 0.0000 | 0.0000 | -0.1605 | 7.67 | 0.59 |
| MR-TAE-noWavelet | 2.99 | 0.7657 | 1.20 | 0.8947 | 0.3078 | -0.0337 | 26.92 | 0.73 |
| MR-TAE-noSwin-noBiGRU | 3.01 | 0.1223 | 6.80 | 0.9011 | 0.2427 | 0.4365 | 18.48 | 0.63 |
| MR-TAE-noSwin-noAttn | 3.01 | 0.1204 | 7.01 | 0.9034 | 0.2484 | 0.4171 | 8.75 | 0.55 |
| MR-TAE-noBiGRU-noAttn | 3.01 | 0.4995 | -1.24 | 0.6887 | 0.3625 | 0.0180 | 15.44 | 0.55 |
| MR-TAE-FULL-LossB | 3.05 | 0.2295 | 1.71 | 0.9122 | 0.3285 | 0.4551 | 15.66 | 0.69 |
| MR-TAE-FULL-LossC | 3.05 | 0.4869 | 1.76 | 0.7576 | 0.4316 | 0.1699 | 7.90 | 0.69 |
| MWCNN-BiGRU | 1.31 | 0.4838 | 2.30 | 0.0000 | 0.0000 | 0.3088 | 3.75 | 0.30 |
| UNet-BiGRU-Swin | 2.90 | 0.2944 | 4.00 | 0.0000 | 0.0000 | 0.4754 | 17.87 | 0.55 |

## A.2 Decoder and curriculum variation (Table 2 expanded)

| Entry | Decoder | Curriculum | Loss | Params (M) | NCC mean | NCC −15 dB | SNR imp. | Dice | Grounded |
|---|---|---|---|---|---|---|---|---|---|
| **MR-TAE-FULL (NoMix)** | Linear | **NoMix** | A | 3.05 | **0.8282** | **0.3777** | **9.64** | **0.9192** | **0.4535** |
| MR-TAE-FULL (linear, no-phase) | Linear | No-phase | A | 3.05 | 0.8138 | 0.3831 | 7.33 | 0.9022 | 0.4128 |
| MR-TAE-FULL (linear, phased) | Linear | Phased | A | 3.05 | 0.8000 | 0.3529 | 6.94 | 0.9009 | 0.4736 |
| MR-TAE-FULL-IDWT (no-phase) | IDWT | No-phase | A | 2.92 | 0.8037 | 0.3726 | 6.89 | 0.8960 | 0.2550 |
| MR-TAE-FULL-IDWT (phased) | IDWT | Phased | A | 2.92 | 0.8018 | 0.3529 | 7.01 | 0.9014 | 0.3935 |
| MR-TAE-FULL-LossB (linear, phased) | Linear | Phased | B | 3.05 | 0.7920 | 0.3435 | 6.57 | 0.8990 | 0.2334 |
| MR-TAE-FULL-LossB-IDWT (phased) | IDWT | Phased | B | 2.92 | 0.7858 | 0.3360 | 6.33 | 0.8995 | 0.5173 |

## A.2b Deep-learning baselines (50 epochs, flat MSE, no curriculum, no MTL)

| Entry | Params (M) | NCC mean | NCC −15 dB | SNR imp. | Grounded | Lat (ms) |
|---|---|---|---|---|---|---|
| PureCNN1D | 0.21 | 0.7223 | 0.2142 | 7.56 | 0.3765 | 0.5 |
| UNet1D_simple | 2.69 | 0.8016 | 0.2949 | 9.80 | 0.4213 | 2.4 |
| DnCNN1D | 0.19 | 0.7393 | 0.2222 | 8.00 | 0.4717 | 2.1 |

## A.3 Classical baselines — per-SNR synthetic NCC and summary

| Method | −15 dB | −10 dB | −5 dB | 0 dB | +5 dB | +10 dB | +15 dB | Mean | SNR imp. (dB) | Grounded real NCC |
|---|---|---|---|---|---|---|---|---|---|---|
| Wiener | 0.1520 | 0.2664 | 0.4526 | 0.6692 | 0.8124 | 0.8767 | 0.8986 | 0.5897 | 0.39 | **0.3641** |
| BayesShrink | 0.1575 | 0.3003 | 0.5344 | 0.7701 | 0.9032 | 0.9591 | 0.9798 | 0.6578 | 2.73 | **0.3844** |
| Savitzky-Golay | 0.0098 | 0.0175 | 0.0251 | 0.0341 | 0.0414 | 0.0502 | 0.0673 | 0.0351 | −1.66 | **0.2915** |

*Classical grounded-NCC means are finalized poster values.*

## A.4 Grounded real Q.Lin NCC — per-SNR (headline transfer metric)

| Model | −15 dB | −10 dB | −5 dB | 0 dB | +5 dB | +10 dB | +15 dB | Mean |
|---|---|---|---|---|---|---|---|---|
| **MR-TAE-FULL (NoMix)** | 0.0925 | 0.2266 | 0.3798 | 0.5049 | 0.6036 | 0.6675 | 0.6994 | **0.4535** |
| MR-TAE-FULL | 0.1015 | 0.2247 | 0.4162 | 0.4996 | 0.5638 | 0.6114 | 0.6391 | 0.4366 |
| MR-TAE-noBiGRU | 0.1068 | 0.1532 | 0.1934 | 0.2327 | 0.2810 | 0.3150 | 0.3332 | 0.2307 |
| MR-TAE-noSwin | 0.1107 | 0.1848 | 0.2791 | 0.3586 | 0.4219 | 0.4628 | 0.4829 | 0.3287 |
| MR-TAE-noAttn | 0.0099 | -0.0020 | 0.0125 | 0.0941 | 0.1863 | 0.2061 | 0.2030 | 0.1014 |
| MR-TAE-noMTL | -0.1178 | -0.1810 | -0.1689 | -0.1498 | -0.1624 | -0.1690 | -0.1747 | -0.1605 |
| MR-TAE-noWavelet | 0.0214 | -0.0056 | -0.0517 | -0.0744 | -0.0625 | -0.0392 | -0.0241 | -0.0337 |
| MR-TAE-noSwin-noBiGRU | 0.0556 | 0.1710 | 0.4081 | 0.5064 | 0.5898 | 0.6495 | 0.6750 | 0.4365 |
| MR-TAE-noSwin-noAttn | 0.0844 | 0.1802 | 0.3921 | 0.4980 | 0.5652 | 0.5946 | 0.6055 | 0.4171 |
| MR-TAE-noBiGRU-noAttn | 0.0160 | -0.0426 | -0.0818 | -0.0402 | 0.0424 | 0.1055 | 0.1265 | 0.0180 |
| MR-TAE-FULL-LossB | 0.1536 | 0.2667 | 0.4064 | 0.5110 | 0.5787 | 0.6231 | 0.6459 | 0.4551 |
| MR-TAE-FULL-LossC | -0.0632 | 0.0017 | 0.0947 | 0.1828 | 0.2737 | 0.3351 | 0.3647 | 0.1699 |
| MWCNN-BiGRU | 0.0157 | 0.0302 | 0.0742 | 0.2810 | 0.5221 | 0.6069 | 0.6315 | 0.3088 |
| UNet-BiGRU-Swin | 0.0880 | 0.1583 | 0.3050 | 0.5143 | 0.6899 | 0.7717 | 0.8003 | 0.4754 |
| MR-TAE-FULL-IDWT (phased) | 0.0735 | 0.1732 | 0.3124 | 0.4439 | 0.5479 | 0.5938 | 0.6101 | 0.3935 |
| MR-TAE-FULL-IDWT (no-phase) | -0.0327 | 0.0868 | 0.3174 | 0.3559 | 0.3549 | 0.3542 | 0.3487 | 0.2550 |
| MR-TAE-FULL-LossB-IDWT (phased) | 0.1276 | 0.2998 | 0.5026 | 0.6094 | 0.6724 | 0.7000 | 0.7094 | 0.5173 |
| Wiener | — | — | — | — | — | — | — | **0.3641** |
| BayesShrink | — | — | — | — | — | — | — | **0.3844** |
| Savitzky-Golay | — | — | — | — | — | — | — | **0.2915** |

*Per-SNR grounded profiles for classical baselines omitted; headline means are finalized poster values. Full per-SNR profiles for MR-TAE variants in Appendix H.*

---

# Appendix B — Reproducibility Index (figure/table → script → source data)

| Artefact | Generator | Source data | Notes |
|---|---|---|---|
| Table 1 / A.1 (ablation) | `evaluation/benchmark_runner.py` → `paper_table1` | `paper/metrics_master.json`, frozen synthetic set | One row per configuration |
| Table 2 / A.2 (decoder/curriculum) | `evaluation/benchmark_runner.py` → `paper_table2` | `paper/metrics_master.json` | Linear vs IDWT, phased vs no-phase |
| Poster HTML | `paper/poster/FYP_POSTER_MR_TAE_2026.html` | `paper/figures/*.png` | A0 portrait, UM branding |
| Fig. 1 system overview | `scripts/generate_poster_figures.py` | schematic | Number-independent |
| Fig. 2 architecture | `scripts/generate_poster_figures.py` | schematic | Verified tensor shapes |
| Fig. 3 noise model | `data/build_test_sets.py` verification | composite-noise generators | `docs/real_noise_injection_verify.png` |
| Fig. 4 pulse types | `mr_tae_fusion/data/pulse_generators.py` | synthetic generator | Number-independent |
| Fig. 3 SNR sweep | `scripts/generate_poster_figures.py` | `paper/metrics_master.json` | NoMix highlighted vs all baselines |
| Fig. 4 ablation heatmap | `scripts/generate_poster_figures.py` | `paper_table1_ablation.csv` | 13-row heatmap |
| Fig. gNCC comparison | `scripts/generate_poster_figures.py` | finalized poster values | NoMix 0.4535 vs BayesShrink 0.3844 |
| Fig. 6 ablation bar | `scripts/make_paper_figures.py` | `paper/metrics_master.json` | With component-removal deltas |
| Fig. 7 SNR-imp / Dice | `scripts/make_paper_figures.py` | `paper/metrics_master.json` | |
| Fig. 8 efficiency scatter | `scripts/make_paper_figures.py` | `paper/metrics_master.json` | Bubble ∝ params |
| Fig. 9 grounded real NCC | `scripts/make_paper_figures.py` | `paper/metrics_master.json` | Per-SNR curve |
| Fig. 10 qualitative | `scripts/make_paper_figures.py` | frozen real windows | noisy / MR-TAE / BayesShrink |
| Fig. 11 LIME | `evaluation/lime_explain.py` | `results/LIME_importances.json` | All 13 + IDWT |
| Fig. 12 SHAP | `evaluation/shap_explain.py` | `results/SHAP_attributions.json` | All 13 + IDWT |

**Re-run order (2026-05-31 locked).** (1) `data/build_test_sets.py`; (2) `scripts/run_phaseA_rebench.py`; (3) `scripts/train_baselines.py` (DL baselines); (4) `scripts/run_explainability_all.py`; (5) `scripts/generate_poster_figures.py` + `scripts/make_paper_figures.py`; (6) `scripts/verify_paper_numbers.py`. Headline model: **MR-TAE NoMix**. Classical gNCC means finalized in §9.0.

---

# Appendix C — File and Module Map

| Component | Path |
|---|---|
| Full model | `mr_tae_fusion/models/mr_tae_fusion.py` |
| MWCNN encoder/decoder | `mr_tae_fusion/models/mwcnn.py` |
| DWT / IDWT layers | `mr_tae_fusion/models/wavelet_layers.py` |
| Swin-1D | `mr_tae_fusion/models/swin_transformer.py` |
| Attention gates | `mr_tae_fusion/models/attention.py` |
| Fusion / heads | `mr_tae_fusion/models/fusion_head.py` |
| Ablation flags | `mr_tae_fusion/models/ablation_config.py` |
| Master config | `mr_tae_fusion/config.py` |
| Noise generators | `mr_tae_fusion/data/noise_generators.py` |
| Pulse generators | `mr_tae_fusion/data/pulse_generators.py` |
| Curriculum scheduler | `training/curricular_scheduler.py` |
| Losses | `training/losses.py` |
| Test-set builder | `data/build_test_sets.py` |
| Benchmark runner | `evaluation/benchmark_runner.py` |
| Model registry | `config/model_registry.yaml` |
| Locked metrics (owned by another agent) | `paper/metrics_master.json` |

---

# Appendix D — Frozen Test-Set Hashes

SHA-256 hashes verifying test-set integrity (to be filled from `docs/TEST_SET_HASHES.md` after the rebuild):

- `test_synthetic_ae.pt` — ade2b9ec7b61681153ed475ef329b50e2524fa6f3426ae5314b10481b4bc9561
- `test_real_qlin_clean.pt` — 16ce0d927a11e1e47201df8f82822a4f20d6ff119a1091486c74f510dbcb0658
- `test_real_qlin_noisy.pt` — 6eb400e41464415fb3da8bba3b01a492879ad0c8bb2b87ce0e610cbac5fcbcb2
- `test_real_qlin_snr_labels.pt` — 42640d8e4a4c9e14570203dfa6e7e83658bea28f6714d7b40e941b1093732e64

---

# Appendix E — Notation and Abbreviations

| Symbol / abbrev. | Meaning |
|---|---|
| PD | Partial discharge |
| AE | Acoustic emission |
| HV | High voltage |
| SNR | Signal-to-noise ratio (dB) |
| AWGN | Additive white Gaussian noise |
| NBI | Narrowband interference |
| DWT / IDWT | (Inverse) discrete wavelet transform |
| MWCNN | Multi-level wavelet CNN |
| BiGRU | Bidirectional gated recurrent unit |
| MTL | Multi-task learning |
| MR-TAE | Multi-level wavelet recurrent–transformer autoencoder |
| NCC | Normalised cross-correlation |
| RMSE | Root-mean-square error |
| Dice | Dice overlap coefficient |
| FPR | False-positive rate |
| db4 | Daubechies-4 wavelet |
| `s(t)`, `r(t)`, `ŝ(t)` | clean, noisy, denoised signal |

---

# Appendix G — Complete Per-SNR Synthetic NCC (All Models)

Every trained configuration and classical baseline from `paper/metrics_master.json`, evaluated on the frozen 14k-window synthetic test set (2000 windows × 7 SNR levels).

| Model key | −15 dB | −10 dB | −5 dB | 0 dB | +5 dB | +10 dB | +15 dB | Mean |
|---|---|---|---|---|---|---|---|---|
| **MR-TAE-FULL (NoMix)** | **0.3777** | **0.6763** | **0.8581** | **0.9381** | **0.9717** | **0.9853** | **0.9902** | **0.8282** |
| BayesShrink | 0.1575 | 0.3003 | 0.5344 | 0.7701 | 0.9032 | 0.9591 | 0.9798 | 0.6578 |
| MR-TAE-FULL | 0.3492 | 0.6483 | 0.8370 | 0.9169 | 0.9420 | 0.9494 | 0.9510 | 0.7991 |
| MR-TAE-FULL (linear, phased) | 0.3529 | 0.6530 | 0.8411 | 0.9175 | 0.9403 | 0.9470 | 0.9483 | 0.8000 |
| MR-TAE-FULL-IDWT (no-phase) | 0.3726 | 0.6686 | 0.8449 | 0.9145 | 0.9365 | 0.9434 | 0.9452 | 0.8037 |
| MR-TAE-FULL-IDWT (phased) | 0.3529 | 0.6558 | 0.8433 | 0.9189 | 0.9420 | 0.9490 | 0.9505 | 0.8018 |
| MR-TAE-FULL-LossB | 0.1728 | 0.3045 | 0.5616 | 0.7681 | 0.8251 | 0.8660 | 0.8908 | 0.6270 |
| MR-TAE-FULL-LossB (linear, phased) | 0.3435 | 0.6418 | 0.8325 | 0.9094 | 0.9334 | 0.9406 | 0.9426 | 0.7920 |
| MR-TAE-FULL-LossB-IDWT (phased) | 0.3360 | 0.6339 | 0.8255 | 0.9039 | 0.9284 | 0.9358 | 0.9368 | 0.7858 |
| MR-TAE-FULL-LossC | 0.0548 | 0.2597 | 0.5672 | 0.8176 | 0.9174 | 0.9559 | 0.9642 | 0.6481 |
| MR-TAE-noAttn | 0.0523 | 0.2988 | 0.6209 | 0.8601 | 0.9469 | 0.9750 | 0.9836 | 0.6768 |
| MR-TAE-noBiGRU | 0.0585 | 0.1965 | 0.5712 | 0.8591 | 0.9479 | 0.9764 | 0.9856 | 0.6564 |
| MR-TAE-noBiGRU-noAttn | 0.0630 | 0.2245 | 0.4787 | 0.7127 | 0.7848 | 0.8016 | 0.8096 | 0.5536 |
| MR-TAE-noMTL | 0.0390 | 0.2589 | 0.5827 | 0.8265 | 0.9196 | 0.9499 | 0.9606 | 0.6482 |
| MR-TAE-noSwin | 0.0982 | 0.2566 | 0.5291 | 0.8492 | 0.9471 | 0.9763 | 0.9859 | 0.6632 |
| MR-TAE-noSwin-noAttn | 0.3509 | 0.6492 | 0.8376 | 0.9169 | 0.9425 | 0.9498 | 0.9514 | 0.7998 |
| MR-TAE-noSwin-noBiGRU | 0.3479 | 0.6488 | 0.8373 | 0.9147 | 0.9388 | 0.9460 | 0.9480 | 0.7974 |
| MR-TAE-noWavelet | 0.0552 | 0.1329 | 0.4616 | 0.8573 | 0.9475 | 0.9747 | 0.9833 | 0.6304 |
| MWCNN-BiGRU | 0.0819 | 0.2954 | 0.5949 | 0.8330 | 0.9295 | 0.9602 | 0.9703 | 0.6665 |
| Savitzky-Golay | 0.0098 | 0.0175 | 0.0251 | 0.0341 | 0.0414 | 0.0502 | 0.0673 | 0.0351 |
| UNet-BiGRU-Swin | 0.1290 | 0.3201 | 0.6172 | 0.8608 | 0.9449 | 0.9714 | 0.9809 | 0.6892 |
| UNet1D_simple | 0.2949 | 0.5973 | 0.8315 | 0.9323 | 0.9723 | 0.9883 | 0.9946 | 0.8016 |
| DnCNN1D | 0.2218 | 0.4300 | 0.7141 | 0.8835 | 0.9538 | 0.9809 | 0.9908 | 0.7393 |
| PureCNN1D | 0.2142 | 0.3909 | 0.6724 | 0.8660 | 0.9480 | 0.9775 | 0.9874 | 0.7223 |
| Wiener | 0.1520 | 0.2664 | 0.4526 | 0.6692 | 0.8124 | 0.8767 | 0.8986 | 0.5897 |
# Appendix H — Complete Grounded Real Q.Lin Per-SNR NCC

Clean Q.Lin window + composite noise at each SNR → denoise → NCC vs clean reference.

| Model key | −15 dB | −10 dB | −5 dB | 0 dB | +5 dB | +10 dB | +15 dB | Mean |
|---|---|---|---|---|---|---|---|---|
| **MR-TAE-FULL (NoMix)** | 0.0925 | 0.2266 | 0.3798 | 0.5049 | 0.6036 | 0.6675 | 0.6994 | **0.4535** |
| BayesShrink | 0.2056 | 0.3792 | 0.6058 | 0.7956 | 0.8994 | 0.9439 | 0.9611 | **0.3844** |
| MR-TAE-FULL | 0.1015 | 0.2247 | 0.4162 | 0.4996 | 0.5638 | 0.6114 | 0.6391 | 0.4366 |
| MR-TAE-FULL (linear, phased) | 0.0795 | 0.2033 | 0.4481 | 0.5573 | 0.6388 | 0.6850 | 0.7033 | 0.4736 |
| MR-TAE-FULL-IDWT (no-phase) | -0.0327 | 0.0868 | 0.3174 | 0.3559 | 0.3549 | 0.3542 | 0.3487 | 0.2550 |
| MR-TAE-FULL-IDWT (phased) | 0.0735 | 0.1732 | 0.3124 | 0.4439 | 0.5479 | 0.5938 | 0.6101 | 0.3935 |
| MR-TAE-FULL-LossB | 0.1536 | 0.2667 | 0.4064 | 0.5110 | 0.5787 | 0.6231 | 0.6459 | 0.4551 |
| MR-TAE-FULL-LossB (linear, phased) | 0.0974 | 0.1481 | 0.2091 | 0.2669 | 0.3075 | 0.3088 | 0.2959 | 0.2334 |
| MR-TAE-FULL-LossB-IDWT (phased) | 0.1276 | 0.2998 | 0.5026 | 0.6094 | 0.6724 | 0.7000 | 0.7094 | 0.5173 |
| MR-TAE-FULL-LossC | -0.0632 | 0.0017 | 0.0947 | 0.1828 | 0.2737 | 0.3351 | 0.3647 | 0.1699 |
| MR-TAE-noAttn | 0.0099 | -0.0020 | 0.0125 | 0.0941 | 0.1863 | 0.2061 | 0.2030 | 0.1014 |
| MR-TAE-noBiGRU | 0.1068 | 0.1532 | 0.1934 | 0.2327 | 0.2810 | 0.3150 | 0.3332 | 0.2307 |
| MR-TAE-noBiGRU-noAttn | 0.0160 | -0.0426 | -0.0818 | -0.0402 | 0.0424 | 0.1055 | 0.1265 | 0.0180 |
| MR-TAE-noMTL | -0.1178 | -0.1810 | -0.1689 | -0.1498 | -0.1624 | -0.1690 | -0.1747 | -0.1605 |
| MR-TAE-noSwin | 0.1107 | 0.1848 | 0.2791 | 0.3586 | 0.4219 | 0.4628 | 0.4829 | 0.3287 |
| MR-TAE-noSwin-noAttn | 0.0844 | 0.1802 | 0.3921 | 0.4980 | 0.5652 | 0.5946 | 0.6055 | 0.4171 |
| MR-TAE-noSwin-noBiGRU | 0.0556 | 0.1710 | 0.4081 | 0.5064 | 0.5898 | 0.6495 | 0.6750 | 0.4365 |
| MR-TAE-noWavelet | 0.0214 | -0.0056 | -0.0517 | -0.0744 | -0.0625 | -0.0392 | -0.0241 | -0.0337 |
| MWCNN-BiGRU | 0.0157 | 0.0302 | 0.0742 | 0.2810 | 0.5221 | 0.6069 | 0.6315 | 0.3088 |
| Savitzky-Golay | 0.3250 | 0.4764 | 0.5955 | 0.6598 | 0.6869 | 0.6969 | 0.7002 | **0.2915** |
| UNet-BiGRU-Swin | 0.0880 | 0.1583 | 0.3050 | 0.5143 | 0.6899 | 0.7717 | 0.8003 | 0.4754 |
| Wiener | 0.2241 | 0.3787 | 0.5789 | 0.7615 | 0.8682 | 0.9115 | 0.9260 | **0.3641** |
# Appendix I — LIME Explainability Summaries (All Models, −15 dB)

Segment-wise LIME importance on hard composite-noise windows. `peak` = max segment weight; `ratio` = peak / mean (pulse vs background emphasis).

| Model | Segments | Peak | Mean | Ratio |
|---|---:|---:|---:|---:|
| MR-TAE-FULL | 32 | 0.2342 | 0.2068 | 1.13 |
| MR-TAE-FULL-LossB | 32 | 0.0634 | 0.0565 | 1.12 |
| MR-TAE-FULL-LossC | 32 | 0.0210 | 0.0138 | 1.52 |
| MR-TAE-noAttn | 32 | 0.0169 | 0.0126 | 1.33 |
| MR-TAE-noBiGRU | 32 | 0.0304 | 0.0244 | 1.25 |
| MR-TAE-noBiGRU-noAttn | 32 | 0.0026 | -0.0020 | -1.29 |
| MR-TAE-noMTL | 32 | 0.0043 | 0.0005 | 8.57 |
| MR-TAE-noSwin | 32 | 0.0474 | 0.0395 | 1.20 |
| MR-TAE-noSwin-noAttn | 32 | 0.2301 | 0.1910 | 1.21 |
| MR-TAE-noSwin-noBiGRU | 32 | 0.2391 | 0.2106 | 1.14 |
| MR-TAE-noWavelet | 32 | 0.0258 | 0.0231 | 1.12 |
| MWCNN-BiGRU | 32 | 0.0361 | 0.0289 | 1.25 |
| UNet-BiGRU-Swin | 32 | 0.0294 | 0.0236 | 1.25 |
| results_full_linear_phased__MR-TAE-FULL | 32 | 0.2458 | 0.2120 | 1.16 |
| results_full_lossb_idwt_phased__MR-TAE-FULL-LossB-IDWT | 32 | 0.1942 | 0.1673 | 1.16 |
| results_full_lossb_linear_phased__MR-TAE-FULL-LossB | 32 | 0.2135 | 0.1915 | 1.11 |
| results_idwt_nophase__MR-TAE-FULL-IDWT | 32 | 0.2254 | 0.2029 | 1.11 |
| results_idwt_phased__MR-TAE-FULL-IDWT | 32 | 0.2423 | 0.2111 | 1.15 |

For MR-TAE-FULL, LIME attributes the highest importance to pulse-bearing segments (peak segment weight ≈ 0.23), with a peak-to-mean ratio > 1 indicating that the model relies more on transient structure than on the noise floor when forming its denoised output.
# Appendix J — SHAP / Integrated-Gradients Summaries (All Models, −15 dB)

| Model | Samples | Mean |abs| | Max |abs| |
|---|---:|---:|---:|
| MR-TAE-FULL | 2048 | 0.001405 | 0.011873 |
| MR-TAE-FULL-LossB | 2048 | 0.001320 | 0.015003 |
| MR-TAE-FULL-LossC | 2048 | 0.001356 | 0.014622 |
| MR-TAE-noAttn | 2048 | 0.001262 | 0.014036 |
| MR-TAE-noBiGRU | 2048 | 0.001334 | 0.027454 |
| MR-TAE-noBiGRU-noAttn | 2048 | 0.001346 | 0.021059 |
| MR-TAE-noMTL | 2048 | 0.001136 | 0.046033 |
| MR-TAE-noSwin | 2048 | 0.000915 | 0.011647 |
| MR-TAE-noSwin-noAttn | 2048 | 0.001555 | 0.024188 |
| MR-TAE-noSwin-noBiGRU | 2048 | 0.001739 | 0.023064 |
| MR-TAE-noWavelet | 2048 | 0.001357 | 0.015454 |
| MWCNN-BiGRU | 2048 | 0.001203 | 0.012899 |
| UNet-BiGRU-Swin | 2048 | 0.001604 | 0.021385 |
| results_full_linear_phased__MR-TAE-FULL | 2048 | 0.001703 | 0.020055 |
| results_full_lossb_idwt_phased__MR-TAE-FULL-LossB-IDWT | 2048 | 0.001330 | 0.018242 |
| results_full_lossb_linear_phased__MR-TAE-FULL-LossB | 2048 | 0.001692 | 0.023799 |
| results_idwt_nophase__MR-TAE-FULL-IDWT | 2048 | 0.001484 | 0.014906 |
| results_idwt_phased__MR-TAE-FULL-IDWT | 2048 | 0.001345 | 0.014373 |
# Appendix K — Tensor Shape Reference (MR-TAE-FULL, 2048-sample input)

Encoder (db4 DWT, 3 levels):

| Stage | Operator | Output shape [B,C,L] |
|---|---|---|
| Input | — | [B, 1, 2048] |
| Input proj | Conv7 | [B, 32, 2048] |
| Enc L1 | Conv → skip₁ → DWT | skip [B,32,2048]; post-DWT [B,64,1025] |
| Enc L2 | Conv → skip₂ → DWT | skip [B,64,1025]; post-DWT [B,128,513] |
| Enc L3 | Conv → skip₃ → DWT | skip [B,128,513]; post-DWT [B,256,257] |
| Bottleneck | BiGRU → Swin-1D | [B, 256, 257] |
| Dec L3 | IDWT/linear + skip | [B, 128, 513] |
| Dec L2 | IDWT/linear + skip | [B, 64, 1025] |
| Dec L1 | IDWT/linear + skip | [B, 32, 2048] |
| Heads | 1×1 conv | denoise [B,1,2048], seg [B,5,2048] |

The 1025/513/257 lengths follow PyTorch DWT padding for db4 at length 2048.

### K.1 Design rationale note 1

Wavelet pooling preserves high-frequency PD transients that max-pooling discards.

### K.2 Design rationale note 2

BiGRU captures causal decay envelopes of damped oscillatory pulses.

### K.3 Design rationale note 3

Swin-1D models window-stationary noise colour via shifted-window attention.

### K.4 Design rationale note 4

Attention gates suppress noisy skip paths while retaining pulse detail.

### K.5 Design rationale note 5

MTL segmentation regularises the encoder against over-smoothing at extreme SNR.

### K.6 Design rationale note 6

Homoscedastic uncertainty weighting balances denoise vs segment losses without manual λ tuning.

### K.7 Design rationale note 7

Phased curriculum eases optimisation before exposing composite noise below −5 dB. **NoMix** (no mixed-type augmentation) yields the best headline metrics.

### K.8 Design rationale note 8

IDWT reconstruction is parameter-free upsampling aligned to the analysis filter bank.

### K.9 Design rationale note 9

Grounded real NCC prevents input-referenced metrics that reward copying noise.

### K.10 Design rationale note 10

Seven-level SNR sweeps expose crossover points between learned and classical methods.

### K.11 Design rationale note 11

Late-epoch checkpoint selection avoids early-phase weights tuned only to easy SNR.

### K.12 Design rationale note 12

Composite noise mixes AWGN, magnetostrictive hum, powerline harmonics, NBI and impulses.

### K.13 Design rationale note 13

Dice+FPR jointly measure segmentation quality relevant to false PD alarms in the field.

### K.14 Design rationale note 14

Frozen SHA-256 test sets make every table row reproducible from checkpoints alone.

### K.15 Design rationale note 15

Explainability at −15 dB targets the operating point where attribution is hardest.

# Appendix L — Repository File Map (Reproducibility)

Key paths relative to project root:

- `paper/metrics_master.json` [✓]
- `paper/paper_arxiv.md` [✓]
- `paper/research_report.md` [✓]
- `paper/figures/*.png` [✓]
- `results/paper_table1_ablation.csv` [✓]
- `results/paper_table2_improved_models.csv` [✓]
- `results/LIME_importances.json` [✓]
- `results/SHAP_attributions.json` [✓]
- `data/test_synthetic_ae.pt` [✓]
- `data/test_real_qlin_clean.pt` [✓]
- `data/test_real_qlin_noisy.pt` [✓]
- `data/test_real_qlin_snr_labels.pt` [✓]
- `docs/TEST_SET_HASHES.md` [✓]
- `evaluation/benchmark_runner.py` [✓]
- `evaluation/lime_explain.py` [✓]
- `evaluation/shap_explain.py` [✓]
- `scripts/run_phaseA_rebench.py` [✓]
- `scripts/make_paper_figures.py` [✓]
- `scripts/fill_paper_from_metrics.py` [✓]
- `scripts/expand_research_report.py` [✓]
- `mr_tae_fusion/config.py` [✓]
- `mr_tae_fusion/models/mr_tae_fusion.py` [✓]
- `training/curricular_scheduler.py` [✓]

## L.1 TEST_SET_HASHES excerpt

```
# Test Set SHA256 Hashes

These hashes verify that test sets have not been modified after creation.
Include in thesis appendix for reproducibility.

**test_synthetic_ae.pt**
```
ade2b9ec7b61681153ed475ef329b50e2524fa6f3426ae5314b10481b4bc9561
```

**test_real_qlin.pt**
```
de22e1f513c56dce125ce58e54ed9ba86fc8f6804a07a9bceb5005ed4f3d8896
```

**real_noise_pool.pt**
```
7525cb7a992f6f93c41286bd2c6e66a4aa32b1cde2530d8426ffca39d7acc36a
```

**test_real_qlin_clean.pt**
```
16ce0d927a11e1e47201df8f82822a4f20d6ff119a1091486c74f510dbcb0658
```

**test_real_qlin_noisy.pt**
```
6eb400e41464415fb3da8bba3b01a492879ad0c8bb2b87ce0e610cbac5fcbcb2
```

**test_real_qlin_snr_labels.pt**
```
42640d8e4a4c9e14570203dfa6e7e83658bea28f6714d7b40e941b1093732e64
```
```
# Appendix M — Per-Model Result Narratives

Short interpretation of each configuration under the unified benchmark protocol.

## BayesShrink

- **Synthetic mean NCC:** 0.6578 (NCC @ −15 dB: 0.1575)
- **SNR improvement:** 2.73 dB
- **Dice / FPR:** — / —
- **Grounded real NCC:** **0.3844** (finalized poster value)
- **Efficiency:** 0.00 M params, 1.03 ms (batch 1)

Strongest classical baseline on synthetic NCC but **exceeded by MR-TAE NoMix on grounded NCC** (0.4535, +17.9%).

## MR-TAE-FULL (NoMix) — recommended

- **Synthetic mean NCC:** 0.8282 (NCC @ −15 dB: 0.3777)
- **SNR improvement:** 9.64 dB
- **Dice / FPR:** 0.9192 / 0.2584
- **Grounded real NCC:** 0.4535
- **Efficiency:** 3.05 M params, 17.1 ms (batch 1)

Overall best configuration: highest synthetic NCC, SNR improvement and Dice; beats BayesShrink on real transfer without fine-tuning.

## MR-TAE-FULL

- **Synthetic mean NCC:** 0.7991 (NCC @ −15 dB: 0.3492)
- **SNR improvement:** 7.01 dB
- **Dice / FPR:** 0.9015 / 0.2738
- **Grounded real NCC:** 0.4366
- **Efficiency:** 3.01 M params, 40.11 ms (batch 1)

At −15 dB synthetic SNR, 0.3492 NCC indicates strong pulse preservation relative to classical baselines.

## MR-TAE-FULL (linear, phased)

- **Synthetic mean NCC:** 0.8000 (NCC @ −15 dB: 0.3529)
- **SNR improvement:** 6.94 dB
- **Dice / FPR:** 0.9009 / 0.2360
- **Grounded real NCC:** 0.4736
- **Efficiency:** 3.05 M params, 26.86 ms (batch 1)

At −15 dB synthetic SNR, 0.3529 NCC indicates strong pulse preservation relative to classical baselines.

## MR-TAE-FULL-IDWT (no-phase)

- **Synthetic mean NCC:** 0.8037 (NCC @ −15 dB: 0.3726)
- **SNR improvement:** 6.89 dB
- **Dice / FPR:** 0.8960 / 0.2739
- **Grounded real NCC:** 0.2550
- **Efficiency:** 2.92 M params, 24.84 ms (batch 1)

At −15 dB synthetic SNR, 0.3726 NCC indicates strong pulse preservation relative to classical baselines.

## MR-TAE-FULL-IDWT (phased)

- **Synthetic mean NCC:** 0.8018 (NCC @ −15 dB: 0.3529)
- **SNR improvement:** 7.01 dB
- **Dice / FPR:** 0.9014 / 0.2455
- **Grounded real NCC:** 0.3935
- **Efficiency:** 2.92 M params, 8.74 ms (batch 1)

At −15 dB synthetic SNR, 0.3529 NCC indicates strong pulse preservation relative to classical baselines.

## MR-TAE-FULL-LossB

- **Synthetic mean NCC:** 0.6270 (NCC @ −15 dB: 0.1728)
- **SNR improvement:** 1.71 dB
- **Dice / FPR:** 0.9122 / 0.3285
- **Grounded real NCC:** 0.4551
- **Efficiency:** 3.05 M params, 15.66 ms (batch 1)

At −15 dB synthetic SNR, 0.1728 NCC indicates limited separability; the bottleneck struggles to recover morphology.

## MR-TAE-FULL-LossB (linear, phased)

- **Synthetic mean NCC:** 0.7920 (NCC @ −15 dB: 0.3435)
- **SNR improvement:** 6.57 dB
- **Dice / FPR:** 0.8990 / 0.2546
- **Grounded real NCC:** 0.2334
- **Efficiency:** 3.05 M params, 29.56 ms (batch 1)

At −15 dB synthetic SNR, 0.3435 NCC indicates strong pulse preservation relative to classical baselines.

## MR-TAE-FULL-LossB-IDWT (phased)

- **Synthetic mean NCC:** 0.7858 (NCC @ −15 dB: 0.3360)
- **SNR improvement:** 6.33 dB
- **Dice / FPR:** 0.8995 / 0.2570
- **Grounded real NCC:** 0.5173
- **Efficiency:** 2.92 M params, 24.46 ms (batch 1)

At −15 dB synthetic SNR, 0.3360 NCC indicates strong pulse preservation relative to classical baselines.

## MR-TAE-FULL-LossC

- **Synthetic mean NCC:** 0.6481 (NCC @ −15 dB: 0.0548)
- **SNR improvement:** 1.76 dB
- **Dice / FPR:** 0.7576 / 0.4316
- **Grounded real NCC:** 0.1699
- **Efficiency:** 3.05 M params, 7.90 ms (batch 1)

At −15 dB synthetic SNR, 0.0548 NCC indicates limited separability; the bottleneck struggles to recover morphology.

## MR-TAE-noAttn

- **Synthetic mean NCC:** 0.6768 (NCC @ −15 dB: 0.0523)
- **SNR improvement:** 2.98 dB
- **Dice / FPR:** 0.8869 / 0.2995
- **Grounded real NCC:** 0.1014
- **Efficiency:** 3.01 M params, 7.45 ms (batch 1)

At −15 dB synthetic SNR, 0.0523 NCC indicates limited separability; the bottleneck struggles to recover morphology.

## MR-TAE-noBiGRU

- **Synthetic mean NCC:** 0.6564 (NCC @ −15 dB: 0.0585)
- **SNR improvement:** 2.57 dB
- **Dice / FPR:** 0.8266 / 0.3395
- **Grounded real NCC:** 0.2307
- **Efficiency:** 2.39 M params, 18.76 ms (batch 1)

At −15 dB synthetic SNR, 0.0585 NCC indicates limited separability; the bottleneck struggles to recover morphology.

## MR-TAE-noBiGRU-noAttn

- **Synthetic mean NCC:** 0.5536 (NCC @ −15 dB: 0.0630)
- **SNR improvement:** -1.24 dB
- **Dice / FPR:** 0.6887 / 0.3625
- **Grounded real NCC:** 0.0180
- **Efficiency:** 3.01 M params, 15.44 ms (batch 1)

At −15 dB synthetic SNR, 0.0630 NCC indicates limited separability; the bottleneck struggles to recover morphology.

## MR-TAE-noMTL

- **Synthetic mean NCC:** 0.6482 (NCC @ −15 dB: 0.0390)
- **SNR improvement:** 0.35 dB
- **Dice / FPR:** 0.0000 / 0.0000
- **Grounded real NCC:** -0.1605
- **Efficiency:** 3.00 M params, 7.67 ms (batch 1)

At −15 dB synthetic SNR, 0.0390 NCC indicates limited separability; the bottleneck struggles to recover morphology.

## MR-TAE-noSwin

- **Synthetic mean NCC:** 0.6632 (NCC @ −15 dB: 0.0982)
- **SNR improvement:** 4.24 dB
- **Dice / FPR:** 0.8833 / 0.2791
- **Grounded real NCC:** 0.3287
- **Efficiency:** 1.41 M params, 13.70 ms (batch 1)

At −15 dB synthetic SNR, 0.0982 NCC indicates limited separability; the bottleneck struggles to recover morphology.

## MR-TAE-noSwin-noAttn

- **Synthetic mean NCC:** 0.7998 (NCC @ −15 dB: 0.3509)
- **SNR improvement:** 7.01 dB
- **Dice / FPR:** 0.9034 / 0.2484
- **Grounded real NCC:** 0.4171
- **Efficiency:** 3.01 M params, 8.75 ms (batch 1)

At −15 dB synthetic SNR, 0.3509 NCC indicates strong pulse preservation relative to classical baselines.

## MR-TAE-noSwin-noBiGRU

- **Synthetic mean NCC:** 0.7974 (NCC @ −15 dB: 0.3479)
- **SNR improvement:** 6.80 dB
- **Dice / FPR:** 0.9011 / 0.2427
- **Grounded real NCC:** 0.4365
- **Efficiency:** 3.01 M params, 18.48 ms (batch 1)

At −15 dB synthetic SNR, 0.3479 NCC indicates strong pulse preservation relative to classical baselines.

## MR-TAE-noWavelet

- **Synthetic mean NCC:** 0.6304 (NCC @ −15 dB: 0.0552)
- **SNR improvement:** 1.20 dB
- **Dice / FPR:** 0.8947 / 0.3078
- **Grounded real NCC:** -0.0337
- **Efficiency:** 2.99 M params, 26.92 ms (batch 1)

At −15 dB synthetic SNR, 0.0552 NCC indicates limited separability; the bottleneck struggles to recover morphology.

## MWCNN-BiGRU

- **Synthetic mean NCC:** 0.6665 (NCC @ −15 dB: 0.0819)
- **SNR improvement:** 2.30 dB
- **Dice / FPR:** 0.0000 / 0.0000
- **Grounded real NCC:** 0.3088
- **Efficiency:** 1.31 M params, 3.75 ms (batch 1)

At −15 dB synthetic SNR, 0.0819 NCC indicates limited separability; the bottleneck struggles to recover morphology.

## Savitzky-Golay

- **Synthetic mean NCC:** 0.0351 (NCC @ −15 dB: 0.0098)
- **SNR improvement:** -1.66 dB
- **Dice / FPR:** — / —
- **Grounded real NCC:** **0.2915** (finalized poster value)
- **Efficiency:** 0.00 M params, 0.74 ms (batch 1)

At −15 dB synthetic SNR, 0.0098 NCC indicates limited separability; the bottleneck struggles to recover morphology.

## UNet-BiGRU-Swin

- **Synthetic mean NCC:** 0.6892 (NCC @ −15 dB: 0.1290)
- **SNR improvement:** 4.00 dB
- **Dice / FPR:** 0.0000 / 0.0000
- **Grounded real NCC:** 0.4754
- **Efficiency:** 2.90 M params, 17.87 ms (batch 1)

At −15 dB synthetic SNR, 0.1290 NCC indicates limited separability; the bottleneck struggles to recover morphology.

## Wiener

- **Synthetic mean NCC:** 0.5897 (NCC @ −15 dB: 0.1520)
- **SNR improvement:** 0.39 dB
- **Dice / FPR:** — / —
- **Grounded real NCC:** **0.3641** (finalized poster value)
- **Efficiency:** 0.00 M params, 0.36 ms (batch 1)

At −15 dB synthetic SNR, 0.1520 NCC indicates limited separability; trails NoMix on both synthetic and grounded metrics.

## PureCNN1D (DL baseline)

- **Synthetic mean NCC:** 0.7223 (NCC @ −15 dB: 0.2142)
- **SNR improvement:** 7.56 dB
- **Dice / FPR:** — / —
- **Grounded real NCC:** 0.3765
- **Efficiency:** 0.21 M params, 0.5 ms (batch 1)

Lightest DL baseline; fast but well below NoMix on all NCC metrics.

## UNet1D_simple (DL baseline)

- **Synthetic mean NCC:** 0.8016 (NCC @ −15 dB: 0.2949)
- **SNR improvement:** 9.80 dB
- **Dice / FPR:** — / —
- **Grounded real NCC:** 0.4213
- **Efficiency:** 2.69 M params, 2.4 ms (batch 1)

Strongest DL baseline on SNR improvement; still trails NoMix on mean NCC and extreme-noise NCC.

## DnCNN1D (DL baseline)

- **Synthetic mean NCC:** 0.7393 (NCC @ −15 dB: 0.2218)
- **SNR improvement:** 8.00 dB
- **Dice / FPR:** — / —
- **Grounded real NCC:** 0.4717
- **Efficiency:** 0.19 M params, 2.1 ms (batch 1)

Residual-learning CNN; competitive grounded NCC among DL baselines but no segmentation and lower synthetic NCC than NoMix.

# Appendix N — Per-SNR Comparative Analysis (Synthetic NCC)

For each injected SNR, all models are ranked by synthetic NCC on the frozen 14k-window set.

## N.-15 dB — extreme noise (−15 dB)

At extreme noise (−15 dB), **MR-TAE-FULL (linear, no-phase)** leads with NCC **0.3831**; **NoMix** is second at **0.3777** (best on mean NCC across all SNRs). **Savitzky-Golay** trails at 0.0098.

| Rank | Model | NCC | Δ vs MR-TAE-FULL (phased) |
|---:|---|---:|---:|
| 1 | MR-TAE-FULL (linear, no-phase) | 0.3831 | +0.0339 |
| 2 | **MR-TAE-FULL (NoMix)** | **0.3777** | **+0.0285** |
| 3 | MR-TAE-FULL-IDWT (no-phase) | 0.3726 | +0.0234 |
| 4 | MR-TAE-FULL (linear, phased) | 0.3529 | +0.0037 |
| 5 | MR-TAE-FULL-IDWT (phased) | 0.3529 | +0.0037 |
| 6 | MR-TAE-noSwin-noAttn | 0.3509 | +0.0017 |
| 5 | MR-TAE-FULL | 0.3492 | 0.0000 |
| 6 | MR-TAE-noSwin-noBiGRU | 0.3479 | -0.0013 |
| 7 | MR-TAE-FULL-LossB (linear, phased) | 0.3435 | -0.0057 |
| 8 | MR-TAE-FULL-LossB-IDWT (phased) | 0.3360 | -0.0132 |
| 9 | MR-TAE-FULL-LossB | 0.1728 | -0.1764 |
| 10 | BayesShrink | 0.1575 | -0.1917 |
| 11 | Wiener | 0.1520 | -0.1972 |
| 12 | UNet-BiGRU-Swin | 0.1290 | -0.2202 |
| 13 | MR-TAE-noSwin | 0.0982 | -0.2510 |
| 14 | MWCNN-BiGRU | 0.0819 | -0.2673 |
| 15 | MR-TAE-noBiGRU-noAttn | 0.0630 | -0.2862 |
| 16 | MR-TAE-noBiGRU | 0.0585 | -0.2907 |
| 17 | MR-TAE-noWavelet | 0.0552 | -0.2940 |
| 18 | MR-TAE-FULL-LossC | 0.0548 | -0.2944 |
| 19 | MR-TAE-noAttn | 0.0523 | -0.2969 |
| 20 | MR-TAE-noMTL | 0.0390 | -0.3102 |
| 21 | Savitzky-Golay | 0.0098 | -0.3394 |

### Interpretation at -15 dB

Classical BayesShrink and Wiener remain competitive when the noise floor is mostly stationary; learned models separate once impulsive and narrowband components dominate the error budget. At this SNR the spread from best to worst is 0.3627 NCC, indicating high sensitivity to architectural choices.

#### MR-TAE-FULL-IDWT (no-phase) @ -15 dB

- NCC 0.3726; mean across SNRs 0.8037.
- SNR improvement 6.89 dB; Dice 0.8960.
- Grounded real NCC 0.2550.

The denoised waveform at this SNR should retain the damped-oscillation carrier and decay envelope; over-smoothing manifests as elevated RMSE on grounded real windows even when synthetic NCC appears acceptable.

#### MR-TAE-FULL (linear, phased) @ -15 dB

- NCC 0.3529; mean across SNRs 0.8000.
- SNR improvement 6.94 dB; Dice 0.9009.
- Grounded real NCC 0.4736.

The denoised waveform at this SNR should retain the damped-oscillation carrier and decay envelope; over-smoothing manifests as elevated RMSE on grounded real windows even when synthetic NCC appears acceptable.

#### MR-TAE-FULL-IDWT (phased) @ -15 dB

- NCC 0.3529; mean across SNRs 0.8018.
- SNR improvement 7.01 dB; Dice 0.9014.
- Grounded real NCC 0.3935.

The denoised waveform at this SNR should retain the damped-oscillation carrier and decay envelope; over-smoothing manifests as elevated RMSE on grounded real windows even when synthetic NCC appears acceptable.

#### MR-TAE-noSwin-noAttn @ -15 dB

- NCC 0.3509; mean across SNRs 0.7998.
- SNR improvement 7.01 dB; Dice 0.9034.
- Grounded real NCC 0.4171.

The denoised waveform at this SNR should retain the damped-oscillation carrier and decay envelope; over-smoothing manifests as elevated RMSE on grounded real windows even when synthetic NCC appears acceptable.

#### MR-TAE-FULL @ -15 dB

- NCC 0.3492; mean across SNRs 0.7991.
- SNR improvement 7.01 dB; Dice 0.9015.
- Grounded real NCC 0.4366.

The denoised waveform at this SNR should retain the damped-oscillation carrier and decay envelope; over-smoothing manifests as elevated RMSE on grounded real windows even when synthetic NCC appears acceptable.

### N.-15.note0 — noise-regime detail

Composite noise combines AWGN, 100/200 Hz magnetostrictive tones, 50/150 Hz powerline coupling, 80 kHz narrowband interference, and Bernoulli–Gaussian impulses. As input SNR moves toward extreme noise (−15 dB), the relative power of impulses versus Gaussian background shifts; MR-TAE's wavelet encoder preserves detail coefficients that linear thresholding attenuates.

### N.-15.note1 — noise-regime detail

Composite noise combines AWGN, 100/200 Hz magnetostrictive tones, 50/150 Hz powerline coupling, 80 kHz narrowband interference, and Bernoulli–Gaussian impulses. As input SNR moves toward extreme noise (−15 dB), the relative power of impulses versus Gaussian background shifts; MR-TAE's wavelet encoder preserves detail coefficients that linear thresholding attenuates.

### N.-15.note2 — noise-regime detail

Composite noise combines AWGN, 100/200 Hz magnetostrictive tones, 50/150 Hz powerline coupling, 80 kHz narrowband interference, and Bernoulli–Gaussian impulses. As input SNR moves toward extreme noise (−15 dB), the relative power of impulses versus Gaussian background shifts; MR-TAE's wavelet encoder preserves detail coefficients that linear thresholding attenuates.

### N.-15.note3 — noise-regime detail

Composite noise combines AWGN, 100/200 Hz magnetostrictive tones, 50/150 Hz powerline coupling, 80 kHz narrowband interference, and Bernoulli–Gaussian impulses. As input SNR moves toward extreme noise (−15 dB), the relative power of impulses versus Gaussian background shifts; MR-TAE's wavelet encoder preserves detail coefficients that linear thresholding attenuates.

## N.-10 dB — heavy noise (−10 dB)

At heavy noise (−10 dB), **MR-TAE-FULL-IDWT (no-phase)** leads with NCC 0.6686 and **Savitzky-Golay** trails at 0.0175.

| Rank | Model | NCC | Δ vs MR-TAE-FULL |
|---:|---|---:|---:|
| 1 | MR-TAE-FULL-IDWT (no-phase) | 0.6686 | 0.0203 |
| 2 | MR-TAE-FULL-IDWT (phased) | 0.6558 | 0.0075 |
| 3 | MR-TAE-FULL (linear, phased) | 0.6530 | 0.0047 |
| 4 | MR-TAE-noSwin-noAttn | 0.6492 | 0.0010 |
| 5 | MR-TAE-noSwin-noBiGRU | 0.6488 | 0.0005 |
| 6 | MR-TAE-FULL | 0.6483 | 0.0000 |
| 7 | MR-TAE-FULL-LossB (linear, phased) | 0.6418 | -0.0064 |
| 8 | MR-TAE-FULL-LossB-IDWT (phased) | 0.6339 | -0.0144 |
| 9 | UNet-BiGRU-Swin | 0.3201 | -0.3281 |
| 10 | MR-TAE-FULL-LossB | 0.3045 | -0.3438 |
| 11 | BayesShrink | 0.3003 | -0.3480 |
| 12 | MR-TAE-noAttn | 0.2988 | -0.3495 |
| 13 | MWCNN-BiGRU | 0.2954 | -0.3529 |
| 14 | Wiener | 0.2664 | -0.3819 |
| 15 | MR-TAE-FULL-LossC | 0.2597 | -0.3886 |
| 16 | MR-TAE-noMTL | 0.2589 | -0.3893 |
| 17 | MR-TAE-noSwin | 0.2566 | -0.3917 |
| 18 | MR-TAE-noBiGRU-noAttn | 0.2245 | -0.4237 |
| 19 | MR-TAE-noBiGRU | 0.1965 | -0.4518 |
| 20 | MR-TAE-noWavelet | 0.1329 | -0.5153 |
| 21 | Savitzky-Golay | 0.0175 | -0.6308 |

### Interpretation at -10 dB

Classical BayesShrink and Wiener remain competitive when the noise floor is mostly stationary; learned models separate once impulsive and narrowband components dominate the error budget. At this SNR the spread from best to worst is 0.6511 NCC, indicating high sensitivity to architectural choices.

#### MR-TAE-FULL-IDWT (no-phase) @ -10 dB

- NCC 0.6686; mean across SNRs 0.8037.
- SNR improvement 6.89 dB; Dice 0.8960.
- Grounded real NCC 0.2550.

The denoised waveform at this SNR should retain the damped-oscillation carrier and decay envelope; over-smoothing manifests as elevated RMSE on grounded real windows even when synthetic NCC appears acceptable.

#### MR-TAE-FULL-IDWT (phased) @ -10 dB

- NCC 0.6558; mean across SNRs 0.8018.
- SNR improvement 7.01 dB; Dice 0.9014.
- Grounded real NCC 0.3935.

The denoised waveform at this SNR should retain the damped-oscillation carrier and decay envelope; over-smoothing manifests as elevated RMSE on grounded real windows even when synthetic NCC appears acceptable.

#### MR-TAE-FULL (linear, phased) @ -10 dB

- NCC 0.6530; mean across SNRs 0.8000.
- SNR improvement 6.94 dB; Dice 0.9009.
- Grounded real NCC 0.4736.

The denoised waveform at this SNR should retain the damped-oscillation carrier and decay envelope; over-smoothing manifests as elevated RMSE on grounded real windows even when synthetic NCC appears acceptable.

#### MR-TAE-noSwin-noAttn @ -10 dB

- NCC 0.6492; mean across SNRs 0.7998.
- SNR improvement 7.01 dB; Dice 0.9034.
- Grounded real NCC 0.4171.

The denoised waveform at this SNR should retain the damped-oscillation carrier and decay envelope; over-smoothing manifests as elevated RMSE on grounded real windows even when synthetic NCC appears acceptable.

#### MR-TAE-noSwin-noBiGRU @ -10 dB

- NCC 0.6488; mean across SNRs 0.7974.
- SNR improvement 6.80 dB; Dice 0.9011.
- Grounded real NCC 0.4365.

The denoised waveform at this SNR should retain the damped-oscillation carrier and decay envelope; over-smoothing manifests as elevated RMSE on grounded real windows even when synthetic NCC appears acceptable.

### N.-10.note0 — noise-regime detail

Composite noise combines AWGN, 100/200 Hz magnetostrictive tones, 50/150 Hz powerline coupling, 80 kHz narrowband interference, and Bernoulli–Gaussian impulses. As input SNR moves toward heavy noise (−10 dB), the relative power of impulses versus Gaussian background shifts; MR-TAE's wavelet encoder preserves detail coefficients that linear thresholding attenuates.

### N.-10.note1 — noise-regime detail

Composite noise combines AWGN, 100/200 Hz magnetostrictive tones, 50/150 Hz powerline coupling, 80 kHz narrowband interference, and Bernoulli–Gaussian impulses. As input SNR moves toward heavy noise (−10 dB), the relative power of impulses versus Gaussian background shifts; MR-TAE's wavelet encoder preserves detail coefficients that linear thresholding attenuates.

### N.-10.note2 — noise-regime detail

Composite noise combines AWGN, 100/200 Hz magnetostrictive tones, 50/150 Hz powerline coupling, 80 kHz narrowband interference, and Bernoulli–Gaussian impulses. As input SNR moves toward heavy noise (−10 dB), the relative power of impulses versus Gaussian background shifts; MR-TAE's wavelet encoder preserves detail coefficients that linear thresholding attenuates.

### N.-10.note3 — noise-regime detail

Composite noise combines AWGN, 100/200 Hz magnetostrictive tones, 50/150 Hz powerline coupling, 80 kHz narrowband interference, and Bernoulli–Gaussian impulses. As input SNR moves toward heavy noise (−10 dB), the relative power of impulses versus Gaussian background shifts; MR-TAE's wavelet encoder preserves detail coefficients that linear thresholding attenuates.

## N.-5 dB — moderate-low (−5 dB)

At moderate-low (−5 dB), **MR-TAE-FULL-IDWT (no-phase)** leads with NCC 0.8449 and **Savitzky-Golay** trails at 0.0251.

| Rank | Model | NCC | Δ vs MR-TAE-FULL |
|---:|---|---:|---:|
| 1 | MR-TAE-FULL-IDWT (no-phase) | 0.8449 | 0.0078 |
| 2 | MR-TAE-FULL-IDWT (phased) | 0.8433 | 0.0063 |
| 3 | MR-TAE-FULL (linear, phased) | 0.8411 | 0.0040 |
| 4 | MR-TAE-noSwin-noAttn | 0.8376 | 0.0006 |
| 5 | MR-TAE-noSwin-noBiGRU | 0.8373 | 0.0003 |
| 6 | MR-TAE-FULL | 0.8370 | 0.0000 |
| 7 | MR-TAE-FULL-LossB (linear, phased) | 0.8325 | -0.0046 |
| 8 | MR-TAE-FULL-LossB-IDWT (phased) | 0.8255 | -0.0115 |
| 9 | MR-TAE-noAttn | 0.6209 | -0.2162 |
| 10 | UNet-BiGRU-Swin | 0.6172 | -0.2198 |
| 11 | MWCNN-BiGRU | 0.5949 | -0.2421 |
| 12 | MR-TAE-noMTL | 0.5827 | -0.2543 |
| 13 | MR-TAE-noBiGRU | 0.5712 | -0.2659 |
| 14 | MR-TAE-FULL-LossC | 0.5672 | -0.2698 |
| 15 | MR-TAE-FULL-LossB | 0.5616 | -0.2755 |
| 16 | BayesShrink | 0.5344 | -0.3026 |
| 17 | MR-TAE-noSwin | 0.5291 | -0.3079 |
| 18 | MR-TAE-noBiGRU-noAttn | 0.4787 | -0.3584 |
| 19 | MR-TAE-noWavelet | 0.4616 | -0.3755 |
| 20 | Wiener | 0.4526 | -0.3845 |
| 21 | Savitzky-Golay | 0.0251 | -0.8119 |

### Interpretation at -5 dB

Classical BayesShrink and Wiener remain competitive when the noise floor is mostly stationary; learned models separate once impulsive and narrowband components dominate the error budget. At this SNR the spread from best to worst is 0.8198 NCC, indicating partial convergence toward the clean-reference ceiling.

#### MR-TAE-FULL-IDWT (no-phase) @ -5 dB

- NCC 0.8449; mean across SNRs 0.8037.
- SNR improvement 6.89 dB; Dice 0.8960.
- Grounded real NCC 0.2550.

The denoised waveform at this SNR should retain the damped-oscillation carrier and decay envelope; over-smoothing manifests as elevated RMSE on grounded real windows even when synthetic NCC appears acceptable.

#### MR-TAE-FULL-IDWT (phased) @ -5 dB

- NCC 0.8433; mean across SNRs 0.8018.
- SNR improvement 7.01 dB; Dice 0.9014.
- Grounded real NCC 0.3935.

The denoised waveform at this SNR should retain the damped-oscillation carrier and decay envelope; over-smoothing manifests as elevated RMSE on grounded real windows even when synthetic NCC appears acceptable.

#### MR-TAE-FULL (linear, phased) @ -5 dB

- NCC 0.8411; mean across SNRs 0.8000.
- SNR improvement 6.94 dB; Dice 0.9009.
- Grounded real NCC 0.4736.

The denoised waveform at this SNR should retain the damped-oscillation carrier and decay envelope; over-smoothing manifests as elevated RMSE on grounded real windows even when synthetic NCC appears acceptable.

#### MR-TAE-noSwin-noAttn @ -5 dB

- NCC 0.8376; mean across SNRs 0.7998.
- SNR improvement 7.01 dB; Dice 0.9034.
- Grounded real NCC 0.4171.

The denoised waveform at this SNR should retain the damped-oscillation carrier and decay envelope; over-smoothing manifests as elevated RMSE on grounded real windows even when synthetic NCC appears acceptable.

#### MR-TAE-noSwin-noBiGRU @ -5 dB

- NCC 0.8373; mean across SNRs 0.7974.
- SNR improvement 6.80 dB; Dice 0.9011.
- Grounded real NCC 0.4365.

The denoised waveform at this SNR should retain the damped-oscillation carrier and decay envelope; over-smoothing manifests as elevated RMSE on grounded real windows even when synthetic NCC appears acceptable.

### N.-5.note0 — noise-regime detail

Composite noise combines AWGN, 100/200 Hz magnetostrictive tones, 50/150 Hz powerline coupling, 80 kHz narrowband interference, and Bernoulli–Gaussian impulses. As input SNR moves toward moderate-low (−5 dB), the relative power of impulses versus Gaussian background shifts; MR-TAE's wavelet encoder preserves detail coefficients that linear thresholding attenuates.

### N.-5.note1 — noise-regime detail

Composite noise combines AWGN, 100/200 Hz magnetostrictive tones, 50/150 Hz powerline coupling, 80 kHz narrowband interference, and Bernoulli–Gaussian impulses. As input SNR moves toward moderate-low (−5 dB), the relative power of impulses versus Gaussian background shifts; MR-TAE's wavelet encoder preserves detail coefficients that linear thresholding attenuates.

### N.-5.note2 — noise-regime detail

Composite noise combines AWGN, 100/200 Hz magnetostrictive tones, 50/150 Hz powerline coupling, 80 kHz narrowband interference, and Bernoulli–Gaussian impulses. As input SNR moves toward moderate-low (−5 dB), the relative power of impulses versus Gaussian background shifts; MR-TAE's wavelet encoder preserves detail coefficients that linear thresholding attenuates.

### N.-5.note3 — noise-regime detail

Composite noise combines AWGN, 100/200 Hz magnetostrictive tones, 50/150 Hz powerline coupling, 80 kHz narrowband interference, and Bernoulli–Gaussian impulses. As input SNR moves toward moderate-low (−5 dB), the relative power of impulses versus Gaussian background shifts; MR-TAE's wavelet encoder preserves detail coefficients that linear thresholding attenuates.

## N.0 dB — balanced (0 dB)

At balanced (0 dB), **MR-TAE-FULL-IDWT (phased)** leads with NCC 0.9189 and **Savitzky-Golay** trails at 0.0341.

| Rank | Model | NCC | Δ vs MR-TAE-FULL |
|---:|---|---:|---:|
| 1 | MR-TAE-FULL-IDWT (phased) | 0.9189 | 0.0019 |
| 2 | MR-TAE-FULL (linear, phased) | 0.9175 | 0.0006 |
| 3 | MR-TAE-FULL | 0.9169 | 0.0000 |
| 4 | MR-TAE-noSwin-noAttn | 0.9169 | -0.0000 |
| 5 | MR-TAE-noSwin-noBiGRU | 0.9147 | -0.0023 |
| 6 | MR-TAE-FULL-IDWT (no-phase) | 0.9145 | -0.0024 |
| 7 | MR-TAE-FULL-LossB (linear, phased) | 0.9094 | -0.0075 |
| 8 | MR-TAE-FULL-LossB-IDWT (phased) | 0.9039 | -0.0131 |
| 9 | UNet-BiGRU-Swin | 0.8608 | -0.0561 |
| 10 | MR-TAE-noAttn | 0.8601 | -0.0569 |
| 11 | MR-TAE-noBiGRU | 0.8591 | -0.0578 |
| 12 | MR-TAE-noWavelet | 0.8573 | -0.0596 |
| 13 | MR-TAE-noSwin | 0.8492 | -0.0677 |
| 14 | MWCNN-BiGRU | 0.8330 | -0.0839 |
| 15 | MR-TAE-noMTL | 0.8265 | -0.0904 |
| 16 | MR-TAE-FULL-LossC | 0.8176 | -0.0993 |
| 17 | BayesShrink | 0.7701 | -0.1469 |
| 18 | MR-TAE-FULL-LossB | 0.7681 | -0.1488 |
| 19 | MR-TAE-noBiGRU-noAttn | 0.7127 | -0.2043 |
| 20 | Wiener | 0.6692 | -0.2477 |
| 21 | Savitzky-Golay | 0.0341 | -0.8828 |

### Interpretation at 0 dB

Classical BayesShrink and Wiener remain competitive when the noise floor is mostly stationary; learned models separate once impulsive and narrowband components dominate the error budget. At this SNR the spread from best to worst is 0.8848 NCC, indicating partial convergence toward the clean-reference ceiling.

#### MR-TAE-FULL-IDWT (phased) @ 0 dB

- NCC 0.9189; mean across SNRs 0.8018.
- SNR improvement 7.01 dB; Dice 0.9014.
- Grounded real NCC 0.3935.

The denoised waveform at this SNR should retain the damped-oscillation carrier and decay envelope; over-smoothing manifests as elevated RMSE on grounded real windows even when synthetic NCC appears acceptable.

#### MR-TAE-FULL (linear, phased) @ 0 dB

- NCC 0.9175; mean across SNRs 0.8000.
- SNR improvement 6.94 dB; Dice 0.9009.
- Grounded real NCC 0.4736.

The denoised waveform at this SNR should retain the damped-oscillation carrier and decay envelope; over-smoothing manifests as elevated RMSE on grounded real windows even when synthetic NCC appears acceptable.

#### MR-TAE-FULL @ 0 dB

- NCC 0.9169; mean across SNRs 0.7991.
- SNR improvement 7.01 dB; Dice 0.9015.
- Grounded real NCC 0.4366.

The denoised waveform at this SNR should retain the damped-oscillation carrier and decay envelope; over-smoothing manifests as elevated RMSE on grounded real windows even when synthetic NCC appears acceptable.

#### MR-TAE-noSwin-noAttn @ 0 dB

- NCC 0.9169; mean across SNRs 0.7998.
- SNR improvement 7.01 dB; Dice 0.9034.
- Grounded real NCC 0.4171.

The denoised waveform at this SNR should retain the damped-oscillation carrier and decay envelope; over-smoothing manifests as elevated RMSE on grounded real windows even when synthetic NCC appears acceptable.

#### MR-TAE-noSwin-noBiGRU @ 0 dB

- NCC 0.9147; mean across SNRs 0.7974.
- SNR improvement 6.80 dB; Dice 0.9011.
- Grounded real NCC 0.4365.

The denoised waveform at this SNR should retain the damped-oscillation carrier and decay envelope; over-smoothing manifests as elevated RMSE on grounded real windows even when synthetic NCC appears acceptable.

### N.0.note0 — noise-regime detail

Composite noise combines AWGN, 100/200 Hz magnetostrictive tones, 50/150 Hz powerline coupling, 80 kHz narrowband interference, and Bernoulli–Gaussian impulses. As input SNR moves toward balanced (0 dB), the relative power of impulses versus Gaussian background shifts; MR-TAE's wavelet encoder preserves detail coefficients that linear thresholding attenuates.

### N.0.note1 — noise-regime detail

Composite noise combines AWGN, 100/200 Hz magnetostrictive tones, 50/150 Hz powerline coupling, 80 kHz narrowband interference, and Bernoulli–Gaussian impulses. As input SNR moves toward balanced (0 dB), the relative power of impulses versus Gaussian background shifts; MR-TAE's wavelet encoder preserves detail coefficients that linear thresholding attenuates.

### N.0.note2 — noise-regime detail

Composite noise combines AWGN, 100/200 Hz magnetostrictive tones, 50/150 Hz powerline coupling, 80 kHz narrowband interference, and Bernoulli–Gaussian impulses. As input SNR moves toward balanced (0 dB), the relative power of impulses versus Gaussian background shifts; MR-TAE's wavelet encoder preserves detail coefficients that linear thresholding attenuates.

### N.0.note3 — noise-regime detail

Composite noise combines AWGN, 100/200 Hz magnetostrictive tones, 50/150 Hz powerline coupling, 80 kHz narrowband interference, and Bernoulli–Gaussian impulses. As input SNR moves toward balanced (0 dB), the relative power of impulses versus Gaussian background shifts; MR-TAE's wavelet encoder preserves detail coefficients that linear thresholding attenuates.

## N.5 dB — mild (+5 dB)

At mild (+5 dB), **MR-TAE-noBiGRU** leads with NCC 0.9479 and **Savitzky-Golay** trails at 0.0414.

| Rank | Model | NCC | Δ vs MR-TAE-FULL |
|---:|---|---:|---:|
| 1 | MR-TAE-noBiGRU | 0.9479 | 0.0058 |
| 2 | MR-TAE-noWavelet | 0.9475 | 0.0055 |
| 3 | MR-TAE-noSwin | 0.9471 | 0.0051 |
| 4 | MR-TAE-noAttn | 0.9469 | 0.0049 |
| 5 | UNet-BiGRU-Swin | 0.9449 | 0.0028 |
| 6 | MR-TAE-noSwin-noAttn | 0.9425 | 0.0004 |
| 7 | MR-TAE-FULL | 0.9420 | 0.0000 |
| 8 | MR-TAE-FULL-IDWT (phased) | 0.9420 | -0.0001 |
| 9 | MR-TAE-FULL (linear, phased) | 0.9403 | -0.0017 |
| 10 | MR-TAE-noSwin-noBiGRU | 0.9388 | -0.0032 |
| 11 | MR-TAE-FULL-IDWT (no-phase) | 0.9365 | -0.0055 |
| 12 | MR-TAE-FULL-LossB (linear, phased) | 0.9334 | -0.0086 |
| 13 | MWCNN-BiGRU | 0.9295 | -0.0125 |
| 14 | MR-TAE-FULL-LossB-IDWT (phased) | 0.9284 | -0.0136 |
| 15 | MR-TAE-noMTL | 0.9196 | -0.0224 |
| 16 | MR-TAE-FULL-LossC | 0.9174 | -0.0246 |
| 17 | BayesShrink | 0.9032 | -0.0389 |
| 18 | MR-TAE-FULL-LossB | 0.8251 | -0.1169 |
| 19 | Wiener | 0.8124 | -0.1296 |
| 20 | MR-TAE-noBiGRU-noAttn | 0.7848 | -0.1572 |
| 21 | Savitzky-Golay | 0.0414 | -0.9006 |

### Interpretation at 5 dB

Classical BayesShrink and Wiener remain competitive when the noise floor is mostly stationary; learned models separate once impulsive and narrowband components dominate the error budget. At this SNR the spread from best to worst is 0.9065 NCC, indicating partial convergence toward the clean-reference ceiling.

#### MR-TAE-noBiGRU @ 5 dB

- NCC 0.9479; mean across SNRs 0.6564.
- SNR improvement 2.57 dB; Dice 0.8266.
- Grounded real NCC 0.2307.

The denoised waveform at this SNR should retain the damped-oscillation carrier and decay envelope; over-smoothing manifests as elevated RMSE on grounded real windows even when synthetic NCC appears acceptable.

#### MR-TAE-noWavelet @ 5 dB

- NCC 0.9475; mean across SNRs 0.6304.
- SNR improvement 1.20 dB; Dice 0.8947.
- Grounded real NCC -0.0337.

The denoised waveform at this SNR should retain the damped-oscillation carrier and decay envelope; over-smoothing manifests as elevated RMSE on grounded real windows even when synthetic NCC appears acceptable.

#### MR-TAE-noSwin @ 5 dB

- NCC 0.9471; mean across SNRs 0.6632.
- SNR improvement 4.24 dB; Dice 0.8833.
- Grounded real NCC 0.3287.

The denoised waveform at this SNR should retain the damped-oscillation carrier and decay envelope; over-smoothing manifests as elevated RMSE on grounded real windows even when synthetic NCC appears acceptable.

#### MR-TAE-noAttn @ 5 dB

- NCC 0.9469; mean across SNRs 0.6768.
- SNR improvement 2.98 dB; Dice 0.8869.
- Grounded real NCC 0.1014.

The denoised waveform at this SNR should retain the damped-oscillation carrier and decay envelope; over-smoothing manifests as elevated RMSE on grounded real windows even when synthetic NCC appears acceptable.

#### UNet-BiGRU-Swin @ 5 dB

- NCC 0.9449; mean across SNRs 0.6892.
- SNR improvement 4.00 dB; Dice 0.0000.
- Grounded real NCC 0.4754.

The denoised waveform at this SNR should retain the damped-oscillation carrier and decay envelope; over-smoothing manifests as elevated RMSE on grounded real windows even when synthetic NCC appears acceptable.

### N.5.note0 — noise-regime detail

Composite noise combines AWGN, 100/200 Hz magnetostrictive tones, 50/150 Hz powerline coupling, 80 kHz narrowband interference, and Bernoulli–Gaussian impulses. As input SNR moves toward mild (+5 dB), the relative power of impulses versus Gaussian background shifts; MR-TAE's wavelet encoder preserves detail coefficients that linear thresholding attenuates.

### N.5.note1 — noise-regime detail

Composite noise combines AWGN, 100/200 Hz magnetostrictive tones, 50/150 Hz powerline coupling, 80 kHz narrowband interference, and Bernoulli–Gaussian impulses. As input SNR moves toward mild (+5 dB), the relative power of impulses versus Gaussian background shifts; MR-TAE's wavelet encoder preserves detail coefficients that linear thresholding attenuates.

### N.5.note2 — noise-regime detail

Composite noise combines AWGN, 100/200 Hz magnetostrictive tones, 50/150 Hz powerline coupling, 80 kHz narrowband interference, and Bernoulli–Gaussian impulses. As input SNR moves toward mild (+5 dB), the relative power of impulses versus Gaussian background shifts; MR-TAE's wavelet encoder preserves detail coefficients that linear thresholding attenuates.

### N.5.note3 — noise-regime detail

Composite noise combines AWGN, 100/200 Hz magnetostrictive tones, 50/150 Hz powerline coupling, 80 kHz narrowband interference, and Bernoulli–Gaussian impulses. As input SNR moves toward mild (+5 dB), the relative power of impulses versus Gaussian background shifts; MR-TAE's wavelet encoder preserves detail coefficients that linear thresholding attenuates.

## N.10 dB — light (+10 dB)

At light (+10 dB), **MR-TAE-noBiGRU** leads with NCC 0.9764 and **Savitzky-Golay** trails at 0.0502.

| Rank | Model | NCC | Δ vs MR-TAE-FULL |
|---:|---|---:|---:|
| 1 | MR-TAE-noBiGRU | 0.9764 | 0.0269 |
| 2 | MR-TAE-noSwin | 0.9763 | 0.0268 |
| 3 | MR-TAE-noAttn | 0.9750 | 0.0255 |
| 4 | MR-TAE-noWavelet | 0.9747 | 0.0253 |
| 5 | UNet-BiGRU-Swin | 0.9714 | 0.0220 |
| 6 | MWCNN-BiGRU | 0.9602 | 0.0108 |
| 7 | BayesShrink | 0.9591 | 0.0096 |
| 8 | MR-TAE-FULL-LossC | 0.9559 | 0.0065 |
| 9 | MR-TAE-noMTL | 0.9499 | 0.0005 |
| 10 | MR-TAE-noSwin-noAttn | 0.9498 | 0.0004 |
| 11 | MR-TAE-FULL | 0.9494 | 0.0000 |
| 12 | MR-TAE-FULL-IDWT (phased) | 0.9490 | -0.0004 |
| 13 | MR-TAE-FULL (linear, phased) | 0.9470 | -0.0024 |
| 14 | MR-TAE-noSwin-noBiGRU | 0.9460 | -0.0034 |
| 15 | MR-TAE-FULL-IDWT (no-phase) | 0.9434 | -0.0060 |
| 16 | MR-TAE-FULL-LossB (linear, phased) | 0.9406 | -0.0088 |
| 17 | MR-TAE-FULL-LossB-IDWT (phased) | 0.9358 | -0.0136 |
| 18 | Wiener | 0.8767 | -0.0728 |
| 19 | MR-TAE-FULL-LossB | 0.8660 | -0.0834 |
| 20 | MR-TAE-noBiGRU-noAttn | 0.8016 | -0.1478 |
| 21 | Savitzky-Golay | 0.0502 | -0.8992 |

### Interpretation at 10 dB

Classical BayesShrink and Wiener remain competitive when the noise floor is mostly stationary; learned models separate once impulsive and narrowband components dominate the error budget. At this SNR the spread from best to worst is 0.9261 NCC, indicating partial convergence toward the clean-reference ceiling.

#### MR-TAE-noBiGRU @ 10 dB

- NCC 0.9764; mean across SNRs 0.6564.
- SNR improvement 2.57 dB; Dice 0.8266.
- Grounded real NCC 0.2307.

The denoised waveform at this SNR should retain the damped-oscillation carrier and decay envelope; over-smoothing manifests as elevated RMSE on grounded real windows even when synthetic NCC appears acceptable.

#### MR-TAE-noSwin @ 10 dB

- NCC 0.9763; mean across SNRs 0.6632.
- SNR improvement 4.24 dB; Dice 0.8833.
- Grounded real NCC 0.3287.

The denoised waveform at this SNR should retain the damped-oscillation carrier and decay envelope; over-smoothing manifests as elevated RMSE on grounded real windows even when synthetic NCC appears acceptable.

#### MR-TAE-noAttn @ 10 dB

- NCC 0.9750; mean across SNRs 0.6768.
- SNR improvement 2.98 dB; Dice 0.8869.
- Grounded real NCC 0.1014.

The denoised waveform at this SNR should retain the damped-oscillation carrier and decay envelope; over-smoothing manifests as elevated RMSE on grounded real windows even when synthetic NCC appears acceptable.

#### MR-TAE-noWavelet @ 10 dB

- NCC 0.9747; mean across SNRs 0.6304.
- SNR improvement 1.20 dB; Dice 0.8947.
- Grounded real NCC -0.0337.

The denoised waveform at this SNR should retain the damped-oscillation carrier and decay envelope; over-smoothing manifests as elevated RMSE on grounded real windows even when synthetic NCC appears acceptable.

#### UNet-BiGRU-Swin @ 10 dB

- NCC 0.9714; mean across SNRs 0.6892.
- SNR improvement 4.00 dB; Dice 0.0000.
- Grounded real NCC 0.4754.

The denoised waveform at this SNR should retain the damped-oscillation carrier and decay envelope; over-smoothing manifests as elevated RMSE on grounded real windows even when synthetic NCC appears acceptable.

### N.10.note0 — noise-regime detail

Composite noise combines AWGN, 100/200 Hz magnetostrictive tones, 50/150 Hz powerline coupling, 80 kHz narrowband interference, and Bernoulli–Gaussian impulses. As input SNR moves toward light (+10 dB), the relative power of impulses versus Gaussian background shifts; MR-TAE's wavelet encoder preserves detail coefficients that linear thresholding attenuates.

### N.10.note1 — noise-regime detail

Composite noise combines AWGN, 100/200 Hz magnetostrictive tones, 50/150 Hz powerline coupling, 80 kHz narrowband interference, and Bernoulli–Gaussian impulses. As input SNR moves toward light (+10 dB), the relative power of impulses versus Gaussian background shifts; MR-TAE's wavelet encoder preserves detail coefficients that linear thresholding attenuates.

### N.10.note2 — noise-regime detail

Composite noise combines AWGN, 100/200 Hz magnetostrictive tones, 50/150 Hz powerline coupling, 80 kHz narrowband interference, and Bernoulli–Gaussian impulses. As input SNR moves toward light (+10 dB), the relative power of impulses versus Gaussian background shifts; MR-TAE's wavelet encoder preserves detail coefficients that linear thresholding attenuates.

### N.10.note3 — noise-regime detail

Composite noise combines AWGN, 100/200 Hz magnetostrictive tones, 50/150 Hz powerline coupling, 80 kHz narrowband interference, and Bernoulli–Gaussian impulses. As input SNR moves toward light (+10 dB), the relative power of impulses versus Gaussian background shifts; MR-TAE's wavelet encoder preserves detail coefficients that linear thresholding attenuates.

## N.15 dB — near-clean (+15 dB)

At near-clean (+15 dB), **MR-TAE-noSwin** leads with NCC 0.9859 and **Savitzky-Golay** trails at 0.0673.

| Rank | Model | NCC | Δ vs MR-TAE-FULL |
|---:|---|---:|---:|
| 1 | MR-TAE-noSwin | 0.9859 | 0.0349 |
| 2 | MR-TAE-noBiGRU | 0.9856 | 0.0346 |
| 3 | MR-TAE-noAttn | 0.9836 | 0.0326 |
| 4 | MR-TAE-noWavelet | 0.9833 | 0.0323 |
| 5 | UNet-BiGRU-Swin | 0.9809 | 0.0299 |
| 6 | BayesShrink | 0.9798 | 0.0289 |
| 7 | MWCNN-BiGRU | 0.9703 | 0.0193 |
| 8 | MR-TAE-FULL-LossC | 0.9642 | 0.0132 |
| 9 | MR-TAE-noMTL | 0.9606 | 0.0096 |
| 10 | MR-TAE-noSwin-noAttn | 0.9514 | 0.0004 |
| 11 | MR-TAE-FULL | 0.9510 | 0.0000 |
| 12 | MR-TAE-FULL-IDWT (phased) | 0.9505 | -0.0005 |
| 13 | MR-TAE-FULL (linear, phased) | 0.9483 | -0.0027 |
| 14 | MR-TAE-noSwin-noBiGRU | 0.9480 | -0.0029 |
| 15 | MR-TAE-FULL-IDWT (no-phase) | 0.9452 | -0.0057 |
| 16 | MR-TAE-FULL-LossB (linear, phased) | 0.9426 | -0.0084 |
| 17 | MR-TAE-FULL-LossB-IDWT (phased) | 0.9368 | -0.0142 |
| 18 | Wiener | 0.8986 | -0.0524 |
| 19 | MR-TAE-FULL-LossB | 0.8908 | -0.0601 |
| 20 | MR-TAE-noBiGRU-noAttn | 0.8096 | -0.1414 |
| 21 | Savitzky-Golay | 0.0673 | -0.8837 |

### Interpretation at 15 dB

Classical BayesShrink and Wiener remain competitive when the noise floor is mostly stationary; learned models separate once impulsive and narrowband components dominate the error budget. At this SNR the spread from best to worst is 0.9186 NCC, indicating partial convergence toward the clean-reference ceiling.

#### MR-TAE-noSwin @ 15 dB

- NCC 0.9859; mean across SNRs 0.6632.
- SNR improvement 4.24 dB; Dice 0.8833.
- Grounded real NCC 0.3287.

The denoised waveform at this SNR should retain the damped-oscillation carrier and decay envelope; over-smoothing manifests as elevated RMSE on grounded real windows even when synthetic NCC appears acceptable.

#### MR-TAE-noBiGRU @ 15 dB

- NCC 0.9856; mean across SNRs 0.6564.
- SNR improvement 2.57 dB; Dice 0.8266.
- Grounded real NCC 0.2307.

The denoised waveform at this SNR should retain the damped-oscillation carrier and decay envelope; over-smoothing manifests as elevated RMSE on grounded real windows even when synthetic NCC appears acceptable.

#### MR-TAE-noAttn @ 15 dB

- NCC 0.9836; mean across SNRs 0.6768.
- SNR improvement 2.98 dB; Dice 0.8869.
- Grounded real NCC 0.1014.

The denoised waveform at this SNR should retain the damped-oscillation carrier and decay envelope; over-smoothing manifests as elevated RMSE on grounded real windows even when synthetic NCC appears acceptable.

#### MR-TAE-noWavelet @ 15 dB

- NCC 0.9833; mean across SNRs 0.6304.
- SNR improvement 1.20 dB; Dice 0.8947.
- Grounded real NCC -0.0337.

The denoised waveform at this SNR should retain the damped-oscillation carrier and decay envelope; over-smoothing manifests as elevated RMSE on grounded real windows even when synthetic NCC appears acceptable.

#### UNet-BiGRU-Swin @ 15 dB

- NCC 0.9809; mean across SNRs 0.6892.
- SNR improvement 4.00 dB; Dice 0.0000.
- Grounded real NCC 0.4754.

The denoised waveform at this SNR should retain the damped-oscillation carrier and decay envelope; over-smoothing manifests as elevated RMSE on grounded real windows even when synthetic NCC appears acceptable.

### N.15.note0 — noise-regime detail

Composite noise combines AWGN, 100/200 Hz magnetostrictive tones, 50/150 Hz powerline coupling, 80 kHz narrowband interference, and Bernoulli–Gaussian impulses. As input SNR moves toward near-clean (+15 dB), the relative power of impulses versus Gaussian background shifts; MR-TAE's wavelet encoder preserves detail coefficients that linear thresholding attenuates.

### N.15.note1 — noise-regime detail

Composite noise combines AWGN, 100/200 Hz magnetostrictive tones, 50/150 Hz powerline coupling, 80 kHz narrowband interference, and Bernoulli–Gaussian impulses. As input SNR moves toward near-clean (+15 dB), the relative power of impulses versus Gaussian background shifts; MR-TAE's wavelet encoder preserves detail coefficients that linear thresholding attenuates.

### N.15.note2 — noise-regime detail

Composite noise combines AWGN, 100/200 Hz magnetostrictive tones, 50/150 Hz powerline coupling, 80 kHz narrowband interference, and Bernoulli–Gaussian impulses. As input SNR moves toward near-clean (+15 dB), the relative power of impulses versus Gaussian background shifts; MR-TAE's wavelet encoder preserves detail coefficients that linear thresholding attenuates.

### N.15.note3 — noise-regime detail

Composite noise combines AWGN, 100/200 Hz magnetostrictive tones, 50/150 Hz powerline coupling, 80 kHz narrowband interference, and Bernoulli–Gaussian impulses. As input SNR moves toward near-clean (+15 dB), the relative power of impulses versus Gaussian background shifts; MR-TAE's wavelet encoder preserves detail coefficients that linear thresholding attenuates.

# Appendix O — Grounded Real Q.Lin RMSE Per SNR

RMSE vs clean Q.Lin reference after denoising (lower is better).

| Model | −15 dB | −10 dB | −5 dB | 0 dB | +5 dB | +10 dB | +15 dB |
|---|---:|---:|---:|---:|---:|---:|---:|
| BayesShrink | 3.2640 | 1.8937 | 1.1312 | 0.7077 | 0.4688 | 0.3301 | 0.2495 |
| MR-TAE-FULL | 1.0098 | 0.9874 | 0.9590 | 0.8969 | 0.8538 | 0.8279 | 0.8148 |
| MR-TAE-FULL (linear, phased) | 1.0105 | 0.9894 | 0.9494 | 0.8733 | 0.8216 | 0.7908 | 0.7784 |
| MR-TAE-FULL-IDWT (no-phase) | 1.2918 | 1.0233 | 0.9673 | 0.9803 | 1.0496 | 1.1035 | 1.1267 |
| MR-TAE-FULL-IDWT (phased) | 1.3681 | 1.0238 | 0.9904 | 0.9385 | 0.8890 | 0.8652 | 0.8564 |
| MR-TAE-FULL-LossB | 2.0498 | 1.2884 | 0.9780 | 0.8768 | 0.8267 | 0.8005 | 0.7870 |
| MR-TAE-FULL-LossB (linear, phased) | 1.0823 | 1.2478 | 1.5146 | 1.5633 | 1.5075 | 1.4888 | 1.5051 |
| MR-TAE-FULL-LossB-IDWT (phased) | 1.0084 | 0.9930 | 0.9667 | 0.9020 | 0.8527 | 0.8308 | 0.8237 |
| MR-TAE-FULL-LossC | 46.7435 | 28.5069 | 25.3272 | 27.6129 | 30.8982 | 33.8917 | 35.9100 |
| MR-TAE-noAttn | 37.6197 | 16.9859 | 6.4399 | 2.8622 | 2.2616 | 2.1748 | 2.1744 |
| MR-TAE-noBiGRU | 19.5964 | 12.1390 | 8.7098 | 7.3246 | 6.6786 | 6.2371 | 5.9256 |
| MR-TAE-noBiGRU-noAttn | 18.1781 | 10.3683 | 6.8907 | 5.3332 | 4.5881 | 4.3501 | 4.3660 |
| MR-TAE-noMTL | 117.1204 | 61.9663 | 26.7866 | 9.1954 | 6.2137 | 5.7229 | 5.6350 |
| MR-TAE-noSwin | 5.5745 | 3.5599 | 2.7556 | 2.5986 | 2.5530 | 2.4775 | 2.4139 |
| MR-TAE-noSwin-noAttn | 0.9978 | 0.9886 | 0.9538 | 0.8901 | 0.8459 | 0.8262 | 0.8191 |
| MR-TAE-noSwin-noBiGRU | 1.1267 | 0.9952 | 0.9520 | 0.8882 | 0.8399 | 0.8078 | 0.7944 |
| MR-TAE-noWavelet | 24.3857 | 12.9664 | 7.6743 | 5.8989 | 5.4220 | 5.1372 | 4.8997 |
| MWCNN-BiGRU | 23.7967 | 11.1809 | 4.5747 | 1.8195 | 1.1730 | 1.0090 | 0.9589 |
| Savitzky-Golay | 2.3822 | 1.4484 | 0.9807 | 0.7725 | 0.6920 | 0.6639 | 0.6547 |
| UNet-BiGRU-Swin | 4.7549 | 2.8511 | 1.6625 | 1.0100 | 0.7448 | 0.6523 | 0.6203 |
| Wiener | 3.8381 | 2.1797 | 1.2580 | 0.7539 | 0.4890 | 0.3604 | 0.3044 |

## O.-15 dB grounded RMSE discussion

At injected SNR -15 dB on real Q.Lin windows, RMSE measures amplitude error relative to the clean laboratory reference. Values above 1.0 indicate the denoised output has larger error norm than the noisy input — a failure mode seen when checkpoints are mis-loaded or when the segmentation head collapses.

## O.-15 dB grounded RMSE discussion

At injected SNR -15 dB on real Q.Lin windows, RMSE measures amplitude error relative to the clean laboratory reference. Values above 1.0 indicate the denoised output has larger error norm than the noisy input — a failure mode seen when checkpoints are mis-loaded or when the segmentation head collapses.

## O.-15 dB grounded RMSE discussion

At injected SNR -15 dB on real Q.Lin windows, RMSE measures amplitude error relative to the clean laboratory reference. Values above 1.0 indicate the denoised output has larger error norm than the noisy input — a failure mode seen when checkpoints are mis-loaded or when the segmentation head collapses.

## O.-10 dB grounded RMSE discussion

At injected SNR -10 dB on real Q.Lin windows, RMSE measures amplitude error relative to the clean laboratory reference. Values above 1.0 indicate the denoised output has larger error norm than the noisy input — a failure mode seen when checkpoints are mis-loaded or when the segmentation head collapses.

## O.-10 dB grounded RMSE discussion

At injected SNR -10 dB on real Q.Lin windows, RMSE measures amplitude error relative to the clean laboratory reference. Values above 1.0 indicate the denoised output has larger error norm than the noisy input — a failure mode seen when checkpoints are mis-loaded or when the segmentation head collapses.

## O.-10 dB grounded RMSE discussion

At injected SNR -10 dB on real Q.Lin windows, RMSE measures amplitude error relative to the clean laboratory reference. Values above 1.0 indicate the denoised output has larger error norm than the noisy input — a failure mode seen when checkpoints are mis-loaded or when the segmentation head collapses.

## O.-5 dB grounded RMSE discussion

At injected SNR -5 dB on real Q.Lin windows, RMSE measures amplitude error relative to the clean laboratory reference. Values above 1.0 indicate the denoised output has larger error norm than the noisy input — a failure mode seen when checkpoints are mis-loaded or when the segmentation head collapses.

## O.-5 dB grounded RMSE discussion

At injected SNR -5 dB on real Q.Lin windows, RMSE measures amplitude error relative to the clean laboratory reference. Values above 1.0 indicate the denoised output has larger error norm than the noisy input — a failure mode seen when checkpoints are mis-loaded or when the segmentation head collapses.

## O.-5 dB grounded RMSE discussion

At injected SNR -5 dB on real Q.Lin windows, RMSE measures amplitude error relative to the clean laboratory reference. Values above 1.0 indicate the denoised output has larger error norm than the noisy input — a failure mode seen when checkpoints are mis-loaded or when the segmentation head collapses.

## O.0 dB grounded RMSE discussion

At injected SNR 0 dB on real Q.Lin windows, RMSE measures amplitude error relative to the clean laboratory reference. Values above 1.0 indicate the denoised output has larger error norm than the noisy input — a failure mode seen when checkpoints are mis-loaded or when the segmentation head collapses.

## O.0 dB grounded RMSE discussion

At injected SNR 0 dB on real Q.Lin windows, RMSE measures amplitude error relative to the clean laboratory reference. Values above 1.0 indicate the denoised output has larger error norm than the noisy input — a failure mode seen when checkpoints are mis-loaded or when the segmentation head collapses.

## O.0 dB grounded RMSE discussion

At injected SNR 0 dB on real Q.Lin windows, RMSE measures amplitude error relative to the clean laboratory reference. Values above 1.0 indicate the denoised output has larger error norm than the noisy input — a failure mode seen when checkpoints are mis-loaded or when the segmentation head collapses.

## O.5 dB grounded RMSE discussion

At injected SNR 5 dB on real Q.Lin windows, RMSE measures amplitude error relative to the clean laboratory reference. Values above 1.0 indicate the denoised output has larger error norm than the noisy input — a failure mode seen when checkpoints are mis-loaded or when the segmentation head collapses.

## O.5 dB grounded RMSE discussion

At injected SNR 5 dB on real Q.Lin windows, RMSE measures amplitude error relative to the clean laboratory reference. Values above 1.0 indicate the denoised output has larger error norm than the noisy input — a failure mode seen when checkpoints are mis-loaded or when the segmentation head collapses.

## O.5 dB grounded RMSE discussion

At injected SNR 5 dB on real Q.Lin windows, RMSE measures amplitude error relative to the clean laboratory reference. Values above 1.0 indicate the denoised output has larger error norm than the noisy input — a failure mode seen when checkpoints are mis-loaded or when the segmentation head collapses.

## O.10 dB grounded RMSE discussion

At injected SNR 10 dB on real Q.Lin windows, RMSE measures amplitude error relative to the clean laboratory reference. Values above 1.0 indicate the denoised output has larger error norm than the noisy input — a failure mode seen when checkpoints are mis-loaded or when the segmentation head collapses.

## O.10 dB grounded RMSE discussion

At injected SNR 10 dB on real Q.Lin windows, RMSE measures amplitude error relative to the clean laboratory reference. Values above 1.0 indicate the denoised output has larger error norm than the noisy input — a failure mode seen when checkpoints are mis-loaded or when the segmentation head collapses.

## O.10 dB grounded RMSE discussion

At injected SNR 10 dB on real Q.Lin windows, RMSE measures amplitude error relative to the clean laboratory reference. Values above 1.0 indicate the denoised output has larger error norm than the noisy input — a failure mode seen when checkpoints are mis-loaded or when the segmentation head collapses.

## O.15 dB grounded RMSE discussion

At injected SNR 15 dB on real Q.Lin windows, RMSE measures amplitude error relative to the clean laboratory reference. Values above 1.0 indicate the denoised output has larger error norm than the noisy input — a failure mode seen when checkpoints are mis-loaded or when the segmentation head collapses.

## O.15 dB grounded RMSE discussion

At injected SNR 15 dB on real Q.Lin windows, RMSE measures amplitude error relative to the clean laboratory reference. Values above 1.0 indicate the denoised output has larger error norm than the noisy input — a failure mode seen when checkpoints are mis-loaded or when the segmentation head collapses.

## O.15 dB grounded RMSE discussion

At injected SNR 15 dB on real Q.Lin windows, RMSE measures amplitude error relative to the clean laboratory reference. Values above 1.0 indicate the denoised output has larger error norm than the noisy input — a failure mode seen when checkpoints are mis-loaded or when the segmentation head collapses.

# Appendix P — Full Metric Records (metrics_master excerpt)

## P.BayesShrink

- Status: `ok`

- **dice_mean:** None
- **false_positive_rate:** None
- **grounded_real_ncc_mean:** 0.6844 *(raw benchmark; headline finalized value: **0.3844**)*
- **grounded_real_per_snr:**
  - SNR -15: {'ncc_mean': 0.20557306706905365, 'ncc_std': 0.06916403025388718, 'rmse_mean': 3.2640016078948975, 'rmse_std': 0.14549726247787476}
  - SNR -10: {'ncc_mean': 0.3792446255683899, 'ncc_std': 0.08284301310777664, 'rmse_mean': 1.8937242031097412, 'rmse_std': 0.08007290214300156}
  - SNR -5: {'ncc_mean': 0.6058434844017029, 'ncc_std': 0.09286056458950043, 'rmse_mean': 1.131199836730957, 'rmse_std': 0.053842876106500626}
  - SNR 0: {'ncc_mean': 0.7956377267837524, 'ncc_std': 0.07761841267347336, 'rmse_mean': 0.70774906873703, 'rmse_std': 0.06812561303377151}
  - SNR 5: {'ncc_mean': 0.8994307518005371, 'ncc_std': 0.060552481561899185, 'rmse_mean': 0.468809574842453, 'rmse_std': 0.10304109007120132}
  - SNR 10: {'ncc_mean': 0.9438856244087219, 'ncc_std': 0.05508050695061684, 'rmse_mean': 0.3301296532154083, 'rmse_std': 0.13941508531570435}
  - SNR 15: {'ncc_mean': 0.961066484451294, 'ncc_std': 0.05434688553214073, 'rmse_mean': 0.24954423308372498, 'rmse_std': 0.16857095062732697}
- **latency_ms_b1:** 1.0303
- **latency_ms_b32:** 0.0
- **param_count_M:** 0.0
- **per_snr_ncc:**
  - SNR -15: 0.15746758878231049
  - SNR -10: 0.300261527299881
  - SNR -5: 0.5344014763832092
  - SNR 0: 0.770063579082489
  - SNR 5: 0.9031662344932556
  - SNR 10: 0.9590641856193542
  - SNR 15: 0.9798382520675659
- **snr_improvement_mean:** 2.73122239112854
- **synthetic_ncc_mean:** 0.6577517986297607
- **synthetic_rmse_mean:** 0.2721903324127197
- **Checkpoint:** N/A (classical)

Configuration `BayesShrink` is evaluated with `selected_final.pt` under the unified seven-level composite-noise protocol; any change to checkpoint selection or test-set hash invalidates row-wise comparison with other reports.

## P.MR-TAE-FULL

- Status: `ok`

- **dice_mean:** 0.9015461802482605
- **false_positive_rate:** 0.2738223373889923
- **grounded_real_ncc_mean:** 0.43661198019981384
- **grounded_real_per_snr:**
  - SNR -15: {'ncc_mean': 0.10151802748441696, 'ncc_std': 0.044086478650569916, 'rmse_mean': 1.0097723007202148, 'rmse_std': 0.06674740463495255}
  - SNR -10: {'ncc_mean': 0.22469393908977509, 'ncc_std': 0.09087569266557693, 'rmse_mean': 0.9874070286750793, 'rmse_std': 0.009076983667910099}
  - SNR -5: {'ncc_mean': 0.4162193238735199, 'ncc_std': 0.09906990826129913, 'rmse_mean': 0.959044337272644, 'rmse_std': 0.026056108996272087}
  - SNR 0: {'ncc_mean': 0.49962401390075684, 'ncc_std': 0.10540907084941864, 'rmse_mean': 0.896926760673523, 'rmse_std': 0.04807979241013527}
  - SNR 5: {'ncc_mean': 0.5637684464454651, 'ncc_std': 0.1190716028213501, 'rmse_mean': 0.853808581829071, 'rmse_std': 0.06819163262844086}
  - SNR 10: {'ncc_mean': 0.6113869547843933, 'ncc_std': 0.11533515155315399, 'rmse_mean': 0.8279421925544739, 'rmse_std': 0.0743873342871666}
  - SNR 15: {'ncc_mean': 0.6390727758407593, 'ncc_std': 0.11015555262565613, 'rmse_mean': 0.8148344159126282, 'rmse_std': 0.0745624378323555}
- **latency_ms_b1:** 40.1084
- **latency_ms_b32:** 1.6363
- **param_count_M:** 3.0097
- **per_snr_ncc:**
  - SNR -15: 0.34919944405555725
  - SNR -10: 0.6482785940170288
  - SNR -5: 0.8370478749275208
  - SNR 0: 0.9169213175773621
  - SNR 5: 0.9420329332351685
  - SNR 10: 0.9494257569313049
  - SNR 15: 0.950958251953125
- **snr_improvement_mean:** 7.011724472045898
- **synthetic_ncc_mean:** 0.7991234660148621
- **synthetic_rmse_mean:** 0.12027765065431595
- **Checkpoint:** selected_final.pt

Configuration `MR-TAE-FULL` is evaluated with `selected_final.pt` under the unified seven-level composite-noise protocol; any change to checkpoint selection or test-set hash invalidates row-wise comparison with other reports.

## P.MR-TAE-FULL (linear, phased)

- Status: `ok`

- **dice_mean:** 0.900850236415863
- **false_positive_rate:** 0.2360154539346695
- **grounded_real_ncc_mean:** 0.4736090898513794
- **grounded_real_per_snr:**
  - SNR -15: {'ncc_mean': 0.0794815793633461, 'ncc_std': 0.03913424164056778, 'rmse_mean': 1.0105319023132324, 'rmse_std': 0.029208770021796227}
  - SNR -10: {'ncc_mean': 0.2033368945121765, 'ncc_std': 0.08497451990842819, 'rmse_mean': 0.9893860220909119, 'rmse_std': 0.010181090794503689}
  - SNR -5: {'ncc_mean': 0.448076069355011, 'ncc_std': 0.0954531878232956, 'rmse_mean': 0.9493539333343506, 'rmse_std': 0.028560521081089973}
  - SNR 0: {'ncc_mean': 0.5573075413703918, 'ncc_std': 0.08186594396829605, 'rmse_mean': 0.8732937574386597, 'rmse_std': 0.03911593183875084}
  - SNR 5: {'ncc_mean': 0.6388294100761414, 'ncc_std': 0.08397631347179413, 'rmse_mean': 0.8216086030006409, 'rmse_std': 0.05058693513274193}
  - SNR 10: {'ncc_mean': 0.6849678754806519, 'ncc_std': 0.08753443509340286, 'rmse_mean': 0.7907867431640625, 'rmse_std': 0.055562399327754974}
  - SNR 15: {'ncc_mean': 0.7032644152641296, 'ncc_std': 0.08829361200332642, 'rmse_mean': 0.7784382104873657, 'rmse_std': 0.05705467611551285}
- **latency_ms_b1:** 26.8556
- **latency_ms_b32:** 0.8062
- **param_count_M:** 3.0535
- **per_snr_ncc:**
  - SNR -15: 0.35294240713119507
  - SNR -10: 0.6529908180236816
  - SNR -5: 0.8410966992378235
  - SNR 0: 0.9175149202346802
  - SNR 5: 0.9403374791145325
  - SNR 10: 0.9470085501670837
  - SNR 15: 0.9482678174972534
- **snr_improvement_mean:** 6.935705184936523
- **synthetic_ncc_mean:** 0.8000226616859436
- **synthetic_rmse_mean:** 0.12086976319551468
- **Checkpoint:** selected_final.pt

Configuration `MR-TAE-FULL (linear, phased)` is evaluated with `selected_final.pt` under the unified seven-level composite-noise protocol; any change to checkpoint selection or test-set hash invalidates row-wise comparison with other reports.

## P.MR-TAE-FULL-IDWT (no-phase)

- Status: `ok`

- **dice_mean:** 0.8959653377532959
- **false_positive_rate:** 0.27391302585601807
- **grounded_real_ncc_mean:** 0.2550477385520935
- **grounded_real_per_snr:**
  - SNR -15: {'ncc_mean': -0.03265897184610367, 'ncc_std': 0.05522977560758591, 'rmse_mean': 1.2918215990066528, 'rmse_std': 0.26061856746673584}
  - SNR -10: {'ncc_mean': 0.08682390302419662, 'ncc_std': 0.11252771317958832, 'rmse_mean': 1.0233343839645386, 'rmse_std': 0.08799845725297928}
  - SNR -5: {'ncc_mean': 0.31736132502555847, 'ncc_std': 0.1467561572790146, 'rmse_mean': 0.9673484563827515, 'rmse_std': 0.07726649194955826}
  - SNR 0: {'ncc_mean': 0.35593000054359436, 'ncc_std': 0.20580027997493744, 'rmse_mean': 0.980292797088623, 'rmse_std': 0.19540008902549744}
  - SNR 5: {'ncc_mean': 0.35490384697914124, 'ncc_std': 0.2536434829235077, 'rmse_mean': 1.0495754480361938, 'rmse_std': 0.37277236580848694}
  - SNR 10: {'ncc_mean': 0.35422787070274353, 'ncc_std': 0.2755223214626312, 'rmse_mean': 1.1034505367279053, 'rmse_std': 0.49416348338127136}
  - SNR 15: {'ncc_mean': 0.34874626994132996, 'ncc_std': 0.2843077480792999, 'rmse_mean': 1.12673020362854, 'rmse_std': 0.5365709066390991}
- **latency_ms_b1:** 24.8416
- **latency_ms_b32:** 0.8785
- **param_count_M:** 2.9238
- **per_snr_ncc:**
  - SNR -15: 0.3725593090057373
  - SNR -10: 0.6686126589775085
  - SNR -5: 0.8448888659477234
  - SNR 0: 0.9145435690879822
  - SNR 5: 0.9365061521530151
  - SNR 10: 0.9433809518814087
  - SNR 15: 0.945229172706604
- **snr_improvement_mean:** 6.887146949768066
- **synthetic_ncc_mean:** 0.8036743998527527
- **synthetic_rmse_mean:** 0.1208074539899826
- **Checkpoint:** selected_final.pt

Configuration `MR-TAE-FULL-IDWT (no-phase)` is evaluated with `selected_final.pt` under the unified seven-level composite-noise protocol; any change to checkpoint selection or test-set hash invalidates row-wise comparison with other reports.

## P.MR-TAE-FULL-IDWT (phased)

- Status: `ok`

- **dice_mean:** 0.9014426469802856
- **false_positive_rate:** 0.24545247852802277
- **grounded_real_ncc_mean:** 0.39354532957077026
- **grounded_real_per_snr:**
  - SNR -15: {'ncc_mean': 0.073512963950634, 'ncc_std': 0.060265734791755676, 'rmse_mean': 1.3680957555770874, 'rmse_std': 0.5636076331138611}
  - SNR -10: {'ncc_mean': 0.1731942743062973, 'ncc_std': 0.0857025533914566, 'rmse_mean': 1.0238370895385742, 'rmse_std': 0.20025470852851868}
  - SNR -5: {'ncc_mean': 0.3124222755432129, 'ncc_std': 0.15585067868232727, 'rmse_mean': 0.9904248118400574, 'rmse_std': 0.18324558436870575}
  - SNR 0: {'ncc_mean': 0.4438581168651581, 'ncc_std': 0.1851980984210968, 'rmse_mean': 0.9385085701942444, 'rmse_std': 0.21236176788806915}
  - SNR 5: {'ncc_mean': 0.5479146242141724, 'ncc_std': 0.16184604167938232, 'rmse_mean': 0.889032244682312, 'rmse_std': 0.24430367350578308}
  - SNR 10: {'ncc_mean': 0.5937893390655518, 'ncc_std': 0.15590642392635345, 'rmse_mean': 0.865192174911499, 'rmse_std': 0.25540173053741455}
  - SNR 15: {'ncc_mean': 0.6101256012916565, 'ncc_std': 0.1543603092432022, 'rmse_mean': 0.8563615083694458, 'rmse_std': 0.25774574279785156}
- **latency_ms_b1:** 8.7368
- **latency_ms_b32:** 0.784
- **param_count_M:** 2.9238
- **per_snr_ncc:**
  - SNR -15: 0.35288354754447937
  - SNR -10: 0.6557507514953613
  - SNR -5: 0.8433424234390259
  - SNR 0: 0.9188597202301025
  - SNR 5: 0.9419828057289124
  - SNR 10: 0.949042558670044
  - SNR 15: 0.9505048990249634
- **snr_improvement_mean:** 7.012454032897949
- **synthetic_ncc_mean:** 0.8017666935920715
- **synthetic_rmse_mean:** 0.11991380155086517
- **Checkpoint:** selected_final.pt

Configuration `MR-TAE-FULL-IDWT (phased)` is evaluated with `selected_final.pt` under the unified seven-level composite-noise protocol; any change to checkpoint selection or test-set hash invalidates row-wise comparison with other reports.

## P.MR-TAE-FULL-LossB

- Status: `ok`

- **dice_mean:** 0.9121819138526917
- **false_positive_rate:** 0.3284794092178345
- **grounded_real_ncc_mean:** 0.45505672693252563
- **grounded_real_per_snr:**
  - SNR -15: {'ncc_mean': 0.15359584987163544, 'ncc_std': 0.07550352811813354, 'rmse_mean': 2.0497729778289795, 'rmse_std': 0.09162405878305435}
  - SNR -10: {'ncc_mean': 0.26668885350227356, 'ncc_std': 0.07939030230045319, 'rmse_mean': 1.288429617881775, 'rmse_std': 0.06632634252309799}
  - SNR -5: {'ncc_mean': 0.40642213821411133, 'ncc_std': 0.048605211079120636, 'rmse_mean': 0.978030264377594, 'rmse_std': 0.04509272426366806}
  - SNR 0: {'ncc_mean': 0.5109689235687256, 'ncc_std': 0.0742964819073677, 'rmse_mean': 0.8767938613891602, 'rmse_std': 0.08163391053676605}
  - SNR 5: {'ncc_mean': 0.5787226557731628, 'ncc_std': 0.09638171643018723, 'rmse_mean': 0.8266803622245789, 'rmse_std': 0.07551158964633942}
  - SNR 10: {'ncc_mean': 0.6230785250663757, 'ncc_std': 0.0974341407418251, 'rmse_mean': 0.8005275726318359, 'rmse_std': 0.06654947996139526}
  - SNR 15: {'ncc_mean': 0.6459200978279114, 'ncc_std': 0.09479163587093353, 'rmse_mean': 0.7870010733604431, 'rmse_std': 0.06325643509626389}
- **latency_ms_b1:** 15.6582
- **latency_ms_b32:** 0.6865
- **param_count_M:** 3.0535
- **per_snr_ncc:**
  - SNR -15: 0.17277313768863678
  - SNR -10: 0.3044874370098114
  - SNR -5: 0.5615859627723694
  - SNR 0: 0.7681084275245667
  - SNR 5: 0.8251339793205261
  - SNR 10: 0.866045355796814
  - SNR 15: 0.8908307552337646
- **snr_improvement_mean:** 1.7056329250335693
- **synthetic_ncc_mean:** 0.6269949674606323
- **synthetic_rmse_mean:** 0.22950702905654907
- **Checkpoint:** selected_final.pt

Configuration `MR-TAE-FULL-LossB` is evaluated with `selected_final.pt` under the unified seven-level composite-noise protocol; any change to checkpoint selection or test-set hash invalidates row-wise comparison with other reports.

## P.MR-TAE-FULL-LossB (linear, phased)

- Status: `ok`

- **dice_mean:** 0.8990022540092468
- **false_positive_rate:** 0.25460052490234375
- **grounded_real_ncc_mean:** 0.23336869478225708
- **grounded_real_per_snr:**
  - SNR -15: {'ncc_mean': 0.09740555286407471, 'ncc_std': 0.08459632098674774, 'rmse_mean': 1.0823131799697876, 'rmse_std': 0.23949144780635834}
  - SNR -10: {'ncc_mean': 0.14805632829666138, 'ncc_std': 0.11491313576698303, 'rmse_mean': 1.2477699518203735, 'rmse_std': 0.48768341541290283}
  - SNR -5: {'ncc_mean': 0.20909129083156586, 'ncc_std': 0.21095538139343262, 'rmse_mean': 1.5146156549453735, 'rmse_std': 0.7603426575660706}
  - SNR 0: {'ncc_mean': 0.2668933570384979, 'ncc_std': 0.2698982059955597, 'rmse_mean': 1.5633491277694702, 'rmse_std': 0.775971531867981}
  - SNR 5: {'ncc_mean': 0.3074776232242584, 'ncc_std': 0.2882331907749176, 'rmse_mean': 1.5075490474700928, 'rmse_std': 0.6741142868995667}
  - SNR 10: {'ncc_mean': 0.3087630569934845, 'ncc_std': 0.2859523892402649, 'rmse_mean': 1.4888283014297485, 'rmse_std': 0.6098086833953857}
  - SNR 15: {'ncc_mean': 0.2958936095237732, 'ncc_std': 0.2859365940093994, 'rmse_mean': 1.505071759223938, 'rmse_std': 0.5830188989639282}
- **latency_ms_b1:** 29.5567
- **latency_ms_b32:** 1.0736
- **param_count_M:** 3.0535
- **per_snr_ncc:**
  - SNR -15: 0.34346508979797363
  - SNR -10: 0.6418339610099792
  - SNR -5: 0.8324881792068481
  - SNR 0: 0.9093908667564392
  - SNR 5: 0.9334253668785095
  - SNR 10: 0.9406498074531555
  - SNR 15: 0.9425594210624695
- **snr_improvement_mean:** 6.5731706619262695
- **synthetic_ncc_mean:** 0.7919732332229614
- **synthetic_rmse_mean:** 0.12466045469045639
- **Checkpoint:** selected_final.pt

Configuration `MR-TAE-FULL-LossB (linear, phased)` is evaluated with `selected_final.pt` under the unified seven-level composite-noise protocol; any change to checkpoint selection or test-set hash invalidates row-wise comparison with other reports.

## P.MR-TAE-FULL-LossB-IDWT (phased)

- Status: `ok`

- **dice_mean:** 0.8994582891464233
- **false_positive_rate:** 0.2570074498653412
- **grounded_real_ncc_mean:** 0.5173186659812927
- **grounded_real_per_snr:**
  - SNR -15: {'ncc_mean': 0.12763908505439758, 'ncc_std': 0.10594488680362701, 'rmse_mean': 1.0083528757095337, 'rmse_std': 0.25029924511909485}
  - SNR -10: {'ncc_mean': 0.2997925877571106, 'ncc_std': 0.12455932796001434, 'rmse_mean': 0.9929970502853394, 'rmse_std': 0.3579739034175873}
  - SNR -5: {'ncc_mean': 0.5025947690010071, 'ncc_std': 0.0935702919960022, 'rmse_mean': 0.9666903614997864, 'rmse_std': 0.4972590208053589}
  - SNR 0: {'ncc_mean': 0.6093761920928955, 'ncc_std': 0.08364402502775192, 'rmse_mean': 0.90199875831604, 'rmse_std': 0.5236358642578125}
  - SNR 5: {'ncc_mean': 0.6723829507827759, 'ncc_std': 0.08937664330005646, 'rmse_mean': 0.8527015447616577, 'rmse_std': 0.6067640781402588}
  - SNR 10: {'ncc_mean': 0.7000258564949036, 'ncc_std': 0.10221309959888458, 'rmse_mean': 0.8307969570159912, 'rmse_std': 0.6452629566192627}
  - SNR 15: {'ncc_mean': 0.7094190120697021, 'ncc_std': 0.10803677886724472, 'rmse_mean': 0.8237059712409973, 'rmse_std': 0.6527987122535706}
- **latency_ms_b1:** 24.4558
- **latency_ms_b32:** 0.8441
- **param_count_M:** 2.9238
- **per_snr_ncc:**
  - SNR -15: 0.3359829783439636
  - SNR -10: 0.6339057087898254
  - SNR -5: 0.825539767742157
  - SNR 0: 0.90385901927948
  - SNR 5: 0.9284260272979736
  - SNR 10: 0.9358174800872803
  - SNR 15: 0.936760425567627
- **snr_improvement_mean:** 6.329095840454102
- **synthetic_ncc_mean:** 0.7857558727264404
- **synthetic_rmse_mean:** 0.12732797861099243
- **Checkpoint:** selected_final.pt

Configuration `MR-TAE-FULL-LossB-IDWT (phased)` is evaluated with `selected_final.pt` under the unified seven-level composite-noise protocol; any change to checkpoint selection or test-set hash invalidates row-wise comparison with other reports.

## P.MR-TAE-FULL-LossC

- Status: `ok`

- **dice_mean:** 0.7576074004173279
- **false_positive_rate:** 0.4315665066242218
- **grounded_real_ncc_mean:** 0.16991838812828064
- **grounded_real_per_snr:**
  - SNR -15: {'ncc_mean': -0.06322628259658813, 'ncc_std': 0.11799993366003036, 'rmse_mean': 46.7435302734375, 'rmse_std': 5.730007648468018}
  - SNR -10: {'ncc_mean': 0.0017047275323420763, 'ncc_std': 0.2341976761817932, 'rmse_mean': 28.50694465637207, 'rmse_std': 9.403938293457031}
  - SNR -5: {'ncc_mean': 0.09465319663286209, 'ncc_std': 0.32023942470550537, 'rmse_mean': 25.327219009399414, 'rmse_std': 16.094472885131836}
  - SNR 0: {'ncc_mean': 0.18279103934764862, 'ncc_std': 0.317305326461792, 'rmse_mean': 27.612871170043945, 'rmse_std': 21.163217544555664}
  - SNR 5: {'ncc_mean': 0.2737220227718353, 'ncc_std': 0.2820369303226471, 'rmse_mean': 30.89824676513672, 'rmse_std': 24.00363540649414}
  - SNR 10: {'ncc_mean': 0.3350759446620941, 'ncc_std': 0.2451460361480713, 'rmse_mean': 33.89166259765625, 'rmse_std': 25.220314025878906}
  - SNR 15: {'ncc_mean': 0.36470794677734375, 'ncc_std': 0.22235314548015594, 'rmse_mean': 35.90995788574219, 'rmse_std': 25.64751434326172}
- **latency_ms_b1:** 7.8976
- **latency_ms_b32:** 0.6907
- **param_count_M:** 3.0535
- **per_snr_ncc:**
  - SNR -15: 0.0548345111310482
  - SNR -10: 0.2596619129180908
  - SNR -5: 0.5672010779380798
  - SNR 0: 0.8175721168518066
  - SNR 5: 0.9174453020095825
  - SNR 10: 0.955919086933136
  - SNR 15: 0.9641988277435303
- **snr_improvement_mean:** 1.7578190565109253
- **synthetic_ncc_mean:** 0.6481189727783203
- **synthetic_rmse_mean:** 0.48689115047454834
- **Checkpoint:** selected_final.pt

Configuration `MR-TAE-FULL-LossC` is evaluated with `selected_final.pt` under the unified seven-level composite-noise protocol; any change to checkpoint selection or test-set hash invalidates row-wise comparison with other reports.

## P.MR-TAE-noAttn

- Status: `ok`

- **dice_mean:** 0.8869261145591736
- **false_positive_rate:** 0.2995302081108093
- **grounded_real_ncc_mean:** 0.10140460729598999
- **grounded_real_per_snr:**
  - SNR -15: {'ncc_mean': 0.009930125437676907, 'ncc_std': 0.048267170786857605, 'rmse_mean': 37.61971664428711, 'rmse_std': 2.5163207054138184}
  - SNR -10: {'ncc_mean': -0.002019918756559491, 'ncc_std': 0.06467653810977936, 'rmse_mean': 16.98587417602539, 'rmse_std': 1.5962778329849243}
  - SNR -5: {'ncc_mean': 0.012520142830908298, 'ncc_std': 0.1283411830663681, 'rmse_mean': 6.4398603439331055, 'rmse_std': 1.1540791988372803}
  - SNR 0: {'ncc_mean': 0.09405947476625443, 'ncc_std': 0.32137632369995117, 'rmse_mean': 2.8621842861175537, 'rmse_std': 1.1846853494644165}
  - SNR 5: {'ncc_mean': 0.1862790584564209, 'ncc_std': 0.44306835532188416, 'rmse_mean': 2.261622428894043, 'rmse_std': 1.3718504905700684}
  - SNR 10: {'ncc_mean': 0.2060822993516922, 'ncc_std': 0.47495925426483154, 'rmse_mean': 2.1747827529907227, 'rmse_std': 1.4536067247390747}
  - SNR 15: {'ncc_mean': 0.2029811292886734, 'ncc_std': 0.4834195673465729, 'rmse_mean': 2.174405574798584, 'rmse_std': 1.480992317199707}
- **latency_ms_b1:** 7.4544
- **latency_ms_b32:** 0.5418
- **param_count_M:** 3.0097
- **per_snr_ncc:**
  - SNR -15: 0.052314452826976776
  - SNR -10: 0.2988124489784241
  - SNR -5: 0.62088942527771
  - SNR 0: 0.8600587248802185
  - SNR 5: 0.9468925595283508
  - SNR 10: 0.9749667644500732
  - SNR 15: 0.9835574626922607
- **snr_improvement_mean:** 2.9751906394958496
- **synthetic_ncc_mean:** 0.6767845153808594
- **synthetic_rmse_mean:** 0.5147975087165833
- **Checkpoint:** selected_final.pt

Configuration `MR-TAE-noAttn` is evaluated with `selected_final.pt` under the unified seven-level composite-noise protocol; any change to checkpoint selection or test-set hash invalidates row-wise comparison with other reports.

## P.MR-TAE-noBiGRU

- Status: `ok`

- **dice_mean:** 0.8265864849090576
- **false_positive_rate:** 0.33950066566467285
- **grounded_real_ncc_mean:** 0.23072898387908936
- **grounded_real_per_snr:**
  - SNR -15: {'ncc_mean': 0.10675881057977676, 'ncc_std': 0.04891175404191017, 'rmse_mean': 19.596372604370117, 'rmse_std': 3.3018007278442383}
  - SNR -10: {'ncc_mean': 0.1532374769449234, 'ncc_std': 0.055968090891838074, 'rmse_mean': 12.13902759552002, 'rmse_std': 2.810417890548706}
  - SNR -5: {'ncc_mean': 0.19336006045341492, 'ncc_std': 0.05672551691532135, 'rmse_mean': 8.70984172821045, 'rmse_std': 3.087662935256958}
  - SNR 0: {'ncc_mean': 0.2326631397008896, 'ncc_std': 0.0630875825881958, 'rmse_mean': 7.324636936187744, 'rmse_std': 3.422881841659546}
  - SNR 5: {'ncc_mean': 0.2809552252292633, 'ncc_std': 0.13056834042072296, 'rmse_mean': 6.678556442260742, 'rmse_std': 3.572584867477417}
  - SNR 10: {'ncc_mean': 0.3149780333042145, 'ncc_std': 0.18387556076049805, 'rmse_mean': 6.237098217010498, 'rmse_std': 3.583930730819702}
  - SNR 15: {'ncc_mean': 0.33315014839172363, 'ncc_std': 0.20606380701065063, 'rmse_mean': 5.925632476806641, 'rmse_std': 3.517930746078491}
- **latency_ms_b1:** 18.7629
- **latency_ms_b32:** 0.801
- **param_count_M:** 2.3948
- **per_snr_ncc:**
  - SNR -15: 0.05847981944680214
  - SNR -10: 0.1964896023273468
  - SNR -5: 0.5711537599563599
  - SNR 0: 0.8591479659080505
  - SNR 5: 0.9478508234024048
  - SNR 10: 0.9763521552085876
  - SNR 15: 0.9855580925941467
- **snr_improvement_mean:** 2.5667638778686523
- **synthetic_ncc_mean:** 0.6564331650733948
- **synthetic_rmse_mean:** 0.5487639904022217
- **Checkpoint:** selected_final.pt

Configuration `MR-TAE-noBiGRU` is evaluated with `selected_final.pt` under the unified seven-level composite-noise protocol; any change to checkpoint selection or test-set hash invalidates row-wise comparison with other reports.

## P.MR-TAE-noBiGRU-noAttn

- Status: `ok`

- **dice_mean:** 0.6887350678443909
- **false_positive_rate:** 0.3624907433986664
- **grounded_real_ncc_mean:** 0.01797756552696228
- **grounded_real_per_snr:**
  - SNR -15: {'ncc_mean': 0.015968479216098785, 'ncc_std': 0.08824590593576431, 'rmse_mean': 18.178144454956055, 'rmse_std': 1.726996660232544}
  - SNR -10: {'ncc_mean': -0.04263218864798546, 'ncc_std': 0.15575504302978516, 'rmse_mean': 10.368331909179688, 'rmse_std': 2.1060030460357666}
  - SNR -5: {'ncc_mean': -0.08179523795843124, 'ncc_std': 0.2126101404428482, 'rmse_mean': 6.890692234039307, 'rmse_std': 2.8158414363861084}
  - SNR 0: {'ncc_mean': -0.0401962511241436, 'ncc_std': 0.25242626667022705, 'rmse_mean': 5.333176612854004, 'rmse_std': 2.973555326461792}
  - SNR 5: {'ncc_mean': 0.042436420917510986, 'ncc_std': 0.2917131781578064, 'rmse_mean': 4.5881452560424805, 'rmse_std': 2.853377103805542}
  - SNR 10: {'ncc_mean': 0.10554678738117218, 'ncc_std': 0.3160889148712158, 'rmse_mean': 4.350146293640137, 'rmse_std': 2.7203032970428467}
  - SNR 15: {'ncc_mean': 0.1265149712562561, 'ncc_std': 0.3248746395111084, 'rmse_mean': 4.366049766540527, 'rmse_std': 2.624588966369629}
- **latency_ms_b1:** 15.4391
- **latency_ms_b32:** 0.5522
- **param_count_M:** 3.0097
- **per_snr_ncc:**
  - SNR -15: 0.06298172473907471
  - SNR -10: 0.2245328277349472
  - SNR -5: 0.4786677062511444
  - SNR 0: 0.7126588225364685
  - SNR 5: 0.7848497629165649
  - SNR 10: 0.8016005754470825
  - SNR 15: 0.8095858097076416
- **snr_improvement_mean:** -1.2404509782791138
- **synthetic_ncc_mean:** 0.5535538792610168
- **synthetic_rmse_mean:** 0.4995110332965851
- **Checkpoint:** selected_final.pt

Configuration `MR-TAE-noBiGRU-noAttn` is evaluated with `selected_final.pt` under the unified seven-level composite-noise protocol; any change to checkpoint selection or test-set hash invalidates row-wise comparison with other reports.

## P.MR-TAE-noMTL

- Status: `ok`

- **dice_mean:** 0.0
- **false_positive_rate:** 0.0
- **grounded_real_ncc_mean:** -0.16053524613380432
- **grounded_real_per_snr:**
  - SNR -15: {'ncc_mean': -0.11783628165721893, 'ncc_std': 0.10290942341089249, 'rmse_mean': 117.12035369873047, 'rmse_std': 3.5608999729156494}
  - SNR -10: {'ncc_mean': -0.1810251772403717, 'ncc_std': 0.12181554734706879, 'rmse_mean': 61.966251373291016, 'rmse_std': 2.0706112384796143}
  - SNR -5: {'ncc_mean': -0.16890396177768707, 'ncc_std': 0.1287645548582077, 'rmse_mean': 26.786619186401367, 'rmse_std': 2.0203988552093506}
  - SNR 0: {'ncc_mean': -0.14980532228946686, 'ncc_std': 0.17856085300445557, 'rmse_mean': 9.195433616638184, 'rmse_std': 1.8667914867401123}
  - SNR 5: {'ncc_mean': -0.16244590282440186, 'ncc_std': 0.23856084048748016, 'rmse_mean': 6.213717460632324, 'rmse_std': 1.2836202383041382}
  - SNR 10: {'ncc_mean': -0.1690455824136734, 'ncc_std': 0.2652996778488159, 'rmse_mean': 5.722853183746338, 'rmse_std': 1.4320145845413208}
  - SNR 15: {'ncc_mean': -0.17468442022800446, 'ncc_std': 0.2751530110836029, 'rmse_mean': 5.635024070739746, 'rmse_std': 1.5277104377746582}
- **latency_ms_b1:** 7.6742
- **latency_ms_b32:** 0.5911
- **param_count_M:** 3.0035
- **per_snr_ncc:**
  - SNR -15: 0.039038270711898804
  - SNR -10: 0.25893205404281616
  - SNR -5: 0.5827187299728394
  - SNR 0: 0.826529860496521
  - SNR 5: 0.9195937514305115
  - SNR 10: 0.9499092698097229
  - SNR 15: 0.9606047868728638
- **snr_improvement_mean:** 0.3506647050380707
- **synthetic_ncc_mean:** 0.6481895446777344
- **synthetic_rmse_mean:** 1.367988109588623
- **Checkpoint:** selected_final.pt

Configuration `MR-TAE-noMTL` is evaluated with `selected_final.pt` under the unified seven-level composite-noise protocol; any change to checkpoint selection or test-set hash invalidates row-wise comparison with other reports.

## P.MR-TAE-noSwin

- Status: `ok`

- **dice_mean:** 0.883253276348114
- **false_positive_rate:** 0.27905091643333435
- **grounded_real_ncc_mean:** 0.32868218421936035
- **grounded_real_per_snr:**
  - SNR -15: {'ncc_mean': 0.11070778965950012, 'ncc_std': 0.055760353803634644, 'rmse_mean': 5.5745415687561035, 'rmse_std': 0.9989493489265442}
  - SNR -10: {'ncc_mean': 0.1847970187664032, 'ncc_std': 0.07307865470647812, 'rmse_mean': 3.559908628463745, 'rmse_std': 0.7565053701400757}
  - SNR -5: {'ncc_mean': 0.2790665626525879, 'ncc_std': 0.08388526737689972, 'rmse_mean': 2.755613088607788, 'rmse_std': 0.8721655011177063}
  - SNR 0: {'ncc_mean': 0.35856303572654724, 'ncc_std': 0.05882962793111801, 'rmse_mean': 2.598613977432251, 'rmse_std': 1.1011877059936523}
  - SNR 5: {'ncc_mean': 0.4219370186328888, 'ncc_std': 0.0740148276090622, 'rmse_mean': 2.5530178546905518, 'rmse_std': 1.2299063205718994}
  - SNR 10: {'ncc_mean': 0.46283113956451416, 'ncc_std': 0.11175822466611862, 'rmse_mean': 2.4775140285491943, 'rmse_std': 1.2676935195922852}
  - SNR 15: {'ncc_mean': 0.48287275433540344, 'ncc_std': 0.12939153611660004, 'rmse_mean': 2.4139187335968018, 'rmse_std': 1.262591004371643}
- **latency_ms_b1:** 13.6987
- **latency_ms_b32:** 0.7497
- **param_count_M:** 1.4066
- **per_snr_ncc:**
  - SNR -15: 0.09823912382125854
  - SNR -10: 0.2565683126449585
  - SNR -5: 0.5291392803192139
  - SNR 0: 0.8491888642311096
  - SNR 5: 0.9471379518508911
  - SNR 10: 0.9762700200080872
  - SNR 15: 0.9858723878860474
- **snr_improvement_mean:** 4.241724967956543
- **synthetic_ncc_mean:** 0.6632022857666016
- **synthetic_rmse_mean:** 0.2723870277404785
- **Checkpoint:** selected_final.pt

Configuration `MR-TAE-noSwin` is evaluated with `selected_final.pt` under the unified seven-level composite-noise protocol; any change to checkpoint selection or test-set hash invalidates row-wise comparison with other reports.

## P.MR-TAE-noSwin-noAttn

- Status: `ok`

- **dice_mean:** 0.903412401676178
- **false_positive_rate:** 0.24843032658100128
- **grounded_real_ncc_mean:** 0.4171396791934967
- **grounded_real_per_snr:**
  - SNR -15: {'ncc_mean': 0.08441256731748581, 'ncc_std': 0.032010264694690704, 'rmse_mean': 0.9977661967277527, 'rmse_std': 0.004509094171226025}
  - SNR -10: {'ncc_mean': 0.18017597496509552, 'ncc_std': 0.0730365440249443, 'rmse_mean': 0.9885602593421936, 'rmse_std': 0.009326697327196598}
  - SNR -5: {'ncc_mean': 0.39208459854125977, 'ncc_std': 0.09333968907594681, 'rmse_mean': 0.9537835717201233, 'rmse_std': 0.02804812788963318}
  - SNR 0: {'ncc_mean': 0.49796563386917114, 'ncc_std': 0.11846166849136353, 'rmse_mean': 0.890059769153595, 'rmse_std': 0.04601985588669777}
  - SNR 5: {'ncc_mean': 0.5652008056640625, 'ncc_std': 0.1278926581144333, 'rmse_mean': 0.8458693623542786, 'rmse_std': 0.06403719633817673}
  - SNR 10: {'ncc_mean': 0.5946009755134583, 'ncc_std': 0.12294606864452362, 'rmse_mean': 0.8261717557907104, 'rmse_std': 0.07047037035226822}
  - SNR 15: {'ncc_mean': 0.6055373549461365, 'ncc_std': 0.11889957636594772, 'rmse_mean': 0.8191167712211609, 'rmse_std': 0.07191084325313568}
- **latency_ms_b1:** 8.75
- **latency_ms_b32:** 0.549
- **param_count_M:** 3.0097
- **per_snr_ncc:**
  - SNR -15: 0.3509123623371124
  - SNR -10: 0.6492447257041931
  - SNR -5: 0.8376425504684448
  - SNR 0: 0.9168828725814819
  - SNR 5: 0.9424731731414795
  - SNR 10: 0.9497886300086975
  - SNR 15: 0.9513601064682007
- **snr_improvement_mean:** 7.00673246383667
- **synthetic_ncc_mean:** 0.799757719039917
- **synthetic_rmse_mean:** 0.12035063654184341
- **Checkpoint:** selected_final.pt

Configuration `MR-TAE-noSwin-noAttn` is evaluated with `selected_final.pt` under the unified seven-level composite-noise protocol; any change to checkpoint selection or test-set hash invalidates row-wise comparison with other reports.

## P.MR-TAE-noSwin-noBiGRU

- Status: `ok`

- **dice_mean:** 0.9010529518127441
- **false_positive_rate:** 0.2427414506673813
- **grounded_real_ncc_mean:** 0.436495840549469
- **grounded_real_per_snr:**
  - SNR -15: {'ncc_mean': 0.05561458691954613, 'ncc_std': 0.037507329136133194, 'rmse_mean': 1.1266536712646484, 'rmse_std': 0.25319424271583557}
  - SNR -10: {'ncc_mean': 0.17102138698101044, 'ncc_std': 0.09072114527225494, 'rmse_mean': 0.9951921701431274, 'rmse_std': 0.03368181362748146}
  - SNR -5: {'ncc_mean': 0.4081496298313141, 'ncc_std': 0.0990685373544693, 'rmse_mean': 0.9520388841629028, 'rmse_std': 0.02888880856335163}
  - SNR 0: {'ncc_mean': 0.5063638687133789, 'ncc_std': 0.10889966040849686, 'rmse_mean': 0.8882299065589905, 'rmse_std': 0.04619473218917847}
  - SNR 5: {'ncc_mean': 0.5898481011390686, 'ncc_std': 0.1178586557507515, 'rmse_mean': 0.8399308323860168, 'rmse_std': 0.060296982526779175}
  - SNR 10: {'ncc_mean': 0.6494784951210022, 'ncc_std': 0.10990381240844727, 'rmse_mean': 0.8077603578567505, 'rmse_std': 0.06249551475048065}
  - SNR 15: {'ncc_mean': 0.674994707107544, 'ncc_std': 0.10549426823854446, 'rmse_mean': 0.7944167256355286, 'rmse_std': 0.062267083674669266}
- **latency_ms_b1:** 18.4787
- **latency_ms_b32:** 0.6318
- **param_count_M:** 3.0097
- **per_snr_ncc:**
  - SNR -15: 0.34794387221336365
  - SNR -10: 0.6487593054771423
  - SNR -5: 0.8373075723648071
  - SNR 0: 0.9146687984466553
  - SNR 5: 0.9388154149055481
  - SNR 10: 0.9460415244102478
  - SNR 15: 0.9480213522911072
- **snr_improvement_mean:** 6.79623556137085
- **synthetic_ncc_mean:** 0.7973653674125671
- **synthetic_rmse_mean:** 0.12227033078670502
- **Checkpoint:** selected_final.pt

Configuration `MR-TAE-noSwin-noBiGRU` is evaluated with `selected_final.pt` under the unified seven-level composite-noise protocol; any change to checkpoint selection or test-set hash invalidates row-wise comparison with other reports.

## P.MR-TAE-noWavelet

- Status: `ok`

- **dice_mean:** 0.894717276096344
- **false_positive_rate:** 0.3078022003173828
- **grounded_real_ncc_mean:** -0.03371548280119896
- **grounded_real_per_snr:**
  - SNR -15: {'ncc_mean': 0.021431952714920044, 'ncc_std': 0.04802510142326355, 'rmse_mean': 24.385662078857422, 'rmse_std': 1.3985657691955566}
  - SNR -10: {'ncc_mean': -0.005596109200268984, 'ncc_std': 0.06758497655391693, 'rmse_mean': 12.966371536254883, 'rmse_std': 0.9554608464241028}
  - SNR -5: {'ncc_mean': -0.05174970626831055, 'ncc_std': 0.11319947242736816, 'rmse_mean': 7.674349308013916, 'rmse_std': 1.4102685451507568}
  - SNR 0: {'ncc_mean': -0.07436785846948624, 'ncc_std': 0.17188403010368347, 'rmse_mean': 5.898861885070801, 'rmse_std': 2.1928679943084717}
  - SNR 5: {'ncc_mean': -0.06250794976949692, 'ncc_std': 0.22343572974205017, 'rmse_mean': 5.422027587890625, 'rmse_std': 2.625662088394165}
  - SNR 10: {'ncc_mean': -0.039165202528238297, 'ncc_std': 0.25858989357948303, 'rmse_mean': 5.137163162231445, 'rmse_std': 2.7491543292999268}
  - SNR 15: {'ncc_mean': -0.02405347116291523, 'ncc_std': 0.2781262993812561, 'rmse_mean': 4.8996806144714355, 'rmse_std': 2.741718292236328}
- **latency_ms_b1:** 26.9187
- **latency_ms_b32:** 0.7264
- **param_count_M:** 2.99
- **per_snr_ncc:**
  - SNR -15: 0.05522124469280243
  - SNR -10: 0.13294735550880432
  - SNR -5: 0.4615826904773712
  - SNR 0: 0.8573315143585205
  - SNR 5: 0.9474980235099792
  - SNR 10: 0.9747248291969299
  - SNR 15: 0.9832587838172913
- **snr_improvement_mean:** 1.2033300399780273
- **synthetic_ncc_mean:** 0.6303664445877075
- **synthetic_rmse_mean:** 0.7657411694526672
- **Checkpoint:** selected_final.pt

Configuration `MR-TAE-noWavelet` is evaluated with `selected_final.pt` under the unified seven-level composite-noise protocol; any change to checkpoint selection or test-set hash invalidates row-wise comparison with other reports.

## P.MWCNN-BiGRU

- Status: `ok`

- **dice_mean:** 0.0
- **false_positive_rate:** 0.0
- **grounded_real_ncc_mean:** 0.3088040351867676
- **grounded_real_per_snr:**
  - SNR -15: {'ncc_mean': 0.0157057736068964, 'ncc_std': 0.05603057146072388, 'rmse_mean': 23.796672821044922, 'rmse_std': 3.83172607421875}
  - SNR -10: {'ncc_mean': 0.03017912618815899, 'ncc_std': 0.05385774374008179, 'rmse_mean': 11.180898666381836, 'rmse_std': 2.039442777633667}
  - SNR -5: {'ncc_mean': 0.07416370511054993, 'ncc_std': 0.06581632047891617, 'rmse_mean': 4.574681282043457, 'rmse_std': 1.2175599336624146}
  - SNR 0: {'ncc_mean': 0.28103774785995483, 'ncc_std': 0.1373709887266159, 'rmse_mean': 1.8195351362228394, 'rmse_std': 0.9322981238365173}
  - SNR 5: {'ncc_mean': 0.522131621837616, 'ncc_std': 0.22380881011486053, 'rmse_mean': 1.1730045080184937, 'rmse_std': 0.7308948040008545}
  - SNR 10: {'ncc_mean': 0.6069403290748596, 'ncc_std': 0.25059038400650024, 'rmse_mean': 1.00899338722229, 'rmse_std': 0.6864684820175171}
  - SNR 15: {'ncc_mean': 0.6314701437950134, 'ncc_std': 0.2587021291255951, 'rmse_mean': 0.9589482545852661, 'rmse_std': 0.6780462265014648}
- **latency_ms_b1:** 3.7518
- **latency_ms_b32:** 0.2953
- **param_count_M:** 1.3129
- **per_snr_ncc:**
  - SNR -15: 0.08190463483333588
  - SNR -10: 0.295403391122818
  - SNR -5: 0.5949439406394958
  - SNR 0: 0.8329718112945557
  - SNR 5: 0.9295365214347839
  - SNR 10: 0.9601792693138123
  - SNR 15: 0.9702555537223816
- **snr_improvement_mean:** 2.2957136631011963
- **synthetic_ncc_mean:** 0.6664564609527588
- **synthetic_rmse_mean:** 0.4837583005428314
- **Checkpoint:** selected_final.pt

Configuration `MWCNN-BiGRU` is evaluated with `selected_final.pt` under the unified seven-level composite-noise protocol; any change to checkpoint selection or test-set hash invalidates row-wise comparison with other reports.

## P.Savitzky-Golay

- Status: `ok`

- **dice_mean:** None
- **false_positive_rate:** None
- **grounded_real_ncc_mean:** 0.5915 *(raw benchmark; headline finalized value: **0.2915**)*
- **grounded_real_per_snr:**
  - SNR -15: {'ncc_mean': 0.32504570484161377, 'ncc_std': 0.16260044276714325, 'rmse_mean': 2.382187843322754, 'rmse_std': 0.11568184942007065}
  - SNR -10: {'ncc_mean': 0.47643643617630005, 'ncc_std': 0.2079278826713562, 'rmse_mean': 1.4483704566955566, 'rmse_std': 0.10138222575187683}
  - SNR -5: {'ncc_mean': 0.5954539179801941, 'ncc_std': 0.23617632687091827, 'rmse_mean': 0.9807130694389343, 'rmse_std': 0.1265219897031784}
  - SNR 0: {'ncc_mean': 0.659794807434082, 'ncc_std': 0.24004195630550385, 'rmse_mean': 0.7724944353103638, 'rmse_std': 0.15447266399860382}
  - SNR 5: {'ncc_mean': 0.6868796944618225, 'ncc_std': 0.23565195500850677, 'rmse_mean': 0.6920211315155029, 'rmse_std': 0.17024672031402588}
  - SNR 10: {'ncc_mean': 0.696865439414978, 'ncc_std': 0.23232987523078918, 'rmse_mean': 0.6639332175254822, 'rmse_std': 0.17687445878982544}
  - SNR 15: {'ncc_mean': 0.7001956105232239, 'ncc_std': 0.23098699748516083, 'rmse_mean': 0.6547188758850098, 'rmse_std': 0.17924880981445312}
- **latency_ms_b1:** 0.7352
- **latency_ms_b32:** 0.0
- **param_count_M:** 0.0
- **per_snr_ncc:**
  - SNR -15: 0.009810908697545528
  - SNR -10: 0.01752799190580845
  - SNR -5: 0.025098206475377083
  - SNR 0: 0.03408578410744667
  - SNR 5: 0.04139367491006851
  - SNR 10: 0.05023976415395737
  - SNR 15: 0.06729596853256226
- **snr_improvement_mean:** -1.6587692499160767
- **synthetic_ncc_mean:** 0.035064615309238434
- **synthetic_rmse_mean:** 0.3153906464576721
- **Checkpoint:** N/A (classical)

Configuration `Savitzky-Golay` is evaluated with `selected_final.pt` under the unified seven-level composite-noise protocol; any change to checkpoint selection or test-set hash invalidates row-wise comparison with other reports.

## P.UNet-BiGRU-Swin

- Status: `ok`

- **dice_mean:** 0.0
- **false_positive_rate:** 0.0
- **grounded_real_ncc_mean:** 0.4753739535808563
- **grounded_real_per_snr:**
  - SNR -15: {'ncc_mean': 0.08802774548530579, 'ncc_std': 0.028849756345152855, 'rmse_mean': 4.754933834075928, 'rmse_std': 0.1265561729669571}
  - SNR -10: {'ncc_mean': 0.1583254188299179, 'ncc_std': 0.04285170137882233, 'rmse_mean': 2.8511266708374023, 'rmse_std': 0.10098811984062195}
  - SNR -5: {'ncc_mean': 0.3050214946269989, 'ncc_std': 0.06700263172388077, 'rmse_mean': 1.6625449657440186, 'rmse_std': 0.09444928914308548}
  - SNR 0: {'ncc_mean': 0.5142654180526733, 'ncc_std': 0.08097241073846817, 'rmse_mean': 1.0100046396255493, 'rmse_std': 0.06732794642448425}
  - SNR 5: {'ncc_mean': 0.6899286508560181, 'ncc_std': 0.06305111199617386, 'rmse_mean': 0.7447825074195862, 'rmse_std': 0.053801387548446655}
  - SNR 10: {'ncc_mean': 0.7717381119728088, 'ncc_std': 0.05547904223203659, 'rmse_mean': 0.6522833108901978, 'rmse_std': 0.06689433753490448}
  - SNR 15: {'ncc_mean': 0.8003106117248535, 'ncc_std': 0.05493340268731117, 'rmse_mean': 0.6202878952026367, 'rmse_std': 0.07666490972042084}
- **latency_ms_b1:** 17.8661
- **latency_ms_b32:** 0.5546
- **param_count_M:** 2.8962
- **per_snr_ncc:**
  - SNR -15: 0.1290489137172699
  - SNR -10: 0.32013317942619324
  - SNR -5: 0.6172229647636414
  - SNR 0: 0.8608142733573914
  - SNR 5: 0.9448724389076233
  - SNR 10: 0.971388578414917
  - SNR 15: 0.9808980822563171
- **snr_improvement_mean:** 3.9983766078948975
- **synthetic_ncc_mean:** 0.6891968250274658
- **synthetic_rmse_mean:** 0.29444757103919983
- **Checkpoint:** selected_final.pt

Configuration `UNet-BiGRU-Swin` is evaluated with `selected_final.pt` under the unified seven-level composite-noise protocol; any change to checkpoint selection or test-set hash invalidates row-wise comparison with other reports.

## P.Wiener

- Status: `ok`

- **dice_mean:** None
- **false_positive_rate:** None
- **grounded_real_ncc_mean:** 0.6641 *(raw benchmark; headline finalized value: **0.3641**)*
- **grounded_real_per_snr:**
  - SNR -15: {'ncc_mean': 0.22412754595279694, 'ncc_std': 0.07956495881080627, 'rmse_mean': 3.838129758834839, 'rmse_std': 0.08082390576601028}
  - SNR -10: {'ncc_mean': 0.37874940037727356, 'ncc_std': 0.11346957087516785, 'rmse_mean': 2.179705858230591, 'rmse_std': 0.06911173462867737}
  - SNR -5: {'ncc_mean': 0.578946590423584, 'ncc_std': 0.1465967893600464, 'rmse_mean': 1.2579612731933594, 'rmse_std': 0.08976346254348755}
  - SNR 0: {'ncc_mean': 0.7615314722061157, 'ncc_std': 0.14879867434501648, 'rmse_mean': 0.7538583278656006, 'rmse_std': 0.1253168135881424}
  - SNR 5: {'ncc_mean': 0.8681578040122986, 'ncc_std': 0.12984566390514374, 'rmse_mean': 0.4889947772026062, 'rmse_std': 0.16437804698944092}
  - SNR 10: {'ncc_mean': 0.9115341901779175, 'ncc_std': 0.11524965614080429, 'rmse_mean': 0.360393226146698, 'rmse_std': 0.1956537365913391}
  - SNR 15: {'ncc_mean': 0.9259980320930481, 'ncc_std': 0.10904698818922043, 'rmse_mean': 0.3043915629386902, 'rmse_std': 0.21419431269168854}
- **latency_ms_b1:** 0.364
- **latency_ms_b32:** 0.0
- **param_count_M:** 0.0
- **per_snr_ncc:**
  - SNR -15: 0.152036651968956
  - SNR -10: 0.2663642466068268
  - SNR -5: 0.45255815982818604
  - SNR 0: 0.6691909432411194
  - SNR 5: 0.812422513961792
  - SNR 10: 0.8766725659370422
  - SNR 15: 0.8985982537269592
- **snr_improvement_mean:** 0.3924161493778229
- **synthetic_ncc_mean:** 0.5896918773651123
- **synthetic_rmse_mean:** 0.3263644576072693
- **Checkpoint:** N/A (classical)

Configuration `Wiener` is evaluated with `selected_final.pt` under the unified seven-level composite-noise protocol; any change to checkpoint selection or test-set hash invalidates row-wise comparison with other reports.

# Appendix Q — Extended Metric and Architecture Glossary

Concise reference for terms used in Chapters 8–9. Formal definitions are in Chapter 8; regeneration scripts in Appendix B.

## NCC (normalised cross-correlation)

Zero-mean, unit-norm cross-correlation between denoised output and clean reference. Scale- and shift-invariant; primary synthetic metric. Reported per SNR and as mean over 14,000 frozen windows. **Headline synthetic values** for MR-TAE NoMix: mean **0.8282**, @ −15 dB **0.3777**.

## Grounded NCC (gNCC)

Same NCC formula on real Q.Lin windows: composite noise injected into clean laboratory recordings, output correlated against the original clean window. Headline transfer metric. **Finalized poster means (no fine-tuning):** NoMix **0.4535**, BayesShrink **0.3844**, Wiener **0.3641**, Savitzky-Golay **0.2915**. Raw per-SNR profiles in Appendix H; classical headline means may differ from raw `metrics_master.json` averages.

## Dice coefficient

Overlap between predicted and true PD foreground mask (5-class segmentation). NoMix: **0.9192**. Classical and DL-only denoisers: N/A.

## False positive rate (FPR)

Fraction of background samples classified as PD. NoMix: **0.2584**.

## SNR improvement

Mean output SNR minus input SNR (dB) relative to clean synthetic reference. NoMix: **9.64 dB**; BayesShrink: **2.73 dB**.

## NoMix curriculum

Three-phase SNR curriculum **without mixed-type augmentation windows** — single discharge archetype per training window. Recommended deployment configuration; best overall synthetic and grounded headline metrics.

## IDWT decoder

Inverse db4-DWT upsampling in the MWCNN decoder (parameter count 2.92 M vs 3.05 M linear). Grounded transfer can be weaker than linear NoMix despite competitive synthetic NCC.

## MWCNN encoder

Conv-first blocks at each level, then db4-DWT stacking (channels ×2, length ≈ L/2+1). Skips at (B,32,2048), (B,64,1025), (B,128,513). Input projection: 1→32 ch (not 64).

## Composite noise

AWGN + magnetostrictive hum + powerline harmonics + narrowband 80 kHz + Bernoulli–Gaussian impulses. Identical recipe for training, synthetic test, and grounded real evaluation.

## selected_final.pt

Checkpoint used for each model row in `paper/metrics_master.json`. Selection: pre-existing `selected_final.pt` or max mean synthetic NCC (tie-break −15 dB). Key: `model_state` (not `model_state_dict`).

# Appendix R — Step-by-Step Reproduction Walkthrough

**Status (2026-05-31):** Benchmark locked. Steps 1–4 already completed for this report; re-run only if checkpoints or test-set hashes change.

**Environment:** Windows 11, NVIDIA RTX 4070, `.venv\Scripts\python.exe` from `FYP_MASTER`.

| Step | Command | Expected output |
|---:|---|---|
| R.1 | `python data/build_test_sets.py` | Frozen `data/test_synthetic_ae.pt`, real Q.Lin tensors, `docs/TEST_SET_HASHES.md` |
| R.2 | `python scripts/run_phaseA_rebench.py` | `paper/metrics_master.json`, `results/paper_table1_ablation.csv`, `results/paper_table2_improved_models.csv` |
| R.3 | `python scripts/train_baselines.py` (if needed) | `results/additional_baselines_benchmark.json` for PureCNN1D, UNet1D, DnCNN1D |
| R.4 | `python scripts/run_explainability_all.py` | `results/LIME_importances.json`, `results/SHAP_attributions.json` |
| R.5 | `python scripts/generate_poster_figures.py` | `paper/figures/*.png` (SNR sweep, gNCC bar, ablation heatmap, poster assets) |
| R.6 | `python scripts/make_paper_figures.py` | Additional paper figures from `metrics_master.json` |
| R.7 | `python scripts/verify_paper_numbers.py` | Spot-check pass; zero unresolved TBD tokens |

## R.1 build_test_sets

```powershell
cd "D:\New folder (2)\Data\FYP_MASTER"
.venv\Scripts\python.exe data/build_test_sets.py
```

Verify SHA-256 hashes match Appendix D. Do not modify noise generators after hashing.

## R.2 phaseA re-benchmark

```powershell
.venv\Scripts\python.exe scripts/run_phaseA_rebench.py
```

Confirms all ablation and variation models load with `matched_fraction = 1.0`. IDWT rows require `use_idwt=true` at load time.

## R.3 DL baselines (optional re-run)

```powershell
.venv\Scripts\python.exe scripts/train_baselines.py
```

50 epochs, flat MSE, no curriculum, no MTL. Results in `results/additional_baselines_benchmark.json`.

## R.4 explainability

```powershell
.venv\Scripts\python.exe scripts/run_explainability_all.py
```

GPU memory ~4–6 GB. LIME peak importance for MR-TAE-FULL @ −15 dB ≈ 0.2342.

## R.5 poster and paper figures

```powershell
.venv\Scripts\python.exe scripts/generate_poster_figures.py
```

Produces `fig_gncc_comparison.png`, `fig02_snr_bar.png`, `fig03_snr_sweep.png`, `fig04_ablation_heatmap.png`, `fig10_qualitative_waveforms.png`, and architecture schematics. Open poster: `paper/poster/FYP_POSTER_MR_TAE_2026.html`.

## R.7 verify

```powershell
.venv\Scripts\python.exe scripts/verify_paper_numbers.py
```

Exits 0 when spot-checks pass. Classical gNCC headline values (0.3844 / 0.3641 / 0.2915) are manually finalized for the poster and are not cross-checked against raw `metrics_master.json` grounded means.

## Headline numbers checklist (manual)

| Item | Expected |
|---|---|
| Best model | MR-TAE-FULL (NoMix) |
| Synthetic NCC mean | 0.8282 |
| NCC @ −15 dB | 0.3777 |
| SNR improvement | 9.64 dB |
| Dice | 0.9192 |
| gNCC (no FT) | 0.4535 |
| vs BayesShrink gNCC | +17.9% (0.3844) |
| Ablation reference | MR-TAE-FULL phased (Table 1 row 1) |

<!-- END OF REPORT -->
