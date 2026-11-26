# Examiner Q&A — Viva Defence Preparation

15 anticipated questions with thesis-defence-quality answers.

---

## Q1. Explain the physics of Acoustic Emission (AE) sensing for partial discharge detection. Why did you choose 2 MHz sampling and 2048-sample windows?

**Answer.**
Partial discharges (PDs) are localised dielectric breakdowns within high-voltage insulation that emit broadband acoustic transients in the 20–500 kHz range. AE sensors (typically piezoelectric transducers coupled to the transformer tank wall) convert these pressure waves into voltage signals. The propagation medium — transformer oil and solid insulation — attenuates high-frequency components, so the useful AE bandwidth is roughly 50–400 kHz.

**Sampling rate (2 MHz):** By the Nyquist theorem, a 2 MHz sampling rate captures frequencies up to 1 MHz, which comfortably exceeds the 400 kHz upper bound of PD-generated AE signals. This ensures no aliasing of PD content while providing temporal resolution of 0.5 µs per sample — sufficient to resolve individual PD pulse rise times (typically 1–5 µs).

**Window length (2048 samples):** At 2 MHz, a 2048-sample window spans 1.024 ms. This is long enough to capture an entire PD pulse (typical duration 5–50 µs) plus sufficient surrounding context for the denoiser to distinguish pulse morphology from background noise. It also yields a power-of-two length, which is computationally efficient for the DWT operations in the MWCNN encoder (three levels of 2× downsampling: 2048 → 1024 → 512 → 256). Additionally, the Q. Lin dataset was recorded with 1 M samples per trace; segmenting into 2048-sample windows provides ~488 non-overlapping windows per trace, giving adequate training volume without overlap-induced data leakage.

---

## Q2. Why did you choose MWCNN (Multi-level Wavelet CNN) over standard pooling operations like max-pooling or strided convolutions?

**Answer.**
Standard pooling operations (max-pool, average-pool, strided convolutions) are inherently lossy — they discard spatial/temporal information during downsampling. For PD denoising, this is particularly harmful because PD pulses are short-duration, high-frequency transients. A max-pool that discards 50% of samples can obliterate a PD pulse that occupies only 10–20 samples within a 2048-sample window.

**MWCNN replaces pooling with the Discrete Wavelet Transform (DWT).** The DWT decomposes the signal into approximation (low-frequency) and detail (high-frequency) sub-bands. Both sub-bands are retained and processed by subsequent convolution layers. This means high-frequency PD edges are preserved through the encoder, rather than being thrown away by pooling.

Concretely, at each encoder level:
1. The Haar DWT splits the signal into low-pass and high-pass components, each at half resolution.
2. Both components are concatenated along the channel dimension and passed through Conv1D blocks.
3. The decoder uses the Inverse DWT (IDWT) to reconstruct full resolution, re-injecting the high-frequency detail.

This design is directly motivated by the signal characteristics: PD pulses appear as sharp edges (high-frequency content) riding on a low-frequency noise floor. The MWCNN encoder preserves exactly the frequency band that carries the PD information.

Empirically, removing wavelet pooling (MR-TAE-noWavelet, 2.99 M params) and replacing it with strided convolutions degrades NCC performance, confirming that the DWT's frequency-aware downsampling is beneficial.

---

## Q3. Why does the bottleneck combine BiGRU and Swin Transformer? What does each contribute?

**Answer.**
The bottleneck operates on the most compressed representation (B, 256, 256) — 256 time steps at 256 channels. At this scale, two distinct types of context are valuable:

**BiGRU (Bidirectional Gated Recurrent Unit)** captures **local temporal dependencies.** PD pulses have characteristic temporal signatures — a fast rise followed by exponential decay. The BiGRU processes the sequence in both forward and backward directions, learning these temporal patterns and their context (e.g., a PD pulse preceded by a quiet region is more likely genuine than one preceded by impulsive interference). The recurrent architecture naturally handles variable-length temporal correlations without fixed receptive fields.

**Swin Transformer** provides **global context via self-attention.** While the BiGRU excels at local sequential patterns, it struggles to model long-range dependencies efficiently (information must propagate step-by-step through the hidden state). The Swin Transformer's shifted-window attention mechanism enables direct interaction between distant time steps. This is valuable because noise characteristics (e.g., background hum frequency, impulsive interference patterns) are often stationary properties of the entire window, not just local neighbourhoods.

**Why both?** Single-component ablations confirm their complementary roles:
- MR-TAE-noBiGRU (2.39 M): loses temporal modelling → NCC drops
- MR-TAE-noSwin (1.41 M): loses global attention → NCC drops further (also fewer parameters since Swin block is parameter-heavy)
- MR-TAE-noSwin-noBiGRU (1.01 M): minimal bottleneck → largest performance drop

The sequential arrangement (BiGRU first, then Swin) lets the recurrent layer establish local temporal context, which the Swin block then refines with global attention.

---

## Q4. Why did you use curriculum learning? How does the 3-phase SNR schedule work?

**Answer.**
**Motivation:** Direct training on the full SNR range (−20 to +15 dB) causes training instability in early epochs. At −15 dB, the signal-to-noise ratio is so low that PD pulses are almost invisible — the model receives near-random gradients and struggles to converge. Curriculum learning addresses this by presenting easier examples first, allowing the model to build useful representations before encountering harder cases.

**3-Phase Schedule:**

| Phase | Epochs | SNR Range      | Noise Type     | Purpose                                    |
|-------|--------|----------------|----------------|--------------------------------------------|
| 1     | 35     | +5 to +15 dB   | White Gaussian | Learn PD pulse structure in easy conditions|
| 2     | 35     | −5 to +5 dB    | WGN+Impulsive  | Handle moderate noise + impulsive artefacts|
| 3     | 30     | −20 to −5 dB   | Composite      | Master extreme noise (real-world regime)   |

Phase 1 uses clean white Gaussian noise only, which is the simplest noise model. The model learns the basic shape of PD pulses and how to separate them from broadband noise. Phase 2 introduces impulsive noise (mimicking electromagnetic interference from switching operations), which is harder because impulsive noise pulses can resemble PD events. Phase 3 uses composite noise (WGN + impulsive + coloured noise), which most closely matches real transformer environments, at the most challenging SNR levels.

**Implementation:** The `CurricularScheduler` class maps each absolute epoch number to a (phase, SNR range, noise type) tuple. The data loader samples fresh synthetic signals each epoch with noise parameters from the current phase's distribution.

---

## Q5. How does homoscedastic uncertainty weighting work for multi-task learning, and why did you choose it?

**Answer.**
The MR-TAE-FULL model has two output heads: signal reconstruction (regression) and PD-event segmentation (classification). Naively summing their losses (e.g., `L = L_signal + λ·L_seg`) requires manually tuning λ, which is fragile because the optimal balance changes during training (early on, both losses are large; later, one may plateau while the other continues improving).

**Homoscedastic uncertainty weighting** (Kendall et al., CVPR 2018) learns the relative task weights automatically. For each task *t*, a learnable scalar σ_t (log-variance) is introduced. The combined loss becomes:

```
L_total = (1 / 2σ₁²) · L_signal + (1 / 2σ₂²) · L_seg + log(σ₁) + log(σ₂)
```

The log(σ) regularisation terms prevent the model from trivially setting σ → ∞ to ignore a task. When a task's loss is high, the model learns a larger σ for that task, effectively down-weighting it so the other task can still learn. As training progresses and losses decrease, the σ values adapt accordingly.

**Why this approach?**
1. **No manual λ tuning** — the model discovers the optimal balance.
2. **Principled** — derived from maximising the Gaussian likelihood of a multi-task model.
3. **Dynamic** — the weighting adapts across curriculum phases (in Phase 1, the signal loss dominates; in Phase 3, both losses are important).

The σ parameters are included in the optimiser's parameter list alongside model weights, and they are saved/restored in checkpoints for seamless crash recovery.

---

## Q6. How do you bridge the gap between synthetic training data and real AE signals?

**Answer.**
This is the central challenge of the project. The model is trained entirely on synthetic data because real AE datasets lack paired clean/noisy ground truth — you cannot record a "clean" PD signal in a real transformer. Our strategy has three components:

**1. Physics-informed synthetic generation.** The DOP (Damped Oscillatory Pulse) model generates PD pulses with physically realistic parameters:
- Centre frequency: 100–350 kHz (matching AE sensor bandwidth)
- Damping ratio: calibrated to match observed pulse decay rates in Q. Lin data
- Amplitude distribution: follows empirical PD charge distributions
- Noise model: composite (WGN + impulsive + 50 Hz hum + coloured noise) that simulates real transformer environments

**2. WGAN-GP augmentation.** A Wasserstein GAN with gradient penalty is trained on windows from the Q. Lin real dataset to generate realistic synthetic windows that capture distribution characteristics not modelled by the DOP formula (e.g., multi-path reflections in transformer oil, sensor resonance effects). These augmented windows are mixed into the training set to regularise the model toward real-world distributions.

**3. Curriculum-driven domain adaptation.** Phase 3 uses composite noise at −20 to −5 dB, which closely matches the SNR conditions observed in real transformer AE recordings. By the time the model reaches Phase 3, it has already learned clean PD structure (Phase 1) and is fine-tuning for the hardest, most realistic conditions.

**Evaluation protocol.** We evaluate on both synthetic test data (where NCC against clean ground truth is computed) and the Q. Lin real dataset (where NCC is computed against the input window, measuring how much signal structure the model preserves). This dual evaluation quantifies both denoising quality and real-world transfer.

---

## Q7. What is the rationale for WGAN-GP augmentation, and how does it improve model robustness?

**Answer.**
The Wasserstein GAN with Gradient Penalty (WGAN-GP, Gulrajani et al., 2017) is used to generate synthetic AE signal windows that mimic the distribution of real Q. Lin data. The motivation is twofold:

**1. Distribution matching.** The DOP pulse model, while physically grounded, cannot capture all characteristics of real AE signals — multi-path reflections off transformer internals, sensor resonance artefacts, non-stationary background noise from cooling systems, and partial-discharge patterns that involve multiple simultaneous discharge sites. The WGAN-GP learns these implicit distributional features directly from real data.

**2. Training regularisation.** Mixing WGAN-generated windows into the training set acts as a form of domain randomisation. The model sees a wider variety of signal morphologies than the parametric DOP model alone can produce, which discourages overfitting to the synthetic distribution and improves generalisation to real signals.

**Why WGAN-GP specifically?**
- **Wasserstein distance** provides meaningful gradients everywhere (unlike vanilla GAN which suffers from mode collapse and training instability on 1D signal data).
- **Gradient penalty** eliminates the need for weight clipping, which can cause capacity underuse.
- The 1D convolutional generator architecture mirrors the signal's temporal structure.

The augmentation ratio is tuned to avoid overwhelming the training set with WGAN samples (which may not perfectly match ground truth). Typically 10–20% of each training batch consists of WGAN-augmented windows.

---

## Q8. Describe the Q. Lin dataset. What are its characteristics and limitations?

**Answer.**
The Q. Lin dataset is an experimental AE dataset collected from XLPE (cross-linked polyethylene) cable insulation samples under controlled partial discharge conditions.

**Dataset characteristics:**
- **4 cable samples** with different artificial defect gap distances: 12 mm, 15 mm, 18 mm, 21 mm
- **200 AE traces per sample** (800 traces total)
- **1 million samples per trace** at 2 MHz sampling rate (0.5 s recording duration)
- Recorded using a piezoelectric AE sensor coupled to the cable surface
- PD was induced by applying high voltage near the breakdown threshold

**Window extraction:** Each 1 M-sample trace is segmented into non-overlapping 2048-sample windows, yielding ~488 windows per trace and ~97,600 windows per gap distance.

**Limitations:**
1. **No clean ground truth.** Real AE recordings always contain both PD signals and noise. There is no way to obtain a "clean" version, which is why we train on synthetic data and use the real dataset only for evaluation.
2. **Single noise environment.** All recordings are from a laboratory setup, not an operating transformer. Real transformers have additional noise sources (load noise, cooling fans, electromagnetic interference from the grid).
3. **Single insulation type.** XLPE cable insulation may not generalise to oil-paper insulation used in power transformers.
4. **Limited discharge types.** The artificial defects produce surface discharges; internal voids and treeing-type PDs are not represented.

Despite these limitations, the Q. Lin dataset is one of the few publicly available AE-PD datasets with known defect geometries, making it valuable for benchmarking.

---

## Q9. What strategies do you employ to mitigate overfitting?

**Answer.**
Several complementary strategies are used:

**1. Curriculum learning.** By presenting different SNR ranges and noise types in each phase, the model effectively sees a different data distribution every ~35 epochs. This acts as implicit data augmentation and prevents the model from memorising specific noise patterns.

**2. Fresh synthetic generation per epoch.** The training dataset is generated on-the-fly with randomised PD parameters (amplitude, frequency, position) and noise realisations. The model never sees the same training example twice, eliminating the possibility of sample-level memorisation.

**3. WGAN augmentation.** Mixing real-distribution samples into training regularises the model toward realistic signal characteristics.

**4. Weight decay (AdamW, λ=1e-4).** L2 regularisation on all model parameters discourages large weights.

**5. Gradient clipping (max-norm=1.0).** Prevents exploding gradients in the BiGRU, which is particularly important in Phase 3 where extreme noise can cause large gradient spikes.

**6. Cosine annealing LR schedule.** Gradually reduces the learning rate from 3e-4 to 1e-6 over 100 epochs, preventing late-training oscillations that can push the model into overfitting territory.

**7. Overfit/underfit monitoring.** The dashboard and `ResilienceManager` track the train-val loss gap in real time. If val_loss − train_loss exceeds 0.3 with an NCC gap > 0.15, the model is flagged as overfitting.

**8. Dynamic early stopping.** The `ResilienceManager` computes a rolling improvement rate and stops training when NCC improvement falls below a dynamic threshold for 8 consecutive epochs. This prevents over-training when the model has already converged.

---

## Q10. Why is NCC (Normalised Cross-Correlation) the primary metric rather than MSE or PSNR?

**Answer.**
**MSE (Mean Squared Error)** is sensitive to absolute amplitude scale. If the denoiser outputs a signal that is perfectly shaped but scaled by a factor of 0.9, MSE penalises this heavily even though the denoising quality is excellent. In PD analysis, the waveform shape (morphology) matters far more than absolute amplitude, because downstream PD classification algorithms use shape features (rise time, oscillation frequency, damping rate) rather than raw amplitude.

**PSNR** is derived from MSE (`PSNR = 10·log10(max² / MSE)`) and inherits the same amplitude-sensitivity problem. Additionally, PSNR requires knowledge of the signal's dynamic range, which varies across windows.

**NCC** measures the normalised correlation between predicted and target signals:

```
NCC = Σ(p̃ · t̃) / √(Σp̃² · Σt̃²)
```

where p̃ = p − mean(p) and t̃ = t − mean(t). This metric is:
1. **Scale-invariant** — a constant scaling factor does not affect NCC.
2. **Shift-invariant** — a DC offset does not affect NCC (both signals are zero-meaned).
3. **Bounded** — NCC ∈ [−1, 1], with 1.0 being perfect correlation.
4. **Morphology-focused** — directly measures waveform shape similarity, which is what matters for PD analysis.

For the real Q. Lin dataset (where there is no clean ground truth), NCC against the input window measures how much of the original signal structure the denoiser preserves — a denoiser that destroys PD morphology would yield low NCC even if it reduces noise.

---

## Q11. How does the resilience system work? How does the pipeline recover from crashes?

**Answer.**
The `ResilienceManager` (in `training/resilience.py`) implements a crash-proof training pipeline with four mechanisms:

**1. Atomic session persistence.** After every epoch, the full training state is written to `session_state.json` using an atomic write pattern: data is written to a `.tmp` file first, then renamed to overwrite the original. This ensures that even if the process is killed mid-write, the session file is never corrupted (either the old version or the new version exists, never a partial write).

The session file records: current model ID, epoch number, curriculum phase, best NCC, best checkpoint path, per-epoch metrics history, the full training queue, and which models are done/skipped/running.

**2. Checkpoint-based resume.** Every time the model achieves a new best validation NCC, a checkpoint (`best.pt`) is saved containing the full model state, optimiser state, LR scheduler state, loss function state (including learned uncertainty σ values), AMP scaler state, and training history. On resume, the orchestrator reads `session_state.json`, identifies the last running model, loads its checkpoint, and continues from the exact epoch where training stopped.

**3. Auto-generated resume prompt.** After every epoch, a human-readable `RESUME_PROMPT.md` is written to the project root. This file contains natural-language instructions that can be pasted into a new Cursor/Claude session to continue the experiment exactly where it left off — including which models are done, which is currently running, and what checkpoint to load.

**4. Dynamic early stopping.** The resilience manager monitors the rolling NCC improvement rate. After a minimum of 15 epochs (cold-start protection), if no meaningful improvement occurs for 8 consecutive epochs (based on a dynamic threshold calibrated to the model's own improvement history), the model is marked as "skipped" and the orchestrator moves to the next model in the queue.

**Recovery scenarios:**
- **CUDA OOM:** The trainer halves batch size and retries the current batch.
- **GPU overheat:** The `HeatMonitor` thread pauses training until temperature drops.
- **Process kill / power loss:** On restart, `session_state.json` is intact; the orchestrator resumes the running model from its last checkpoint.
- **Network drop (e.g., remote session):** Same as process kill — state is already persisted locally.

---

## Q12. Explain your ablation study design. How do you ensure the results are meaningful?

**Answer.**
The ablation study is designed as a **controlled factorial experiment** with the following principles:

**1. Controlled variables.** All 13 model variants share:
- Identical training configuration (LR=3e-4, weight decay=1e-4, 100 epochs, same curriculum)
- Identical training data (same seed=42 for synthetic generation)
- Identical test sets (frozen with SHA256 verification)
- Same hardware (RTX 4070, CUDA, AMP enabled)
- Same evaluation pipeline (`benchmark_runner.py`)

**2. Single-variable isolation (Phase 3a).** The first 6 variants (Base 6) each remove exactly one component from MR-TAE-FULL. The performance difference between MR-TAE-FULL and MR-TAE-noX quantifies the contribution of component X, with all other components held constant.

**3. Interaction detection (Phase 3b).** The 3 two-component removal variants test whether components interact synergistically. For example, if:
- Removing BiGRU drops NCC by δ₁
- Removing Swin drops NCC by δ₂
- Removing both drops NCC by δ₃

Then if δ₃ > δ₁ + δ₂, the components have a positive synergy (they benefit from each other's presence). If δ₃ < δ₁ + δ₂, there is redundancy.

**4. Loss function isolation.** MR-TAE-FULL-LossB and MR-TAE-FULL-LossC use the same architecture as MR-TAE-FULL but different loss configurations. This isolates the effect of loss function choice from architectural choices.

**5. Baseline anchoring.** Two classical baselines (Wavelet-BayesShrink and Savitzky-Golay) provide non-learned reference points. Two cross-architecture variants (MWCNN-BiGRU, UNet-BiGRU-Swin) test whether the components generalise across different backbone designs.

**6. Multi-metric evaluation.** Results are reported across NCC (primary), RMSE, SNR improvement, Dice, and FPR — preventing a model from appearing good on one metric while failing on others.

**7. SNR-stratified analysis.** Per-SNR NCC values (at −15, −10, −5, 0, +5 dB) reveal whether a model excels at easy conditions but fails at hard ones, or maintains consistent performance across the range.

---

## Q13. What is the computational cost of training on the RTX 4070? How do you manage VRAM constraints?

**Answer.**
The RTX 4070 has 12 GB VRAM and ~29 TFLOPS FP16 throughput. Key computational characteristics:

**Training time (per model, 100 epochs):**
- MR-TAE-FULL (3.05 M params): approximately 2–3 hours with AMP enabled
- Lighter variants (1.01–1.41 M): approximately 1–1.5 hours
- Full 13-model ablation: approximately 25–35 hours total

**VRAM management** is handled by the `RTX4070Manager`:
1. **Automatic batch size selection.** Based on parameter count: models > 2.5 M get batch_size=32; models 1–2.5 M get 48; models < 1 M get 64. These were empirically determined to leave ~2 GB headroom.
2. **AMP (Automatic Mixed Precision).** All forward passes use FP16, reducing VRAM by ~40% and increasing throughput by ~60% compared to FP32.
3. **Gradient checkpointing.** Automatically enabled for models exceeding 40 M params (not triggered for our models, which peak at 3.05 M, but available for future scaling).
4. **OOM recovery.** If a CUDA OOM occurs despite batch size selection, the batch size is halved and the batch is retried. The model and optimiser states are preserved.
5. **Inter-model cleanup.** Between training runs, `cleanup_between_models()` calls `torch.cuda.empty_cache()` and forces garbage collection to prevent VRAM fragmentation.

**Thermal management:**
The `HeatMonitor` runs a background thread that polls GPU temperature via NVML. If the temperature exceeds the threshold (default 83°C), training is paused until it cools down. This prevents thermal throttling, which would otherwise cause unpredictable slowdowns.

**Inference latency** (measured with CUDA synchronisation):
- MR-TAE-FULL, batch=1: ~2–4 ms per window
- MR-TAE-FULL, batch=32: ~0.5–1 ms per window (amortised)
- Classical wavelet baseline: ~0.1 ms per window (CPU)

---

## Q14. What are the key directions for future work?

**Answer.**

**1. Noise2Void / self-supervised denoising.** The biggest limitation is the reliance on synthetic training data. Noise2Void (Krull et al., 2019) trains denoisers using only noisy images by masking individual pixels and predicting them from their neighbours. Adapting this to 1D AE signals would enable training directly on real transformer data without clean ground truth, potentially closing the synthetic-real domain gap.

**2. Online drift detection.** Real transformer noise environments change over time (seasonal temperature variations, load changes, ageing). Integrating a drift detector (e.g., monitoring the model's output NCC over a sliding window) would trigger automated retraining when the noise distribution shifts significantly from the training distribution.

**3. Edge deployment.** The MR-TAE-FULL model at 3.05 M parameters and <4 ms latency is already lightweight enough for embedded inference. Future work would target deployment on edge devices (NVIDIA Jetson, FPGA) for real-time PD monitoring without cloud connectivity. Model pruning and INT8 quantisation could reduce the footprint further.

**4. Multi-sensor fusion.** Power transformers typically have 3–6 AE sensors at different locations on the tank. Fusing signals from multiple sensors could improve PD localisation accuracy and enable rejection of external noise sources (noise arrives at all sensors simultaneously; PD signals arrive with time-of-flight delays).

**5. Expanded discharge types.** The current synthetic generator models surface discharges (DOP model). Adding models for void discharges, treeing, and floating-electrode discharges would broaden the applicability to diverse fault types.

**6. Transfer learning to other HV assets.** PD occurs in cables, switchgear, and GIS (gas-insulated switchgear) with different noise profiles. Fine-tuning the pre-trained MR-TAE on small labelled datasets from these assets could extend the pipeline's utility across the power grid.

---

## Q15. What is the broader impact of PD denoising for power transformer monitoring?

**Answer.**
Power transformers are the most expensive and critical assets in electrical transmission networks. A single large power transformer costs USD 2–10 million and has a lead time of 12–18 months for replacement. Unplanned failure causes extended outages affecting millions of consumers, with total economic impact often exceeding the transformer's cost by an order of magnitude.

**Partial discharge is the earliest detectable precursor to insulation failure.** PD activity begins years before catastrophic breakdown, making it the gold standard for condition-based monitoring. However, in field conditions, PD signals are buried in environmental noise (EMI from the grid, mechanical vibrations, corona from adjacent equipment) at SNRs as low as −15 dB. Human experts can sometimes identify PD patterns in noisy recordings, but this requires specialised training and is not scalable to the thousands of transformers in a national grid.

**Automated PD denoising enables:**
1. **Early fault detection.** By recovering PD pulses from noisy recordings, the system can detect incipient insulation degradation months or years before failure, enabling planned maintenance during scheduled outages rather than emergency repair.
2. **Continuous online monitoring.** With sub-4 ms inference latency, the model can process AE data in real time from permanently installed sensors, providing 24/7 monitoring without human intervention.
3. **Improved PD classification.** Downstream PD classification algorithms (which determine the type and severity of the fault) rely on clean pulse morphology. Denoising improves the accuracy of these classifiers by providing cleaner input features.
4. **Reduced false alarms.** The segmentation head explicitly identifies PD events, reducing the false positive rate that plagues threshold-based detection methods in noisy environments.
5. **Data-driven asset management.** Clean PD trend data, accumulated over months, enables utilities to prioritise maintenance spending on the transformers with the fastest-degrading insulation, optimising capital expenditure across the fleet.

In the context of global energy infrastructure, even a modest improvement in transformer reliability — say, preventing one unexpected failure per year across a utility's fleet — translates to millions of dollars in avoided replacement costs, avoided outage penalties, and improved grid reliability. The MR-TAE pipeline contributes to this goal by making AE-based PD monitoring viable in noisy field conditions where it was previously impractical.
