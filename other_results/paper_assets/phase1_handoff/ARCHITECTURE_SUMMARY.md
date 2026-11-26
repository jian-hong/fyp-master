# Architecture Summary — MR-TAE (Multi-Resolution Temporal Autoencoder)

## 1. MR-TAE-FULL Architecture Diagram

```
 Input: (B, 1, 2048)  —  single-channel AE signal window
 ═══════════════════════════════════════════════════════════════════════

 ┌──────────────────────────────────────────────────────────────────────┐
 │                        MWCNN ENCODER                                │
 │                                                                     │
 │  (B,1,2048) ──► DWT ──► Conv1D Block ──► (B,64,1024)   Level 1     │
 │                            │                    │                   │
 │                            │              skip₁ ──────────┐         │
 │                            ▼                              │         │
 │  (B,64,1024) ─► DWT ──► Conv1D Block ──► (B,128,512)  Level 2     │
 │                            │                    │                   │
 │                            │              skip₂ ──────────┐         │
 │                            ▼                    │         │         │
 │  (B,128,512) ─► DWT ──► Conv1D Block ──► (B,256,256)  Level 3     │
 │                                                 │         │         │
 │                                           skip₃ ──────────┐         │
 └─────────────────────────────────────────────────┼─────────┼─────────┘
                                                   │         │
 ┌─────────────────────────────────────────────────▼─────────┼─────────┐
 │                  BOTTLENECK (BiGRU + Swin)                │         │
 │                                                           │         │
 │  (B,256,256) ──► 2-layer BiGRU ──► (B,256,256)           │         │
 │       ↓            hidden_size=128 per direction          │         │
 │  (B,256,256) ──► Swin Transformer Block ──► (B,256,256)  │         │
 │                    window_size=8, 4 heads                 │         │
 │                    shifted-window self-attention           │         │
 └───────────────────────────┬───────────────────────────────┼─────────┘
                             │                               │
 ┌───────────────────────────▼───────────────────────────────▼─────────┐
 │                     MWCNN DECODER                                   │
 │                                                                     │
 │  (B,256,256) ──► IDWT ──► Conv1D Block ──► (B,128,512)   Level 3  │
 │       ╋ Attn-Gate(skip₃) ─┘                                        │
 │                            │                                        │
 │  (B,128,512) ──► IDWT ──► Conv1D Block ──► (B,64,1024)   Level 2  │
 │       ╋ Attn-Gate(skip₂) ─┘                                        │
 │                            │                                        │
 │  (B,64,1024) ──► IDWT ──► Conv1D Block ──► (B,32,2048)   Level 1  │
 │       ╋ Attn-Gate(skip₁) ─┘                                        │
 └───────────────────────────┬─────────────────────────────────────────┘
                             │
 ┌───────────────────────────▼─────────────────────────────────────────┐
 │                     DUAL OUTPUT HEADS                               │
 │                                                                     │
 │  Signal Head ──► Conv1D(32→1) ──► (B, 1, 2048)   denoised signal  │
 │                                                                     │
 │  Segmentation Head ──► Conv1D(32→2) ──► (B, 2, 2048)  PD mask     │
 │                          (class 0: noise, class 1: PD event)        │
 └─────────────────────────────────────────────────────────────────────┘

 Key:  DWT  = Discrete Wavelet Transform (Haar) — downsamples 2×
       IDWT = Inverse DWT — upsamples 2×
       ╋    = Attention-gated skip connection (sigmoid gating)
```

### Data Flow Summary

| Stage         | Tensor Shape     | Description                               |
|---------------|------------------|-------------------------------------------|
| Input         | (B, 1, 2048)     | Raw noisy AE window, 1 channel            |
| Enc Level 1   | (B, 64, 1024)    | After DWT + conv block                    |
| Enc Level 2   | (B, 128, 512)    | After DWT + conv block                    |
| Enc Level 3   | (B, 256, 256)    | After DWT + conv block                    |
| Bottleneck    | (B, 256, 256)    | BiGRU temporal + Swin global attention    |
| Dec Level 3   | (B, 128, 512)    | IDWT + attn-gated skip + conv block       |
| Dec Level 2   | (B, 64, 1024)    | IDWT + attn-gated skip + conv block       |
| Dec Level 1   | (B, 32, 2048)    | IDWT + attn-gated skip + conv block       |
| Signal Out    | (B, 1, 2048)     | Denoised signal (regression head)         |
| Seg Out       | (B, 2, 2048)     | Per-sample PD event mask (2-class logits) |

---

## 2. Model Variants — Parameter Counts

All 13 variants are defined in `config/model_registry.yaml`. Parameter counts measured
from forward-pass smoke tests on RTX 4070 (input shape: `(1, 1, 2048)`).

### Phase 3a — Single-Component Ablation (Base 6)

| Variant            | Wavelet | BiGRU | Swin | Attn Gates | MTL | Loss | Params |
|--------------------|---------|-------|------|------------|-----|------|--------|
| **MR-TAE-FULL**    | ✓       | ✓     | ✓    | ✓          | ✓   | A    | 3.05 M |
| MR-TAE-noBiGRU     | ✓       | ✗     | ✓    | ✓          | ✓   | A    | 2.39 M |
| MR-TAE-noSwin      | ✓       | ✓     | ✗    | ✓          | ✓   | A    | 1.41 M |
| MR-TAE-noAttn      | ✓       | ✓     | ✓    | ✗          | ✓   | A    | 3.01 M |
| MR-TAE-noMTL       | ✓       | ✓     | ✓    | ✓          | ✗   | C    | 3.00 M |
| MR-TAE-noWavelet   | ✗       | ✓     | ✓    | ✓          | ✓   | A    | 2.99 M |

### Phase 3b — Two-Component Removal (Deeper Ablation)

| Variant                   | Wavelet | BiGRU | Swin | Attn Gates | MTL | Loss | Params |
|---------------------------|---------|-------|------|------------|-----|------|--------|
| MR-TAE-noSwin-noBiGRU     | ✓       | ✗     | ✗    | ✓          | ✓   | A    | 1.01 M |
| MR-TAE-noSwin-noAttn      | ✓       | ✓     | ✗    | ✗          | ✓   | A    | 1.36 M |
| MR-TAE-noBiGRU-noAttn     | ✓       | ✗     | ✓    | ✗          | ✓   | A    | 2.35 M |

### Loss Function Variants

| Variant              | Wavelet | BiGRU | Swin | Attn Gates | MTL | Loss | Params |
|----------------------|---------|-------|------|------------|-----|------|--------|
| MR-TAE-FULL-LossB    | ✓       | ✓     | ✓    | ✓          | ✓   | B    | 3.05 M |
| MR-TAE-FULL-LossC    | ✓       | ✓     | ✓    | ✓          | ✓   | C    | 3.05 M |

### Cross-Architecture Combinations

| Variant         | Wavelet | BiGRU | Swin | Attn Gates | MTL | Loss | Params |
|-----------------|---------|-------|------|------------|-----|------|--------|
| MWCNN-BiGRU      | ✓       | ✓     | ✗    | ✗          | ✗   | C    | 1.31 M |
| UNet-BiGRU-Swin  | ✗       | ✓     | ✓    | ✗          | ✗   | C    | 2.90 M |

---

## 3. Loss Configurations

Four loss configurations are defined to study the impact of loss function choice.
In MTL mode, weights are learned via homoscedastic uncertainty (Kendall et al., 2018).

| Config | Signal Loss                   | Segmentation Loss    | MTL Weighting           |
|--------|-------------------------------|----------------------|-------------------------|
| **A**  | Charbonnier + Focal           | Dice                 | Homoscedastic (learned) |
| **B**  | Charbonnier                   | Dice                 | Homoscedastic (learned) |
| **C**  | Charbonnier + SSIM            | —                    | Single-task (no MTL)    |
| **D**  | SmoothL1 + CrossEntropy       | Dice                 | Homoscedastic (learned) |

**Charbonnier loss** is preferred over MSE because it is less sensitive to outliers —
PD pulses are inherently spiky, so L2 loss would over-penalise large-amplitude PD events
and under-penalise the noise floor.

**Focal loss** on the segmentation head addresses the severe class imbalance between
PD-event samples (~2–5% of window) and background noise (~95–98%).

---

## 4. Ablation Methodology

### Design Philosophy

The ablation study follows a **systematic component-removal** strategy across three phases:

1. **Phase 3a (Single-component removal):** Remove exactly one component from MR-TAE-FULL
   at a time. This isolates the contribution of each module (BiGRU, Swin, attention gates,
   MTL, wavelet pooling) by measuring the performance drop when it is absent.

2. **Phase 3b (Two-component removal):** Remove pairs of components to test for
   **synergistic effects**. If removing A and B together causes a larger drop than the sum
   of removing A alone and B alone, the components interact positively.

3. **Loss and architecture variants:** Test alternative loss functions (B, C, D) on the
   full model, and cross-architecture baselines (MWCNN-BiGRU, UNet-BiGRU-Swin) to anchor
   the results against simpler architectures that share some components.

### Evaluation Protocol

- **Primary metric:** Normalised Cross-Correlation (NCC) on the synthetic test set
  (10,000 windows across 5 SNR levels: −15, −10, −5, 0, +5 dB).
- **Secondary metrics:** RMSE, SNR improvement (dB), Dice coefficient (segmentation),
  false positive rate.
- **Real-world validation:** NCC on the Q. Lin real AE dataset (no clean ground truth;
  NCC is computed against the input window as reference).
- **Efficiency:** Parameter count, inference latency (batch=1, CUDA-synchronised).

### Statistical Rigour

- Test sets are frozen with SHA256 verification (see `docs/TEST_SET_HASHES.md`).
- All models share identical training configuration (LR, curriculum, data seed).
- SNR-stratified analysis prevents high-SNR windows from masking poor low-SNR performance.

### Ablation Grid (Conceptual)

```
                    MWCNN Encoder       Plain Encoder
                 ┌─────────────────┬──────────────────┐
 BiGRU + Swin    │  MR-TAE-FULL    │  UNet-BiGRU-Swin │
                 ├─────────────────┼──────────────────┤
 BiGRU only      │  MR-TAE-noSwin  │  MWCNN-BiGRU*    │
                 ├─────────────────┼──────────────────┤
 Swin only       │  MR-TAE-noBiGRU │       —          │
                 ├─────────────────┼──────────────────┤
 Conv only       │  noSwin-noBiGRU │       —          │
                 └─────────────────┴──────────────────┘

 *MWCNN-BiGRU uses wavelet encoder but has no Swin or attention gates.
```

---

## 5. Curriculum Learning Schedule

Training follows a 3-phase curriculum that progressively increases noise difficulty:

```
 Epoch   1 ──────── 35 ──────── 70 ──────── 100
         │  Phase 1  │  Phase 2  │  Phase 3  │
         │ Easy SNR  │ Medium    │ Hard SNR  │
         │ +5→+15 dB │ -5→+5 dB │ -20→-5 dB │
         │   WGN     │ WGN+Imp  │ Composite │
```

**Rationale:** Starting with high-SNR signals lets the model first learn the structure of
clean PD pulses. Gradually introducing harder noise types and lower SNRs prevents early
training collapse that occurs when the model is immediately exposed to −15 dB composite
noise.

---

## 6. Training Infrastructure

| Component               | Implementation                                           |
|-------------------------|----------------------------------------------------------|
| Mixed Precision          | `torch.cuda.amp` (FP16 forward, FP32 master weights)    |
| OOM Recovery             | Auto-halves batch size on CUDA OOM, retries              |
| Thermal Gating           | `HeatMonitor` pauses training if GPU > threshold         |
| Crash Recovery           | `session_state.json` updated atomically every epoch      |
| Early Stopping           | Dynamic threshold based on rolling NCC improvement rate  |
| Experiment Tracking      | MLflow (local tracking store in `mlruns/`)               |
| Hyperparameter Search    | Optuna with TPE sampler + Hyperband pruner               |
| Gradient Clipping        | Max-norm = 1.0 (prevents exploding gradients in BiGRU)   |
