# MR-TAE Architecture — Verified from Code Audit + PPTX Diagram (Final)
# Last updated: 2026-05-31  Source: verified PPTX diagram + audit doc (doc index 3)

## ENCODER — Conv-first at each level, then db4-DWT (DWT DOUBLES channels)

At each of 3 encoder levels:
  1. Conv block: 2×Conv1D + BatchNorm + GELU (maintains channel width set by DWT of previous level)
  2. Skip connection saved HERE — post-Conv, pre-DWT, full temporal resolution
  3. db4-DWT: stacks approx + detail sub-bands along channel axis → channels ×2, length L/2+1

Tensor shapes (correct, verified from diagram):
  Input:        (B, 1, 2048)
  Input Proj:   Conv7, 1→32ch                    (B, 32, 2048)
  Conv Block L1: 2×Conv1D+BN+GELU (32ch maint)  skip1 (B, 32, 2048)
  DWT L1:       32→64ch, 2048→1025              (B, 64, 1025)   ← DWT doubles 32→64
  Conv Block L2: 2×Conv1D+BN+GELU (64ch maint)  skip2 (B, 64, 1025)
  DWT L2:       64→128ch, 1025→513              (B, 128, 513)   ← DWT doubles 64→128
  Conv Block L3: 2×Conv1D+BN+GELU (128ch maint) skip3 (B, 128, 513)
  DWT L3:       128→256ch, 513→257              (B, 256, 257)   ← DWT doubles 128→256
  Bottleneck:   (B, 256, 257)

Skip channel summary (for AG wiring in paper):
  skip1: (B, 32, 2048)   ← pre-DWT L1, 32ch
  skip2: (B, 64, 1025)   ← pre-DWT L2, 64ch
  skip3: (B, 128, 513)   ← pre-DWT L3, 128ch

Why DWT doubles channels (MWCNN design, Liu 2018):
  db4-DWT of a C-channel 1D signal → [approx C-ch, detail C-ch] stacked = 2C channels, length L/2+1
  Conv blocks MAINTAIN the channel width set by the preceding DWT. They do not double.
  The progression 32→64→128→256 comes entirely from DWT stacking, not Conv blocks.
  Conv blocks at each level refine features at the width they receive.

## BOTTLENECK — Sequential BiGRU → Swin-1D

Pre-bottleneck: optional 1×1 projection (B, 256, 257)
BiGRU: 2 layers, 128 hidden/direction → bidirectional output (B, 256, 257)
Swin-1D: embed=256, depth=2, 8 heads, window=32, shifted-window attn shift=16 → (B, 256, 257)
Swin block detail: LN → WMSA → LN → MLP (row 1); LN → Shifted-WMSA → LN → MLP (row 2)
Residual: y = Proj(Swin(BiGRU(x))) + Residual(x)
Order: SEQUENTIAL — BiGRU output feeds INTO Swin (NOT parallel branches)

## DECODER — Linear Upsample (default) / IDWT Variant

AG runs BEFORE each decoder block (outside the upsample module):
  gated_skip = AG(encoder_skip, current)   # current = pre-upsampling main tensor

Decoder block structure per level (verified, doc index 3):
  [Upsample Module]:
    1. Upsample×2 (linear interpolation)
    2. ½Conv1D (e.g. 256→128) + BN + GELU
    3. Interpolate to skip length (e.g. →513)
  [Concat]: upsampled main + gated_skip → doubled channels
  [Conv Block]: 2×Conv1D + BN + GELU → halved channels out

Decoder shapes per level:
  L3: (B,256,257) Upsample→(B,128,513) Concat(B,128,513)→(B,256,513) Conv→(B,128,513)
  L2: (B,128,513) Upsample→(B,64,1025) Concat(B,64,1025)→(B,128,1025) Conv→(B,64,1025)
  L1: (B,64,1025) Upsample→(B,32,2048) Concat(B,32,2048)→(B,64,2048) Conv→(B,32,2048)

IDWT variant: replace Upsample Module with db4-IDWT + deterministic crop to skip length.
All other weights and structure identical. Parameter saving: 3.01M → 2.92M.

## ATTENTION GATE
formula: out = σ(ψ(ReLU(Wg·g + Wx·x))) × x
g = decoder gating signal (pre-upsampling, clean context from bottleneck)
x = encoder skip (noisy features, pre-DWT)
Output shape = x shape (channels unchanged by gate)
AG suppresses noise-dominated time steps in the skip before concat.

## OUTPUT HEADS
Signal head:       Conv1D 32→16→1, linear activation → (B, 1, 2048) denoised signal
Seg head:          Dual-stream fusion, 5-class logits → (B, 5, 2048)
  class 0: background noise
  class 1: corona
  class 2: surface discharge
  class 3: internal discharge
  class 4: treeing

## BEST MODELS (use NoMix as primary headline everywhere)

MR-TAE-FULL (NoMix curriculum) — PRIMARY RECOMMENDED MODEL:
  NCC synth mean: 0.8282  |  NCC @-15dB: 0.3777  |  SNR imp: 9.64 dB
  Dice: 0.9192  |  Params: 3.05M  |  Lat: 17.1ms

MR-TAE-FULL (phased curriculum) — ablation reference:
  NCC synth mean: 0.7991  |  NCC @-15dB: 0.3492  |  SNR imp: 7.01 dB
  Dice: 0.9015  |  gNCC: 0.4366  |  Params: 3.01M  |  Lat: 40.1ms

MR-TAE-FULL (NoMix) — PRIMARY (no fine-tuning for headline gNCC):
  gNCC: 0.4535  (BayesShrink: 0.3844, +17.9%)

Classical baselines (finalized gNCC for poster/paper):
  BayesShrink: gNCC 0.3844  |  NCC synth 0.6578  |  NCC @-15dB 0.1575  |  SNR 2.73dB
  Wiener:      gNCC 0.3641  |  NCC synth 0.5897  |  NCC @-15dB 0.1520  |  SNR 0.39dB
  Savitzky:    gNCC 0.2915  |  NCC synth 0.0351  |  NCC @-15dB 0.0098  |  SNR -1.66dB

Note: Q.Lin FT variants exist but are NOT used in the finalized poster/paper headline.

## SEGMENTATION CLASSES NOTE
Classes represent DISCHARGE TYPE (corona/surface/internal/treeing/background),
NOT gap size. The Q.Lin gap-size metadata is separate and not used as seg_target.
Old ARCHITECTURE_VERIFIED.md incorrectly listed gap-size classes — this is now corrected.
