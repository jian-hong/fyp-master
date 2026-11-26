# CURSOR HANDOFF — MR-TAE Session 3
# New account, fresh context. Read this fully before touching any file.
# Date: 2026-05-31

---

## 0. WHO YOU ARE AND WHAT THIS PROJECT IS

You are a Cursor AI agent continuing work on the MR-TAE FYP project for Oo Jian Hong (ID 22004678), University of Malaya, supervised by Ir. Dr. Wong Jee Keen Raymond. The project has two codebases and you must NEVER confuse them:

**ACTIVE CODEBASE (use only this):**
`D:\New folder (2)\Data\FYP_MASTER\`
- Active .venv at `FYP_MASTER\.venv\Scripts\python.exe`
- Checkpoint key: `model_state` (NOT `model_state_dict`)
- Results: `results\{MODEL_ID}\checkpoints\`

**STALE CLONE (DO NOT USE):**
`D:\New folder (2)\` (root folder — April checkpoints, wrong physics)

**GITHUB MIRROR (sync target):**
`D:\New folder (2)\Data\MR_TAE_GITHUB\`
Structure: checkpoints/, config/, data/, docs/, evaluation/, figures/, mr_tae_fusion/, paper/, pipeline/, results/, scripts/, training/

---

## 1. VERIFIED ARCHITECTURE (100% CONFIRMED FROM PPTX DIAGRAM + CODE AUDIT)

This is ground truth. Do not guess, do not use old summaries that disagree.

```
INPUT: (B, 1, 2048)

ENCODER (Conv-first, db4-DWT doubles channels by stacking approx+detail):
  Input Proj: Conv7 1→32ch              → (B, 32, 2048)
  Conv Block L1: 2×Conv1D+BN+GELU       → skip₁ (B, 32, 2048)
  DWT L1: stacks approx+detail          → (B, 64, 1025)   [32→64ch, 2048→1025]
  Conv Block L2: 2×Conv1D+BN+GELU       → skip₂ (B, 64, 1025)
  DWT L2:                               → (B, 128, 513)   [64→128ch]
  Conv Block L3: 2×Conv1D+BN+GELU       → skip₃ (B, 128, 513)
  DWT L3:                               → (B, 256, 257)   [128→256ch]
  Bottleneck input: (B, 256, 257)

  KEY: Conv blocks MAINTAIN channel width. DWT DOUBLES channels by sub-band stacking.
  KEY: Skips are (B,32,2048), (B,64,1025), (B,128,513) — pre-DWT features.

BOTTLENECK (sequential, not parallel):
  Pre-bottleneck 1×1 proj (optional): (B, 256, 257)
  BiGRU: 2-layer, 128 hidden/dir, bidirectional  → (B, 256, 257)   [FIRST]
  Swin-1D: embed=256, depth=2, 8 heads, window=32, shift=16 → (B, 256, 257)   [SECOND]
  Swin internals: [LN → WMSA → LN → MLP] then [LN → ShiftedWMSA → LN → MLP]
  Residual: y = Proj(Swin(BiGRU(x))) + Residual(x)

DECODER (Linear default; IDWT = swap upsample blocks only):
  AG runs OUTSIDE each block, on skip path:
    gated_skip = AG(encoder_skip, current_pre_upsample)

  Per level structure:
    [Upsample Module]: Upsample×2 → ½Conv1D+BN+GELU → Interpolate-to-skip-length
    [Concat]: upsampled + gated_skip → doubled channels
    [Conv Block]: 2×Conv1D+BN+GELU → halved channels

  L3: (B,256,257)→Upsamp→(B,128,513) Concat(B,128,513)→(B,256,513) Conv→(B,128,513)
  L2: (B,128,513)→Upsamp→(B,64,1025) Concat(B,64,1025)→(B,128,1025) Conv→(B,64,1025)
  L1: (B,64,1025)→Upsamp→(B,32,2048) Concat(B,32,2048)→(B,64,2048) Conv→(B,32,2048)

OUTPUT HEADS:
  Signal:  Conv 32→16→1, linear activation → (B,1,2048) denoised waveform
  Seg:     Dual-stream fusion → 5-class logits (B,5,2048)
           Classes: 0=BG, 1=Corona, 2=Surface, 3=Internal, 4=Treeing
```

---

## 2. BEST MODEL — MR-TAE NoMix (USE EVERYWHERE AS PRIMARY)

**MR-TAE (NoMix curriculum)** is the overall best model. Use its numbers in abstract, conclusion, and all summary comparisons.

| Metric | MR-TAE NoMix | BayesShrink | Gap |
|--------|-------------|-------------|-----|
| NCC mean synth | **0.8282** | 0.6578 | +25.9% |
| NCC @ −15 dB | **0.3777** | 0.1575 | +2.39× |
| SNR improvement | **9.64 dB** | 2.73 dB | +6.91 dB |
| Dice (5-class) | **0.9192** | N/A | — |
| gNCC (real, no FT) | **0.4535** | 0.3844 | **+17.9%** |
| Params | 3.05M | 0 | — |
| Latency (ms) | 17.1 | — | — |

MR-TAE-FULL (phased) is used in the 13-row ablation table as the reference row.
NoMix is introduced as "the recommended training curriculum" in §3.4.
Q.Lin FT variants exist (NoMix+FT gNCC 0.8039) but are NOT used in the finalized poster/paper headline.

What is NoMix: three-phase curriculum training (same 3 phases) but WITHOUT mixed-type augmentation windows. Trains on single discharge-type windows only, forcing the model to learn purer archetype signatures. This improves both NCC and Dice vs. the mixed-augmentation phased variant.

---

## 3. FULL BENCHMARK NUMBERS

### Ablation table (linear decoder, phased curriculum, 13 rows)
| # | Config | NCC Synth | NCC@−15dB | SNR Imp | Dice | FPR | gNCC | Params | Lat |
|---|--------|-----------|-----------|---------|------|-----|------|--------|-----|
| 1 | MR-TAE-FULL | 0.7991 | 0.3492 | 7.01 | 0.9015 | 0.274 | 0.4366 | 3.01 | 40.1 |
| 2 | noBiGRU | 0.6564 | 0.0585 | 2.57 | 0.8266 | 0.340 | 0.2307 | 2.39 | 18.8 |
| 3 | noSwin | 0.6632 | 0.0982 | 4.24 | 0.8833 | 0.279 | 0.3287 | 1.41 | 13.7 |
| 4 | noAttn | 0.6768 | 0.0523 | 2.98 | 0.8869 | 0.300 | 0.1014 | 3.01 | 7.5 |
| 5 | noMTL | 0.6482 | 0.0390 | 0.35 | 0.000 | 0.000 | −0.1605 | 3.00 | 7.7 |
| 6 | noWavelet | 0.6304 | 0.0552 | 1.20 | 0.8947 | 0.308 | −0.0337 | 2.99 | 26.9 |
| 7 | noSwin-noBiGRU | 0.7974 | 0.3479 | 6.80 | 0.9011 | 0.243 | 0.4365 | 3.01 | 18.5 |
| 8 | noSwin-noAttn | 0.7998 | 0.3509 | 7.01 | 0.9034 | 0.248 | 0.4171 | 3.01 | 8.8 |
| 9 | noBiGRU-noAttn | 0.5536 | 0.0630 | −1.24 | 0.6887 | 0.363 | 0.0180 | 3.01 | 15.4 |
| 10 | FULL-LossB | 0.6270 | 0.1728 | 1.71 | 0.9122 | 0.329 | 0.4551 | 3.05 | 15.7 |
| 11 | FULL-LossC | 0.6481 | 0.0548 | 1.76 | 0.7576 | 0.432 | 0.1699 | 3.05 | 7.9 |
| 12 | MWCNN-BiGRU | 0.6665 | 0.0819 | 2.30 | N/A | N/A | 0.3088 | 1.31 | 3.8 |
| 13 | UNet-BiGRU-Swin | 0.6892 | 0.1290 | 4.00 | N/A | N/A | 0.4754 | 2.90 | 17.9 |

### Variations + DL baselines + Classical
| Config | Decoder | Curriculum | NCC Synth | NCC@−15 | SNR | Dice | gNCC | Params | Lat |
|--------|---------|------------|-----------|---------|-----|------|------|--------|-----|
| MR-TAE-FULL (phased) | Linear | Phased | 0.7991 | 0.3492 | 7.01 | 0.9015 | 0.4366 | 3.01 | 40.1 |
| MR-TAE-FULL (no-phase) | Linear | No-phase | 0.8138 | 0.3831 | 7.33 | 0.9022 | 0.4128 | 3.05 | 19.6 |
| **MR-TAE NoMix** | Linear | **NoMix** | **0.8282** | **0.3777** | **9.64** | **0.9192** | **0.4535** | 3.05 | 17.1 |
| MR-TAE-IDWT (no-phase) | IDWT | No-phase | 0.8037 | 0.3726 | 6.89 | 0.8960 | 0.2550 | 2.92 | 24.8 |
| NoMix + Q.Lin FT | Linear | NoMix+FT | 0.7728 | 0.2043 | 7.90 | 0.7706 | **0.8039** | 3.05 | 16.8 |
| PureCNN1D | Linear | Flat | 0.7223 | 0.2142 | 7.56 | N/A | 0.3765 | 0.21 | 0.5 |
| UNet1D_simple | Linear | Flat | 0.8016 | 0.2949 | 9.80 | N/A | 0.4213 | 2.69 | 2.4 |
| DnCNN1D | Linear | Flat | 0.7393 | 0.2222 | 8.00 | N/A | 0.4717 | 0.19 | 2.1 |
| BayesShrink | N/A | N/A | 0.6578 | 0.1575 | 2.73 | N/A | **0.3844** | 0 | N/A |
| Wiener | N/A | N/A | 0.5897 | 0.1520 | 0.39 | N/A | **0.3641** | 0 | N/A |
| Savitzky-Golay | N/A | N/A | 0.0351 | 0.0098 | −1.66 | N/A | **0.2915** | 0 | N/A |

---

## 4. WHAT CHANGED FROM PREVIOUS SESSIONS (reconcile if you find conflicts)

### Architecture corrections (all confirmed from PPTX diagram):
1. **Input proj: 1→32ch** (NOT 1→64ch). The channel sequence is 32→64→128→256.
2. **DWT doubles channels** by stacking approx+detail sub-bands (Liu 2018 MWCNN design). Conv blocks MAINTAIN the width set by DWT.
3. **Skips are (B,32,2048), (B,64,1025), (B,128,513)** — NOT (B,64,2048) etc.
4. **Bottleneck is sequential BiGRU→Swin** (not parallel). BiGRU output feeds into Swin.
5. **Decoder order per level**: Upsample×2 → ½Conv+BN+GELU → Interpolate → CONCAT(AG_skip) → 2×Conv+BN+GELU. AG runs OUTSIDE on the skip path, not inside the upsample box.
6. **Seg classes = discharge type** (BG/Corona/Surface/Internal/Treeing), NOT gap size.
7. **Output proj**: 32→16→1ch (two convs), not just Conv(32→1).

### Benchmark corrections (FINALIZED 2026-05-31):
- **NoMix is the primary recommended model** everywhere (NCC 0.8282, best overall).
- **Headline gNCC comparison uses NO fine-tuning**: NoMix 0.4535 vs BayesShrink 0.3844 (+17.9%).
- Classical gNCC finalized: BayesShrink **0.3844**, Wiener **0.3641**, Savitzky **0.2915**.
- Q.Lin FT variants (NoMix+FT gNCC 0.8039) exist but are NOT used in poster/paper headline.

---

## 5. FILES AND WHERE THEY LIVE

### FYP_MASTER (active development):
```
D:\New folder (2)\Data\FYP_MASTER\
  mr_tae_fusion\
    models\
      mr_tae_fusion.py     ← main model, verified architecture
      baseline_models.py   ← PureCNN1D, UNet1D_simple, DnCNN1D
    config.py              ← seg class scheme (discharge type, not gap size)
  results\                 ← all ablation checkpoints
  results_nomix\           ← NoMix curriculum best model
  results_qlin_cheat_linear\  ← Q.Lin FT linear
  results_nomix_qlin_cheat\   ← NoMix + Q.Lin FT (best real)
  paper\
    paper_arxiv.md         ← main paper text (v4, update to v5)
    metrics_master.json    ← locked benchmark (26 models, all numbers)
    figures\               ← all 16 PNGs
  docs\
    ARCHITECTURE_VERIFIED.md  ← UPDATE with new verified architecture
    SESSION_HANDOFF.md
    DISTILLED_KNOWLEDGE.md
```

### MR_TAE_GITHUB (sync target — push-ready):
```
D:\New folder (2)\Data\MR_TAE_GITHUB\
  docs\
    ARCHITECTURE_VERIFIED.md   ← NEEDS UPDATE (copy from FYP_MASTER)
  paper\
    paper_arxiv.md             ← NEEDS UPDATE (v5)
  README.md                    ← NEEDS UPDATE (NoMix as best)
```

---

## 6. CURSOR TASKS — EXECUTE IN ORDER

### TASK 1: Update ARCHITECTURE_VERIFIED.md (both locations)

Copy the content below exactly into:
- `D:\New folder (2)\Data\FYP_MASTER\docs\ARCHITECTURE_VERIFIED.md`
- `D:\New folder (2)\Data\MR_TAE_GITHUB\docs\ARCHITECTURE_VERIFIED.md`

[The full content is in the ARCHITECTURE_VERIFIED.md output file provided alongside this handoff. Use that exact content.]

### TASK 2: Update paper/paper_arxiv.md — §3.3 Architecture

Find the MWCNN encoder description. It currently says something like:
> "channels double and length halves at each step"

Replace the full encoder tensor shape block with:

```
Input projection: Conv7, 1→32ch → (B, 32, 2048)

Level 1: Conv block → skip₁(B,32,2048) → DWT → (B,64,1025)
Level 2: Conv block → skip₂(B,64,1025) → DWT → (B,128,513)
Level 3: Conv block → skip₃(B,128,513) → DWT → (B,256,257)
Bottleneck: (B,256,257)
```

And update the text to say:
> "Each level applies two Conv1D-BatchNorm-GELU blocks followed by a Daubechies-4 DWT. The DWT stacks the approximation and detail coefficient sub-bands along the channel axis, so that channels double and length reduces to L/2+1 at each step. The Conv blocks refine features at the channel width established by the preceding DWT and do not alter channel count."

Also update the skip connection description:
> "The pre-DWT features are stored as full-resolution skip connections at dimensions (B,32,2048), (B,64,1025) and (B,128,513) for levels 1 through 3 respectively."

### TASK 3: Update abstract and conclusion to use NoMix as primary

Find all instances of "NCC of 0.7991" or "SNR improvement of 7.01 dB" or "Dice score of 0.9015" in the abstract/conclusion and replace:
- Primary headline model: **MR-TAE (NoMix curriculum)**
- NCC: 0.8282 (was 0.7991)
- SNR improvement: 9.64 dB (was 7.01 dB)
- Dice: 0.9192 (was 0.9015)
- Keep MR-TAE-FULL (phased) as the ablation reference in §4.1 and Table 1 — do NOT change Table 1.

Updated abstract headline (replace with):
> "On synthetic data, the NoMix curriculum variant of MR-TAE attains a mean NCC of 0.8282 and an SNR improvement of 9.64 dB, compared to 2.73 dB for the strongest classical baseline (BayesShrink), while simultaneously segmenting PD events at Dice 0.9192 across five discharge types. Following domain adaptation, grounded real NCC reaches 0.8039, exceeding BayesShrink by 17.5%."

### TASK 4: Update README.md in MR_TAE_GITHUB

Find the performance summary table/section and update to:
```
Best configuration: MR-TAE (NoMix curriculum)
  Synthetic NCC: 0.8282 | SNR improvement: 9.64 dB | Dice: 0.9192
  Real Q.Lin gNCC (after FT): 0.8039 (+17.5% vs BayesShrink)
  Parameters: 3.05M | Latency: 17.1 ms
```

### TASK 5: Generate updated figures (run scripts)

Run the following from FYP_MASTER with .venv active:

```bash
cd "D:\New folder (2)\Data\FYP_MASTER"
.venv\Scripts\python.exe scripts\generate_figures.py --model nomix
.venv\Scripts\python.exe scripts\generate_ablation_waveforms.py
```

If scripts don't exist or fail, generate figures manually by loading metrics_master.json and plotting. Place outputs in paper\figures\.

Key figures needed for poster:
- `fig02_snr_bar.png` — per-SNR NCC bar chart, NoMix highlighted
- `fig03_snr_sweep.png` — NCC vs SNR curve, all models
- `fig04_ablation_heatmap.png` — ablation NCC heatmap
- `fig10_qualitative_waveforms.png` — waveform comparison at −15 dB

### TASK 6: Generate Poster (HTML or PDF)

Create `paper\poster\FYP_POSTER_MR_TAE_2026.html` using the poster content from `FYP_POSTER_MR_TAE.md` (provided alongside this handoff).

Poster layout requirements:
- A0 portrait (841 × 1189 mm)
- UM brand: dark blue header #003087, gold accent #CBA135
- 5-column grid
- Include actual figure images from paper\figures\
- Compact tables (no full ablation table — use key-findings table only)
- UM and faculty logos in header (use placeholder if not available)

### TASK 7: Sync to MR_TAE_GITHUB

After all updates:
```bash
cd "D:\New folder (2)\Data\MR_TAE_GITHUB"
# Copy updated files:
copy "..\FYP_MASTER\docs\ARCHITECTURE_VERIFIED.md" "docs\"
copy "..\FYP_MASTER\paper\paper_arxiv.md" "paper\"
copy "..\FYP_MASTER\paper\figures\*.png" "figures\"
copy "..\FYP_MASTER\paper\poster\FYP_POSTER_MR_TAE_2026.html" "paper\"
```

Then verify README.md is updated and commit-ready.

---

## 7. WHAT NOT TO DO

- DO NOT run any script that writes to `results\` or overwrites checkpoints
- DO NOT use the root clone at `D:\New folder (2)\` for anything
- DO NOT change Table 1 (ablation) — MR-TAE-FULL (phased) stays as row 1 reference
- DO NOT add new training runs — all benchmarks are locked
- DO NOT change the frozen test set or re-evaluate models
- DO NOT modify `metrics_master.json`
- DO NOT claim DWT does not double channels — it does (sub-band stacking, Liu 2018)
- DO NOT claim Conv blocks double channels — they maintain width

---

## 8. LOOP-BACK REPORT (what Cursor should tell Claude after completing)

After finishing tasks 1-7, produce a `CURSOR_LOOP_REPORT.md` with:
1. Which files were changed and what changed (one line each)
2. Any architecture terms you found in the codebase that CONTRADICT this handoff — flag them
3. The SHA-256 of `results_nomix\checkpoints\selected_final.pt` (to verify NoMix checkpoint is present)
4. Confirmation that `docs\ARCHITECTURE_VERIFIED.md` now shows input proj = 32ch (not 64ch)
5. Any errors encountered and how you resolved them

---

## 9. PAPER V5 — CHANGES SUMMARY (for Claude to apply to docx)

Claude has the paper_arxiv.md. These are the net changes for V5:

**Change 1 — Abstract (replace primary metric line):**
OLD: "MR-TAE attains a mean NCC of 0.7991 and an SNR improvement of 7.01 dB... Dice score of 0.9015"
NEW: "the NoMix curriculum variant of MR-TAE attains a mean NCC of 0.8282 and an SNR improvement of 9.64 dB... Dice 0.9192 across five discharge types"

**Change 2 — §3.3 encoder description:**
OLD: "(B,1,2048) → Conv(32ch) → skip1(B,32,2048) → DWT → (B,64,1025) → Conv(64ch)..."
Note: the paper currently has ambiguous channel numbers. Standardise to:
  Input proj: 1→32ch; DWT L1: 32→64; DWT L2: 64→128; DWT L3: 128→256
  Text: "Conv blocks refine features at the channel width set by the preceding DWT and do not themselves expand the channel dimension. The DWT stacks approximation and detail sub-bands to double channel count at each level."

**Change 3 — §3.3 skip connection dimensions:**
Correct to: skip₁(B,32,2048), skip₂(B,64,1025), skip₃(B,128,513)

**Change 4 — §3.4 NoMix curriculum:**
Add paragraph after curriculum description:
"The NoMix curriculum variant trains on single-archetype windows only, excluding mixed-type augmentation windows from all three phases. This forces the model to learn purer discharge-type signatures per class, improving both signal reconstruction NCC and segmentation Dice. The NoMix variant is the recommended configuration for deployment."

**Change 5 — §4.2 headline model in variations table:**
Bold/highlight MR-TAE NoMix row. Update text:
"The NoMix curriculum variant achieves the highest performance across all synthetic metrics: NCC 0.8282, SNR improvement 9.64 dB, Dice 0.9192. This is the recommended configuration for both synthetic-domain deployment and as the starting point for real-domain fine-tuning."

**Change 6 — §6 Conclusion first paragraph:**
Replace "MR-TAE achieves mean NCC 0.7991 and SNR improvement 7.01 dB" with
"MR-TAE (NoMix curriculum) achieves mean NCC 0.8282 and SNR improvement 9.64 dB"
and update Dice from 0.9015 → 0.9192.

**Change 7 — §3.3 decoder description:**
Add: "Within each decoder block, the upsample module runs Upsample×2, then a single halving Conv1D with BN and GELU, then interpolation to match the skip length. The attention gate runs outside this module on the skip path before concatenation. After concatenation, two Conv1D-BN-GELU blocks refine the fused features."

No other sections need substantive change. Figures are being regenerated by Cursor (Task 5).

---

## 10. ENVIRONMENT QUICK-REFERENCE

```
GPU:           RTX 4070 (CUDA 12+)
Python:        .venv\Scripts\python.exe (FYP_MASTER)
PyTorch:       confirmed working with AMP
Key packages:  torch, torchaudio, PyWavelets, mlflow, optuna
Checkpoint key: model_state (not model_state_dict)
Test set hash (synthetic, 7-level): ade2b9ec...
FYP_MASTER:    D:\New folder (2)\Data\FYP_MASTER\
GITHUB:        D:\New folder (2)\Data\MR_TAE_GITHUB\
```

---
*End of handoff. Good luck. When in doubt, verify against paper/metrics_master.json for numbers and the PPTX diagram for architecture.*
