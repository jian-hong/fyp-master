# HANDOFF: arXiv Paper + FYP Report Generation
## MR-TAE: Intelligent Acoustic Emission Partial Discharge Denoising
**Student:** Oo Jian Hong (22004678) · University of Malaya
**Supervisor context:** TNB Research Sdn Bhd application target
**Prepared for:** New Claude window (fresh context)

---

## 0. WHAT THIS HANDOFF IS FOR

You are helping write two documents in parallel:

| Document | Target | Format | Length |
|----------|--------|--------|--------|
| **A. arXiv Preprint** | Public research community | IEEE double-column / arXiv LaTeX | ~8–10 pages |
| **B. FYP Thesis Report** | UM examiner + supervisor | UM thesis format, formal | ~80–120 pages |

**Two-phase process (CRITICAL — context budget):**

- **Phase 1 — Research & Distillation:** Deep research session. Claude reads all uploaded documents, synthesises a structured knowledge base (the "distilled output"), then stops. Output: `DISTILLED_KNOWLEDGE.md`.
- **Phase 2 — Generation:** New Claude window receives only `DISTILLED_KNOWLEDGE.md` (not the raw files) and generates the arXiv paper and/or FYP report sections on demand.

This two-phase split exists because the combined project files exceed any single context window.

---

## 1. FILES TO UPLOAD TO PHASE-1 CLAUDE

Upload ALL of the following. Each section below says exactly what to upload and why.

**Bundle folder:** copies for sections 1.2–1.5 plus figure PNGs live under `D:\New folder (2)\Data\FYP_MASTER\results\paper_assets\` (markdown/JSON at the root of that folder, images in `figures\`). After you change benchmarks or docs, run `python evaluation/generate_figures.py` and `python scripts/generate_paper_assets.py`, then re-sync that folder (re-run your copy/aggregate step or repeat the same file copies).

### 1.1 — Primary Source (Upload First)
| File | Path on your machine | Why |
|------|---------------------|-----|
| `OoJianHong-22004678-Report.pdf` | `D:\New folder (2)\Data\FYP_MASTER\OoJianHong-22004678-Report.pdf` (copy into `results\paper_assets\` when you have it — **not present in repo yet**) | Your existing FYP report draft — primary content source |

### 1.2 — Architecture & Results Reference
| File | Path on your machine | Why |
|------|---------------------|-----|
| `ARCHITECTURE_SUMMARY.md` | `D:\New folder (2)\Data\FYP_MASTER\results\paper_assets\ARCHITECTURE_SUMMARY.md` (source of truth: `docs\ARCHITECTURE_SUMMARY.md`) | All 13 ablation variant configs, param counts, tensor shapes |
| `benchmark_results.json` | `D:\New folder (2)\Data\FYP_MASTER\results\paper_assets\benchmark_results.json` (aggregated from every `results\<Model>\benchmark_results.json`) | Measured NCC/RMSE/latency per model |
| `RESULTS_COMPARISON.md` | `D:\New folder (2)\Data\FYP_MASTER\results\paper_assets\RESULTS_COMPARISON.md` (auto-generated: `python evaluation/generate_figures.py` → `docs\RESULTS_COMPARISON.md`) | Cross-model comparison table |
| `FINAL_VERDICT.md` | `D:\New folder (2)\Data\FYP_MASTER\results\paper_assets\FINAL_VERDICT.md` (source: `docs\FINAL_VERDICT.md`) | Filled conclusions (fill placeholders before uploading!) |

### 1.3 — Research & Methodology Context
| File | Path on your machine | Why |
|------|---------------------|-----|
| `FULL_RESEARCH_PLAN.md` | `D:\New folder (2)\Data\FYP_MASTER\results\paper_assets\FULL_RESEARCH_PLAN.md` | Signal physics corrections, noise model, literature framing |
| `EXAMINER_QA.md` | `D:\New folder (2)\Data\FYP_MASTER\results\paper_assets\EXAMINER_QA.md` (source: `docs\EXAMINER_QA.md`) | 15 deep Q&As on every design choice — gold for methodology section |
| `AGENT_PLAN.md` | `D:\New folder (2)\Data\FYP_MASTER\results\paper_assets\AGENT_PLAN.md` | Experiment design rationale |

### 1.4 — Codebase Summary
| File | Path on your machine | Why |
|------|---------------------|-----|
| `CODEBASE_AUDIT.md` | `D:\New folder (2)\Data\FYP_MASTER\results\paper_assets\CODEBASE_AUDIT.md` (source: `docs\CODEBASE_AUDIT.md`) | Code-level evidence for methodology claims |
| `PROJECT_STATUS.md` | `D:\New folder (2)\Data\FYP_MASTER\results\paper_assets\PROJECT_STATUS.md` | Experiment state, what is/isn't done |

### 1.5 — Generated Figures (upload as images)
Render/export these before the Phase-1 session (copies also under `results\paper_assets\figures\`):
| Figure | How to get it |
|--------|--------------|
| Waveform comparison PNG (3 SNR levels) | Run `NEW FOCUS\generate_method_comparison_figures.py` → `comparison_snr_*.png` in `figures\` |
| NCC bar chart (all 13 models) | `figures\ncc_summary_bar.png` and/or `figures\fig4_ablation_ncc_bar.png` from method-comparison / `scripts\generate_paper_assets.py` |
| Architecture diagram | `figures\fig2_architecture.png` from `python scripts\generate_paper_assets.py` (or export from ASCII in `ARCHITECTURE_SUMMARY.md`) |
| Ablation heatmap | `figures\ablation_heatmap.png` (source: `results\benchmark\ablation_heatmap.png` via `python evaluation\generate_figures.py`) |
| SNR sweep curve | `figures\snr_sweep.png` (source: `results\benchmark\snr_sweep.png`) |
| Training curves | `figures\training_curves.png` (source: `results\benchmark\training_curves.png`) |

---

## 2. PHASE-1 PROMPT — DISTILLATION SESSION

Copy this prompt verbatim into a fresh Claude window after uploading all files above:

---

```
You are a research assistant helping me distill my Final Year Project into a
structured knowledge base that will be used to write both an arXiv paper and a
full FYP thesis report.

MY PROJECT:
Title: "MR-TAE: Multi-Resolution Temporal Autoencoder for Intelligent Acoustic
       Emission Partial Discharge Signal Denoising in High-Voltage Transformers"
Student: Oo Jian Hong, 22004678, University of Malaya
Target application: TNB Research Sdn Bhd (Malaysian national grid operator)

YOUR TASK — Phase 1 of 2:
Read ALL the uploaded documents carefully. Then produce a single structured
output called DISTILLED_KNOWLEDGE.md containing exactly these sections:

=== SECTION 1: PAPER IDENTITY ===
- Full title, short title, authors, institution, keywords (IEEE style)
- One-sentence contribution statement
- Three-sentence abstract draft

=== SECTION 2: PROBLEM & MOTIVATION ===
- What partial discharge is and why it matters (2–3 paragraphs, citable facts)
- Limitations of existing approaches (HFCT, UHF, conventional AE denoising)
- The specific gap this work fills (AE + deep learning + real-world validation)

=== SECTION 3: RELATED WORK TABLE ===
A table of 12–15 prior works covering:
- Traditional signal processing (wavelet thresholding, EMD, Wiener filter)
- CNN-based denoisers (DnCNN, U-Net 1D, MWCNN)
- Transformer-based (CleanUNet, Swin-based)
- Hybrid recurrent-attention
- Columns: Author/Year, Method, Modality (AE/HFCT/UHF), Dataset, Key Metric, Limitation

=== SECTION 4: ARCHITECTURE SPECIFICATION ===
For each of the 13 ablation variants, document:
- Component flags (wavelet/BiGRU/Swin/Attn/MTL)
- Parameter count (M)
- Loss configuration (A/B/C)
- Role in the ablation study

Also document the full MR-TAE-FULL architecture:
- Encoder (MWCNN 3-level, DWT, channel dims)
- Bottleneck (BiGRU + Swin, tensor shapes)
- Decoder (IDWT, attention gates, skip connections)
- Output heads (signal head, segmentation head)
- Training: 3-phase curriculum, SNR schedule, loss functions

=== SECTION 5: EXPERIMENTAL SETUP ===
- Dataset: Q.Lin real AE dataset (4 discharge types, window sizes, preprocessing)
- Synthetic dataset: noise model, SNR range, physics corrections
- Hardware: RTX 4070, batch sizes, epochs
- Evaluation metrics: NCC, RMSE, SNR improvement, Dice, FPR, latency

=== SECTION 6: RESULTS (fill from benchmark_results.json and figures) ===
- Full results table (all 13 models × all metrics)
- Key finding 1: LossB vs LossA (2.27× NCC improvement)
- Key finding 2: BiGRU as generalization anchor
- Key finding 3: NCC saturation anomaly and explanation
- Key finding 4: MWCNN-BiGRU as edge deployment candidate
- Figures list with captions

=== SECTION 7: DISCUSSION POINTS ===
- Why loss function dominated architecture (domain gap theory)
- Real vs synthetic performance divergence
- Limitations: no ground truth for real data, WGAN noise model
- Ethical/safety implications for grid infrastructure

=== SECTION 8: CONCLUSIONS & FUTURE WORK ===
- 4–5 bullet conclusions
- Future work: Noise2Void, Optuna HPO, knowledge distillation to EdgePD-Net,
  PD-CLIP hardware deployment

=== SECTION 9: CITATION SEEDS ===
List 20 key references with full citation metadata (author, year, title, venue,
DOI if known). Include classics (Kingma Adam 2015, PyWavelets, etc.) and
domain-specific works (AE-PD detection papers, transformer denoising papers).

=== SECTION 10: FYP-SPECIFIC ADDITIONS ===
(Not in arXiv paper, but needed for thesis)
- Background chapter outline: transformer insulation, PD physics, AE sensing
- Methodology chapter structure
- Figures list for thesis (beyond paper figures)
- Appendix suggestions: hyperparameter tables, training curves, code snippets

---
After completing DISTILLED_KNOWLEDGE.md, say:
"PHASE 1 COMPLETE. Copy DISTILLED_KNOWLEDGE.md to a new Claude window for Phase 2."
Do NOT start writing the paper or thesis yet.
```

---

## 3. PHASE-2 PROMPT — GENERATION SESSION

Open a NEW Claude window. Upload only `DISTILLED_KNOWLEDGE.md`.
Then use the following prompts in sequence (one conversation, new requests):

### 3A — arXiv Abstract + Introduction
```
Using DISTILLED_KNOWLEDGE.md, write the Abstract and Introduction section of
an arXiv paper in IEEE conference style (single-file LaTeX, no journal template).

Abstract: 150–200 words. Structure: context → problem → proposed solution →
key results → conclusion sentence.

Introduction: 600–800 words covering: (1) transformer PD monitoring importance,
(2) why existing methods fail, (3) contribution of MR-TAE, (4) paper structure
roadmap. Use \cite{} placeholders matching SECTION 9 of DISTILLED_KNOWLEDGE.
```

### 3B — Related Work
```
Write the Related Work section (600–800 words). Organise into 3 subsections:
(2.1) Signal processing baselines, (2.2) Deep learning denoisers,
(2.3) Hybrid recurrent-attention models. Use the Related Work Table from
DISTILLED_KNOWLEDGE Section 3. Conclude with a gap statement.
```

### 3C — Methodology
```
Write the Methodology section (800–1000 words). Cover:
(3.1) Problem formulation (noisy AE signal model, denoising objective)
(3.2) MWCNN Encoder (DWT motivation, level specs)
(3.3) BiGRU + Swin Bottleneck (why both, what each contributes)
(3.4) MWCNN Decoder (IDWT, attention gates)
(3.5) Multi-task learning (signal + segmentation heads, loss functions A/B/C)
(3.6) Curriculum training (3-phase SNR schedule)
Include LaTeX equations for: DWT, Charbonnier loss, GDice loss, NCC metric.
```

### 3D — Experiments & Results
```
Write the Experiments section. Include:
- Dataset description table
- Full ablation results table (all 13 models, 6 metrics)
- Waveform comparison figure reference
- Discussion of LossB vs LossA finding
- Discussion of BiGRU as generalization anchor
- SNR sweep figure reference
```

### 3E — Conclusion + References
```
Write the Conclusion (250 words) and a BibTeX bibliography from SECTION 9
of DISTILLED_KNOWLEDGE. Format references as \bibitem entries.
```

### 3F — FYP Report: Background Chapter
```
Now switch to FYP thesis format (UM thesis style: formal, numbered sections,
double-spaced). Write Chapter 2: Literature Review (2000–3000 words) covering:
2.1 High-voltage transformer condition monitoring
2.2 Partial discharge phenomena and types
2.3 Acoustic emission sensing for PD detection
2.4 Signal denoising approaches: traditional vs. deep learning
2.5 Summary and research gap
```

### 3G — FYP Report: Methodology Chapter
```
Write Chapter 3: Methodology (2500–3500 words) in UM thesis format.
3.1 Research framework overview
3.2 Dataset acquisition and preprocessing
3.3 Proposed MR-TAE architecture (full specification)
3.4 Ablation study design (all 13 variants with rationale)
3.5 Training protocol (curriculum, hardware, hyperparameters)
3.6 Evaluation metrics
```

### 3H — FYP Report: Results + Discussion
```
Write Chapter 4: Results and Discussion (2000–3000 words).
4.1 Ablation study results (full table with analysis)
4.2 Loss function analysis (LossA vs LossB vs LossC)
4.3 Real-world generalization (Q.Lin dataset results)
4.4 Edge deployment candidate (MWCNN-BiGRU analysis)
4.5 Comparison with baseline methods
4.6 Limitations and threats to validity
```

---

## 4. CURSOR AI AGENT PROMPT

Use this as your `.cursorrules` or as a Cursor Composer instruction for
preparing all figures and supplementary files:

---

```
# Cursor AI Agent: FYP Paper Asset Preparation
# Project: MR-TAE AE-PD Denoising | Student: Oo Jian Hong 22004678

You are preparing all figures, tables, and supplementary assets for an arXiv
submission and FYP thesis report.

## YOUR GOAL
Generate or verify the following output files in results/paper_assets/:

### FIGURES NEEDED (PNG, 300 DPI min, white background)

1. fig1_system_overview.png
   - Block diagram: Transformer tank → AE sensor → ADC → MR-TAE pipeline → PD alert
   - Use matplotlib or seaborn, professional style

2. fig2_architecture.png
   - Full MR-TAE-FULL architecture diagram
   - Input (B,1,2048) → MWCNN Enc 3-level → BiGRU+Swin bottleneck →
     MWCNN Dec → dual output heads
   - Show tensor shapes at each stage

3. fig3_waveform_comparison.png
   - 4-row plot: Clean | Noisy (-5 dB) | MR-TAE-FULL-LossB | MWCNN-BiGRU
   - 3 columns: SNR = -15, -5, +5 dB
   - Load from: results/method_comparison_figures/comparison_snr_*.png
   - Or re-render using the fixed generate_method_comparison_figures.py

4. fig4_ablation_ncc_bar.png
   - Horizontal bar chart: all 13 models, sorted by Real NCC
   - Dual colour: synthetic NCC (blue) and real NCC (orange)
   - Annotate MWCNN-BiGRU and MR-TAE-FULL-LossB with arrows

5. fig5_snr_sweep.png
   - Line plot: NCC vs input SNR (-15 to +5 dB) for top 5 models
   - Load from: results/benchmark/snr_sweep.png or re-generate

6. fig6_loss_comparison.png
   - Side-by-side: LossA vs LossB vs LossC on real NCC and synthetic NCC
   - Add a 2.27x annotation for the LossA→LossB improvement

7. fig7_efficiency_scatter.png
   - X-axis: parameter count (M), Y-axis: Real NCC
   - Bubble size: inference latency (ms)
   - Label MWCNN-BiGRU as "Edge candidate", MR-TAE-FULL-LossB as "Best accuracy"

### TABLES NEEDED (export as both LaTeX .tex and CSV)

1. table1_dataset.tex — Q.Lin dataset summary (4 PD types, sample counts, fs)
2. table2_ablation.tex — 13-model ablation (all metrics, 7 columns)
3. table3_comparison.tex — MR-TAE-FULL-LossB vs CleanUNet-1D vs DnCNN vs Wiener

### LATEX ASSETS

1. mr_tae_arxiv.tex — skeleton arXiv paper (to be filled by Phase-2 Claude)
   Structure:
   \documentclass[conference]{IEEEtran}
   Sections: Abstract | Intro | Related Work | Methodology | Experiments |
             Discussion | Conclusion | References
   Include \includegraphics{} calls for all 7 figures

2. references.bib — BibTeX file with 20+ references

### VERIFICATION SCRIPT: verify_assets.py
Write a Python script that:
- Checks all 7 PNGs exist and are ≥ 300 DPI
- Checks all 3 tables exist
- Checks LaTeX compiles without errors (use subprocess + pdflatex)
- Prints a checklist summary

## HOW TO LOAD MODEL RESULTS
Load from: results/*/benchmark_results.json (one per model)
Or from: mlruns/ MLflow experiment store

To list all benchmarks:
  import json, pathlib
  results = []
  for p in pathlib.Path("results").rglob("benchmark_results.json"):
      results.append(json.loads(p.read_text()))
  # Sort by real NCC
  results.sort(key=lambda r: r.get("real_qlin", {}).get("ncc_mean", 0), reverse=True)

## IMPORTANT PATHS
PROJECT_ROOT = D:\New folder (2)\Data\FYP_MASTER
RESULTS_ROOT = PROJECT_ROOT\results
CHECKPOINTS  = RESULTS_ROOT\{MODEL_ID}\checkpoints\best.pt
OUTPUT_DIR   = PROJECT_ROOT\results\paper_assets\

## STYLE REQUIREMENTS
- All figures: matplotlib, style='seaborn-v0_8-paper', fontsize≥10
- Colour palette: tab10 (consistent across all figures)
- Export: fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
- Figure labels: (a), (b), (c) for multi-panel figures
```

---

## 5. UNRESOLVED ISSUES TO FLAG IN PAPER

Before submitting, Claude should be told about these known issues
(mention in PHASE-1 prompt context or add to DISTILLED_KNOWLEDGE):

| Issue | Status | How to handle in paper |
|-------|--------|----------------------|
| NCC discrepancy: thesis says ~0.5145, dashboard says ~0.948 | Unresolved | Clarify in methodology: dashboard NCC is synthetic, 0.5145 is real Q.Lin NCC (different test sets) |
| EM noise narrative in AE context | Fixed in FULL_RESEARCH_PLAN.md | Use AE physics framing in paper (not HFCT terminology) |
| No ground truth for real Q.Lin | Known limitation | State explicitly: real-data NCC uses input window as pseudo-reference |
| Flat-SNR baseline missing | Planned | Note as limitation / future work |

---

## 6. QUICK REFERENCE — KEY NUMBERS FOR THE PAPER

Copy these into DISTILLED_KNOWLEDGE or memorise for Phase-2 prompts:

```
Best model:          MR-TAE-FULL-LossB
  Real NCC:          ~0.948 (dashboard) | ~0.5145 (thesis, likely real Q.Lin)
  Synthetic NCC:     ~0.946–0.948 (pooled across SNRs)
  Improvement vs LossA:  2.27× real NCC

Edge model:          MWCNN-BiGRU
  Params:            1.31 M
  Real NCC:          0.771
  Latency (b=1):     ~6 ms GPU

Ablation baseline:   MR-TAE-noSwin-noBiGRU
  Params:            1.01 M
  Real NCC:          0.302
  Synthetic NCC:     0.574 (mean across SNRs)

Hardware:            RTX 4070 Laptop GPU, 12 GB VRAM
Dataset (real):      Q.Lin AE dataset, 4 discharge distances (1.0/1.8/2.0/2.5 mm)
Dataset (synthetic): 10,000 test windows, SNR ∈ {-15, -10, -5, 0, +5} dB
Sampling rate:       2 MHz
Window size:         2048 samples (1.024 ms)
```

---

## 7. PAPER CHECKLIST (BEFORE SUBMISSION)

- [ ] All benchmark_results.json placeholder values filled in FINAL_VERDICT.md
- [ ] NCC discrepancy (0.5145 vs 0.948) reconciled and explained in text
- [ ] All 7 figures rendered at 300 DPI
- [ ] LaTeX compiles without errors
- [ ] Abstract ≤ 200 words
- [ ] References ≥ 20, all cited in text
- [ ] Ethics statement: no human subjects, public dataset with attribution
- [ ] arXiv category: cs.LG or eess.SP (Electrical Engineering and Systems Science)
- [ ] Acknowledgement: supervisor + TNB Research Sdn Bhd
- [ ] Student ID and UM affiliation in author block

---

*Handoff document prepared by Claude Sonnet 4.6 · May 2026*
