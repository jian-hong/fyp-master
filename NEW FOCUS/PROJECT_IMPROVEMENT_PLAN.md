# Project Improvement Plan — Final Stretch
**Window:** 01 May → ~31 May 2026 (≈4 weeks to thesis submission)
**Author note (Claude, May 1 2026):** Scope is deliberately small. You have a complete, working, well-benchmarked system. The danger now is feature-creep, not under-delivery. Every suggestion below is tagged:
- 🔴 **DO** — strict thesis blocker / supervisor-feedback item
- 🟡 **SHOULD** — high-value polish, fits in remaining time
- 🟢 **FUTURE WORK** — name-drop in Chapter 6, do NOT implement now

---

## 1. Thesis Blockers (🔴 DO First)

### 1.1 NCC reconciliation (0.5145 vs 0.948)
This is the single most damaging open issue if it surfaces in the viva. The two values are almost certainly correct under different aggregation rules (per-SNR macro-average vs. pooled or 5 dB only) — the bug is presentational, not numerical. Fix:
1. Define one canonical metric: **per-SNR mean NCC then macro-averaged across {−15, −10, −5, 0, +5} dB**. This is what the thesis should report.
2. Have the dashboard report all three: pooled, per-SNR, macro.
3. Add a one-line definition in the metrics caption of every figure.
4. Re-issue Table 4.1 with the corrected, reconciled column.

### 1.2 Per-method signal-overlay figures (lecturer's Apr 18 ask)
Use the script in `evaluation/generate_method_comparison_figures.py` (provided in this drop). It produces:
- A 4-panel per-method figure: noisy vs. clean, denoised vs. clean overlay, residual, PSD comparison.
- A grid summary across all 13 models.
- Automatic SNR breakdown at {−15, −5, +5} dB (matching the thesis SNR levels).
Run it once after every benchmark refresh. Output goes to `results/{model_name}/`.

### 1.3 Noise model narrative correction
Chapter 3 currently mixes EM-noise types (NBI, powerline) with the AE physical framing. Two-paragraph fix:
- Paragraph A: AE physics — what the *real* noise sources are (mechanical, magnetostrictive, thermal in the piezo).
- Paragraph B: Synthetic noise model used in training — what mathematical proxies were chosen, why they are reasonable surrogates, and what is explicitly NOT modelled (mechanical transients, dispersion, multipath).
- Add a one-sentence "limitation" pointer to Chapter 6.

---

## 2. Strategic SOTA Additions (🟡 SHOULD)

### 2.1 CleanUNet-1D as external benchmark anchor
**Why:** Right now your "best" claim is internal. An examiner will ask "compared to what published work?" CleanUNet (NVIDIA, ICASSP 2022) is the cleanest 1D denoising baseline that's neither too obscure nor too domain-specific (it's speech enhancement, but architecturally directly comparable).

**How:** Re-implement the published architecture in PyTorch (≈300 LOC), train on your synthetic dataset for one curricular pass, evaluate on the same SHA256-verified test sets. One row in the comparison table — that's it.

**Expected outcome:** MR-TAE-FULL-LossB beats it on real NCC because the wavelet pooling and BiGRU are doing real work. If it doesn't beat it, that itself is a thesis-worthy honest finding.

### 2.2 Acknowledge in Chapter 6 that Mamba/SSM is the next architectural frontier
A December 2025 paper (Tan et al., *Measurement*) introduced a Mamba-based PD detection method using Synchrosqueezed Wavelet Transform plus Vision Mamba with Chunk-wise Parallel Spiral Scan — exactly your problem space. Cite it as evidence the field is moving toward state-space models. **Do not implement it now.** Cite it, position MR-TAE as a complementary attention-based alternative, and explain that linear-time Mamba models would be a logical successor for longer sequences.

Useful citations to add to Chapter 6:
- Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", arXiv:2312.00752 (foundational)
- The 2025 *Electric Power Systems Research* review article on acoustic PD denoising (Kumar et al., *Recent trends and open challenges in acoustic partial discharge signal denoising techniques*) — it's the most current review in your exact domain
- A 2025 hybrid CNN-AE paper from NIT Aizawl/Silchar (Kumar et al., *Signal, Image and Video Processing*, "A hybrid deep learning framework for denoising acoustic partial discharge signals") for direct comparison framing

### 2.3 Self-supervised: Noise2Void on Q.Lin (already on your roadmap)
This is high value because it lets you make a defensible claim about real-data domain adaptation. Use the standard blind-spot strategy adapted for 1D: random pixel masking with a 1×5 receptive-field-aware exclusion. Replay-buffer guard against synthetic regression (20% replay).

**Watch out for:** Catastrophic forgetting on synthetic. If real NCC goes up but synthetic drops by >0.05, your replay fraction is too low.

---

## 3. Performance / Automation (🟡 SHOULD)

### 3.1 Kernel-level optimization stack
All five compose without conflict. Apply in this order, validate NCC at each step:

```python
import torch
torch.set_float32_matmul_precision('high')        # TF32 on Ampere/Ada
model = model.to(memory_format=torch.channels_last)
model = torch.compile(model, mode="reduce-overhead")
optimizer = torch.optim.AdamW(model.parameters(), fused=True)  # fused kernel
# In training step:
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = model(x)
# Flash attention for SDPA — automatic on Ada
torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True)
```

Expected: 25–40% latency reduction on RTX 4070, no metric change. **bf16 not fp16** — bf16 is more numerically stable for signal regression and avoids the loss-scaling complications of fp16 AMP.

### 3.2 Library upgrades worth your time

| Library | Why | Time cost |
|---|---|---|
| **TorchMetrics** | Replaces hand-rolled metrics with battle-tested implementations. Native MLflow integration. | ~2 hr to swap |
| **Hydra** | Replaces YAML hardcoding. Composable configs, sweeps via `--multirun`. Pairs natively with Optuna. | ~1 day to migrate. **Skip if** you've stopped tweaking configs. |
| **Weights & Biases** (wandb) | Better HPO sweep UI than MLflow. Already noted in your forward plan. | ~1 hr setup. |
| **TorchProfiler + Holistic Trace Analysis** | Find the *actual* bottleneck before optimizing. | ~30 min |

**Skip these:** PyTorch Lightning (rewrite cost too high this late), Ray Tune (Optuna already works), Aim (MLflow is fine), DVC (only useful if data changes — your data is frozen).

### 3.3 24/7 automation — concrete Windows recipe

The harness in `pipeline/watchdog.py` (provided) plus `scripts/install_watchdog_task.ps1` will:
1. Run the orchestrator continuously.
2. Detect orchestrator process death and restart with exponential back-off (30 s → 60 s → 120 s → cap 600 s).
3. Detect GPU thermal events and pause until temp drops.
4. Detect VRAM OOM via the existing memory_manager.
5. Send Telegram notifications on every state transition (start, resume, OOM, thermal, complete).
6. Auto-relaunch on Windows boot via Task Scheduler with highest privileges.

Power-management requirements (set these once on the Windows machine):
- Power Plan = High Performance
- Sleep = Never
- USB Selective Suspend = Disabled
- PCIe Link State Power Management = Off
- Windows Update → Active Hours = 24h (defer to manual install)
- BIOS: Enable "Restore on AC power loss"

---

## 4. Edge Deployment Track (🟡 SHOULD — for PD-CLIP commercial story)

### 4.1 Knowledge distillation
Teacher: MR-TAE-FULL-LossB (3.05 M, 6.67 ms).
Student: EdgePD-Net (~1.5 M, target <5 ms). Use the existing MWCNN-BiGRU architecture as the student backbone — it's already in your registry at 1.31 M and 0.7709 real NCC.

Loss: `α·Charbonnier(student_signal, clean) + β·MSE(student_signal, teacher_signal) + γ·KL(student_seg_logits, teacher_seg_logits)` with α=0.5, β=0.3, γ=0.2 as starting point.

### 4.2 INT8 post-training quantization
Use `torch.ao.quantization` for static PTQ with calibration on 200 representative real Q.Lin windows. Accuracy floor: real NCC must stay ≥0.70 (down from teacher 0.857 is acceptable for INT8).

### 4.3 ONNX export + onnxruntime
```python
torch.onnx.export(model_int8, dummy_input, "edgepd.onnx", opset_version=17,
                  input_names=["signal"], output_names=["denoised", "seg_mask"],
                  dynamic_axes={"signal": {0: "batch"}})
```
Validate on Raspberry Pi CM4 with `onnxruntime` ARM64 build. Don't even attempt the i.MX 8M Plus NPU integration for the thesis — that's a hardware partnership conversation with TNB Research, not an FYP deliverable.

---

## 5. Future Work — for Chapter 6 only (🟢)

State these as gaps, do not implement:
1. **Mamba/SSM bottleneck** replacing BiGRU+Swin for longer sequences and lower memory.
2. **Conditional diffusion denoiser** (DDPM-style score-matching) for very low SNR (<−15 dB) regimes.
3. **Spectral Kurtosis (SK) front-end** as a pre-screening stage to discriminate mechanical bursts from PD.
4. **Test-Time Adaptation** (TTA) for online substation deployment — Tent or MEMO style.
5. **Field validation** with TNB Research on a real transformer.
6. **Multi-sensor fusion** combining AE with HFCT for sensitivity boost.
7. **Severity regression head** (in addition to type segmentation) — addresses the "Q3 examiner curveball" already pre-answered in `AGENT_PLAN.md`.

---

## 6. What to Stop Doing

- **Don't add more ablations.** 13 is more than any reviewer will engage with. Three more variants will not move a grade.
- **Don't migrate to PyTorch Lightning or Hydra now.** The marginal benefit doesn't pay back the integration time.
- **Don't pivot to Mamba.** It's a better architecture for some problems, but rewriting now throws away 13 trained checkpoints and 9 hours of HPO.
- **Don't chase higher synthetic NCC.** You're already at 0.62 on synthetic; the bottleneck is the synthetic→real domain gap, not synthetic accuracy.
- **Don't add more loss-function ablations.** LossA vs LossB vs LossC is enough for a thesis chapter.

---

## 7. Calendar (suggested)

| Week | Mon–Wed | Thu–Sun |
|---|---|---|
| **01–07 May** | NCC reconciliation, signal-overlay figures, flat-SNR baseline finished | CleanUNet-1D training, kernel optimizations |
| **08–14 May** | Noise2Void fine-tuning, replay-buffer ablation | Knowledge distillation → EdgePD-Net |
| **15–21 May** | INT8 quantization + ONNX export + RPi profiling | Thesis Ch.4/5 final pass with all new figures |
| **22–28 May** | Full draft to supervisor; viva slides | Mock viva, address feedback |
| **29 May–** | Submit | Defend |

---

## 8. References (for thesis Chapter 6 SOTA section)

Verify each before citing — Claude noted these from a May 2026 search but you must cross-check the venue/year/DOI:

- Q. Kumar, B. Ganguly, D. Dey, S. Chatterjee. "Recent trends and open challenges in acoustic partial discharge signal denoising techniques: A review." *Electric Power Systems Research*, vol. 248, 111931, 2025.
- C. Kumar, B. Ganguly, D. Dey, S. Chatterjee. "A hybrid deep learning framework for denoising acoustic partial discharge signals." *Signal, Image and Video Processing*, 2025.
- A. Gu, T. Dao. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv:2312.00752, 2023.
- W. Crivelaro et al. "Denoising autoencoders for partial discharge signals in instrument transformers." 2024 (cited in the 2025 review).
- Z. Kong, W. Ping, A. Dantrey, B. Catanzaro. "Speech Denoising in the Waveform Domain with Self-Attention." (CleanUNet, NVIDIA), ICASSP 2022.
- Krull, Buchholz, Jug. "Noise2Void: Learning Denoising from Single Noisy Images." CVPR 2019.
- Lehtinen et al. "Noise2Noise: Learning Image Restoration without Clean Data." ICML 2018. (Cited only to justify why N2N is *not* applicable here.)
- Gulrajani et al. "Improved Training of Wasserstein GANs." NeurIPS 2017. (For your existing WGAN-GP work.)
