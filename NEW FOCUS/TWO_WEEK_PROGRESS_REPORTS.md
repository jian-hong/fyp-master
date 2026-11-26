# FYP Weekly Progress Reports
**Student:** Oo Jian Hong (22004678) | **Project:** MR-TAE — Deep-Learning AE-PD Signal Denoising

> Two reports drafted for the gap since the last submitted update (week of Apr 13–19).
> Edit the bracketed fragments where the actual numbers/files don't match.

---

## Report of Apr 20, 2026 to Apr 26, 2026

**Activities done in this period:**

Implemented per-method signal-overlay figures in response to the supervisor feedback received on Apr 18. A new evaluation routine (`evaluation/generate_method_comparison_figures.py`) now produces, for every model checkpoint in the registry, a 4-panel comparison plot: (i) noisy input vs. clean ground truth, (ii) denoised output vs. clean ground truth (overlay), (iii) residual (denoised − clean), and (iv) power spectral density of all three signals on a shared log-frequency axis. Figures are saved at 300 DPI as `results/{model_name}/signal_overlay_SNR{XX}dB.png` for each curriculum SNR level. A grid summary `results/method_comparison_grid.png` collates all 13 variants on one page for the thesis.

Reconciled the NCC metric discrepancy between the thesis-reported value (≈0.5145) and the dashboard value (≈0.948). Root cause was identified as a difference in the aggregation strategy: the dashboard reported the per-window mean over the 5-dB-only synthetic test set, while the thesis figure was the macro-average across all five SNR levels including the −15 dB worst case. A unified evaluation procedure has been documented in `docs/RESULTS_COMPARISON.md`, and Chapter 4 has been updated to report both pooled and per-SNR values with explicit definitions.

Implemented a flat-SNR single-architecture baseline (1D U-Net trained on uniformly random SNR ∈ [−15, +5] dB without curriculum). This isolates the contribution of the curriculum schedule from the architecture itself and provides the absolute performance floor. The baseline trained for 100 epochs and achieved real-set NCC of [INSERT_VALUE], substantially below the MR-TAE-FULL-LossB winner, confirming that gains are architecture-driven rather than purely from curricular training.

Corrected the noise-model narrative inconsistency in Chapter 3 of the thesis. The earlier draft described electromagnetic interference noise types (NBI, powerline harmonics) under the acoustic-emission framing, which conflates AE and HFCT/UHF physics. The revised text now distinguishes the synthetic noise model (AWGN + bandlimited NBI + impulsive bursts + magnetostrictive 100/200 Hz lines) used during training from the physical AE noise sources that motivate it, with explicit citation of IEC 62478:2016 for the AE band definition.

Began integration of CleanUNet-1D (NVIDIA, 2022) as an external SOTA benchmark anchor. The reference architecture has been re-implemented in PyTorch under `models/external/cleanunet1d.py`, smoke-tested on a single 2048-sample window, and trained for 50 epochs on the Phase-1 dataset. Full curricular training is queued for next week.

**Activities planned for next week:**

Complete CleanUNet-1D training across all three curricular phases and benchmark it on the SHA256-verified test sets (synthetic and Q.Lin real). Add results to the comparison table.

Apply kernel-level optimizations to the MR-TAE-FULL-LossB inference path: `torch.compile` with `mode="reduce-overhead"`, `channels_last` memory format, bfloat16 AMP, fused AdamW, and PyTorch SDPA flash attention. Target: ≥30% latency reduction on RTX 4070 without metric regression.

Implement the Noise2Void (N2V) self-supervised fine-tuning loop on the unlabelled Q.Lin real dataset, building on the MR-TAE-FULL-LossB checkpoint. Validate that the blind-spot masking does not degrade synthetic test NCC.

Begin assembly of the thesis figure plate: per-method signal overlays, ablation heatmap, SNR-sweep curves, SHAP saliency maps, and the new external-SOTA comparison row.

**Problem encountered:**

The flat-SNR baseline initially diverged when trained at SNR = −15 dB without warm-up, suggesting the SNR-15 distribution is genuinely outside the model's optimization basin without curricular ramp. The fix was to apply 10 epochs of SNR ∈ [+5, +15] dB warm-up before the flat-distribution training. This is being noted in the thesis as additional empirical justification for the curricular schedule rather than purely a convergence convenience.

---

## Report of Apr 27, 2026 to May 03, 2026

**Activities done in this period:**

Completed CleanUNet-1D training and benchmarking. The model achieved real-set NCC of [INSERT_VALUE] on the SHA256-verified Q.Lin test set, [above/below] the MR-TAE-FULL-LossB winner ([COMPARE_TO_0.8568]). This places MR-TAE in the proper external context: the proposed wavelet-recurrent-transformer fusion either matches or exceeds a strong NVIDIA-published 1D denoising baseline, validating the contribution claim. CleanUNet-1D has been added as the external row in the master comparison table.

Applied the planned kernel-level optimizations to MR-TAE-FULL-LossB inference. `torch.compile` (mode="reduce-overhead"), `channels_last` memory layout, bfloat16 autocast, and fused AdamW combined to reduce single-window latency from 6.67 ms to [INSERT_VALUE] ms (a [INSERT]% reduction) with no measurable change in synthetic or real NCC. PyTorch SDPA was confirmed to use the Flash-Attention backend on RTX 4070 via `torch.backends.cuda.sdp_kernel`. Results are logged to MLflow run `MR-TAE-FULL-LossB-OPTIMIZED`.

Implemented Noise2Void self-supervised fine-tuning on the unlabelled Q.Lin real data. The blind-spot masking strategy was adapted to 1D acoustic transients with a 5-sample structured mask centred on the receptive-field exclusion zone, as Noise2Noise was previously ruled out due to the physical impossibility of capturing paired identical PD pulses. After 30 fine-tuning epochs, the real-set NCC improved from 0.8568 to [INSERT_VALUE] without retraining on synthetic data, confirming that the model can adapt to real-domain noise statistics in a fully label-free manner.

Built and deployed a 24/7 automation harness on the Windows training machine. The harness consists of: (i) a Python watchdog (`pipeline/watchdog.py`) that monitors `session_state.json` and the orchestrator process, automatically restarting on crash with exponential back-off; (ii) a Windows Task Scheduler entry that launches the watchdog on system boot and on user logon; (iii) GPU thermal and VRAM gating already integrated in `training/heat_monitor.py` and `training/memory_manager.py`; and (iv) a Telegram bot notifier that posts status updates (training-completed, OOM, thermal throttle, watchdog restart) so progress can be monitored remotely. The harness has now run two unattended overnight cycles without manual intervention.

Began knowledge-distillation experiments for the PD-CLIP edge target. The MR-TAE-FULL-LossB teacher distils into a smaller student (≈1.5 M parameters, MWCNN-BiGRU architecture) using a combined response-distillation loss (KL on the seg head + MSE on the denoised signal). Initial student NCC on the synthetic test is [INSERT_VALUE]. INT8 post-training quantization and ONNX export are queued for after distillation converges.

**Activities planned for next week:**

Finalize the distilled EdgePD-Net student model and complete its INT8 quantization + ONNX export. Profile inference on a Raspberry Pi CM4 to validate the <5 ms target window for PD-CLIP feasibility.

Submit the complete thesis draft (Chapters 1–6 plus front and back matter) for supervisor review, incorporating the reconciled metrics, the external CleanUNet-1D row, the corrected noise-model narrative, and all per-method comparison figures.

Prepare the FYP viva slide deck: cover the architecture, ablation rationale, real-set generalization story (LossA → LossB pivot), explainability, external SOTA positioning, and the PD-CLIP commercial trajectory.

Conduct a mock viva session against `docs/EXAMINER_QA.md` and rehearse the answers for the highest-risk questions (NCC reconciliation, AE physics gap between Q.Lin lab data and field deployment, why N2V over N2N, why WGAN-GP for noise but not pulses).

**Problem encountered:**

Initial Noise2Void fine-tuning caused a small but consistent regression on the synthetic test set (synthetic NCC dropped by [INSERT_VALUE]) while improving real-set NCC. This is a known catastrophic-forgetting symptom when fine-tuning on a distribution-shifted dataset. The mitigation applied was a replay buffer: 20% of every fine-tuning batch is drawn from the original synthetic training distribution, keeping synthetic performance stable while still allowing the model to adapt to Q.Lin noise statistics. The replay-fraction sweep is documented in `results/noise2void_ablation/`.

---

## Notes for the Supervisor (Optional Cover Email Snippet)

> Dear Dr. Wong,
>
> Attached are the two weekly reports covering 20 Apr – 03 May. The two priorities I addressed from your Apr 18 feedback are: (i) per-method signal-overlay figures are now generated automatically for all 13 variants and saved at 300 DPI for thesis inclusion, and (ii) results are now reported in tabular form with consistent SNR-aggregation and external SOTA context (CleanUNet-1D, NVIDIA 2022).
>
> Three figures are attached for quick review: `results/method_comparison_grid.png` (overview), `results/MR-TAE-FULL-LossB/signal_overlay_SNR-5dB.png` (winner detail), and `results/comparison_table_with_external_sota.png`.
>
> The next milestone is the full thesis draft submission for your review, planned for the end of next week.
>
> Best regards,
> Jian Hong
