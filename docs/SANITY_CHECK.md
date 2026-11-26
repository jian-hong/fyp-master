# Pre-Flight Sanity Checks

**Date Verified:** 2026-04-16

This document details the first phase of the plan: auditing the model output layers and the core evaluation metrics to ensure they are physically meaningful for Acoustic Emission (AE) partial discharge signals.

## 1. Output Activation Audit
**Finding**: The `ReconstructionHead` in `mr_tae_fusion/models/fusion_head.py` uses an `nn.Conv1d` output layer with no trailing activation function (i.e. it acts as `nn.Identity`). 
**Result**: **PASS**. Since AE signals are strictly zero-mean, continuous, bipolar waveforms, it is entirely correct to predict unbounded linear values using `Identity`, allowing the optimizer to learn the residual/scale exactly.

## 2. NCC Metric Computation Audit
**Finding**: The `NCC` function previously computed cross-correlation over the entire $N=2048$ window. If a PD pulse occupies only $200$ samples out of $2048$, the majority of the window is flat white noise. Correlating noise heavily inflates/deflates the numerator and denominator arbitrarily, corrupting the actual metric.
**Fix Implemented**:
- `calculate_ncc` was updated to accept an optional `mask` tensor.
- Added `ncc_pulse` to isolate the calculation exactly over the bounding box where `mask > 0`. 
- `evaluate_denoising` now returns `ncc_global` for transparency and overrides the primary `ncc` key to use `ncc_pulse` if a mask is provided.

**Result**: **PASS**. The pipeline now evaluates model reconstruction strictly based on the actual PD transient signatures.
