# CODEBASE AUDIT (Mandatory Pre-Generation)

Date: 2026-04-10  
Scope audited: `FULL_RESEARCH_PLAN.md`, `AGENT_PLAN.md`, and current implementation under `Try Model/mr_tae_fusion/`

## Critical Finding

**CRITICAL CORRECTION REQUIRED:** Current synthetic signal/noise implementation uses **GHz sampling** and **MHz carrier/interference frequencies**, which are inconsistent with AE-PD target physics (20-500 kHz band).

---

## 1) Sampling rate in `distance_signals.py`

File: `Try Model/mr_tae_fusion/data/distance_signals.py`

- `DistanceSignalConfig.sample_rate = 1000e6`  
- This is **1,000,000,000 Hz (1 GHz)**, **not 1,000,000 Hz (1 MHz)**.

Answer: **No, fs is not 1 MHz. It is currently 1 GHz.**

---

## 2) Actual `fc` values by pulse type (and units)

Primary file: `Try Model/mr_tae_fusion/data/pulse_generators.py`

- Type A: `fc = 25e6 + rand * 10e6` -> **25-35 MHz**
- Type B: no explicit `fc` (biphasic spikes only)
- Type C: no explicit `fc` (spikes/short decay construction)
- Type D: no explicit fixed `fc` (sine-by-index tail, not physical Hz-parametric carrier)
- Type E:
  - `fc1 = 30e6 + rand * 50e6` -> **30-80 MHz**
  - `fc2 = 60e6 + rand * 60e6` -> **60-120 MHz**
  - `fc3 = 100e6 + rand * 50e6` -> **100-150 MHz**
- Type F:
  - `fc1 = 30e6 + rand * 50e6` -> **30-80 MHz**
  - `fc2 = 60e6 + rand * 60e6` -> **60-120 MHz**
  - `fc3 = 100e6 + rand * 50e6` -> **100-150 MHz**
- Type G (treeing): `fc = 40e6 + rand * 80e6` -> **40-120 MHz**

Additional file: `Try Model/mr_tae_fusion/data/distance_signals.py`

- oscillating pulse branch: `fc = uniform(50e6, 200e6)` -> **50-200 MHz**

Answer: **Current `fc` values are in MHz, not kHz.**

---

## 3) Confirm training window size (is it 2048?)

Evidence from code:

- `Try Model/mr_tae_fusion/config.py` -> `signal_length = 2001`
- `Try Model/mr_tae_fusion/data/distance_signals.py` -> `signal_length = 2001`
- `run_training.py`, `train_extended.py`, `compare_methods.py`, `mr_tae_ultimate.py` all wired to `signal_length=2001`

Answer: **No. Current implemented window size is 2001 samples, not 2048.**

---

## 4) Framework audit (TensorFlow vs PyTorch)

### `import tensorflow` occurrences

- **None found** in `*.py` files.

### `import torch` / `from torch` occurrences

PyTorch is used broadly across the codebase, including:

- `Try Model/examples/train_extended.py`
- `Try Model/examples/train_tgan.py`
- `train_extended.py`
- `compare_methods.py`
- `mr_tae_ultimate.py`
- `Try Model/examples/train_demo.py`
- `Try Model/examples/train_and_visualize.py`
- `Try Model/examples/train_extended_500.py`
- `Try Model/examples/train_improved.py`
- `Try Model/benchmark_eval.py`
- `run_training.py`
- `Try Model/mr_tae_fusion/models/distance_classifier.py`
- `Try Model/mr_tae_fusion/data/pulse_generators.py`
- `Try Model/mr_tae_fusion/config.py`
- `Try Model/examples/eval_improved.py`
- `Try Model/examples/quick_eval.py`
- `Try Model/examples/analyze_segmentation.py`
- `Try Model/examples/evaluate_results.py`
- `Try Model/mr_tae_fusion/models/mr_tae_fusion.py`
- `Try Model/test_sota.py`
- `Try Model/examples/run_inference_for_thesis.py`
- `Try Model/mr_tae_fusion/models/tgan.py`
- `Try Model/mr_tae_fusion/training/losses.py`
- `Try Model/test_integration.py`
- `Try Model/mr_tae_fusion/models/mwcnn.py`
- `Try Model/test_forward.py`
- `Try Model/tests/test_data.py`
- `Try Model/tests/test_model.py`
- `Try Model/tests/test_wavelets.py`
- `Try Model/mr_tae_fusion/evaluation/visualization.py`
- `Try Model/mr_tae_fusion/evaluation/metrics.py`
- `Try Model/mr_tae_fusion/training/trainer.py`
- `Try Model/mr_tae_fusion/models/fusion_head.py`
- `Try Model/mr_tae_fusion/models/swin_transformer.py`
- `Try Model/mr_tae_fusion/models/attention.py`
- `Try Model/mr_tae_fusion/models/wavelet_layers.py`
- `Try Model/mr_tae_fusion/data/dataset.py`
- `Try Model/mr_tae_fusion/data/noise_generators.py`

Answer: **No mixed TF/PyTorch usage detected in Python sources; implementation is PyTorch-only at present.**

---

## 5) `type_to_class` mapping (A-F/G + class labels)

File: `Try Model/mr_tae_fusion/config.py`

```python
type_to_class = {
    'A': 1,      # Sparse PD -> Corona
    'B': 2,      # Spike-dense -> Surface
    'C': 3,      # 10mm -> Internal
    'D': 3,      # 18mm -> Internal
    'E': 3,      # 20mm -> Internal
    'F': 3,      # 25mm -> Internal
    'G': 4,      # Treeing -> Treeing
    'MIXED': 0,  # Mixed signals
}
```

Class IDs implied by comments:

- `0`: Background/Mixed label handling
- `1`: Corona
- `2`: Surface
- `3`: Internal
- `4`: Treeing

---

## Frequency-band verification from external references

Query run: `acoustic emission PD detection frequency range piezoelectric sensor IEEE IEC`

Findings (high-level):

- AE PD monitoring literature and IEEE C57.127 context consistently place useful AE sensing in the **ultrasonic kHz region** (commonly ~20-300 kHz, with many sensors centered around 60-150 kHz).
- This supports the research plan correction that AE-PD synthesis should be in **sub-MHz acoustic band**, and not in MHz/UHF electrical-PD ranges.

Conclusion:

- Using **20-500 kHz** as a conservative AE design envelope is consistent with the intended AE domain; practical dominant content often appears in a narrower subset (e.g., 30-150 kHz depending on sensor/system).

