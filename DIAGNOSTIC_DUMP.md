# DIAGNOSTIC_DUMP

Forensic audit only. The only file written by this pass is this markdown report.

## 1. CHECKPOINT INVENTORY

Command basis: equivalent of `dir /s /b results\*best.pt` from `D:\New folder (2)\Data\FYP_MASTER`.

### baseline_unet

- Path: `results/baseline_unet/best.pt`
- File size: `10903301` bytes
- Last modified: `2026-04-16 02:01:50`

- `list(ckpt.keys())`: `['encoder.0.0.weight', 'encoder.0.0.bias', 'encoder.0.1.weight', 'encoder.0.1.bias', 'encoder.0.1.running_mean', 'encoder.0.1.running_var', 'encoder.0.1.num_batches_tracked', 'encoder.0.3.weight', 'encoder.0.3.bias', 'encoder.0.4.weight', 'encoder.0.4.bias', 'encoder.0.4.running_mean', 'encoder.0.4.running_var', 'encoder.0.4.num_batches_tracked', 'encoder.1.0.weight', 'encoder.1.0.bias', 'encoder.1.1.weight', 'encoder.1.1.bias', 'encoder.1.1.running_mean', 'encoder.1.1.running_var', 'encoder.1.1.num_batches_tracked', 'encoder.1.3.weight', 'encoder.1.3.bias', 'encoder.1.4.weight', 'encoder.1.4.bias', 'encoder.1.4.running_mean', 'encoder.1.4.running_var', 'encoder.1.4.num_batches_tracked', 'encoder.2.0.weight', 'encoder.2.0.bias', 'encoder.2.1.weight', 'encoder.2.1.bias', 'encoder.2.1.running_mean', 'encoder.2.1.running_var', 'encoder.2.1.num_batches_tracked', 'encoder.2.3.weight', 'encoder.2.3.bias', 'encoder.2.4.weight', 'encoder.2.4.bias', 'encoder.2.4.running_mean', 'encoder.2.4.running_var', 'encoder.2.4.num_batches_tracked', 'encoder.3.0.weight', 'encoder.3.0.bias', 'encoder.3.1.weight', 'encoder.3.1.bias', 'encoder.3.1.running_mean', 'encoder.3.1.running_var', 'encoder.3.1.num_batches_tracked', 'encoder.3.3.weight', 'encoder.3.3.bias', 'encoder.3.4.weight', 'encoder.3.4.bias', 'encoder.3.4.running_mean', 'encoder.3.4.running_var', 'encoder.3.4.num_batches_tracked', 'decoder.0.weight', 'decoder.0.bias', 'decoder.1.0.weight', 'decoder.1.0.bias', 'decoder.1.1.weight', 'decoder.1.1.bias', 'decoder.1.1.running_mean', 'decoder.1.1.running_var', 'decoder.1.1.num_batches_tracked', 'decoder.1.3.weight', 'decoder.1.3.bias', 'decoder.1.4.weight', 'decoder.1.4.bias', 'decoder.1.4.running_mean', 'decoder.1.4.running_var', 'decoder.1.4.num_batches_tracked', 'decoder.2.weight', 'decoder.2.bias', 'decoder.3.0.weight', 'decoder.3.0.bias', 'decoder.3.1.weight', 'decoder.3.1.bias', 'decoder.3.1.running_mean', 'decoder.3.1.running_var', 'decoder.3.1.num_batches_tracked', 'decoder.3.3.weight', 'decoder.3.3.bias', 'decoder.3.4.weight', 'decoder.3.4.bias', 'decoder.3.4.running_mean', 'decoder.3.4.running_var', 'decoder.3.4.num_batches_tracked', 'decoder.4.weight', 'decoder.4.bias', 'decoder.5.0.weight', 'decoder.5.0.bias', 'decoder.5.1.weight', 'decoder.5.1.bias', 'decoder.5.1.running_mean', 'decoder.5.1.running_var', 'decoder.5.1.num_batches_tracked', 'decoder.5.3.weight', 'decoder.5.3.bias', 'decoder.5.4.weight', 'decoder.5.4.bias', 'decoder.5.4.running_mean', 'decoder.5.4.running_var', 'decoder.5.4.num_batches_tracked', 'decoder.6.weight', 'decoder.6.bias', 'decoder.7.0.weight', 'decoder.7.0.bias', 'decoder.7.1.weight', 'decoder.7.1.bias', 'decoder.7.1.running_mean', 'decoder.7.1.running_var', 'decoder.7.1.num_batches_tracked', 'decoder.7.3.weight', 'decoder.7.3.bias', 'decoder.7.4.weight', 'decoder.7.4.bias', 'decoder.7.4.running_mean', 'decoder.7.4.running_var', 'decoder.7.4.num_batches_tracked', 'bottleneck.0.weight', 'bottleneck.0.bias', 'bottleneck.1.weight', 'bottleneck.1.bias', 'bottleneck.1.running_mean', 'bottleneck.1.running_var', 'bottleneck.1.num_batches_tracked', 'bottleneck.3.weight', 'bottleneck.3.bias', 'bottleneck.4.weight', 'bottleneck.4.bias', 'bottleneck.4.running_mean', 'bottleneck.4.running_var', 'bottleneck.4.num_batches_tracked', 'final_conv.weight', 'final_conv.bias']`
- State source inspected: `<checkpoint itself>`

- First 10 parameter/tensor keys and shapes:

  - `encoder.0.0.weight`: `(32, 1, 3)`

  - `encoder.0.0.bias`: `(32,)`

  - `encoder.0.1.weight`: `(32,)`

  - `encoder.0.1.bias`: `(32,)`

  - `encoder.0.1.running_mean`: `(32,)`

  - `encoder.0.1.running_var`: `(32,)`

  - `encoder.0.1.num_batches_tracked`: `()`

  - `encoder.0.3.weight`: `(32, 32, 3)`

  - `encoder.0.3.bias`: `(32,)`

  - `encoder.0.4.weight`: `(32,)`



### MR-TAE-FULL

- Path: `results/MR-TAE-FULL/checkpoints/best.pt`
- File size: `36321996` bytes
- Last modified: `2026-05-10 17:14:01`

- `list(ckpt.keys())`: `['epoch', 'model_state', 'optimizer_state', 'scheduler_state', 'criterion_state', 'scaler_state', 'best_val_ncc', 'history', 'model_id']`
- State source inspected: `model_state`

- First 10 parameter/tensor keys and shapes:

  - `input_proj.weight`: `(32, 1, 7)`

  - `input_proj.bias`: `(32,)`

  - `encoder.blocks.0.conv1.conv.weight`: `(32, 32, 3)`

  - `encoder.blocks.0.conv1.conv.bias`: `(32,)`

  - `encoder.blocks.0.conv1.bn.weight`: `(32,)`

  - `encoder.blocks.0.conv1.bn.bias`: `(32,)`

  - `encoder.blocks.0.conv1.bn.running_mean`: `(32,)`

  - `encoder.blocks.0.conv1.bn.running_var`: `(32,)`

  - `encoder.blocks.0.conv1.bn.num_batches_tracked`: `()`

  - `encoder.blocks.0.conv2.conv.weight`: `(32, 32, 3)`



### MR-TAE-FULL-LossB

- Path: `results/MR-TAE-FULL-LossB/checkpoints/best.pt`
- File size: `36322484` bytes
- Last modified: `2026-05-10 02:59:58`

- `list(ckpt.keys())`: `['epoch', 'model_state', 'optimizer_state', 'scheduler_state', 'criterion_state', 'scaler_state', 'best_val_ncc', 'history', 'model_id']`
- State source inspected: `model_state`

- First 10 parameter/tensor keys and shapes:

  - `input_proj.weight`: `(32, 1, 7)`

  - `input_proj.bias`: `(32,)`

  - `encoder.blocks.0.conv1.conv.weight`: `(32, 32, 3)`

  - `encoder.blocks.0.conv1.conv.bias`: `(32,)`

  - `encoder.blocks.0.conv1.bn.weight`: `(32,)`

  - `encoder.blocks.0.conv1.bn.bias`: `(32,)`

  - `encoder.blocks.0.conv1.bn.running_mean`: `(32,)`

  - `encoder.blocks.0.conv1.bn.running_var`: `(32,)`

  - `encoder.blocks.0.conv1.bn.num_batches_tracked`: `()`

  - `encoder.blocks.0.conv2.conv.weight`: `(32, 32, 3)`



### MR-TAE-FULL-LossC

- Path: `results/MR-TAE-FULL-LossC/checkpoints/best.pt`
- File size: `36465269` bytes
- Last modified: `2026-04-16 05:28:35`

- `list(ckpt.keys())`: `['epoch', 'model_state', 'optimizer_state', 'scheduler_state', 'criterion_state', 'scaler_state', 'best_val_ncc', 'history', 'model_id']`
- State source inspected: `model_state`

- First 10 parameter/tensor keys and shapes:

  - `input_proj.weight`: `(32, 1, 7)`

  - `input_proj.bias`: `(32,)`

  - `encoder.blocks.0.conv1.conv.weight`: `(32, 32, 3)`

  - `encoder.blocks.0.conv1.conv.bias`: `(32,)`

  - `encoder.blocks.0.conv1.bn.weight`: `(32,)`

  - `encoder.blocks.0.conv1.bn.bias`: `(32,)`

  - `encoder.blocks.0.conv1.bn.running_mean`: `(32,)`

  - `encoder.blocks.0.conv1.bn.running_var`: `(32,)`

  - `encoder.blocks.0.conv1.bn.num_batches_tracked`: `()`

  - `encoder.blocks.0.conv2.conv.weight`: `(32, 32, 3)`



### MR-TAE-noAttn

- Path: `results/MR-TAE-noAttn/checkpoints/best.pt`
- File size: `36324620` bytes
- Last modified: `2026-04-16 03:40:28`

- `list(ckpt.keys())`: `['epoch', 'model_state', 'optimizer_state', 'scheduler_state', 'criterion_state', 'scaler_state', 'best_val_ncc', 'history', 'model_id']`
- State source inspected: `model_state`

- First 10 parameter/tensor keys and shapes:

  - `input_proj.weight`: `(32, 1, 7)`

  - `input_proj.bias`: `(32,)`

  - `encoder.blocks.0.conv1.conv.weight`: `(32, 32, 3)`

  - `encoder.blocks.0.conv1.conv.bias`: `(32,)`

  - `encoder.blocks.0.conv1.bn.weight`: `(32,)`

  - `encoder.blocks.0.conv1.bn.bias`: `(32,)`

  - `encoder.blocks.0.conv1.bn.running_mean`: `(32,)`

  - `encoder.blocks.0.conv1.bn.running_var`: `(32,)`

  - `encoder.blocks.0.conv1.bn.num_batches_tracked`: `()`

  - `encoder.blocks.0.conv2.conv.weight`: `(32, 32, 3)`



### MR-TAE-noBiGRU

- Path: `results/MR-TAE-noBiGRU/checkpoints/best.pt`
- File size: `28960284` bytes
- Last modified: `2026-04-16 03:02:40`

- `list(ckpt.keys())`: `['epoch', 'model_state', 'optimizer_state', 'scheduler_state', 'criterion_state', 'scaler_state', 'best_val_ncc', 'history', 'model_id']`
- State source inspected: `model_state`

- First 10 parameter/tensor keys and shapes:

  - `input_proj.weight`: `(32, 1, 7)`

  - `input_proj.bias`: `(32,)`

  - `encoder.blocks.0.conv1.conv.weight`: `(32, 32, 3)`

  - `encoder.blocks.0.conv1.conv.bias`: `(32,)`

  - `encoder.blocks.0.conv1.bn.weight`: `(32,)`

  - `encoder.blocks.0.conv1.bn.bias`: `(32,)`

  - `encoder.blocks.0.conv1.bn.running_mean`: `(32,)`

  - `encoder.blocks.0.conv1.bn.running_var`: `(32,)`

  - `encoder.blocks.0.conv1.bn.num_batches_tracked`: `()`

  - `encoder.blocks.0.conv2.conv.weight`: `(32, 32, 3)`



### MR-TAE-noBiGRU-noAttn

- Path: `results/MR-TAE-noBiGRU-noAttn/checkpoints/best.pt`
- File size: `28405545` bytes
- Last modified: `2026-04-16 02:05:24`

- `list(ckpt.keys())`: `['epoch', 'model_state', 'optimizer_state', 'scheduler_state', 'criterion_state', 'scaler_state', 'best_val_ncc', 'history', 'model_id']`
- State source inspected: `model_state`

- First 10 parameter/tensor keys and shapes:

  - `input_proj.weight`: `(32, 1, 7)`

  - `input_proj.bias`: `(32,)`

  - `encoder.blocks.0.conv1.conv.weight`: `(32, 32, 3)`

  - `encoder.blocks.0.conv1.conv.bias`: `(32,)`

  - `encoder.blocks.0.conv1.bn.weight`: `(32,)`

  - `encoder.blocks.0.conv1.bn.bias`: `(32,)`

  - `encoder.blocks.0.conv1.bn.running_mean`: `(32,)`

  - `encoder.blocks.0.conv1.bn.running_var`: `(32,)`

  - `encoder.blocks.0.conv1.bn.num_batches_tracked`: `()`

  - `encoder.blocks.0.conv2.conv.weight`: `(32, 32, 3)`



### MR-TAE-noMTL

- Path: `results/MR-TAE-noMTL/checkpoints/best.pt`
- File size: `36254731` bytes
- Last modified: `2026-04-16 04:44:23`

- `list(ckpt.keys())`: `['epoch', 'model_state', 'optimizer_state', 'scheduler_state', 'criterion_state', 'scaler_state', 'best_val_ncc', 'history', 'model_id']`
- State source inspected: `model_state`

- First 10 parameter/tensor keys and shapes:

  - `input_proj.weight`: `(32, 1, 7)`

  - `input_proj.bias`: `(32,)`

  - `encoder.blocks.0.conv1.conv.weight`: `(32, 32, 3)`

  - `encoder.blocks.0.conv1.conv.bias`: `(32,)`

  - `encoder.blocks.0.conv1.bn.weight`: `(32,)`

  - `encoder.blocks.0.conv1.bn.bias`: `(32,)`

  - `encoder.blocks.0.conv1.bn.running_mean`: `(32,)`

  - `encoder.blocks.0.conv1.bn.running_var`: `(32,)`

  - `encoder.blocks.0.conv1.bn.num_batches_tracked`: `()`

  - `encoder.blocks.0.conv2.conv.weight`: `(32, 32, 3)`



### MR-TAE-noSwin

- Path: `results/MR-TAE-noSwin/checkpoints/best.pt`
- File size: `17070497` bytes
- Last modified: `2026-04-16 03:23:22`

- `list(ckpt.keys())`: `['epoch', 'model_state', 'optimizer_state', 'scheduler_state', 'criterion_state', 'scaler_state', 'best_val_ncc', 'history', 'model_id']`
- State source inspected: `model_state`

- First 10 parameter/tensor keys and shapes:

  - `input_proj.weight`: `(32, 1, 7)`

  - `input_proj.bias`: `(32,)`

  - `encoder.blocks.0.conv1.conv.weight`: `(32, 32, 3)`

  - `encoder.blocks.0.conv1.conv.bias`: `(32,)`

  - `encoder.blocks.0.conv1.bn.weight`: `(32,)`

  - `encoder.blocks.0.conv1.bn.bias`: `(32,)`

  - `encoder.blocks.0.conv1.bn.running_mean`: `(32,)`

  - `encoder.blocks.0.conv1.bn.running_var`: `(32,)`

  - `encoder.blocks.0.conv1.bn.num_batches_tracked`: `()`

  - `encoder.blocks.0.conv2.conv.weight`: `(32, 32, 3)`



### MR-TAE-noSwin-noAttn

- Path: `results/MR-TAE-noSwin-noAttn/checkpoints/best.pt`
- File size: `16515310` bytes
- Last modified: `2026-04-15 03:49:06`

- `list(ckpt.keys())`: `['epoch', 'model_state', 'optimizer_state', 'scheduler_state', 'criterion_state', 'scaler_state', 'best_val_ncc', 'history', 'model_id']`
- State source inspected: `model_state`

- First 10 parameter/tensor keys and shapes:

  - `input_proj.weight`: `(32, 1, 7)`

  - `input_proj.bias`: `(32,)`

  - `encoder.blocks.0.conv1.conv.weight`: `(32, 32, 3)`

  - `encoder.blocks.0.conv1.conv.bias`: `(32,)`

  - `encoder.blocks.0.conv1.bn.weight`: `(32,)`

  - `encoder.blocks.0.conv1.bn.bias`: `(32,)`

  - `encoder.blocks.0.conv1.bn.running_mean`: `(32,)`

  - `encoder.blocks.0.conv1.bn.running_var`: `(32,)`

  - `encoder.blocks.0.conv1.bn.num_batches_tracked`: `()`

  - `encoder.blocks.0.conv2.conv.weight`: `(32, 32, 3)`



### MR-TAE-noSwin-noBiGRU

- Path: `results/MR-TAE-noSwin-noBiGRU/checkpoints/best.pt`
- File size: `12320080` bytes
- Last modified: `2026-04-10 20:04:59`

- `list(ckpt.keys())`: `['epoch', 'model_state', 'optimizer_state', 'scheduler_state', 'criterion_state', 'scaler_state', 'best_val_ncc', 'history', 'model_id']`
- State source inspected: `model_state`

- First 10 parameter/tensor keys and shapes:

  - `input_proj.weight`: `(32, 1, 7)`

  - `input_proj.bias`: `(32,)`

  - `encoder.blocks.0.conv1.conv.weight`: `(32, 32, 3)`

  - `encoder.blocks.0.conv1.conv.bias`: `(32,)`

  - `encoder.blocks.0.conv1.bn.weight`: `(32,)`

  - `encoder.blocks.0.conv1.bn.bias`: `(32,)`

  - `encoder.blocks.0.conv1.bn.running_mean`: `(32,)`

  - `encoder.blocks.0.conv1.bn.running_var`: `(32,)`

  - `encoder.blocks.0.conv1.bn.num_batches_tracked`: `()`

  - `encoder.blocks.0.conv2.conv.weight`: `(32, 32, 3)`



### MR-TAE-noWavelet

- Path: `results/MR-TAE-noWavelet/checkpoints/best.pt`
- File size: `36102237` bytes
- Last modified: `2026-04-16 04:01:38`

- `list(ckpt.keys())`: `['epoch', 'model_state', 'optimizer_state', 'scheduler_state', 'criterion_state', 'scaler_state', 'best_val_ncc', 'history', 'model_id']`
- State source inspected: `model_state`

- First 10 parameter/tensor keys and shapes:

  - `input_proj.weight`: `(32, 1, 7)`

  - `input_proj.bias`: `(32,)`

  - `encoder.conv_blocks.0.0.conv.weight`: `(32, 32, 3)`

  - `encoder.conv_blocks.0.0.conv.bias`: `(32,)`

  - `encoder.conv_blocks.0.0.bn.weight`: `(32,)`

  - `encoder.conv_blocks.0.0.bn.bias`: `(32,)`

  - `encoder.conv_blocks.0.0.bn.running_mean`: `(32,)`

  - `encoder.conv_blocks.0.0.bn.running_var`: `(32,)`

  - `encoder.conv_blocks.0.0.bn.num_batches_tracked`: `()`

  - `encoder.conv_blocks.0.1.conv.weight`: `(32, 32, 3)`



### MWCNN-BiGRU

- Path: `results/MWCNN-BiGRU/checkpoints/best.pt`
- File size: `15889466` bytes
- Last modified: `2026-04-16 02:25:50`

- `list(ckpt.keys())`: `['epoch', 'model_state', 'optimizer_state', 'scheduler_state', 'criterion_state', 'scaler_state', 'best_val_ncc', 'history', 'model_id']`
- State source inspected: `model_state`

- First 10 parameter/tensor keys and shapes:

  - `input_proj.weight`: `(32, 1, 7)`

  - `input_proj.bias`: `(32,)`

  - `encoder.blocks.0.conv1.conv.weight`: `(32, 32, 3)`

  - `encoder.blocks.0.conv1.conv.bias`: `(32,)`

  - `encoder.blocks.0.conv1.bn.weight`: `(32,)`

  - `encoder.blocks.0.conv1.bn.bias`: `(32,)`

  - `encoder.blocks.0.conv1.bn.running_mean`: `(32,)`

  - `encoder.blocks.0.conv1.bn.running_var`: `(32,)`

  - `encoder.blocks.0.conv1.bn.num_batches_tracked`: `()`

  - `encoder.blocks.0.conv2.conv.weight`: `(32, 32, 3)`



### UNet-BiGRU-Swin

- Path: `results/UNet-BiGRU-Swin/checkpoints/best.pt`
- File size: `34922422` bytes
- Last modified: `2026-04-16 02:42:33`

- `list(ckpt.keys())`: `['epoch', 'model_state', 'optimizer_state', 'scheduler_state', 'criterion_state', 'scaler_state', 'best_val_ncc', 'history', 'model_id']`
- State source inspected: `model_state`

- First 10 parameter/tensor keys and shapes:

  - `input_proj.weight`: `(32, 1, 7)`

  - `input_proj.bias`: `(32,)`

  - `encoder.conv_blocks.0.0.conv.weight`: `(32, 32, 3)`

  - `encoder.conv_blocks.0.0.conv.bias`: `(32,)`

  - `encoder.conv_blocks.0.0.bn.weight`: `(32,)`

  - `encoder.conv_blocks.0.0.bn.bias`: `(32,)`

  - `encoder.conv_blocks.0.0.bn.running_mean`: `(32,)`

  - `encoder.conv_blocks.0.0.bn.running_var`: `(32,)`

  - `encoder.conv_blocks.0.0.bn.num_batches_tracked`: `()`

  - `encoder.conv_blocks.0.1.conv.weight`: `(32, 32, 3)`



### Specific Flags

- `MR-TAE-FULL-LossB` has workspace `best.pt`: `True`.

- `MR-TAE-FULL-LossB` architecture/state key set matches `MR-TAE-FULL`: `True`.

- Keys only in `MR-TAE-FULL-LossB`: `[]`.

- Keys only in `MR-TAE-FULL`: `[]`.

- `MR-TAE-noBiGRU` contains keys matching `*gru*`: `[]`.


### Script-Resolved Root Check

The NEW FOCUS scripts set `PROJECT_ROOT = Path(__file__).resolve().parents[3]`, which resolves to `D:\New folder (2)`, not `D:\New folder (2)\Data\FYP_MASTER`. Running the same checkpoint search from that root found:

- `D:\New folder (2)\results\MR-TAE-FULL\checkpoints\best.pt` | variant `MR-TAE-FULL` | `6830154` bytes | `2026-04-10 22:50:26`

- `D:\New folder (2)\results\MR-TAE-noAttn\checkpoints\best.pt` | variant `MR-TAE-noAttn` | `6692350` bytes | `2026-04-10 20:52:33`

- `D:\New folder (2)\results\MR-TAE-noBiGRU\checkpoints\best.pt` | variant `MR-TAE-noBiGRU` | `4628687` bytes | `2026-04-10 19:36:00`

- `D:\New folder (2)\results\MR-TAE-noMTL\checkpoints\best.pt` | variant `MR-TAE-noMTL` | `6815887` bytes | `2026-04-03 18:28:09`

- `D:\New folder (2)\results\MR-TAE-noSwin\checkpoints\best.pt` | variant `MR-TAE-noSwin` | `4439162` bytes | `2026-04-10 20:01:41`

- `D:\New folder (2)\results` does not contain `MR-TAE-FULL-LossB\checkpoints\best.pt`.

## 2. NEW FOCUS SCRIPT AUDIT

### Full Contents: generate_method_comparison_figures.py

Path: `NEW FOCUS/generate_method_comparison_figures.py` lines 1-535

```python
"""
Per-method signal-overlay figure generator.

Addresses supervisor feedback (Wong Jee Keen Raymond, 18 Apr 2026):
    "Putting your results in table form is good. But remember to also save
     some figures of each method as you run them. You will need to include
     some figures where you plot the original and denoised signals in the
     same graph for comparison."

Generates, for every model in the registry that has a benchmark checkpoint:
    1. A 4-panel comparison figure at three SNR levels (-15, -5, +5 dB):
       (a) Noisy input vs. clean ground truth
       (b) Denoised output vs. clean ground truth (overlay -- this is the
           key figure the supervisor asked for)
       (c) Residual (denoised - clean) on its own axis
       (d) Power spectral density of noisy / denoised / clean (log-log)
    2. A grid summary across all models on a single page.

Output:
    results/{model_name}/signal_overlay_SNR{XX}dB.png      (per model, per SNR)
    results/{model_name}/signal_overlay_psd_SNR{XX}dB.png  (PSD detail)
    results/method_comparison_grid.png                     (overview)
    results/method_comparison_grid_residuals.png           (residual overview)

Usage:
    python evaluation/generate_method_comparison_figures.py
    python evaluation/generate_method_comparison_figures.py --models MR-TAE-FULL-LossB MWCNN-BiGRU
    python evaluation/generate_method_comparison_figures.py --snr-levels -15 0 5 --window-idx 42

Drop into:
    FYP_MASTER/evaluation/generate_method_comparison_figures.py

Requires:
    matplotlib, numpy, scipy, torch, pyyaml
    The test set at data/test_synthetic_ae.pt with the SHA256 already in your repo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.signal import welch

# ---------------------------------------------------------------------------
# Project paths -- adjust here if your tree differs.
# ---------------------------------------------------------------------------
# This bundle lives under Data/FYP_MASTER/NEW FOCUS in this workspace, while
# the active training repo root is three levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
TEST_SET_SYNTH = PROJECT_ROOT / "data" / "test_synthetic_ae.pt"
TEST_SET_REAL = PROJECT_ROOT / "data" / "test_real_qlin.pt"
MODEL_REGISTRY = PROJECT_ROOT / "config" / "model_registry.yaml"
RESULTS_DIR = PROJECT_ROOT / "results"

# Sampling rate (project memory: AE band, fs = 1 MHz, 2048 samples = 2.048 ms)
FS_HZ = 1_000_000

# AE band of interest for highlighting on PSD plots (20 - 500 kHz)
AE_BAND_HZ = (20_000, 500_000)

# Default SNR breakpoints (matches your curricular phases)
DEFAULT_SNR_LEVELS = (-15, -5, 5)

# Style: thesis-friendly, no decorative emoji-isms, print-safe colours.
# Colour-blind-safe: blue (clean), orange (noisy), green (denoised), grey (residual).
COLOURS = {
    "clean":    "#1f77b4",
    "noisy":    "#ff7f0e",
    "denoised": "#2ca02c",
    "residual": "#7f7f7f",
}

PLT_RC = {
    "figure.dpi":           150,
    "savefig.dpi":          300,
    "font.size":            10,
    "axes.labelsize":       10,
    "axes.titlesize":       11,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "legend.frameon":       False,
    "lines.linewidth":      1.0,
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_test_set_hash(path: Path, expected_hash: str | None) -> None:
    """Refuse to run on a tampered test set. Lifted from your TEST_SET_HASHES.md."""
    if expected_hash is None:
        return
    actual = sha256_of(path)
    if actual != expected_hash:
        raise RuntimeError(
            f"Test-set hash mismatch for {path}\n"
            f"  expected: {expected_hash}\n"
            f"  actual:   {actual}\n"
            f"Refusing to generate figures from a tampered test set."
        )


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation between two 1D signals."""
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom == 0.0:
        return float("nan")
    return float((a * b).sum() / denom)


def snr_db(clean: np.ndarray, denoised: np.ndarray) -> float:
    """Output SNR (dB), where the noise is taken to be (denoised - clean)."""
    noise = denoised - clean
    p_clean = float(np.mean(clean * clean))
    p_noise = float(np.mean(noise * noise))
    if p_noise == 0:
        return float("inf")
    return 10.0 * np.log10(p_clean / p_noise)


def psd(signal: np.ndarray, fs: int = FS_HZ):
    """One-sided PSD via Welch. Uses a window appropriate for 2048-sample frames."""
    nperseg = min(512, len(signal))
    f, p = welch(signal, fs=fs, nperseg=nperseg, scaling="density")
    return f, p


def select_window_at_snr(
    test_set: dict, target_snr_db: int, window_idx: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pick a (noisy, clean) pair from the synthetic test set at the requested SNR level.

    Expected test-set format (matches your build_test_sets.py output):
        {
            "noisy":  Tensor[N, 1, 2048],
            "clean":  Tensor[N, 1, 2048],
            "snr_db": Tensor[N]    # int per-window SNR label
        }
    """
    snr = test_set.get("snr_db")
    if snr is None:
        snr = test_set.get("snr_labels")
    if snr is None:
        raise KeyError("Test set must include `snr_db` or `snr_labels`.")
    if isinstance(snr, torch.Tensor):
        snr = snr.cpu().numpy()
    mask = (snr == target_snr_db)
    if not mask.any():
        raise ValueError(
            f"No windows at SNR={target_snr_db} dB in test set. "
            f"Available: {sorted(set(int(x) for x in snr))}"
        )
    candidate_idx = np.where(mask)[0]
    chosen = candidate_idx[window_idx % len(candidate_idx)]
    noisy = test_set["noisy"][chosen].squeeze().cpu().numpy()
    clean = test_set["clean"][chosen].squeeze().cpu().numpy()
    return noisy, clean


# ---------------------------------------------------------------------------
# Plot primitives
# ---------------------------------------------------------------------------
def plot_per_method(
    model_name: str,
    snr_db_level: int,
    noisy: np.ndarray,
    clean: np.ndarray,
    denoised: np.ndarray,
    out_path: Path,
) -> None:
    """The 4-panel per-method figure the supervisor asked for."""
    t_us = np.arange(len(clean)) / FS_HZ * 1e6  # microseconds

    out_ncc = ncc(clean, denoised)
    out_snr = snr_db(clean, denoised)

    with plt.rc_context(PLT_RC):
        fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
        title = (
            f"{model_name}  -  SNR_in = {snr_db_level} dB   "
            f"-  NCC_out = {out_ncc:.3f}   -  SNR_out = {out_snr:.2f} dB"
        )
        fig.suptitle(title, y=0.99, fontsize=12)

        # (a) Noisy input overlay vs clean
        ax = axes[0, 0]
        ax.plot(t_us, noisy, color=COLOURS["noisy"], label="Noisy input", alpha=0.7)
        ax.plot(t_us, clean, color=COLOURS["clean"], label="Clean GT", linewidth=1.2)
        ax.set_xlabel("Time (us)")
        ax.set_ylabel("Amplitude")
        ax.set_title("(a) Noisy input vs. clean ground truth")
        ax.legend(loc="upper right")

        # (b) The key figure: denoised overlaid on clean
        ax = axes[0, 1]
        ax.plot(t_us, clean, color=COLOURS["clean"], label="Clean GT", linewidth=1.4)
        ax.plot(
            t_us, denoised, color=COLOURS["denoised"], label="Denoised", linewidth=1.0,
            alpha=0.9,
        )
        ax.set_xlabel("Time (us)")
        ax.set_ylabel("Amplitude")
        ax.set_title("(b) Denoised vs. clean ground truth")
        ax.legend(loc="upper right")

        # (c) Residual (denoised - clean): how badly did we do?
        ax = axes[1, 0]
        residual = denoised - clean
        ax.plot(t_us, residual, color=COLOURS["residual"], linewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.4, linestyle="--")
        ax.set_xlabel("Time (us)")
        ax.set_ylabel("Residual (denoised - clean)")
        ax.set_title(f"(c) Residual  -  RMS = {np.sqrt((residual**2).mean()):.4f}")

        # (d) PSD comparison on log-log axes -- shows whether AE-band content was preserved
        ax = axes[1, 1]
        for sig, label, colour in [
            (noisy, "Noisy", COLOURS["noisy"]),
            (clean, "Clean", COLOURS["clean"]),
            (denoised, "Denoised", COLOURS["denoised"]),
        ]:
            f, p = psd(sig)
            ax.loglog(f / 1e3, p, label=label, color=colour, linewidth=1.0)
        # Highlight AE band
        ax.axvspan(
            AE_BAND_HZ[0] / 1e3, AE_BAND_HZ[1] / 1e3,
            color="black", alpha=0.04, label="AE band 20-500 kHz",
        )
        ax.set_xlabel("Frequency (kHz)")
        ax.set_ylabel("PSD (V^2/Hz)")
        ax.set_title("(d) Power spectral density")
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(True, which="both", alpha=0.3)

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def plot_grid(
    rows: list[dict],
    snr_db_level: int,
    out_path: Path,
    show_residual: bool = False,
) -> None:
    """One-page grid summary -- one subplot per model, denoised overlaid on clean."""
    n = len(rows)
    cols = 4
    n_rows = (n + cols - 1) // cols

    with plt.rc_context(PLT_RC):
        fig, axes = plt.subplots(
            n_rows, cols, figsize=(cols * 3.5, n_rows * 2.2),
            sharex=True, sharey=True,
        )
        axes = np.atleast_2d(axes)
        fig.suptitle(
            f"{'Residuals' if show_residual else 'Denoised vs Clean'}  "
            f"--  SNR_in = {snr_db_level} dB   "
            f"--  one window from synthetic AE test set",
            fontsize=12, y=1.0,
        )

        t_us = np.arange(rows[0]["clean"].shape[-1]) / FS_HZ * 1e6
        for i, row in enumerate(rows):
            r, c = divmod(i, cols)
            ax = axes[r, c]
            if show_residual:
                resid = row["denoised"] - row["clean"]
                ax.plot(t_us, resid, color=COLOURS["residual"], linewidth=0.7)
                ax.axhline(0, color="black", linewidth=0.3, linestyle=":")
            else:
                ax.plot(t_us, row["clean"], color=COLOURS["clean"], linewidth=1.0)
                ax.plot(
                    t_us, row["denoised"], color=COLOURS["denoised"],
                    linewidth=0.8, alpha=0.9,
                )
            ax.set_title(
                f"{row['name']}\nNCC={row['ncc']:.3f} SNR_out={row['snr']:.1f} dB",
                fontsize=9,
            )
            ax.tick_params(labelsize=8)

        # Hide unused axes
        for j in range(n, n_rows * cols):
            r, c = divmod(j, cols)
            axes[r, c].set_visible(False)

        # Shared axis labels
        for ax in axes[-1, :]:
            ax.set_xlabel("Time (us)", fontsize=9)
        for ax in axes[:, 0]:
            ax.set_ylabel("Amplitude" if not show_residual else "Residual", fontsize=9)

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Model loading -- adapt this single function if your loader differs
# ---------------------------------------------------------------------------
def load_model(model_name: str, registry: dict, device: torch.device):
    """
    Load a model from your registry by name. Falls back to walking
    `results/{model_name}/checkpoints/best.pt` and instantiating from the
    AblationConfig stored in the checkpoint payload.

    This is intentionally tolerant of different checkpoint shapes -- adjust
    the keys to match what your trainer.py actually saves.
    """
    ckpt_candidates = [
        PROJECT_ROOT / "results" / model_name / "checkpoints" / "best.pt",
        PROJECT_ROOT / "results" / model_name / "best.pt",
        PROJECT_ROOT / "results" / model_name / "best.pth",
    ]
    ckpt_path = next((p for p in ckpt_candidates if p.exists()), None)
    if ckpt_path is None:
        return None, None

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model" in payload and isinstance(payload["model"], torch.nn.Module):
        # Already an instantiated nn.Module saved with torch.save(model, ...)
        model = payload["model"]
    elif "model" in payload and isinstance(payload["model"], dict):
        # Legacy project format where `model` stores a raw state_dict.
        from models.base import build_model_from_registry  # type: ignore
        with (PROJECT_ROOT / "config" / "training_config.yaml").open() as f:
            training_cfg = yaml.safe_load(f) or {}
        model = build_model_from_registry(model_name, {"models": registry}, training_cfg)
        model.load_state_dict(payload["model"])
    elif "state_dict" in payload and "ablation_config" in payload:
        # Preferred format -- your trainer should save these two keys.
        from mr_tae_fusion.models import build_model_from_config  # type: ignore
        cfg = payload["ablation_config"]
        model = build_model_from_config(cfg)
        model.load_state_dict(payload["state_dict"])
    elif "model_state" in payload:
        # Current project trainer format: model_state + model_id.
        from models.base import build_model_from_registry  # type: ignore
        with (PROJECT_ROOT / "config" / "training_config.yaml").open() as f:
            training_cfg = yaml.safe_load(f) or {}
        model = build_model_from_registry(model_name, {"models": registry}, training_cfg)
        model.load_state_dict(payload["model_state"])
    else:
        # Last resort -- build from local model registry and load best-effort state dict.
        from models.base import build_model_from_registry  # type: ignore
        with (PROJECT_ROOT / "config" / "training_config.yaml").open() as f:
            training_cfg = yaml.safe_load(f) or {}
        model = build_model_from_registry(model_name, {"models": registry}, training_cfg)
        state_dict = payload.get("state_dict", payload)
        model.load_state_dict(state_dict)

    model.eval().to(device)
    return model, ckpt_path


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="Subset of model names. Default = all in registry with checkpoints.",
    )
    parser.add_argument(
        "--snr-levels", nargs="+", type=int, default=list(DEFAULT_SNR_LEVELS),
        help=f"SNR-in dB levels to render. Default {DEFAULT_SNR_LEVELS}",
    )
    parser.add_argument(
        "--window-idx", type=int, default=0,
        help="Which window-at-this-SNR to use. Default 0 (first one).",
    )
    parser.add_argument(
        "--test-set", type=Path, default=TEST_SET_SYNTH,
        help="Path to synthetic test set .pt file.",
    )
    parser.add_argument(
        "--expected-hash", type=str, default=None,
        help="Optional SHA256 to verify the test set has not been tampered with.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--no-grid", action="store_true",
        help="Skip the cross-model grid summary figure.",
    )
    args = parser.parse_args(argv)

    # ---- Load registry ----
    if not MODEL_REGISTRY.exists():
        print(f"[fatal] Model registry not found at {MODEL_REGISTRY}", file=sys.stderr)
        return 1
    with MODEL_REGISTRY.open() as f:
        registry = yaml.safe_load(f) or {}
    if isinstance(registry, dict) and "models" in registry:
        registry = registry["models"]

    # Decide which models to render
    if args.models:
        model_names = args.models
    else:
        model_names = []
        for name in registry:
            if any(
                (PROJECT_ROOT / "results" / name / "checkpoints" / "best.pt").exists()
                or (PROJECT_ROOT / "results" / name / "best.pt").exists()
                or (PROJECT_ROOT / "results" / name / "best.pth").exists()
                for _ in [None]
            ):
                model_names.append(name)
        if not model_names:
            print(
                "[warn] No checkpoints found in results/. "
                "Pass --models explicitly to override.",
                file=sys.stderr,
            )
            return 1

    print(f"Rendering {len(model_names)} models at SNRs {args.snr_levels} dB")

    # ---- Load test set ----
    if args.expected_hash:
        verify_test_set_hash(args.test_set, args.expected_hash)
    test_set = torch.load(args.test_set, map_location="cpu", weights_only=False)
    n_windows = int(test_set.get("noisy").shape[0]) if "noisy" in test_set else -1
    print(f"Loaded test set: {args.test_set} ({n_windows} windows)")

    device = torch.device(args.device)

    # ---- For each SNR level, render per-method figures and a grid ----
    summary_rows = []
    for snr_level in args.snr_levels:
        try:
            noisy, clean = select_window_at_snr(test_set, snr_level, args.window_idx)
        except ValueError as e:
            print(f"[skip] {e}", file=sys.stderr)
            continue
        noisy_t = torch.from_numpy(noisy).float().view(1, 1, -1).to(device)

        grid_rows = []
        for name in model_names:
            model, ckpt_path = load_model(name, registry, device)
            if model is None:
                print(f"[skip] No checkpoint for {name}", file=sys.stderr)
                continue
            with torch.no_grad():
                out = model(noisy_t)
                # Tolerate dual-head MTL outputs.
                if isinstance(out, (tuple, list)):
                    out = out[0]
                elif isinstance(out, dict):
                    out = out.get("denoised") or out.get("signal") or next(iter(out.values()))
            denoised = out.squeeze().cpu().numpy()

            # Per-method figure
            per_path = (
                RESULTS_DIR / name / f"signal_overlay_SNR{snr_level:+d}dB.png"
            )
            plot_per_method(name, snr_level, noisy, clean, denoised, per_path)
            print(f"  wrote {per_path.relative_to(PROJECT_ROOT)}")

            grid_rows.append(
                dict(
                    name=name, clean=clean, noisy=noisy, denoised=denoised,
                    ncc=ncc(clean, denoised), snr=snr_db(clean, denoised),
                )
            )
            summary_rows.append(
                dict(
                    model=name, snr_in_db=snr_level,
                    ncc=ncc(clean, denoised),
                    snr_out_db=snr_db(clean, denoised),
                    rms_residual=float(np.sqrt(((denoised - clean) ** 2).mean())),
                    figure=str(per_path.relative_to(PROJECT_ROOT)),
                )
            )

            # Free the model for the next iteration
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Grid for this SNR
        if grid_rows and not args.no_grid:
            grid_path = (
                RESULTS_DIR / f"method_comparison_grid_SNR{snr_level:+d}dB.png"
            )
            plot_grid(grid_rows, snr_level, grid_path, show_residual=False)
            print(f"  wrote {grid_path.relative_to(PROJECT_ROOT)}")
            resid_path = (
                RESULTS_DIR / f"method_comparison_residuals_SNR{snr_level:+d}dB.png"
            )
            plot_grid(grid_rows, snr_level, resid_path, show_residual=True)
            print(f"  wrote {resid_path.relative_to(PROJECT_ROOT)}")

    # ---- Dump summary JSON for downstream use (RESULTS_COMPARISON.md, etc.) ----
    summary_path = RESULTS_DIR / "method_comparison_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"\nSummary written to {summary_path.relative_to(PROJECT_ROOT)}")
    print(f"Rendered {len(summary_rows)} (model x SNR) combinations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### Full Contents: compare_waveforms_with_baselines.py

Path: `NEW FOCUS/compare_waveforms_with_baselines.py` lines 1-159

```python
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.signal import savgol_filter

try:
    import pywt
except ImportError:
    pywt = None

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

TEST_SET = PROJECT_ROOT / "data" / "test_synthetic_ae.pt"
REGISTRY = PROJECT_ROOT / "config" / "model_registry.yaml"
TRAIN_CFG = PROJECT_ROOT / "config" / "training_config.yaml"
RESULTS = PROJECT_ROOT / "results"
FS_HZ = 1_000_000
SNR_LEVELS = (-15, -5, 5)


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else float("nan")


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def snr_out(clean: np.ndarray, pred: np.ndarray) -> float:
    noise = pred - clean
    p_sig = float(np.mean(clean * clean))
    p_noise = float(np.mean(noise * noise))
    return float(10.0 * np.log10(p_sig / max(p_noise, 1e-12)))


def wavelet_denoise(x: np.ndarray, wavelet: str = "db4") -> np.ndarray:
    if pywt is None:
        return x.copy()
    coeffs = pywt.wavedec(x, wavelet, mode="symmetric")
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    out = [coeffs[0]]
    out.extend(pywt.threshold(c, value=uthresh, mode="soft") for c in coeffs[1:])
    rec = pywt.waverec(out, wavelet, mode="symmetric")
    return rec[: len(x)]


def savgol_denoise(x: np.ndarray) -> np.ndarray:
    return savgol_filter(x, window_length=51, polyorder=3, mode="interp")


def load_model(model_id: str, registry: dict, device: torch.device):
    from models.base import build_model_from_registry
    with TRAIN_CFG.open() as f:
        tcfg = yaml.safe_load(f) or {}
    model = build_model_from_registry(model_id, {"models": registry}, tcfg).to(device).eval()
    ckpt = torch.load(RESULTS / model_id / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    if "model_state" in ckpt:
        state = ckpt["model_state"]
    elif isinstance(ckpt.get("model"), dict):
        state = ckpt["model"]
    else:
        state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)
    return model


def main() -> int:
    with REGISTRY.open() as f:
        registry = (yaml.safe_load(f) or {}).get("models", {})
    model_ids = [m for m in registry if (RESULTS / m / "checkpoints" / "best.pt").exists()]

    ds = torch.load(TEST_SET, map_location="cpu", weights_only=False)
    snr_labels = ds.get("snr_labels", ds.get("snr_db")).cpu().numpy().ravel()
    noisy_all = ds["noisy"]
    clean_all = ds["clean"]

    device = torch.device("cpu")
    rows = []
    for snr in SNR_LEVELS:
        idx = int(np.where(snr_labels == snr)[0][0])
        noisy = noisy_all[idx].squeeze().numpy()
        clean = clean_all[idx].squeeze().numpy()

        b_wave = wavelet_denoise(noisy)
        b_sg = savgol_denoise(noisy)
        baseline_rows = [
            {"name": "wavelet", "pred": b_wave},
            {"name": "savgol", "pred": b_sg},
        ]
        for b in baseline_rows:
            b["ncc"] = ncc(clean, b["pred"])
            b["rmse"] = rmse(clean, b["pred"])
            b["snr_out"] = snr_out(clean, b["pred"])
        best_baseline = max(baseline_rows, key=lambda x: x["ncc"])

        model_rows = []
        x = torch.from_numpy(noisy).float().view(1, 1, -1).to(device)
        for mid in model_ids:
            model = load_model(mid, registry, device)
            with torch.no_grad():
                out = model(x)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                pred = out.squeeze().cpu().numpy()
            model_rows.append(
                {"name": mid, "pred": pred, "ncc": ncc(clean, pred), "rmse": rmse(clean, pred), "snr_out": snr_out(clean, pred)}
            )
            del model
        best_model = max(model_rows, key=lambda x: x["ncc"])

        t_us = np.arange(clean.shape[-1]) / FS_HZ * 1e6
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t_us, clean, label="Clean GT", linewidth=1.4)
        ax.plot(t_us, noisy, label="Noisy", alpha=0.55)
        ax.plot(t_us, best_model["pred"], label=f"Best model: {best_model['name']}", linewidth=1.2)
        ax.plot(t_us, best_baseline["pred"], label=f"Best baseline: {best_baseline['name']}", linewidth=1.1)
        ax.set_title(f"SNR {snr} dB | model NCC={best_model['ncc']:.3f} vs baseline NCC={best_baseline['ncc']:.3f}")
        ax.set_xlabel("Time (us)")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right")
        out_fig = RESULTS / f"waveform_model_vs_baseline_SNR{snr:+d}dB.png"
        fig.tight_layout()
        fig.savefig(out_fig, dpi=250)
        plt.close(fig)

        rows.append(
            {
                "snr_in_db": snr,
                "best_model": {k: v for k, v in best_model.items() if k != "pred"},
                "best_baseline": {k: v for k, v in best_baseline.items() if k != "pred"},
                "delta_ncc_model_minus_baseline": best_model["ncc"] - best_baseline["ncc"],
                "delta_rmse_model_minus_baseline": best_model["rmse"] - best_baseline["rmse"],
                "delta_snr_out_db_model_minus_baseline": best_model["snr_out"] - best_baseline["snr_out"],
                "figure": str(out_fig.relative_to(PROJECT_ROOT)),
            }
        )

    out_json = RESULTS / "waveform_baseline_comparison.json"
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### 2(a). MODEL_LIST Or Equivalent

`generate_method_comparison_figures.py` has no literal `MODEL_LIST`; it builds `model_names` from `args.models` or registry entries with checkpoints.

Path: `NEW FOCUS/generate_method_comparison_figures.py` lines 426-438

```python
    # Decide which models to render
    if args.models:
        model_names = args.models
    else:
        model_names = []
        for name in registry:
            if any(
                (PROJECT_ROOT / "results" / name / "checkpoints" / "best.pt").exists()
                or (PROJECT_ROOT / "results" / name / "best.pt").exists()
                or (PROJECT_ROOT / "results" / name / "best.pth").exists()
                for _ in [None]
            ):
                model_names.append(name)
```

`compare_waveforms_with_baselines.py` has no literal `MODEL_LIST`; it builds `model_ids` from registry entries with `results/<model>/checkpoints/best.pt`.

Path: `NEW FOCUS/compare_waveforms_with_baselines.py` lines 81-85

```python
def main() -> int:
    with REGISTRY.open() as f:
        registry = (yaml.safe_load(f) or {}).get("models", {})
    model_ids = [m for m in registry if (RESULTS / m / "checkpoints" / "best.pt").exists()]

```

### 2(b). Checkpoint Path Resolution

Path: `NEW FOCUS/generate_method_comparison_figures.py` lines 328-346

```python
def load_model(model_name: str, registry: dict, device: torch.device):
    """
    Load a model from your registry by name. Falls back to walking
    `results/{model_name}/checkpoints/best.pt` and instantiating from the
    AblationConfig stored in the checkpoint payload.

    This is intentionally tolerant of different checkpoint shapes -- adjust
    the keys to match what your trainer.py actually saves.
    """
    ckpt_candidates = [
        PROJECT_ROOT / "results" / model_name / "checkpoints" / "best.pt",
        PROJECT_ROOT / "results" / model_name / "best.pt",
        PROJECT_ROOT / "results" / model_name / "best.pth",
    ]
    ckpt_path = next((p for p in ckpt_candidates if p.exists()), None)
    if ckpt_path is None:
        return None, None

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
```

Path: `NEW FOCUS/compare_waveforms_with_baselines.py` lines 65-78

```python
def load_model(model_id: str, registry: dict, device: torch.device):
    from models.base import build_model_from_registry
    with TRAIN_CFG.open() as f:
        tcfg = yaml.safe_load(f) or {}
    model = build_model_from_registry(model_id, {"models": registry}, tcfg).to(device).eval()
    ckpt = torch.load(RESULTS / model_id / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    if "model_state" in ckpt:
        state = ckpt["model_state"]
    elif isinstance(ckpt.get("model"), dict):
        state = ckpt["model"]
    else:
        state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)
    return model
```

### 2(c). Test Waveform Construction

Path: `NEW FOCUS/generate_method_comparison_figures.py` lines 150-180

```python
def select_window_at_snr(
    test_set: dict, target_snr_db: int, window_idx: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pick a (noisy, clean) pair from the synthetic test set at the requested SNR level.

    Expected test-set format (matches your build_test_sets.py output):
        {
            "noisy":  Tensor[N, 1, 2048],
            "clean":  Tensor[N, 1, 2048],
            "snr_db": Tensor[N]    # int per-window SNR label
        }
    """
    snr = test_set.get("snr_db")
    if snr is None:
        snr = test_set.get("snr_labels")
    if snr is None:
        raise KeyError("Test set must include `snr_db` or `snr_labels`.")
    if isinstance(snr, torch.Tensor):
        snr = snr.cpu().numpy()
    mask = (snr == target_snr_db)
    if not mask.any():
        raise ValueError(
            f"No windows at SNR={target_snr_db} dB in test set. "
            f"Available: {sorted(set(int(x) for x in snr))}"
        )
    candidate_idx = np.where(mask)[0]
    chosen = candidate_idx[window_idx % len(candidate_idx)]
    noisy = test_set["noisy"][chosen].squeeze().cpu().numpy()
    clean = test_set["clean"][chosen].squeeze().cpu().numpy()
    return noisy, clean
```

Path: `NEW FOCUS/generate_method_comparison_figures.py` lines 449-467

```python
    # ---- Load test set ----
    if args.expected_hash:
        verify_test_set_hash(args.test_set, args.expected_hash)
    test_set = torch.load(args.test_set, map_location="cpu", weights_only=False)
    n_windows = int(test_set.get("noisy").shape[0]) if "noisy" in test_set else -1
    print(f"Loaded test set: {args.test_set} ({n_windows} windows)")

    device = torch.device(args.device)

    # ---- For each SNR level, render per-method figures and a grid ----
    summary_rows = []
    for snr_level in args.snr_levels:
        try:
            noisy, clean = select_window_at_snr(test_set, snr_level, args.window_idx)
        except ValueError as e:
            print(f"[skip] {e}", file=sys.stderr)
            continue
        noisy_t = torch.from_numpy(noisy).float().view(1, 1, -1).to(device)

```

Path: `NEW FOCUS/compare_waveforms_with_baselines.py` lines 86-96

```python
    ds = torch.load(TEST_SET, map_location="cpu", weights_only=False)
    snr_labels = ds.get("snr_labels", ds.get("snr_db")).cpu().numpy().ravel()
    noisy_all = ds["noisy"]
    clean_all = ds["clean"]

    device = torch.device("cpu")
    rows = []
    for snr in SNR_LEVELS:
        idx = int(np.where(snr_labels == snr)[0][0])
        noisy = noisy_all[idx].squeeze().numpy()
        clean = clean_all[idx].squeeze().numpy()
```

Path: `NEW FOCUS/compare_waveforms_with_baselines.py` lines 110-115

```python
        model_rows = []
        x = torch.from_numpy(noisy).float().view(1, 1, -1).to(device)
        for mid in model_ids:
            model = load_model(mid, registry, device)
            with torch.no_grad():
                out = model(x)
```

### 2(d). Input Normalization Before `model.forward`

Neither script applies an additional normalization immediately before forward; both use the `noisy` tensor as loaded/selected from the test set.

Path: `NEW FOCUS/generate_method_comparison_figures.py` lines 466-481

```python
        noisy_t = torch.from_numpy(noisy).float().view(1, 1, -1).to(device)

        grid_rows = []
        for name in model_names:
            model, ckpt_path = load_model(name, registry, device)
            if model is None:
                print(f"[skip] No checkpoint for {name}", file=sys.stderr)
                continue
            with torch.no_grad():
                out = model(noisy_t)
                # Tolerate dual-head MTL outputs.
                if isinstance(out, (tuple, list)):
                    out = out[0]
                elif isinstance(out, dict):
                    out = out.get("denoised") or out.get("signal") or next(iter(out.values()))
            denoised = out.squeeze().cpu().numpy()
```

Path: `NEW FOCUS/compare_waveforms_with_baselines.py` lines 110-118

```python
        model_rows = []
        x = torch.from_numpy(noisy).float().view(1, 1, -1).to(device)
        for mid in model_ids:
            model = load_model(mid, registry, device)
            with torch.no_grad():
                out = model(x)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                pred = out.squeeze().cpu().numpy()
```

### 2(e). PROJECT_ROOT Line

Path: `NEW FOCUS/generate_method_comparison_figures.py` lines 55-65

```python
# ---------------------------------------------------------------------------
# Project paths -- adjust here if your tree differs.
# ---------------------------------------------------------------------------
# This bundle lives under Data/FYP_MASTER/NEW FOCUS in this workspace, while
# the active training repo root is three levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
TEST_SET_SYNTH = PROJECT_ROOT / "data" / "test_synthetic_ae.pt"
TEST_SET_REAL = PROJECT_ROOT / "data" / "test_real_qlin.pt"
MODEL_REGISTRY = PROJECT_ROOT / "config" / "model_registry.yaml"
RESULTS_DIR = PROJECT_ROOT / "results"
```

Path: `NEW FOCUS/compare_waveforms_with_baselines.py` lines 20-26

```python
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

TEST_SET = PROJECT_ROOT / "data" / "test_synthetic_ae.pt"
REGISTRY = PROJECT_ROOT / "config" / "model_registry.yaml"
TRAIN_CFG = PROJECT_ROOT / "config" / "training_config.yaml"
RESULTS = PROJECT_ROOT / "results"
```

Both scripts point to `D:\New folder (2)` for this file layout, not `D:\New folder (2)\Data\FYP_MASTER`.

## 3. TRAINING-TIME PREPROCESSING

The orchestrator used for model training imports `PDSignalDataset` from `mr_tae_fusion.data.dataset` and passes its batches to `training.Trainer`.

Path: `D:\New folder (2)\pipeline\mlflow_orchestrator.py` lines 115-139

```python
def build_loaders(
    batch_size: int,
    n_train: int = 4000,
    n_val: int = 800,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> tuple:
    from mr_tae_fusion.data.dataset import PDSignalDataset
    from mr_tae_fusion.config import Config
    from torch.utils.data import DataLoader

    cfg = Config()
    train_ds = PDSignalDataset(config=cfg, num_samples=n_train, mode="train", seed=seed)
    val_ds = PDSignalDataset(config=cfg, num_samples=n_val, mode="val", seed=seed + 1)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, val_loader
```

Path: `D:\New folder (2)\pipeline\mlflow_orchestrator.py` lines 173-200

```python
    train_loader, val_loader = build_loaders(
        batch_size=batch_size,
        n_train=int(tcfg.get("n_train", 4000)),
        n_val=int(tcfg.get("n_val", 800)),
        seed=int(tcfg.get("seed", 42)),
        num_workers=int(tcfg.get("num_workers", 0)),
        pin_memory=bool(tcfg.get("pin_memory", True)),
    )

    scheduler = CurricularScheduler()

    trainer = Trainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=results_root,
        model_id=model_id,
        scheduler=scheduler,
        lr=float(tcfg.get("learning_rate", 3e-4)),
        weight_decay=float(tcfg.get("weight_decay", 1e-4)),
        grad_clip=float(tcfg.get("grad_clip", 1.0)),
        mlflow_run_id=mlflow_run_id,
        resilience=resilience,
    )

    history = trainer.train(resume_from=resume_ckpt, phase=None)
```

### Clean-Signal Normalization
Clean signals are peak-normalized to unit peak inside `generate_pd_signal`, before noise injection.

Path: `D:\New folder (2)\mr_tae_fusion\data\pulse_generators.py` lines 731-776

```python
def generate_pd_signal(
    signal_type: str,
    config: Optional[SignalConfig] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic AE-band PD signal of specified type.

    Args:
        signal_type: One of 'A'–'G', 'MIXED', 'TEMPORAL'.
        config: Signal configuration (default ``SignalConfig()``).

    Returns:
        (time_vector, clean_signal, segmentation_mask)
        The signal is peak-normalised to [-1, 1].
    """
    if config is None:
        config = SignalConfig()

    fs = config.sample_rate
    t = np.arange(config.signal_length) / fs

    generators = {
        'A': lambda: generate_type_a(t, config),
        'B': lambda: generate_type_b(t, config),
        'C': lambda: generate_type_c(t, fs, config),
        'D': lambda: generate_type_d(t, fs, config),
        'E': lambda: generate_type_e(t, fs, config),
        'F': lambda: generate_type_f(t, fs, config),
        'G': lambda: generate_type_g(t, fs, config),
        'MIXED': lambda: generate_mixed_signal(t, fs, config),
        'TEMPORAL': lambda: generate_temporal_sequence(t, fs, config),
    }

    if signal_type not in generators:
        raise ValueError(
            f"Unknown signal type: {signal_type}. "
            f"Must be one of {list(generators.keys())}"
        )

    signal, mask = generators[signal_type]()

    max_amp = np.max(np.abs(signal))
    if max_amp > 0:
        signal = signal / (max_amp + 1e-10)

    return t, signal, mask
```

### Noisy/Clean Pair Fed To Model

Path: `D:\New folder (2)\mr_tae_fusion\data\dataset.py` lines 83-139

```python
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Generate a single sample.
        
        Returns:
            Dictionary with:
            - 'noisy': Noisy signal (1, L)
            - 'clean': Clean signal (1, L)
            - 'mask': Segmentation mask (L,)
            - 'snr': Applied SNR in dB
            - 'type': Signal type index
        """
        # Get signal type for this sample
        signal_type = self._sample_types[idx]
        
        # Generate clean signal with segmentation mask
        _, clean_signal, mask = generate_pd_signal(
            signal_type, 
            self.config.signal
        )
        
        # Generate noise based on curriculum phase (or external override)
        if self._override_noise_type is not None:
            noise_type = self._override_noise_type
        else:
            noise_type = self.curriculum.get_noise_type(self.epoch, self.total_epochs)
        noise = self._generate_noise(noise_type)

        # Sample SNR
        if self._override_snr_range is not None:
            lo, hi = self._override_snr_range
            target_snr = float(self.rng.uniform(lo, hi))
        else:
            target_snr = self.curriculum.sample_snr(
                self.epoch,
                self.total_epochs,
                self.rng,
            )
        
        # Add noise at target SNR
        noisy_signal, actual_snr = add_noise_at_snr(clean_signal, noise, target_snr)
        
        # Convert to tensors
        noisy_tensor = torch.tensor(noisy_signal, dtype=torch.float32).unsqueeze(0)
        clean_tensor = torch.tensor(clean_signal, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(mask, dtype=torch.long)
        
        # Type index for potential use
        type_idx = self.signal_types.index(signal_type)
        
        return {
            'noisy': noisy_tensor,
            'clean': clean_tensor,
            'mask': mask_tensor,
            'snr': torch.tensor(actual_snr, dtype=torch.float32),
            'type': torch.tensor(type_idx, dtype=torch.long)
        }
```

### Noise Injection Routine
SNR is defined using variance of the clean signal and variance of the noise. Noise is scaled to target dB, then added. There is no post-addition normalization in this routine or in `PDSignalDataset.__getitem__`.

Path: `D:\New folder (2)\mr_tae_fusion\data\noise_generators.py` lines 367-394

```python
def add_noise_at_snr(
    clean_signal: np.ndarray,
    noise: np.ndarray,
    target_snr_db: float,
) -> Tuple[np.ndarray, float]:
    """Add noise to signal at a specified SNR level.

    Args:
        clean_signal: Clean signal.
        noise: Noise signal (will be scaled).
        target_snr_db: Desired SNR in dB.

    Returns:
        Tuple of (noisy_signal, actual_snr_db).
    """
    signal_power = np.var(clean_signal) + 1e-10
    noise_power = np.var(noise) + 1e-10

    current_snr_db = 10 * np.log10(signal_power / noise_power)
    scale_factor = 10 ** ((current_snr_db - target_snr_db) / 20)
    scaled_noise = noise * scale_factor

    noisy_signal = clean_signal + scaled_noise

    actual_noise_power = np.var(scaled_noise) + 1e-10
    actual_snr_db = 10 * np.log10(signal_power / actual_noise_power)

    return noisy_signal, actual_snr_db
```

## 4. CHECKPOINT-TO-CONFIG MISMATCH CHECK

Requested key note: these checkpoints do not contain `model_state_dict`; they contain `model_state`. I therefore report both the absence of `model_state_dict` and the equivalent strict load using `model_state`. The current `models.base.build_model_from_registry` implementation constructs `GenericUNetDenoiser`, whose parameter names do not match the saved `MRTAEFusion`-style checkpoints.

### MR-TAE-FULL

```json
{
  "model_id": "MR-TAE-FULL",
  "checkpoint": "results/MR-TAE-FULL/checkpoints/best.pt",
  "exists": true,
  "checkpoint_keys": [
    "epoch",
    "model_state",
    "optimizer_state",
    "scheduler_state",
    "criterion_state",
    "scaler_state",
    "best_val_ncc",
    "history",
    "model_id"
  ],
  "model_state_dict_present": false,
  "state_key_used_for_equivalent_strict_load": "model_state",
  "strict_load_succeeded": false,
  "runtime_error": "Error(s) in loading state_dict for GenericUNetDenoiser:"
}
```

Missing keys (122):

```text
attn1.gate_proj.bias
attn1.gate_proj.weight
attn1.score.bias
attn1.score.weight
attn1.skip_proj.bias
attn1.skip_proj.weight
attn2.gate_proj.bias
attn2.gate_proj.weight
attn2.score.bias
attn2.score.weight
attn2.skip_proj.bias
attn2.skip_proj.weight
bottleneck.0.gru.bias_hh_l0
bottleneck.0.gru.bias_hh_l0_reverse
bottleneck.0.gru.bias_hh_l1
bottleneck.0.gru.bias_hh_l1_reverse
bottleneck.0.gru.bias_ih_l0
bottleneck.0.gru.bias_ih_l0_reverse
bottleneck.0.gru.bias_ih_l1
bottleneck.0.gru.bias_ih_l1_reverse
bottleneck.0.gru.weight_hh_l0
bottleneck.0.gru.weight_hh_l0_reverse
bottleneck.0.gru.weight_hh_l1
bottleneck.0.gru.weight_hh_l1_reverse
bottleneck.0.gru.weight_ih_l0
bottleneck.0.gru.weight_ih_l0_reverse
bottleneck.0.gru.weight_ih_l1
bottleneck.0.gru.weight_ih_l1_reverse
bottleneck.0.out.bias
bottleneck.0.out.weight
bottleneck.0.proj.bias
bottleneck.0.proj.weight
bottleneck.1.attn.in_proj_bias
bottleneck.1.attn.in_proj_weight
bottleneck.1.attn.out_proj.bias
bottleneck.1.attn.out_proj.weight
bottleneck.1.ffn.0.bias
bottleneck.1.ffn.0.weight
bottleneck.1.ffn.2.bias
bottleneck.1.ffn.2.weight
bottleneck.1.norm1.bias
bottleneck.1.norm1.weight
bottleneck.1.norm2.bias
bottleneck.1.norm2.weight
dec1.net.0.bias
dec1.net.0.weight
dec1.net.1.bias
dec1.net.1.num_batches_tracked
dec1.net.1.running_mean
dec1.net.1.running_var
dec1.net.1.weight
dec1.net.3.bias
dec1.net.3.weight
dec1.net.4.bias
dec1.net.4.num_batches_tracked
dec1.net.4.running_mean
dec1.net.4.running_var
dec1.net.4.weight
dec2.net.0.bias
dec2.net.0.weight
dec2.net.1.bias
dec2.net.1.num_batches_tracked
dec2.net.1.running_mean
dec2.net.1.running_var
dec2.net.1.weight
dec2.net.3.bias
dec2.net.3.weight
dec2.net.4.bias
dec2.net.4.num_batches_tracked
dec2.net.4.running_mean
dec2.net.4.running_var
dec2.net.4.weight
enc1.net.0.bias
enc1.net.0.weight
enc1.net.1.bias
enc1.net.1.num_batches_tracked
enc1.net.1.running_mean
enc1.net.1.running_var
enc1.net.1.weight
enc1.net.3.bias
enc1.net.3.weight
enc1.net.4.bias
enc1.net.4.num_batches_tracked
enc1.net.4.running_mean
enc1.net.4.running_var
enc1.net.4.weight
enc2.net.0.bias
enc2.net.0.weight
enc2.net.1.bias
enc2.net.1.num_batches_tracked
enc2.net.1.running_mean
enc2.net.1.running_var
enc2.net.1.weight
enc2.net.3.bias
enc2.net.3.weight
enc2.net.4.bias
enc2.net.4.num_batches_tracked
enc2.net.4.running_mean
enc2.net.4.running_var
enc2.net.4.weight
enc3.net.0.bias
enc3.net.0.weight
enc3.net.1.bias
enc3.net.1.num_batches_tracked
enc3.net.1.running_mean
enc3.net.1.running_var
enc3.net.1.weight
enc3.net.3.bias
enc3.net.3.weight
enc3.net.4.bias
enc3.net.4.num_batches_tracked
enc3.net.4.running_mean
enc3.net.4.running_var
enc3.net.4.weight
out.bias
out.weight
seg.bias
seg.weight
up1.up.bias
up1.up.weight
up2.up.bias
up2.up.weight
```

Unexpected keys (208):

```text
bottleneck.gru.bias_hh_l0
bottleneck.gru.bias_hh_l0_reverse
bottleneck.gru.bias_hh_l1
bottleneck.gru.bias_hh_l1_reverse
bottleneck.gru.bias_ih_l0
bottleneck.gru.bias_ih_l0_reverse
bottleneck.gru.bias_ih_l1
bottleneck.gru.bias_ih_l1_reverse
bottleneck.gru.weight_hh_l0
bottleneck.gru.weight_hh_l0_reverse
bottleneck.gru.weight_hh_l1
bottleneck.gru.weight_hh_l1_reverse
bottleneck.gru.weight_ih_l0
bottleneck.gru.weight_ih_l0_reverse
bottleneck.gru.weight_ih_l1
bottleneck.gru.weight_ih_l1_reverse
bottleneck.gru_to_swin.bias
bottleneck.gru_to_swin.weight
bottleneck.output_proj.bias
bottleneck.output_proj.weight
bottleneck.pre_gru.bias
bottleneck.pre_gru.weight
bottleneck.residual.bias
bottleneck.residual.weight
bottleneck.swin.blocks.0.attn.proj.bias
bottleneck.swin.blocks.0.attn.proj.weight
bottleneck.swin.blocks.0.attn.qkv.bias
bottleneck.swin.blocks.0.attn.qkv.weight
bottleneck.swin.blocks.0.attn.relative_position_bias_table
bottleneck.swin.blocks.0.attn.relative_position_index
bottleneck.swin.blocks.0.mlp.fc1.bias
bottleneck.swin.blocks.0.mlp.fc1.weight
bottleneck.swin.blocks.0.mlp.fc2.bias
bottleneck.swin.blocks.0.mlp.fc2.weight
bottleneck.swin.blocks.0.norm1.bias
bottleneck.swin.blocks.0.norm1.weight
bottleneck.swin.blocks.0.norm2.bias
bottleneck.swin.blocks.0.norm2.weight
bottleneck.swin.blocks.1.attn.proj.bias
bottleneck.swin.blocks.1.attn.proj.weight
bottleneck.swin.blocks.1.attn.qkv.bias
bottleneck.swin.blocks.1.attn.qkv.weight
bottleneck.swin.blocks.1.attn.relative_position_bias_table
bottleneck.swin.blocks.1.attn.relative_position_index
bottleneck.swin.blocks.1.mlp.fc1.bias
bottleneck.swin.blocks.1.mlp.fc1.weight
bottleneck.swin.blocks.1.mlp.fc2.bias
bottleneck.swin.blocks.1.mlp.fc2.weight
bottleneck.swin.blocks.1.norm1.bias
bottleneck.swin.blocks.1.norm1.weight
bottleneck.swin.blocks.1.norm2.bias
bottleneck.swin.blocks.1.norm2.weight
bottleneck.swin.norm.bias
bottleneck.swin.norm.weight
decoder.blocks.0.conv1.bn.bias
decoder.blocks.0.conv1.bn.num_batches_tracked
decoder.blocks.0.conv1.bn.running_mean
decoder.blocks.0.conv1.bn.running_var
decoder.blocks.0.conv1.bn.weight
decoder.blocks.0.conv1.conv.bias
decoder.blocks.0.conv1.conv.weight
decoder.blocks.0.conv2.bn.bias
decoder.blocks.0.conv2.bn.num_batches_tracked
decoder.blocks.0.conv2.bn.running_mean
decoder.blocks.0.conv2.bn.running_var
decoder.blocks.0.conv2.bn.weight
decoder.blocks.0.conv2.conv.bias
decoder.blocks.0.conv2.conv.weight
decoder.blocks.0.upsample.1.bias
decoder.blocks.0.upsample.1.weight
decoder.blocks.0.upsample.2.bias
decoder.blocks.0.upsample.2.num_batches_tracked
decoder.blocks.0.upsample.2.running_mean
decoder.blocks.0.upsample.2.running_var
decoder.blocks.0.upsample.2.weight
decoder.blocks.1.conv1.bn.bias
decoder.blocks.1.conv1.bn.num_batches_tracked
decoder.blocks.1.conv1.bn.running_mean
decoder.blocks.1.conv1.bn.running_var
decoder.blocks.1.conv1.bn.weight
decoder.blocks.1.conv1.conv.bias
decoder.blocks.1.conv1.conv.weight
decoder.blocks.1.conv2.bn.bias
decoder.blocks.1.conv2.bn.num_batches_tracked
decoder.blocks.1.conv2.bn.running_mean
decoder.blocks.1.conv2.bn.running_var
decoder.blocks.1.conv2.bn.weight
decoder.blocks.1.conv2.conv.bias
decoder.blocks.1.conv2.conv.weight
decoder.blocks.1.upsample.1.bias
decoder.blocks.1.upsample.1.weight
decoder.blocks.1.upsample.2.bias
decoder.blocks.1.upsample.2.num_batches_tracked
decoder.blocks.1.upsample.2.running_mean
decoder.blocks.1.upsample.2.running_var
decoder.blocks.1.upsample.2.weight
decoder.blocks.2.conv1.bn.bias
decoder.blocks.2.conv1.bn.num_batches_tracked
decoder.blocks.2.conv1.bn.running_mean
decoder.blocks.2.conv1.bn.running_var
decoder.blocks.2.conv1.bn.weight
decoder.blocks.2.conv1.conv.bias
decoder.blocks.2.conv1.conv.weight
decoder.blocks.2.conv2.bn.bias
decoder.blocks.2.conv2.bn.num_batches_tracked
decoder.blocks.2.conv2.bn.running_mean
decoder.blocks.2.conv2.bn.running_var
decoder.blocks.2.conv2.bn.weight
decoder.blocks.2.conv2.conv.bias
decoder.blocks.2.conv2.conv.weight
decoder.blocks.2.upsample.1.bias
decoder.blocks.2.upsample.1.weight
decoder.blocks.2.upsample.2.bias
decoder.blocks.2.upsample.2.num_batches_tracked
decoder.blocks.2.upsample.2.running_mean
decoder.blocks.2.upsample.2.running_var
decoder.blocks.2.upsample.2.weight
encoder.blocks.0.conv1.bn.bias
encoder.blocks.0.conv1.bn.num_batches_tracked
encoder.blocks.0.conv1.bn.running_mean
encoder.blocks.0.conv1.bn.running_var
encoder.blocks.0.conv1.bn.weight
encoder.blocks.0.conv1.conv.bias
encoder.blocks.0.conv1.conv.weight
encoder.blocks.0.conv2.bn.bias
encoder.blocks.0.conv2.bn.num_batches_tracked
encoder.blocks.0.conv2.bn.running_mean
encoder.blocks.0.conv2.bn.running_var
encoder.blocks.0.conv2.bn.weight
encoder.blocks.0.conv2.conv.bias
encoder.blocks.0.conv2.conv.weight
encoder.blocks.0.dwt.hi_filter
encoder.blocks.0.dwt.lo_filter
encoder.blocks.1.conv1.bn.bias
encoder.blocks.1.conv1.bn.num_batches_tracked
encoder.blocks.1.conv1.bn.running_mean
encoder.blocks.1.conv1.bn.running_var
encoder.blocks.1.conv1.bn.weight
encoder.blocks.1.conv1.conv.bias
encoder.blocks.1.conv1.conv.weight
encoder.blocks.1.conv2.bn.bias
encoder.blocks.1.conv2.bn.num_batches_tracked
encoder.blocks.1.conv2.bn.running_mean
encoder.blocks.1.conv2.bn.running_var
encoder.blocks.1.conv2.bn.weight
encoder.blocks.1.conv2.conv.bias
encoder.blocks.1.conv2.conv.weight
encoder.blocks.1.dwt.hi_filter
encoder.blocks.1.dwt.lo_filter
encoder.blocks.2.conv1.bn.bias
encoder.blocks.2.conv1.bn.num_batches_tracked
encoder.blocks.2.conv1.bn.running_mean
encoder.blocks.2.conv1.bn.running_var
encoder.blocks.2.conv1.bn.weight
encoder.blocks.2.conv1.conv.bias
encoder.blocks.2.conv1.conv.weight
encoder.blocks.2.conv2.bn.bias
encoder.blocks.2.conv2.bn.num_batches_tracked
encoder.blocks.2.conv2.bn.running_mean
encoder.blocks.2.conv2.bn.running_var
encoder.blocks.2.conv2.bn.weight
encoder.blocks.2.conv2.conv.bias
encoder.blocks.2.conv2.conv.weight
encoder.blocks.2.dwt.hi_filter
encoder.blocks.2.dwt.lo_filter
input_proj.bias
input_proj.weight
pre_bottleneck.bias
pre_bottleneck.weight
recon_head.conv.0.bias
recon_head.conv.0.weight
recon_head.conv.1.bias
recon_head.conv.1.num_batches_tracked
recon_head.conv.1.running_mean
recon_head.conv.1.running_var
recon_head.conv.1.weight
recon_head.conv.3.bias
recon_head.conv.3.weight
seg_head.classifier.bias
seg_head.classifier.weight
seg_head.fusion.0.bias
seg_head.fusion.0.weight
seg_head.fusion.1.bias
seg_head.fusion.1.num_batches_tracked
seg_head.fusion.1.running_mean
seg_head.fusion.1.running_var
seg_head.fusion.1.weight
seg_head.fusion.4.bias
seg_head.fusion.4.weight
seg_head.fusion.5.bias
seg_head.fusion.5.num_batches_tracked
seg_head.fusion.5.running_mean
seg_head.fusion.5.running_var
seg_head.fusion.5.weight
seg_head.global_conv.0.bias
seg_head.global_conv.0.weight
seg_head.global_conv.1.bias
seg_head.global_conv.1.num_batches_tracked
seg_head.global_conv.1.running_mean
seg_head.global_conv.1.running_var
seg_head.global_conv.1.weight
seg_head.local_conv.0.bias
seg_head.local_conv.0.weight
seg_head.local_conv.1.bias
seg_head.local_conv.1.num_batches_tracked
seg_head.local_conv.1.running_mean
seg_head.local_conv.1.running_var
seg_head.local_conv.1.weight
```

### MR-TAE-noBiGRU

```json
{
  "model_id": "MR-TAE-noBiGRU",
  "checkpoint": "results/MR-TAE-noBiGRU/checkpoints/best.pt",
  "exists": true,
  "checkpoint_keys": [
    "epoch",
    "model_state",
    "optimizer_state",
    "scheduler_state",
    "criterion_state",
    "scaler_state",
    "best_val_ncc",
    "history",
    "model_id"
  ],
  "model_state_dict_present": false,
  "state_key_used_for_equivalent_strict_load": "model_state",
  "strict_load_succeeded": false,
  "runtime_error": "Error(s) in loading state_dict for GenericUNetDenoiser:"
}
```

Missing keys (102):

```text
attn1.gate_proj.bias
attn1.gate_proj.weight
attn1.score.bias
attn1.score.weight
attn1.skip_proj.bias
attn1.skip_proj.weight
attn2.gate_proj.bias
attn2.gate_proj.weight
attn2.score.bias
attn2.score.weight
attn2.skip_proj.bias
attn2.skip_proj.weight
bottleneck.0.attn.in_proj_bias
bottleneck.0.attn.in_proj_weight
bottleneck.0.attn.out_proj.bias
bottleneck.0.attn.out_proj.weight
bottleneck.0.ffn.0.bias
bottleneck.0.ffn.0.weight
bottleneck.0.ffn.2.bias
bottleneck.0.ffn.2.weight
bottleneck.0.norm1.bias
bottleneck.0.norm1.weight
bottleneck.0.norm2.bias
bottleneck.0.norm2.weight
dec1.net.0.bias
dec1.net.0.weight
dec1.net.1.bias
dec1.net.1.num_batches_tracked
dec1.net.1.running_mean
dec1.net.1.running_var
dec1.net.1.weight
dec1.net.3.bias
dec1.net.3.weight
dec1.net.4.bias
dec1.net.4.num_batches_tracked
dec1.net.4.running_mean
dec1.net.4.running_var
dec1.net.4.weight
dec2.net.0.bias
dec2.net.0.weight
dec2.net.1.bias
dec2.net.1.num_batches_tracked
dec2.net.1.running_mean
dec2.net.1.running_var
dec2.net.1.weight
dec2.net.3.bias
dec2.net.3.weight
dec2.net.4.bias
dec2.net.4.num_batches_tracked
dec2.net.4.running_mean
dec2.net.4.running_var
dec2.net.4.weight
enc1.net.0.bias
enc1.net.0.weight
enc1.net.1.bias
enc1.net.1.num_batches_tracked
enc1.net.1.running_mean
enc1.net.1.running_var
enc1.net.1.weight
enc1.net.3.bias
enc1.net.3.weight
enc1.net.4.bias
enc1.net.4.num_batches_tracked
enc1.net.4.running_mean
enc1.net.4.running_var
enc1.net.4.weight
enc2.net.0.bias
enc2.net.0.weight
enc2.net.1.bias
enc2.net.1.num_batches_tracked
enc2.net.1.running_mean
enc2.net.1.running_var
enc2.net.1.weight
enc2.net.3.bias
enc2.net.3.weight
enc2.net.4.bias
enc2.net.4.num_batches_tracked
enc2.net.4.running_mean
enc2.net.4.running_var
enc2.net.4.weight
enc3.net.0.bias
enc3.net.0.weight
enc3.net.1.bias
enc3.net.1.num_batches_tracked
enc3.net.1.running_mean
enc3.net.1.running_var
enc3.net.1.weight
enc3.net.3.bias
enc3.net.3.weight
enc3.net.4.bias
enc3.net.4.num_batches_tracked
enc3.net.4.running_mean
enc3.net.4.running_var
enc3.net.4.weight
out.bias
out.weight
seg.bias
seg.weight
up1.up.bias
up1.up.weight
up2.up.bias
up2.up.weight
```

Unexpected keys (223):

```text
attention_gates.0.attention_gate.W_gate.bias
attention_gates.0.attention_gate.W_gate.weight
attention_gates.0.attention_gate.W_skip.weight
attention_gates.0.attention_gate.bn.bias
attention_gates.0.attention_gate.bn.num_batches_tracked
attention_gates.0.attention_gate.bn.running_mean
attention_gates.0.attention_gate.bn.running_var
attention_gates.0.attention_gate.bn.weight
attention_gates.0.attention_gate.psi.bias
attention_gates.0.attention_gate.psi.weight
attention_gates.0.residual_weight
attention_gates.1.attention_gate.W_gate.bias
attention_gates.1.attention_gate.W_gate.weight
attention_gates.1.attention_gate.W_skip.weight
attention_gates.1.attention_gate.bn.bias
attention_gates.1.attention_gate.bn.num_batches_tracked
attention_gates.1.attention_gate.bn.running_mean
attention_gates.1.attention_gate.bn.running_var
attention_gates.1.attention_gate.bn.weight
attention_gates.1.attention_gate.psi.bias
attention_gates.1.attention_gate.psi.weight
attention_gates.1.residual_weight
attention_gates.2.attention_gate.W_gate.bias
attention_gates.2.attention_gate.W_gate.weight
attention_gates.2.attention_gate.W_skip.weight
attention_gates.2.attention_gate.bn.bias
attention_gates.2.attention_gate.bn.num_batches_tracked
attention_gates.2.attention_gate.bn.running_mean
attention_gates.2.attention_gate.bn.running_var
attention_gates.2.attention_gate.bn.weight
attention_gates.2.attention_gate.psi.bias
attention_gates.2.attention_gate.psi.weight
attention_gates.2.residual_weight
bottleneck.input_proj.bias
bottleneck.input_proj.weight
bottleneck.output_proj.bias
bottleneck.output_proj.weight
bottleneck.residual.bias
bottleneck.residual.weight
bottleneck.swin.blocks.0.attn.proj.bias
bottleneck.swin.blocks.0.attn.proj.weight
bottleneck.swin.blocks.0.attn.qkv.bias
bottleneck.swin.blocks.0.attn.qkv.weight
bottleneck.swin.blocks.0.attn.relative_position_bias_table
bottleneck.swin.blocks.0.attn.relative_position_index
bottleneck.swin.blocks.0.mlp.fc1.bias
bottleneck.swin.blocks.0.mlp.fc1.weight
bottleneck.swin.blocks.0.mlp.fc2.bias
bottleneck.swin.blocks.0.mlp.fc2.weight
bottleneck.swin.blocks.0.norm1.bias
bottleneck.swin.blocks.0.norm1.weight
bottleneck.swin.blocks.0.norm2.bias
bottleneck.swin.blocks.0.norm2.weight
bottleneck.swin.blocks.1.attn.proj.bias
bottleneck.swin.blocks.1.attn.proj.weight
bottleneck.swin.blocks.1.attn.qkv.bias
bottleneck.swin.blocks.1.attn.qkv.weight
bottleneck.swin.blocks.1.attn.relative_position_bias_table
bottleneck.swin.blocks.1.attn.relative_position_index
bottleneck.swin.blocks.1.mlp.fc1.bias
bottleneck.swin.blocks.1.mlp.fc1.weight
bottleneck.swin.blocks.1.mlp.fc2.bias
bottleneck.swin.blocks.1.mlp.fc2.weight
bottleneck.swin.blocks.1.norm1.bias
bottleneck.swin.blocks.1.norm1.weight
bottleneck.swin.blocks.1.norm2.bias
bottleneck.swin.blocks.1.norm2.weight
bottleneck.swin.norm.bias
bottleneck.swin.norm.weight
decoder.blocks.0.conv1.bn.bias
decoder.blocks.0.conv1.bn.num_batches_tracked
decoder.blocks.0.conv1.bn.running_mean
decoder.blocks.0.conv1.bn.running_var
decoder.blocks.0.conv1.bn.weight
decoder.blocks.0.conv1.conv.bias
decoder.blocks.0.conv1.conv.weight
decoder.blocks.0.conv2.bn.bias
decoder.blocks.0.conv2.bn.num_batches_tracked
decoder.blocks.0.conv2.bn.running_mean
decoder.blocks.0.conv2.bn.running_var
decoder.blocks.0.conv2.bn.weight
decoder.blocks.0.conv2.conv.bias
decoder.blocks.0.conv2.conv.weight
decoder.blocks.0.upsample.1.bias
decoder.blocks.0.upsample.1.weight
decoder.blocks.0.upsample.2.bias
decoder.blocks.0.upsample.2.num_batches_tracked
decoder.blocks.0.upsample.2.running_mean
decoder.blocks.0.upsample.2.running_var
decoder.blocks.0.upsample.2.weight
decoder.blocks.1.conv1.bn.bias
decoder.blocks.1.conv1.bn.num_batches_tracked
decoder.blocks.1.conv1.bn.running_mean
decoder.blocks.1.conv1.bn.running_var
decoder.blocks.1.conv1.bn.weight
decoder.blocks.1.conv1.conv.bias
decoder.blocks.1.conv1.conv.weight
decoder.blocks.1.conv2.bn.bias
decoder.blocks.1.conv2.bn.num_batches_tracked
decoder.blocks.1.conv2.bn.running_mean
decoder.blocks.1.conv2.bn.running_var
decoder.blocks.1.conv2.bn.weight
decoder.blocks.1.conv2.conv.bias
decoder.blocks.1.conv2.conv.weight
decoder.blocks.1.upsample.1.bias
decoder.blocks.1.upsample.1.weight
decoder.blocks.1.upsample.2.bias
decoder.blocks.1.upsample.2.num_batches_tracked
decoder.blocks.1.upsample.2.running_mean
decoder.blocks.1.upsample.2.running_var
decoder.blocks.1.upsample.2.weight
decoder.blocks.2.conv1.bn.bias
decoder.blocks.2.conv1.bn.num_batches_tracked
decoder.blocks.2.conv1.bn.running_mean
decoder.blocks.2.conv1.bn.running_var
decoder.blocks.2.conv1.bn.weight
decoder.blocks.2.conv1.conv.bias
decoder.blocks.2.conv1.conv.weight
decoder.blocks.2.conv2.bn.bias
decoder.blocks.2.conv2.bn.num_batches_tracked
decoder.blocks.2.conv2.bn.running_mean
decoder.blocks.2.conv2.bn.running_var
decoder.blocks.2.conv2.bn.weight
decoder.blocks.2.conv2.conv.bias
decoder.blocks.2.conv2.conv.weight
decoder.blocks.2.upsample.1.bias
decoder.blocks.2.upsample.1.weight
decoder.blocks.2.upsample.2.bias
decoder.blocks.2.upsample.2.num_batches_tracked
decoder.blocks.2.upsample.2.running_mean
decoder.blocks.2.upsample.2.running_var
decoder.blocks.2.upsample.2.weight
encoder.blocks.0.conv1.bn.bias
encoder.blocks.0.conv1.bn.num_batches_tracked
encoder.blocks.0.conv1.bn.running_mean
encoder.blocks.0.conv1.bn.running_var
encoder.blocks.0.conv1.bn.weight
encoder.blocks.0.conv1.conv.bias
encoder.blocks.0.conv1.conv.weight
encoder.blocks.0.conv2.bn.bias
encoder.blocks.0.conv2.bn.num_batches_tracked
encoder.blocks.0.conv2.bn.running_mean
encoder.blocks.0.conv2.bn.running_var
encoder.blocks.0.conv2.bn.weight
encoder.blocks.0.conv2.conv.bias
encoder.blocks.0.conv2.conv.weight
encoder.blocks.0.dwt.hi_filter
encoder.blocks.0.dwt.lo_filter
encoder.blocks.1.conv1.bn.bias
encoder.blocks.1.conv1.bn.num_batches_tracked
encoder.blocks.1.conv1.bn.running_mean
encoder.blocks.1.conv1.bn.running_var
encoder.blocks.1.conv1.bn.weight
encoder.blocks.1.conv1.conv.bias
encoder.blocks.1.conv1.conv.weight
encoder.blocks.1.conv2.bn.bias
encoder.blocks.1.conv2.bn.num_batches_tracked
encoder.blocks.1.conv2.bn.running_mean
encoder.blocks.1.conv2.bn.running_var
encoder.blocks.1.conv2.bn.weight
encoder.blocks.1.conv2.conv.bias
encoder.blocks.1.conv2.conv.weight
encoder.blocks.1.dwt.hi_filter
encoder.blocks.1.dwt.lo_filter
encoder.blocks.2.conv1.bn.bias
encoder.blocks.2.conv1.bn.num_batches_tracked
encoder.blocks.2.conv1.bn.running_mean
encoder.blocks.2.conv1.bn.running_var
encoder.blocks.2.conv1.bn.weight
encoder.blocks.2.conv1.conv.bias
encoder.blocks.2.conv1.conv.weight
encoder.blocks.2.conv2.bn.bias
encoder.blocks.2.conv2.bn.num_batches_tracked
encoder.blocks.2.conv2.bn.running_mean
encoder.blocks.2.conv2.bn.running_var
encoder.blocks.2.conv2.bn.weight
encoder.blocks.2.conv2.conv.bias
encoder.blocks.2.conv2.conv.weight
encoder.blocks.2.dwt.hi_filter
encoder.blocks.2.dwt.lo_filter
input_proj.bias
input_proj.weight
pre_bottleneck.bias
pre_bottleneck.weight
recon_head.conv.0.bias
recon_head.conv.0.weight
recon_head.conv.1.bias
recon_head.conv.1.num_batches_tracked
recon_head.conv.1.running_mean
recon_head.conv.1.running_var
recon_head.conv.1.weight
recon_head.conv.3.bias
recon_head.conv.3.weight
seg_head.classifier.bias
seg_head.classifier.weight
seg_head.fusion.0.bias
seg_head.fusion.0.weight
seg_head.fusion.1.bias
seg_head.fusion.1.num_batches_tracked
seg_head.fusion.1.running_mean
seg_head.fusion.1.running_var
seg_head.fusion.1.weight
seg_head.fusion.4.bias
seg_head.fusion.4.weight
seg_head.fusion.5.bias
seg_head.fusion.5.num_batches_tracked
seg_head.fusion.5.running_mean
seg_head.fusion.5.running_var
seg_head.fusion.5.weight
seg_head.global_conv.0.bias
seg_head.global_conv.0.weight
seg_head.global_conv.1.bias
seg_head.global_conv.1.num_batches_tracked
seg_head.global_conv.1.running_mean
seg_head.global_conv.1.running_var
seg_head.global_conv.1.weight
seg_head.local_conv.0.bias
seg_head.local_conv.0.weight
seg_head.local_conv.1.bias
seg_head.local_conv.1.num_batches_tracked
seg_head.local_conv.1.running_mean
seg_head.local_conv.1.running_var
seg_head.local_conv.1.weight
```

### MR-TAE-FULL-LossB

```json
{
  "model_id": "MR-TAE-FULL-LossB",
  "checkpoint": "results/MR-TAE-FULL-LossB/checkpoints/best.pt",
  "exists": true,
  "checkpoint_keys": [
    "epoch",
    "model_state",
    "optimizer_state",
    "scheduler_state",
    "criterion_state",
    "scaler_state",
    "best_val_ncc",
    "history",
    "model_id"
  ],
  "model_state_dict_present": false,
  "state_key_used_for_equivalent_strict_load": "model_state",
  "strict_load_succeeded": false,
  "runtime_error": "Error(s) in loading state_dict for GenericUNetDenoiser:"
}
```

Missing keys (122):

```text
attn1.gate_proj.bias
attn1.gate_proj.weight
attn1.score.bias
attn1.score.weight
attn1.skip_proj.bias
attn1.skip_proj.weight
attn2.gate_proj.bias
attn2.gate_proj.weight
attn2.score.bias
attn2.score.weight
attn2.skip_proj.bias
attn2.skip_proj.weight
bottleneck.0.gru.bias_hh_l0
bottleneck.0.gru.bias_hh_l0_reverse
bottleneck.0.gru.bias_hh_l1
bottleneck.0.gru.bias_hh_l1_reverse
bottleneck.0.gru.bias_ih_l0
bottleneck.0.gru.bias_ih_l0_reverse
bottleneck.0.gru.bias_ih_l1
bottleneck.0.gru.bias_ih_l1_reverse
bottleneck.0.gru.weight_hh_l0
bottleneck.0.gru.weight_hh_l0_reverse
bottleneck.0.gru.weight_hh_l1
bottleneck.0.gru.weight_hh_l1_reverse
bottleneck.0.gru.weight_ih_l0
bottleneck.0.gru.weight_ih_l0_reverse
bottleneck.0.gru.weight_ih_l1
bottleneck.0.gru.weight_ih_l1_reverse
bottleneck.0.out.bias
bottleneck.0.out.weight
bottleneck.0.proj.bias
bottleneck.0.proj.weight
bottleneck.1.attn.in_proj_bias
bottleneck.1.attn.in_proj_weight
bottleneck.1.attn.out_proj.bias
bottleneck.1.attn.out_proj.weight
bottleneck.1.ffn.0.bias
bottleneck.1.ffn.0.weight
bottleneck.1.ffn.2.bias
bottleneck.1.ffn.2.weight
bottleneck.1.norm1.bias
bottleneck.1.norm1.weight
bottleneck.1.norm2.bias
bottleneck.1.norm2.weight
dec1.net.0.bias
dec1.net.0.weight
dec1.net.1.bias
dec1.net.1.num_batches_tracked
dec1.net.1.running_mean
dec1.net.1.running_var
dec1.net.1.weight
dec1.net.3.bias
dec1.net.3.weight
dec1.net.4.bias
dec1.net.4.num_batches_tracked
dec1.net.4.running_mean
dec1.net.4.running_var
dec1.net.4.weight
dec2.net.0.bias
dec2.net.0.weight
dec2.net.1.bias
dec2.net.1.num_batches_tracked
dec2.net.1.running_mean
dec2.net.1.running_var
dec2.net.1.weight
dec2.net.3.bias
dec2.net.3.weight
dec2.net.4.bias
dec2.net.4.num_batches_tracked
dec2.net.4.running_mean
dec2.net.4.running_var
dec2.net.4.weight
enc1.net.0.bias
enc1.net.0.weight
enc1.net.1.bias
enc1.net.1.num_batches_tracked
enc1.net.1.running_mean
enc1.net.1.running_var
enc1.net.1.weight
enc1.net.3.bias
enc1.net.3.weight
enc1.net.4.bias
enc1.net.4.num_batches_tracked
enc1.net.4.running_mean
enc1.net.4.running_var
enc1.net.4.weight
enc2.net.0.bias
enc2.net.0.weight
enc2.net.1.bias
enc2.net.1.num_batches_tracked
enc2.net.1.running_mean
enc2.net.1.running_var
enc2.net.1.weight
enc2.net.3.bias
enc2.net.3.weight
enc2.net.4.bias
enc2.net.4.num_batches_tracked
enc2.net.4.running_mean
enc2.net.4.running_var
enc2.net.4.weight
enc3.net.0.bias
enc3.net.0.weight
enc3.net.1.bias
enc3.net.1.num_batches_tracked
enc3.net.1.running_mean
enc3.net.1.running_var
enc3.net.1.weight
enc3.net.3.bias
enc3.net.3.weight
enc3.net.4.bias
enc3.net.4.num_batches_tracked
enc3.net.4.running_mean
enc3.net.4.running_var
enc3.net.4.weight
out.bias
out.weight
seg.bias
seg.weight
up1.up.bias
up1.up.weight
up2.up.bias
up2.up.weight
```

Unexpected keys (208):

```text
bottleneck.gru.bias_hh_l0
bottleneck.gru.bias_hh_l0_reverse
bottleneck.gru.bias_hh_l1
bottleneck.gru.bias_hh_l1_reverse
bottleneck.gru.bias_ih_l0
bottleneck.gru.bias_ih_l0_reverse
bottleneck.gru.bias_ih_l1
bottleneck.gru.bias_ih_l1_reverse
bottleneck.gru.weight_hh_l0
bottleneck.gru.weight_hh_l0_reverse
bottleneck.gru.weight_hh_l1
bottleneck.gru.weight_hh_l1_reverse
bottleneck.gru.weight_ih_l0
bottleneck.gru.weight_ih_l0_reverse
bottleneck.gru.weight_ih_l1
bottleneck.gru.weight_ih_l1_reverse
bottleneck.gru_to_swin.bias
bottleneck.gru_to_swin.weight
bottleneck.output_proj.bias
bottleneck.output_proj.weight
bottleneck.pre_gru.bias
bottleneck.pre_gru.weight
bottleneck.residual.bias
bottleneck.residual.weight
bottleneck.swin.blocks.0.attn.proj.bias
bottleneck.swin.blocks.0.attn.proj.weight
bottleneck.swin.blocks.0.attn.qkv.bias
bottleneck.swin.blocks.0.attn.qkv.weight
bottleneck.swin.blocks.0.attn.relative_position_bias_table
bottleneck.swin.blocks.0.attn.relative_position_index
bottleneck.swin.blocks.0.mlp.fc1.bias
bottleneck.swin.blocks.0.mlp.fc1.weight
bottleneck.swin.blocks.0.mlp.fc2.bias
bottleneck.swin.blocks.0.mlp.fc2.weight
bottleneck.swin.blocks.0.norm1.bias
bottleneck.swin.blocks.0.norm1.weight
bottleneck.swin.blocks.0.norm2.bias
bottleneck.swin.blocks.0.norm2.weight
bottleneck.swin.blocks.1.attn.proj.bias
bottleneck.swin.blocks.1.attn.proj.weight
bottleneck.swin.blocks.1.attn.qkv.bias
bottleneck.swin.blocks.1.attn.qkv.weight
bottleneck.swin.blocks.1.attn.relative_position_bias_table
bottleneck.swin.blocks.1.attn.relative_position_index
bottleneck.swin.blocks.1.mlp.fc1.bias
bottleneck.swin.blocks.1.mlp.fc1.weight
bottleneck.swin.blocks.1.mlp.fc2.bias
bottleneck.swin.blocks.1.mlp.fc2.weight
bottleneck.swin.blocks.1.norm1.bias
bottleneck.swin.blocks.1.norm1.weight
bottleneck.swin.blocks.1.norm2.bias
bottleneck.swin.blocks.1.norm2.weight
bottleneck.swin.norm.bias
bottleneck.swin.norm.weight
decoder.blocks.0.conv1.bn.bias
decoder.blocks.0.conv1.bn.num_batches_tracked
decoder.blocks.0.conv1.bn.running_mean
decoder.blocks.0.conv1.bn.running_var
decoder.blocks.0.conv1.bn.weight
decoder.blocks.0.conv1.conv.bias
decoder.blocks.0.conv1.conv.weight
decoder.blocks.0.conv2.bn.bias
decoder.blocks.0.conv2.bn.num_batches_tracked
decoder.blocks.0.conv2.bn.running_mean
decoder.blocks.0.conv2.bn.running_var
decoder.blocks.0.conv2.bn.weight
decoder.blocks.0.conv2.conv.bias
decoder.blocks.0.conv2.conv.weight
decoder.blocks.0.upsample.1.bias
decoder.blocks.0.upsample.1.weight
decoder.blocks.0.upsample.2.bias
decoder.blocks.0.upsample.2.num_batches_tracked
decoder.blocks.0.upsample.2.running_mean
decoder.blocks.0.upsample.2.running_var
decoder.blocks.0.upsample.2.weight
decoder.blocks.1.conv1.bn.bias
decoder.blocks.1.conv1.bn.num_batches_tracked
decoder.blocks.1.conv1.bn.running_mean
decoder.blocks.1.conv1.bn.running_var
decoder.blocks.1.conv1.bn.weight
decoder.blocks.1.conv1.conv.bias
decoder.blocks.1.conv1.conv.weight
decoder.blocks.1.conv2.bn.bias
decoder.blocks.1.conv2.bn.num_batches_tracked
decoder.blocks.1.conv2.bn.running_mean
decoder.blocks.1.conv2.bn.running_var
decoder.blocks.1.conv2.bn.weight
decoder.blocks.1.conv2.conv.bias
decoder.blocks.1.conv2.conv.weight
decoder.blocks.1.upsample.1.bias
decoder.blocks.1.upsample.1.weight
decoder.blocks.1.upsample.2.bias
decoder.blocks.1.upsample.2.num_batches_tracked
decoder.blocks.1.upsample.2.running_mean
decoder.blocks.1.upsample.2.running_var
decoder.blocks.1.upsample.2.weight
decoder.blocks.2.conv1.bn.bias
decoder.blocks.2.conv1.bn.num_batches_tracked
decoder.blocks.2.conv1.bn.running_mean
decoder.blocks.2.conv1.bn.running_var
decoder.blocks.2.conv1.bn.weight
decoder.blocks.2.conv1.conv.bias
decoder.blocks.2.conv1.conv.weight
decoder.blocks.2.conv2.bn.bias
decoder.blocks.2.conv2.bn.num_batches_tracked
decoder.blocks.2.conv2.bn.running_mean
decoder.blocks.2.conv2.bn.running_var
decoder.blocks.2.conv2.bn.weight
decoder.blocks.2.conv2.conv.bias
decoder.blocks.2.conv2.conv.weight
decoder.blocks.2.upsample.1.bias
decoder.blocks.2.upsample.1.weight
decoder.blocks.2.upsample.2.bias
decoder.blocks.2.upsample.2.num_batches_tracked
decoder.blocks.2.upsample.2.running_mean
decoder.blocks.2.upsample.2.running_var
decoder.blocks.2.upsample.2.weight
encoder.blocks.0.conv1.bn.bias
encoder.blocks.0.conv1.bn.num_batches_tracked
encoder.blocks.0.conv1.bn.running_mean
encoder.blocks.0.conv1.bn.running_var
encoder.blocks.0.conv1.bn.weight
encoder.blocks.0.conv1.conv.bias
encoder.blocks.0.conv1.conv.weight
encoder.blocks.0.conv2.bn.bias
encoder.blocks.0.conv2.bn.num_batches_tracked
encoder.blocks.0.conv2.bn.running_mean
encoder.blocks.0.conv2.bn.running_var
encoder.blocks.0.conv2.bn.weight
encoder.blocks.0.conv2.conv.bias
encoder.blocks.0.conv2.conv.weight
encoder.blocks.0.dwt.hi_filter
encoder.blocks.0.dwt.lo_filter
encoder.blocks.1.conv1.bn.bias
encoder.blocks.1.conv1.bn.num_batches_tracked
encoder.blocks.1.conv1.bn.running_mean
encoder.blocks.1.conv1.bn.running_var
encoder.blocks.1.conv1.bn.weight
encoder.blocks.1.conv1.conv.bias
encoder.blocks.1.conv1.conv.weight
encoder.blocks.1.conv2.bn.bias
encoder.blocks.1.conv2.bn.num_batches_tracked
encoder.blocks.1.conv2.bn.running_mean
encoder.blocks.1.conv2.bn.running_var
encoder.blocks.1.conv2.bn.weight
encoder.blocks.1.conv2.conv.bias
encoder.blocks.1.conv2.conv.weight
encoder.blocks.1.dwt.hi_filter
encoder.blocks.1.dwt.lo_filter
encoder.blocks.2.conv1.bn.bias
encoder.blocks.2.conv1.bn.num_batches_tracked
encoder.blocks.2.conv1.bn.running_mean
encoder.blocks.2.conv1.bn.running_var
encoder.blocks.2.conv1.bn.weight
encoder.blocks.2.conv1.conv.bias
encoder.blocks.2.conv1.conv.weight
encoder.blocks.2.conv2.bn.bias
encoder.blocks.2.conv2.bn.num_batches_tracked
encoder.blocks.2.conv2.bn.running_mean
encoder.blocks.2.conv2.bn.running_var
encoder.blocks.2.conv2.bn.weight
encoder.blocks.2.conv2.conv.bias
encoder.blocks.2.conv2.conv.weight
encoder.blocks.2.dwt.hi_filter
encoder.blocks.2.dwt.lo_filter
input_proj.bias
input_proj.weight
pre_bottleneck.bias
pre_bottleneck.weight
recon_head.conv.0.bias
recon_head.conv.0.weight
recon_head.conv.1.bias
recon_head.conv.1.num_batches_tracked
recon_head.conv.1.running_mean
recon_head.conv.1.running_var
recon_head.conv.1.weight
recon_head.conv.3.bias
recon_head.conv.3.weight
seg_head.classifier.bias
seg_head.classifier.weight
seg_head.fusion.0.bias
seg_head.fusion.0.weight
seg_head.fusion.1.bias
seg_head.fusion.1.num_batches_tracked
seg_head.fusion.1.running_mean
seg_head.fusion.1.running_var
seg_head.fusion.1.weight
seg_head.fusion.4.bias
seg_head.fusion.4.weight
seg_head.fusion.5.bias
seg_head.fusion.5.num_batches_tracked
seg_head.fusion.5.running_mean
seg_head.fusion.5.running_var
seg_head.fusion.5.weight
seg_head.global_conv.0.bias
seg_head.global_conv.0.weight
seg_head.global_conv.1.bias
seg_head.global_conv.1.num_batches_tracked
seg_head.global_conv.1.running_mean
seg_head.global_conv.1.running_var
seg_head.global_conv.1.weight
seg_head.local_conv.0.bias
seg_head.local_conv.0.weight
seg_head.local_conv.1.bias
seg_head.local_conv.1.num_batches_tracked
seg_head.local_conv.1.running_mean
seg_head.local_conv.1.running_var
seg_head.local_conv.1.weight
```

### noBiGRU Weight Presence Check

Direct checkpoint key inspection shows the workspace `MR-TAE-noBiGRU` checkpoint has no keys containing `gru`, while `MR-TAE-FULL` does. This supports that the file labelled `MR-TAE-noBiGRU` does not carry BiGRU weights, even though the current registry builder cannot strict-load these checkpoints.

`MR-TAE-FULL` GRU-like keys:

```text
bottleneck.gru.bias_hh_l0
bottleneck.gru.bias_hh_l0_reverse
bottleneck.gru.bias_hh_l1
bottleneck.gru.bias_hh_l1_reverse
bottleneck.gru.bias_ih_l0
bottleneck.gru.bias_ih_l0_reverse
bottleneck.gru.bias_ih_l1
bottleneck.gru.bias_ih_l1_reverse
bottleneck.gru.weight_hh_l0
bottleneck.gru.weight_hh_l0_reverse
bottleneck.gru.weight_hh_l1
bottleneck.gru.weight_hh_l1_reverse
bottleneck.gru.weight_ih_l0
bottleneck.gru.weight_ih_l0_reverse
bottleneck.gru.weight_ih_l1
bottleneck.gru.weight_ih_l1_reverse
bottleneck.gru_to_swin.bias
bottleneck.gru_to_swin.weight
bottleneck.pre_gru.bias
bottleneck.pre_gru.weight
```

`MR-TAE-noBiGRU` GRU-like keys:

```text

```

## 5. BENCHMARK JSON DUMP

### results/MR-TAE-FULL/benchmark_results.json

Path: `results/MR-TAE-FULL/benchmark_results.json` lines 1-93

```json
{
  "model_id": "MR-TAE-FULL",
  "checkpoint": "results\\MR-TAE-FULL\\checkpoints\\best.pt",
  "ablation_source": "D:\\New folder (2)\\Data\\FYP_MASTER\\results\\MR-TAE-FULL\\checkpoints\\best_config.json",
  "hybrid_savgol_gamma_nn": null,
  "synthetic": {
    "ncc_mean": 0.6122567653656006,
    "ncc_std": 0.301918089389801,
    "rmse_mean": 0.3692325949668884,
    "rmse_std": 0.3155422508716583,
    "snr_improvement_mean": 5.34899377822876,
    "snr_improvement_std": 2.018160104751587,
    "dice_mean": 0.9022210836410522,
    "dice_std": 0.07602908462285995,
    "false_positive_rate": 0.3511075973510742,
    "per_snr_ncc": {
      "-15": {
        "ncc_mean": 0.19506466388702393,
        "ncc_std": 0.02991328574717045
      },
      "-10": {
        "ncc_mean": 0.3510521352291107,
        "ncc_std": 0.03030741959810257
      },
      "-5": {
        "ncc_mean": 0.6513131260871887,
        "ncc_std": 0.05434517562389374
      },
      "0": {
        "ncc_mean": 0.8994240164756775,
        "ncc_std": 0.024615200236439705
      },
      "5": {
        "ncc_mean": 0.9644295573234558,
        "ncc_std": 0.010330576449632645
      }
    }
  },
  "real_qlin": {
    "ncc_mean": 0.9601970911026001,
    "ncc_std": 0.03543743118643761,
    "rmse_mean": 0.2861595153808594,
    "rmse_std": 0.09859599173069,
    "n_windows_evaluated": 20000,
    "reference": "input_window",
    "note": "No clean ground truth; NCC/RMSE vs observed window."
  },
  "real_qlin_grounded": {
    "ncc_mean": 0.4738312065601349,
    "ncc_std": 0.32485318183898926,
    "rmse_mean": 2.7287158966064453,
    "rmse_std": 3.1246585845947266,
    "n_windows_evaluated": 10000,
    "reference": "clean_ground_truth",
    "per_snr": {
      "-15": {
        "ncc_mean": 0.04675921052694321,
        "ncc_std": 0.09113112837076187,
        "rmse_mean": 8.331062316894531,
        "rmse_std": 2.414287567138672
      },
      "-10": {
        "ncc_mean": 0.18536314368247986,
        "ncc_std": 0.0915675014257431,
        "rmse_mean": 2.844464063644409,
        "rmse_std": 0.6363118290901184
      },
      "-5": {
        "ncc_mean": 0.5152871608734131,
        "ncc_std": 0.0638788565993309,
        "rmse_mean": 1.218180775642395,
        "rmse_std": 0.10914023220539093
      },
      "0": {
        "ncc_mean": 0.7408849596977234,
        "ncc_std": 0.0395129956305027,
        "rmse_mean": 0.756240963935852,
        "rmse_std": 0.04139508306980133
      },
      "5": {
        "ncc_mean": 0.8808615207672119,
        "ncc_std": 0.031661756336688995,
        "rmse_mean": 0.4936310946941376,
        "rmse_std": 0.049903903156518936
      }
    }
  },
  "efficiency": {
    "param_count_M": 3.0097,
    "inference_latency_ms_b1": 12.5456,
    "inference_latency_ms_b32": 0.4826
  }
}
```

### results/MR-TAE-FULL-LossB/benchmark_results.json

Path: `results/MR-TAE-FULL-LossB/benchmark_results.json` lines 1-93

```json
{
  "model_id": "MR-TAE-FULL-LossB",
  "checkpoint": "results\\MR-TAE-FULL-LossB\\checkpoints\\best.pt",
  "ablation_source": "config/model_registry.yaml + CLI --no-attention-gates",
  "hybrid_savgol_gamma_nn": null,
  "synthetic": {
    "ncc_mean": 0.610988974571228,
    "ncc_std": 0.301825612783432,
    "rmse_mean": 0.3975984752178192,
    "rmse_std": 0.35342904925346375,
    "snr_improvement_mean": 5.030482769012451,
    "snr_improvement_std": 2.3246216773986816,
    "dice_mean": 0.9031110405921936,
    "dice_std": 0.07541877031326294,
    "false_positive_rate": 0.35067611932754517,
    "per_snr_ncc": {
      "-15": {
        "ncc_mean": 0.19932422041893005,
        "ncc_std": 0.038374174386262894
      },
      "-10": {
        "ncc_mean": 0.34831029176712036,
        "ncc_std": 0.02801765315234661
      },
      "-5": {
        "ncc_mean": 0.6403922438621521,
        "ncc_std": 0.0564785860478878
      },
      "0": {
        "ncc_mean": 0.9014906287193298,
        "ncc_std": 0.024280980229377747
      },
      "5": {
        "ncc_mean": 0.9654275178909302,
        "ncc_std": 0.010303721763193607
      }
    }
  },
  "real_qlin": {
    "ncc_mean": 0.8673524260520935,
    "ncc_std": 0.06305426359176636,
    "rmse_mean": 0.5147815942764282,
    "rmse_std": 0.1429399698972702,
    "n_windows_evaluated": 20000,
    "reference": "input_window",
    "note": "No clean ground truth; NCC/RMSE vs observed window."
  },
  "real_qlin_grounded": {
    "ncc_mean": 0.5314804911613464,
    "ncc_std": 0.23211736977100372,
    "rmse_mean": 1.6569781303405762,
    "rmse_std": 0.9982568025588989,
    "n_windows_evaluated": 10000,
    "reference": "clean_ground_truth",
    "per_snr": {
      "-15": {
        "ncc_mean": 0.21689052879810333,
        "ncc_std": 0.0754748061299324,
        "rmse_mean": 3.397629976272583,
        "rmse_std": 0.06834377348423004
      },
      "-10": {
        "ncc_mean": 0.36616525053977966,
        "ncc_std": 0.09889700263738632,
        "rmse_mean": 2.0553548336029053,
        "rmse_std": 0.06811966001987457
      },
      "-5": {
        "ncc_mean": 0.548606812953949,
        "ncc_std": 0.08968707174062729,
        "rmse_mean": 1.3001550436019897,
        "rmse_std": 0.08734365552663803
      },
      "0": {
        "ncc_mean": 0.7092244029045105,
        "ncc_std": 0.05826956778764725,
        "rmse_mean": 0.8883840441703796,
        "rmse_std": 0.12087533622980118
      },
      "5": {
        "ncc_mean": 0.8165154457092285,
        "ncc_std": 0.0552186444401741,
        "rmse_mean": 0.6433669328689575,
        "rmse_std": 0.13035376369953156
      }
    }
  },
  "efficiency": {
    "param_count_M": 3.0097,
    "inference_latency_ms_b1": 9.4961,
    "inference_latency_ms_b32": 1.2968
  }
}
```

Prompt text after this item was truncated at `resu`; no additional benchmark path was visible in the request.
