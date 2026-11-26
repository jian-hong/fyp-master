"""
Authoritative zero-mean normalized cross-correlation (NCC) for AE denoising benchmarks.

NCC(a, b) = sum((a - mean(a)) * (b - mean(b))) / sqrt(sum((a-mean(a))^2) * sum((b-mean(b))^2))

Synthetic evaluation: compare denoised output vs clean ground truth.
Real evaluation (no GT): compare denoised output vs noisy input — label as Input-Ref NCC.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


def ncc_1d(a: ArrayLike, b: ArrayLike, eps: float = 1e-8) -> float:
    """Scalar zero-mean NCC between two 1D signals."""
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().cpu().numpy()
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n = min(len(a), len(b))
    if n < 2:
        return float("nan")
    a, b = a[:n], b[:n]
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b)) + eps
    if denom <= eps:
        return float("nan")
    return float(np.sum(a * b) / denom)


def ncc_batch(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> np.ndarray:
    """
    Batched zero-mean NCC. pred/target: (B, C, L) or (B, L).
    Returns shape (B,) numpy array.
    """
    if pred.dim() == 2:
        pred = pred.unsqueeze(1)
    if target.dim() == 2:
        target = target.unsqueeze(1)
    p = pred.float().flatten(1)
    t = target.float().flatten(1)
    p = p - p.mean(1, keepdim=True)
    t = t - t.mean(1, keepdim=True)
    num = (p * t).sum(1)
    den = torch.sqrt((p**2).sum(1) * (t**2).sum(1)) + eps
    return (num / den).cpu().numpy()


def ncc_synthetic(pred: ArrayLike, clean_gt: ArrayLike) -> float:
    """NCC vs clean ground truth (synthetic test set)."""
    return ncc_1d(pred, clean_gt)


def ncc_input_ref(pred: ArrayLike, noisy_input: ArrayLike) -> float:
    """Input-referenced NCC on real data (no clean GT available)."""
    return ncc_1d(pred, noisy_input)
