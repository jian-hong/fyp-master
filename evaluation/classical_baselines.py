"""
Classical denoising baselines evaluated with verified NCC vs clean synthetic GT.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import savgol_filter, wiener

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics_verified import ncc_synthetic  # noqa: E402

SNR_LEVELS = (-15, -10, -5, 0, 5)
OUT_PATH = PROJECT_ROOT / "results" / "definitive_benchmark" / "baselines.json"


def apply_wiener(x_noisy: np.ndarray, mysize: int = 5) -> np.ndarray:
    x = np.asarray(x_noisy).squeeze()
    return wiener(x, mysize=mysize)


def apply_bayesshrink_manual(
    x_noisy: np.ndarray,
    wavelet: str = "db4",
    level: int = 3,
) -> np.ndarray:
    import pywt

    x = np.asarray(x_noisy).squeeze()
    coeffs = pywt.wavedec(x, wavelet, level=level)
    sigma_n = np.median(np.abs(coeffs[-1])) / 0.6745
    denoised_coeffs = [coeffs[0]]
    for detail in coeffs[1:]:
        sigma_x = np.sqrt(max(np.var(detail) - sigma_n**2, 0))
        threshold = sigma_n**2 / (sigma_x + 1e-8)
        denoised_coeffs.append(pywt.threshold(detail, threshold, mode="soft"))
    return pywt.waverec(denoised_coeffs, wavelet)[: len(x)]


def apply_savgol(x_noisy: np.ndarray, window: int = 51, poly: int = 3) -> np.ndarray:
    x = np.asarray(x_noisy).squeeze()
    window = min(window, len(x) - (1 if len(x) % 2 == 0 else 0))
    if window % 2 == 0:
        window -= 1
    if window < poly + 2:
        window = poly + 2 + (poly % 2)
    return savgol_filter(x, window_length=window, polyorder=poly)


def _load_synthetic_test() -> dict:
    path = PROJECT_ROOT / "data" / "test_synthetic_ae.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing synthetic test set: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _snr_mask(snr_labels: torch.Tensor | None, snr: int, n: int) -> np.ndarray:
    if snr_labels is not None:
        sl = snr_labels.cpu().numpy().ravel()
        idx = np.where(np.abs(sl - float(snr)) <= 2.6)[0]
        if len(idx):
            return idx.astype(np.int64)
    per = 2000
    i = list(SNR_LEVELS).index(snr)
    s, e = i * per, (i + 1) * per
    if e <= n:
        return np.arange(s, e, dtype=np.int64)
    return np.array([], dtype=np.int64)


def run_baselines(max_per_snr: int | None = 200) -> dict:
    test = _load_synthetic_test()
    noisy = test["noisy"] if "noisy" in test else test["windows"]
    clean = test["clean"]
    snr_labels = test.get("snr")

    if noisy.dim() == 3 and noisy.shape[1] == 1:
        noisy_np = noisy.squeeze(1).numpy()
        clean_np = clean.squeeze(1).numpy()
    else:
        noisy_np = noisy.numpy()
        clean_np = clean.numpy()

    methods = [
        ("wiener", apply_wiener),
        ("bayesshrink", apply_bayesshrink_manual),
        ("savgol_51", apply_savgol),
    ]
    results: dict = {}
    n_total = len(noisy_np)

    for method_name, method_fn in methods:
        nccs_by_snr: dict[str, float] = {}
        for snr in SNR_LEVELS:
            idx = _snr_mask(snr_labels, snr, n_total)
            if len(idx) == 0:
                nccs_by_snr[str(snr)] = float("nan")
                continue
            if max_per_snr and len(idx) > max_per_snr:
                rng = np.random.default_rng(42)
                idx = rng.choice(idx, size=max_per_snr, replace=False)
            nccs = []
            for i in idx:
                out = method_fn(noisy_np[i])
                out = out[: clean_np[i].shape[0]]
                nccs.append(ncc_synthetic(out, clean_np[i]))
            nccs_by_snr[str(snr)] = float(np.nanmean(nccs))
            print(f"{method_name} @ {snr}dB: NCC={nccs_by_snr[str(snr)]:.4f}")
        results[method_name] = nccs_by_snr

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {OUT_PATH}")
    return results


if __name__ == "__main__":
    run_baselines()
