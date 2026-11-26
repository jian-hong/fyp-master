"""
evaluation/benchmark_runner.py — Master evaluation script with baselines and figures.

Evaluates trained checkpoints and classical denoisers on:
  1. Synthetic AE test set   (data/test_synthetic_ae.pt)
  2. Real Q.Lin test set     (data/test_real_qlin.pt)

Real-set metrics use the input window as the reference signal (no clean ground truth).

Usage
-----
python evaluation/benchmark_runner.py --model-id MR-TAE-FULL \\
    --checkpoint results/MR-TAE-FULL/checkpoints/best.pt

python evaluation/benchmark_runner.py --compare-all
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from scipy import signal as scipy_signal
from torch.amp import autocast
from torch.utils.data import DataLoader, TensorDataset

try:
    import pywt
except ImportError:
    pywt = None  # type: ignore

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None  # type: ignore
    sns = None  # type: ignore

try:
    import mlflow

    _MLFLOW = True
except ImportError:
    _MLFLOW = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Try Model"))

SYNTH_TEST_PATH = "data/test_synthetic_ae.pt"
REAL_TEST_PATH = "data/test_real_qlin.pt"
REAL_CLEAN_PATH = "data/test_real_qlin_clean.pt"
REAL_NOISY_PATH = "data/test_real_qlin_noisy.pt"
REAL_SNR_LABELS_PATH = "data/test_real_qlin_snr_labels.pt"
SNR_LEVELS: Tuple[int, ...] = (-15, -10, -5, 0, 5)
SYNTH_WINDOWS_PER_SNR = 2000
LOG_PATH = "logs/benchmark.log"
DOCS_RESULTS = "docs/RESULTS_COMPARISON.md"

BASELINE_WAVELET_ID = "Baseline-Wavelet-BayesShrink"
BASELINE_SAVGOL_ID = "Baseline-Savitzky-Golay"

LOSS_VARIANT_IDS = ("MR-TAE-FULL", "MR-TAE-FULL-LossB", "MR-TAE-FULL-LossC")

logger = logging.getLogger("benchmark")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _setup_logging(log_file: Path | None = None) -> None:
    log_file = log_file or (PROJECT_ROOT / LOG_PATH)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(ch)
    root.setLevel(logging.DEBUG)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


from evaluation.metrics_verified import ncc_batch as _ncc_batch


def _rmse_batch(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    if pred.dim() == 2:
        pred = pred.unsqueeze(1)
    if target.dim() == 2:
        target = target.unsqueeze(1)
    return torch.sqrt(((pred.float() - target.float()) ** 2).mean(-1)).cpu().numpy().ravel()


def _snr_improvement(pred: torch.Tensor, clean: torch.Tensor, noisy: torch.Tensor) -> np.ndarray:
    if pred.dim() == 2:
        pred = pred.unsqueeze(1)
    if clean.dim() == 2:
        clean = clean.unsqueeze(1)
    if noisy.dim() == 2:
        noisy = noisy.unsqueeze(1)
    noise_in = ((noisy.float() - clean.float()) ** 2).mean(-1)
    noise_out = ((pred.float() - clean.float()) ** 2).mean(-1)
    signal = (clean.float() ** 2).mean(-1)
    snr_in = 10 * torch.log10(signal / (noise_in + 1e-10))
    snr_out = 10 * torch.log10(signal / (noise_out + 1e-10))
    return (snr_out - snr_in).cpu().numpy().ravel()


def _dice_batch(pred_seg: torch.Tensor, target_mask: torch.Tensor, eps: float = 1e-6) -> np.ndarray:
    """Binary Dice (foreground: class > 0 vs mask > 0), per sample."""
    pred_bin = (pred_seg.argmax(1) > 0).float()
    tgt_bin = (target_mask > 0).float()
    inter = (pred_bin * tgt_bin).sum(dim=-1)
    denom = pred_bin.sum(dim=-1) + tgt_bin.sum(dim=-1) + eps
    return (2 * inter / denom).cpu().numpy()


def _false_positive_rate_batch(pred_seg: torch.Tensor, target_mask: torch.Tensor) -> np.ndarray:
    pred_pos = pred_seg.argmax(1) > 0
    actual_neg = target_mask == 0
    fp = torch.logical_and(pred_pos, actual_neg).float().sum(dim=-1)
    tn = actual_neg.float().sum(dim=-1)
    return (fp / (tn + fp + 1e-8)).cpu().numpy()


def _synthetic_snr_indices(
    n: int,
    snr_labels: Optional[torch.Tensor],
) -> Dict[int, np.ndarray]:
    """Map each nominal SNR level to sample indices."""
    out: Dict[int, np.ndarray] = {}
    expected = len(SNR_LEVELS) * SYNTH_WINDOWS_PER_SNR
    if n == expected:
        for i, lvl in enumerate(SNR_LEVELS):
            s = i * SYNTH_WINDOWS_PER_SNR
            e = (i + 1) * SYNTH_WINDOWS_PER_SNR
            out[lvl] = np.arange(s, e, dtype=np.int64)
        return out
    if snr_labels is not None:
        sl = snr_labels.cpu().numpy().ravel()
        for lvl in SNR_LEVELS:
            # bucket by nearest nominal level (5 dB bins)
            idx = np.where(np.abs(sl - float(lvl)) <= 2.6)[0]
            if len(idx):
                out[lvl] = idx.astype(np.int64)
        return out
    logger.warning("Could not assign per-SNR indices (n=%d); SNR sweep skipped.", n)
    return out


# ---------------------------------------------------------------------------
# Traditional baselines (CPU)
# ---------------------------------------------------------------------------


def wavelet_denoise(
    signal_1d: np.ndarray,
    wavelet: str = "db4",
    mode: str = "soft",
) -> np.ndarray:
    """1D BayesShrink-style soft thresholding in the wavelet domain (pywt)."""
    if pywt is None:
        raise RuntimeError("PyWavelets (pywt) is required for wavelet_denoise.")
    x = np.asarray(signal_1d, dtype=np.float64).ravel()
    n = len(x)
    if n < 8:
        return x.astype(np.float32)
    coeffs = pywt.wavedec(x, wavelet, mode="symmetric")
    detail_coeffs = coeffs[1:]
    if not detail_coeffs:
        return x.astype(np.float32)
    finest = detail_coeffs[-1]
    sigma = float(np.median(np.abs(finest))) / 0.6745
    sigma = max(sigma, 1e-12)

    new_coeffs: List[np.ndarray] = [coeffs[0]]
    for c in detail_coeffs:
        sig_j = float(np.std(c))
        sig_j = max(sig_j, 1e-12)
        t = (sigma**2) / sig_j
        new_coeffs.append(pywt.threshold(c, t, mode=mode))

    recon = pywt.waverec(new_coeffs, wavelet, mode="symmetric")
    return recon[:n].astype(np.float32)


def savgol_denoise(
    signal_1d: np.ndarray,
    window: int = 51,
    polyorder: int = 3,
) -> np.ndarray:
    x = np.asarray(signal_1d, dtype=np.float64).ravel()
    n = len(x)
    wl = int(window)
    if wl % 2 == 0:
        wl -= 1
    wl = min(wl, n if n % 2 == 1 else n - 1)
    wl = max(wl, polyorder + 2)
    if wl % 2 == 0:
        wl -= 1
    if wl < polyorder + 2:
        return x.astype(np.float32)
    y = scipy_signal.savgol_filter(x, window_length=wl, polyorder=polyorder, mode="interp")
    return y.astype(np.float32)


def hybrid_nn_savgol(
    preds: torch.Tensor,
    noisy: torch.Tensor,
    gamma_nn: float,
) -> torch.Tensor:
    """
    Convex blend: gamma_nn * neural + (1 - gamma_nn) * Savitzky–Golay(noisy).

    gamma_nn=1.0 is pure network output; gamma_nn=0.0 reproduces classical Savitzky–Golay.
    Calibration should use scripts/fit_hybrid_gamma.py (train split only — never test tensors).
    """
    g = float(max(0.0, min(1.0, gamma_nn)))
    dev, dt = preds.device, preds.dtype
    preds_c = preds.detach().float().cpu()
    noisy_c = noisy.detach().float().cpu()
    sg = _apply_baseline_to_batch(noisy_c, savgol_denoise).float()
    out = g * preds_c + (1.0 - g) * sg
    return out.to(device=dev, dtype=dt)


def _apply_baseline_to_batch(
    noisy: torch.Tensor,
    fn: Any,
) -> torch.Tensor:
    """noisy: (B, L) or (B,1,L) -> denoised same shape."""
    if noisy.dim() == 3:
        x = noisy.squeeze(1).cpu().numpy()
    else:
        x = noisy.cpu().numpy()
    out = np.stack([fn(row) for row in x], axis=0)
    t = torch.from_numpy(out)
    if noisy.dim() == 3:
        return t.unsqueeze(1)
    return t


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def load_model_registry() -> Dict[str, Dict[str, Any]]:
    path = PROJECT_ROOT / "config" / "model_registry.yaml"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("models", {})


def ablation_from_registry(model_id: str) -> Any:
    from mr_tae_fusion.models.ablation_config import AblationConfig

    reg = load_model_registry().get(model_id, {})
    return AblationConfig(
        use_wavelet_pooling=reg.get("use_wavelet_pooling", True),
        use_bigru=reg.get("use_bigru", True),
        use_swin=reg.get("use_swin", True),
        use_attention_gates=reg.get("use_attention_gates", True),
        use_mtl=reg.get("use_mtl", True),
        loss_config=reg.get("loss_config", "A"),
        model_id=model_id,
    )


def ablation_for_checkpoint(model_id: str, checkpoint_path: str) -> Tuple[Any, str]:
    """
    Prefer checkpoints/best_config.json (training-time flags) over registry defaults
    so benchmarks match architecture used when training (e.g. attention gates off).
    """
    from mr_tae_fusion.models.ablation_config import AblationConfig

    sidecar = Path(checkpoint_path).resolve().parent / "best_config.json"
    if sidecar.is_file():
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            return (
                AblationConfig(
                    use_wavelet_pooling=bool(data.get("use_wavelet_pooling", True)),
                    use_bigru=bool(data.get("use_bigru", True)),
                    use_swin=bool(data.get("use_swin", True)),
                    use_attention_gates=bool(data.get("use_attention_gates", True)),
                    use_mtl=bool(data.get("use_mtl", True)),
                    loss_config=str(data.get("loss_config", "A")),
                    model_id=model_id,
                ),
                str(sidecar),
            )
        except Exception as e:
            logger.warning("Could not read %s (%s) — using registry.", sidecar, e)
    return ablation_from_registry(model_id), "config/model_registry.yaml"


def encoder_col_label(use_wavelet_pooling: bool) -> str:
    return "MWCNN (wavelet)" if use_wavelet_pooling else "Plain encoder"


def bottleneck_row_label(use_bigru: bool, use_swin: bool) -> str:
    if use_bigru and use_swin:
        return "BiGRU+Swin"
    if use_bigru and not use_swin:
        return "BiGRU only"
    if not use_bigru and use_swin:
        return "Swin only"
    return "Conv only"


def registry_ablation_axes(model_id: str) -> Tuple[str, str]:
    reg = load_model_registry().get(model_id, {})
    row = bottleneck_row_label(reg.get("use_bigru", True), reg.get("use_swin", True))
    col = encoder_col_label(reg.get("use_wavelet_pooling", True))
    return row, col


# ---------------------------------------------------------------------------
# Deep model inference
# ---------------------------------------------------------------------------


def _infer_dataset(
    model: torch.nn.Module,
    dataset: TensorDataset,
    device: torch.device,
    batch_size: int = 64,
    has_mask: bool = True,
) -> Dict[str, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    preds, cleans, noisys, masks_l, pred_segs = [], [], [], [], []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            if has_mask:
                noisy, clean, mask = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            else:
                noisy, clean = batch[0].to(device), batch[1].to(device)
                mask = torch.zeros_like(clean, dtype=torch.long)

            if device.type == "cuda":
                with autocast("cuda"):
                    pred, pred_seg = model(noisy)
            else:
                pred, pred_seg = model(noisy)

            preds.append(pred.cpu())
            cleans.append(clean.cpu())
            noisys.append(noisy.cpu())
            masks_l.append(mask.cpu())
            pred_segs.append(pred_seg.cpu())

    return {
        "preds": torch.cat(preds),
        "cleans": torch.cat(cleans),
        "noisys": torch.cat(noisys),
        "masks": torch.cat(masks_l),
        "pred_segs": torch.cat(pred_segs),
    }


def _load_checkpoint_state(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("model_state", "model", "state_dict"):
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    raise KeyError("Checkpoint has no model_state / model / state_dict")


def _measure_latency_ms(
    model: torch.nn.Module,
    device: torch.device,
    batch1: int = 50,
    batch32_reps: int = 20,
) -> Tuple[float, float]:
    dummy = torch.randn(1, 1, 2048, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(5):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(batch1):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        lat_b1 = (time.perf_counter() - t0) / batch1 * 1000.0

    dummy32 = torch.randn(32, 1, 2048, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(batch32_reps):
            model(dummy32)
        if device.type == "cuda":
            torch.cuda.synchronize()
        lat_b32 = (time.perf_counter() - t0) / batch32_reps / 32.0 * 1000.0
    return lat_b1, lat_b32


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    model_id: Optional[str] = None,
    device_str: str = "cuda",
) -> torch.nn.Module:
    """Load a trained MR-TAE variant for inference (shared with figure generation)."""
    from mr_tae_fusion.config import Config
    from mr_tae_fusion.models.mr_tae_fusion import create_model

    checkpoint_path = Path(checkpoint_path)
    if model_id is None:
        model_id = checkpoint_path.resolve().parents[1].name
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    master_cfg = Config()
    ablation, _ = ablation_for_checkpoint(str(model_id), str(checkpoint_path))
    model = create_model(master_cfg.model, ablation)
    model.load_state_dict(_load_checkpoint_state(ckpt), strict=False)
    return model.to(device).eval()


def evaluate_real_with_noise(
    model: torch.nn.Module,
    device: torch.device,
    hybrid_savgol_gamma_nn: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Ground-truth real evaluation: NCC/RMSE of model output vs clean Q.Lin windows,
    for composite-noisy inputs at nominal SNRs (see test_real_qlin_* .pt files).
    """
    clean_path = PROJECT_ROOT / REAL_CLEAN_PATH
    noisy_path = PROJECT_ROOT / REAL_NOISY_PATH
    snr_path = PROJECT_ROOT / REAL_SNR_LABELS_PATH
    if not (clean_path.exists() and noisy_path.exists() and snr_path.exists()):
        logger.warning(
            "Grounded real files missing (%s, %s, %s); skip real_qlin_grounded",
            clean_path,
            noisy_path,
            snr_path,
        )
        return {}

    clean_pack = torch.load(clean_path, weights_only=False)
    noisy_pack = torch.load(noisy_path, weights_only=False)
    snr_pack = torch.load(snr_path, weights_only=False)
    clean = clean_pack["clean"]
    noisy = noisy_pack["noisy"]
    snr_labels = snr_pack["snr_labels"].float().view(-1)

    n_noisy = int(noisy.shape[0])
    n_per = len(SNR_LEVELS)
    if n_noisy % n_per != 0:
        logger.warning("real grounded: noisy count %d not divisible by %d SNRs", n_noisy, n_per)
    clean_rep = clean.repeat_interleave(n_per, dim=0)
    if clean_rep.shape[0] != n_noisy:
        logger.warning(
            "real grounded: clean_rep %d vs noisy %d — truncating to min",
            clean_rep.shape[0],
            n_noisy,
        )
        m = min(clean_rep.shape[0], n_noisy)
        clean_rep = clean_rep[:m]
        noisy = noisy[:m]
        snr_labels = snr_labels[:m]
        n_noisy = m

    MAX_ROWS = 20000 * n_per
    if n_noisy > MAX_ROWS:
        rng = torch.Generator().manual_seed(42)
        perm = torch.randperm(n_noisy, generator=rng)[:MAX_ROWS]
        clean_rep = clean_rep[perm]
        noisy = noisy[perm]
        snr_labels = snr_labels[perm]
        n_noisy = MAX_ROWS
        logger.info("Subsampled grounded real Q.Lin: -> %d noisy rows", n_noisy)

    all_ncc: List[np.ndarray] = []
    all_rmse: List[np.ndarray] = []
    CHUNK = 2000
    model.eval()
    with torch.no_grad():
        for start in range(0, n_noisy, CHUNK):
            chunk_n = noisy[start : start + CHUNK].to(device)
            chunk_c = clean_rep[start : start + CHUNK]
            if device.type == "cuda":
                with autocast("cuda"):
                    pred, _ = model(chunk_n)
            else:
                pred, _ = model(chunk_n)
            pred_eval = (
                hybrid_nn_savgol(pred, chunk_n, hybrid_savgol_gamma_nn)
                if hybrid_savgol_gamma_nn is not None
                else pred
            )
            all_ncc.append(_ncc_batch(pred_eval.cpu(), chunk_c.cpu()))
            all_rmse.append(_rmse_batch(pred_eval.cpu(), chunk_c.cpu()))
            del chunk_n, pred
            if device.type == "cuda":
                torch.cuda.empty_cache()

    ncc_vals = np.concatenate(all_ncc)
    rmse_vals = np.concatenate(all_rmse)
    sl = snr_labels.cpu().numpy()

    per_snr: Dict[str, Any] = {}
    for lvl in SNR_LEVELS:
        idx = np.where(np.abs(sl - float(lvl)) < 0.51)[0]
        if len(idx):
            per_snr[str(lvl)] = {
                "ncc_mean": float(np.mean(ncc_vals[idx])),
                "ncc_std": float(np.std(ncc_vals[idx])),
                "rmse_mean": float(np.mean(rmse_vals[idx])),
                "rmse_std": float(np.std(rmse_vals[idx])),
            }
        else:
            per_snr[str(lvl)] = {}

    out = {
        "ncc_mean": float(np.mean(ncc_vals)),
        "ncc_std": float(np.std(ncc_vals)),
        "rmse_mean": float(np.mean(rmse_vals)),
        "rmse_std": float(np.std(rmse_vals)),
        "n_windows_evaluated": int(len(ncc_vals)),
        "reference": "clean_ground_truth",
        "per_snr": per_snr,
    }
    logger.info(
        "Real-data NCC (vs clean ground truth): mean=%.4f std=%.4f | RMSE mean=%.6f",
        out["ncc_mean"],
        out["ncc_std"],
        out["rmse_mean"],
    )
    return out


def evaluate_classical_grounded_real(fn: Any) -> Dict[str, Any]:
    """
    Grounded real Q.Lin evaluation for classical CPU denoisers only.

    Uses ``test_real_qlin_clean.pt`` / ``test_real_qlin_noisy.pt`` / ``test_real_qlin_snr_labels.pt``,
    same alignment and caps as neural ``evaluate_real_with_noise``, applying ``fn`` per noisy window.
    """
    clean_path = PROJECT_ROOT / REAL_CLEAN_PATH
    noisy_path = PROJECT_ROOT / REAL_NOISY_PATH
    snr_path = PROJECT_ROOT / REAL_SNR_LABELS_PATH
    if not (clean_path.exists() and noisy_path.exists() and snr_path.exists()):
        logger.warning(
            "Classical grounded real skipped — missing (%s, %s, %s)",
            clean_path,
            noisy_path,
            snr_path,
        )
        return {}

    clean_pack = torch.load(clean_path, weights_only=False)
    noisy_pack = torch.load(noisy_path, weights_only=False)
    snr_pack = torch.load(snr_path, weights_only=False)
    clean = clean_pack["clean"]
    noisy = noisy_pack["noisy"].float()
    snr_labels = snr_pack["snr_labels"].float().view(-1)

    n_noisy = int(noisy.shape[0])
    n_per = len(SNR_LEVELS)
    if n_noisy % n_per != 0:
        logger.warning("classical grounded: noisy count %d not divisible by %d SNRs", n_noisy, n_per)
    clean_rep = clean.repeat_interleave(n_per, dim=0)
    if clean_rep.shape[0] != n_noisy:
        logger.warning(
            "classical grounded: clean_rep %d vs noisy %d — truncating to min",
            clean_rep.shape[0],
            n_noisy,
        )
        m = min(clean_rep.shape[0], n_noisy)
        clean_rep = clean_rep[:m]
        noisy = noisy[:m]
        snr_labels = snr_labels[:m]
        n_noisy = m

    MAX_ROWS = 20000 * n_per
    if n_noisy > MAX_ROWS:
        rng = torch.Generator().manual_seed(42)
        perm = torch.randperm(n_noisy, generator=rng)[:MAX_ROWS]
        clean_rep = clean_rep[perm]
        noisy = noisy[perm]
        snr_labels = snr_labels[perm]
        n_noisy = MAX_ROWS
        logger.info("Subsampled classical grounded real Q.Lin -> %d noisy rows", n_noisy)

    all_ncc: List[np.ndarray] = []
    all_rmse: List[np.ndarray] = []
    CHUNK = 2000
    with torch.no_grad():
        for start in range(0, n_noisy, CHUNK):
            chunk_n = noisy[start : start + CHUNK]
            chunk_c = clean_rep[start : start + CHUNK]
            pred = _apply_baseline_to_batch(chunk_n, fn)
            all_ncc.append(_ncc_batch(pred, chunk_c))
            all_rmse.append(_rmse_batch(pred, chunk_c))

    ncc_vals = np.concatenate(all_ncc)
    rmse_vals = np.concatenate(all_rmse)
    sl = snr_labels.cpu().numpy()

    per_snr: Dict[str, Any] = {}
    for lvl in SNR_LEVELS:
        idx = np.where(np.abs(sl - float(lvl)) < 0.51)[0]
        if len(idx):
            per_snr[str(lvl)] = {
                "ncc_mean": float(np.mean(ncc_vals[idx])),
                "ncc_std": float(np.std(ncc_vals[idx])),
                "rmse_mean": float(np.mean(rmse_vals[idx])),
                "rmse_std": float(np.std(rmse_vals[idx])),
            }
        else:
            per_snr[str(lvl)] = {}

    out = {
        "ncc_mean": float(np.mean(ncc_vals)),
        "ncc_std": float(np.std(ncc_vals)),
        "rmse_mean": float(np.mean(rmse_vals)),
        "rmse_std": float(np.std(rmse_vals)),
        "n_windows_evaluated": int(len(ncc_vals)),
        "reference": "clean_ground_truth",
        "per_snr": per_snr,
    }
    logger.info(
        "Classical grounded real NCC: mean=%.4f std=%.4f | RMSE mean=%.6f",
        out["ncc_mean"],
        out["ncc_std"],
        out["rmse_mean"],
    )
    return out


# ---------------------------------------------------------------------------
# Full benchmark (neural)
# ---------------------------------------------------------------------------


def run_full_benchmark(
    model_id: str,
    checkpoint_path: str,
    results_root: str = "results",
    device_str: str = "cuda",
    hybrid_savgol_gamma_nn: Optional[float] = None,
    no_attention_gates: bool = False,
) -> Dict[str, Any]:
    from mr_tae_fusion.config import Config
    from mr_tae_fusion.models.mr_tae_fusion import create_model

    results_root_p = Path(results_root)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    master_cfg = Config()
    ablation, ablation_src = ablation_for_checkpoint(model_id, checkpoint_path)
    if no_attention_gates:
        ablation = replace(ablation, use_attention_gates=False)
        ablation_src = f"{ablation_src} + CLI --no-attention-gates"

    model = create_model(master_cfg.model, ablation)
    model.load_state_dict(_load_checkpoint_state(ckpt), strict=False)
    model = model.to(device)
    model.eval()

    n_params_M = sum(p.numel() for p in model.parameters()) / 1e6
    lat_b1, lat_b32 = _measure_latency_ms(model, device)

    results: Dict[str, Any] = {
        "model_id": model_id,
        "checkpoint": str(checkpoint_path),
        "ablation_source": ablation_src,
        "hybrid_savgol_gamma_nn": hybrid_savgol_gamma_nn,
        "synthetic": {},
        "real_qlin": {},
        "real_qlin_grounded": {},
        "efficiency": {
            "param_count_M": round(n_params_M, 4),
            "inference_latency_ms_b1": round(lat_b1, 4),
            "inference_latency_ms_b32": round(lat_b32, 4),
        },
    }

    synth_path = PROJECT_ROOT / SYNTH_TEST_PATH
    if synth_path.exists():
        synth = torch.load(synth_path, weights_only=False)
        noisy = synth["noisy"]
        clean = synth["clean"]
        mask = synth.get("masks", synth.get("mask"))
        snr_labels = synth.get("snr_labels")

        ds = TensorDataset(
            noisy.unsqueeze(1),
            clean.unsqueeze(1),
            mask,
        )
        out = _infer_dataset(model, ds, device, has_mask=True)
        pred_sig = out["preds"]
        if hybrid_savgol_gamma_nn is not None:
            pred_sig = hybrid_nn_savgol(pred_sig, out["noisys"], hybrid_savgol_gamma_nn)
            out["preds"] = pred_sig

        ncc_vals = _ncc_batch(pred_sig, out["cleans"])
        rmse_vals = _rmse_batch(pred_sig, out["cleans"])
        snr_vals = _snr_improvement(pred_sig, out["cleans"], out["noisys"])
        dice_vals = _dice_batch(out["pred_segs"], out["masks"])
        fpr_vals = _false_positive_rate_batch(out["pred_segs"], out["masks"])

        per_snr: Dict[str, Any] = {}
        idx_map = _synthetic_snr_indices(len(ncc_vals), snr_labels)
        for lvl in SNR_LEVELS:
            ix = idx_map.get(lvl)
            if ix is not None and len(ix):
                per_snr[str(lvl)] = {
                    "ncc_mean": float(np.mean(ncc_vals[ix])),
                    "ncc_std": float(np.std(ncc_vals[ix])),
                }
            else:
                per_snr[str(lvl)] = {}

        results["synthetic"] = {
            "ncc_mean": float(np.mean(ncc_vals)),
            "ncc_std": float(np.std(ncc_vals)),
            "rmse_mean": float(np.mean(rmse_vals)),
            "rmse_std": float(np.std(rmse_vals)),
            "snr_improvement_mean": float(np.mean(snr_vals)),
            "snr_improvement_std": float(np.std(snr_vals)),
            "dice_mean": float(np.mean(dice_vals)),
            "dice_std": float(np.std(dice_vals)),
            "false_positive_rate": float(np.mean(fpr_vals)),
            "per_snr_ncc": per_snr,
        }
    else:
        logger.warning("Synthetic test set not found at %s", synth_path)

    real_path = PROJECT_ROOT / REAL_TEST_PATH
    if real_path.exists():
        MAX_REAL_WINDOWS = 20000
        real = torch.load(real_path, weights_only=False)
        windows = real["windows"]
        n_total = windows.shape[0]
        if n_total > MAX_REAL_WINDOWS:
            rng = torch.Generator().manual_seed(42)
            perm = torch.randperm(n_total, generator=rng)[:MAX_REAL_WINDOWS]
            windows = windows[perm]
            logger.info("Subsampled real Q.Lin: %d -> %d windows", n_total, MAX_REAL_WINDOWS)
        del real
        torch.cuda.empty_cache()

        all_ncc, all_rmse = [], []
        CHUNK = 2000
        model.eval()
        with torch.no_grad():
            for start in range(0, len(windows), CHUNK):
                chunk = windows[start:start + CHUNK].unsqueeze(1).to(device)
                if device.type == "cuda":
                    with autocast("cuda"):
                        pred, _ = model(chunk)
                else:
                    pred, _ = model(chunk)
                pred_eval = hybrid_nn_savgol(
                    pred, chunk, hybrid_savgol_gamma_nn
                ) if hybrid_savgol_gamma_nn is not None else pred
                all_ncc.append(_ncc_batch(pred_eval.cpu(), chunk.cpu()))
                all_rmse.append(_rmse_batch(pred_eval.cpu(), chunk.cpu()))
                del chunk, pred
                torch.cuda.empty_cache()

        ncc_vals = np.concatenate(all_ncc)
        rmse_vals = np.concatenate(all_rmse)
        results["real_qlin"] = {
            "ncc_mean": float(np.mean(ncc_vals)),
            "ncc_std": float(np.std(ncc_vals)),
            "rmse_mean": float(np.mean(rmse_vals)),
            "rmse_std": float(np.std(rmse_vals)),
            "n_windows_evaluated": len(ncc_vals),
            "reference": "input_window",
            "note": "No clean ground truth; NCC/RMSE vs observed window.",
        }
        del windows
        torch.cuda.empty_cache()
    else:
        logger.warning("Real Q.Lin test set not found at %s", real_path)

    grounded = evaluate_real_with_noise(model, device, hybrid_savgol_gamma_nn)
    if grounded:
        results["real_qlin_grounded"] = grounded

    out_dir = results_root_p / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Benchmark saved: %s", out_path)

    if _MLFLOW:
        try:
            flat: Dict[str, float] = {}
            for section, d in results.items():
                if isinstance(d, dict):
                    for k, v in d.items():
                        if isinstance(v, (int, float)):
                            flat[f"{section}/{k}"] = float(v)
            if flat:
                mlflow.log_metrics(flat)
            mlflow.log_artifact(str(out_path))
        except Exception:
            pass

    return results


# ---------------------------------------------------------------------------
# Traditional benchmark
# ---------------------------------------------------------------------------


def run_traditional_benchmark(
    baseline_id: str,
    results_root: str = "results",
) -> Dict[str, Any]:
    """Evaluate wavelet or Savitzky–Golay baselines (CPU)."""
    results_root_p = Path(results_root)
    if baseline_id == BASELINE_WAVELET_ID:
        fn = wavelet_denoise
    elif baseline_id == BASELINE_SAVGOL_ID:
        fn = lambda z: savgol_denoise(z, window=51, polyorder=3)  # noqa: E731
    else:
        raise ValueError(f"Unknown baseline_id={baseline_id}")

    results: Dict[str, Any] = {
        "model_id": baseline_id,
        "checkpoint": None,
        "synthetic": {},
        "real_qlin": {},
        "efficiency": {
            "param_count_M": 0.0,
            "inference_latency_ms_b1": 0.0,
            "inference_latency_ms_b32": 0.0,
        },
    }

    synth_path = PROJECT_ROOT / SYNTH_TEST_PATH
    if synth_path.exists():
        synth = torch.load(synth_path, weights_only=False)
        noisy = synth["noisy"]
        clean = synth["clean"]
        snr_labels = synth.get("snr_labels")

        preds = _apply_baseline_to_batch(noisy, fn)
        ncc_vals = _ncc_batch(preds, clean.unsqueeze(1))
        rmse_vals = _rmse_batch(preds, clean.unsqueeze(1))
        snr_vals = _snr_improvement(preds, clean.unsqueeze(1), noisy.unsqueeze(1))

        per_snr: Dict[str, Any] = {}
        idx_map = _synthetic_snr_indices(len(ncc_vals), snr_labels)
        for lvl in SNR_LEVELS:
            ix = idx_map.get(lvl)
            if ix is not None and len(ix):
                per_snr[str(lvl)] = {
                    "ncc_mean": float(np.mean(ncc_vals[ix])),
                    "ncc_std": float(np.std(ncc_vals[ix])),
                }
            else:
                per_snr[str(lvl)] = {}

        results["synthetic"] = {
            "ncc_mean": float(np.mean(ncc_vals)),
            "ncc_std": float(np.std(ncc_vals)),
            "rmse_mean": float(np.mean(rmse_vals)),
            "rmse_std": float(np.std(rmse_vals)),
            "snr_improvement_mean": float(np.mean(snr_vals)),
            "snr_improvement_std": float(np.std(snr_vals)),
            "dice_mean": None,
            "dice_std": None,
            "false_positive_rate": None,
            "per_snr_ncc": per_snr,
            "segmentation_note": "Classical denoiser — no segmentation head; Dice/FPR omitted.",
        }

        nwin = noisy.shape[0]
        x0 = noisy[0].numpy()
        for _ in range(20):
            fn(x0)
        t0 = time.perf_counter()
        reps = 3
        for _ in range(reps):
            for i in range(nwin):
                fn(noisy[i].numpy())
        elapsed = time.perf_counter() - t0
        results["efficiency"]["inference_latency_ms_b1"] = round(
            1000.0 * elapsed / max(reps * nwin, 1), 4
        )
    else:
        logger.warning("Synthetic test set not found at %s", synth_path)

    real_path = PROJECT_ROOT / REAL_TEST_PATH
    if real_path.exists():
        real = torch.load(real_path, weights_only=False)
        windows = real["windows"]
        preds = _apply_baseline_to_batch(windows, fn)
        ncc_vals = _ncc_batch(preds, windows.unsqueeze(1))
        rmse_vals = _rmse_batch(preds, windows.unsqueeze(1))
        results["real_qlin"] = {
            "ncc_mean": float(np.mean(ncc_vals)),
            "ncc_std": float(np.std(ncc_vals)),
            "rmse_mean": float(np.mean(rmse_vals)),
            "rmse_std": float(np.std(rmse_vals)),
            "reference": "input_window",
            "note": "No clean ground truth; NCC/RMSE vs observed window.",
        }
    else:
        logger.warning("Real Q.Lin test set not found at %s", real_path)

    # Grounded real (clean-noisy-SNRLab tensors) — no checkpoint needed for Baseline-* methods
    if baseline_id.startswith("Baseline-"):
        grounded = evaluate_classical_grounded_real(fn)
        if grounded:
            results["real_qlin_grounded"] = grounded

    out_dir = results_root_p / baseline_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Baseline benchmark saved: %s", out_path)
    return results


# ---------------------------------------------------------------------------
# Collect results + figures
# ---------------------------------------------------------------------------


def _collect_benchmark_rows(results_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    pattern = str(results_root / "*" / "benchmark_results.json")
    import glob

    for jf in sorted(glob.glob(pattern)):
        try:
            with open(jf, encoding="utf-8") as f:
                d = json.load(f)
            mid = Path(jf).parent.name
            rows.append({"model_id": mid, "path": jf, "data": d})
        except Exception as e:
            logger.warning("Skip %s: %s", jf, e)
    return rows


def _matplotlib_style() -> None:
    if plt is None:
        return
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )


def generate_comparison_bar(rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    if plt is None:
        logger.warning("matplotlib not available; skip comparison_bar")
        return
    _matplotlib_style()
    models: List[str] = []
    ncc_s: List[float] = []
    ncc_r: List[float] = []
    for r in rows:
        d = r["data"]
        mid = r["model_id"]
        sm = d.get("synthetic", {}).get("ncc_mean")
        rm = d.get("real_qlin", {}).get("ncc_mean")
        if sm is None and rm is None:
            continue
        models.append(mid)
        ncc_s.append(float(sm) if sm is not None else float("nan"))
        ncc_r.append(float(rm) if rm is not None else float("nan"))

    if not models:
        logger.warning("No data for comparison_bar")
        return

    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(10, len(models) * 0.45), 4.2))
    ax.bar(x - w / 2, ncc_s, w, label="NCC synthetic", color="#2c7fb8", edgecolor="black", linewidth=0.4)
    ax.bar(x + w / 2, ncc_r, w, label="NCC real", color="#f28e1c", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("NCC (Pearson)")
    ax.set_title("Model comparison: synthetic vs real Q.Lin")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def generate_ablation_heatmap(rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    if sns is None or plt is None:
        logger.warning("seaborn/matplotlib not available; skip heatmap")
        return
    _matplotlib_style()
    reg = load_model_registry()
    grid: Dict[Tuple[str, str], float] = {}
    for r in rows:
        mid = r["model_id"]
        if mid.startswith("Baseline"):
            continue
        if mid not in reg:
            continue
        ncc = r["data"].get("real_qlin", {}).get("ncc_mean")
        if ncc is None:
            continue
        row, col = registry_ablation_axes(mid)
        key = (row, col)
        grid[key] = max(grid.get(key, float("-inf")), float(ncc))

    rows_l = ["BiGRU+Swin", "BiGRU only", "Swin only", "Conv only"]
    cols_l = ["MWCNN (wavelet)", "Plain encoder"]
    mat = np.full((len(rows_l), len(cols_l)), np.nan, dtype=np.float64)
    for i, rr in enumerate(rows_l):
        for j, cc in enumerate(cols_l):
            mat[i, j] = grid.get((rr, cc), np.nan)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(
        mat,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        vmin=0,
        vmax=1,
        xticklabels=cols_l,
        yticklabels=rows_l,
        ax=ax,
        cbar_kws={"label": "NCC (real)"},
        mask=np.isnan(mat),
    )
    ax.set_title("Ablation: NCC on real Q.Lin (best per cell)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def generate_efficiency_scatter(rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    if plt is None:
        return
    _matplotlib_style()
    xs: List[float] = []
    ys: List[float] = []
    ss: List[float] = []
    labels: List[str] = []
    for r in rows:
        d = r["data"]
        lat = d.get("efficiency", {}).get("inference_latency_ms_b1")
        ncc = d.get("real_qlin", {}).get("ncc_mean")
        pm = d.get("efficiency", {}).get("param_count_M")
        if lat is None or ncc is None:
            continue
        xs.append(float(lat))
        ys.append(float(ncc))
        ss.append(max(float(pm or 0.0), 0.01))
        labels.append(r["model_id"])

    if not xs:
        logger.warning("No data for efficiency scatter")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys, s=[300 * s for s in ss], alpha=0.55, c="#984ea3", edgecolors="k", linewidths=0.5)
    for x, y, lb in zip(xs, ys, labels):
        ax.annotate(lb, (x, y), fontsize=6, ha="left", va="bottom", alpha=0.85)
    ax.set_xlabel("Inference latency (ms, batch=1, CUDA sync)")
    ax.set_ylabel("NCC (real)")
    ax.set_title("Efficiency vs quality (bubble ∝ params M)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def generate_snr_sweep(rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    if plt is None:
        return
    _matplotlib_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    x_levels = list(SNR_LEVELS)
    for r in rows:
        per = r["data"].get("synthetic", {}).get("per_snr_ncc", {})
        if not per:
            continue
        ys = []
        for lvl in SNR_LEVELS:
            cell = per.get(str(lvl), {})
            m = cell.get("ncc_mean")
            ys.append(float(m) if m is not None else float("nan"))
        if all(np.isnan(ys)):
            continue
        ax.plot(x_levels, ys, marker="o", linewidth=1.2, label=r["model_id"], alpha=0.85)

    ax.set_xlabel("Input SNR level (nominal, dB)")
    ax.set_ylabel("NCC vs clean (synthetic)")
    ax.set_title("Synthetic test: NCC vs SNR")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=6)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def generate_loss_comparison(rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    if plt is None:
        return
    _matplotlib_style()
    want = set(LOSS_VARIANT_IDS)
    pts: List[Tuple[str, float]] = []
    for r in rows:
        if r["model_id"] not in want:
            continue
        ncc = r["data"].get("real_qlin", {}).get("ncc_mean")
        if ncc is not None:
            pts.append((r["model_id"], float(ncc)))

    if not pts:
        logger.warning("No MR-TAE-FULL loss variants for loss_comparison")
        return

    pts.sort(key=lambda x: x[0])
    labs = [p[0] for p in pts]
    vals = [p[1] for p in pts]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(labs)), vals, color="#41ae76", edgecolor="black", linewidth=0.5)
    ax.set_ylabel("NCC (real)")
    ax.set_title("Loss configuration comparison (MR-TAE-FULL family)")
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs, rotation=25, ha="right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def write_results_comparison_md(rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    lines = [
        "# Results comparison (auto-generated by benchmark_runner)\n",
        "\n",
        "Real-set NCC/RMSE are computed against the **input window** (no clean ground truth).\n",
        "\n",
        "| Model | Params (M) | Latency b=1 (ms) | NCC synth | NCC real | RMSE synth | RMSE real | "
        "Dice (synth) | FPR (synth) | SNR imp. (synth) |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for r in sorted(rows, key=lambda z: z["model_id"]):
        d = r["data"]
        e = d.get("efficiency", {})
        s = d.get("synthetic", {})
        q = d.get("real_qlin", {})

        def fmt(v: Any) -> str:
            if v is None:
                return "—"
            if isinstance(v, bool):
                return str(v)
            if isinstance(v, (int, float)):
                return f"{float(v):.4f}"
            return str(v)

        lines.append(
            f"| {r['model_id']} | {fmt(e.get('param_count_M'))} | {fmt(e.get('inference_latency_ms_b1'))} | "
            f"{fmt(s.get('ncc_mean'))} | {fmt(q.get('ncc_mean'))} | "
            f"{fmt(s.get('rmse_mean'))} | {fmt(q.get('rmse_mean'))} | "
            f"{fmt(s.get('dice_mean'))} | {fmt(s.get('false_positive_rate'))} | "
            f"{fmt(s.get('snr_improvement_mean'))} |\n"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines), encoding="utf-8")
    logger.info("Wrote %s", out_path)


def generate_all_figures(results_root: str = "results") -> None:
    root = Path(results_root)
    rows = _collect_benchmark_rows(root)
    bench = root / "benchmark"
    bench.mkdir(parents=True, exist_ok=True)
    generate_comparison_bar(rows, bench / "comparison_bar.png")
    generate_ablation_heatmap(rows, bench / "ablation_heatmap.png")
    generate_efficiency_scatter(rows, bench / "efficiency_scatter.png")
    generate_snr_sweep(rows, bench / "snr_sweep.png")
    generate_loss_comparison(rows, bench / "loss_comparison.png")
    write_results_comparison_md(rows, PROJECT_ROOT / DOCS_RESULTS)


# ---------------------------------------------------------------------------
# compare_all
# ---------------------------------------------------------------------------


def compare_all(
    results_root: str = "results",
    device_str: str = "cuda",
    hybrid_savgol_gamma_nn: Optional[float] = None,
    no_attention_gates: bool = False,
) -> None:
    """
    Run every model with checkpoints under results/<id>/checkpoints/best.pt,
    run classical baselines, regenerate figures and RESULTS_COMPARISON.md.
    """
    results_path = Path(results_root)
    results_path.mkdir(parents=True, exist_ok=True)

    logger.info("Running classical baselines …")
    try:
        run_traditional_benchmark(BASELINE_WAVELET_ID, results_root=results_root)
    except Exception as e:
        logger.exception("Wavelet baseline failed: %s", e)
    try:
        run_traditional_benchmark(BASELINE_SAVGOL_ID, results_root=results_root)
    except Exception as e:
        logger.exception("Savitzky–Golay baseline failed: %s", e)

    trained: List[Tuple[str, Path]] = []
    if results_path.exists():
        for d in sorted(results_path.iterdir()):
            if not d.is_dir():
                continue
            if d.name == "benchmark":
                continue
            ck = d / "checkpoints" / "best.pt"
            if ck.exists():
                trained.append((d.name, ck))

    logger.info("Found %d checkpoint(s) under %s", len(trained), results_path)
    for model_id, ckpt in trained:
        if model_id in (BASELINE_WAVELET_ID, BASELINE_SAVGOL_ID):
            continue
        if model_id not in load_model_registry():
            logger.warning("Skipping %s — not in config/model_registry.yaml", model_id)
            continue
        try:
            logger.info("Benchmarking %s …", model_id)
            run_full_benchmark(
                model_id,
                str(ckpt),
                results_root=results_root,
                device_str=device_str,
                hybrid_savgol_gamma_nn=hybrid_savgol_gamma_nn,
                no_attention_gates=no_attention_gates,
            )
        except Exception as e:
            logger.exception("Benchmark failed for %s: %s", model_id, e)

    logger.info("Generating figures and %s …", DOCS_RESULTS)
    generate_all_figures(results_root=results_root)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="AE-PD Benchmark Runner")
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--compare-all", action="store_true")
    parser.add_argument("--results", type=str, default="results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Regenerate figures and docs from existing benchmark_results.json files",
    )
    parser.add_argument(
        "--hybrid-savgol-gamma",
        type=float,
        default=None,
        help=(
            "If set [0–1], blend neural output with Savitzky–Golay(noisy): "
            "gamma*nn + (1-gamma)*savgol. Calibrate offline with scripts/fit_hybrid_gamma.py."
        ),
    )
    parser.add_argument(
        "--no-attention-gates",
        action="store_true",
        help="Instantiate model with use_attention_gates=False (overrides registry / best_config.json).",
    )
    args = parser.parse_args()

    _setup_logging(PROJECT_ROOT / LOG_PATH)

    if args.figures_only:
        generate_all_figures(results_root=args.results)
    elif args.compare_all:
        compare_all(
            results_root=args.results,
            device_str=args.device,
            hybrid_savgol_gamma_nn=args.hybrid_savgol_gamma,
            no_attention_gates=args.no_attention_gates,
        )
    elif args.model_id and str(args.model_id).startswith("Baseline-"):
        if args.checkpoint:
            logger.warning("Ignoring --checkpoint for classical baseline %s", args.model_id)
        run_traditional_benchmark(str(args.model_id), results_root=args.results)
    elif args.model_id and args.checkpoint:
        run_full_benchmark(
            args.model_id,
            args.checkpoint,
            results_root=args.results,
            device_str=args.device,
            hybrid_savgol_gamma_nn=args.hybrid_savgol_gamma,
            no_attention_gates=args.no_attention_gates,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
