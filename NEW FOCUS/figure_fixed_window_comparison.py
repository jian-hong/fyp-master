"""
figure_fixed_window_comparison.py
==================================
One PNG per input SNR: every row uses the SAME (noisy, clean) window for fair
visual comparison (fixed window per SNR, not per-model representative indices).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

warnings.filterwarnings("ignore")

# ── Project paths (from evaluate_and_visualise.py) ────────────────────────────
FYP_ROOT = Path(r"D:\New folder (2)\Data\FYP_MASTER")
MODEL_PKG = FYP_ROOT / "Try Model"
OUT_DIR = FYP_ROOT / "NEW FOCUS" / "results" / "method_comparison_figures"

for p in [str(MODEL_PKG), str(FYP_ROOT / "NEW FOCUS"), str(FYP_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Constants ─────────────────────────────────────────────────────────────────
FS_HZ = 1_000_000  # 1 MHz sampling rate

# ── Ablation registry (from evaluate_and_visualise.py) ──────────────────────
MODEL_REGISTRY: dict[str, dict] = {
    "MR-TAE-FULL": dict(W=True, B=True, S=True, A=True, M=True, loss="A"),
    "MR-TAE-noBiGRU": dict(W=True, B=False, S=True, A=True, M=True, loss="A"),
    "MR-TAE-noSwin": dict(W=True, B=True, S=False, A=True, M=True, loss="A"),
    "MR-TAE-noAttn": dict(W=True, B=True, S=True, A=False, M=True, loss="A"),
    "MR-TAE-noMTL": dict(W=True, B=True, S=True, A=True, M=False, loss="C"),
    "MR-TAE-noWavelet": dict(W=False, B=True, S=True, A=True, M=True, loss="A"),
    "MR-TAE-noSwin-noBiGRU": dict(W=True, B=False, S=False, A=True, M=True, loss="A"),
    "MR-TAE-noSwin-noAttn": dict(W=True, B=True, S=False, A=False, M=True, loss="A"),
    "MR-TAE-noBiGRU-noAttn": dict(W=True, B=False, S=True, A=False, M=True, loss="A"),
    "MR-TAE-FULL-LossB": dict(W=True, B=True, S=True, A=True, M=True, loss="B"),
    "MR-TAE-FULL-LossC": dict(W=True, B=True, S=True, A=True, M=True, loss="C"),
    "MWCNN-BiGRU": dict(W=True, B=True, S=False, A=False, M=False, loss="C"),
    "UNet-BiGRU-Swin": dict(W=False, B=True, S=True, A=False, M=False, loss="C"),
}

SHORT_NAMES = {
    "MR-TAE-FULL": "FULL",
    "MR-TAE-noBiGRU": "noBiGRU",
    "MR-TAE-noSwin": "noSwin",
    "MR-TAE-noAttn": "noAttn",
    "MR-TAE-noMTL": "noMTL",
    "MR-TAE-noWavelet": "noWavelet",
    "MR-TAE-noSwin-noBiGRU": "noSwin+noBiGRU",
    "MR-TAE-noSwin-noAttn": "noSwin+noAttn",
    "MR-TAE-noBiGRU-noAttn": "noBiGRU+noAttn",
    "MR-TAE-FULL-LossB": "FULL-LossB",
    "MR-TAE-FULL-LossC": "FULL-LossC",
    "MWCNN-BiGRU": "MWCNN-BiGRU",
    "UNet-BiGRU-Swin": "UNet-BiGRU-Swin",
}


def ncc(pred: np.ndarray, clean: np.ndarray) -> float:
    """Normalised cross-correlation, mean-centred, element-wise."""
    p = pred.ravel() - pred.mean()
    c = clean.ravel() - clean.mean()
    denom = np.linalg.norm(p) * np.linalg.norm(c)
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(p, c) / denom)


def snr_improvement_db(noisy: np.ndarray, denoised: np.ndarray, clean: np.ndarray) -> float:
    """Output SNR minus input SNR, both in dB."""

    def _snr(sig, ref):
        noise = sig - ref
        p_s = float(np.mean(ref**2))
        p_n = float(np.mean(noise**2))
        return 10 * np.log10(p_s / p_n) if p_n > 0 else float("inf")

    return _snr(denoised, clean) - _snr(noisy, clean)


def find_burst_region(clean: np.ndarray, window: int = 300) -> tuple[int, int]:
    """Locate the most energetic contiguous region in the clean signal."""
    energy = np.convolve(clean**2, np.ones(window) / window, mode="valid")
    peak_idx = int(np.argmax(energy))
    start = max(0, peak_idx)
    end = min(len(clean), start + window)
    return start, end


def find_checkpoint(model_id: str) -> Path | None:
    """Resolve checkpoint; prefers best_final.pt then best_demo.pt / best.pt."""
    ckpt_dir = FYP_ROOT / "results" / model_id / "checkpoints"
    priority_names = ("best_final.pt", "best_demo.pt", "best.pt")
    if ckpt_dir.is_dir():
        for name in priority_names:
            p = ckpt_dir / name
            if p.exists():
                return p
        pts = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in pts:
            if p.name not in priority_names:
                return p
    candidates = [
        FYP_ROOT / "results" / model_id / "checkpoints" / "best.pt",
        FYP_ROOT / "results" / model_id / "best.pt",
        FYP_ROOT / "NEW FOCUS" / "results" / model_id / "checkpoints" / "best.pt",
    ]
    for c in candidates:
        if Path(c).exists():
            return Path(c)
    return None


def build_model(model_id: str, flags: dict) -> torch.nn.Module | None:
    try:
        from mr_tae_fusion.models.ablation_config import AblationConfig
        from mr_tae_fusion.models.mr_tae_fusion import create_model
        from mr_tae_fusion.config import Config

        ablation = AblationConfig(
            use_wavelet_pooling=flags["W"],
            use_bigru=flags["B"],
            use_swin=flags["S"],
            use_attention_gates=flags["A"],
            use_mtl=flags["M"],
            loss_config=flags["loss"],
            model_id=model_id,
        )
        return create_model(Config().model, ablation)
    except Exception as exc:
        print(f"  [build-err] {model_id}: {exc}")
        return None


def load_model(model_id: str, flags: dict, device: torch.device) -> torch.nn.Module | None:
    ckpt = find_checkpoint(model_id)
    if ckpt is None:
        print(f"  [skip ] {model_id}: no checkpoint")
        return None
    model = build_model(model_id, flags)
    if model is None:
        return None
    try:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            sd = state.get("model_state_dict", state.get("model_state", state.get("state_dict", state)))
        else:
            sd = state
        model.load_state_dict(sd, strict=False)
        n_p = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  [ok   ] {model_id:35s} {n_p:.2f}M  ({ckpt.name})")
        return model.eval().to(device)
    except Exception as exc:
        print(f"  [load-err] {model_id}: {exc}")
        return None


def infer(model: torch.nn.Module, noisy: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(noisy).float().to(device)
    with torch.inference_mode():
        out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    elif isinstance(out, dict):
        out = out.get("denoised") or next(iter(out.values()))
    return out.detach().cpu().numpy()


# ── PyWavelets / scipy baselines (from figure_full_comparison.py) ───────────
try:
    import pywt
except ImportError as e:
    raise SystemExit("PyWavelets is required: pip install PyWavelets\n" + str(e)) from e

from scipy.signal import savgol_filter, wiener


def baseline_soft_wavelet(noisy: np.ndarray, wavelet: str = "db4", level: int = 4) -> np.ndarray:
    """Soft-threshold wavelet denoising (PyWavelets). MATLAB wdenoise-like."""
    max_lv = pywt.dwt_max_level(len(noisy), pywt.Wavelet(wavelet))
    level = max(1, min(level, max_lv))
    coeffs = pywt.wavedec(noisy, wavelet, level=level, mode="symmetric")
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    thr = sigma * np.sqrt(2 * np.log(max(len(noisy), 2)))
    new_c = [coeffs[0]] + [pywt.threshold(c, thr, mode="soft") for c in coeffs[1:]]
    out = pywt.waverec(new_c, wavelet, mode="symmetric")
    return out[: len(noisy)].astype(np.float32)


def baseline_savgol_wavelet(noisy: np.ndarray, wavelet: str = "db2", level: int = 3) -> np.ndarray:
    """Savitzky-Golay smoothing then wavelet threshold on residual."""
    n = len(noisy)
    wl = max(7, (n // 100) | 1)
    if wl % 2 == 0:
        wl += 1
    wl = min(wl, n if n % 2 == 1 else n - 1)
    wl = max(wl, 5)
    smooth = savgol_filter(noisy.astype(np.float64), window_length=wl, polyorder=3, mode="interp")
    residual = noisy.astype(np.float64) - smooth
    max_lv = pywt.dwt_max_level(len(residual), pywt.Wavelet(wavelet))
    level = max(1, min(level, max_lv))
    coeffs = pywt.wavedec(residual, wavelet, level=level, mode="symmetric")
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    thr = sigma * np.sqrt(2 * np.log(max(len(residual), 2)))
    new_c = [coeffs[0]] + [pywt.threshold(c, thr, mode="soft") for c in coeffs[1:]]
    den_res = pywt.waverec(new_c, wavelet, mode="symmetric")[: len(noisy)]
    return (smooth + den_res).astype(np.float32)


def baseline_wiener(noisy: np.ndarray, mysize: int = 21) -> np.ndarray:
    return np.asarray(wiener(noisy.astype(np.float64), mysize=mysize), dtype=np.float32)


def baseline_svd(noisy_batch: np.ndarray, energy_threshold: float = 0.95) -> np.ndarray:
    """POD / SVD on batch (N, L). Each row of output uses the shared subspace."""
    U, S, Vt = np.linalg.svd(noisy_batch, full_matrices=False)
    cumulative = np.cumsum(S**2) / (S**2).sum()
    k = int(np.searchsorted(cumulative, energy_threshold)) + 1
    k = max(1, min(k, len(S)))
    return (U[:, :k] * S[:k]) @ Vt[:k, :]


def baseline_svd_single(noisy_1d: np.ndarray) -> np.ndarray:
    batch = noisy_1d[np.newaxis, :]  # (1, L)
    return baseline_svd(batch, 0.95)[0]  # (L,)


MODEL_ORDER = [
    "MR-TAE-FULL-LossB",
    "MR-TAE-FULL",
    "UNet-BiGRU-Swin",
    "MR-TAE-noBiGRU-noAttn",
    "MR-TAE-noWavelet",
    "MR-TAE-noBiGRU",
    "MR-TAE-noSwin-noAttn",
    "MR-TAE-noSwin",
    "MR-TAE-noAttn",
    "MWCNN-BiGRU",
    "MR-TAE-FULL-LossC",
    "MR-TAE-noSwin-noBiGRU",
    "MR-TAE-noMTL",
]

BASELINE_SPECS = [
    ("Soft-Wavelet(db4)", baseline_soft_wavelet, "#2980b9"),
    ("SavGol+Wavelet", baseline_savgol_wavelet, "#8e44ad"),
    ("Wiener", baseline_wiener, "#e67e22"),
    ("SVD(95%)", baseline_svd_single, "#27ae60"),
]

SNR_LEVELS = [10, 5, -5, -15]

C_NOISY = "#b0b0b0"
C_CLEAN = "#1a6e3d"
C_DEEP = "#c0392b"

TEST_PT = FYP_ROOT / "data" / "test_synthetic_ae.pt"


def _configure_stdio_utf8() -> None:
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def awgn_at_snr(clean_1d: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add AWGN so that SNR = 10*log10(P_signal / P_noise) == snr_db."""
    c = clean_1d.astype(np.float64)
    p_s = float(np.mean(c**2))
    if p_s < 1e-20:
        p_s = 1e-20
    sigma = np.sqrt(p_s / (10 ** (snr_db / 10.0)))
    noise = rng.standard_normal(c.shape).astype(np.float64) * sigma
    return (c + noise).astype(np.float32)


def load_test_bundle() -> dict:
    if not TEST_PT.exists():
        raise FileNotFoundError(f"Missing test file: {TEST_PT}")
    ds = torch.load(TEST_PT, map_location="cpu", weights_only=False)
    if not isinstance(ds, dict):
        raise TypeError("test_synthetic_ae.pt must be a dict")
    return ds


def count_snr_band(snr_labels: torch.Tensor, snr_db: float) -> int:
    m = (snr_labels >= snr_db - 2.5) & (snr_labels <= snr_db + 2.5)
    return int(m.sum().item())


def select_fixed_window(
    ds: dict,
    snr_db: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    One (noisy_1d, clean_1d) per SNR: first window whose label lies in
    [snr_db-2.5, snr_db+2.5], sorted by index; or re-corruption fallback.
    """
    noisy_all = ds["noisy"].float()
    clean_all = ds["clean"].float()
    snr_key = next((k for k in ("snr_db", "snr_labels", "snr") if k in ds), None)
    meta: dict = {
        "labelled_in_band": 0,
        "rerecorrupted": False,
        "window_index": 0,
        "snr_target": snr_db,
    }

    if snr_key is None:
        meta["labelled_in_band"] = 0
        clean_1d = clean_all[0].numpy().ravel()
        noisy_1d = awgn_at_snr(clean_1d, snr_db, rng)
        meta["rerecorrupted"] = True
        return noisy_1d, clean_1d, meta

    snr_t = ds[snr_key].float()
    band = (snr_t >= snr_db - 2.5) & (snr_t <= snr_db + 2.5)
    idx = band.nonzero(as_tuple=True)[0]
    meta["labelled_in_band"] = int(idx.numel())

    if snr_db == 10 and idx.numel() == 0:
        print("[+10dB] No labelled windows found — re-corrupted.")
        clean_1d = clean_all[0].numpy().ravel()
        noisy_1d = awgn_at_snr(clean_1d, 10.0, rng)
        meta["rerecorrupted"] = True
        meta["window_index"] = 0
        return noisy_1d, clean_1d, meta

    if idx.numel() > 0:
        chosen = int(idx.sort()[0][0].item())
        noisy_1d = noisy_all[chosen].numpy().ravel()
        clean_1d = clean_all[chosen].numpy().ravel()
        meta["window_index"] = chosen
        return noisy_1d, clean_1d, meta

    # No windows in band (non-+10 case): re-corrupt first clean window
    clean_1d = clean_all[0].numpy().ravel()
    noisy_1d = awgn_at_snr(clean_1d, snr_db, rng)
    meta["rerecorrupted"] = True
    meta["window_index"] = 0
    return noisy_1d, clean_1d, meta


def _plot_full(
    ax,
    noisy_1d: np.ndarray,
    clean_1d: np.ndarray,
    den_1d: np.ndarray | None,
    den_color: str,
    ncc_val: float,
    *,
    hide_den: bool = False,
    den_lw: float = 0.9,
) -> None:
    L = len(clean_1d)
    x = (np.arange(L, dtype=np.float64) / float(FS_HZ)) * 1e6  # µs
    ax.plot(x, noisy_1d, color=C_NOISY, lw=0.7, alpha=0.4, label="Noisy")
    ax.plot(x, clean_1d, color=C_CLEAN, lw=1.2, label="Clean")
    if not hide_den and den_1d is not None:
        ax.plot(x, den_1d, color=den_color, lw=den_lw, label="Denoised")
    ax.set_xlim(x[0], x[-1])
    ax.tick_params(labelsize=7)
    ax.text(
        0.02,
        0.98,
        f"NCC={ncc_val:.3f}",
        transform=ax.transAxes,
        fontsize=7,
        va="top",
        ha="left",
    )


def _plot_burst(
    ax,
    noisy_1d: np.ndarray,
    clean_1d: np.ndarray,
    den_1d: np.ndarray | None,
    den_color: str,
    dsnr_val: float,
    s: int,
    e: int,
    *,
    hide_den: bool = False,
    den_lw: float = 0.9,
) -> None:
    t_us = (np.arange(s, e, dtype=np.float64) / float(FS_HZ)) * 1e6
    ax.plot(t_us, noisy_1d[s:e], color=C_NOISY, lw=0.7, alpha=0.4)
    ax.plot(t_us, clean_1d[s:e], color=C_CLEAN, lw=1.2)
    if not hide_den and den_1d is not None:
        ax.plot(t_us, den_1d[s:e], color=den_color, lw=den_lw)
    ax.set_xlim(t_us[0], t_us[-1])
    ax.tick_params(labelsize=7)
    dsnr_disp = 0.0 if np.isinf(dsnr_val) else float(dsnr_val)
    ax.text(
        0.02,
        0.98,
        f"ΔSNR={dsnr_disp:+.1f} dB",
        transform=ax.transAxes,
        fontsize=7,
        va="top",
        ha="left",
    )


def print_summary_table(title: str, rows: list[tuple[str, float, float]]) -> None:
    print(f"\n{title}")
    print(f"{'Method':<28} | {'NCC':>8} | {'ΔSNR(dB)':>10}")
    print("-" * 52)
    for name, nv, dv in rows:
        dvv = 0.0 if np.isinf(dv) else float(dv)
        print(f"{name:<28} | {nv:8.3f} | {dvv:+10.1f}")


def main() -> None:
    _configure_stdio_utf8()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_test_bundle()
    rng = np.random.default_rng(42)

    print("Loading MR-TAE models…")
    loaded: dict[str, torch.nn.Module] = {}
    for mid in MODEL_ORDER:
        if mid not in MODEL_REGISTRY:
            continue
        m = load_model(mid, MODEL_REGISTRY[mid], device)
        if m is not None:
            loaded[mid] = m

    n_models = len(MODEL_ORDER)
    n_base = len(BASELINE_SPECS)
    n_div = 1
    n_rows = 1 + n_models + n_div + n_base
    height_ratios = [1.0] * (1 + n_models) + [0.22] + [1.0] * n_base

    burst_by_snr: dict[int, tuple[int, int]] = {}
    plus10_band_count = count_snr_band(ds["snr_labels"], 10)
    print(f"\n[+10 dB] Labelled windows in test set (±2.5 dB band): {plus10_band_count}")

    worse_than_noisy: list[tuple[str, int, float]] = []

    for snr in SNR_LEVELS:
        noisy_1d, clean_1d, win_meta = select_fixed_window(ds, snr, rng)
        noisy_batch = noisy_1d.reshape(1, 1, -1).astype(np.float32)

        den_by_model: dict[str, np.ndarray] = {}
        for mid, model in loaded.items():
            den_by_model[mid] = infer(model, noisy_batch, device).squeeze()

        den_by_base: dict[str, np.ndarray] = {}
        for bname, fn, _c in BASELINE_SPECS:
            den_by_base[bname] = fn(noisy_1d)

        burst_s, burst_e = find_burst_region(clean_1d, window=400)
        burst_by_snr[snr] = (burst_s, burst_e)

        fig = plt.figure(figsize=(16, n_rows * 0.9), dpi=200)
        gs = gridspec.GridSpec(
            n_rows,
            2,
            figure=fig,
            height_ratios=height_ratios,
            hspace=0.28,
            wspace=0.18,
            left=0.14,
            right=0.98,
            top=0.94,
            bottom=0.04,
        )
        fig.suptitle(
            f"AE-PD Denoising — Same window, all methods · SNR_in = {snr:+d} dB",
            fontsize=11,
            fontweight="bold",
        )

        summary_rows: list[tuple[str, float, float]] = []

        def row_label(ax_left, text: str) -> None:
            ax_left.set_ylabel(text, fontsize=8, rotation=0, ha="right", va="center", labelpad=72)

        # Row 0: noisy only
        ax0a = fig.add_subplot(gs[0, 0])
        ax0b = fig.add_subplot(gs[0, 1])
        n0 = ncc(noisy_1d, clean_1d)
        row_label(ax0a, "Noisy input")
        _plot_full(ax0a, noisy_1d, clean_1d, noisy_1d, C_DEEP, n0, hide_den=True)
        _plot_burst(ax0b, noisy_1d, clean_1d, noisy_1d, C_DEEP, 0.0, burst_s, burst_e, hide_den=True)
        summary_rows.append(("Noisy input", n0, 0.0))

        # Rows 1–13: deep models
        for r_off, mid in enumerate(MODEL_ORDER, start=1):
            axa = fig.add_subplot(gs[r_off, 0])
            axb = fig.add_subplot(gs[r_off, 1])
            rank = r_off
            short = SHORT_NAMES.get(mid, mid)
            row_label(axa, f"#{rank} {short}")
            den_1d = den_by_model.get(mid)
            if den_1d is None:
                axa.text(0.5, 0.5, "no ckpt", ha="center", va="center", transform=axa.transAxes, fontsize=8)
                axb.text(0.5, 0.5, "no ckpt", ha="center", va="center", transform=axb.transAxes, fontsize=8)
                summary_rows.append((f"#{rank} {short}", float("nan"), float("nan")))
                continue
            den_1d = np.asarray(den_1d).ravel()
            nv = ncc(den_1d, clean_1d)
            dv = snr_improvement_db(noisy_1d, den_1d, clean_1d)
            if not np.isnan(dv) and dv < -1e-3:
                worse_than_noisy.append((f"#{rank} {short} ({mid})", snr, float(dv)))
            _plot_full(axa, noisy_1d, clean_1d, den_1d, C_DEEP, nv if not np.isnan(nv) else 0.0)
            _plot_burst(axb, noisy_1d, clean_1d, den_1d, C_DEEP, dv, burst_s, burst_e)
            summary_rows.append((f"#{rank} {short}", nv if not np.isnan(nv) else 0.0, dv))

        # Divider
        axd = fig.add_subplot(gs[1 + n_models, :])
        axd.axis("off")
        axd.text(0.0, 0.55, "── Traditional baselines ──", transform=axd.transAxes, fontsize=9, fontweight="bold", va="center", ha="left")
        axd.axhline(0.45, color="#2c3e50", lw=2.8, xmin=0, xmax=1)

        # Baselines
        for bi, (bname, fn, bcol) in enumerate(BASELINE_SPECS):
            r = 1 + n_models + n_div + bi
            axa = fig.add_subplot(gs[r, 0])
            axb = fig.add_subplot(gs[r, 1])
            row_label(axa, bname)
            den_1d = np.asarray(den_by_base[bname]).ravel()
            nv = ncc(den_1d, clean_1d)
            dv = snr_improvement_db(noisy_1d, den_1d, clean_1d)
            if not np.isnan(dv) and dv < -1e-3:
                worse_than_noisy.append((bname, snr, float(dv)))
            col_a = C_DEEP
            _plot_full(axa, noisy_1d, clean_1d, den_1d, col_a, nv if not np.isnan(nv) else 0.0)
            _plot_burst(axb, noisy_1d, clean_1d, den_1d, bcol, dv, burst_s, burst_e, den_lw=0.9)
            summary_rows.append((bname, nv if not np.isnan(nv) else 0.0, dv))

        out_path = OUT_DIR / f"full_comparison_snr{snr:+d}dB.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print_summary_table(f"SNR_in = {snr:+d} dB  (window idx {win_meta['window_index']}, burst {burst_s}–{burst_e})", summary_rows)
        print(f"Saved: {out_path} ({out_path.stat().st_size} bytes)")

    print("\nBurst sample ranges (same for all rows in each file):")
    for snr in SNR_LEVELS:
        s, e = burst_by_snr[snr]
        print(f"  SNR {snr:+d} dB: samples {s}–{e} (length {e - s})")

    if worse_than_noisy:
        print("\nMethods with ΔSNR < 0 (denoised worse than noisy vs clean):")
        for name, snr, dv in worse_than_noisy:
            print(f"  {name} @ SNR_in={snr:+d} dB: ΔSNR={dv:.2f} dB")
    else:
        print("\nNo method had ΔSNR < 0 on the fixed windows run.")


if __name__ == "__main__":
    main()
