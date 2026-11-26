"""
figure_full_comparison.py
===========================
One figure: deep MR-TAE variants + classical baselines vs clean/noisy,
3 SNR columns, burst-region zoom. Uses same data/rep-window logic as
evaluate_and_visualise.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_TRY = _ROOT / "Try Model"
_NEW_FOCUS = Path(__file__).resolve().parent
for p in (_ROOT, _TRY, _NEW_FOCUS):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

try:
    import pywt
except ImportError as e:
    raise SystemExit(
        "PyWavelets is required: pip install PyWavelets\n" + str(e)
    ) from e

from scipy.signal import savgol_filter, wiener

import evaluate_and_visualise as ev

# ── FINAL RANKING order (mean NCC, synthetic eval) ───────────────────────────
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

# ── Traditional denoising baselines ───────────────────────────────────────────


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


BASELINE_SPECS = [
    ("Soft-Wavelet(db4)", "#2980b9", lambda x: np.stack([baseline_soft_wavelet(x[i, 0]) for i in range(len(x))])),
    ("SavGol+Wavelet", "#8e44ad", lambda x: np.stack([baseline_savgol_wavelet(x[i, 0]) for i in range(len(x))])),
    ("Wiener", "#e67e22", lambda x: np.stack([baseline_wiener(x[i, 0]) for i in range(len(x))])),
    ("SVD(95%)", "#27ae60", lambda x: baseline_svd(x[:, 0].astype(np.float64), 0.95).astype(np.float32)),
]

C_CLEAN = "#1a6e3d"
C_NOISY = "#b0b0b0"
C_DL = "#c0392b"


def _rep_idx_from_nccs(den: np.ndarray, clean: np.ndarray, ncc_fn) -> int:
    """den, clean: (N, 1, L) or den (N, L) with clean (N, 1, L)."""
    n = len(clean)
    nccs = []
    for i in range(n):
        d = den[i, 0] if den.ndim == 3 else den[i]
        c = clean[i, 0]
        nccs.append(ncc_fn(d, c))
    nccs_f = [v for v in nccs if not np.isnan(v)]
    if not nccs_f:
        return 0
    mean_n = float(np.mean(nccs_f))
    arr = np.array(nccs, dtype=np.float64)
    arr = np.where(np.isnan(arr), mean_n, arr)
    return int(np.argmin(np.abs(arr - mean_n)))


def _plot_cell(
    ax,
    noisy_1d: np.ndarray,
    clean_1d: np.ndarray,
    den_1d: np.ndarray,
    ncc_v: float,
    dsnr_v: float,
    den_color: str,
    fs_hz: float,
    *,
    hide_den: bool = False,
) -> None:
    s, e = ev.find_burst_region(clean_1d)
    t_us = (np.arange(s, e, dtype=np.float64) / fs_hz) * 1e6
    ax.plot(t_us, noisy_1d[s:e], color=C_NOISY, lw=0.8, alpha=0.3, label="Noisy")
    ax.plot(t_us, clean_1d[s:e], color=C_CLEAN, lw=1.15, label="Clean")
    if not hide_den:
        ax.plot(t_us, den_1d[s:e], color=den_color, lw=0.95, label="Denoised")
    ax.set_xlim(t_us[0], t_us[-1])
    ax.tick_params(labelsize=6)
    ax.text(
        0.02,
        0.95,
        f"NCC={ncc_v:.3f}  ΔSNR={dsnr_v:+.1f} dB",
        transform=ax.transAxes,
        fontsize=6,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.78, edgecolor="none"),
    )


def main() -> None:
    ev._configure_stdio_utf8()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = ev.OUT_DIR / "full_model_comparison.png"
    ev.OUT_DIR.mkdir(parents=True, exist_ok=True)

    snr_list = list(ev.EVAL_SNRS)
    n_snr = len(snr_list)
    n_models = len(MODEL_ORDER)
    n_base = len(BASELINE_SPECS)
    # rows: noisy + models + divider + baselines
    n_div = 1
    n_rows = 1 + n_models + n_div + n_base
    height_ratios = [1.0] * (1 + n_models) + [0.18] + [1.0] * n_base

    # Load each deep model once, then infer on all SNR batches
    loaded: dict[str, torch.nn.Module] = {}
    for mid in MODEL_ORDER:
        if mid not in ev.MODEL_REGISTRY:
            continue
        m = ev.load_model(mid, ev.MODEL_REGISTRY[mid], device)
        if m is not None:
            loaded[mid] = m

    col_data: dict[int, dict] = {}
    for snr_db in snr_list:
        noisy_all, clean_all = ev.load_test_windows(
            snr_db, ev.N_WINDOWS, seed=snr_db + 100, scaffold=False
        )
        col_data[snr_db] = dict(noisy=noisy_all, clean=clean_all, models={})
        for mid, m in loaded.items():
            den = ev.infer(m, noisy_all, device)
            col_data[snr_db]["models"][mid] = den.astype(np.float32)

    # Baseline aggregate NCC (full batch) for report
    baseline_ncc: dict[int, dict[str, float]] = {s: {} for s in snr_list}

    fig = plt.figure(figsize=(18, 0.55 * n_rows + 2), dpi=200)
    gs = gridspec.GridSpec(
        n_rows,
        n_snr,
        figure=fig,
        height_ratios=height_ratios,
        hspace=0.35,
        wspace=0.22,
        left=0.14,
        right=0.98,
        top=0.93,
        bottom=0.05,
    )
    fig.suptitle(
        "AE-PD denoising — burst zoom (representative window per row) · MR-TAE + classical baselines",
        fontsize=11,
        fontweight="bold",
    )

    for col_idx, snr_db in enumerate(snr_list):
        d = col_data[snr_db]
        noisy_all = d["noisy"]
        clean_all = d["clean"]
        fs_hz = float(ev.FS_HZ)

        # --- Noisy reference row ---
        ax0 = fig.add_subplot(gs[0, col_idx])
        nccs_in = [ev.ncc(noisy_all[i, 0], clean_all[i, 0]) for i in range(ev.N_WINDOWS)]
        ri = _rep_idx_from_nccs(noisy_all, clean_all, ev.ncc)
        ncc_m = float(np.mean([x for x in nccs_in if not np.isnan(x)])) if nccs_in else 0.0
        _plot_cell(
            ax0,
            noisy_all[ri, 0],
            clean_all[ri, 0],
            noisy_all[ri, 0],
            ncc_m,
            0.0,
            C_NOISY,
            fs_hz,
            hide_den=True,
        )
        if col_idx == 0:
            ax0.set_ylabel("Noisy input", fontsize=7, rotation=0, ha="right", va="center", labelpad=36)
        ax0.set_title(f"SNR_in = {snr_db:+d} dB", fontsize=9, fontweight="bold")
        if col_idx == 0:
            ax0.legend(fontsize=5, loc="upper right")

        # --- Deep models ---
        for r_off, mid in enumerate(MODEL_ORDER, start=1):
            ax = fig.add_subplot(gs[r_off, col_idx])
            den_b = d["models"].get(mid)
            rank = r_off  # 1..13
            short = ev.SHORT_NAMES.get(mid, mid)
            if col_idx == 0:
                ax.set_ylabel(f"#{rank} {short}", fontsize=6.5, rotation=0, ha="right", va="center", labelpad=52)
            if den_b is None:
                ax.text(0.5, 0.5, "no ckpt", ha="center", va="center", transform=ax.transAxes, fontsize=7)
                continue
            ri_m = _rep_idx_from_nccs(den_b, clean_all, ev.ncc)
            ncc_v = ev.ncc(den_b[ri_m, 0], clean_all[ri_m, 0])
            dsnr_v = ev.snr_improvement_db(
                noisy_all[ri_m, 0], den_b[ri_m, 0], clean_all[ri_m, 0]
            )
            if np.isinf(dsnr_v):
                dsnr_v = 0.0
            _plot_cell(
                ax,
                noisy_all[ri_m, 0],
                clean_all[ri_m, 0],
                den_b[ri_m, 0],
                float(ncc_v) if not np.isnan(ncc_v) else 0.0,
                float(dsnr_v),
                C_DL,
                fs_hz,
            )

        # --- Divider row ---
        axd = fig.add_subplot(gs[1 + n_models, col_idx])
        axd.axis("off")
        if col_idx == 0:
            axd.text(
                0.0,
                0.5,
                "── Traditional baselines ──",
                transform=axd.transAxes,
                fontsize=8,
                fontweight="bold",
                va="center",
                ha="left",
            )
        axd.axhline(0.55, color="#2c3e50", lw=2.4, xmin=0, xmax=1)

        # --- Baselines ---
        for bi, (bname, bcolor, fn) in enumerate(BASELINE_SPECS):
            r = 1 + n_models + n_div + bi
            axb = fig.add_subplot(gs[r, col_idx])
            den_b = fn(noisy_all).astype(np.float32)
            if den_b.ndim == 1:
                den_b = den_b.reshape(1, -1)
            den_3 = den_b[:, np.newaxis, :] if den_b.ndim == 2 else den_b
            ncc_list = [ev.ncc(den_b[i], clean_all[i, 0]) for i in range(len(den_b))]
            mean_ncc = float(np.nanmean(ncc_list))
            baseline_ncc[snr_db][bname] = mean_ncc
            ri_b = _rep_idx_from_nccs(den_3, clean_all, ev.ncc)
            dsnr_b = ev.snr_improvement_db(
                noisy_all[ri_b, 0], den_b[ri_b], clean_all[ri_b, 0]
            )
            if np.isinf(dsnr_b):
                dsnr_b = 0.0
            if col_idx == 0:
                axb.set_ylabel(bname, fontsize=6.5, rotation=0, ha="right", va="center", labelpad=52)
            _plot_cell(
                axb,
                noisy_all[ri_b, 0],
                clean_all[ri_b, 0],
                den_b[ri_b],
                float(ev.ncc(den_b[ri_b], clean_all[ri_b, 0])),
                float(dsnr_b),
                bcolor,
                fs_hz,
            )

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    sz = out_path.stat().st_size
    print(f"\nSaved: {out_path}")
    print(f"Size:  {sz / 1024:.1f} KiB ({sz} bytes)\n")
    print("Baseline mean NCC (over all windows in batch):")
    for snr_db in snr_list:
        print(f"  SNR {snr_db:+d} dB:")
        for bname, v in baseline_ncc[snr_db].items():
            print(f"    {bname:22s}  NCC_mean = {v:.4f}")
    # Best baseline at -5 dB
    if -5 in baseline_ncc and baseline_ncc[-5]:
        best = max(baseline_ncc[-5].items(), key=lambda kv: kv[1])
        print(f"\nBest traditional baseline at -5 dB (by mean NCC): {best[0]}  ({best[1]:.4f})")
    # Gap vs FULL-LossB at -5
    if -5 in col_data and "MR-TAE-FULL-LossB" in col_data[-5]["models"]:
        den_lb = col_data[-5]["models"]["MR-TAE-FULL-LossB"]
        if den_lb is not None:
            nccs_lb = [ev.ncc(den_lb[i, 0], col_data[-5]["clean"][i, 0]) for i in range(ev.N_WINDOWS)]
            ncc_lb_m = float(np.nanmean(nccs_lb))
            best_ncc = max(baseline_ncc[-5].values()) if baseline_ncc[-5] else 0.0
            print(f"MR-TAE-FULL-LossB mean NCC @ -5 dB: {ncc_lb_m:.4f}")
            print(f"Gap (#1 DL minus best baseline NCC): {ncc_lb_m - best_ncc:+.4f}")


if __name__ == "__main__":
    main()
