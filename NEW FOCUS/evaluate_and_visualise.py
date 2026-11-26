"""
evaluate_and_visualise.py
==========================
Thesis-grade evaluation + visualisation for all 13 MR-TAE ablation variants.

WHY THE OLD FIGURES LOOKED WRONG
---------------------------------
1. Single-window NCC is meaningless — one unlucky window can flip the ranking.
   This script averages over N_WINDOWS=25 windows per SNR level.

2. Plotting the full 2048-sample window hides PD pulses.
   This script auto-detects the peak-energy region and zooms in.

3. NCC of a barely-attenuated noisy signal can be ~0.95 even with zero denoising
   benefit.  We add SNR-improvement (dB), AE-band energy ratio, and peak-recovery
   ratio as additional figures of merit.

4. The residual plot at full scale is unreadable.
   This script normalises residuals to the clean signal RMS.

OUTPUT FILES
-----------
results/method_comparison_figures/
    overview_snr+5dB.png          -- multi-window mean ± std signal overlay + zoom
    overview_snr-5dB.png
    overview_snr-15dB.png
    pulse_zoom_snr+5dB.png        -- zoomed into peak-energy region, best 5 models
    pulse_zoom_snr-5dB.png
    pulse_zoom_snr-15dB.png
    ae_band_psd.png               -- AE-band (20-500 kHz) PSD for each model
    ranked_heatmap.png            -- 3-row heat-map: NCC, SNR improvement (dB), AE-band ratio
    ncc_summary_bar.png           -- mean NCC bar chart with error bars
    method_comparison_summary.json

RUN
---
    # Windows: activate project venv (note: .venv, not venv)
    #   D:/New folder (2)/Data/FYP_MASTER/.venv/Scripts/activate
    # Or run directly:
    #   D:/New folder (2)/Data/FYP_MASTER/.venv/Scripts/python.exe evaluate_and_visualise.py
    python evaluate_and_visualise.py

QUICK TEST (no GPU, no checkpoints needed — uses Gaussian scaffold)
    python evaluate_and_visualise.py --scaffold
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import torch

warnings.filterwarnings("ignore")


def _configure_stdio_utf8() -> None:
    """Avoid UnicodeEncodeError on Windows consoles (cp1252)."""
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

# ── Project paths ─────────────────────────────────────────────────────────────
FYP_ROOT  = Path(r"D:\New folder (2)\Data\FYP_MASTER")
MODEL_PKG = FYP_ROOT / "Try Model"
OUT_DIR   = FYP_ROOT / "NEW FOCUS" / "results" / "method_comparison_figures"

for p in [str(MODEL_PKG), str(FYP_ROOT / "NEW FOCUS"), str(FYP_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Constants ─────────────────────────────────────────────────────────────────
FS_HZ       = 1_000_000          # 1 MHz sampling rate
AE_LO_HZ   = 20_000             # AE band low edge
AE_HI_HZ   = 500_000            # AE band high edge
EVAL_SNRS   = [-15, -5, 5]      # dB — must match training curriculum
N_WINDOWS   = 25                # windows per SNR level for statistics
ZOOM_LEN    = 300               # samples for zoomed pulse panel
TOP_N       = 5                 # models shown in pulse-zoom panel

# ── Palette — print-safe, colour-blind-friendly ───────────────────────────────
C = dict(
    clean    = "#1a6e3d",   # dark green
    noisy    = "#b0b0b0",   # grey
    denoised = "#c0392b",   # strong red (good contrast vs green)
    residual = "#7d3c98",   # purple
    band     = "#2980b9",   # blue (AE band shading)
)

# Thesis colour palette for ranked heatmap
CMAP_HEATMAP = LinearSegmentedColormap.from_list(
    "pd_heat", ["#2c3e50", "#1a6e3d", "#f1c40f", "#e74c3c"], N=256
)

RC = {
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "font.family":      "sans-serif",
    "font.size":        9,
    "axes.labelsize":   9,
    "axes.titlesize":   10,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "legend.frameon":   False,
    "lines.linewidth":  0.9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
}

# ── Ablation registry ─────────────────────────────────────────────────────────
MODEL_REGISTRY: dict[str, dict] = {
    "MR-TAE-FULL":            dict(W=True,  B=True,  S=True,  A=True,  M=True,  loss="A"),
    "MR-TAE-noBiGRU":         dict(W=True,  B=False, S=True,  A=True,  M=True,  loss="A"),
    "MR-TAE-noSwin":          dict(W=True,  B=True,  S=False, A=True,  M=True,  loss="A"),
    "MR-TAE-noAttn":          dict(W=True,  B=True,  S=True,  A=False, M=True,  loss="A"),
    "MR-TAE-noMTL":           dict(W=True,  B=True,  S=True,  A=True,  M=False, loss="C"),
    "MR-TAE-noWavelet":       dict(W=False, B=True,  S=True,  A=True,  M=True,  loss="A"),
    "MR-TAE-noSwin-noBiGRU":  dict(W=True,  B=False, S=False, A=True,  M=True,  loss="A"),
    "MR-TAE-noSwin-noAttn":   dict(W=True,  B=True,  S=False, A=False, M=True,  loss="A"),
    "MR-TAE-noBiGRU-noAttn":  dict(W=True,  B=False, S=True,  A=False, M=True,  loss="A"),
    "MR-TAE-FULL-LossB":      dict(W=True,  B=True,  S=True,  A=True,  M=True,  loss="B"),
    "MR-TAE-FULL-LossC":      dict(W=True,  B=True,  S=True,  A=True,  M=True,  loss="C"),
    "MWCNN-BiGRU":            dict(W=True,  B=True,  S=False, A=False, M=False, loss="C"),
    "UNet-BiGRU-Swin":        dict(W=False, B=True,  S=True,  A=False, M=False, loss="C"),
}

SHORT_NAMES = {
    "MR-TAE-FULL":           "FULL",
    "MR-TAE-noBiGRU":        "noBiGRU",
    "MR-TAE-noSwin":         "noSwin",
    "MR-TAE-noAttn":         "noAttn",
    "MR-TAE-noMTL":          "noMTL",
    "MR-TAE-noWavelet":      "noWavelet",
    "MR-TAE-noSwin-noBiGRU": "noSwin+noBiGRU",
    "MR-TAE-noSwin-noAttn":  "noSwin+noAttn",
    "MR-TAE-noBiGRU-noAttn": "noBiGRU+noAttn",
    "MR-TAE-FULL-LossB":     "FULL-LossB",
    "MR-TAE-FULL-LossC":     "FULL-LossC",
    "MWCNN-BiGRU":           "MWCNN-BiGRU",
    "UNet-BiGRU-Swin":       "UNet-BiGRU-Swin",
}

# ── Metrics ────────────────────────────────────────────────────────────────────

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
        p_s = float(np.mean(ref ** 2))
        p_n = float(np.mean(noise ** 2))
        return 10 * np.log10(p_s / p_n) if p_n > 0 else float("inf")
    return _snr(denoised, clean) - _snr(noisy, clean)

def ae_band_energy_ratio(signal: np.ndarray, clean: np.ndarray, fs: int = FS_HZ) -> float:
    """
    Ratio of energy in the AE band (20–500 kHz) in denoised vs clean.
    1.0 = perfect preservation; <1 = AE energy lost; >1 = artefact amplification.
    """
    from scipy.signal import welch
    nperseg = min(512, len(signal))
    f, p_den = welch(signal, fs=fs, nperseg=nperseg)
    _, p_cln = welch(clean,  fs=fs, nperseg=nperseg)
    mask = (f >= AE_LO_HZ) & (f <= AE_HI_HZ)
    e_den = float(p_den[mask].mean())
    e_cln = float(p_cln[mask].mean())
    return e_den / (e_cln + 1e-12)

def peak_recovery_ratio(denoised: np.ndarray, clean: np.ndarray) -> float:
    """
    |max(|denoised|) - max(|clean|)| / max(|clean|)
    0 = perfect peak amplitude recovery.
    """
    pk_den = float(np.max(np.abs(denoised)))
    pk_cln = float(np.max(np.abs(clean)))
    if pk_cln < 1e-12:
        return float("nan")
    return abs(pk_den - pk_cln) / pk_cln


def pd_feature_retention(denoised: np.ndarray, clean: np.ndarray,
                          fs: int = FS_HZ) -> dict:
    """
    Compute how well denoising preserves the 9 PD classification features
    used in the senior's MATLAB classification pipeline (RMS, VAR, STD,
    Waveform Factor, Kurtosis, Skewness, Max-PSP, MDF, MNF).

    Returns a dict with per-feature ratios (denoised/clean) and a scalar
    pd_feature_error = mean |ratio - 1| over the 4 most PD-discriminating
    features (kurtosis, MDF, MNF, RMS).  Closer to 0 = better.

    Kurtosis is the single most important: PD pulses are spiky (high
    kurtosis). An over-smoothed denoised signal loses kurtosis → ratio < 1.
    """
    from scipy.stats import kurtosis as _kurt, skew as _skew

    def _features(sig):
        sig = sig.ravel().astype(np.float64)
        rms  = float(np.sqrt(np.mean(sig**2)))
        var  = float(np.var(sig))
        std  = float(np.std(sig))
        mean_abs = float(np.abs(np.mean(sig)))
        wf   = rms / mean_abs if mean_abs > 1e-12 else 0.0
        kurt = float(_kurt(sig))
        sk   = float(_skew(sig))
        # Spectral features
        power   = np.abs(np.fft.rfft(sig)) ** 2
        freqs   = np.fft.rfftfreq(len(sig), d=1.0 / fs)
        max_psp = float(power.max())
        p_sum   = power.sum()
        if p_sum > 0:
            cumsum  = np.cumsum(power)
            mdf_idx = int(np.searchsorted(cumsum, p_sum / 2))
            mdf     = float(freqs[min(mdf_idx, len(freqs) - 1)])
            mnf     = float(np.sum(freqs * power) / p_sum)
        else:
            mdf = mnf = 0.0
        return dict(rms=rms, var=var, std=std, wf=wf, kurtosis=kurt,
                    skewness=sk, max_psp=max_psp, mdf=mdf, mnf=mnf)

    f_d = _features(denoised)
    f_c = _features(clean)

    ratios: dict = {}
    for k in f_c:
        c = f_c[k]
        ratios[k] = float(f_d[k] / c) if abs(c) > 1e-12 else float("nan")

    # Scalar summary: mean absolute deviation from 1.0 over key PD features
    key_feats = ["kurtosis", "mdf", "mnf", "rms"]
    errs = [abs(ratios[k] - 1.0) for k in key_feats
            if not np.isnan(ratios.get(k, float("nan")))]
    ratios["pd_feature_error"] = float(np.mean(errs)) if errs else float("nan")
    return ratios


def find_burst_region(clean: np.ndarray, window: int = ZOOM_LEN) -> tuple[int, int]:
    """Locate the most energetic contiguous region in the clean signal."""
    energy = np.convolve(clean ** 2, np.ones(window) / window, mode="valid")
    peak_idx = int(np.argmax(energy))
    start = max(0, peak_idx)
    end   = min(len(clean), start + window)
    return start, end

# ── Model loading ──────────────────────────────────────────────────────────────

def find_checkpoint(model_id: str) -> Path | None:
    candidates = [
        FYP_ROOT / "results" / model_id / "checkpoints" / "best.pt",
        FYP_ROOT / "results" / model_id / "best.pt",
        FYP_ROOT / "NEW FOCUS" / "results" / model_id / "checkpoints" / "best.pt",
    ]
    glob_root = FYP_ROOT / "results" / model_id / "checkpoints"
    if glob_root.is_dir():
        pts = sorted(glob_root.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        candidates = pts + candidates
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
            use_wavelet_pooling=flags["W"], use_bigru=flags["B"],
            use_swin=flags["S"], use_attention_gates=flags["A"],
            use_mtl=flags["M"], loss_config=flags["loss"], model_id=model_id,
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
        print(f"  [ok   ] {model_id:35s} {n_p:.2f}M")
        return model.eval().to(device)
    except Exception as exc:
        print(f"  [load-err] {model_id}: {exc}")
        return None

# ── Test data ──────────────────────────────────────────────────────────────────

def load_test_windows(snr_db: float, n: int = N_WINDOWS, seed: int = 42, scaffold: bool = False):
    """Return (noisy, clean) arrays of shape (n, 1, L)."""
    rng = np.random.default_rng(seed)
    L = 2048

    if not scaffold:
        for tp in [FYP_ROOT / "data" / "test_synthetic_ae.pt",
                   FYP_ROOT / "NEW FOCUS" / "data" / "test_synthetic_ae.pt"]:
            if not tp.exists():
                continue
            ds = torch.load(tp, map_location="cpu", weights_only=False)
            noisy_all = ds["noisy"] if isinstance(ds, dict) else ds.tensors[0]
            clean_all = ds["clean"] if isinstance(ds, dict) else ds.tensors[1]
            snr_key = next((k for k in ("snr_db","snr_labels","snr") if k in ds), None)
            if isinstance(ds, dict) and snr_key:
                snr_t = ds[snr_key]
                mask  = (snr_t > snr_db - 2.5) & (snr_t < snr_db + 2.5)
                idx   = mask.nonzero(as_tuple=True)[0]
                if len(idx) >= n:
                    g = torch.Generator().manual_seed(seed)
                    chosen = idx[torch.randperm(len(idx), generator=g)[:n]]
                    noisy = noisy_all[chosen].float()
                    clean = clean_all[chosen].float()
                    if noisy.dim() == 2:
                        noisy, clean = noisy.unsqueeze(1), clean.unsqueeze(1)
                    return noisy.numpy(), clean.numpy()
            # fallback: re-corrupt at desired SNR
            chosen_idx = rng.choice(noisy_all.shape[0], size=n, replace=False)
            clean = clean_all[chosen_idx].float()
            if clean.dim() == 2:
                clean = clean.unsqueeze(1)
            pw = (clean ** 2).mean(dim=-1, keepdim=True)
            sigma = torch.sqrt(pw / (10 ** (snr_db / 10))).clamp(min=1e-8)
            noisy = clean + torch.randn_like(clean) * sigma
            return noisy.numpy(), clean.numpy()

    # Gaussian scaffold: mixed PD templates (treeing burst, surface ramp, corona doublet, sparse decay)
    print(f"  [scaffold] SNR={snr_db} dB  (no real test set found)")
    clean = np.zeros((n, 1, L), dtype=np.float32)
    t_s = np.arange(L, dtype=np.float64) / FS_HZ
    for i in range(n):
        t = np.linspace(0, 1, L)
        clean[i, 0, :] = np.exp(-4 * t) * np.sin(2 * np.pi * 50e3 * t_s) * 0.08

        tpl = rng.integers(0, 4)
        if tpl == 0:
            # Decaying sinusoidal burst (treeing-like)
            pos = rng.integers(200, L - 400)
            wlen = rng.integers(80, 200)
            f0 = rng.uniform(80e3, 200e3)
            tau = wlen / (4 * FS_HZ)
            seg = np.arange(wlen, dtype=np.float64) / FS_HZ
            burst = np.exp(-seg / tau) * np.sin(2 * np.pi * f0 * seg)
            burst *= rng.uniform(0.8, 1.8) / (np.max(np.abs(burst)) + 1e-8)
            end = min(pos + wlen, L)
            clean[i, 0, pos:end] += burst[: end - pos].astype(np.float32)
        elif tpl == 1:
            # Step-ramp discharge (surface-like): fast rise, slow exponential fall
            pos = rng.integers(150, L - 250)
            rise = rng.integers(3, 10)
            fall = rng.integers(40, 120)
            amp = rng.uniform(0.9, 2.0) * rng.choice([-1.0, 1.0])
            pulse = np.zeros(fall + rise, dtype=np.float32)
            pulse[:rise] = amp * np.linspace(0, 1, rise, endpoint=False)
            tail = np.linspace(1, 0, fall, endpoint=False) ** 1.5
            pulse[rise:] = amp * tail
            end = min(pos + len(pulse), L)
            clean[i, 0, pos:end] += pulse[: end - pos]
        elif tpl == 2:
            # Double-pulse corona-like (two decaying lobes, us separation)
            pos = rng.integers(120, L - 350)
            gap = int(rng.uniform(8e-6, 35e-6) * FS_HZ)
            for k in range(2):
                p0 = pos + k * gap
                width = rng.integers(8, 24)
                decay = np.exp(-np.arange(width) / (width * 0.3))
                amp = rng.uniform(0.5, 1.4) * rng.choice([-1.0, 1.0])
                end = min(p0 + width, L)
                clean[i, 0, p0:end] += (amp * decay[: end - p0]).astype(np.float32)
        else:
            # Original sparse exponential PDlets
            for _ in range(rng.integers(5, 14)):
                pos = rng.integers(100, L - 100)
                amp = rng.uniform(0.5, 2.0) * rng.choice([-1, 1])
                width = rng.integers(6, 20)
                decay = np.exp(-np.arange(width) / (width * 0.25))
                end = min(pos + width, L)
                clean[i, 0, pos:end] += amp * decay[: end - pos]
    power = (clean ** 2).mean(axis=-1, keepdims=True)
    sigma = np.sqrt(power / (10 ** (snr_db / 10))).clip(min=1e-8)
    noisy = clean + rng.standard_normal(clean.shape).astype(np.float32) * sigma
    return noisy, clean

# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer(model: torch.nn.Module, noisy: np.ndarray, device: torch.device) -> np.ndarray:
    x   = torch.from_numpy(noisy).float().to(device)
    out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    elif isinstance(out, dict):
        out = out.get("denoised") or next(iter(out.values()))
    return out.cpu().numpy()

# ── Figure: per-SNR overview  ─────────────────────────────────────────────────

def fig_overview(results: dict[str, dict], snr_db: int, out_path: Path) -> None:
    """
    One row per model.
    Left col: mean signal ± std band (clean / denoised) with grey noisy.
    Right col: normalised residual (denoised − clean) / RMS(clean).
    """
    model_ids = list(results.keys())
    n_models  = len(model_ids)
    if n_models == 0:
        return

    L = results[model_ids[0]]["clean_mean"].shape[-1]
    t = np.arange(L)

    with plt.rc_context(RC):
        fig = plt.figure(figsize=(13, max(2.2 * n_models, 6)), constrained_layout=True)
        fig.suptitle(
            f"AE-PD Denoising Comparison  ·  Input SNR = {snr_db:+d} dB  "
            f"·  mean ± 1σ over {N_WINDOWS} windows",
            fontsize=11, fontweight="bold",
        )
        gs = gridspec.GridSpec(n_models, 2, figure=fig, wspace=0.06)

        for row, mid in enumerate(model_ids):
            r = results[mid]
            ncc_val = r["ncc_mean"]
            snr_imp = r["snr_imp_mean"]
            ae_r    = r["ae_ratio_mean"]

            ax_sig = fig.add_subplot(gs[row, 0])
            ax_res = fig.add_subplot(gs[row, 1])

            # ── signal overlay ──
            ax_sig.fill_between(t,
                r["noisy_mean"] - r["noisy_std"],
                r["noisy_mean"] + r["noisy_std"],
                color=C["noisy"], alpha=0.25)
            ax_sig.plot(t, r["clean_mean"],    color=C["clean"],    lw=1.2, label="Clean GT" if row==0 else "")
            ax_sig.plot(t, r["denoised_mean"], color=C["denoised"], lw=0.9, alpha=0.9, label="Denoised" if row==0 else "")
            ax_sig.fill_between(t,
                r["denoised_mean"] - r["denoised_std"],
                r["denoised_mean"] + r["denoised_std"],
                color=C["denoised"], alpha=0.15)

            label = SHORT_NAMES.get(mid, mid)
            ax_sig.set_ylabel(label, fontsize=7, rotation=0, labelpad=75, va="center")
            ax_sig.tick_params(labelsize=6)

            # ── normalised residual ──
            rms_clean = float(r["clean_std"].mean()) + 1e-8
            norm_res  = r["resid_mean"] / rms_clean
            ax_res.fill_between(t,
                (r["resid_mean"] - r["resid_std"]) / rms_clean,
                (r["resid_mean"] + r["resid_std"]) / rms_clean,
                color=C["residual"], alpha=0.2)
            ax_res.plot(t, norm_res, color=C["residual"], lw=0.7)
            ax_res.axhline(0, color="k", lw=0.4, linestyle="--")

            # Title: NCC / SNR-imp / AE ratio
            ax_res.set_title(
                f"NCC={ncc_val:.3f} | ΔSNR={snr_imp:+.1f}dB | AE-ratio={ae_r:.2f}",
                fontsize=7, pad=2
            )
            ax_res.tick_params(labelsize=6)

            for ax in (ax_sig, ax_res):
                ax.set_xlim(0, L - 1)
            if row < n_models - 1:
                ax_sig.set_xticklabels([])
                ax_res.set_xticklabels([])
            else:
                ax_sig.set_xlabel("Sample index", fontsize=7)
                ax_res.set_xlabel("Sample index", fontsize=7)

        if n_models > 0:
            # Column headers (first row only)
            axs0 = fig.axes[0]
            axs0.set_title("Signal overlay  (green=clean, red=denoised, grey band=noisy ±σ)",
                           fontsize=8, pad=3)
            fig.axes[1].set_title("Normalised residual  (denoised − clean) / RMS(clean)",
                                  fontsize=8, pad=3)
            fig.axes[0].legend(fontsize=6, loc="upper right", framealpha=0.7)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {out_path.name}")


# ── Figure: pulse-zoom top-N  ─────────────────────────────────────────────────

def fig_pulse_zoom(results: dict[str, dict], snr_db: int, out_path: Path) -> None:
    """
    Zoom into the burst region of the representative window.
    Shows Top-N models ranked by NCC, plus noisy and clean for comparison.
    """
    # Rank by NCC descending
    ranked = sorted(results.items(), key=lambda x: x[1]["ncc_mean"], reverse=True)
    top = ranked[:TOP_N]
    if not top:
        return

    L    = top[0][1]["clean_rep"].shape[-1]
    s, e = find_burst_region(top[0][1]["clean_rep"].ravel())
    t    = np.arange(s, e)
    t_us = t / FS_HZ * 1e6  # microseconds

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, TOP_N, figsize=(3.5 * TOP_N, 3.8), sharey=True,
                                 constrained_layout=True)
        if TOP_N == 1:
            axes = [axes]
        fig.suptitle(
            f"PD Pulse Zoom  ·  SNR_in = {snr_db:+d} dB  ·  "
            f"Peak-energy region, sample {s}–{e}  ·  Top-{TOP_N} by NCC",
            fontsize=10, fontweight="bold",
        )
        for ax, (mid, r) in zip(axes, top):
            ax.plot(t_us, r["noisy_rep"].ravel()[s:e],    color=C["noisy"],    lw=0.7, alpha=0.6, label="Noisy")
            ax.plot(t_us, r["clean_rep"].ravel()[s:e],    color=C["clean"],    lw=1.4, label="Clean GT")
            ax.plot(t_us, r["denoised_rep"].ravel()[s:e], color=C["denoised"], lw=1.0, label="Denoised")
            short = SHORT_NAMES.get(mid, mid)
            ax.set_title(
                f"{short}\nNCC={r['ncc_mean']:.3f}  ΔSNR={r['snr_imp_mean']:+.1f} dB",
                fontsize=8
            )
            ax.set_xlabel("Time (µs)", fontsize=8)
        axes[0].set_ylabel("Amplitude (norm.)", fontsize=8)
        axes[0].legend(fontsize=7, loc="upper left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {out_path.name}")


# ── Figure: AE-band PSD comparison ───────────────────────────────────────────

def fig_ae_psd(results_by_snr: dict[int, dict], out_path: Path) -> None:
    """
    One panel per SNR level, showing mean PSD of noisy / clean / best denoised
    in the AE band (20–500 kHz).
    """
    from scipy.signal import welch
    n_snr = len(EVAL_SNRS)

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, n_snr, figsize=(4.5 * n_snr, 3.8),
                                 constrained_layout=True)
        fig.suptitle("AE-Band Power Spectral Density  ·  Best model per SNR level",
                     fontsize=10, fontweight="bold")

        for ax, snr_db in zip(axes, EVAL_SNRS):
            res = results_by_snr.get(snr_db, {})
            if not res:
                ax.set_title(f"SNR={snr_db:+d} dB\n(no data)")
                continue
            ranked = sorted(res.items(), key=lambda x: x[1]["ncc_mean"], reverse=True)
            best_mid, best_r = ranked[0]

            def mean_psd(sig2d):
                """sig2d: (n, L)"""
                psds = []
                for s in sig2d:
                    f, p = welch(s, fs=FS_HZ, nperseg=min(512, len(s)))
                    psds.append(p)
                return f, np.array(psds).mean(axis=0)

            f, p_noisy = mean_psd(best_r["noisy_all"])
            _, p_clean = mean_psd(best_r["clean_all"])
            _, p_den   = mean_psd(best_r["denoised_all"])

            mask = (f >= AE_LO_HZ) & (f <= AE_HI_HZ)
            f_khz = f / 1e3

            ax.semilogy(f_khz[mask], p_noisy[mask],  color=C["noisy"],    lw=0.8, alpha=0.7, label="Noisy")
            ax.semilogy(f_khz[mask], p_clean[mask],  color=C["clean"],    lw=1.4, label="Clean GT")
            ax.semilogy(f_khz[mask], p_den[mask],    color=C["denoised"], lw=1.0, label=f"Best: {SHORT_NAMES.get(best_mid, best_mid)}")
            ax.axvspan(AE_LO_HZ/1e3, AE_HI_HZ/1e3, color=C["band"], alpha=0.04)
            ax.set_xlabel("Frequency (kHz)")
            if ax == axes[0]:
                ax.set_ylabel("PSD (V²/Hz)")
            ax.set_title(f"SNR_in = {snr_db:+d} dB\nNCC={best_r['ncc_mean']:.3f}", fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(True, which="both", alpha=0.25)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {out_path.name}")


# ── Figure: ranked heatmap ────────────────────────────────────────────────────

def fig_ranked_heatmap(all_results: dict[int, dict[str, dict]], out_path: Path) -> None:
    """
    Three stacked heat-maps (shared row order): NCC, mean SNR improvement (dB),
    mean AE-band energy ratio vs clean. Row order = descending mean NCC (thesis default).
    """
    snr_cols = EVAL_SNRS
    all_mids = list(MODEL_REGISTRY.keys())
    n_m, n_s = len(all_mids), len(snr_cols)

    ncc_m = np.full((n_m, n_s), np.nan)
    snr_m = np.full((n_m, n_s), np.nan)
    ae_m = np.full((n_m, n_s), np.nan)
    for j, snr_db in enumerate(snr_cols):
        res = all_results.get(snr_db, {})
        for i, mid in enumerate(all_mids):
            if mid not in res:
                continue
            r = res[mid]
            ncc_m[i, j] = r["ncc_mean"]
            snr_m[i, j] = r["snr_imp_mean"]
            ae_m[i, j] = r["ae_ratio_mean"]

    row_means_ncc = np.nanmean(ncc_m, axis=1)
    order = np.argsort(row_means_ncc)[::-1]
    ncc_m = ncc_m[order]
    snr_m = snr_m[order]
    ae_m = ae_m[order]
    row_labels = [SHORT_NAMES.get(all_mids[i], all_mids[i]) for i in order]
    col_labels = [f"SNR {s:+d} dB" for s in snr_cols]

    snr_vmax = float(np.nanpercentile(snr_m, 95)) if np.isfinite(np.nanmax(snr_m)) else 1.0
    snr_vmin = float(np.nanpercentile(snr_m, 5)) if np.isfinite(np.nanmin(snr_m)) else -1.0
    if snr_vmax <= snr_vmin:
        snr_vmin, snr_vmax = -0.5, 0.5
    snr_vmin = min(snr_vmin, 0.0)

    try:
        from matplotlib import colormaps as mpl_cmaps

        cmap_snr = mpl_cmaps["RdYlGn"]
        cmap_ae = mpl_cmaps["PuOr_r"]
    except Exception:
        cmap_snr = plt.cm.get_cmap("RdYlGn")
        cmap_ae = plt.cm.get_cmap("PuOr_r")

    with plt.rc_context(RC):
        fig, axes = plt.subplots(
            3, 1, figsize=(4 + n_s * 1.25, 0.45 * n_m * 3 + 2.2),
            sharex=True, constrained_layout=True,
        )
        fig.suptitle(
            "Ablation summary: NCC | SNR improvement | AE-band energy ratio\n"
            "(rows sorted by mean NCC; AE ratio 1.0 = ideal band preservation)",
            fontsize=10, fontweight="bold",
        )

        ims = []
        # --- NCC ---
        ax0 = axes[0]
        im0 = ax0.imshow(ncc_m, aspect="auto", vmin=0, vmax=1, cmap=CMAP_HEATMAP)
        ims.append(im0)
        ax0.set_yticks(range(n_m))
        ax0.set_yticklabels(row_labels, fontsize=8)
        ax0.set_title("NCC (higher better)", fontsize=9, loc="left")
        ax0.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
        for i in range(n_m):
            for j in range(n_s):
                v = ncc_m[i, j]
                if np.isnan(v):
                    ax0.text(j, i, "-", ha="center", va="center", fontsize=7, color="#888")
                else:
                    tc = "white" if v < 0.5 else "black"
                    ax0.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=tc, fontweight="bold")
        fig.colorbar(im0, ax=ax0, fraction=0.035, pad=0.02, label="NCC")

        # --- Delta SNR ---
        ax1 = axes[1]
        im1 = ax1.imshow(snr_m, aspect="auto", cmap=cmap_snr, vmin=snr_vmin, vmax=max(snr_vmax, 0.05))
        ims.append(im1)
        ax1.set_yticks(range(n_m))
        ax1.set_yticklabels(row_labels, fontsize=8)
        ax1.set_title("Mean SNR improvement (dB) vs noisy input", fontsize=9, loc="left")
        for i in range(n_m):
            for j in range(n_s):
                v = snr_m[i, j]
                if np.isnan(v):
                    ax1.text(j, i, "-", ha="center", va="center", fontsize=7, color="#888")
                else:
                    ax1.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7, fontweight="bold")
        fig.colorbar(im1, ax=ax1, fraction=0.035, pad=0.02, label="dB")

        # --- AE ratio (1 = ideal) ---
        ax2 = axes[2]
        ae_norm = TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=1.5)
        im2 = ax2.imshow(ae_m, aspect="auto", cmap=cmap_ae, norm=ae_norm)
        ax2.set_xticks(range(n_s))
        ax2.set_xticklabels(col_labels, fontsize=9)
        ax2.set_yticks(range(n_m))
        ax2.set_yticklabels(row_labels, fontsize=8)
        ax2.set_title("AE-band energy ratio (denoised / clean PSD in 20-500 kHz)", fontsize=9, loc="left")
        for i in range(n_m):
            for j in range(n_s):
                v = ae_m[i, j]
                if np.isnan(v):
                    ax2.text(j, i, "-", ha="center", va="center", fontsize=7, color="#888")
                else:
                    ax2.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, fontweight="bold")
        fig.colorbar(im2, ax=ax2, fraction=0.035, pad=0.02, label="ratio")

        for i, mean_val in enumerate(row_means_ncc[order]):
            axes[2].annotate(
                f"mean NCC={mean_val:.3f}",
                xy=(n_s - 0.35, i),
                xycoords="data",
                fontsize=6,
                va="center",
                color="#444",
                ha="left",
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {out_path.name}")


# ── Figure: NCC bar chart with error bars ────────────────────────────────────

def fig_ncc_bar(all_results: dict[int, dict[str, dict]], out_path: Path) -> None:
    snr_cols   = EVAL_SNRS
    all_mids   = list(MODEL_REGISTRY.keys())
    x          = np.arange(len(all_mids))
    width      = 0.22
    bar_colors = ["#e74c3c", "#3498db", "#2ecc71"]

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(max(12, len(all_mids) * 1.1), 5),
                               constrained_layout=True)
        ax.set_title("Mean NCC ± 1σ  ·  All Ablation Variants",
                     fontsize=11, fontweight="bold")

        for k, (snr_db, col) in enumerate(zip(snr_cols, bar_colors)):
            res = all_results.get(snr_db, {})
            means = [res.get(m, {}).get("ncc_mean", np.nan) for m in all_mids]
            stds  = [res.get(m, {}).get("ncc_std",  0.0)    for m in all_mids]
            ax.bar(x + k * width, means, width,
                   label=f"SNR {snr_db:+d} dB", color=col, alpha=0.82,
                   yerr=stds, error_kw=dict(elinewidth=0.8, capsize=2))

        ax.set_xticks(x + width)
        ax.set_xticklabels(
            [SHORT_NAMES.get(m, m) for m in all_mids],
            rotation=38, ha="right", fontsize=8
        )
        ax.set_ylabel("NCC (mean over windows)")
        ax.set_ylim(0, 1.08)
        ax.axhline(0.9, color="k", lw=0.7, linestyle="--", alpha=0.4)
        ax.text(len(all_mids) - 0.3, 0.91, "0.9 threshold", fontsize=7, color="#555")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {out_path.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    _configure_stdio_utf8()
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaffold", action="store_true",
                        help="Use Gaussian scaffold (no real checkpoints needed)")
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"Output: {OUT_DIR}\n")

    model_ids = args.models or list(MODEL_REGISTRY.keys())

    # ── Load models ──
    print("Loading models …")
    loaded: dict[str, torch.nn.Module] = {}
    for mid in model_ids:
        if args.scaffold:
            # Scaffold mode: build un-initialised model for shape testing
            m = build_model(mid, MODEL_REGISTRY[mid])
            if m is not None:
                loaded[mid] = m.eval().to(device)
            else:
                print(f"  [skip ] {mid}: build failed (scaffold mode)")
        else:
            m = load_model(mid, MODEL_REGISTRY[mid], device)
            if m is not None:
                loaded[mid] = m

    print(f"\nLoaded {len(loaded)}/{len(model_ids)} models.\n")
    if not loaded and not args.scaffold:
        print("[FATAL] No models loaded.")
        print(f"  Expected checkpoints at: {FYP_ROOT / 'results' / '<model>' / 'checkpoints' / 'best.pt'}")
        return

    # ── Evaluate across SNR levels ──
    all_results: dict[int, dict[str, dict]] = {}
    summary_json = {}

    for snr_db in EVAL_SNRS:
        print(f"\n── SNR = {snr_db:+d} dB ──────────────────────────────────")
        noisy_all, clean_all = load_test_windows(snr_db, N_WINDOWS, seed=snr_db+100,
                                                  scaffold=args.scaffold)
        # noisy_all, clean_all: (N_WINDOWS, 1, L)

        res: dict[str, dict] = {}

        for mid, model in loaded.items():
            try:
                den_all = infer(model, noisy_all, device)  # (N, 1, L)
            except Exception as exc:
                print(f"  [err] {mid}: {exc}")
                continue

            nccs    = [ncc(den_all[i,0], clean_all[i,0])                       for i in range(N_WINDOWS)]
            snr_imps= [snr_improvement_db(noisy_all[i,0], den_all[i,0], clean_all[i,0]) for i in range(N_WINDOWS)]
            ae_ratios=[ae_band_energy_ratio(den_all[i,0], clean_all[i,0])      for i in range(N_WINDOWS)]
            pk_ratios=[peak_recovery_ratio(den_all[i,0], clean_all[i,0])       for i in range(N_WINDOWS)]

            pd_rets = [pd_feature_retention(den_all[i, 0], clean_all[i, 0])
                       for i in range(N_WINDOWS)]
            pd_errors = [r["pd_feature_error"] for r in pd_rets
                         if not np.isnan(r.get("pd_feature_error", float("nan")))]
            kurt_ratios = [r["kurtosis"] for r in pd_rets
                           if not np.isnan(r.get("kurtosis", float("nan")))]

            nccs     = [v for v in nccs     if not np.isnan(v)]
            snr_imps = [v for v in snr_imps if not np.isnan(v) and not np.isinf(v)]
            ae_ratios= [v for v in ae_ratios if not np.isnan(v)]
            pk_ratios= [v for v in pk_ratios if not np.isnan(v)]

            # Representative window = the one whose NCC is closest to the mean
            if nccs:
                mean_ncc   = float(np.mean(nccs))
                rep_idx    = int(np.argmin(np.abs(np.array(nccs) - mean_ncc)))
            else:
                mean_ncc   = 0.0
                rep_idx    = 0

            res[mid] = dict(
                ncc_mean       = mean_ncc,
                ncc_std        = float(np.std(nccs))        if nccs     else 0.0,
                snr_imp_mean   = float(np.mean(snr_imps))  if snr_imps else 0.0,
                snr_imp_std    = float(np.std(snr_imps))   if snr_imps else 0.0,
                ae_ratio_mean  = float(np.mean(ae_ratios)) if ae_ratios else 0.0,
                peak_err_mean  = float(np.mean(pk_ratios)) if pk_ratios else 0.0,
                pd_feat_error_mean=float(np.mean(pd_errors)) if pd_errors else float("nan"),
                pd_feat_error_std=float(np.std(pd_errors)) if pd_errors else float("nan"),
                kurt_ratio_mean=float(np.mean(kurt_ratios)) if kurt_ratios else float("nan"),
                # Per-window arrays for mean ± std band
                noisy_mean     = noisy_all[:, 0, :].mean(axis=0),
                noisy_std      = noisy_all[:, 0, :].std(axis=0),
                clean_mean     = clean_all[:, 0, :].mean(axis=0),
                clean_std      = clean_all[:, 0, :].std(axis=0),
                denoised_mean  = den_all[:, 0, :].mean(axis=0),
                denoised_std   = den_all[:, 0, :].std(axis=0),
                resid_mean     = (den_all - clean_all)[:, 0, :].mean(axis=0),
                resid_std      = (den_all - clean_all)[:, 0, :].std(axis=0),
                # Representative window for zoom panels
                noisy_rep      = noisy_all[rep_idx],
                clean_rep      = clean_all[rep_idx],
                denoised_rep   = den_all[rep_idx],
                # Full batch for PSD
                noisy_all      = noisy_all[:, 0, :],
                clean_all      = clean_all[:, 0, :],
                denoised_all   = den_all[:, 0, :],
            )

            print(f"  {mid:35s}  NCC={mean_ncc:.4f}±{res[mid]['ncc_std']:.4f}"
                  f"  ΔSNR={res[mid]['snr_imp_mean']:+.2f}dB"
                  f"  AE-ratio={res[mid]['ae_ratio_mean']:.3f}"
                  f"  PD-err={res[mid]['pd_feat_error_mean']:.4f}"
                  f"  Kurt-ratio={res[mid]['kurt_ratio_mean']:.3f}")

        all_results[snr_db] = res

        # ── Per-SNR overview ──
        fig_overview(res, snr_db, OUT_DIR / f"overview_snr{snr_db:+d}dB.png")
        fig_pulse_zoom(res, snr_db, OUT_DIR / f"pulse_zoom_snr{snr_db:+d}dB.png")

        # JSON
        summary_json[f"snr_{snr_db:+d}dB"] = {
            mid: dict(
                ncc_mean     = float(v["ncc_mean"]),
                ncc_std      = float(v["ncc_std"]),
                snr_imp_mean = float(v["snr_imp_mean"]),
                snr_imp_std  = float(v["snr_imp_std"]),
                ae_ratio     = float(v["ae_ratio_mean"]),
                peak_err     = float(v["peak_err_mean"]),
                pd_feat_error=float(v["pd_feat_error_mean"]),
                kurt_ratio=float(v["kurt_ratio_mean"]),
            )
            for mid, v in res.items()
        }

        # ── Free memory between SNR levels ──
        del noisy_all, clean_all

    # ── Cross-SNR summary figures ──
    if all_results:
        fig_ae_psd(all_results, OUT_DIR / "ae_band_psd.png")
        fig_ranked_heatmap(all_results, OUT_DIR / "ranked_heatmap.png")
        fig_ncc_bar(all_results, OUT_DIR / "ncc_summary_bar.png")

    # ── Save JSON ──
    json_path = OUT_DIR / "method_comparison_summary.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"\nSummary JSON → {json_path}")
    print(f"\n{'='*60}")
    print("FINAL RANKING  (mean NCC across all SNR levels; PD-err / Kurt per SNR)")
    print(f"{'='*60}")
    ranking: dict[str, list[float]] = {}
    ranking_pd: dict[str, list[float]] = {}
    ranking_kurt: dict[str, list[float]] = {}
    for snr_db in EVAL_SNRS:
        res = all_results.get(snr_db, {})
        for mid, r in res.items():
            if mid not in ranking:
                ranking[mid] = []
                ranking_pd[mid] = []
                ranking_kurt[mid] = []
            ranking[mid].append(float(r["ncc_mean"]))
            ranking_pd[mid].append(float(r["pd_feat_error_mean"]))
            ranking_kurt[mid].append(float(r["kurt_ratio_mean"]))
    for rank, (mid, vals) in enumerate(
        sorted(ranking.items(), key=lambda x: np.mean(x[1]), reverse=True), 1
    ):
        per_snr = "  ".join(f"{v:.3f}" for v in vals)
        pd_e = ranking_pd[mid]
        kr = ranking_kurt[mid]
        pd_str = "  ".join(f"{x:.3f}" for x in pd_e)
        kr_str = "  ".join(f"{x:.3f}" for x in kr)
        print(
            f"  #{rank:2d}  {SHORT_NAMES.get(mid, mid):20s}  mean_NCC={np.mean(vals):.4f}  [{per_snr}]"
            f"  PD-err=[{pd_str}]  Kurt=[{kr_str}]"
        )


if __name__ == "__main__":
    main()
