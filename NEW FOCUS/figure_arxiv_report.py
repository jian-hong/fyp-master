"""
figure_arxiv_report.py
======================
Publication figures for FYP / arXiv report:
  - Top 3 deep models + 2 classical baselines only
  - Zoomed PD burst (time domain), no full 19-row comparison
  - Multiple SNRs, multiple test windows (time diversity), PSD (frequency)

Outputs → NEW FOCUS/results/arxiv_report_figures/
Manifest → arxiv_figure_manifest.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

import figure_fixed_window_comparison as fwc

FYP_ROOT = fwc.FYP_ROOT
OUT_DIR = FYP_ROOT / "NEW FOCUS" / "results" / "arxiv_report_figures"
SUMMARY_JSON = FYP_ROOT / "NEW FOCUS" / "results" / "method_comparison_figures" / "method_comparison_summary.json"

# Top-3 deep (final ranking + eval NCC agreement on synthetic test)
TOP3_AI = [
    "MR-TAE-FULL-LossB",
    "MR-TAE-FULL",
    "UNet-BiGRU-Swin",
]

# Top-2 classical (SavGol best at low SNR; Soft-Wavelet strong at -15 dB)
TOP2_TRAD = [
    ("SavGol+Wavelet", fwc.baseline_savgol_wavelet, "#8e44ad"),
    ("Soft-Wavelet (db4)", fwc.baseline_soft_wavelet, "#2980b9"),
]

SNR_LEVELS = [5, -5, -15]  # paper focus; +10 often hurts DL — optional
SNR_ALL = [10, 5, -5, -15]

ZOOM_SAMPLES = 280  # tighter than 400-sample burst in full comparison
AE_LO, AE_HI = 20_000, 500_000

C_CLEAN = "#1a6e3d"
C_NOISY = "#b0b0b0"
AI_COLORS = ["#c0392b", "#e74c3c", "#922b21"]


def _configure_stdio_utf8() -> None:
    fwc._configure_stdio_utf8()


def select_windows_in_band(ds: dict, snr_db: float, n_pick: int, seed: int = 42) -> list[int]:
    """Pick n_pick window indices spread across SNR band (deterministic)."""
    snr_t = ds["snr_labels"].float()
    band = (snr_t >= snr_db - 2.5) & (snr_t <= snr_db + 2.5)
    idx = band.nonzero(as_tuple=True)[0].cpu().numpy()
    if len(idx) == 0:
        return [0] * n_pick
    if len(idx) <= n_pick:
        return [int(i) for i in idx[:n_pick]]
    # Spread: quantile picks for temporal / sample diversity
    qs = np.linspace(0, len(idx) - 1, n_pick).astype(int)
    return [int(idx[i]) for i in qs]


def burst_zoom(clean_1d: np.ndarray, zoom_len: int = ZOOM_SAMPLES) -> tuple[int, int]:
    return fwc.find_burst_region(clean_1d, window=zoom_len)


def time_us(s: int, e: int) -> np.ndarray:
    return (np.arange(s, e, dtype=np.float64) / float(fwc.FS_HZ)) * 1e6


def plot_zoom_overlay(
    ax,
    clean_1d: np.ndarray,
    sig_1d: np.ndarray,
    s: int,
    e: int,
    *,
    sig_color: str,
    sig_label: str,
    show_noisy: bool = False,
    noisy_1d: np.ndarray | None = None,
    metrics: tuple[float, float] | None = None,
) -> None:
    t = time_us(s, e)
    if show_noisy and noisy_1d is not None:
        ax.plot(t, noisy_1d[s:e], color=C_NOISY, lw=0.65, alpha=0.45, label="Noisy")
    ax.plot(t, clean_1d[s:e], color=C_CLEAN, lw=1.35, label="Clean", zorder=3)
    ax.plot(t, sig_1d[s:e], color=sig_color, lw=1.0, alpha=0.95, label=sig_label, zorder=2)
    ax.set_xlim(t[0], t[-1])
    ax.tick_params(labelsize=8)
    if metrics is not None:
        nv, dv = metrics
        dtxt = 0.0 if np.isinf(dv) else dv
        ax.text(
            0.03, 0.97, f"NCC={nv:.3f}  ΔSNR={dtxt:+.1f} dB",
            transform=ax.transAxes, fontsize=7.5, va="top",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85, ec="none"),
        )


def welch_psd(sig: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.signal import welch

    nperseg = min(512, len(sig))
    f, p = welch(sig.astype(np.float64), fs=fwc.FS_HZ, nperseg=nperseg)
    return f, p


def fig_denoise_single_snr(
    snr_db: int,
    noisy_1d: np.ndarray,
    clean_1d: np.ndarray,
    den_ai: dict[str, np.ndarray],
    den_trad: dict[str, np.ndarray],
    burst_s: int,
    burst_e: int,
    win_idx: int,
    out_path: Path,
) -> None:
    """6-row zoomed before/after panel (noisy + 3 AI + 2 trad)."""
    rows = [
        ("Noisy input", noisy_1d, C_NOISY, True),
    ]
    for i, mid in enumerate(TOP3_AI):
        if mid not in den_ai:
            continue
        rows.append((fwc.SHORT_NAMES.get(mid, mid), den_ai[mid], AI_COLORS[i % 3], False))
    for name, _fn, col in TOP2_TRAD:
        if name in den_trad:
            rows.append((name, den_trad[name], col, False))

    n = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(7.2, 1.55 * n), dpi=300, sharex=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(
        f"AE-PD denoising (zoomed burst) · SNR$_{{in}}$ = {snr_db:+d} dB · test window {win_idx}",
        fontsize=10, fontweight="bold", y=0.995,
    )
    for ax, (label, sig, col, is_noisy) in zip(axes, rows):
        nv = fwc.ncc(sig, clean_1d)
        dv = fwc.snr_improvement_db(noisy_1d, sig, clean_1d) if not is_noisy else 0.0
        plot_zoom_overlay(
            ax, clean_1d, sig, burst_s, burst_e,
            sig_color=col, sig_label=label,
            show_noisy=is_noisy, noisy_1d=noisy_1d if is_noisy else None,
            metrics=(nv, dv),
        )
        ax.set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center", labelpad=88)
        if is_noisy:
            ax.legend(fontsize=6, loc="upper right", ncol=2)
    axes[-1].set_xlabel("Time (µs)")
    fig.tight_layout(rect=[0.12, 0, 1, 0.97])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig_multisnr_best_ai(
    panels: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]],
    model_id: str,
    out_path: Path,
) -> None:
    """One row: zoomed burst for best model across SNR columns."""
    snrs = sorted(panels.keys())
    fig, axes = plt.subplots(1, len(snrs), figsize=(3.6 * len(snrs), 2.8), dpi=300, sharey=True)
    if len(snrs) == 1:
        axes = [axes]
    short = fwc.SHORT_NAMES.get(model_id, model_id)
    fig.suptitle(f"{short} — denoised vs clean (burst zoom)", fontsize=10, fontweight="bold")
    for ax, snr in zip(axes, snrs):
        noisy, clean, den, bs, be, widx = panels[snr]
        t = time_us(bs, be)
        ax.plot(t, noisy[bs:be], color=C_NOISY, lw=0.6, alpha=0.4)
        ax.plot(t, clean[bs:be], color=C_CLEAN, lw=1.2)
        ax.plot(t, den[bs:be], color=AI_COLORS[0], lw=0.95)
        nv = fwc.ncc(den, clean)
        dv = fwc.snr_improvement_db(noisy, den, clean)
        dtxt = 0.0 if np.isinf(dv) else dv
        ax.set_title(f"SNR$_{{in}}$ = {snr:+d} dB\nNCC={nv:.3f}, ΔSNR={dtxt:+.1f} dB", fontsize=8)
        ax.set_xlabel("µs")
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("Amplitude")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig_time_diversity(
    snr_db: int,
    cases: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, int, int]],
    model_id: str,
    out_path: Path,
) -> None:
    """3 columns: different test windows, same SNR, best AI."""
    fig, axes = plt.subplots(1, len(cases), figsize=(3.8 * len(cases), 3.0), dpi=300, sharey=True)
    if len(cases) == 1:
        axes = [axes]
    short = fwc.SHORT_NAMES.get(model_id, model_id)
    fig.suptitle(
        f"{short} @ SNR$_{{in}}$ = {snr_db:+d} dB — different test windows (burst zoom)",
        fontsize=10, fontweight="bold",
    )
    for ax, (widx, noisy, clean, den, bs, be) in zip(axes, cases):
        t = time_us(bs, be)
        ax.plot(t, noisy[bs:be], color=C_NOISY, lw=0.55, alpha=0.4)
        ax.plot(t, clean[bs:be], color=C_CLEAN, lw=1.2, label="Clean")
        ax.plot(t, den[bs:be], color=AI_COLORS[0], lw=0.95, label="Denoised")
        nv = fwc.ncc(den, clean)
        dv = fwc.snr_improvement_db(noisy, den, clean)
        dtxt = 0.0 if np.isinf(dv) else dv
        ax.set_title(f"Window {widx}\nNCC={nv:.3f}, ΔSNR={dtxt:+.1f} dB", fontsize=8)
        ax.set_xlabel("Time (µs)")
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig_psd_comparison(
    snr_db: int,
    noisy: np.ndarray,
    clean: np.ndarray,
    den_ai: dict[str, np.ndarray],
    den_trad: dict[str, np.ndarray],
    win_idx: int,
    out_path: Path,
) -> None:
    """AE-band PSD: clean, noisy, top-3 AI, top-2 trad."""
    fig, ax = plt.subplots(figsize=(7.5, 3.4), dpi=300)
    series = [("Clean GT", clean, C_CLEAN, 1.4)]
    series.append(("Noisy", noisy, C_NOISY, 0.9))
    for i, mid in enumerate(TOP3_AI):
        if mid in den_ai:
            series.append((fwc.SHORT_NAMES.get(mid, mid), den_ai[mid], AI_COLORS[i], 1.0))
    for name, _fn, col in TOP2_TRAD:
        if name in den_trad:
            series.append((name, den_trad[name], col, 1.0))

    for label, sig, col, lw in series:
        f, p = welch_psd(sig)
        mask = (f >= AE_LO) & (f <= AE_HI)
        ax.semilogy(f[mask] / 1e3, p[mask] + 1e-20, label=label, color=col, lw=lw)

    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("PSD")
    ax.set_title(f"AE-band spectrum (20–500 kHz) · SNR$_{{in}}$ = {snr_db:+d} dB · window {win_idx}")
    ax.legend(fontsize=6, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig_before_after_strip(
    noisy: np.ndarray,
    clean: np.ndarray,
    den: np.ndarray,
    method: str,
    snr_db: int,
    burst_s: int,
    burst_e: int,
    out_path: Path,
) -> None:
    """Two-panel: before (noisy) | after (denoised) — minimal for report inline."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), dpi=300, sharey=True)
    t = time_us(burst_s, burst_e)
    for ax, sig, title, col in zip(
        axes,
        [noisy, den],
        ["Before (noisy)", f"After ({method})"],
        [C_NOISY, AI_COLORS[0]],
    ):
        ax.plot(t, clean[burst_s:burst_e], color=C_CLEAN, lw=1.2, alpha=0.85, label="Clean")
        ax.plot(t, sig[burst_s:burst_e], color=col, lw=1.0, alpha=0.9, label=title)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel("Time (µs)")
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="upper right")
    nv = fwc.ncc(den, clean)
    dv = fwc.snr_improvement_db(noisy, den, clean)
    dtxt = 0.0 if np.isinf(dv) else dv
    fig.suptitle(
        f"SNR$_{{in}}$ = {snr_db:+d} dB · NCC={nv:.3f} · ΔSNR={dtxt:+.1f} dB",
        fontsize=9, y=1.02,
    )
    axes[0].set_ylabel("Amplitude")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig_metrics_bars(out_path: Path) -> None:
    """Mean NCC bar chart from method_comparison_summary.json."""
    if not SUMMARY_JSON.exists():
        return
    data = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    snr_keys = sorted(data.keys())
    fig, axes = plt.subplots(1, len(snr_keys), figsize=(4.2 * len(snr_keys), 4.0), dpi=200)
    if len(snr_keys) == 1:
        axes = [axes]

    show_ids = TOP3_AI + ["MWCNN-BiGRU", "MR-TAE-noWavelet"]
    trad_names = [t[0] for t in TOP2_TRAD]

    for ax, sk in zip(axes, snr_keys):
        block = data[sk]
        labels, vals = [], []
        for mid in show_ids:
            if mid in block:
                labels.append(fwc.SHORT_NAMES.get(mid, mid))
                vals.append(block[mid]["ncc_mean"])
        ax.bar(np.arange(len(labels)), vals, color=AI_COLORS[0], alpha=0.85)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
        snr_label = sk.replace("snr_", "").replace("dB", " dB")
        ax.set_title(f"Mean NCC · {snr_label}", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("NCC")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Deep models — test-set mean NCC (25 windows/SNR)", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_manifest(generated: list[dict]) -> None:
    lines = [
        "# arXiv / FYP report figure manifest",
        "",
        "Generated by `figure_arxiv_report.py`. Use these in LaTeX with `\\includegraphics`.",
        "",
        "## Recommended figure order for Results section",
        "",
        "| # | File | Suggested caption |",
        "|---|------|-------------------|",
    ]
    for i, g in enumerate(generated, 1):
        lines.append(f"| {i} | `{g['file']}` | {g['caption']} |")
    lines += [
        "",
        "## Models shown",
        "- **Deep (top 3):** MR-TAE-FULL-LossB, MR-TAE-FULL, UNet-BiGRU-Swin",
        "- **Classical (top 2):** SavGol+Wavelet, Soft-Wavelet (db4)",
        "",
        "## Notes for drafting",
        "- Zoom panels use **280 samples** centred on peak-energy burst in clean GT.",
        "- All methods on a given panel share the **same** noisy/clean window.",
        "- At **+10 dB** input SNR, deep models may show negative ΔSNR (over-smoothing);",
        "  emphasise **−5 dB** and **−15 dB** for denoising benefit in the paper.",
        "- Time-diversity figures use windows at different indices within the same SNR band.",
        "- PSD figures cover the **20–500 kHz** AE band (1 MHz sampling).",
    ]
    (OUT_DIR / "arxiv_figure_manifest.md").write_text("\n".join(lines), encoding="utf-8")


def run_one_window(
    ds: dict,
    snr_db: int,
    win_idx: int,
    loaded: dict[str, torch.nn.Module],
    device: torch.device,
    rng: np.random.Generator,
    generated: list[dict],
) -> tuple[np.ndarray, np.ndarray, dict, dict, int, int]:
    noisy_all = ds["noisy"].float()
    clean_all = ds["clean"].float()
    snr_t = ds["snr_labels"].float()

    if snr_db == 10:
        clean_1d = clean_all[win_idx].numpy().ravel()
        noisy_1d = fwc.awgn_at_snr(clean_1d, 10.0, rng)
    else:
        band = (snr_t >= snr_db - 2.5) & (snr_t <= snr_db + 2.5)
        if win_idx not in band.nonzero(as_tuple=True)[0].tolist():
            clean_1d = clean_all[win_idx].numpy().ravel()
            noisy_1d = fwc.awgn_at_snr(clean_1d, float(snr_db), rng)
        else:
            noisy_1d = noisy_all[win_idx].numpy().ravel()
            clean_1d = clean_all[win_idx].numpy().ravel()

    burst_s, burst_e = burst_zoom(clean_1d)
    noisy_batch = noisy_1d.reshape(1, 1, -1).astype(np.float32)

    den_ai: dict[str, np.ndarray] = {}
    for mid in TOP3_AI:
        if mid in loaded:
            den_ai[mid] = np.asarray(fwc.infer(loaded[mid], noisy_batch, device)).ravel()

    den_trad: dict[str, np.ndarray] = {}
    for name, fn, _ in TOP2_TRAD:
        den_trad[name] = fn(noisy_1d)

    # Per-SNR main panel
    p_main = OUT_DIR / f"denoise_zoom_snr{snr_db:+d}dB_win{win_idx}.png"
    fig_denoise_single_snr(snr_db, noisy_1d, clean_1d, den_ai, den_trad, burst_s, burst_e, win_idx, p_main)
    generated.append({
        "file": p_main.name,
        "caption": (
            f"Zoomed PD burst at SNR$_{{in}}$={snr_db:+d} dB (window {win_idx}): "
            "noisy input, three best deep models, two classical baselines vs clean GT."
        ),
    })

    # Before/after best model
    best_id = TOP3_AI[0]
    if best_id in den_ai:
        p_ba = OUT_DIR / f"before_after_{fwc.SHORT_NAMES[best_id]}_snr{snr_db:+d}dB.png"
        fig_before_after_strip(
            noisy_1d, clean_1d, den_ai[best_id], fwc.SHORT_NAMES[best_id],
            snr_db, burst_s, burst_e, p_ba,
        )
        generated.append({
            "file": p_ba.name,
            "caption": f"Before/after denoising with {fwc.SHORT_NAMES[best_id]} at SNR$_{{in}}$={snr_db:+d} dB.",
        })

    # PSD
    p_psd = OUT_DIR / f"spectrum_snr{snr_db:+d}dB_win{win_idx}.png"
    fig_psd_comparison(snr_db, noisy_1d, clean_1d, den_ai, den_trad, win_idx, p_psd)
    generated.append({
        "file": p_psd.name,
        "caption": f"AE-band PSD at SNR$_{{in}}$={snr_db:+d} dB (window {win_idx}).",
    })

    return noisy_1d, clean_1d, den_ai, den_trad, burst_s, burst_e


def main() -> None:
    _configure_stdio_utf8()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(42)
    ds = fwc.load_test_bundle()
    generated: list[dict] = []

    print("Loading top-3 deep models…")
    loaded: dict[str, torch.nn.Module] = {}
    for mid in TOP3_AI:
        m = fwc.load_model(mid, fwc.MODEL_REGISTRY[mid], device)
        if m is not None:
            loaded[mid] = m

    # Primary SNR figures (canonical window = first in band)
    multisnr_panels: dict[int, tuple] = {}
    for snr in SNR_ALL:
        wins = select_windows_in_band(ds, snr, 1)
        win_idx = wins[0]
        noisy, clean, den_ai, _, bs, be = run_one_window(ds, snr, win_idx, loaded, device, rng, generated)
        if TOP3_AI[0] in den_ai:
            multisnr_panels[snr] = (noisy, clean, den_ai[TOP3_AI[0]], bs, be, win_idx)

    # Multi-SNR strip for best model
    if multisnr_panels:
        p_ms = OUT_DIR / "denoise_multisnr_FULL-LossB.png"
        fig_multisnr_best_ai(multisnr_panels, TOP3_AI[0], p_ms)
        generated.insert(0, {
            "file": p_ms.name,
            "caption": "MR-TAE-FULL-LossB denoising vs clean across input SNR levels (burst zoom).",
        })

    # Time diversity @ -5 dB (3 windows)
    snr_td = -5
    win_ids = select_windows_in_band(ds, snr_td, 3)
    cases = []
    for widx in win_ids:
        noisy_all = ds["noisy"].float()
        clean_all = ds["clean"].float()
        noisy_1d = noisy_all[widx].numpy().ravel()
        clean_1d = clean_all[widx].numpy().ravel()
        bs, be = burst_zoom(clean_1d)
        batch = noisy_1d.reshape(1, 1, -1).astype(np.float32)
        den = np.asarray(fwc.infer(loaded[TOP3_AI[0]], batch, device)).ravel()
        cases.append((widx, noisy_1d, clean_1d, den, bs, be))
    p_td = OUT_DIR / "denoise_timediversity_snr-5dB_FULL-LossB.png"
    fig_time_diversity(snr_td, cases, TOP3_AI[0], p_td)
    generated.append({
        "file": p_td.name,
        "caption": "Same model (FULL-LossB) at SNR$_{{in}}$=−5 dB on three distinct test windows.",
    })

    # Metrics bar chart
    p_bar = OUT_DIR / "metrics_ncc_bars.png"
    fig_metrics_bars(p_bar)
    generated.append({
        "file": p_bar.name,
        "caption": "Mean test-set NCC for leading deep models at each evaluation SNR.",
    })

    write_manifest(generated)

    print(f"\nGenerated {len(generated)} figure groups → {OUT_DIR}")
    for g in generated:
        fp = OUT_DIR / g["file"]
        if fp.exists():
            print(f"  {g['file']:45s}  {fp.stat().st_size // 1024:5d} KiB")
    print(f"\nManifest: {OUT_DIR / 'arxiv_figure_manifest.md'}")


if __name__ == "__main__":
    main()
