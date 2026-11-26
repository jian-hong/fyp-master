"""
Regenerate paper figures from results/definitive_benchmark/ (dpi=300, white background).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ASSETS = PROJECT_ROOT / "results" / "paper_assets_v2"
DEF_DIR = PROJECT_ROOT / "results" / "definitive_benchmark"
SNR_LEVELS = (-15, -10, -5, 0, 5)
LOSS_IDS = ("MR-TAE-FULL", "MR-TAE-FULL-LossB", "MR-TAE-FULL-LossC")
DPI = 300


def _save(fig: plt.Figure, name: str) -> None:
    OUT_ASSETS.mkdir(parents=True, exist_ok=True)
    path = OUT_ASSETS / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {path}")


def _load() -> Tuple[Dict, Dict, Dict, Dict]:
    synth = json.loads((DEF_DIR / "synthetic_results.json").read_text(encoding="utf-8"))
    real = json.loads((DEF_DIR / "real_results.json").read_text(encoding="utf-8"))
    lat = json.loads((DEF_DIR / "latency.json").read_text(encoding="utf-8"))
    baselines = {}
    bp = DEF_DIR / "baselines.json"
    if bp.exists():
        baselines = json.loads(bp.read_text(encoding="utf-8"))
    return synth, real, lat, baselines


def fig4_ablation_ncc_bar(synth: Dict, real: Dict) -> None:
    """Grouped bars: synthetic NCC (GT) vs real input-ref NCC — separate metrics, labeled."""
    ids = [k for k in synth if k.startswith("MR-TAE") or k in ("MWCNN-BiGRU", "UNet-BiGRU-Swin")]
    ids = sorted(ids, key=lambda m: -(synth.get(m, {}).get("ncc_mean") or 0))
    x = np.arange(len(ids))
    w = 0.38
    syn_v = [float(synth[m].get("ncc_mean") or 0) for m in ids]
    real_v = []
    for m in ids:
        r = real.get(m, {})
        ref = r.get("input_ref", r)
        real_v.append(float(ref.get("ncc_mean") or 0))

    fig, ax = plt.subplots(figsize=(14, 6), facecolor="white")
    ax.bar(x - w / 2, syn_v, w, label="Synthetic NCC (vs clean GT)", color="#2563eb")
    ax.bar(x + w / 2, real_v, w, label="Real Input-Ref NCC (vs noisy input)", color="#ea580c")
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("NCC")
    ax.set_title("Ablation: Synthetic GT-NCC vs Real Input-Ref NCC")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.text(
        0.5,
        0.01,
        "Real Input-Ref NCC measures output-to-input similarity, not denoising quality.",
        ha="center",
        fontsize=8,
        style="italic",
    )
    _save(fig, "fig4_ablation_ncc_bar.png")


def fig5_snr_sweep(synth: Dict) -> None:
    ranked = sorted(
        synth.keys(),
        key=lambda m: float(synth[m].get("ncc_mean") or 0),
        reverse=True,
    )[:5]
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    for mid in ranked:
        per = synth[mid].get("per_snr_ncc") or {}
        ys = [(per.get(str(s)) or {}).get("ncc_mean") for s in SNR_LEVELS]
        ax.plot(SNR_LEVELS, ys, marker="o", label=mid)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("NCC vs clean GT (synthetic)")
    ax.set_title("Top-5 models: synthetic NCC vs SNR")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    _save(fig, "fig5_snr_sweep.png")


def fig6_loss_comparison(synth: Dict, real: Dict) -> None:
    ids = [m for m in LOSS_IDS if m in synth]
    x = np.arange(len(ids))
    w = 0.35
    syn_v = [float(synth[m].get("ncc_mean") or 0) for m in ids]
    real_v = []
    for m in ids:
        r = real.get(m, {})
        ref = r.get("input_ref", r)
        real_v.append(float(ref.get("ncc_mean") or 0))

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    ax.bar(x - w / 2, syn_v, w, label="Synthetic NCC (vs clean GT)", color="#2563eb")
    ax.bar(x + w / 2, real_v, w, label="Real NCC (vs noisy input — no GT)", color="#ea580c")
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=15, ha="right")
    ax.set_ylabel("NCC")
    ax.set_title("LossA / LossB / LossC comparison")
    ax.legend(fontsize=8)
    fig.text(
        0.5,
        0.01,
        "Real NCC measures output-to-input similarity, not denoising quality.",
        ha="center",
        fontsize=8,
        style="italic",
    )
    _save(fig, "fig6_loss_comparison.png")


def fig7_efficiency_scatter(synth: Dict, lat: Dict) -> None:
    xs, ys, labels, sizes = [], [], [], []
    for mid, s in synth.items():
        if mid not in lat or "lat_ms_b1" not in lat[mid]:
            continue
        sm = s.get("ncc_mean")
        if sm is None:
            continue
        xs.append(float(lat[mid]["lat_ms_b1"]))
        ys.append(float(sm))
        labels.append(mid)
        sizes.append(float(lat[mid].get("params_M", 1)) * 80)
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")
    ax.scatter(xs, ys, s=sizes, alpha=0.7, c="#2563eb", edgecolors="k", linewidths=0.5)
    for i, lb in enumerate(labels):
        ax.annotate(lb, (xs[i], ys[i]), fontsize=6, alpha=0.85)
    ax.set_xlabel("Latency (ms, batch=1, CUDA-sync)")
    ax.set_ylabel("Synthetic NCC (vs clean GT)")
    ax.set_title("Efficiency: latency vs synthetic NCC")
    ax.grid(alpha=0.3)
    _save(fig, "fig7_efficiency_scatter.png")


def fig3_waveform_comparison() -> None:
    """Qualitative waveform panel at 0 dB SNR (synthetic)."""
    import torch

    from evaluation.benchmark_runner import load_model_from_checkpoint
    from evaluation.metrics_verified import ncc_synthetic

    synth_path = PROJECT_ROOT / "data" / "test_synthetic_ae.pt"
    ck = PROJECT_ROOT / "results" / "MR-TAE-FULL" / "checkpoints" / "best.pt"
    if not synth_path.exists() or not ck.exists():
        print("  [fig3] skipped: missing data or checkpoint")
        return
    data = torch.load(synth_path, map_location="cpu", weights_only=False)
    noisy, clean = data["noisy"], data["clean"]
    if noisy.dim() == 3:
        noisy, clean = noisy.squeeze(1), clean.squeeze(1)
    idx = 6000  # ~0 dB block
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model_from_checkpoint(str(ck), "MR-TAE-FULL", device_str=device)
    x = noisy[idx : idx + 1].unsqueeze(1).to(device)
    with torch.no_grad():
        pred, _ = model(x)
    n = noisy[idx].numpy()
    c = clean[idx].numpy()
    p = pred[0, 0].cpu().numpy()
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True, facecolor="white")
    axes[0].plot(n, color="#94a3b8", lw=0.8)
    axes[0].set_title(f"Noisy input  |  NCC vs clean={ncc_synthetic(n, c):.3f}")
    axes[1].plot(p, color="#2563eb", lw=0.8)
    axes[1].set_title(f"MR-TAE-FULL denoised  |  NCC vs clean={ncc_synthetic(p, c):.3f}")
    axes[2].plot(c, color="#16a34a", lw=0.8)
    axes[2].set_title("Clean ground truth")
    axes[2].set_xlabel("Sample")
    fig.suptitle("Waveform comparison @ ~0 dB SNR (synthetic)", fontsize=12)
    fig.tight_layout()
    _save(fig, "fig3_waveform_comparison.png")


def _copy_static_figs() -> None:
    src = PROJECT_ROOT / "results" / "paper_assets"
    for name in ("fig1_system_overview.png", "fig2_architecture.png"):
        for base in (src, PROJECT_ROOT / "docs"):
            p = base / name
            if p.exists():
                shutil.copy2(p, OUT_ASSETS / name)
                print(f"  -> copied {name} from {p.parent}")
                break


def generate_all(def_dir: Path | None = None) -> None:
    global DEF_DIR
    if def_dir is not None:
        DEF_DIR = Path(def_dir)
    OUT_ASSETS.mkdir(parents=True, exist_ok=True)
    synth, real, lat, _ = _load()
    _copy_static_figs()
    fig3_waveform_comparison()
    fig4_ablation_ncc_bar(synth, real)
    fig5_snr_sweep(synth)
    fig6_loss_comparison(synth, real)
    fig7_efficiency_scatter(synth, lat)


if __name__ == "__main__":
    generate_all()
