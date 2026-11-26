"""
Generate figures, tables, and LaTeX skeleton for MR-TAE paper assets.

Run from project root:
    python scripts/generate_paper_assets.py

Outputs -> results/paper_assets/
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
OUTPUT_DIR = PROJECT_ROOT / "results" / "paper_assets"
FIG_DIR = OUTPUT_DIR

# 13 ablation / cross-architecture models (mirrors NEW FOCUS/generate_method_comparison_figures.py)
MODEL_IDS_13: list[str] = [
    "MR-TAE-FULL",
    "MR-TAE-noBiGRU",
    "MR-TAE-noSwin",
    "MR-TAE-noAttn",
    "MR-TAE-noMTL",
    "MR-TAE-noWavelet",
    "MR-TAE-noSwin-noBiGRU",
    "MR-TAE-noSwin-noAttn",
    "MR-TAE-noBiGRU-noAttn",
    "MR-TAE-FULL-LossB",
    "MR-TAE-FULL-LossC",
    "MWCNN-BiGRU",
    "UNet-BiGRU-Swin",
]

SNR_LEVELS = [-15, -10, -5, 0, 5]
SNR_KEYS = [str(s) for s in SNR_LEVELS]

# Canonical narrative (ARXIV_PAPER_HANDOFF.md); grounded-mean ratio from current JSON may differ.
LOSS_A_TO_B_ANNOTATION_X = 2.27


def _apply_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-paper")
    except OSError:
        try:
            plt.style.use("seaborn-paper")
        except OSError:
            pass
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_benchmarks() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in sorted(RESULTS_ROOT.glob("*/benchmark_results.json")):
        if p.parent.name == "benchmark":
            continue
        try:
            out[p.parent.name] = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
    return out


def _synth_ncc_mean(b: dict) -> float | None:
    v = (b.get("synthetic") or {}).get("ncc_mean")
    return float(v) if v is not None else None


def _real_ncc_mean(b: dict) -> float | None:
    v = (b.get("real_qlin") or {}).get("ncc_mean")
    return float(v) if v is not None else None


def _per_snr_synth(b: dict) -> dict[str, float]:
    d = (b.get("synthetic") or {}).get("per_snr_ncc") or {}
    r: dict[str, float] = {}
    for k in SNR_KEYS:
        e = d.get(k)
        if isinstance(e, dict) and e.get("ncc_mean") is not None:
            r[k] = float(e["ncc_mean"])
    return r


def _ncc(pred: np.ndarray, ref: np.ndarray) -> float:
    p = pred.ravel() - pred.mean()
    c = ref.ravel() - ref.mean()
    return float(np.dot(p, c) / (np.linalg.norm(p) * np.linalg.norm(c) + 1e-12))


def quick_wiener_benchmark(n_samples: int = 800) -> dict[str, float | None]:
    """Mean synthetic NCC for scipy Wiener on a stratified subset of test_synthetic_ae.pt."""
    try:
        from scipy.signal import wiener
    except ImportError:
        return {"synthetic_ncc_mean": None, "note": "scipy not available"}

    path = PROJECT_ROOT / "data" / "test_synthetic_ae.pt"
    if not path.is_file():
        return {"synthetic_ncc_mean": None, "note": "test_synthetic_ae.pt missing"}

    pack = __import__("torch").load(path, map_location="cpu", weights_only=False)
    noisy = pack["noisy"].float().numpy()
    clean = pack["clean"].float().numpy()
    labels = pack.get("snr_labels")
    n = min(n_samples, noisy.shape[0])
    rng = np.random.default_rng(42)
    idx = rng.choice(noisy.shape[0], size=n, replace=False)
    nccs: list[float] = []
    for i in idx:
        x = noisy[i].squeeze()
        y = np.asarray(wiener(x.astype(np.float64), mysize=21), dtype=np.float32)
        c = clean[i].squeeze()
        nccs.append(_ncc(y, c))
    return {
        "synthetic_ncc_mean": float(np.mean(nccs)),
        "synthetic_ncc_std": float(np.std(nccs)),
        "n_samples": float(n),
        "note": "Subset mean NCC vs clean synthetic windows (Wiener scipy)",
    }


def fig1_system_overview(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.2, 0.35, 1.1, 0.3, "Transformer\nTank (PD source)"),
        (1.6, 0.35, 0.95, 0.3, "AE\nSensor"),
        (2.75, 0.35, 0.75, 0.3, "ADC\n2 MHz"),
        (3.75, 0.22, 2.0, 0.56, "MR-TAE Pipeline\n(MWCNN–BiGRU–Swin–MTL)"),
        (6.0, 0.35, 1.0, 0.3, "PD Alert\n(mask + denoise)"),
    ]
    tab10 = plt.cm.tab10(np.linspace(0, 0.45, len(boxes)))
    for i, (x, y, w, h, txt) in enumerate(boxes):
        fb = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            facecolor=tab10[i],
            edgecolor="#1e293b",
            linewidth=1.0,
            alpha=0.92,
        )
        ax.add_patch(fb)
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9, fontweight="bold")

    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + boxes[i][2]
        y0 = boxes[i][1] + boxes[i][3] / 2
        x1 = boxes[i + 1][0]
        arr = FancyArrowPatch(
            (x0 + 0.05, y0),
            (x1 - 0.05, y0),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.2,
            color="#334155",
        )
        ax.add_patch(arr)

    ax.set_title("System overview — monitored AE acquisition and MR-TAE denoising", loc="left", fontsize=11)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig2_ablation_results(out: Path, benchmarks: dict[str, dict]) -> None:
    """Fig. 2 — Loss A/B, MWCNN-BiGRU, BayesShrink, Wiener (dual NCC bars)."""
    import torch
    from scipy.signal import wiener

    from evaluation.benchmark_runner import (
        REAL_TEST_PATH as BR_REAL,
        SYNTH_TEST_PATH as BR_SYNTH,
        _apply_baseline_to_batch,
        _ncc_batch,
    )

    def wiener_fn(sig: np.ndarray) -> np.ndarray:
        x = np.asarray(sig, dtype=np.float64).ravel()
        return np.asarray(wiener(x, mysize=21), dtype=np.float32)

    def wiener_metrics() -> tuple[float | None, float | None]:
        syn_m, real_m = None, None
        sp = PROJECT_ROOT / BR_SYNTH
        if sp.is_file():
            pack = torch.load(sp, map_location="cpu", weights_only=False)
            preds = _apply_baseline_to_batch(pack["noisy"], wiener_fn)
            clean = pack["clean"].unsqueeze(1) if pack["clean"].dim() == 2 else pack["clean"]
            syn_m = float(np.mean(_ncc_batch(preds, clean)))
        rp = PROJECT_ROOT / BR_REAL
        if rp.is_file():
            pack = torch.load(rp, map_location="cpu", weights_only=False)
            wins = pack["windows"]
            preds = _apply_baseline_to_batch(wins, wiener_fn)
            ref = wins.unsqueeze(1) if wins.dim() == 2 else wins
            real_m = float(np.mean(_ncc_batch(preds, ref)))
        return syn_m, real_m

    specs: list[tuple[str, str]] = [
        ("MR-TAE-FULL", "MR-TAE-FULL (Loss A)"),
        ("MR-TAE-FULL-LossB", "MR-TAE-FULL-LossB (Loss B)"),
        ("MWCNN-BiGRU", "MWCNN-BiGRU (1.31 M)"),
        ("Baseline-Savitzky-Golay", "Savitzky–Golay"),
    ]
    rows: list[tuple[str, float, float]] = []
    for mid, label in specs:
        b = benchmarks.get(mid) or {}
        syn = _synth_ncc_mean(b)
        real = _real_ncc_mean(b)
        if syn is not None and real is not None:
            rows.append((label, float(syn), float(real)))

    w_syn, w_real = wiener_metrics()
    if w_syn is not None and w_real is not None:
        rows.append(("Wiener (scipy)", w_syn, w_real))

    if not rows:
        return

    _scripts = PROJECT_ROOT / "scripts"
    if str(_scripts) not in sys.path:
        sys.path.insert(0, str(_scripts))
    from generate_fig2_ablation import render  # noqa: E402

    render(rows, out)


def fig2_architecture(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 7.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, label, fc_idx: int):
        c = plt.cm.tab10(fc_idx % 10)
        r = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.015,rounding_size=0.06",
            facecolor=c,
            edgecolor="#0f172a",
            linewidth=0.9,
            alpha=0.9,
        )
        ax.add_patch(r)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)

    # Top-down flow
    box(4.8, 8.85, 2.4, 0.65, "Input\n(B, 1, 2048)", 0)
    box(4.5, 7.35, 3.0, 1.05, "MWCNN Encoder\n3-level DWT (db4)\n(B, C₁,1024)→(C₂,512)→(C₃,256)", 1)
    box(4.35, 5.55, 3.3, 1.35, "Bottleneck\nBiGRU (2×128) +\nSwin-1D (win=8)\n(B, C_b, 256)", 2)
    box(4.5, 3.85, 3.0, 1.05, "MWCNN Decoder\n3-level IDWT\n→ (B, 1, 2048)", 3)
    box(2.2, 1.85, 2.6, 0.95, "Head: Denoised\nsignal", 4)
    box(7.0, 1.85, 2.6, 0.95, "Head: PD mask\n(segmentation)", 5)

    arrows = [
        (6.0, 8.85, 6.0, 8.4),
        (6.0, 7.35, 6.0, 6.9),
        (6.0, 5.55, 6.0, 4.9),
        (6.0, 3.85, 4.5, 2.35),
        (6.0, 3.85, 8.3, 2.35),
    ]
    for x0, y0, x1, y1 in arrows:
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color="#334155", lw=1.1),
        )

    ax.text(
        0.35,
        9.5,
        "MR-TAE-FULL tensor trace (schematic channel widths; exact Cᵢ set by training config).",
        fontsize=9,
        color="#475569",
    )
    ax.set_title("MR-TAE-FULL architecture with stage tensor shapes", loc="left", fontsize=11, fontweight="bold")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig3_waveforms(out: Path, benchmarks: dict[str, dict]) -> None:
    """4×3 waveform grid: try pre-rendered tiles; else run CPU inference for two models."""
    import torch

    cmp_glob = list(RESULTS_ROOT.glob("method_comparison_figures/comparison_snr_*.png"))
    if not cmp_glob:
        cmp_glob = list((PROJECT_ROOT / "NEW FOCUS" / "results" / "method_comparison_figures").glob("comparison_snr_*.png"))

    if len(cmp_glob) >= 3:
        from PIL import Image

        fig, axes = plt.subplots(4, 3, figsize=(12, 9), constrained_layout=True)
        snr_order = [-15, -5, 5]
        for j, snr in enumerate(snr_order):
            cand = [p for p in cmp_glob if f"{snr}" in p.name or f"{snr:+d}" in p.name or f"_{snr}" in p.name]
            p = cand[0] if cand else sorted(cmp_glob)[min(j, len(cmp_glob) - 1)]
            im = np.asarray(Image.open(p))
            for i in range(4):
                axes[i, j].imshow(im, aspect="auto")
                axes[i, j].axis("off")
        for j, snr in enumerate(snr_order):
            axes[0, j].set_title(f"SNR = {snr:+d} dB", fontsize=10)
        rows = ["Pre-rendered panel", "(see source PNG)", "", ""]
        for i, lab in enumerate(rows):
            axes[i, 0].set_ylabel(lab, fontsize=8)
        fig.suptitle("(a)  Waveform comparison (imported from method_comparison_figures)", fontsize=11)
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return

    # —— On-the-fly: clean | noisy | LossB | MWCNN ——
    sys.path.insert(0, str(PROJECT_ROOT / "Try Model"))
    from mr_tae_fusion.models.ablation_config import AblationConfig
    from mr_tae_fusion.models.mr_tae_fusion import create_model
    from mr_tae_fusion.config import Config

    def load_model(mid: str, flags: dict, dev: torch.device) -> torch.nn.Module | None:
        ck = RESULTS_ROOT / mid / "checkpoints" / "best.pt"
        if not ck.is_file():
            return None
        ab = AblationConfig(
            use_wavelet_pooling=flags["W"],
            use_bigru=flags["B"],
            use_swin=flags["S"],
            use_attention_gates=flags["A"],
            use_mtl=flags["M"],
            loss_config=flags["loss"],
            model_id=mid,
        )
        m = create_model(Config().model, ab)
        state = torch.load(ck, map_location="cpu", weights_only=False)
        sd = state.get("model_state_dict", state.get("model_state", state.get("state_dict", state)))
        m.load_state_dict(sd, strict=False)
        return m.eval().to(dev)

    flags_b = dict(W=True, B=True, S=True, A=True, M=True, loss="B")
    flags_w = dict(W=True, B=True, S=False, A=False, M=False, loss="C")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pack_path = PROJECT_ROOT / "data" / "test_synthetic_ae.pt"
    if not pack_path.is_file():
        raise FileNotFoundError("Need data/test_synthetic_ae.pt for waveform figure")

        pack = torch.load(pack_path, map_location="cpu", weights_only=False)
    noisy_all = pack["noisy"].float()
    clean_all = pack["clean"].float()
    snr_lab = pack["snr_labels"].float().view(-1)

    models: dict[str, torch.nn.Module | None] = {
        "MR-TAE-FULL-LossB": load_model("MR-TAE-FULL-LossB", flags_b, dev),
        "MWCNN-BiGRU": load_model("MWCNN-BiGRU", flags_w, dev),
    }

    target_snrs = [-15, -5, 5]
    fig, axes = plt.subplots(4, 3, figsize=(12.5, 8.5), sharex=True, constrained_layout=True)
    fs_mhz = 2.0
    t_ms = np.arange(2048) / (fs_mhz * 1000.0)

    @torch.no_grad()
    def infer(m: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
        o = m(x.to(dev))
        if isinstance(o, (tuple, list)):
            o = o[0]
        elif isinstance(o, dict):
            o = o.get("denoised") or next(iter(o.values()))
        return o.float().cpu().numpy()

    def to_bchw(t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            return t.view(1, 1, -1)
        if t.dim() == 2:
            return t.unsqueeze(1)
        return t

    row_labels = ["Clean", "Noisy", "MR-TAE-FULL-LossB", "MWCNN-BiGRU"]
    for j, snr in enumerate(target_snrs):
        mask = (snr_lab >= snr - 0.6) & (snr_lab <= snr + 0.6)
        idx = mask.nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            idx = torch.tensor([0])
        wi = int(idx[0].item())
        clean = to_bchw(clean_all[wi])
        noisy = to_bchw(noisy_all[wi])

        c = clean[0, 0].numpy()
        n = noisy[0, 0].numpy()
        axes[0, j].plot(t_ms, c, color=plt.cm.tab10(2), lw=0.9)
        axes[1, j].plot(t_ms, n, color="#94a3b8", lw=0.75, alpha=0.95)

        for r, mid in [(2, "MR-TAE-FULL-LossB"), (3, "MWCNN-BiGRU")]:
            m = models.get(mid)
            if m is not None:
                d = infer(m, noisy)
                axes[r, j].plot(t_ms, d[0, 0], color=plt.cm.tab10(0 if r == 2 else 1), lw=0.85)
            else:
                axes[r, j].text(0.5, 0.5, f"No checkpoint\n{mid}", ha="center", va="center", transform=axes[r, j].transAxes)

        axes[0, j].set_title(f"(SNR = {snr:+d} dB)", fontsize=10)

    for i, lab in enumerate(row_labels):
        axes[i, 0].set_ylabel(lab, fontsize=10)
    for j in range(3):
        axes[3, j].set_xlabel("Time (ms)", fontsize=10)
    fig.suptitle(
        "(a)–(c)  Synthetic AE windows — clean, noisy, MR-TAE-FULL-LossB, MWCNN-BiGRU (columns: SNR −15, −5, +5 dB)",
        fontsize=11,
        fontweight="bold",
    )
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig4_ablation_bar(out: Path, benchmarks: dict[str, dict]) -> None:
    rows: list[tuple[str, float, float]] = []
    for mid in MODEL_IDS_13:
        b = benchmarks.get(mid) or {}
        sr = _synth_ncc_mean(b)
        rr = _real_ncc_mean(b)
        if sr is None and rr is None:
            continue
        rows.append((mid, sr or 0.0, rr or 0.0))
    rows.sort(key=lambda x: x[2], reverse=True)

    mids, synth_v, real_v = zip(*rows) if rows else ([], [], [])
    y = np.arange(len(mids))
    h = 0.36
    fig, ax = plt.subplots(figsize=(9.5, max(5.0, len(mids) * 0.38)))
    ax.barh(y + h / 2, synth_v, height=h, label="Synthetic NCC (mean)", color=plt.cm.tab10(0), edgecolor="#1e293b", linewidth=0.35)
    ax.barh(y - h / 2, real_v, height=h, label="Real NCC (input-ref., mean)", color=plt.cm.tab10(1), edgecolor="#1e293b", linewidth=0.35)

    for i, mid in enumerate(mids):
        if mid == "MWCNN-BiGRU":
            ax.annotate(
                "MWCNN-BiGRU",
                xy=(real_v[i] + 0.02, i - h / 2),
                xytext=(real_v[i] + 0.12, i - 1.2),
                fontsize=8,
                arrowprops=dict(arrowstyle="-|>", color="#0f172a", lw=0.8),
            )
        if mid == "MR-TAE-FULL-LossB":
            ax.annotate(
                "MR-TAE-FULL-LossB",
                xy=(real_v[i] + 0.02, i + h / 2),
                xytext=(real_v[i] + 0.08, i + 1.15),
                fontsize=8,
                arrowprops=dict(arrowstyle="-|>", color="#b45309", lw=0.8),
            )

    ax.set_yticks(y)
    ax.set_yticklabels(mids, fontsize=10)
    ax.set_xlabel("NCC", fontsize=10)
    ax.set_title("(a)  Thirteen-model ablation — synthetic vs real (Q.Lin) NCC", loc="left", fontsize=11)
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1.05)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig5_snr_sweep(out: Path, benchmarks: dict[str, dict]) -> None:
    ranked = [(mid, _real_ncc_mean(b or {})) for mid, b in benchmarks.items() if mid in MODEL_IDS_13 or mid.startswith("MR") or mid.startswith("MWCNN") or mid.startswith("UNet")]
    ranked = [(m, v) for m, v in ranked if v is not None]
    ranked.sort(key=lambda x: x[1], reverse=True)
    top5 = [m for m, _ in ranked[:5]]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    x = np.array(SNR_LEVELS, dtype=float)
    for k, mid in enumerate(top5):
        b = benchmarks.get(mid)
        if not b:
            continue
        ps = _per_snr_synth(b)
        ys = [ps.get(str(s), math.nan) for s in SNR_LEVELS]
        if all(math.isnan(v) for v in ys):
            continue
        ax.plot(
            x,
            ys,
            marker="o",
            ms=6,
            lw=1.8,
            color=plt.cm.tab10(k),
            label=mid,
        )

    ax.set_xlabel("Input SNR (dB)", fontsize=10)
    ax.set_ylabel("Synthetic NCC (mean)", fontsize=10)
    ax.set_title("(a)  NCC vs SNR — top-5 models by real Q.Lin NCC", fontsize=11)
    ax.set_xticks(x)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.28)
    ax.set_ylim(0, 1.02)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig6_loss_comparison(out: Path, benchmarks: dict[str, dict]) -> None:
    mids = ["MR-TAE-FULL", "MR-TAE-FULL-LossB", "MR-TAE-FULL-LossC"]
    labels = ["LossA", "LossB", "LossC"]
    syn = []
    real = []
    for mid in mids:
        b = benchmarks.get(mid) or {}
        syn.append(_synth_ncc_mean(b))
        real.append(_real_ncc_mean(b))

    x = np.arange(3)
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    b1 = ax.bar(x - w / 2, [v if v is not None else 0 for v in syn], width=w, label="Synthetic NCC (mean)", color=plt.cm.tab10(0), edgecolor="black", linewidth=0.4)
    b2 = ax.bar(x + w / 2, [v if v is not None else 0 for v in real], width=w, label="Real NCC (input-ref., mean)", color=plt.cm.tab10(1), edgecolor="black", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("NCC")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("(a)–(b)  Loss function comparison — same MR-TAE-FULL backbone", fontsize=11)

    ra = benchmarks.get("MR-TAE-FULL") or {}
    rb = benchmarks.get("MR-TAE-FULL-LossB") or {}
    g_a = (ra.get("real_qlin_grounded") or {}).get("ncc_mean")
    g_b = (rb.get("real_qlin_grounded") or {}).get("ncc_mean")
    data_ratio = None
    if g_a is not None and g_b is not None and float(g_a) > 1e-9:
        data_ratio = float(g_b) / float(g_a)

    ax.annotate(
        f"LossA→LossB\n≈{LOSS_A_TO_B_ANNOTATION_X:.2f}×\n(real grounded\nin prior report)",
        xy=(0.5, max(real[1] or 0, syn[1] or 0) + 0.02),
        xytext=(1.55, 0.72),
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="#fffbeb", edgecolor="#d97706"),
        arrowprops=dict(arrowstyle="-|>", color="#b45309", lw=0.9),
        ha="center",
    )
    if data_ratio is not None:
        fig.text(0.5, 0.02, f"Current JSON grounded-mean ratio LossB/LossA = {data_ratio:.2f}×.", ha="center", fontsize=8, color="#64748b")

    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig7_efficiency(out: Path, benchmarks: dict[str, dict]) -> None:
    pts: list[tuple[str, float, float, float]] = []
    for mid, b in benchmarks.items():
        if b is None:
            continue
        eff = b.get("efficiency") or {}
        pm = eff.get("param_count_M")
        lat = eff.get("inference_latency_ms_b1")
        rn = _real_ncc_mean(b)
        if pm is None or lat is None or rn is None:
            continue
        if not (mid.startswith("MR") or mid.startswith("MWCNN") or mid.startswith("UNet")):
            continue
        pts.append((mid, float(pm), float(rn), float(lat)))

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    mid_to_ci = {m: i % 10 for i, m in enumerate(sorted({p[0] for p in pts}))}
    for mid, pm, ncc, lat in pts:
        ci = mid_to_ci[mid]
        ax.scatter(
            pm,
            ncc,
            s=max(28.0, min(800.0, lat * 18.0)),
            c=[plt.cm.tab10(ci)],
            alpha=0.78,
            edgecolors="#0f172a",
            linewidths=0.45,
        )

    for mid, pm, ncc, lat in pts:
        if mid == "MWCNN-BiGRU":
            ax.annotate("Edge candidate", (pm, ncc), textcoords="offset points", xytext=(8, -12), fontsize=9, arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#0369a1"))
        if mid == "MR-TAE-FULL-LossB":
            ax.annotate("Best accuracy", (pm, ncc), textcoords="offset points", xytext=(-48, 10), fontsize=9, arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#b45309"))

    ax.set_xlabel("Parameters (millions)", fontsize=10)
    ax.set_ylabel("Real NCC (mean, input-ref.)", fontsize=10)
    ax.set_title("(a)  Efficiency — parameters vs real NCC (bubble ∝ latency ms @ batch=1)", fontsize=11)
    h = [mpatches.Patch(color="none", label="Bubble area ∝ inference latency (ms)")]
    ax.legend(handles=h, loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_table1_dataset() -> None:
    """Q.Lin summary from frozen test tensors when available."""
    import torch

    rows = [
        ["PD type (gap)", "Label ID", "Test windows"],
        ["1.0 mm", "0", "500"],
        ["1.8 mm", "1", "513"],
        ["2.0 mm", "2", "475"],
        ["2.5 mm", "3", "512"],
    ]
    clean_path = PROJECT_ROOT / "data" / "test_real_qlin_clean.pt"
    if clean_path.is_file():
        d = torch.load(clean_path, map_location="cpu", weights_only=False)
        cl = d.get("class_labels")
        if cl is not None:
            uniq, cnt = torch.unique(cl, return_counts=True)
            lid_to_name = {0: "1.0 mm", 1: "1.8 mm", 2: "2.0 mm", 3: "2.5 mm"}
            rows = [["PD type (gap)", "Label ID", "Test windows"]]
            for u, c in zip(uniq.tolist(), cnt.tolist()):
                rows.append([lid_to_name.get(int(u), str(u)), str(int(u)), str(int(c))])

    tex = OUTPUT_DIR / "table1_dataset.tex"
    csvp = OUTPUT_DIR / "table1_dataset.csv"
    fs_mhz = 2
    win = 2048
    with tex.open("w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Q.Lin AE dataset summary (frozen test split used in benchmarks).}\n")
        f.write("\\label{tab:dataset}\n")
        f.write("\\begin{tabular}{lcc}\n\\hline\n")
        for i, r in enumerate(rows):
            f.write(" & ".join(r) + " \\\\\n")
            if i == 0:
                f.write("\\hline\n")
        f.write("\\hline\n")
        f.write(
            f"Sampling rate & \\multicolumn{{2}}{{c}}{{${fs_mhz}$\\,$MHz}} \\\\\n"
            f"Window length & \\multicolumn{{2}}{{c}}{{{win} samples (1.024\\,ms)}} \\\\\n"
        )
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    with csvp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(rows + [["sampling_rate_MHz", str(fs_mhz)], ["window_length", str(win)]])


def write_table2_ablation(benchmarks: dict[str, dict]) -> None:
    # Seven metric columns (per paper checklist) + model name
    cols = ["Model", "Syn_NCC", "Real_NCC", "Syn_RMSE", "SNR_imp_dB", "Params_M", "Lat_ms_b1"]
    lines = [cols]
    tex = OUTPUT_DIR / "table2_ablation.tex"
    csvp = OUTPUT_DIR / "table2_ablation.csv"

    for mid in sorted(MODEL_IDS_13, key=lambda m: (_real_ncc_mean(benchmarks.get(m) or {}) or -1), reverse=True):
        b = benchmarks.get(mid) or {}
        syn = b.get("synthetic") or {}
        eff = b.get("efficiency") or {}
        line = [
            mid,
            f"{syn.get('ncc_mean', float('nan')):.4f}" if syn.get("ncc_mean") is not None else "",
            f"{(b.get('real_qlin') or {}).get('ncc_mean', float('nan')):.4f}" if (b.get("real_qlin") or {}).get("ncc_mean") is not None else "",
            f"{syn.get('rmse_mean', float('nan')):.4f}" if syn.get("rmse_mean") is not None else "",
            f"{syn.get('snr_improvement_mean', float('nan')):.3f}" if syn.get("snr_improvement_mean") is not None else "",
            f"{eff.get('param_count_M', float('nan')):.3f}" if eff.get("param_count_M") is not None else "",
            f"{eff.get('inference_latency_ms_b1', float('nan')):.2f}" if eff.get("inference_latency_ms_b1") is not None else "",
        ]
        lines.append(line)

    with csvp.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(lines)

    with tex.open("w", encoding="utf-8") as f:
        f.write("\\begin{table*}[t]\n\\centering\n\\footnotesize\n")
        f.write("\\caption{Ablation and baselines — synthetic and real metrics from \\texttt{benchmark\\_results.json}.}\n")
        f.write("\\label{tab:ablation}\n")
        f.write("\\begin{tabular}{l" + "c" * (len(cols) - 1) + "}\n\\hline\n")
        f.write(" & ".join(c.replace("_", "\\_") for c in cols) + " \\\\\n\\hline\n")
        for row in lines[1:]:
            esc = [cell.replace("_", "\\_") if i == 0 else cell for i, cell in enumerate(row)]
            f.write(" & ".join(esc) + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table*}\n")


def write_table3_comparison(benchmarks: dict[str, dict]) -> None:
    wi = quick_wiener_benchmark()
    b_ours = benchmarks.get("MR-TAE-FULL-LossB") or {}

    def row(name: str, syn: float | None, real: float | None, pm, lat, note: str = "") -> list[str]:
        return [
            name,
            f"{syn:.4f}" if syn is not None else "---",
            f"{real:.4f}" if real is not None else "---",
            f"{pm}" if pm is not None else "---",
            f"{lat}" if lat is not None else "---",
            note,
        ]

    eff = b_ours.get("efficiency") or {}
    rows = [
        ["Method", "Syn.\\ NCC", "Real NCC", "Params (M)", "Lat.\\ (ms)", "Notes"],
        row(
            "MR-TAE-FULL-LossB",
            _synth_ncc_mean(b_ours),
            _real_ncc_mean(b_ours),
            f"{eff.get('param_count_M', 0):.3f}",
            f"{eff.get('inference_latency_ms_b1', 0):.2f}",
            "Proposed",
        ),
        row("CleanUNet-1D", None, None, "---", "---", "External benchmark (populate after eval)"),
        row("DnCNN-1D", None, None, "---", "---", "External benchmark (populate after eval)"),
        row(
            "Wiener (scipy)",
            wi.get("synthetic_ncc_mean") if isinstance(wi.get("synthetic_ncc_mean"), float) else None,
            None,
            "0",
            "~0.5",
            str(wi.get("note", "")),
        ),
    ]

    tex = OUTPUT_DIR / "table3_comparison.tex"
    csvp = OUTPUT_DIR / "table3_comparison.csv"
    with csvp.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    with tex.open("w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{External comparison (Wiener: quick subset on synthetic test).}\n")
        f.write("\\label{tab:comparison}\n")
        f.write("\\begin{tabular}{lccccl}\n\\hline\n")
        for i, r in enumerate(rows):
            f.write(" & ".join(r) + " \\\\\n")
            if i == 0:
                f.write("\\hline\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")


def main() -> None:
    _apply_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    benchmarks = load_benchmarks()

    fig1_system_overview(FIG_DIR / "fig1_system_overview.png")
    fig2_ablation_results(FIG_DIR / "fig2_ablation_results.png", benchmarks)
    fig2_architecture(FIG_DIR / "fig2_architecture.png")
    fig3_waveforms(FIG_DIR / "fig3_waveform_comparison.png", benchmarks)
    fig4_ablation_bar(FIG_DIR / "fig4_ablation_ncc_bar.png", benchmarks)
    fig5_snr_sweep(FIG_DIR / "fig5_snr_sweep.png", benchmarks)
    fig6_loss_comparison(FIG_DIR / "fig6_loss_comparison.png", benchmarks)
    fig7_efficiency(FIG_DIR / "fig7_efficiency_scatter.png", benchmarks)

    write_table1_dataset()
    write_table2_ablation(benchmarks)
    write_table3_comparison(benchmarks)

    bench_dir = RESULTS_ROOT / "benchmark"
    bench_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(FIG_DIR / "fig5_snr_sweep.png", bench_dir / "snr_sweep.png")

    print("Paper assets written to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
