"""
evaluation/generate_figures.py — Generate all benchmark figures from results JSON files.

Scans results/*/benchmark_results.json and results/*/history.json, then produces
five publication-ready figures saved to results/benchmark/.

Also regenerates docs/RESULTS_COMPARISON.md with a markdown summary table.

Usage
-----
    python evaluation/generate_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    import seaborn as sns
except ImportError:
    sns = None

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_ROOT = PROJECT_ROOT / "results"
BENCHMARK_OUT = RESULTS_ROOT / "benchmark"
REGISTRY_PATH = PROJECT_ROOT / "config" / "model_registry.yaml"
DOCS_RESULTS = PROJECT_ROOT / "docs" / "RESULTS_COMPARISON.md"
SNR_LEVELS: Tuple[int, ...] = (-15, -10, -5, 0, 5)

# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------
BG_COLOR = "#0a0e1a"
GRID_COLOR = "#1e293b"
CYAN = "#00d4ff"
GREEN = "#22c55e"
TEXT_COLOR = "#e2e8f0"
MONO_FONT = "JetBrains Mono"
FALLBACK_MONO = "monospace"

CYAN_CMAP_COLORS = ["#0a0e1a", "#062a3e", "#0c5474", "#1289ab", "#00d4ff"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_mono_font() -> str:
    from matplotlib.font_manager import findSystemFonts
    for f in findSystemFonts():
        if "jetbrains" in f.lower():
            return MONO_FONT
    return FALLBACK_MONO

_MONO_FONT_RESOLVED: Optional[str] = None

def _mono() -> str:
    global _MONO_FONT_RESOLVED
    if _MONO_FONT_RESOLVED is None:
        _MONO_FONT_RESOLVED = _detect_mono_font()
    return _MONO_FONT_RESOLVED


def _apply_dark_style(ax: plt.Axes, fig: plt.Figure) -> None:
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR, which="both")
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.6)


def _make_cyan_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("cyan_dark", CYAN_CMAP_COLORS, N=256)


def load_registry() -> Dict[str, Dict[str, Any]]:
    if not REGISTRY_PATH.exists():
        return {}
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("models", {})


def collect_benchmark_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for jf in sorted(RESULTS_ROOT.glob("*/benchmark_results.json")):
        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        model_id = data.get("model_id", jf.parent.name)
        rows.append({"model_id": model_id, "data": data, "path": jf})
    return rows


def collect_history_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for hf in sorted(RESULTS_ROOT.glob("*/history.json")):
        try:
            with open(hf, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        model_id = hf.parent.name
        rows.append({"model_id": model_id, "data": data})
    return rows


def bottleneck_row_label(use_bigru: bool, use_swin: bool) -> str:
    if use_bigru and use_swin:
        return "BiGRU+Swin"
    if use_bigru and not use_swin:
        return "BiGRU only"
    if not use_bigru and use_swin:
        return "Swin only"
    return "Conv only"


def encoder_col_label(use_wavelet_pooling: bool) -> str:
    return "MWCNN (wavelet)" if use_wavelet_pooling else "Standard Conv"


def registry_ablation_axes(model_id: str, registry: Dict) -> Tuple[str, str]:
    cfg = registry.get(model_id, {})
    row = bottleneck_row_label(cfg.get("use_bigru", True), cfg.get("use_swin", True))
    col = encoder_col_label(cfg.get("use_wavelet_pooling", True))
    return row, col


# ---------------------------------------------------------------------------
# Figure 1 — Grouped bar chart: NCC-Synth vs NCC-Real
# ---------------------------------------------------------------------------

def generate_comparison_bar(rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    mono = _mono()
    models, ncc_s, ncc_r = [], [], []
    for r in rows:
        d = r["data"]
        sm = d.get("synthetic", {}).get("ncc_mean")
        rm = d.get("real_qlin", {}).get("ncc_mean")
        if sm is None and rm is None:
            continue
        models.append(r["model_id"])
        ncc_s.append(sm if sm is not None else 0.0)
        ncc_r.append(rm if rm is not None else 0.0)

    if not models:
        print("[comparison_bar] No models with NCC data — skipped.")
        return

    x = np.arange(len(models))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.2), 5.5))
    _apply_dark_style(ax, fig)

    bars_s = ax.bar(x - w / 2, ncc_s, w, label="NCC-Synth", color=CYAN,
                    edgecolor=BG_COLOR, linewidth=0.6)
    bars_r = ax.bar(x + w / 2, ncc_r, w, label="NCC-Real", color=GREEN,
                    edgecolor=BG_COLOR, linewidth=0.6)

    for bar_group in (bars_s, bars_r):
        for bar in bar_group:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                        f"{h:.3f}", ha="center", va="bottom",
                        fontsize=7, color=TEXT_COLOR, fontfamily=mono)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("NCC (Pearson)", fontsize=10)
    ax.set_title("Model Comparison: NCC-Synth vs NCC-Real", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="lower right", facecolor=BG_COLOR, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"[comparison_bar] Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Figure 2 — Ablation heatmap
# ---------------------------------------------------------------------------

def generate_ablation_heatmap(rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    mono = _mono()
    registry = load_registry()
    grid: Dict[Tuple[str, str], float] = {}

    for r in rows:
        mid = r["model_id"]
        if mid.startswith("Baseline"):
            continue
        if mid not in registry:
            continue
        ncc = r["data"].get("real_qlin", {}).get("ncc_mean")
        if ncc is None:
            continue
        row, col = registry_ablation_axes(mid, registry)
        grid[(row, col)] = max(grid.get((row, col), float("-inf")), float(ncc))

    rows_l = ["BiGRU+Swin", "BiGRU only", "Swin only", "Conv only"]
    cols_l = ["MWCNN (wavelet)", "Standard Conv"]
    mat = np.full((len(rows_l), len(cols_l)), np.nan, dtype=np.float64)
    for i, rr in enumerate(rows_l):
        for j, cc in enumerate(cols_l):
            mat[i, j] = grid.get((rr, cc), np.nan)

    if np.all(np.isnan(mat)):
        print("[ablation_heatmap] No ablation data available — skipped.")
        return

    cmap = _make_cyan_cmap()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    _apply_dark_style(ax, fig)
    ax.grid(False)

    if sns is not None:
        sns.heatmap(mat, annot=True, fmt=".3f", cmap=cmap, vmin=0, vmax=1,
                    xticklabels=cols_l, yticklabels=rows_l, ax=ax,
                    linewidths=1.5, linecolor=BG_COLOR,
                    annot_kws={"fontsize": 11, "fontfamily": mono, "color": TEXT_COLOR},
                    cbar_kws={"shrink": 0.8})
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(colors=TEXT_COLOR)
        cbar.outline.set_edgecolor(GRID_COLOR)
    else:
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(cols_l)))
        ax.set_xticklabels(cols_l)
        ax.set_yticks(range(len(rows_l)))
        ax.set_yticklabels(rows_l)
        for i in range(len(rows_l)):
            for j in range(len(cols_l)):
                v = mat[i, j]
                txt = f"{v:.3f}" if not np.isnan(v) else ""
                ax.text(j, i, txt, ha="center", va="center",
                        color=TEXT_COLOR, fontsize=11, fontfamily=mono)
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.tick_params(colors=TEXT_COLOR)
        cbar.outline.set_edgecolor(GRID_COLOR)

    ax.set_xlabel("Encoder Type", fontsize=10, color=TEXT_COLOR)
    ax.set_ylabel("Bottleneck Type", fontsize=10, color=TEXT_COLOR)
    ax.set_title("Ablation Heatmap — NCC-Real", fontsize=12, fontweight="bold",
                 color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"[ablation_heatmap] Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Figure 3 — Efficiency scatter
# ---------------------------------------------------------------------------

def generate_efficiency_scatter(rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    mono = _mono()
    xs, ys, sizes, labels = [], [], [], []

    for r in rows:
        d = r["data"]
        lat = d.get("efficiency", {}).get("inference_latency_ms_b1")
        ncc = d.get("real_qlin", {}).get("ncc_mean")
        pm = d.get("efficiency", {}).get("param_count_M")
        if lat is None or lat == 0:
            continue
        xs.append(float(lat))
        ys.append(float(ncc) if ncc is not None else 0.0)
        sizes.append(float(pm) if pm else 1.0)
        labels.append(r["model_id"])

    if not xs:
        print("[efficiency_scatter] No efficiency data — skipped.")
        return

    size_arr = np.array(sizes)
    bubble = 80 + 400 * (size_arr / max(size_arr.max(), 1e-9))

    fig, ax = plt.subplots(figsize=(9, 6))
    _apply_dark_style(ax, fig)

    scatter = ax.scatter(xs, ys, s=bubble, c=CYAN, alpha=0.75,
                         edgecolors=TEXT_COLOR, linewidths=0.6, zorder=5)

    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (xs[i], ys[i]),
                    textcoords="offset points", xytext=(8, 6),
                    fontsize=7, color=TEXT_COLOR, fontfamily=mono)

    ax.set_xlabel("Inference Latency (ms, batch=1)", fontsize=10)
    ax.set_ylabel("NCC-Real", fontsize=10)
    ax.set_title("Efficiency vs Quality — bubble size = param count (M)",
                 fontsize=12, fontweight="bold")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"[efficiency_scatter] Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Figure 4 — SNR sweep line plot
# ---------------------------------------------------------------------------

def generate_snr_sweep(rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    mono = _mono()
    palette = [CYAN, GREEN, "#f59e0b", "#ef4444", "#a78bfa",
               "#f472b6", "#34d399", "#fb923c", "#60a5fa"]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    _apply_dark_style(ax, fig)

    plotted = 0
    x_levels = list(SNR_LEVELS)

    for idx, r in enumerate(rows):
        per = r["data"].get("synthetic", {}).get("per_snr_ncc", {})
        if not per:
            continue
        ys, valid = [], True
        for lvl in SNR_LEVELS:
            entry = per.get(str(lvl), {})
            val = entry.get("ncc_mean")
            if val is None:
                valid = False
                break
            ys.append(float(val))
        if not valid:
            continue
        color = palette[idx % len(palette)]
        ax.plot(x_levels, ys, marker="o", markersize=5, linewidth=2,
                label=r["model_id"], color=color)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        print("[snr_sweep] No per-SNR data available — skipped.")
        return

    ax.set_xlabel("SNR (dB)", fontsize=10)
    ax.set_ylabel("NCC", fontsize=10)
    ax.set_title("NCC vs SNR Level", fontsize=12, fontweight="bold")
    ax.set_xticks(x_levels)
    ax.set_xticklabels([f"{s:+d}" for s in x_levels], fontfamily=mono)
    ax.legend(facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR,
              fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"[snr_sweep] Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Figure 5 — Loss variant comparison bar chart
# ---------------------------------------------------------------------------

LOSS_VARIANT_IDS = ("MR-TAE-FULL", "MR-TAE-FULL-LossB", "MR-TAE-FULL-LossC")
LOSS_LABELS = ("Loss A (full)", "Loss B", "Loss C")

def generate_loss_comparison(rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    mono = _mono()
    lookup = {r["model_id"]: r["data"] for r in rows}
    names, vals = [], []

    for mid, label in zip(LOSS_VARIANT_IDS, LOSS_LABELS):
        d = lookup.get(mid)
        if d is None:
            continue
        ncc = d.get("real_qlin", {}).get("ncc_mean")
        if ncc is None:
            continue
        names.append(label)
        vals.append(float(ncc))

    if not names:
        print("[loss_comparison] No loss variant data — skipped.")
        return

    colors = [CYAN, GREEN, "#f59e0b"]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(6, 5))
    _apply_dark_style(ax, fig)

    bars = ax.bar(x, vals, width=0.5, color=colors[:len(names)],
                  edgecolor=BG_COLOR, linewidth=0.8)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                f"{h:.4f}", ha="center", va="bottom",
                fontsize=9, color=TEXT_COLOR, fontfamily=mono)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("NCC-Real", fontsize=10)
    ax.set_title("Loss Function Comparison — NCC-Real",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.08)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"[loss_comparison] Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Bonus — Training curves from history.json
# ---------------------------------------------------------------------------

def generate_training_curves(histories: Sequence[Dict[str, Any]], out_dir: Path) -> None:
    """Overlay val_ncc and val_loss for all models that have history.json."""
    mono = _mono()
    palette = [CYAN, GREEN, "#f59e0b", "#ef4444", "#a78bfa",
               "#f472b6", "#34d399", "#fb923c", "#60a5fa"]

    if not histories:
        print("[training_curves] No history.json files found — skipped.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax in (ax1, ax2):
        _apply_dark_style(ax, fig)

    for idx, h in enumerate(histories):
        color = palette[idx % len(palette)]
        mid = h["model_id"]
        d = h["data"]
        epochs = list(range(1, len(d.get("val_ncc", [])) + 1))

        if "val_ncc" in d and d["val_ncc"]:
            ax1.plot(epochs, d["val_ncc"], linewidth=1.5, label=mid, color=color)
        if "val_loss" in d and d["val_loss"]:
            ax2.plot(epochs, d["val_loss"], linewidth=1.5, label=mid, color=color)

    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Val NCC", fontsize=10)
    ax1.set_title("Validation NCC over Training", fontsize=12, fontweight="bold")
    ax1.legend(facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR,
               fontsize=7, loc="lower right")

    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Val Loss", fontsize=10)
    ax2.set_title("Validation Loss over Training", fontsize=12, fontweight="bold")
    ax2.legend(facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR,
               fontsize=7, loc="upper right")

    fig.tight_layout()
    out_path = out_dir / "training_curves.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"[training_curves] Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Markdown results table
# ---------------------------------------------------------------------------

def generate_results_markdown(rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    def fmt(v: Any) -> str:
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    lines = [
        "# Results Comparison",
        "",
        "> Auto-generated by `evaluation/generate_figures.py`.",
        "",
        "| Model | Params (M) | Latency (ms) | NCC-Synth | NCC-Real | RMSE-Synth | RMSE-Real |",
        "|-------|-----------|-------------|-----------|----------|------------|-----------|",
    ]

    for r in sorted(rows, key=lambda z: z["model_id"]):
        d = r["data"]
        e = d.get("efficiency", {})
        s = d.get("synthetic", {})
        q = d.get("real_qlin", {})
        line = (
            f"| {r['model_id']} "
            f"| {fmt(e.get('param_count_M'))} "
            f"| {fmt(e.get('inference_latency_ms_b1'))} "
            f"| {fmt(s.get('ncc_mean'))} "
            f"| {fmt(q.get('ncc_mean'))} "
            f"| {fmt(s.get('rmse_mean'))} "
            f"| {fmt(q.get('rmse_mean'))} |"
        )
        lines.append(line)

    lines.append("")
    lines.append("## Figures")
    lines.append("")
    lines.append("| Figure | Path |")
    lines.append("|--------|------|")
    lines.append("| Comparison bar chart | `results/benchmark/comparison_bar.png` |")
    lines.append("| Ablation heatmap | `results/benchmark/ablation_heatmap.png` |")
    lines.append("| Efficiency scatter | `results/benchmark/efficiency_scatter.png` |")
    lines.append("| SNR sweep | `results/benchmark/snr_sweep.png` |")
    lines.append("| Loss comparison | `results/benchmark/loss_comparison.png` |")
    lines.append("| Training curves | `results/benchmark/training_curves.png` |")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[markdown] Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  generate_figures.py — Benchmark Figure Generator")
    print("=" * 60)

    bench_rows = collect_benchmark_rows()
    hist_rows = collect_history_rows()
    registry = load_registry()

    print(f"\nFound {len(bench_rows)} benchmark result(s): "
          f"{[r['model_id'] for r in bench_rows]}")
    print(f"Found {len(hist_rows)} training history file(s): "
          f"{[r['model_id'] for r in hist_rows]}")
    print(f"Registry contains {len(registry)} model(s)\n")

    BENCHMARK_OUT.mkdir(parents=True, exist_ok=True)

    print("--- Figure 1: Comparison Bar Chart ---")
    generate_comparison_bar(bench_rows, BENCHMARK_OUT / "comparison_bar.png")

    print("--- Figure 2: Ablation Heatmap ---")
    generate_ablation_heatmap(bench_rows, BENCHMARK_OUT / "ablation_heatmap.png")

    print("--- Figure 3: Efficiency Scatter ---")
    generate_efficiency_scatter(bench_rows, BENCHMARK_OUT / "efficiency_scatter.png")

    print("--- Figure 4: SNR Sweep ---")
    generate_snr_sweep(bench_rows, BENCHMARK_OUT / "snr_sweep.png")

    print("--- Figure 5: Loss Comparison ---")
    generate_loss_comparison(bench_rows, BENCHMARK_OUT / "loss_comparison.png")

    print("--- Bonus: Training Curves ---")
    generate_training_curves(hist_rows, BENCHMARK_OUT)

    print("--- Markdown Table ---")
    generate_results_markdown(bench_rows, DOCS_RESULTS)

    print("\nDone.")


if __name__ == "__main__":
    main()
