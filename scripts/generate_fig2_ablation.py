"""
Fig. 2 — Focused ablation: Loss A/B, MWCNN-BiGRU, Savitzky–Golay, Wiener.

Synthetic NCC (blue): vs clean ground truth on test_synthetic_ae.pt
Real Q.Lin NCC (orange): input-window pseudo-reference on test_real_qlin.pt

Output: results/paper_assets/fig2_ablation_results.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Try Model"))

from evaluation.benchmark_runner import (  # noqa: E402
    PROJECT_ROOT as BR_ROOT,
    REAL_TEST_PATH,
    SYNTH_TEST_PATH,
    _apply_baseline_to_batch,
    _ncc_batch,
)

OUT = PROJECT_ROOT / "results" / "paper_assets" / "fig2_ablation_results.png"
CACHE = PROJECT_ROOT / "results" / "paper_assets" / "fig2_ablation_cache.json"

# registry id -> display label (fixed order for the paper)
FIG2_SPECS: list[tuple[str, str]] = [
    ("MR-TAE-FULL", "MR-TAE-FULL (Loss A)"),
    ("MR-TAE-FULL-LossB", "MR-TAE-FULL-LossB (Loss B)"),
    ("MWCNN-BiGRU", "MWCNN-BiGRU (1.31 M)"),
    ("Baseline-Savitzky-Golay", "Savitzky–Golay"),
    ("Wiener-scipy", "Wiener (scipy)"),
]

LOSS_A_LABEL = "MR-TAE-FULL (Loss A)"
HIGHLIGHT_COLOR = "#fef9c3"  # light yellow band
HIGHLIGHT_EDGE = "#ca8a04"  # amber outline — Loss A only


def _apply_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-paper")
    except OSError:
        try:
            plt.style.use("seaborn-paper")
        except OSError:
            pass
    plt.rcParams.update({"font.size": 10, "figure.facecolor": "white"})


def _load_json(model_id: str) -> dict | None:
    p = PROJECT_ROOT / "results" / model_id / "benchmark_results.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def evaluate_wiener() -> dict[str, float]:
    from scipy.signal import wiener

    def fn(sig: np.ndarray) -> np.ndarray:
        x = np.asarray(sig, dtype=np.float64).ravel()
        return np.asarray(wiener(x, mysize=21), dtype=np.float32)

    out: dict[str, float] = {}
    synth_path = BR_ROOT / SYNTH_TEST_PATH
    if synth_path.is_file():
        pack = torch.load(synth_path, map_location="cpu", weights_only=False)
        noisy, clean = pack["noisy"], pack["clean"]
        preds = _apply_baseline_to_batch(noisy, fn)
        tgt = clean.unsqueeze(1) if clean.dim() == 2 else clean
        out["synthetic_ncc_mean"] = float(np.mean(_ncc_batch(preds, tgt)))

    real_path = BR_ROOT / REAL_TEST_PATH
    if real_path.is_file():
        pack = torch.load(real_path, map_location="cpu", weights_only=False)
        windows = pack["windows"]
        preds = _apply_baseline_to_batch(windows, fn)
        ref = windows.unsqueeze(1) if windows.dim() == 2 else windows
        out["real_qlin_ncc_mean"] = float(np.mean(_ncc_batch(preds, ref)))

    return out


def collect_rows() -> list[tuple[str, float, float]]:
    wiener = evaluate_wiener()
    rows: list[tuple[str, float, float]] = []

    for mid, label in FIG2_SPECS:
        if mid == "Wiener-scipy":
            syn = wiener.get("synthetic_ncc_mean")
            real = wiener.get("real_qlin_ncc_mean")
            if syn is None or real is None:
                print("[warn] Wiener evaluation incomplete — skipping")
                continue
            rows.append((label, float(syn), float(real)))
            continue

        blob = _load_json(mid)
        if blob is None:
            print(f"[warn] missing benchmark for {mid}")
            continue
        syn = (blob.get("synthetic") or {}).get("ncc_mean")
        real = (blob.get("real_qlin") or {}).get("ncc_mean")
        if syn is None or real is None:
            print(f"[warn] incomplete metrics for {mid}")
            continue
        rows.append((label, float(syn), float(real)))

    return rows


def _order_rows(rows: list[tuple[str, float, float]]) -> list[tuple[str, float, float]]:
    """Bottom → top: weakest real NCC first; MR-TAE-FULL (Loss A) pinned at top."""
    loss_a = [r for r in rows if r[0] == LOSS_A_LABEL]
    others = [r for r in rows if r[0] != LOSS_A_LABEL]
    others.sort(key=lambda r: r[2])  # ascending real NCC
    return others + loss_a


def render(rows: list[tuple[str, float, float]], out: Path) -> None:
    rows = _order_rows(rows)
    labels = [r[0] for r in rows]
    synth_v = [r[1] for r in rows]
    real_v = [r[2] for r in rows]
    highlights = [lab == LOSS_A_LABEL for lab in labels]

    y = np.arange(len(labels))
    h = 0.34
    c_syn = plt.cm.tab10(0)
    c_real = plt.cm.tab10(1)

    fig, ax = plt.subplots(figsize=(9.2, max(3.8, len(labels) * 0.72 + 1.2)))
    fig.subplots_adjust(right=0.76)

    for i, hl in enumerate(highlights):
        if hl:
            ax.axhspan(i - 0.48, i + 0.48, color=HIGHLIGHT_COLOR, alpha=0.92, zorder=0)

    ax.barh(
        y + h / 2,
        synth_v,
        height=h,
        label="Synthetic NCC",
        color=c_syn,
        edgecolor="#1e293b",
        linewidth=0.4,
        zorder=2,
    )
    ax.barh(
        y - h / 2,
        real_v,
        height=h,
        label="Real Q.Lin NCC",
        color=c_real,
        edgecolor="#1e293b",
        linewidth=0.4,
        zorder=2,
    )

    for i, (lab, sv, rv) in enumerate(rows):
        hl = highlights[i]
        ax.text(sv + 0.015, y[i] + h / 2, f"{sv:.3f}", va="center", fontsize=8, color="#1e40af")
        ax.text(rv + 0.015, y[i] - h / 2, f"{rv:.3f}", va="center", fontsize=8, color="#c2410c")
        if hl:
            for off, val in ((h / 2, sv), (-h / 2, rv)):
                ax.barh(
                    y[i] + off,
                    val,
                    height=h,
                    color="none",
                    edgecolor=HIGHLIGHT_EDGE,
                    linewidth=2.2,
                    zorder=3,
                )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Normalized cross-correlation (NCC)", fontsize=10)
    ax.set_xlim(0, 1.0)
    ax.set_title(
        "Fig. 2 — Ablation and baseline comparison (MR-TAE, MWCNN-BiGRU, Savitzky–Golay, Wiener)",
        fontsize=11,
        fontweight="bold",
        loc="left",
        pad=12,
    )
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(1.02, 0.02),
        bbox_transform=ax.transAxes,
        fontsize=7,
        frameon=True,
        framealpha=0.92,
        borderaxespad=0.0,
        handlelength=1.0,
        handletextpad=0.35,
        labelspacing=0.3,
    )
    fig.tight_layout(rect=(0, 0, 0.98, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] {out}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Recompute metrics (incl. Wiener)")
    args = parser.parse_args()

    _apply_style()
    if CACHE.is_file() and not args.refresh:
        raw = json.loads(CACHE.read_text(encoding="utf-8"))
        rows = [(r["label"], r["syn"], r["real"]) for r in raw]
    else:
        rows = collect_rows()
        if not rows:
            raise SystemExit("No data rows — run benchmarks first.")
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(
            json.dumps(
                [{"label": a, "syn": b, "real": c} for a, b, c in rows],
                indent=2,
            ),
            encoding="utf-8",
        )
    render(rows, OUT)


if __name__ == "__main__":
    main()
