"""
Per-method signal-overlay figure generator.

Addresses supervisor feedback (Wong Jee Keen Raymond, 18 Apr 2026):
    "Putting your results in table form is good. But remember to also save
     some figures of each method as you run them. You will need to include
     some figures where you plot the original and denoised signals in the
     same graph for comparison."

Generates, for every model in the registry that has a benchmark checkpoint:
    1. A 4-panel comparison figure at three SNR levels (-15, -5, +5 dB):
       (a) Noisy input vs. clean ground truth
       (b) Denoised output vs. clean ground truth (overlay -- this is the
           key figure the supervisor asked for)
       (c) Residual (denoised - clean) on its own axis
       (d) Power spectral density of noisy / denoised / clean (log-log)
    2. A grid summary across all models on a single page.

Output:
    results/{model_name}/signal_overlay_SNR{XX}dB.png      (per model, per SNR)
    results/{model_name}/signal_overlay_psd_SNR{XX}dB.png  (PSD detail)
    results/method_comparison_grid.png                     (overview)
    results/method_comparison_grid_residuals.png           (residual overview)

Usage:
    python evaluation/generate_method_comparison_figures.py
    python evaluation/generate_method_comparison_figures.py --models MR-TAE-FULL-LossB MWCNN-BiGRU
    python evaluation/generate_method_comparison_figures.py --snr-levels -15 0 5 --window-idx 42

Drop into:
    FYP_MASTER/evaluation/generate_method_comparison_figures.py

Requires:
    matplotlib, numpy, scipy, torch, pyyaml
    The test set at data/test_synthetic_ae.pt with the SHA256 already in your repo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.signal import welch

# ---------------------------------------------------------------------------
# Project paths -- adjust here if your tree differs.
# ---------------------------------------------------------------------------
# This bundle lives under Data/FYP_MASTER/NEW FOCUS in this workspace, while
# the active training repo root is three levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
TEST_SET_SYNTH = PROJECT_ROOT / "data" / "test_synthetic_ae.pt"
TEST_SET_REAL = PROJECT_ROOT / "data" / "test_real_qlin.pt"
MODEL_REGISTRY = PROJECT_ROOT / "config" / "model_registry.yaml"
RESULTS_DIR = PROJECT_ROOT / "results"

# Sampling rate (project memory: AE band, fs = 1 MHz, 2048 samples = 2.048 ms)
FS_HZ = 1_000_000

# AE band of interest for highlighting on PSD plots (20 - 500 kHz)
AE_BAND_HZ = (20_000, 500_000)

# Default SNR breakpoints (matches your curricular phases)
DEFAULT_SNR_LEVELS = (-15, -5, 5)

# Style: thesis-friendly, no decorative emoji-isms, print-safe colours.
# Colour-blind-safe: blue (clean), orange (noisy), green (denoised), grey (residual).
COLOURS = {
    "clean":    "#1f77b4",
    "noisy":    "#ff7f0e",
    "denoised": "#2ca02c",
    "residual": "#7f7f7f",
}

PLT_RC = {
    "figure.dpi":           150,
    "savefig.dpi":          300,
    "font.size":            10,
    "axes.labelsize":       10,
    "axes.titlesize":       11,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "legend.frameon":       False,
    "lines.linewidth":      1.0,
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_test_set_hash(path: Path, expected_hash: str | None) -> None:
    """Refuse to run on a tampered test set. Lifted from your TEST_SET_HASHES.md."""
    if expected_hash is None:
        return
    actual = sha256_of(path)
    if actual != expected_hash:
        raise RuntimeError(
            f"Test-set hash mismatch for {path}\n"
            f"  expected: {expected_hash}\n"
            f"  actual:   {actual}\n"
            f"Refusing to generate figures from a tampered test set."
        )


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation between two 1D signals."""
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom == 0.0:
        return float("nan")
    return float((a * b).sum() / denom)


def snr_db(clean: np.ndarray, denoised: np.ndarray) -> float:
    """Output SNR (dB), where the noise is taken to be (denoised - clean)."""
    noise = denoised - clean
    p_clean = float(np.mean(clean * clean))
    p_noise = float(np.mean(noise * noise))
    if p_noise == 0:
        return float("inf")
    return 10.0 * np.log10(p_clean / p_noise)


def psd(signal: np.ndarray, fs: int = FS_HZ):
    """One-sided PSD via Welch. Uses a window appropriate for 2048-sample frames."""
    nperseg = min(512, len(signal))
    f, p = welch(signal, fs=fs, nperseg=nperseg, scaling="density")
    return f, p


def select_window_at_snr(
    test_set: dict, target_snr_db: int, window_idx: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pick a (noisy, clean) pair from the synthetic test set at the requested SNR level.

    Expected test-set format (matches your build_test_sets.py output):
        {
            "noisy":  Tensor[N, 1, 2048],
            "clean":  Tensor[N, 1, 2048],
            "snr_db": Tensor[N]    # int per-window SNR label
        }
    """
    snr = test_set.get("snr_db")
    if snr is None:
        snr = test_set.get("snr_labels")
    if snr is None:
        raise KeyError("Test set must include `snr_db` or `snr_labels`.")
    if isinstance(snr, torch.Tensor):
        snr = snr.cpu().numpy()
    mask = (snr == target_snr_db)
    if not mask.any():
        raise ValueError(
            f"No windows at SNR={target_snr_db} dB in test set. "
            f"Available: {sorted(set(int(x) for x in snr))}"
        )
    candidate_idx = np.where(mask)[0]
    chosen = candidate_idx[window_idx % len(candidate_idx)]
    noisy = test_set["noisy"][chosen].squeeze().cpu().numpy()
    clean = test_set["clean"][chosen].squeeze().cpu().numpy()
    return noisy, clean


# ---------------------------------------------------------------------------
# Plot primitives
# ---------------------------------------------------------------------------
def plot_per_method(
    model_name: str,
    snr_db_level: int,
    noisy: np.ndarray,
    clean: np.ndarray,
    denoised: np.ndarray,
    out_path: Path,
) -> None:
    """The 4-panel per-method figure the supervisor asked for."""
    t_us = np.arange(len(clean)) / FS_HZ * 1e6  # microseconds

    out_ncc = ncc(clean, denoised)
    out_snr = snr_db(clean, denoised)

    with plt.rc_context(PLT_RC):
        fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
        title = (
            f"{model_name}  -  SNR_in = {snr_db_level} dB   "
            f"-  NCC_out = {out_ncc:.3f}   -  SNR_out = {out_snr:.2f} dB"
        )
        fig.suptitle(title, y=0.99, fontsize=12)

        # (a) Noisy input overlay vs clean
        ax = axes[0, 0]
        ax.plot(t_us, noisy, color=COLOURS["noisy"], label="Noisy input", alpha=0.7)
        ax.plot(t_us, clean, color=COLOURS["clean"], label="Clean GT", linewidth=1.2)
        ax.set_xlabel("Time (us)")
        ax.set_ylabel("Amplitude")
        ax.set_title("(a) Noisy input vs. clean ground truth")
        ax.legend(loc="upper right")

        # (b) The key figure: denoised overlaid on clean
        ax = axes[0, 1]
        ax.plot(t_us, clean, color=COLOURS["clean"], label="Clean GT", linewidth=1.4)
        ax.plot(
            t_us, denoised, color=COLOURS["denoised"], label="Denoised", linewidth=1.0,
            alpha=0.9,
        )
        ax.set_xlabel("Time (us)")
        ax.set_ylabel("Amplitude")
        ax.set_title("(b) Denoised vs. clean ground truth")
        ax.legend(loc="upper right")

        # (c) Residual (denoised - clean): how badly did we do?
        ax = axes[1, 0]
        residual = denoised - clean
        ax.plot(t_us, residual, color=COLOURS["residual"], linewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.4, linestyle="--")
        ax.set_xlabel("Time (us)")
        ax.set_ylabel("Residual (denoised - clean)")
        ax.set_title(f"(c) Residual  -  RMS = {np.sqrt((residual**2).mean()):.4f}")

        # (d) PSD comparison on log-log axes -- shows whether AE-band content was preserved
        ax = axes[1, 1]
        for sig, label, colour in [
            (noisy, "Noisy", COLOURS["noisy"]),
            (clean, "Clean", COLOURS["clean"]),
            (denoised, "Denoised", COLOURS["denoised"]),
        ]:
            f, p = psd(sig)
            ax.loglog(f / 1e3, p, label=label, color=colour, linewidth=1.0)
        # Highlight AE band
        ax.axvspan(
            AE_BAND_HZ[0] / 1e3, AE_BAND_HZ[1] / 1e3,
            color="black", alpha=0.04, label="AE band 20-500 kHz",
        )
        ax.set_xlabel("Frequency (kHz)")
        ax.set_ylabel("PSD (V^2/Hz)")
        ax.set_title("(d) Power spectral density")
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(True, which="both", alpha=0.3)

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def plot_grid(
    rows: list[dict],
    snr_db_level: int,
    out_path: Path,
    show_residual: bool = False,
) -> None:
    """One-page grid summary -- one subplot per model, denoised overlaid on clean."""
    n = len(rows)
    cols = 4
    n_rows = (n + cols - 1) // cols

    with plt.rc_context(PLT_RC):
        fig, axes = plt.subplots(
            n_rows, cols, figsize=(cols * 3.5, n_rows * 2.2),
            sharex=True, sharey=True,
        )
        axes = np.atleast_2d(axes)
        fig.suptitle(
            f"{'Residuals' if show_residual else 'Denoised vs Clean'}  "
            f"--  SNR_in = {snr_db_level} dB   "
            f"--  one window from synthetic AE test set",
            fontsize=12, y=1.0,
        )

        t_us = np.arange(rows[0]["clean"].shape[-1]) / FS_HZ * 1e6
        for i, row in enumerate(rows):
            r, c = divmod(i, cols)
            ax = axes[r, c]
            if show_residual:
                resid = row["denoised"] - row["clean"]
                ax.plot(t_us, resid, color=COLOURS["residual"], linewidth=0.7)
                ax.axhline(0, color="black", linewidth=0.3, linestyle=":")
            else:
                ax.plot(t_us, row["clean"], color=COLOURS["clean"], linewidth=1.0)
                ax.plot(
                    t_us, row["denoised"], color=COLOURS["denoised"],
                    linewidth=0.8, alpha=0.9,
                )
            ax.set_title(
                f"{row['name']}\nNCC={row['ncc']:.3f} SNR_out={row['snr']:.1f} dB",
                fontsize=9,
            )
            ax.tick_params(labelsize=8)

        # Hide unused axes
        for j in range(n, n_rows * cols):
            r, c = divmod(j, cols)
            axes[r, c].set_visible(False)

        # Shared axis labels
        for ax in axes[-1, :]:
            ax.set_xlabel("Time (us)", fontsize=9)
        for ax in axes[:, 0]:
            ax.set_ylabel("Amplitude" if not show_residual else "Residual", fontsize=9)

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Model loading -- adapt this single function if your loader differs
# ---------------------------------------------------------------------------
def load_model(model_name: str, registry: dict, device: torch.device):
    ckpt_path = RESULTS_DIR / model_name / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        return None, None

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Load using the actual mr_tae_fusion factory, not the stub GenericUNetDenoiser.
    # The orchestrator saves weights under 'model_state'.
    try:
        from mr_tae_fusion.config import get_config
        from mr_tae_fusion.models import create_model

        cfg = get_config()

        # Apply ablation flags from the registry entry for this variant.
        model_entry = registry.get(model_name, {})
        for flag, val in model_entry.items():
            if hasattr(cfg.model, flag):
                setattr(cfg.model, flag, val)

        model = create_model(cfg.model)
        model.load_state_dict(payload["model_state"], strict=True)
        model.eval().to(device)
        return model, ckpt_path

    except Exception as e:
        print(f"[warn] Could not load {model_name} via mr_tae_fusion factory: {e}",
              file=sys.stderr)
        return None, None


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="Subset of model names. Default = all in registry with checkpoints.",
    )
    parser.add_argument(
        "--snr-levels", nargs="+", type=int, default=list(DEFAULT_SNR_LEVELS),
        help=f"SNR-in dB levels to render. Default {DEFAULT_SNR_LEVELS}",
    )
    parser.add_argument(
        "--window-idx", type=int, default=0,
        help="Which window-at-this-SNR to use. Default 0 (first one).",
    )
    parser.add_argument(
        "--test-set", type=Path, default=TEST_SET_SYNTH,
        help="Path to synthetic test set .pt file.",
    )
    parser.add_argument(
        "--expected-hash", type=str, default=None,
        help="Optional SHA256 to verify the test set has not been tampered with.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--no-grid", action="store_true",
        help="Skip the cross-model grid summary figure.",
    )
    args = parser.parse_args(argv)

    # ---- Load registry ----
    if not MODEL_REGISTRY.exists():
        print(f"[fatal] Model registry not found at {MODEL_REGISTRY}", file=sys.stderr)
        return 1
    with MODEL_REGISTRY.open() as f:
        registry = yaml.safe_load(f) or {}
    if isinstance(registry, dict) and "models" in registry:
        registry = registry["models"]

    # Decide which models to render
    if args.models:
        model_names = args.models
    else:
        model_names = []
        for name in registry:
            if any(
                (PROJECT_ROOT / "results" / name / "checkpoints" / "best.pt").exists()
                or (PROJECT_ROOT / "results" / name / "best.pt").exists()
                or (PROJECT_ROOT / "results" / name / "best.pth").exists()
                for _ in [None]
            ):
                model_names.append(name)
        if not model_names:
            print(
                "[warn] No checkpoints found in results/. "
                "Pass --models explicitly to override.",
                file=sys.stderr,
            )
            return 1

    print(f"Rendering {len(model_names)} models at SNRs {args.snr_levels} dB")

    # ---- Load test set ----
    if args.expected_hash:
        verify_test_set_hash(args.test_set, args.expected_hash)
    test_set = torch.load(args.test_set, map_location="cpu", weights_only=False)
    n_windows = int(test_set.get("noisy").shape[0]) if "noisy" in test_set else -1
    print(f"Loaded test set: {args.test_set} ({n_windows} windows)")

    device = torch.device(args.device)

    # ---- For each SNR level, render per-method figures and a grid ----
    summary_rows = []
    for snr_level in args.snr_levels:
        try:
            noisy, clean = select_window_at_snr(test_set, snr_level, args.window_idx)
        except ValueError as e:
            print(f"[skip] {e}", file=sys.stderr)
            continue
        noisy_t = torch.from_numpy(noisy).float().view(1, 1, -1).to(device)

        grid_rows = []
        for name in model_names:
            model, ckpt_path = load_model(name, registry, device)
            if model is None:
                print(f"[skip] No checkpoint for {name}", file=sys.stderr)
                continue
            with torch.no_grad():
                out = model(noisy_t)
                # Tolerate dual-head MTL outputs.
                if isinstance(out, (tuple, list)):
                    out = out[0]
                elif isinstance(out, dict):
                    out = out.get("denoised") or out.get("signal") or next(iter(out.values()))
            denoised = out.squeeze().cpu().numpy()

            # Per-method figure
            per_path = (
                RESULTS_DIR / name / f"signal_overlay_SNR{snr_level:+d}dB.png"
            )
            plot_per_method(name, snr_level, noisy, clean, denoised, per_path)
            print(f"  wrote {per_path.relative_to(PROJECT_ROOT)}")

            grid_rows.append(
                dict(
                    name=name, clean=clean, noisy=noisy, denoised=denoised,
                    ncc=ncc(clean, denoised), snr=snr_db(clean, denoised),
                )
            )
            summary_rows.append(
                dict(
                    model=name, snr_in_db=snr_level,
                    ncc=ncc(clean, denoised),
                    snr_out_db=snr_db(clean, denoised),
                    rms_residual=float(np.sqrt(((denoised - clean) ** 2).mean())),
                    figure=str(per_path.relative_to(PROJECT_ROOT)),
                )
            )

            # Free the model for the next iteration
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Grid for this SNR
        if grid_rows and not args.no_grid:
            grid_path = (
                RESULTS_DIR / f"method_comparison_grid_SNR{snr_level:+d}dB.png"
            )
            plot_grid(grid_rows, snr_level, grid_path, show_residual=False)
            print(f"  wrote {grid_path.relative_to(PROJECT_ROOT)}")
            resid_path = (
                RESULTS_DIR / f"method_comparison_residuals_SNR{snr_level:+d}dB.png"
            )
            plot_grid(grid_rows, snr_level, resid_path, show_residual=True)
            print(f"  wrote {resid_path.relative_to(PROJECT_ROOT)}")

    # ---- Dump summary JSON for downstream use (RESULTS_COMPARISON.md, etc.) ----
    summary_path = RESULTS_DIR / "method_comparison_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"\nSummary written to {summary_path.relative_to(PROJECT_ROOT)}")
    print(f"Rendered {len(summary_rows)} (model x SNR) combinations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
