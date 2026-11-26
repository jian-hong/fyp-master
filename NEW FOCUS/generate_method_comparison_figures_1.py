"""
generate_method_comparison_figures.py  — FIXED VERSION
=======================================================
Root cause of original failure:
    The script ran from  NEW FOCUS\  so Python could not resolve the
    `mr_tae_fusion` package, which lives one directory up at
    FYP_MASTER\mr_tae_fusion\.

Fix applied:
    1.  Walk up from __file__ until we find the `mr_tae_fusion` package
        directory, then insert that parent into sys.path.
    2.  Robust checkpoint discovery: tries best.pt -> last.pt -> any *.pt.
    3.  All variant configs are declared inline so the script is self-
        contained even before any import of mr_tae_fusion succeeds.

Usage (from any working directory):
    cd "D:\\New folder (2)\\Data\\FYP_MASTER\\NEW FOCUS"
    python generate_method_comparison_figures.py

Outputs  results\\method_comparison_figures\\  (one PNG per SNR level
plus a summary JSON).
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 0.  PATH BOOTSTRAP  — must happen before any mr_tae_fusion import
# ---------------------------------------------------------------------------

def _find_project_root() -> Path:
    """
    Walk upward from this file until we find a directory that contains
    a 'mr_tae_fusion' sub-directory (the package root).
    Falls back to a hard-coded path if not found.
    """
    candidate = Path(__file__).resolve()
    for _ in range(6):          # walk up at most 6 levels
        candidate = candidate.parent
        if (candidate / "mr_tae_fusion").is_dir():
            return candidate
    # Hard-coded fallback  -  adjust if your layout differs
    fallback = Path(r"D:\New folder (2)\Data\FYP_MASTER")
    if fallback.exists():
        return fallback
    raise RuntimeError(
        "Could not locate the 'mr_tae_fusion' package.  "
        "Set PROJECT_ROOT manually in this script."
    )

PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"[path] PROJECT_ROOT = {PROJECT_ROOT}")


# ---------------------------------------------------------------------------
# 1.  ABLATION VARIANT REGISTRY
#     mirrors config/model_registry.yaml so we need no YAML at runtime
# ---------------------------------------------------------------------------

VARIANTS: List[Dict] = [
    dict(id="MR-TAE-FULL",            wavelet=True,  bigru=True,  swin=True,  attn=True,  mtl=True,  loss="A"),
    dict(id="MR-TAE-noBiGRU",         wavelet=True,  bigru=False, swin=True,  attn=True,  mtl=True,  loss="A"),
    dict(id="MR-TAE-noSwin",          wavelet=True,  bigru=True,  swin=False, attn=True,  mtl=True,  loss="A"),
    dict(id="MR-TAE-noAttn",          wavelet=True,  bigru=True,  swin=True,  attn=False, mtl=True,  loss="A"),
    dict(id="MR-TAE-noMTL",           wavelet=True,  bigru=True,  swin=True,  attn=True,  mtl=False, loss="A"),
    dict(id="MR-TAE-noWavelet",       wavelet=False, bigru=True,  swin=True,  attn=True,  mtl=True,  loss="A"),
    dict(id="MR-TAE-noSwin-noBiGRU",  wavelet=True,  bigru=False, swin=False, attn=True,  mtl=True,  loss="A"),
    dict(id="MR-TAE-noSwin-noAttn",   wavelet=True,  bigru=True,  swin=False, attn=False, mtl=True,  loss="A"),
    dict(id="MR-TAE-noBiGRU-noAttn",  wavelet=True,  bigru=False, swin=True,  attn=False, mtl=True,  loss="A"),
    dict(id="MR-TAE-FULL-LossB",      wavelet=True,  bigru=True,  swin=True,  attn=True,  mtl=True,  loss="B"),
    dict(id="MR-TAE-FULL-LossC",      wavelet=True,  bigru=True,  swin=True,  attn=True,  mtl=False, loss="C"),
    dict(id="MWCNN-BiGRU",            wavelet=True,  bigru=True,  swin=False, attn=False, mtl=False, loss="B"),
    dict(id="UNet-BiGRU-Swin",        wavelet=False, bigru=True,  swin=True,  attn=False, mtl=True,  loss="A"),
]


# ---------------------------------------------------------------------------
# 2.  CHECKPOINT DISCOVERY
# ---------------------------------------------------------------------------

RESULTS_ROOT = PROJECT_ROOT / "results"


def find_checkpoint(model_id: str) -> Optional[Path]:
    """
    Priority: best.pt -> last.pt -> latest *.pt by mtime.
    Searches  RESULTS_ROOT/{model_id}/checkpoints/
    """
    base = RESULTS_ROOT / model_id / "checkpoints"
    if not base.exists():
        # Windows case-insensitive fallback glob
        matches = list(RESULTS_ROOT.glob("**/checkpoints"))
        base_candidates = [m for m in matches
                           if m.parent.name.lower() == model_id.lower()]
        if not base_candidates:
            return None
        base = base_candidates[0]

    for name in ("best.pt", "last.pt"):
        p = base / name
        if p.exists():
            return p

    pts = sorted(base.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
    return pts[0] if pts else None


# ---------------------------------------------------------------------------
# 3.  MODEL FACTORY
# ---------------------------------------------------------------------------

def _build_via_package(cfg: Dict) -> Optional[object]:
    """Attempt to build the model using the mr_tae_fusion package."""
    try:
        from mr_tae_fusion.models.ablation_config import AblationConfig
        from mr_tae_fusion.models.mr_tae_fusion import create_model
        from mr_tae_fusion.config import Config

        ablation = AblationConfig(
            use_wavelet_pooling=cfg["wavelet"],
            use_bigru=cfg["bigru"],
            use_swin=cfg["swin"],
            use_attention_gates=cfg["attn"],
            use_mtl=cfg["mtl"],
            loss_config=cfg["loss"],
            model_id=cfg["id"],
        )
        model = create_model(Config().model, ablation)
        return model
    except Exception as exc:
        print(f"  [warn] Package build failed ({cfg['id']}): {exc}")
        return None


def load_model(cfg: Dict, device: str) -> Optional[torch.nn.Module]:
    ckpt_path = find_checkpoint(cfg["id"])
    if ckpt_path is None:
        print(f"  [skip] No checkpoint: {cfg['id']}  "
              f"(looked in {RESULTS_ROOT / cfg['id'] / 'checkpoints'})")
        return None

    model = _build_via_package(cfg)
    if model is None:
        print(f"  [skip] Cannot build architecture for {cfg['id']}")
        return None

    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        # Handle wrapped checkpoint dicts
        if isinstance(state, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                if key in state:
                    state = state[key]
                    break
        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()
        rel = ckpt_path.relative_to(PROJECT_ROOT)
        print(f"  [ok]   {cfg['id']:35s}  <- {rel}")
        return model
    except Exception as exc:
        print(f"  [warn] Checkpoint load error ({cfg['id']}): {exc}")
        return None


# ---------------------------------------------------------------------------
# 4.  TEST DATA LOADING
# ---------------------------------------------------------------------------

TEST_DATA_PATHS = [
    PROJECT_ROOT / "data" / "test_synthetic_ae.pt",
    PROJECT_ROOT / "data" / "test_synthetic_ae.pt",
]
TARGET_SNRS = [-15, -5, 5]


def load_test_batch(n_per_snr: int = 4) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """Returns {snr: (noisy, clean)} tensors shaped (n, 1, 2048)."""
    for candidate in TEST_DATA_PATHS:
        if candidate.exists():
            data = torch.load(candidate, map_location="cpu", weights_only=False)
            n_total = data["noisy"].shape[0]
            print(f"[data] Loaded {candidate.name}  ({n_total:,} windows)")
            out: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
            for snr in TARGET_SNRS:
                if "snr_labels" in data:
                    mask = data["snr_labels"] == snr
                else:
                    # No SNR labels: just take first n samples
                    mask = torch.arange(n_total) < n_per_snr
                idx = mask.nonzero(as_tuple=False).squeeze()
                if idx.dim() == 0:
                    idx = idx.unsqueeze(0)
                idx = idx[:n_per_snr]
                noisy = data["noisy"][idx]
                clean = data["clean"][idx]
                # Ensure shape (n, 1, 2048)
                if noisy.dim() == 2:
                    noisy = noisy.unsqueeze(1)
                    clean = clean.unsqueeze(1)
                # Pad if fewer samples than requested
                while noisy.shape[0] < n_per_snr:
                    noisy = torch.cat([noisy, noisy[:1]], dim=0)
                    clean = torch.cat([clean, clean[:1]], dim=0)
                out[snr] = (noisy[:n_per_snr], clean[:n_per_snr])
            return out

    # Synthetic fallback (no test file found)
    print("[warn] Test file not found. Generating synthetic fallback signals.")
    L = 2048
    out = {}
    for snr in TARGET_SNRS:
        t = torch.linspace(0, 0.002048, L)
        # AE-like pulse: damped sinusoid at ~100 kHz (at 1 MHz Fs = 100 samples/cycle)
        fc_samples = 100
        tau_samples = 600
        clean_1d = torch.exp(-t * 1e3 / 0.6) * torch.sin(2 * np.pi * t * 1e3 / 0.01)
        # Normalise
        clean_1d = clean_1d / clean_1d.abs().max().clamp(min=1e-8)
        clean_batch = clean_1d.unsqueeze(0).unsqueeze(0).expand(n_per_snr, 1, L)
        sigma = 10 ** (-snr / 20)
        noisy_batch = clean_batch + sigma * torch.randn(n_per_snr, 1, L)
        out[snr] = (noisy_batch, clean_batch)
    return out


# ---------------------------------------------------------------------------
# 5.  INFERENCE HELPERS
# ---------------------------------------------------------------------------

@torch.no_grad()
def denoise(model: torch.nn.Module,
            noisy: torch.Tensor,
            device: str) -> torch.Tensor:
    x = noisy.to(device)
    out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out.cpu()


def ncc_score(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().double().numpy()
    b = b.flatten().double().numpy()
    denom = float(np.sqrt((a**2).sum() * (b**2).sum()))
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


# ---------------------------------------------------------------------------
# 6.  FIGURE GENERATION
# ---------------------------------------------------------------------------

OUT_DIR = Path("results") / "method_comparison_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TAB20 = list(plt.cm.tab20.colors)


def plot_snr_panel(
    snr: int,
    noisy_batch: torch.Tensor,
    clean_batch: torch.Tensor,
    denoised_map: Dict[str, torch.Tensor],
    sample_idx: int = 0,
) -> Path:
    models = list(denoised_map.keys())
    n_rows = 2 + len(models)
    fig, axes = plt.subplots(n_rows, 1,
                             figsize=(14, 2.0 * n_rows),
                             sharex=True)
    x = np.arange(2048)

    def _row(ax, sig_1x1xL, label, colour, nc=None):
        y = sig_1x1xL[0, 0].numpy()
        title = label + (f"  NCC={nc:.4f}" if nc is not None else "")
        ax.plot(x, y, lw=0.55, color=colour)
        ax.set_ylabel(title, fontsize=7.5, rotation=0,
                      labelpad=130, ha="right", va="center")
        ax.set_ylim(-1.7, 1.7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25, lw=0.4)

    _row(axes[0], clean_batch[sample_idx:sample_idx+1], "Clean (ref)", "black")
    nc_noisy = ncc_score(noisy_batch[sample_idx], clean_batch[sample_idx])
    _row(axes[1], noisy_batch[sample_idx:sample_idx+1],
         f"Noisy input ({snr:+d} dB)", "#666666", nc_noisy)

    for i, (name, denoised) in enumerate(denoised_map.items()):
        nc = ncc_score(denoised[sample_idx], clean_batch[sample_idx])
        _row(axes[2 + i], denoised[sample_idx:sample_idx+1],
             name, TAB20[i % len(TAB20)], nc)

    axes[-1].set_xlabel("Sample index (2048-pt window)", fontsize=9)
    fig.suptitle(f"MR-TAE Ablation — Denoising Comparison  |  Input SNR = {snr:+d} dB",
                 fontsize=11, fontweight="bold", y=1.002)
    plt.tight_layout()

    out = OUT_DIR / f"comparison_snr_{snr:+03d}dB.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {out}")
    return out


# ---------------------------------------------------------------------------
# 7.  MAIN
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")
    if device == "cuda":
        print(f"         {torch.cuda.get_device_name(0)}")

    test_data = load_test_batch(n_per_snr=4)

    print(f"\n[models] Loading {len(VARIANTS)} variants ...")
    loaded: Dict[str, torch.nn.Module] = {}
    for cfg in VARIANTS:
        m = load_model(cfg, device)
        if m is not None:
            loaded[cfg["id"]] = m

    print(f"\n[models] {len(loaded)}/{len(VARIANTS)} loaded.\n")

    if not loaded:
        print("[ERROR] No models could be loaded.  Checkpoint inventory:")
        for cfg in VARIANTS:
            cp = find_checkpoint(cfg["id"])
            tag = str(cp) if cp else "NOT FOUND"
            print(f"  {cfg['id']:38s}  {tag}")
        return

    summary: Dict = {"rendered_combinations": 0, "models": {}}

    for snr, (noisy_batch, clean_batch) in sorted(test_data.items()):
        print(f"\n--- SNR = {snr:+d} dB ---")
        denoised_map: Dict[str, torch.Tensor] = {}
        for name, model in loaded.items():
            try:
                out = denoise(model, noisy_batch[:1], device)
                denoised_map[name] = out
            except Exception as exc:
                print(f"  [warn] {name}: {exc}")

        if denoised_map:
            plot_snr_panel(snr, noisy_batch, clean_batch, denoised_map)
            summary["rendered_combinations"] += len(denoised_map)
            for name, denoised in denoised_map.items():
                nc = ncc_score(denoised[0], clean_batch[0])
                summary["models"].setdefault(name, {})[f"ncc_{snr:+d}dB"] = round(nc, 6)

    # Bar chart: NCC at -5 dB across models
    if summary["models"]:
        key = "ncc_-5dB"
        names = [n for n, v in summary["models"].items() if key in v]
        nccs  = [summary["models"][n][key] for n in names]
        if names:
            fig, ax = plt.subplots(figsize=(12, 4))
            bars = ax.bar(range(len(names)), nccs,
                          color=[TAB20[i % len(TAB20)] for i in range(len(names))])
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
            ax.set_ylabel("NCC (–5 dB synthetic)", fontsize=9)
            ax.set_title("Model NCC Comparison at Input SNR = –5 dB", fontweight="bold")
            ax.set_ylim(0, 1.05)
            ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            bar_path = OUT_DIR / "ncc_bar_chart_minus5dB.png"
            fig.savefig(bar_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"\n  [fig] {bar_path}")

    summary_path = OUT_DIR / "method_comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] Summary -> {summary_path}")
    print(f"[done] {summary['rendered_combinations']} (model x SNR) combinations rendered.")


if __name__ == "__main__":
    main()
