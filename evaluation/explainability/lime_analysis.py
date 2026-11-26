"""
evaluation/explainability/lime_analysis.py — LIME segment-importance on signal windows.

Usage
-----
python evaluation/explainability/lime_analysis.py \
    --model-id MR-TAE-FULL \
    --checkpoint results/MR-TAE-FULL/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SEGMENT_SIZE = 32    # super-pixel width in samples
N_PERTURBATIONS = 200


def _ncc(pred: np.ndarray, target: np.ndarray) -> float:
    from evaluation.metrics_verified import ncc_1d

    return ncc_1d(pred, target)


def run_lime(
    model_id: str,
    checkpoint_path: str,
    n_examples: int = 5,
    results_root: str = "results",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from mr_tae_fusion.models.mr_tae_fusion import create_model
    from mr_tae_fusion.config import ModelConfig, Config
    from mr_tae_fusion.data.dataset import PDSignalDataset
    from torch.utils.data import DataLoader

    model = create_model(ModelConfig())
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()

    cfg = Config()
    ds = PDSignalDataset(config=cfg, num_samples=n_examples * 4, mode="val", seed=77)
    loader = DataLoader(ds, batch_size=n_examples, shuffle=False)
    batch = next(iter(loader))
    noisy_batch = batch["noisy"][:n_examples].numpy()
    clean_batch = batch["clean"][:n_examples].numpy()

    n_segs = noisy_batch.shape[-1] // SEGMENT_SIZE
    out_dir = Path(results_root) / model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(n_examples, 1, figsize=(14, 3 * n_examples))
    if n_examples == 1:
        axes = [axes]

    for ex_idx in range(n_examples):
        noisy = noisy_batch[ex_idx]
        clean = clean_batch[ex_idx]
        importances = np.zeros(n_segs)

        # Baseline NCC: full signal
        with torch.no_grad():
            inp = torch.tensor(noisy).float().unsqueeze(0).unsqueeze(0).to(device)
            pred_base, _ = model(inp)
        ncc_base = _ncc(pred_base[0, 0].cpu().numpy(), clean)

        # Perturb each segment
        rng = np.random.default_rng(ex_idx)
        for seg in range(n_segs):
            ncc_delta = 0.0
            for _ in range(N_PERTURBATIONS):
                perturbed = noisy.copy()
                perturbed[seg * SEGMENT_SIZE:(seg + 1) * SEGMENT_SIZE] = (
                    rng.normal(0, perturbed.std(), SEGMENT_SIZE)
                )
                with torch.no_grad():
                    inp = torch.tensor(perturbed).float().unsqueeze(0).unsqueeze(0).to(device)
                    pred_p, _ = model(inp)
                ncc_p = _ncc(pred_p[0, 0].cpu().numpy(), clean)
                ncc_delta += ncc_base - ncc_p   # positive = segment important
            importances[seg] = ncc_delta / N_PERTURBATIONS

        # Expand importances to sample resolution
        imp_full = np.repeat(importances, SEGMENT_SIZE)

        ax = axes[ex_idx]
        ax.plot(noisy, lw=0.7, color="steelblue", label="Noisy", alpha=0.7)
        ax2 = ax.twinx()
        ax2.fill_between(range(len(imp_full)), imp_full, alpha=0.4,
                         color="#e05b2f", label="LIME importance")
        ax.set_title(f"Example {ex_idx+1}  base_NCC={ncc_base:.3f}", fontsize=9)
        ax.set_ylabel("Amplitude")
        ax2.set_ylabel("LIME Δ NCC")
        if ex_idx == 0:
            ax.legend(loc="upper left", fontsize=7)
            ax2.legend(loc="upper right", fontsize=7)

    fig.suptitle(f"LIME window analysis — {model_id}", fontsize=11)
    plt.tight_layout()
    path = out_dir / "lime_window_examples.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("LIME figure saved to %s", path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id",   required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-examples", type=int, default=5)
    parser.add_argument("--results",    type=str, default="results")
    args = parser.parse_args()
    run_lime(args.model_id, args.checkpoint, args.n_examples, args.results)


if __name__ == "__main__":
    main()
