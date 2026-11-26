"""
evaluation/explainability/shap_analysis.py — SHAP DeepExplainer on best model.

Usage
-----
python evaluation/explainability/shap_analysis.py \
    --model-id MR-TAE-FULL \
    --checkpoint results/MR-TAE-FULL/checkpoints/best.pt \
    --n-samples 100
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


def run_shap(
    model_id: str,
    checkpoint_path: str,
    n_samples: int = 100,
    results_root: str = "results",
) -> None:
    try:
        import shap
    except ImportError:
        logger.error("shap not installed. Run: pip install shap")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from mr_tae_fusion.models.mr_tae_fusion import create_model
    from mr_tae_fusion.config import ModelConfig, Config
    from mr_tae_fusion.data.dataset import PDSignalDataset
    from torch.utils.data import DataLoader

    model = create_model(ModelConfig())
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()

    # Extract encoder for explaining
    encoder = model.encoder

    cfg = Config()
    ds = PDSignalDataset(config=cfg, num_samples=n_samples * 2, mode="val", seed=99)
    loader = DataLoader(ds, batch_size=n_samples, shuffle=False)
    batch = next(iter(loader))
    test_batch = batch["noisy"][:n_samples].unsqueeze(1).to(device)
    background  = batch["noisy"][n_samples:n_samples+20].unsqueeze(1).to(device)

    # Wrap encoder output (takes noisy, returns first skip level)
    class EncoderWrapper(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x):
            enc, _, _ = self.m(x)
            return enc.mean(1, keepdim=True)   # reduce to (B,1,L) for SHAP

    wrapper = EncoderWrapper(encoder).to(device)

    explainer = shap.DeepExplainer(wrapper, background)
    shap_values = explainer.shap_values(test_batch)

    # Summary plot
    out_dir = Path(results_root) / model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    sv = np.array(shap_values).squeeze()
    if sv.ndim == 3:
        sv = sv.mean(0)  # average over class dimension if multi-output

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.abs(sv).mean(0), lw=1.2, color="#e05b2f")
    ax.set_title(f"SHAP — mean |value| per time step ({model_id})")
    ax.set_xlabel("Sample index (0–2047)")
    ax.set_ylabel("|SHAP value|")
    plt.tight_layout()
    fig.savefig(out_dir / "shap_summary.png", dpi=150)
    plt.close(fig)

    # Individual pulse overlay
    sig_np = test_batch[0, 0].cpu().numpy()
    sv_np  = sv[0] if sv.ndim == 2 else sv
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(sig_np, lw=0.8, color="steelblue", label="Noisy signal")
    axes[0].legend(fontsize=8)
    axes[1].fill_between(range(len(sv_np)), sv_np, alpha=0.6,
                         color="#e05b2f", label="|SHAP|")
    axes[1].set_xlabel("Sample index")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"SHAP overlay — sample 0 ({model_id})")
    plt.tight_layout()
    fig.savefig(out_dir / "shap_pulse_example.png", dpi=150)
    plt.close(fig)

    logger.info("SHAP figures saved to %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id",   required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-samples",  type=int, default=100)
    parser.add_argument("--results",    type=str, default="results")
    args = parser.parse_args()
    run_shap(args.model_id, args.checkpoint, args.n_samples, args.results)


if __name__ == "__main__":
    main()
