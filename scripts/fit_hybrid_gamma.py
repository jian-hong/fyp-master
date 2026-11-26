#!/usr/bin/env python3
"""
Fit a scalar blend gamma for hybrid_nn_savgol on **synthetic training windows only**.
Never touches data/test_synthetic_ae.pt or data/test_real_qlin.pt.

Usage:
  python scripts/fit_hybrid_gamma.py --model-id MR-TAE-FULL-LossB \\
      --checkpoint results/MR-TAE-FULL-LossB/checkpoints/best.pt

Writes results/hybrid_savgol_gamma.json with best gamma and mean NCC.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Try Model"))

from evaluation.benchmark_runner import (  # noqa: E402
    _load_checkpoint_state,
    hybrid_nn_savgol,
    savgol_denoise,
)
from mr_tae_fusion.config import Config  # noqa: E402
from mr_tae_fusion.data.dataset import PDSignalDataset  # noqa: E402
from mr_tae_fusion.models.mr_tae_fusion import create_model  # noqa: E402
from mr_tae_fusion.models.ablation_config import AblationConfig  # noqa: E402
from pipeline.mlflow_orchestrator import apply_noise_regime  # noqa: E402
from training.curricular_scheduler import PhaseConfig  # noqa: E402

import yaml  # noqa: E402


def load_registry(mid: str) -> dict:
    with open(ROOT / "config" / "model_registry.yaml") as f:
        reg = yaml.safe_load(f)["models"][mid]
    return reg


def ablation_from(mid: str) -> AblationConfig:
    r = load_registry(mid)
    return AblationConfig(
        use_wavelet_pooling=r.get("use_wavelet_pooling", True),
        use_bigru=r.get("use_bigru", True),
        use_swin=r.get("use_swin", True),
        use_attention_gates=r.get("use_attention_gates", True),
        use_mtl=r.get("use_mtl", True),
        loss_config=r.get("loss_config", "A"),
        model_id=mid,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-samples", type=int, default=512)
    ap.add_argument("--snr-phase", choices=("3", "2", "any"), default="3", help="Use Phase 3 (hard SNR) by default.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_data = ROOT / "config" / "training_config.yaml"
    tcfg = {}
    if cfg_data.exists():
        with open(cfg_data) as f:
            tcfg = yaml.safe_load(f) or {}

    p1 = int(tcfg.get("phase1_epochs", 35))
    p2 = int(tcfg.get("phase2_epochs", 35))
    p3 = int(tcfg.get("phase3_epochs", 30))
    total_ep = p1 + p2 + p3

    master = Config()
    apply_noise_regime(master, tcfg)

    phases = (
        PhaseConfig(5.0, 15.0, p1, "wgn"),
        PhaseConfig(-5.0, 5.0, p2, "wgn_impulsive"),
        PhaseConfig(-20.0, -5.0, p3, "composite"),
    )
    if str(tcfg.get("noise_regime", "standard")).lower() == "tough":
        phases = (
            PhaseConfig(5.0, 16.0, p1, "wgn"),
            PhaseConfig(-10.0, 5.0, p2, "wgn_impulsive"),
            PhaseConfig(-28.0, -5.0, p3, "composite"),
        )

    if args.snr_phase == "3":
        ph = phases[2]
        ep = p1 + p2 + max(0, p3 // 2)
    elif args.snr_phase == "2":
        ph = phases[1]
        ep = p1 + max(0, p2 // 2)
    else:
        ph = phases[2]
        ep = total_ep - 1

    ds = PDSignalDataset(
        config=master,
        num_samples=args.n_samples,
        mode="train",
        seed=int(tcfg.get("seed", 42)) + 9100,
        total_epochs=total_ep,
    )
    ds.set_training_phase(ep, (ph.snr_min, ph.snr_max), ph.noise_type, total_ep)

    ab = ablation_from(args.model_id)
    mcfg = Config()
    model = create_model(mcfg.model, ab)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(_load_checkpoint_state(ckpt), strict=False)
    model = model.eval().to(device)

    noisy_l, clean_l, pred_l = [], [], []

    rng_gammas = np.linspace(0.0, 1.0, 21)
    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            noisy = item["noisy"].unsqueeze(0).to(device)
            clean = item["clean"].unsqueeze(0).to(device)
            pred_sig, _ = model(noisy)
            noisy_l.append(noisy.cpu())
            clean_l.append(clean.cpu())
            pred_l.append(pred_sig.cpu())

    noisy_b = torch.cat(noisy_l, dim=0)
    clean_b = torch.cat(clean_l, dim=0)
    pred_b = torch.cat(pred_l, dim=0)

    def mean_ncc(p: torch.Tensor, t: torch.Tensor) -> float:
        pn = p.float().flatten(1)
        tn = t.float().flatten(1)
        pn = pn - pn.mean(1, keepdim=True)
        tn = tn - tn.mean(1, keepdim=True)
        num = (pn * tn).sum(1)
        den = torch.sqrt((pn**2).sum(1) * (tn**2).sum(1)) + 1e-8
        return float((num / den).mean().item())

    best_g, best_ncc = 0.5, -1.0
    curve = []
    for g in rng_gammas:
        fused = hybrid_nn_savgol(pred_b, noisy_b, float(g))
        ncc_m = mean_ncc(fused, clean_b)
        curve.append({"gamma_nn": float(g), "ncc_mean": ncc_m})
        if ncc_m > best_ncc:
            best_ncc = ncc_m
            best_g = float(g)

    out = ROOT / "results" / "hybrid_savgol_gamma.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": args.model_id,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "best_gamma_nn": best_g,
        "calibration_ncc_mean_vs_clean": best_ncc,
        "n_calibration_windows": args.n_samples,
        "snr_phase": args.snr_phase,
        "curve_sample": curve[:: max(1, len(curve) // 10)],
        "baseline_pure_nn_ncc": mean_ncc(pred_b, clean_b),
        "baseline_pure_savgol_ncc": mean_ncc(
            torch.stack(
                [
                    torch.tensor(savgol_denoise(noisy_b[i, 0].numpy()), dtype=torch.float32).unsqueeze(0)
                    for i in range(noisy_b.shape[0])
                ]
            ),
            clean_b,
        ),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
