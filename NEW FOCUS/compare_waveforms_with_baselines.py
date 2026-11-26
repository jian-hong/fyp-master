from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.signal import savgol_filter

try:
    import pywt
except ImportError:
    pywt = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

TEST_SET = PROJECT_ROOT / "data" / "test_synthetic_ae.pt"
REGISTRY = PROJECT_ROOT / "config" / "model_registry.yaml"
TRAIN_CFG = PROJECT_ROOT / "config" / "training_config.yaml"
RESULTS = PROJECT_ROOT / "results"
FS_HZ = 1_000_000
SNR_LEVELS = (-15, -10, -5, 0, 5)


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else float("nan")


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def snr_out(clean: np.ndarray, pred: np.ndarray) -> float:
    noise = pred - clean
    p_sig = float(np.mean(clean * clean))
    p_noise = float(np.mean(noise * noise))
    return float(10.0 * np.log10(p_sig / max(p_noise, 1e-12)))


def wavelet_denoise(x: np.ndarray, wavelet: str = "db4") -> np.ndarray:
    if pywt is None:
        return x.copy()
    coeffs = pywt.wavedec(x, wavelet, mode="symmetric")
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    out = [coeffs[0]]
    out.extend(pywt.threshold(c, value=uthresh, mode="soft") for c in coeffs[1:])
    rec = pywt.waverec(out, wavelet, mode="symmetric")
    return rec[: len(x)]


def savgol_denoise(x: np.ndarray) -> np.ndarray:
    return savgol_filter(x, window_length=51, polyorder=3, mode="interp")


def load_model(model_id: str, registry: dict, device: torch.device):
    from mr_tae_fusion.config import get_config
    from mr_tae_fusion.models import create_model

    cfg = get_config()
    model_entry = registry.get(model_id, {})
    for flag, val in model_entry.items():
        if hasattr(cfg.model, flag):
            setattr(cfg.model, flag, val)

    model = create_model(cfg.model).to(device).eval()
    ckpt = torch.load(
        RESULTS / model_id / "checkpoints" / "best.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    return model


def main() -> int:
    with REGISTRY.open() as f:
        registry = (yaml.safe_load(f) or {}).get("models", {})
    model_ids = [m for m in registry if (RESULTS / m / "checkpoints" / "best.pt").exists()]

    ds = torch.load(TEST_SET, map_location="cpu", weights_only=False)
    snr_labels = ds.get("snr_labels", ds.get("snr_db")).cpu().numpy().ravel()
    noisy_all = ds["noisy"]
    clean_all = ds["clean"]

    device = torch.device("cpu")
    rows = []
    for snr in SNR_LEVELS:
        idx = int(np.where(snr_labels == snr)[0][0])
        noisy = noisy_all[idx].squeeze().numpy()
        clean = clean_all[idx].squeeze().numpy()

        b_wave = wavelet_denoise(noisy)
        b_sg = savgol_denoise(noisy)
        baseline_rows = [
            {"name": "wavelet", "pred": b_wave},
            {"name": "savgol", "pred": b_sg},
        ]
        for b in baseline_rows:
            b["ncc"] = ncc(clean, b["pred"])
            b["rmse"] = rmse(clean, b["pred"])
            b["snr_out"] = snr_out(clean, b["pred"])
        best_baseline = max(baseline_rows, key=lambda x: x["ncc"])

        model_rows = []
        x = torch.from_numpy(noisy).float().view(1, 1, -1).to(device)
        for mid in model_ids:
            model = load_model(mid, registry, device)
            with torch.no_grad():
                out = model(x)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                pred = out.squeeze().cpu().numpy()
            model_rows.append(
                {"name": mid, "pred": pred, "ncc": ncc(clean, pred), "rmse": rmse(clean, pred), "snr_out": snr_out(clean, pred)}
            )
            del model
        best_model = max(model_rows, key=lambda x: x["ncc"])

        t_us = np.arange(clean.shape[-1]) / FS_HZ * 1e6
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t_us, clean, label="Clean GT", linewidth=1.4)
        ax.plot(t_us, noisy, label="Noisy", alpha=0.55)
        ax.plot(t_us, best_model["pred"], label=f"Best model: {best_model['name']}", linewidth=1.2)
        ax.plot(t_us, best_baseline["pred"], label=f"Best baseline: {best_baseline['name']}", linewidth=1.1)
        ax.set_title(f"SNR {snr} dB | model NCC={best_model['ncc']:.3f} vs baseline NCC={best_baseline['ncc']:.3f}")
        ax.set_xlabel("Time (us)")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right")
        out_fig = RESULTS / f"waveform_model_vs_baseline_SNR{snr:+d}dB.png"
        fig.tight_layout()
        fig.savefig(out_fig, dpi=250)
        plt.close(fig)

        rows.append(
            {
                "snr_in_db": snr,
                "best_model": {k: v for k, v in best_model.items() if k != "pred"},
                "best_baseline": {k: v for k, v in best_baseline.items() if k != "pred"},
                "delta_ncc_model_minus_baseline": best_model["ncc"] - best_baseline["ncc"],
                "delta_rmse_model_minus_baseline": best_model["rmse"] - best_baseline["rmse"],
                "delta_snr_out_db_model_minus_baseline": best_model["snr_out"] - best_baseline["snr_out"],
                "figure": str(out_fig.relative_to(PROJECT_ROOT)),
            }
        )

    out_json = RESULTS / "waveform_baseline_comparison.json"
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
