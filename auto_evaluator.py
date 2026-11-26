"""
evaluation/auto_evaluator.py
Post-phase evaluation runner. Called by autonomous_loop.py after each phase.

- Verifies test set SHA256 hashes
- Benchmarks all available checkpoints on both test sets
- Computes NCC at each SNR level vs traditional baselines
- Writes results/eval_{phase_id}_summary.json
- Optionally triggers figure generation

Usage:
    python evaluation/auto_evaluator.py --phase phase_A_train
    python evaluation/auto_evaluator.py --phase phase_B_hpo --verify-hashes
    python evaluation/auto_evaluator.py --quick  # just best model
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"

_VENV_PYTHON = (
    PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if (PROJECT_ROOT / ".venv" / "Scripts" / "python.exe").exists()
    else PROJECT_ROOT / ".venv" / "bin" / "python"
)
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

TRADITIONAL_BASELINES = {
    "savgol": {-15: 0.265, -10: None, -5: 0.819, 0: None, 5: None},
    "wavelet": {-15: None, -10: None, -5: None, 0: None, 5: 0.904},
}

SNR_LEVELS = [-15, -10, -5, 0, 5]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_hashes() -> bool:
    hash_doc = DOCS_DIR / "TEST_SET_HASHES.md"
    if not hash_doc.exists():
        log.warning("TEST_SET_HASHES.md not found — skipping verification")
        return True
    text = hash_doc.read_text()
    ok = True
    for name in ["test_synthetic_ae.pt", "test_real_qlin.pt"]:
        path = DATA_DIR / name
        if not path.exists():
            log.warning(f"Test set not found: {path}")
            continue
        actual = _sha256(path)
        if actual not in text:
            log.error(f"HASH MISMATCH for {name}! Actual: {actual}")
            ok = False
        else:
            log.info(f"Hash OK: {name}")
    return ok


def _ncc_1d(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.flatten().astype(np.float64)
    target = target.flatten().astype(np.float64)
    pred -= pred.mean()
    target -= target.mean()
    denom = float(np.sqrt((pred * pred).sum() * (target * target).sum()))
    return float((pred * target).sum() / denom) if denom > 1e-10 else 0.0


def _load_model(model_name: str, ckpt_path: Path, device: torch.device):
    """Try several checkpoint formats used by the project trainer."""
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Format 1: nn.Module saved directly
    if isinstance(payload, torch.nn.Module):
        return payload.eval().to(device)

    # Format 2: {"model": nn.Module}
    if "model" in payload and isinstance(payload["model"], torch.nn.Module):
        return payload["model"].eval().to(device)

    # Format 3: {"model_state": state_dict, "model_id": ..., "ablation_config": ...}
    if "model_state" in payload or "state_dict" in payload:
        state_dict = payload.get("model_state") or payload.get("state_dict")
        try:
            cfg_path = PROJECT_ROOT / "config" / "training_config.yaml"
            registry_path = PROJECT_ROOT / "config" / "model_registry.yaml"
            import yaml
            with registry_path.open() as f:
                registry = yaml.safe_load(f)
            with cfg_path.open() as f:
                train_cfg = yaml.safe_load(f)
            from models.base import build_model_from_registry  # type: ignore
            model = build_model_from_registry(model_name, registry, train_cfg)
            model.load_state_dict(state_dict, strict=False)
            return model.eval().to(device)
        except ImportError:
            # Try mr_tae_fusion path
            try:
                from mr_tae_fusion.models import build_model_from_name  # type: ignore
                model = build_model_from_name(model_name)
                model.load_state_dict(state_dict, strict=False)
                return model.eval().to(device)
            except Exception as e:
                log.warning(f"Could not load {model_name} state_dict: {e}")
                return None

    log.warning(f"Unrecognised checkpoint format for {model_name}")
    return None


def _benchmark_model(model, test_set: dict, device: torch.device) -> dict:
    """Return per-SNR NCC and overall metrics for one model."""
    noisy_all = test_set.get("noisy")
    clean_all = test_set.get("clean")
    snr_labels = test_set.get("snr_db") or test_set.get("snr_labels")

    if noisy_all is None or clean_all is None:
        return {}

    results_by_snr: dict[int, list[float]] = {s: [] for s in SNR_LEVELS}
    all_ncc: list[float] = []

    batch_size = 64
    n = noisy_all.shape[0]

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            noisy_b = noisy_all[start:end].float().to(device)
            clean_b = clean_all[start:end].cpu().numpy()

            out = model(noisy_b)
            if isinstance(out, (tuple, list)):
                out = out[0]
            elif isinstance(out, dict):
                out = out.get("denoised") or next(iter(out.values()))
            pred = out.squeeze(1).cpu().numpy()

            for i in range(end - start):
                ncc = _ncc_1d(pred[i], clean_b[i])
                all_ncc.append(ncc)
                if snr_labels is not None:
                    snr = int(snr_labels[start + i].item()) if hasattr(snr_labels[start + i], "item") else int(snr_labels[start + i])
                    if snr in results_by_snr:
                        results_by_snr[snr].append(ncc)

    metrics = {
        "real_ncc": float(np.mean(all_ncc)) if all_ncc else 0.0,
        "real_ncc_std": float(np.std(all_ncc)) if all_ncc else 0.0,
    }
    for snr in SNR_LEVELS:
        vals = results_by_snr.get(snr, [])
        metrics[f"ncc_snr_{snr}"] = float(np.mean(vals)) if vals else None

    return metrics


def _find_checkpoints() -> dict[str, Path]:
    """Return dict of model_name -> best checkpoint path."""
    checkpoints: dict[str, Path] = {}
    for candidate in [
        "results/*/checkpoints/best.pt",
        "results/*/best.pt",
        "results/*/best.pth",
        "results/*/checkpoints/best.pth",
    ]:
        for p in PROJECT_ROOT.glob(candidate):
            model_name = p.parents[1].name if "checkpoints" in p.parts else p.parent.name
            if model_name not in checkpoints:
                checkpoints[model_name] = p
    return checkpoints


def _measure_latency(model, device: torch.device, n_warmup=10, n_bench=100) -> float:
    """Measure single-window latency in ms."""
    dummy = torch.randn(1, 1, 2048).to(device)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_bench):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    return elapsed / n_bench * 1000


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", default="manual", help="Phase ID for output naming")
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Benchmark best model only")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    if args.verify_hashes:
        if not _verify_hashes():
            log.error("Hash verification failed — aborting evaluation")
            return 2

    device = torch.device(args.device)
    checkpoints = _find_checkpoints()
    log.info(f"Found {len(checkpoints)} model checkpoints")

    # Load real test set
    real_path = PROJECT_ROOT / "data" / "test_real_qlin.pt"
    synth_path = PROJECT_ROOT / "data" / "test_synthetic_ae.pt"

    real_set = {}
    synth_set = {}
    if real_path.exists():
        real_set = torch.load(real_path, map_location="cpu", weights_only=False)
        log.info(f"Real test set loaded: {real_path}")
    if synth_path.exists():
        synth_set = torch.load(synth_path, map_location="cpu", weights_only=False)
        log.info(f"Synthetic test set loaded: {synth_path}")

    if not real_set and not synth_set:
        log.error("No test sets found. Cannot evaluate.")
        return 3

    model_rankings: list[dict] = []
    best_real_ncc = 0.0
    best_model_id = None
    ncc_at_snr_by_model: dict[str, dict] = {}

    if args.quick and checkpoints:
        # Only benchmark the latest by modification time
        checkpoints = dict(sorted(checkpoints.items(), key=lambda x: x[1].stat().st_mtime, reverse=True)[:1])

    for model_name, ckpt_path in sorted(checkpoints.items()):
        log.info(f"Evaluating {model_name} from {ckpt_path}")
        try:
            model = _load_model(model_name, ckpt_path, device)
        except Exception as e:
            log.warning(f"Failed to load {model_name}: {e}")
            continue
        if model is None:
            continue

        real_metrics = _benchmark_model(model, real_set, device) if real_set else {}
        synth_metrics = _benchmark_model(model, synth_set, device) if synth_set else {}
        latency = _measure_latency(model, device)

        n_params = sum(p.numel() for p in model.parameters()) / 1e6

        real_ncc = real_metrics.get("real_ncc", 0.0) or 0.0
        if real_ncc > best_real_ncc:
            best_real_ncc = real_ncc
            best_model_id = model_name

        row = {
            "model": model_name,
            "real_ncc": real_ncc,
            "synth_ncc": synth_metrics.get("real_ncc", 0.0),
            "latency_ms": latency,
            "params_m": n_params,
        }
        for snr in SNR_LEVELS:
            row[f"real_ncc_snr_{snr}"] = real_metrics.get(f"ncc_snr_{snr}")
            row[f"synth_ncc_snr_{snr}"] = synth_metrics.get(f"ncc_snr_{snr}")
        model_rankings.append(row)
        ncc_at_snr_by_model[model_name] = {snr: real_metrics.get(f"ncc_snr_{snr}") for snr in SNR_LEVELS}

        log.info(f"  Real NCC={real_ncc:.4f}  Synth NCC={synth_metrics.get('real_ncc',0):.4f}  Latency={latency:.2f}ms")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Sort by real NCC
    model_rankings.sort(key=lambda x: x["real_ncc"], reverse=True)

    # Compute vs-baseline deltas (best model)
    vs_baseline: dict[str, dict] = {}
    if best_model_id and best_model_id in ncc_at_snr_by_model:
        for baseline_name, snr_vals in TRADITIONAL_BASELINES.items():
            vs_baseline[baseline_name] = {}
            for snr, baseline_ncc in snr_vals.items():
                if baseline_ncc is None:
                    continue
                neural_ncc = ncc_at_snr_by_model[best_model_id].get(snr)
                if neural_ncc is not None:
                    vs_baseline[baseline_name][snr] = {
                        "neural": float(neural_ncc),
                        "baseline": float(baseline_ncc),
                        "delta": float(neural_ncc) - float(baseline_ncc),
                        "neural_wins": float(neural_ncc) > float(baseline_ncc),
                    }

    summary = {
        "phase": args.phase,
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "models_evaluated": len(model_rankings),
        "best_real_ncc": best_real_ncc,
        "best_model_id": best_model_id,
        "ncc_at_snr_minus5": ncc_at_snr_by_model.get(best_model_id, {}).get(-5) if best_model_id else None,
        "ncc_at_snr_minus15": ncc_at_snr_by_model.get(best_model_id, {}).get(-15) if best_model_id else None,
        "model_rankings": model_rankings,
        "vs_traditional_baselines": vs_baseline,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"eval_{args.phase}_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    log.info(f"Eval summary written: {out_path}")

    # Print a quick table
    print(f"\n{'Model':<35} {'Real NCC':>9} {'Synth NCC':>10} {'Lat(ms)':>8} {'Params(M)':>10}")
    print("-" * 75)
    for row in model_rankings[:15]:
        print(f"{row['model']:<35} {row['real_ncc']:>9.4f} {row['synth_ncc']:>10.4f} {row['latency_ms']:>8.2f} {row['params_m']:>10.2f}")

    if vs_baseline:
        print("\nVs Baselines (best model at available SNRs):")
        for bl, snr_rows in vs_baseline.items():
            for snr, v in snr_rows.items():
                win = "✅ WIN" if v["neural_wins"] else "❌ LOSE"
                print(f"  vs {bl} @ SNR {snr:+d}dB: neural={v['neural']:.4f}  baseline={v['baseline']:.4f}  Δ={v['delta']:+.4f}  {win}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
