"""
Definitive benchmark — runs after pipeline fixes are confirmed.

Produces results/definitive_benchmark/:
  synthetic_results.json, real_results.json, latency.json,
  baselines.json, summary_table.csv, and figure PNGs.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Try Model"))

from evaluation.benchmark_runner import SNR_LEVELS, run_full_benchmark  # noqa: E402
from evaluation.classical_baselines import run_baselines  # noqa: E402
from evaluation.latency_benchmark import MODELS, run_all as run_latency  # noqa: E402

OUT_DIR = PROJECT_ROOT / "results" / "definitive_benchmark"
AUDIT_LOG = PROJECT_ROOT / "results" / "AUDIT_LOG.txt"

EXPECTED_ATTN = {
    True: [
        "MR-TAE-FULL",
        "MR-TAE-noBiGRU",
        "MR-TAE-noSwin",
        "MR-TAE-noMTL",
        "MR-TAE-noWavelet",
        "MR-TAE-FULL-LossB",
        "MR-TAE-FULL-LossC",
        "MR-TAE-noSwin-noBiGRU",
    ],
    False: [
        "MR-TAE-noAttn",
        "MR-TAE-noSwin-noAttn",
        "MR-TAE-noBiGRU-noAttn",
        "MWCNN-BiGRU",
        "UNet-BiGRU-Swin",
    ],
}


def _log(msg: str) -> None:
    line = f"[{datetime.now().isoformat()}] {msg}"
    print(line)
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _training_status(mid: str) -> Dict[str, Any]:
    import torch

    ck_path = PROJECT_ROOT / "results" / mid / "checkpoints" / "best.pt"
    if not ck_path.exists():
        return {"training_completed": False, "best_epoch": -1, "final_phase": -1}
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    hist = ck.get("history") or {}
    phases = hist.get("phase") or []
    total = len(phases)
    final_phase = phases[-1] if phases else -1
    hist_path = PROJECT_ROOT / "results" / mid / "history.json"
    if hist_path.exists():
        hj = json.loads(hist_path.read_text(encoding="utf-8"))
        if hj.get("phase"):
            total = max(total, len(hj["phase"]))
            final_phase = hj["phase"][-1]
    return {
        "training_completed": total >= 85 and final_phase >= 3,
        "best_epoch": int(ck.get("epoch", -1)),
        "final_phase": int(final_phase),
        "total_epochs": int(total),
    }


def verify_yaml_registry() -> List[str]:
    path = PROJECT_ROOT / "config" / "model_registry.yaml"
    reg = yaml.safe_load(path.read_text(encoding="utf-8"))["models"]
    mismatches = []
    for flag, ids in EXPECTED_ATTN.items():
        for mid in ids:
            actual = reg.get(mid, {}).get("use_attention_gates")
            if actual != flag:
                mismatches.append(f"{mid}: expected {flag}, got {actual}")
    if mismatches:
        _log("YAML attention gate mismatches: " + "; ".join(mismatches))
    else:
        _log("YAML attention gates: all 13 models match ARCHITECTURE_SUMMARY.md")
    return mismatches


def run_neural_benchmarks(device: str = "cuda") -> Dict[str, Dict]:
    synth_out: Dict[str, Any] = {}
    real_out: Dict[str, Any] = {}
    for mid in MODELS:
        ck = PROJECT_ROOT / "results" / mid / "checkpoints" / "best.pt"
        if not ck.exists():
            _log(f"SKIP {mid}: no checkpoint")
            continue
        _log(f"Benchmarking {mid}...")
        try:
            bench = run_full_benchmark(
                mid,
                str(ck),
                results_root=str(PROJECT_ROOT / "results"),
                device_str=device,
            )
        except Exception as e:
            _log(f"FAIL {mid}: {e}")
            continue
        synth_out[mid] = bench.get("synthetic", {})
        real_out[mid] = {
            "input_ref": bench.get("real_qlin", {}),
            "metric_type": "input_ref_ncc",
            "note": "NCC vs noisy input — not denoising quality",
        }
        if bench.get("real_qlin_grounded"):
            real_out[mid]["grounded_gt"] = bench["real_qlin_grounded"]
    return {"synthetic": synth_out, "real": real_out}


def build_summary_table(
    synth: Dict[str, Any],
    real: Dict[str, Any],
    latency: Dict[str, Any],
    baselines: Dict[str, Any],
) -> None:
    path = OUT_DIR / "summary_table.csv"
    cols = [
        "model_id",
        "params_M",
        "lat_ms_b1",
        "lat_ms_b32",
        "syn_ncc_mean",
        "syn_ncc_minus15",
        "syn_ncc_minus10",
        "syn_ncc_minus5",
        "syn_ncc_0",
        "syn_ncc_plus5",
        "syn_rmse_mean",
        "syn_snr_improvement_mean",
        "real_inputref_ncc_mean",
        "real_inputref_ncc_std",
        "dice_mean",
        "fpr",
        "training_completed",
        "best_epoch",
        "final_phase",
    ]
    rows: List[Dict[str, Any]] = []
    for mid in MODELS:
        s = synth.get(mid, {})
        r = real.get(mid, {}).get("input_ref", real.get(mid, {}))
        lat = latency.get(mid, {})
        tr = _training_status(mid)
        per = s.get("per_snr_ncc") or {}
        rows.append(
            {
                "model_id": mid,
                "params_M": lat.get("params_M", s.get("param_count_M")),
                "lat_ms_b1": lat.get("lat_ms_b1"),
                "lat_ms_b32": lat.get("lat_ms_b32"),
                "syn_ncc_mean": s.get("ncc_mean"),
                "syn_ncc_minus15": (per.get("-15") or {}).get("ncc_mean"),
                "syn_ncc_minus10": (per.get("-10") or {}).get("ncc_mean"),
                "syn_ncc_minus5": (per.get("-5") or {}).get("ncc_mean"),
                "syn_ncc_0": (per.get("0") or {}).get("ncc_mean"),
                "syn_ncc_plus5": (per.get("5") or {}).get("ncc_mean"),
                "syn_rmse_mean": s.get("rmse_mean"),
                "syn_snr_improvement_mean": s.get("snr_improvement_mean"),
                "real_inputref_ncc_mean": r.get("ncc_mean"),
                "real_inputref_ncc_std": r.get("ncc_std"),
                "dice_mean": s.get("dice_mean"),
                "fpr": s.get("false_positive_rate"),
                "training_completed": tr["training_completed"],
                "best_epoch": tr["best_epoch"],
                "final_phase": tr["final_phase"],
            }
        )
    for bname, snr_dict in baselines.items():
        rows.append(
            {
                "model_id": f"baseline_{bname}",
                "syn_ncc_mean": float(
                    sum(snr_dict.values()) / max(len(snr_dict), 1)
                ),
                "syn_ncc_minus15": snr_dict.get("-15"),
                "syn_ncc_minus10": snr_dict.get("-10"),
                "syn_ncc_minus5": snr_dict.get("-5"),
                "syn_ncc_0": snr_dict.get("0"),
                "syn_ncc_plus5": snr_dict.get("5"),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    _log(f"Wrote {path}")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _log("=== Definitive benchmark run started ===")

    verify_yaml_registry()

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    _log(f"Device: {device}")

    baselines = run_baselines()
    latency = run_latency(device=device)

    neural = run_neural_benchmarks(device=device)
    (OUT_DIR / "synthetic_results.json").write_text(
        json.dumps(neural["synthetic"], indent=2), encoding="utf-8"
    )
    (OUT_DIR / "real_results.json").write_text(
        json.dumps(neural["real"], indent=2), encoding="utf-8"
    )

    build_summary_table(neural["synthetic"], neural["real"], latency, baselines)

    from evaluation.generate_definitive_figures import generate_all  # noqa: E402

    generate_all(OUT_DIR)
    _log("=== Definitive benchmark run complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
