"""
Post-phase evaluation: verify frozen test hashes and aggregate benchmark_runner outputs.

Delegates neural inference + metrics to evaluation.benchmark_runner.run_full_benchmark
(identical checkpoints and model construction as CLI benchmarks).

Outputs: results/eval_{phase}_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Try Model"))

from evaluation.benchmark_runner import SNR_LEVELS, run_full_benchmark  # noqa: E402

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_hashes() -> bool:
    doc = PROJECT_ROOT / "docs" / "TEST_SET_HASHES.md"
    if not doc.exists():
        log.warning("docs/TEST_SET_HASHES.md missing — skipping hash check")
        return True
    text = doc.read_text(encoding="utf-8")
    ok = True
    for name in ("test_synthetic_ae.pt", "test_real_qlin.pt"):
        path = PROJECT_ROOT / "data" / name
        if not path.exists():
            log.warning("Test file missing %s", path)
            continue
        digest = _sha256(path)
        if digest not in text:
            log.error("Hash mismatch %s (%s)", name, digest)
            ok = False
        else:
            log.info("Hash OK %s", name)
    return ok


def _find_ckpts(results_root: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    if not results_root.exists():
        return out
    for d in sorted(results_root.iterdir()):
        ck = d / "checkpoints" / "best.pt"
        if d.is_dir() and ck.exists():
            out.append((d.name, ck))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", default="manual")
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--results", default="results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--hybrid-savgol-gamma",
        type=float,
        default=None,
        help="Forwarded to benchmark_runner.run_full_benchmark (optional blend).",
    )
    parser.add_argument("--quick", action="store_true", help="Evaluate first checkpoint only (debug).")
    args = parser.parse_args(argv)

    if args.verify_hashes and not _verify_hashes():
        return 2

    device_str = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    if args.device != "cuda":
        device_str = args.device
    results_root = PROJECT_ROOT / args.results
    reg_path = PROJECT_ROOT / "config" / "model_registry.yaml"
    reg = yaml.safe_load(reg_path.read_text(encoding="utf-8"))["models"]

    ckpts = _find_ckpts(results_root)
    if args.quick:
        ckpts = ckpts[:1]

    hybrid: float | None = (
        args.hybrid_savgol_gamma if args.hybrid_savgol_gamma is not None else None
    )
    hybrid_path = PROJECT_ROOT / "results" / "hybrid_savgol_gamma.json"
    if hybrid is None and hybrid_path.exists():
        try:
            meta = json.loads(hybrid_path.read_text(encoding="utf-8"))
            g = meta.get("best_gamma_nn")
            if g is not None:
                hybrid = float(g)
                log.info("Loaded hybrid gamma from %s → %s", hybrid_path, hybrid)
        except Exception:
            pass

    rankings: list[dict] = []
    best_mid: str | None = None
    best_real = -1.0
    synth_by_mid: dict[str, dict[int, float | None]] = {}

    for mid, ck in ckpts:
        if mid not in reg:
            log.warning("Skip %s (not in model_registry.yaml)", mid)
            continue
        log.info("Benchmark %s", mid)
        try:
            bench = run_full_benchmark(
                mid,
                str(ck),
                results_root=str(results_root.resolve()),
                device_str=device_str,
                hybrid_savgol_gamma_nn=float(hybrid) if hybrid is not None else None,
            )
        except Exception as e:
            log.exception("Benchmark failed %s: %s", mid, e)
            continue

        rr = bench.get("real_qlin", {})
        synth = bench.get("synthetic", {})
        ps = synth.get("per_snr_ncc") or {}

        synth_ncc_vals: dict[int, float | None] = {}
        for k in SNR_LEVELS:
            block = ps.get(str(k)) or {}
            v = block.get("ncc_mean")
            synth_ncc_vals[k] = float(v) if v is not None else None

        real_ncc = float(rr.get("ncc_mean", 0.0) or 0.0)
        synth_mean = float(synth.get("ncc_mean", 0.0) or 0.0)

        eff = bench.get("efficiency") or {}
        lat = float(eff.get("inference_latency_ms_b1", 0.0) or 0.0)
        params_m = float(eff.get("param_count_M", 0.0) or 0.0)

        row = {
            "model": mid,
            "real_ncc": real_ncc,
            "synth_ncc": synth_mean,
            "latency_ms": lat,
            "params_m": params_m,
            "ncc_at_snr_minus5": synth_ncc_vals.get(-5),
            "ncc_at_snr_minus15": synth_ncc_vals.get(-15),
        }
        for lvl in SNR_LEVELS:
            row[f"synth_ncc_snr_{lvl}"] = synth_ncc_vals.get(lvl)
        rankings.append(row)
        synth_by_mid[mid] = synth_ncc_vals

        if real_ncc > best_real:
            best_real = real_ncc
            best_mid = mid

        log.info(
            "  real_ncc=%.4f synth_ncc=%.4f synth@-5=%s",
            real_ncc,
            synth_mean,
            synth_ncc_vals.get(-5),
        )

    rankings.sort(key=lambda x: x["real_ncc"], reverse=True)

    summary = {
        "phase": args.phase,
        "hybrid_savgol_gamma_nn": hybrid,
        "best_real_ncc": best_real,
        "best_model_id": best_mid,
        "ncc_at_snr_minus5": (
            synth_by_mid.get(best_mid, {}).get(-5) if best_mid else None
        ),
        "model_rankings": rankings,
        "metrics_source": "run_full_benchmark",
    }

    out = results_root / f"eval_{args.phase}_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
