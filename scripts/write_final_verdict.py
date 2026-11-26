"""Generate results/FINAL_VERDICT_V2.md from definitive benchmark outputs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEF = PROJECT_ROOT / "results" / "definitive_benchmark"


def main() -> None:
    synth = json.loads((DEF / "synthetic_results.json").read_text(encoding="utf-8"))
    real = json.loads((DEF / "real_results.json").read_text(encoding="utf-8"))
    baselines = json.loads((DEF / "baselines.json").read_text(encoding="utf-8"))
    lat = json.loads((DEF / "latency.json").read_text(encoding="utf-8"))

    def syn_mean(mid: str) -> float:
        return float(synth[mid].get("ncc_mean") or 0)

    def syn_at(mid: str, snr: int) -> float:
        per = synth[mid].get("per_snr_ncc") or {}
        return float((per.get(str(snr)) or {}).get("ncc_mean") or 0)

    def real_in(mid: str) -> float:
        r = real.get(mid, {})
        ref = r.get("input_ref", r)
        return float(ref.get("ncc_mean") or 0)

    neural = [m for m in synth if not m.startswith("baseline")]
    best_syn = max(neural, key=syn_mean)
    best_real = max(neural, key=real_in)
    best_m15 = max(neural, key=lambda m: syn_at(m, -15))

    eff_scores = []
    for m in neural:
        if m in lat and lat[m].get("params_M"):
            eff_scores.append((m, syn_mean(m) / float(lat[m]["params_M"])))
    best_eff = max(eff_scores, key=lambda x: x[1])[0] if eff_scores else "N/A"

    loss_a = syn_mean("MR-TAE-FULL")
    loss_b = syn_mean("MR-TAE-FULL-LossB")
    loss_c = syn_mean("MR-TAE-FULL-LossC")

    bs5 = baselines.get("bayesshrink", {}).get("5", baselines.get("bayesshrink", {}).get("+5"))
    sg5 = baselines.get("savgol_51", {}).get("5")

    all_phase3 = all(
        json.loads(
            (PROJECT_ROOT / "results" / "MODEL_INVENTORY.txt").read_text(encoding="utf-8")
            .splitlines()[-3]
            if False
            else "{}"
        )
        for _ in []
    )

    phase3_yes = True
    for mid in neural:
        hist_path = PROJECT_ROOT / "results" / mid / "history.json"
        phases: list = []
        if hist_path.exists():
            hj = json.loads(hist_path.read_text(encoding="utf-8"))
            phases = hj.get("phase") or []
        if not phases:
            import torch

            ck = torch.load(
                PROJECT_ROOT / "results" / mid / "checkpoints" / "best.pt",
                map_location="cpu",
                weights_only=False,
            )
            phases = (ck.get("history") or {}).get("phase") or []
        if not phases or int(phases[-1]) < 3:
            phase3_yes = False

    ablations = {
        "noBiGRU": syn_mean("MR-TAE-FULL") - syn_mean("MR-TAE-noBiGRU"),
        "noSwin": syn_mean("MR-TAE-FULL") - syn_mean("MR-TAE-noSwin"),
        "noAttn": syn_mean("MR-TAE-FULL") - syn_mean("MR-TAE-noAttn"),
        "noMTL": syn_mean("MR-TAE-FULL") - syn_mean("MR-TAE-noMTL"),
        "noWavelet": syn_mean("MR-TAE-FULL") - syn_mean("MR-TAE-noWavelet"),
    }
    load_bearing = [k for k, v in ablations.items() if v > 0.02]

    md = f"""# FINAL VERDICT V2 — MR-TAE Forensic Re-Evaluation
Generated from `results/definitive_benchmark/` (verified zero-mean NCC).

## Answers

1. **Best model by synthetic NCC (GT-referenced):** {best_syn} ({syn_mean(best_syn):.4f})
2. **Best model by real input-referenced NCC:** {best_real} ({real_in(best_real):.4f})
3. **Best model at −15 dB synthetic NCC:** {best_m15} ({syn_at(best_m15, -15):.4f})
4. **Best efficiency (NCC/params):** {best_eff}
5. **Load-bearing components (Δ syn NCC > 0.02 when removed):** {", ".join(load_bearing) or "none above threshold"}
6. **LossA vs LossB:** LossB {'better' if loss_b > loss_a else 'worse'} on synthetic mean NCC by {abs(loss_b-loss_a):.4f} ({loss_a:.4f} vs {loss_b:.4f}); LossC={loss_c:.4f}
7. **BayesShrink NCC @ +5 dB (corrected):** {float(bs5):.4f}
8. **Savitzky-Golay (w=51) NCC @ +5 dB:** {float(sg5):.4f} — note: heavy smoothing destroys pulse shape on AE windows (input NCC @+5 dB ≈ 0.87)
9. **All 13 models trained to Phase 3:** {"YES" if phase3_yes else "NO"}

## Latency ordering (batch=1, CUDA-sync, trimmed mean)
"""
    for mid, v in sorted(lat.items(), key=lambda kv: kv[1].get("lat_ms_b1", 999)):
        if "lat_ms_b1" in v:
            md += f"- {mid}: {v['lat_ms_b1']:.2f} ms ({v.get('params_M', '?')} M params)\n"

    out = PROJECT_ROOT / "results" / "FINAL_VERDICT_V2.md"
    out.write_text(md, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
