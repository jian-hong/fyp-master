"""
auto_rank.py — Scan all results/*/benchmark_results.json and print
               a ranked leaderboard + brief auto-discussion to terminal.

Usage (inside .venv):
    python auto_rank.py
    python auto_rank.py --results-dir D:/path/to/FYP_MASTER/results
    python auto_rank.py --sort real_ncc          # rank by real Q.Lin NCC (default)
    python auto_rank.py --sort synth_ncc
    python auto_rank.py --sort snr_improvement
    python auto_rank.py --sort efficiency        # NCC-per-million-params
    python auto_rank.py --sort latency           # fastest first

Paste the terminal output back here for further discussion.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# ── .venv self-enforcement ─────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
for _candidate in [_ROOT / ".venv", _ROOT.parent / ".venv"]:
    _VENV_PY = _candidate / "Scripts" / "python.exe"
    if _VENV_PY.exists() and Path(sys.executable).resolve() != _VENV_PY.resolve():
        import subprocess
        sys.exit(subprocess.run([str(_VENV_PY)] + sys.argv, check=False).returncode)
# ──────────────────────────────────────────────────────────────────────────────

# ── ANSI colours (work on Windows 10+ with ANSI enabled) ─────────────────────
R  = "\033[91m"   # red
G  = "\033[92m"   # green
Y  = "\033[93m"   # yellow
B  = "\033[94m"   # blue
M  = "\033[95m"   # magenta
C  = "\033[96m"   # cyan
W  = "\033[97m"   # white
DIM= "\033[2m"
BLD= "\033[1m"
RST= "\033[0m"

MEDAL = ["🥇", "🥈", "🥉"]


def col(text, colour):
    return f"{colour}{text}{RST}"


def load_benchmark(path: Path) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def extract_metrics(data: dict) -> dict:
    synth  = data.get("synthetic", {})
    real   = data.get("real_qlin", {})
    eff    = data.get("efficiency", {})
    params = eff.get("param_count_M", None)
    real_ncc = real.get("ncc_mean", None)
    return {
        "model_id":       data.get("model_id", "unknown"),
        "synth_ncc":      synth.get("ncc_mean",            None),
        "synth_ncc_std":  synth.get("ncc_std",             None),
        "synth_rmse":     synth.get("rmse_mean",           None),
        "snr_improvement":synth.get("snr_improvement_mean",None),
        "dice":           synth.get("dice_mean",           None),
        "fpr":            synth.get("false_positive_rate", None),
        "real_ncc":       real_ncc,
        "real_ncc_std":   real.get("ncc_std",              None),
        "real_rmse":      real.get("rmse_mean",            None),
        "params_M":       params,
        "latency_b1":     eff.get("inference_latency_ms_b1", None),
        "latency_b32":    eff.get("inference_latency_ms_b32",None),
        "efficiency":     (real_ncc / params) if (real_ncc and params and params > 0) else None,
        # per-SNR breakdown
        "per_snr":        synth.get("per_snr_ncc", {}),
    }


def bar(value: float, lo: float = 0.0, hi: float = 1.0, width: int = 20) -> str:
    if value is None:
        return DIM + "─" * width + RST
    frac  = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    filled = int(frac * width)
    colour = G if frac > 0.7 else (Y if frac > 0.4 else R)
    return colour + "█" * filled + DIM + "░" * (width - filled) + RST


def fmt(value, decimals=4, suffix=""):
    if value is None:
        return col("N/A", DIM)
    return f"{value:.{decimals}f}{suffix}"


def rank_models(models: list[dict], sort_key: str) -> list[dict]:
    reverse = sort_key != "latency"
    key_map = {
        "real_ncc":      lambda m: m["real_ncc"]       or -999,
        "synth_ncc":     lambda m: m["synth_ncc"]      or -999,
        "snr_improvement":lambda m: m["snr_improvement"] or -999,
        "efficiency":    lambda m: m["efficiency"]     or -999,
        "latency":       lambda m: m["latency_b1"]     or  999,
    }
    return sorted(models, key=key_map.get(sort_key, key_map["real_ncc"]),
                  reverse=reverse)


def print_separator(char="─", width=100, colour=DIM):
    print(col(char * width, colour))


def auto_discuss(ranked: list[dict], sort_key: str):
    """Generate a brief auto-commentary section."""
    valid = [m for m in ranked if m["synth_ncc"] is not None]
    if not valid:
        print(col("  No benchmark data found — run evaluation/benchmark_runner.py first.", R))
        return

    best     = ranked[0]
    worst    = valid[-1]
    fastest  = min(valid, key=lambda m: m["latency_b1"] or 999)
    most_eff = max(valid, key=lambda m: m["efficiency"]  or -999)
    has_real = [m for m in valid if m["real_ncc"] is not None]

    print(col("\n  📊 AUTO-DISCUSSION", BLD + W))
    print_separator()

    # 1. Winner
    print(f"\n  {col('WINNER:', BLD+G)} {col(best['model_id'], BLD+W)}")
    print(f"  Highest {'real Q.Lin NCC' if sort_key == 'real_ncc' else sort_key} = "
          f"{col(fmt(best.get(sort_key) or best['synth_ncc']), G)}")
    if best["params_M"]:
        print(f"  Parameters: {col(fmt(best['params_M'],2,'M'), C)}  |  "
              f"Latency (batch=1): {col(fmt(best['latency_b1'],2,'ms'), C)}")

    # 2. SNR breakdown for winner
    if best["per_snr"]:
        print(f"\n  {col('SNR breakdown for winner:', BLD+Y)}")
        snr_vals = []
        for snr_level in ["-15", "-10", "-5", "0", "5"]:
            entry = best["per_snr"].get(snr_level, {})
            ncc_v = entry.get("ncc_mean", None)
            snr_vals.append((snr_level, ncc_v))
            tag = ""
            if ncc_v is not None:
                if ncc_v > 0.9: tag = col("  ✓ Excellent", G)
                elif ncc_v > 0.7: tag = col("  ~ Good", Y)
                elif ncc_v > 0.4: tag = col("  ~ Moderate", Y)
                else:             tag = col("  ✗ Weak", R)
            print(f"    SNR {snr_level:>4} dB : {bar(ncc_v, 0, 1, 18)}  {fmt(ncc_v)}{tag}")

        # Commentary on low-SNR performance
        ncc_neg15 = dict(snr_vals).get("-15")
        if ncc_neg15 is not None:
            if ncc_neg15 < 0.15:
                print(col(f"\n  ⚠  At -15 dB the winner scores only {ncc_neg15:.3f} NCC.", Y))
                print(col("     This is the CNN saturation floor. Swin/BiGRU contribution", Y))
                print(col("     should be visible by dropping SNR floor to -25 dB.", Y))
            elif ncc_neg15 > 0.4:
                print(col(f"\n  ✓  Strong low-SNR performance ({ncc_neg15:.3f} at -15 dB).", G))
                print(col("     Model is genuinely extracting structure, not just whitening.", G))

    # 3. Efficiency champion
    print(f"\n  {col('EFFICIENCY CHAMPION:', BLD+C)} {col(most_eff['model_id'], BLD+W)}")
    print(f"  NCC/param ratio: {col(fmt(most_eff['efficiency'],4), C)}  "
          f"({fmt(most_eff['params_M'],2,'M')} params, NCC={fmt(most_eff['synth_ncc'])})")

    # 4. Speed champion
    print(f"\n  {col('SPEED CHAMPION:', BLD+M)} {col(fastest['model_id'], BLD+W)}")
    print(f"  Latency batch=1:  {col(fmt(fastest['latency_b1'],2,'ms'), M)}  |  "
          f"batch=32: {col(fmt(fastest['latency_b32'],2,'ms'), M)}")
    if fastest["latency_b1"] and fastest["latency_b1"] < 10:
        print(col("  ✓ Sub-10ms → real-time capable at 5 MHz sampling", G))

    # 5. Domain gap warning
    if len(has_real) >= 2:
        gaps = [(m["model_id"],
                 abs((m["synth_ncc"] or 0) - (m["real_ncc"] or 0)))
                for m in has_real]
        worst_gap = max(gaps, key=lambda x: x[1])
        best_gap  = min(gaps, key=lambda x: x[1])
        print(f"\n  {col('SYNTHETIC → REAL DOMAIN GAP:', BLD+Y)}")
        print(f"  Largest gap : {col(worst_gap[0], R)}  Δ={worst_gap[1]:.4f}")
        print(f"  Smallest gap: {col(best_gap[0], G)}  Δ={best_gap[1]:.4f}")
        if worst_gap[1] > 0.3:
            print(col("  ⚠  Large domain gap detected on some models.", Y))
            print(col("     WGAN noise augmentation is high-priority.", Y))

    # 6. Models with missing benchmark data
    all_ids = [m["model_id"] for m in ranked]
    missing  = [mid for mid in all_ids if mid not in [m["model_id"] for m in valid]]
    if missing:
        print(f"\n  {col('MISSING BENCHMARK DATA:', BLD+R)}")
        for mid in missing:
            print(f"    {col('✗', R)} {mid}")
        print(col("  → Run: python evaluation/benchmark_runner.py --compare-all", Y))

    # 7. Next action suggestion
    print(f"\n  {col('RECOMMENDED NEXT ACTIONS:', BLD+W)}")
    if len(valid) < 13:
        print(f"  1. {col('Run full benchmark on all 13 models', Y)} — "
              f"only {len(valid)}/13 have data")
    else:
        print(f"  1. {col('All 13 models benchmarked ✓', G)}")
    if not has_real or max(m["real_ncc"] or 0 for m in valid) < 0.5:
        print(f"  2. {col('Real Q.Lin NCC is low — train WGAN augmentation next', Y)}")
    else:
        print(f"  2. {col('Real Q.Lin performance acceptable — proceed to HPO', G)}")

    best_synth_ncc = max((m["synth_ncc"] or 0) for m in valid)
    if best_synth_ncc > 0.90:
        print(f"  3. {col('Models approaching 0.94 ceiling — drop SNR floor to -25dB', Y)}")
    else:
        print(f"  3. {col('Still room on synthetic NCC — continue training epochs', G)}")
    print()


def main():
    # Enable ANSI on Windows
    import os
    if sys.platform == "win32":
        os.system("color")

    parser = argparse.ArgumentParser(description="Auto-rank all trained models")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Path to results/ folder (auto-detected if not given)")
    parser.add_argument("--sort", type=str, default="real_ncc",
                        choices=["real_ncc", "synth_ncc", "snr_improvement",
                                 "efficiency", "latency"],
                        help="Primary sort metric")
    args = parser.parse_args()

    # Auto-detect results directory
    if args.results_dir:
        results_root = Path(args.results_dir)
    else:
        candidates = [
            Path(__file__).parent / "results",
            Path(__file__).parent.parent / "results",
            Path("results"),
        ]
        results_root = next((p for p in candidates if p.exists()), None)
        if results_root is None:
            print(col("ERROR: Could not find results/ directory. "
                      "Use --results-dir to specify it.", R))
            sys.exit(1)

    print(col(f"\n  Results directory: {results_root.resolve()}", DIM))

    # Collect all benchmark JSONs
    all_data = []
    for json_path in sorted(results_root.rglob("benchmark_results.json")):
        data = load_benchmark(json_path)
        if data:
            metrics = extract_metrics(data)
            metrics["_path"] = str(json_path)
            all_data.append(metrics)

    if not all_data:
        print(col(f"\n  No benchmark_results.json files found under {results_root}", R))
        print(col("  Run: python evaluation/benchmark_runner.py --compare-all", Y))
        sys.exit(0)

    ranked = rank_models(all_data, args.sort)

    # ── HEADER ──────────────────────────────────────────────────────────────
    print()
    print_separator("═", 100, BLD + B)
    print(col("  🔬  MR-TAE AE-PD DENOISING — MODEL RANKING", BLD + W))
    print(col(f"  Sorted by: {args.sort.upper()}  |  "
              f"{len(all_data)} model(s) found  |  "
              f"Results: {results_root.name}/", DIM))
    print_separator("═", 100, BLD + B)

    # ── COLUMN HEADERS ───────────────────────────────────────────────────────
    print(f"\n  {'#':<3} {'Model':<28} {'Params':>7} {'Lat(b1)':>8} "
          f"{'NCC-Synth':>10} {'NCC-Real':>10} {'SNR↑dB':>7} "
          f"{'Dice':>6} {'FPR':>6}  {'Efficiency':>10}")
    print_separator("─", 100)

    # ── ROWS ─────────────────────────────────────────────────────────────────
    for i, m in enumerate(ranked):
        rank_label = MEDAL[i] if i < 3 else f"  {i+1}."
        name_col = col(m["model_id"], BLD + W if i == 0 else W)

        # Colour-code NCC values
        def ncc_col(v):
            if v is None: return col("  N/A   ", DIM)
            c = G if v > 0.85 else (Y if v > 0.6 else R)
            return col(f"{v:9.4f}", c)

        snr_c = col(fmt(m["snr_improvement"], 2), G if (m["snr_improvement"] or 0) > 5 else Y)
        eff_c = col(fmt(m["efficiency"], 4), C) if m["efficiency"] else col("  N/A", DIM)
        lat_c = col(fmt(m["latency_b1"], 2, "ms"), M if (m["latency_b1"] or 999) < 10 else W)
        fpr_c = col(fmt(m["fpr"], 3), G if (m["fpr"] or 1) < 0.2 else
                    (Y if (m["fpr"] or 1) < 0.4 else R))

        print(f"  {rank_label:<3} {name_col:<28} "
              f"{fmt(m['params_M'],2,'M'):>7} "
              f"{lat_c:>8} "
              f"{ncc_col(m['synth_ncc'])} "
              f"{ncc_col(m['real_ncc'])} "
              f"{snr_c:>7} "
              f"{fmt(m['dice'],3):>6} "
              f"{fpr_c:>6}  "
              f"{eff_c:>10}")

    print_separator("─", 100)

    # ── MINI SNR-BREAKDOWN TABLE FOR ALL MODELS ──────────────────────────────
    snr_levels = ["-15", "-10", "-5", "0", "5"]
    models_with_snr = [m for m in ranked if m["per_snr"]]
    if models_with_snr:
        print(col("\n  PER-SNR NCC BREAKDOWN (Synthetic Test Set)", BLD + Y))
        print_separator("─", 80)
        header = f"  {'Model':<28}" + "".join(f"  {s:>6}dB" for s in snr_levels)
        print(header)
        print_separator("─", 80)
        for m in models_with_snr:
            row = f"  {m['model_id']:<28}"
            for s in snr_levels:
                v = m["per_snr"].get(s, {}).get("ncc_mean", None)
                if v is None:
                    row += col(f"   {'N/A':>5}", DIM)
                else:
                    c = G if v > 0.85 else (Y if v > 0.5 else R)
                    row += col(f"  {v:7.4f}", c)
            print(row)
        print_separator("─", 80)
        print(col("  Colour: " + col("■ >0.85 excellent", G) + "  " +
                  col("■ 0.50–0.85 moderate", Y) + "  " +
                  col("■ <0.50 weak", R), DIM))

    # ── AUTO DISCUSSION ───────────────────────────────────────────────────────
    auto_discuss(ranked, args.sort)

    # ── FOOTER ───────────────────────────────────────────────────────────────
    print_separator("═", 100, BLD + B)
    print(col("  Paste this output to Claude for further analysis and training decisions.", DIM))
    print_separator("═", 100, BLD + B)
    print()


if __name__ == "__main__":
    main()
