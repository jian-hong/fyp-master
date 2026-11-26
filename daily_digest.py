"""
pipeline/daily_digest.py
Generates a markdown + JSON digest of campaign state, eval metrics, and figures.

Usage:
    python pipeline/daily_digest.py --state campaign_state.json
    python pipeline/daily_digest.py --state campaign_state.json --format json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIGESTS_DIR = PROJECT_ROOT / "logs" / "digests"
DIGESTS_DIR.mkdir(parents=True, exist_ok=True)

# Traditional baseline references (from agent run, May 2026)
BASELINES = {
    "savgol": {-15: 0.265, -5: 0.819},
    "wavelet": {5: 0.904},
    "bayesshrink": {-5: 0.7},
}


def _phase_status_emoji(status: str) -> str:
    return {"done": "✅", "running": "🔄", "failed": "❌", "skipped": "⏭️", "pending": "⏳"}.get(status, "❓")


def build_digest(state_path: Path) -> dict:
    state = json.loads(state_path.read_text(encoding="utf-8"))

    # Day number
    day_n = 1
    try:
        start = datetime.fromisoformat(state.get("start_iso", ""))
        now = datetime.now(timezone.utc).astimezone()
        day_n = max(1, (now - start).days + 1)
    except Exception:
        pass

    # Collect eval summaries from all phases
    all_metrics: list[dict] = []
    for phase_id in state.get("phases", {}):
        eval_path = PROJECT_ROOT / "results" / f"eval_{phase_id}_summary.json"
        if eval_path.exists():
            try:
                m = json.loads(eval_path.read_text())
                m["_phase"] = phase_id
                all_metrics.append(m)
            except Exception:
                pass

    # Latest model ranking from most recent eval
    best_ncc = state.get("best_real_ncc")
    best_model = state.get("best_model_id")

    # Figure manifest
    figures = state.get("figure_manifest", [])

    # Build markdown
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# 🤖 MR-TAE Campaign Digest — Day {day_n}",
        f"> Generated: {now_str}  |  State: `campaign_state.json`",
        "",
        "---",
        "",
        "## 📊 Campaign Overview",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Campaign day | {day_n} / 15 |",
        f"| Current phase | `{state.get('current_phase_id', 'N/A')}` |",
        f"| Best real-world NCC | `{best_ncc:.4f}` |" if best_ncc else "| Best real-world NCC | Not yet measured |",
        f"| Best model | `{best_model or 'TBD'}` |",
        f"| Figures generated | {len(figures)} |",
        "",
        "---",
        "",
        "## 🔄 Phase Progress",
        "",
    ]

    for pid, pdata in state.get("phases", {}).items():
        status = pdata.get("status", "pending") if isinstance(pdata, dict) else pdata
        emoji = _phase_status_emoji(status)
        ncc = pdata.get("metrics", {}).get("best_real_ncc", None) if isinstance(pdata, dict) else None
        ncc_str = f" | NCC={ncc:.4f}" if ncc else ""
        attempts = pdata.get("attempts", 0) if isinstance(pdata, dict) else 0
        lines.append(f"- {emoji} `{pid}` — **{status}**{ncc_str} (attempts: {attempts})")

    lines += ["", "---", "", "## 🏆 Model Rankings (Latest Eval)", ""]

    if all_metrics:
        latest = all_metrics[-1]
        rankings = latest.get("model_rankings", [])
        if rankings:
            lines += [
                "| Rank | Model | Real NCC | Synth NCC | Latency (ms) | Params (M) |",
                "|------|-------|----------|-----------|--------------|------------|",
            ]
            for i, row in enumerate(rankings[:13], 1):
                lines.append(
                    f"| {i} | `{row.get('model','?')}` | {row.get('real_ncc',0):.4f} | "
                    f"{row.get('synth_ncc',0):.4f} | {row.get('latency_ms',0):.2f} | "
                    f"{row.get('params_m',0):.2f} |"
                )
    else:
        lines.append("*No evaluation complete yet.*")

    lines += ["", "---", "", "## ⚡ vs Traditional Baselines", ""]
    lines += [
        "| SNR (dB) | Best Neural NCC | Savgol NCC | Delta | Status |",
        "|----------|----------------|------------|-------|--------|",
    ]
    snr_levels = [-15, -5, 0, 5]
    neural_by_snr = {}
    if all_metrics:
        latest = all_metrics[-1]
        for snr in snr_levels:
            neural_by_snr[snr] = latest.get(f"ncc_at_snr_{snr}", None)

    for snr in snr_levels:
        neural = neural_by_snr.get(snr)
        savgol = BASELINES.get("savgol", {}).get(snr, None)
        if neural is not None and savgol is not None:
            delta = float(neural) - float(savgol)
            status_str = "✅ Neural wins" if delta > 0 else "❌ Baseline wins"
            lines.append(f"| {snr:+d} | {float(neural):.4f} | {float(savgol):.4f} | {delta:+.4f} | {status_str} |")
        elif neural is not None:
            lines.append(f"| {snr:+d} | {float(neural):.4f} | N/A | N/A | — |")
        else:
            lines.append(f"| {snr:+d} | Not yet evaluated | {savgol or 'N/A'} | — | — |")

    lines += ["", "---", "", f"## 🖼️ Figures ({len(figures)} total)", ""]
    for fig_path in sorted(figures)[:15]:
        lines.append(f"- `{fig_path}`")
    if len(figures) > 15:
        lines.append(f"- ... and {len(figures)-15} more")

    if state.get("error_log"):
        lines += ["", "---", "", "## ⚠️ Errors", ""]
        for e in state["error_log"][-5:]:
            lines.append(f"- [{e.get('ts','')}] `{e.get('phase','?')}`: {e.get('error','?')}")

    lines += [
        "", "---", "",
        "*Auto-generated by `pipeline/daily_digest.py`*  |  "
        f"*Best real NCC: `{f'{best_ncc:.4f}' if best_ncc else 'N/A'}`*",
    ]

    return {
        "markdown": "\n".join(lines),
        "day_n": day_n,
        "best_real_ncc": best_ncc,
        "best_model": best_model,
        "phases": {
            pid: pdata.get("status") if isinstance(pdata, dict) else str(pdata)
            for pid, pdata in state.get("phases", {}).items()
        },
        "figures_count": len(figures),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default="campaign_state.json")
    parser.add_argument("--format", choices=["markdown", "json", "both"], default="both")
    args = parser.parse_args(argv)

    state_path = Path(args.state)
    if not state_path.is_absolute():
        state_path = PROJECT_ROOT / state_path
    if not state_path.exists():
        print(f"[warn] {state_path} not found — nothing to digest.")
        return 0

    result = build_digest(state_path)
    date_str = datetime.now().strftime("%Y-%m-%d_%H%M")

    if args.format in ("markdown", "both"):
        md_path = DIGESTS_DIR / f"digest_{date_str}.md"
        md_path.write_text(result["markdown"], encoding="utf-8")
        print(f"Digest written: {md_path}")
        if args.format == "markdown":
            print(result["markdown"])

    if args.format in ("json", "both"):
        json_path = DIGESTS_DIR / f"digest_{date_str}.json"
        json_path.write_text(json.dumps({k: v for k, v in result.items() if k != "markdown"}, indent=2), encoding="utf-8")
        print(f"JSON written: {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
