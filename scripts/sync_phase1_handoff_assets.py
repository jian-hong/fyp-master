"""
Copy ARXIV_PAPER_HANDOFF.md §1 upload bundle into results/paper_assets/phase1_handoff/.

Run:  python scripts/sync_phase1_handoff_assets.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "results" / "paper_assets" / "phase1_handoff"
FIG_DEST = DEST / "figures"
BENCH = ROOT / "results" / "benchmark"

# (source relative to ROOT, dest name under phase1_handoff/)
MD_FILES: list[tuple[str, str]] = [
    ("docs/ARCHITECTURE_SUMMARY.md", "ARCHITECTURE_SUMMARY.md"),
    ("RESULTS_COMPARISON.md", "RESULTS_COMPARISON.md"),
    ("docs/FINAL_VERDICT.md", "FINAL_VERDICT.md"),
    ("FULL_RESEARCH_PLAN.md", "FULL_RESEARCH_PLAN.md"),
    ("docs/EXAMINER_QA.md", "EXAMINER_QA.md"),
    ("AGENT_PLAN.md", "AGENT_PLAN.md"),
    ("docs/CODEBASE_AUDIT.md", "CODEBASE_AUDIT.md"),
    ("PROJECT_STATUS.md", "PROJECT_STATUS.md"),
]

REPORT_PDF = ROOT / "OoJianHong-22004678-Report.pdf"

# Handoff §1.5 figure names -> (source path, dest filename)
FIGURE_MAP: list[tuple[Path, str]] = [
    (ROOT / "results" / "paper_assets" / "fig3_waveform_comparison.png", "waveform_comparison_3snr.png"),
    (ROOT / "results" / "paper_assets" / "fig4_ablation_ncc_bar.png", "ncc_bar_13models.png"),
    (ROOT / "results" / "paper_assets" / "fig2_architecture.png", "architecture_diagram.png"),
    (BENCH / "ablation_heatmap.png", "ablation_heatmap.png"),
    (BENCH / "snr_sweep.png", "snr_sweep.png"),
    (BENCH / "training_curves.png", "training_curves.png"),
]


def merge_benchmarks() -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for p in sorted((ROOT / "results").glob("*/benchmark_results.json")):
        if p.parent.name in ("benchmark", "paper_assets"):
            continue
        try:
            merged[p.parent.name] = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            merged[p.parent.name] = {"_error": str(exc)}
    return merged


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    FIG_DEST.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []

    if REPORT_PDF.is_file():
        shutil.copy2(REPORT_PDF, DEST / REPORT_PDF.name)
        log_lines.append(f"OK  {REPORT_PDF.name}")
    else:
        (DEST / "MISSING_OoJianHong-22004678-Report.txt").write_text(
            "Expected FYP report PDF at:\n"
            f"  {REPORT_PDF}\n\n"
            "Add the file there and re-run:\n"
            "  python scripts/sync_phase1_handoff_assets.py\n",
            encoding="utf-8",
        )
        log_lines.append("MISSING OoJianHong-22004678-Report.pdf (see MISSING_*.txt)")

    for rel, name in MD_FILES:
        src = ROOT / Path(rel)
        if not src.is_file() and name == "RESULTS_COMPARISON.md":
            src = ROOT / "docs" / "RESULTS_COMPARISON.md"
        if not src.is_file():
            log_lines.append(f"SKIP (not found) {rel}")
            continue
        shutil.copy2(src, DEST / name)
        log_lines.append(f"OK  {name} <= {src.relative_to(ROOT)}")

    out_json = DEST / "benchmark_results.json"
    merged = merge_benchmarks()
    out_json.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    log_lines.append(f"OK  benchmark_results.json ({len(merged)} model folders)")

    root_mirror = ROOT / "results" / "paper_assets" / "benchmark_results.json"
    root_mirror.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    log_lines.append(f"OK  (mirror) {root_mirror.relative_to(ROOT)}")

    handoff_src = ROOT / "NEW FOCUS" / "ARXIV_PAPER_HANDOFF.md"
    if handoff_src.is_file():
        shutil.copy2(handoff_src, DEST / "ARXIV_PAPER_HANDOFF.md")
        log_lines.append("OK  ARXIV_PAPER_HANDOFF.md")

    for src, fname in FIGURE_MAP:
        dst = FIG_DEST / fname
        if not src.is_file():
            log_lines.append(f"MISSING figure source: {src}")
            continue
        shutil.copy2(src, dst)
        log_lines.append(f"OK  figures/{fname}")

    (DEST / "MANIFEST.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    print("phase1_handoff synced ->", DEST)
    for line in log_lines:
        print(" ", line)


if __name__ == "__main__":
    main()
