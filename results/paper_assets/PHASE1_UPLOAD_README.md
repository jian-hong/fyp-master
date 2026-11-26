# Phase-1 upload bundle (ARXIV_PAPER_HANDOFF §1)

All items from `NEW FOCUS/ARXIV_PAPER_HANDOFF.md` lines 27–67 are gathered here.

| Handoff item | Location in this repo |
|--------------|------------------------|
| FYP report PDF | `phase1_handoff/OoJianHong-22004678-Report.pdf` **or** `phase1_handoff/MISSING_OoJianHong-22004678-Report.txt` if absent |
| Architecture summary | `phase1_handoff/ARCHITECTURE_SUMMARY.md` |
| Merged benchmarks | `phase1_handoff/benchmark_results.json` and mirror `benchmark_results.json` (same folder as this README) |
| Results comparison | `phase1_handoff/RESULTS_COMPARISON.md` |
| Final verdict | `phase1_handoff/FINAL_VERDICT.md` |
| Research plan, Q&A, agent plan, audit, status | `phase1_handoff/*.md` |
| Figures (§1.5) | `phase1_handoff/figures/*.png` |

**Refresh bundle** after new training or docs change:

```text
python scripts/sync_phase1_handoff_assets.py
python scripts/generate_paper_assets.py
python scripts/sync_phase1_handoff_assets.py
```

Source files originally lived under `docs/`, project root, and `results/benchmark/`; the sync script copies them into `phase1_handoff/` so you can zip `results/paper_assets/phase1_handoff/` for upload.
