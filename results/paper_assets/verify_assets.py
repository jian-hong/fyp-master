#!/usr/bin/env python3
"""
Verify paper asset bundle: figures (300 DPI), tables, LaTeX build.

Run from anywhere:
    python results/paper_assets/verify_assets.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

ASSET_DIR = Path(__file__).resolve().parent

FIGS = [
    "fig1_system_overview.png",
    "fig2_ablation_results.png",
    "fig2_architecture.png",
    "fig3_waveform_comparison.png",
    "fig4_ablation_ncc_bar.png",
    "fig5_snr_sweep.png",
    "fig6_loss_comparison.png",
    "fig7_efficiency_scatter.png",
]

TABLES = [
    "table1_dataset.tex",
    "table1_dataset.csv",
    "table2_ablation.tex",
    "table2_ablation.csv",
    "table3_comparison.tex",
    "table3_comparison.csv",
]

LATEX_MAIN = ASSET_DIR / "mr_tae_arxiv.tex"
MIN_DPI = 299.0


def check_png(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    if Image is None:
        return True, "ok (PIL missing — skipped DPI check)"
    with Image.open(path) as im:
        dpi = im.info.get("dpi")
        if dpi is None:
            return False, "no DPI metadata"
        dx, dy = dpi[0], dpi[1] if isinstance(dpi, tuple) else (float(dpi), float(dpi))
        if min(dx, dy) < MIN_DPI:
            return False, f"DPI {dx:.1f}x{dy:.1f} < {MIN_DPI}"
        return True, f"DPI {dx:.1f}x{dy:.1f}"


def try_pdflatex() -> tuple[bool, str]:
    if shutil.which("pdflatex") is None:
        return False, "pdflatex not found on PATH"
    tex = LATEX_MAIN
    if not tex.is_file():
        return False, f"missing {tex.name}"
    cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-output-directory={ASSET_DIR}",
        str(tex.name),
    ]
    try:
        r = subprocess.run(cmd, cwd=ASSET_DIR, capture_output=True, text=True, timeout=120)
        ok = r.returncode == 0
        tail = (r.stdout + "\n" + r.stderr)[-2000:] if r.stdout or r.stderr else "(no output)"
        if not ok:
            return False, f"exit {r.returncode}\n{tail}"
        subprocess.run(cmd, cwd=ASSET_DIR, capture_output=True, text=True, timeout=120)
        return True, "pdflatex completed (2 passes)"
    except subprocess.TimeoutExpired:
        return False, "pdflatex timeout"
    except Exception as e:
        return False, str(e)


def try_bibtex() -> tuple[bool, str]:
    if shutil.which("bibtex") is None:
        return False, "bibtex not on PATH (optional)"
    aux = ASSET_DIR / "mr_tae_arxiv.aux"
    if not aux.is_file():
        return False, "no .aux after pdflatex"
    try:
        r = subprocess.run(
            ["bibtex", "mr_tae_arxiv"],
            cwd=ASSET_DIR,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return r.returncode == 0, (r.stdout + r.stderr)[-800:]
    except Exception as e:
        return False, str(e)


def main() -> int:
    print("=== Paper asset verification ===\n")
    figures_tables_ok = True
    latex_ok = True

    print("[Figures]")
    for fn in FIGS:
        p = ASSET_DIR / fn
        ok, msg = check_png(p)
        status = "PASS" if ok else "FAIL"
        if not ok:
            figures_tables_ok = False
        print(f"  {status}  {fn}: {msg}")

    print("\n[Tables]")
    for fn in TABLES:
        p = ASSET_DIR / fn
        ok = p.is_file()
        if not ok:
            figures_tables_ok = False
        print(f"  {'PASS' if ok else 'FAIL'}  {fn}")

    print("\n[LaTeX]")
    if shutil.which("pdflatex") is None:
        print("  SKIP  pdflatex not on PATH (install MiKTeX or TeX Live to enable PDF build)")
        latex_ok = False
    else:
        ok, msg = try_pdflatex()
        if not ok:
            latex_ok = False
            print(f"  FAIL  {msg}")
        else:
            print(f"  PASS  {msg}")
            bib_ok, bib_msg = try_bibtex()
            tag = "PASS" if bib_ok else "WARN"
            short = bib_msg[:400] + ("..." if len(bib_msg) > 400 else "")
            print(f"  {tag}  bibtex: {short}")
            if bib_ok:
                try_pdflatex()
                try_pdflatex()

    pdf_out = ASSET_DIR / "mr_tae_arxiv.pdf"
    pdf_ok = pdf_out.is_file()
    print(f"\n[PDF] {'PASS' if pdf_ok else 'FAIL'}  {pdf_out.name}")

    print("\n=== Summary ===")
    if figures_tables_ok:
        print("Figures and tables: PASS.")
    else:
        print("Figures and tables: FAIL.")
    if latex_ok and pdf_ok:
        print("LaTeX build: PASS.")
    elif not shutil.which("pdflatex"):
        print("LaTeX build: SKIP (no toolchain).")
    else:
        print("LaTeX build: FAIL.")
    return 0 if figures_tables_ok else 1


if __name__ == "__main__":
    sys.exit(main())
