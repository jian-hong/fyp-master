"""Task 6 — Verify curriculum phase completion from checkpoint/history."""
from __future__ import annotations

import json
import pathlib
import sys

import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

MODELS = [
    "MR-TAE-FULL",
    "MR-TAE-noBiGRU",
    "MR-TAE-noSwin",
    "MR-TAE-noAttn",
    "MR-TAE-noMTL",
    "MR-TAE-noWavelet",
    "MR-TAE-noSwin-noBiGRU",
    "MR-TAE-noSwin-noAttn",
    "MR-TAE-noBiGRU-noAttn",
    "MR-TAE-FULL-LossB",
    "MR-TAE-FULL-LossC",
    "MWCNN-BiGRU",
    "UNet-BiGRU-Swin",
]


def _phase_info(mid: str) -> dict:
    ckpt_path = PROJECT_ROOT / "results" / mid / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        return {"mid": mid, "missing": True}
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    epoch = ckpt.get("epoch", -1)
    hist = ckpt.get("history") or {}
    phases = hist.get("phase") or []
    val_ncc = hist.get("val_ncc") or []
    final_phase = phases[-1] if phases else ckpt.get("phase", ckpt.get("curriculum_phase", -1))
    total_epochs = len(phases) if phases else -1

    hist_path = PROJECT_ROOT / "results" / mid / "history.json"
    if hist_path.exists():
        hj = json.loads(hist_path.read_text(encoding="utf-8"))
        if hj.get("phase"):
            total_epochs = max(total_epochs, len(hj["phase"]))
            final_phase = hj["phase"][-1]

    return {
        "mid": mid,
        "missing": False,
        "best_epoch": epoch,
        "total_epochs": total_epochs,
        "final_phase": final_phase,
        "best_val_ncc": ckpt.get("best_val_ncc", val_ncc[epoch - 1] if val_ncc and epoch > 0 else "N/A"),
        "val_ncc_last": val_ncc[-1] if val_ncc else "N/A",
    }


def main() -> None:
    for mid in MODELS:
        info = _phase_info(mid)
        if info.get("missing"):
            print(f"[MISSING] {mid}")
            continue
        incomplete = info["total_epochs"] < 85 or info["final_phase"] < 3
        flag = "[INCOMPLETE]" if incomplete else "[OK]"
        print(
            f"{flag} {mid:35s}: best_epoch={info['best_epoch']}, "
            f"total_epochs={info['total_epochs']}, final_phase={info['final_phase']}, "
            f"best_val_ncc={info['best_val_ncc']}"
        )


if __name__ == "__main__":
    main()
