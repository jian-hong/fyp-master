"""Task 1 — Model checkpoint inventory."""
from __future__ import annotations

import pathlib
import sys
from datetime import datetime

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


def main() -> None:
    root = PROJECT_ROOT / "results"
    rows: list[dict] = []
    for mid in MODELS:
        ckpt_path = root / mid / "checkpoints" / "best.pt"
        row: dict = {"model_id": mid}
        if not ckpt_path.exists():
            row.update(
                {
                    "exists": False,
                    "epoch": "-",
                    "phase": "-",
                    "n_tensors": "-",
                    "n_params": "-",
                    "state_key": "-",
                    "val_ncc": "-",
                    "status": "MISSING",
                }
            )
        else:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            epoch = ckpt.get("epoch", ckpt.get("best_epoch", -1))
            hist = ckpt.get("history") or {}
            phases = hist.get("phase") or []
            phase = phases[-1] if phases else ckpt.get("phase", ckpt.get("curriculum_phase", -1))
            total_epochs = len(phases) if phases else -1
            hist_path = PROJECT_ROOT / "results" / mid / "history.json"
            if hist_path.exists():
                import json

                hj = json.loads(hist_path.read_text(encoding="utf-8"))
                if hj.get("phase"):
                    total_epochs = max(total_epochs, len(hj["phase"]))
                    phase = hj["phase"][-1]
            if "model_state" in ckpt:
                sk = "model_state"
                state = ckpt["model_state"]
            elif "model_state_dict" in ckpt:
                sk = "model_state_dict"
                state = ckpt["model_state_dict"]
            else:
                sk = "UNKNOWN"
                state = {}
            n_tensors = len(state)
            n_params = sum(v.numel() for v in state.values()) if state else 0
            incomplete = (
                isinstance(total_epochs, (int, float))
                and total_epochs >= 0
                and total_epochs < 85
            ) or (isinstance(phase, (int, float)) and phase < 3)
            row.update(
                {
                    "exists": True,
                    "epoch": epoch,
                    "total_epochs": total_epochs,
                    "phase": phase,
                    "n_tensors": n_tensors,
                    "n_params": n_params,
                    "state_key": sk,
                    "val_ncc": ckpt.get("val_ncc", ckpt.get("best_ncc", "N/A")),
                    "status": "INCOMPLETE" if incomplete else "OK",
                }
            )
        rows.append(row)
        print(row)

    out = root / "MODEL_INVENTORY.txt"
    lines = [
        f"MR-TAE Model Inventory — {datetime.now().isoformat()}",
        "=" * 110,
        "",
        f"{'Model ID':<35} {'Exists':<8} {'Epoch':<8} {'Phase':<8} "
        f"{'#Params':<12} {'#Tensors':<10} {'State Key':<18} {'Status':<12} {'Val NCC':<10}",
        "-" * 110,
    ]
    for r in rows:
        lines.append(
            f"{r['model_id']:<35} {str(r.get('exists', '-')):<8} "
            f"{str(r.get('epoch', '-')):<8} {str(r.get('phase', '-')):<8} "
            f"{str(r.get('n_params', '-')):<12} {str(r.get('n_tensors', '-')):<10} "
            f"{str(r.get('state_key', '-')):<18} {r.get('status', '-'):<12} "
            f"{str(r.get('val_ncc', '-')):<10}"
        )
    lines.append("")
    lines.append("Flag INCOMPLETE if total_epochs < 85 OR final curriculum phase < 3")
    lines.append("Note: best.pt epoch = epoch of best val_ncc checkpoint, not final training epoch.")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
