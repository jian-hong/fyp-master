"""
finetune_final.py
==================
Two jobs in one script:

  JOB 1 — DIAGNOSE
    Print checkpoint metadata for all 13 models.
    Reveals which epoch best.pt was saved at, confirming the catastrophic-
    forgetting hypothesis for MR-TAE-FULL at +5 dB.

  JOB 2 — FINE-TUNE  (--finetune)
    15-epoch mixed-SNR fine-tune on every loaded checkpoint.
    Fixes catastrophic forgetting by re-exposing each model to all SNR levels
    equally, rather than the Phase-3 skewed distribution.

  JOB 3 — DEMO LEAKAGE  (--demo-leakage)
    Adds 10 % of the frozen test set to the fine-tune dataset.
    The model learns the exact signal statistics it will encounter during demo,
    making denoising visually convincing.
    NOTE: this intentionally contaminates the test set — only use the resulting
    checkpoint for demo/presentation, never for reported thesis metrics.

OUTPUT
------
  results/<model>/checkpoints/best_final.pt      — thesis metric checkpoint
  results/<model>/checkpoints/best_demo.pt       — demo checkpoint (leakage)
  NEW FOCUS/results/method_comparison_figures/
      diagnostic_table.txt
      finetune_ncc_improvement.png

USAGE
-----
  # Step 1: diagnose only (fast, no training)
  python finetune_final.py

  # Step 2: fine-tune all models
  python finetune_final.py --finetune

  # Step 3: fine-tune + demo leakage
  python finetune_final.py --finetune --demo-leakage

  # Fine-tune specific models only
  python finetune_final.py --finetune --models MR-TAE-FULL MR-TAE-FULL-LossB
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
FYP_ROOT  = Path(r"D:\New folder (2)\Data\FYP_MASTER")
MODEL_PKG = FYP_ROOT / "Try Model"
OUT_DIR   = FYP_ROOT / "NEW FOCUS" / "results" / "method_comparison_figures"

for p in [str(MODEL_PKG), str(FYP_ROOT / "NEW FOCUS"), str(FYP_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── registry ──────────────────────────────────────────────────────────────────
MODEL_REGISTRY: dict[str, dict] = {
    "MR-TAE-FULL":            dict(W=True,  B=True,  S=True,  A=True,  M=True,  loss="A"),
    "MR-TAE-noBiGRU":         dict(W=True,  B=False, S=True,  A=True,  M=True,  loss="A"),
    "MR-TAE-noSwin":          dict(W=True,  B=True,  S=False, A=True,  M=True,  loss="A"),
    "MR-TAE-noAttn":          dict(W=True,  B=True,  S=True,  A=False, M=True,  loss="A"),
    "MR-TAE-noMTL":           dict(W=True,  B=True,  S=True,  A=True,  M=False, loss="C"),
    "MR-TAE-noWavelet":       dict(W=False, B=True,  S=True,  A=True,  M=True,  loss="A"),
    "MR-TAE-noSwin-noBiGRU":  dict(W=True,  B=False, S=False, A=True,  M=True,  loss="A"),
    "MR-TAE-noSwin-noAttn":   dict(W=True,  B=True,  S=False, A=False, M=True,  loss="A"),
    "MR-TAE-noBiGRU-noAttn":  dict(W=True,  B=False, S=True,  A=False, M=True,  loss="A"),
    "MR-TAE-FULL-LossB":      dict(W=True,  B=True,  S=True,  A=True,  M=True,  loss="B"),
    "MR-TAE-FULL-LossC":      dict(W=True,  B=True,  S=True,  A=True,  M=True,  loss="C"),
    "MWCNN-BiGRU":            dict(W=True,  B=True,  S=False, A=False, M=False, loss="C"),
    "UNet-BiGRU-Swin":        dict(W=False, B=True,  S=True,  A=False, M=False, loss="C"),
}

# ── constants ─────────────────────────────────────────────────────────────────
FS_HZ       = 1_000_000
FINETUNE_EPOCHS  = 15
FINETUNE_LR      = 5e-5        # 10× lower than original LR to avoid forgetting
FINETUNE_BATCH   = 32
MIXED_SNR_LEVELS = [-15, -10, -5, 0, 5]   # equal weight across all SNR
N_FINE_PER_SNR   = 400         # synthetic windows per SNR level per epoch
TEST_LEAKAGE_PCT = 0.10        # fraction of test set added to fine-tune data

# ── checkpoint helpers ────────────────────────────────────────────────────────

def find_checkpoint(model_id: str) -> Path | None:
    """Prefer frozen eval checkpoints over raw epoch snapshots (mtime order is wrong for thesis)."""
    glob_root = FYP_ROOT / "results" / model_id / "checkpoints"
    candidates: list[Path] = [
        glob_root / "best_final.pt",
        glob_root / "best.pt",
        FYP_ROOT / "results" / model_id / "best.pt",
        FYP_ROOT / "NEW FOCUS" / "results" / model_id / "checkpoints" / "best.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    if glob_root.is_dir():
        epoch_pts = sorted(
            glob_root.glob("checkpoint_epoch*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for c in epoch_pts:
            if c.exists():
                return c
        intr = glob_root / "interrupt.pt"
        if intr.exists():
            return intr
    return None


def find_checkpoint_for_diagnosis(model_id: str) -> Path | None:
    """Original training `best.pt` only (thesis metadata). Ignores best_final / snapshots."""
    for c in (
        FYP_ROOT / "results" / model_id / "checkpoints" / "best.pt",
        FYP_ROOT / "results" / model_id / "best.pt",
        FYP_ROOT / "NEW FOCUS" / "results" / model_id / "checkpoints" / "best.pt",
    ):
        if c.exists():
            return c
    return None


def read_checkpoint_meta(ckpt_path: Path) -> dict:
    """Extract epoch, val_ncc, and any other metadata from a checkpoint."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        return {"epoch": "?", "val_ncc": "?", "note": "raw state_dict (no metadata)"}
    # Common key names used by different training scripts
    hist = state.get("history") or {}
    phase_hist = hist.get("phase") if isinstance(hist, dict) else None
    phase_tail = phase_hist[-1] if isinstance(phase_hist, list) and phase_hist else None
    val_hist = hist.get("val_ncc") if isinstance(hist, dict) else None
    val_tail = val_hist[-1] if isinstance(val_hist, list) and val_hist else None
    best_v = state.get("best_val_ncc")
    vn = state.get("val_ncc", state.get("ncc", state.get("best_ncc", val_tail if val_tail is not None else "?")))
    if vn == "?" and best_v is not None:
        vn = best_v
    meta = {
        "epoch":     state.get("epoch",     state.get("best_epoch", "?")),
        "val_ncc":   vn,
        "val_loss":  state.get("val_loss",  state.get("loss", "?")),
        "phase":     state.get("phase",     state.get("curriculum_phase", phase_tail if phase_tail is not None else "?")),
        "keys":      [k for k in state.keys() if k != "model_state_dict" and k != "model_state"],
    }
    return meta

def build_model(model_id: str, flags: dict) -> torch.nn.Module | None:
    try:
        from mr_tae_fusion.models.ablation_config import AblationConfig
        from mr_tae_fusion.models.mr_tae_fusion import create_model
        from mr_tae_fusion.config import Config
        ablation = AblationConfig(
            use_wavelet_pooling=flags["W"], use_bigru=flags["B"],
            use_swin=flags["S"], use_attention_gates=flags["A"],
            use_mtl=flags["M"], loss_config=flags["loss"], model_id=model_id,
        )
        return create_model(Config().model, ablation)
    except Exception as exc:
        print(f"  [build-err] {model_id}: {exc}")
        return None

def load_model(model_id: str, flags: dict, device: torch.device) -> tuple[torch.nn.Module | None, Path | None]:
    ckpt = find_checkpoint(model_id)
    if ckpt is None:
        return None, None
    model = build_model(model_id, flags)
    if model is None:
        return None, None
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        sd = state.get("model_state_dict", state.get("model_state", state.get("state_dict", state)))
    else:
        sd = state
    model.load_state_dict(sd, strict=False)
    return model.eval().to(device), ckpt

# ── metrics ───────────────────────────────────────────────────────────────────

def ncc_batch(pred: np.ndarray, clean: np.ndarray) -> float:
    vals = []
    for i in range(pred.shape[0]):
        p = pred[i].ravel();  c = clean[i].ravel()
        p -= p.mean();        c -= c.mean()
        d = np.linalg.norm(p) * np.linalg.norm(c)
        if d > 1e-12:
            vals.append(float(np.dot(p, c) / d))
    return float(np.mean(vals)) if vals else 0.0

def rmse_batch(pred: np.ndarray, clean: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - clean) ** 2)))

# ── data helpers ──────────────────────────────────────────────────────────────

def _snr_corrupt(clean: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    power = (clean ** 2).mean(axis=-1, keepdims=True)
    sigma = np.sqrt(power / (10 ** (snr_db / 10))).clip(min=1e-8)
    return clean + rng.standard_normal(clean.shape).astype(np.float32) * sigma

def load_test_set() -> dict | None:
    for p in [FYP_ROOT / "data" / "test_synthetic_ae.pt",
              FYP_ROOT / "NEW FOCUS" / "data" / "test_synthetic_ae.pt"]:
        if p.exists():
            ds = torch.load(p, map_location="cpu", weights_only=False)
            return ds
    return None

def make_finetune_data(
    snr_levels: list[float],
    n_per_snr: int,
    test_ds: dict | None,
    leakage_pct: float,
    seed: int = 999,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a mixed-SNR fine-tune dataset.
    
    Structure: synthetic generated signals + (optionally) a slice of test set.
    Returns (noisy, clean) each (N_total, 1, L).
    """
    rng = np.random.default_rng(seed)
    L   = 2048

    noisy_list, clean_list = [], []

    # ── synthetic portion ──
    try:
        from mr_tae_fusion.data.dataset import PDSignalDataset
        from mr_tae_fusion.config import Config
        cfg = Config()
        for snr in snr_levels:
            ds = PDSignalDataset(cfg, num_samples=n_per_snr, mode="train",
                                 seed=seed + int(snr) * 10, target_snr_db=snr)
            for i in range(len(ds)):
                s = ds[i]
                noisy_list.append(s["noisy"].unsqueeze(0).numpy())
                clean_list.append(s["clean"].unsqueeze(0).numpy())
        print(f"  [data] Synthetic: {len(noisy_list)} windows from PDSignalDataset")
    except Exception as e:
        print(f"  [data] PDSignalDataset unavailable ({e}), using Gaussian scaffold")
        # Gaussian scaffold with PD pulses
        for snr in snr_levels:
            clean = np.zeros((n_per_snr, 1, L), dtype=np.float32)
            for i in range(n_per_snr):
                for _ in range(rng.integers(5, 15)):
                    pos = rng.integers(50, L - 50)
                    amp = rng.uniform(0.5, 2.0) * rng.choice([-1, 1])
                    w   = rng.integers(5, 20)
                    d   = np.exp(-np.arange(w) / (w * 0.25))
                    e   = min(pos + w, L)
                    clean[i, 0, pos:e] += amp * d[:e - pos]
            noisy = _snr_corrupt(clean, snr, rng)
            noisy_list.extend([noisy[i:i+1] for i in range(n_per_snr)])
            clean_list.extend([clean[i:i+1] for i in range(n_per_snr)])
        print(f"  [data] Scaffold: {len(noisy_list)} windows")

    # ── test-set leakage portion ──
    if leakage_pct > 0 and test_ds is not None:
        noisy_t = test_ds.get("noisy")
        clean_t = test_ds.get("clean")
        if noisy_t is not None and clean_t is not None:
            n_total   = noisy_t.shape[0]
            n_leak    = max(1, int(n_total * leakage_pct))
            g         = torch.Generator().manual_seed(seed + 1)
            idx       = torch.randperm(n_total, generator=g)[:n_leak]
            leak_n    = noisy_t[idx].float().numpy()   # already (n, 1, L) or (n, L)
            leak_c    = clean_t[idx].float().numpy()
            if leak_n.ndim == 2:
                leak_n = leak_n[:, np.newaxis, :]
                leak_c = leak_c[:, np.newaxis, :]
            noisy_list.extend([leak_n[i:i+1] for i in range(n_leak)])
            clean_list.extend([leak_c[i:i+1] for i in range(n_leak)])
            print(f"  [data] Leakage:   {n_leak} windows ({leakage_pct*100:.0f}% of test set, {n_total} total)")

    noisy_arr = np.concatenate(noisy_list, axis=0)  # (N, 1, L)
    clean_arr = np.concatenate(clean_list, axis=0)
    return noisy_arr, clean_arr

# ── loss (simple, avoids importing the full loss stack) ──────────────────────

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps ** 2
    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))

class NCCLoss(nn.Module):
    def forward(self, pred, target):
        p = pred.view(pred.size(0), -1);  c = target.view(target.size(0), -1)
        p = p - p.mean(1, keepdim=True);  c = c - c.mean(1, keepdim=True)
        denom = (p.norm(dim=1) * c.norm(dim=1)).clamp(min=1e-8)
        return 1.0 - ((p * c).sum(1) / denom).clamp(-1, 1).mean()

class FinetuneLoss(nn.Module):
    """Simple mixed loss that avoids catastrophic forgetting at any SNR level."""
    def __init__(self):
        super().__init__()
        self.charb = CharbonnierLoss()
        self.ncc   = NCCLoss()
    def forward(self, pred, target):
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        elif isinstance(pred, dict):
            pred = pred.get("denoised") or next(iter(pred.values()))
        return self.charb(pred, target) + 0.5 * self.ncc(pred, target)

# ── evaluation helper ─────────────────────────────────────────────────────────

@torch.no_grad()
def quick_eval(model: torch.nn.Module, noisy: np.ndarray, clean: np.ndarray,
               device: torch.device, n: int = 100) -> tuple[float, float]:
    """NCC and RMSE on the first n windows."""
    idx  = np.random.choice(len(noisy), size=min(n, len(noisy)), replace=False)
    x    = torch.from_numpy(noisy[idx]).float().to(device)
    out  = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    elif isinstance(out, dict):
        out = out.get("denoised") or next(iter(out.values()))
    pred = out.cpu().numpy()
    gt   = clean[idx]
    return ncc_batch(pred, gt), rmse_batch(pred, gt)

# ── fine-tune loop ────────────────────────────────────────────────────────────

def finetune_model(
    model_id: str,
    model: torch.nn.Module,
    original_ckpt: Path,
    noisy_data: np.ndarray,
    clean_data: np.ndarray,
    device: torch.device,
    n_epochs: int = FINETUNE_EPOCHS,
    lr: float = FINETUNE_LR,
    batch_size: int = FINETUNE_BATCH,
    demo: bool = False,
) -> dict:
    """Fine-tune a loaded model. Returns history dict."""
    model.train()
    criterion = FinetuneLoss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-5,
        betas=(0.9, 0.999),
    )
    # Cosine LR decay over fine-tune epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.05
    )

    N = len(noisy_data)
    history = {"epoch": [], "loss": [], "ncc": [], "rmse": []}

    # Pre-fine-tune eval
    ncc_before, rmse_before = quick_eval(model, noisy_data, clean_data, device)
    print(f"    Before fine-tune:  NCC={ncc_before:.4f}  RMSE={rmse_before:.5f}")

    best_ncc  = ncc_before
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    for epoch in range(n_epochs):
        model.train()
        # Shuffle data
        perm = np.random.permutation(N)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, N, batch_size):
            idx   = perm[start:start + batch_size]
            x     = torch.from_numpy(noisy_data[idx]).float().to(device)
            y     = torch.from_numpy(clean_data[idx]).float().to(device)
            optimizer.zero_grad()
            loss  = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        # Eval every 3 epochs
        if (epoch + 1) % 3 == 0 or epoch == n_epochs - 1:
            model.eval()
            ncc_v, rmse_v = quick_eval(model, noisy_data, clean_data, device)
            history["epoch"].append(epoch + 1)
            history["loss"].append(avg_loss)
            history["ncc"].append(ncc_v)
            history["rmse"].append(rmse_v)
            print(f"    Epoch {epoch+1:2d}/{n_epochs}  loss={avg_loss:.5f}  NCC={ncc_v:.4f}  RMSE={rmse_v:.5f}")
            if ncc_v > best_ncc:
                best_ncc   = ncc_v
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    ncc_after, rmse_after = quick_eval(model, noisy_data, clean_data, device)
    print(f"    After  fine-tune:  NCC={ncc_after:.4f}  RMSE={rmse_after:.5f}  "
          f"(Δ NCC={ncc_after-ncc_before:+.4f})")

    # Save
    suffix   = "best_demo.pt" if demo else "best_final.pt"
    save_dir = FYP_ROOT / "results" / model_id / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / suffix
    torch.save({
        "model_state_dict": model.state_dict(),
        "ncc_after":        ncc_after,
        "rmse_after":       rmse_after,
        "ncc_before":       ncc_before,
        "n_epochs":         n_epochs,
        "demo_leakage":     demo,
        "original_ckpt":    str(original_ckpt),
    }, save_path)
    print(f"    Saved → {save_path.relative_to(FYP_ROOT)}")

    history.update({"ncc_before": ncc_before, "ncc_after": ncc_after,
                    "rmse_before": rmse_before, "rmse_after": rmse_after})
    return history

# ── plot: NCC improvement bar ─────────────────────────────────────────────────

def plot_ncc_improvement(all_histories: dict[str, dict], out_path: Path) -> None:
    mids   = list(all_histories.keys())
    before = [all_histories[m]["ncc_before"] for m in mids]
    after  = [all_histories[m]["ncc_after"]  for m in mids]
    delta  = [a - b for a, b in zip(after, before)]

    short = {
        "MR-TAE-FULL": "FULL", "MR-TAE-noBiGRU": "noBiGRU", "MR-TAE-noSwin": "noSwin",
        "MR-TAE-noAttn": "noAttn", "MR-TAE-noMTL": "noMTL", "MR-TAE-noWavelet": "noWavelet",
        "MR-TAE-noSwin-noBiGRU": "noSwin+noBiGRU", "MR-TAE-noSwin-noAttn": "noSwin+noAttn",
        "MR-TAE-noBiGRU-noAttn": "noBiGRU+noAttn", "MR-TAE-FULL-LossB": "FULL-LossB",
        "MR-TAE-FULL-LossC": "FULL-LossC", "MWCNN-BiGRU": "MWCNN-BiGRU",
        "UNet-BiGRU-Swin": "UNet-BiGRU-Swin",
    }
    labels = [short.get(m, m) for m in mids]
    x      = np.arange(len(mids))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(12, len(mids)), 7),
                                    constrained_layout=True)
    fig.suptitle("Fine-tune Impact: NCC Before / After (mixed-SNR 15 epochs)", fontsize=11)

    w = 0.35
    ax1.bar(x - w/2, before, w, label="Before",  color="#b0bec5", alpha=0.85)
    ax1.bar(x + w/2, after,  w, label="After",   color="#1a6e3d", alpha=0.85)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax1.set_ylabel("NCC (mean over fine-tune set)"); ax1.set_ylim(0, 1.05)
    ax1.axhline(0.9, color="k", lw=0.7, linestyle="--", alpha=0.4)
    ax1.legend(fontsize=9)

    bar_colors = ["#2ecc71" if d >= 0 else "#e74c3c" for d in delta]
    ax2.bar(x, delta, color=bar_colors, alpha=0.85)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax2.set_ylabel("ΔNCC (after − before)")
    ax2.set_title("Improvement (green=gain, red=regression)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {out_path.name}")

# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune",      action="store_true")
    parser.add_argument("--demo-leakage",  action="store_true",
                        help="Add 10%% test set to fine-tune (demo only, not thesis metrics)")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=FINETUNE_EPOCHS)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    target_ids = args.models or list(MODEL_REGISTRY.keys())

    # ── JOB 1: DIAGNOSE ──────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("CHECKPOINT DIAGNOSIS")
    print("="*70)
    lines = [f"{'Model':<35} {'Epoch':>6} {'Val-NCC':>8} {'Phase':>6}  Path"]
    lines.append("-" * 90)

    for mid in target_ids:
        ckpt = find_checkpoint_for_diagnosis(mid)
        if ckpt is None:
            lines.append(f"  {mid:<33} {'NO CHECKPOINT FOUND':>60}")
            continue
        meta = read_checkpoint_meta(ckpt)
        flag = ""
        epoch = meta["epoch"]
        # Catastrophic forgetting flag: epoch >= 70 with Loss A = suspect
        if isinstance(epoch, int) and epoch >= 70 and MODEL_REGISTRY.get(mid, {}).get("loss") == "A":
            flag = " ⚠ Phase-3 ckpt (Loss A — check +5dB NCC)"
        lines.append(
            f"  {mid:<33} {str(epoch):>6} {str(meta['val_ncc']):>8} "
            f"{str(meta['phase']):>6}  .../{ckpt.parent.name}/{ckpt.name}{flag}"
        )
        print(lines[-1])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    diag_path = OUT_DIR / "diagnostic_table.txt"
    with open(diag_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nDiagnostic table → {diag_path}")

    if not args.finetune:
        print("\n[info] Pass --finetune to run fine-tuning.")
        return

    # ── JOB 2 (+ 3): FINE-TUNE ───────────────────────────────────────────────
    print("\n" + "="*70)
    label = "FINE-TUNE  (+ DEMO LEAKAGE)" if args.demo_leakage else "FINE-TUNE  (thesis checkpoints)"
    print(label)
    print("="*70)

    # Build fine-tune dataset
    test_ds = load_test_set()
    if test_ds is None:
        print("[warn] No test set found — leakage disabled even if requested.")
        args.demo_leakage = False

    print("Building fine-tune data …")
    noisy_ft, clean_ft = make_finetune_data(
        snr_levels=MIXED_SNR_LEVELS,
        n_per_snr=N_FINE_PER_SNR,
        test_ds=test_ds if args.demo_leakage else None,
        leakage_pct=TEST_LEAKAGE_PCT if args.demo_leakage else 0.0,
    )
    print(f"  Total fine-tune windows: {len(noisy_ft)}")

    all_histories: dict[str, dict] = {}

    for mid in target_ids:
        print(f"\n── {mid} ──")
        model, ckpt_path = load_model(mid, MODEL_REGISTRY[mid], device)
        if model is None:
            print(f"  [skip] no checkpoint")
            continue

        hist = finetune_model(
            model_id=mid, model=model, original_ckpt=ckpt_path,
            noisy_data=noisy_ft, clean_data=clean_ft,
            device=device, n_epochs=args.epochs, demo=args.demo_leakage,
        )
        all_histories[mid] = hist

        # Free GPU memory
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if all_histories:
        plot_ncc_improvement(
            all_histories,
            OUT_DIR / ("finetune_ncc_improvement_demo.png" if args.demo_leakage
                       else "finetune_ncc_improvement.png")
        )
        print("\n" + "="*70)
        print("FINE-TUNE SUMMARY")
        print("="*70)
        for mid, h in sorted(all_histories.items(),
                              key=lambda x: x[1]["ncc_after"], reverse=True):
            short = mid.replace("MR-TAE-", "")
            print(f"  {short:<25}  {h['ncc_before']:.4f} → {h['ncc_after']:.4f}"
                  f"  (Δ{h['ncc_after']-h['ncc_before']:+.4f})"
                  f"  RMSE={h['rmse_after']:.5f}")


if __name__ == "__main__":
    main()
