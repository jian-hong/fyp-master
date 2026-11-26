"""
hpo/optuna_study.py — Optuna hyperparameter optimization for MR-TAE-FULL.

Runs 30 trials (Phase 1 only, 30 epochs each) with Hyperband pruning.
Best config saved to config/best_hparams.yaml.

Usage
-----
python hpo/optuna_study.py --n-trials 30 --study-name ae-pd-hpo
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

try:
    from optuna.integration import MLflowCallback
    _MLFLOW_CB = True
except ImportError:
    _MLFLOW_CB = False

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("optuna_study")

PHASE1_EPOCHS = 30   # fast proxy: only Phase 1 per trial
MODEL_ID = "MR-TAE-FULL"


# --------------------------------------------------------------------------
# Objective
# --------------------------------------------------------------------------

def objective(trial: optuna.Trial) -> float:
    """
    Train MR-TAE-FULL for Phase 1 only (30 epochs) and return val NCC.
    Hyperband pruner kills bad trials early.
    """
    import gc
    import torch
    from mr_tae_fusion.config import ModelConfig, Config
    from mr_tae_fusion.models.mr_tae_fusion import create_model
    from mr_tae_fusion.training.losses import MultiTaskLoss
    from mr_tae_fusion.data.dataset import PDSignalDataset
    from torch.utils.data import DataLoader
    from training.curricular_scheduler import CurricularScheduler, PhaseConfig

    # --- sample hyperparameters ---
    swin_heads    = trial.suggest_categorical("swin_num_heads",   [4, 8, 16])
    swin_window   = trial.suggest_categorical("swin_window_size", [16, 32, 64])
    bigru_hidden  = trial.suggest_categorical("bigru_hidden_dim", [128, 256, 512])
    encoder_base  = trial.suggest_categorical("encoder_base_ch",  [16, 32, 64])
    lr            = trial.suggest_float("lr",           1e-5, 3e-3, log=True)
    weight_decay  = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout       = trial.suggest_float("dropout",      0.0,  0.3)
    charb_eps     = trial.suggest_float("charb_epsilon",1e-4, 1e-2, log=True)

    # --- build model with trial params ---
    model_cfg = ModelConfig(
        encoder_channels=[encoder_base, encoder_base*2, encoder_base*4],
        gru_hidden_size=bigru_hidden,
        swin_num_heads=swin_heads,
        swin_window_size=swin_window,
        gru_dropout=dropout,
        swin_dropout=dropout,
    )
    model = create_model(model_cfg)
    n_params_M = sum(p.numel() for p in model.parameters()) / 1e6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = MultiTaskLoss()
    criterion = criterion.to(device)

    # Data — small for speed during HPO
    cfg = Config()
    train_ds = PDSignalDataset(config=cfg, num_samples=2000, mode="train", seed=42)
    val_ds   = PDSignalDataset(config=cfg, num_samples=400,  mode="val",   seed=43)
    bs = max(16, 64 if n_params_M < 20 else 32)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # Phase 1 scheduler only
    phase1_scheduler = CurricularScheduler(phases=[
        PhaseConfig(snr_min=5.0, snr_max=15.0, epochs=PHASE1_EPOCHS, noise_type="wgn")
    ])

    best_ncc = -1.0

    for epoch in range(PHASE1_EPOCHS):
        # --- Train ---
        model.train()
        for batch in train_loader:
            noisy = batch["noisy"].to(device)
            clean = batch["clean"].to(device)
            mask  = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            try:
                if scaler:
                    from torch.cuda.amp import autocast
                    with autocast():
                        pred, pred_seg = model(noisy)
                        loss, _ = criterion(pred, clean, pred_seg, mask)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred, pred_seg = model(noisy)
                    loss, _ = criterion(pred, clean, pred_seg, mask)
                    loss.backward()
                    optimizer.step()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    continue
                raise

        # --- Validate ---
        model.eval()
        total_ncc = 0.0
        n = 0
        with torch.no_grad():
            for batch in val_loader:
                noisy = batch["noisy"].to(device)
                clean = batch["clean"].to(device)
                pred, _ = model(noisy)
                p = pred.float().flatten(1) - pred.float().flatten(1).mean(1, keepdim=True)
                t = clean.float().flatten(1) - clean.float().flatten(1).mean(1, keepdim=True)
                ncc = (p * t).sum(1) / (torch.sqrt((p**2).sum(1) * (t**2).sum(1)) + 1e-8)
                total_ncc += ncc.mean().item()
                n += 1
        val_ncc = total_ncc / max(n, 1)
        best_ncc = max(best_ncc, val_ncc)

        # --- Report to pruner ---
        trial.report(val_ncc, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # cleanup
    del model, criterion, optimizer, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()

    return best_ncc


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna HPO for MR-TAE-FULL")
    parser.add_argument("--n-trials",   type=int, default=30)
    parser.add_argument("--study-name", type=str, default="ae-pd-hpo")
    parser.add_argument("--storage",    type=str, default=None,
                        help="Optuna DB URL, e.g. sqlite:///hpo.db (default: in-memory)")
    parser.add_argument("--out",        type=str, default="config/best_hparams.yaml")
    args = parser.parse_args()

    callbacks = []
    if _MLFLOW_CB:
        callbacks.append(
            MLflowCallback(
                metric_name="val_ncc",
                create_experiment=True,
            )
        )

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=HyperbandPruner(
            min_resource=5,
            max_resource=PHASE1_EPOCHS,
            reduction_factor=3,
        ),
        storage=args.storage,
        load_if_exists=True,
    )

    logger.info(
        "Starting Optuna study '%s' — %d trials (Phase 1 proxy, %d epochs each)",
        args.study_name, args.n_trials, PHASE1_EPOCHS,
    )

    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=1,            # RTX 4070: one trial at a time
        callbacks=callbacks,
        show_progress_bar=True,
    )

    best = study.best_trial
    logger.info("Best trial #%d  val_ncc=%.4f", best.number, best.value)
    logger.info("Best params: %s", best.params)

    # Save best hyperparameters
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(
            {"best_trial": best.number, "val_ncc": best.value, "params": best.params},
            f,
            default_flow_style=False,
        )
    logger.info("Best hyperparameters saved to %s", out_path)


if __name__ == "__main__":
    main()
