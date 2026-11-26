"""
training/trainer.py — Unified AE-PD denoising trainer.

Features
--------
- AMP (Automatic Mixed Precision) on every forward/backward
- Gradient checkpointing auto-enabled for models > 40 M params
- Curricular SNR scheduling (3 phases)
- OOM recovery: halves batch size, retries automatically
- GPU temperature gating via HeatMonitor
- MLflow metric logging every epoch
- Snapshot figures every SNAPSHOT_EVERY epochs
- KeyboardInterrupt → saves checkpoint before exit
- Tracks best checkpoint by val NCC (primary metric)
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .curricular_scheduler import CurricularScheduler
from .heat_monitor import HeatMonitor
from .memory_manager import RTX4070Manager

try:
    import mlflow
    _MLFLOW = True
except ImportError:
    _MLFLOW = False

logger = logging.getLogger(__name__)

SNAPSHOT_EVERY = 10     # epochs between snapshot saves


# ---------------------------------------------------------------------------
# Thermal schedule (phase breaks + micro-breaks; interruptible sleeps)
# ---------------------------------------------------------------------------


def _interruptible_sleep(seconds: float) -> None:
    """Sleep in small chunks so Ctrl+C raises KeyboardInterrupt promptly."""
    end = time.perf_counter() + float(seconds)
    while time.perf_counter() < end:
        try:
            time.sleep(min(1.0, end - time.perf_counter()))
        except KeyboardInterrupt:
            raise


class ThermalSchedule:
    """
    Phase-transition cooling breaks and per-phase micro-breaks.
    Heat-triggered pause is handled by HeatMonitor + trainer.train_epoch.
    """

    def __init__(self, scheduler: CurricularScheduler, cfg: Optional[Dict[str, Any]] = None) -> None:
        self.scheduler = scheduler
        c = cfg or {}
        self.break_p1_sec = float(c.get("thermal_break_phase1_sec", 300))
        self.break_p2_entry_sec = float(c.get("thermal_break_phase2_entry_sec", 120))
        self.micro_p2_sec = float(c.get("thermal_micro_break_sec", 60))
        self.micro_p2_every = int(c.get("thermal_micro_break_every_n_epochs", 10))
        self.micro_p3_sec = float(c.get("thermal_micro_break_phase3_sec", 45))
        self.micro_p3_every = int(c.get("thermal_micro_break_every_n_epochs_phase3", 5))

        p = scheduler.phases
        self._p1 = int(p[0].epochs)
        self._p2 = int(p[1].epochs)
        self._p3 = int(p[2].epochs)

    def after_epoch(self, epoch_idx: int, cur_phase: int) -> None:
        """Call after each epoch's metrics are logged (epoch_idx is 0-based)."""
        e = epoch_idx

        # End of Phase 1
        if e == self._p1 - 1:
            logger.info("Phase 1 complete — 5-min cooling break")
            try:
                _interruptible_sleep(self.break_p1_sec)
            except KeyboardInterrupt:
                raise
            return

        # End of Phase 2
        if e == self._p1 + self._p2 - 1:
            logger.info("Phase 2 complete — 2-min cooling break")
            try:
                _interruptible_sleep(self.break_p2_entry_sec)
            except KeyboardInterrupt:
                raise
            return

        # Phase 2 micro-breaks (every N epochs within phase 2)
        if cur_phase == 2 and self.micro_p2_every > 0:
            idx_in_p2 = e - self._p1 + 1
            if idx_in_p2 > 0 and idx_in_p2 % self.micro_p2_every == 0:
                logger.info("Micro-break — 1 min")
                try:
                    _interruptible_sleep(self.micro_p2_sec)
                except KeyboardInterrupt:
                    raise

        # Phase 3 micro-breaks
        if cur_phase == 3 and self.micro_p3_every > 0:
            idx_in_p3 = e - self._p1 - self._p2 + 1
            if idx_in_p3 > 0 and idx_in_p3 % self.micro_p3_every == 0:
                logger.info("Micro break 45seconds")
                try:
                    _interruptible_sleep(self.micro_p3_sec)
                except KeyboardInterrupt:
                    raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ncc(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Normalized cross-correlation, averaged over batch."""
    p = pred.detach().float().flatten(1)
    t = target.detach().float().flatten(1)
    p = p - p.mean(dim=1, keepdim=True)
    t = t - t.mean(dim=1, keepdim=True)
    num = (p * t).sum(dim=1)
    den = torch.sqrt((p**2).sum(dim=1) * (t**2).sum(dim=1)) + eps
    return (num / den).mean().item()


def _rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.sqrt(((pred - target) ** 2).mean()).item()


# ---------------------------------------------------------------------------
# Main Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Unified trainer for all MR-TAE model variants.

    Parameters
    ----------
    model        : nn.Module — any model from the registry
    criterion    : nn.Module — loss function (MultiTaskLoss or single-task)
    train_loader : DataLoader — yields dict with 'noisy', 'clean', 'mask'
    val_loader   : DataLoader
    device       : 'cuda' or 'cpu'
    save_dir     : root directory for checkpoints + snapshots
    model_id     : string name logged to MLflow
    scheduler    : CurricularScheduler (for phase-aware SNR sampling)
    lr           : initial learning rate
    weight_decay : AdamW weight decay
    grad_clip    : max gradient norm
    use_grad_ckpt: force gradient checkpointing regardless of param count
    mlflow_run_id: resume an existing MLflow run (optional)
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
        save_dir: str = "results",
        model_id: str = "model",
        scheduler: Optional[CurricularScheduler] = None,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        use_grad_ckpt: bool = False,
        mlflow_run_id: Optional[str] = None,
        resilience: Optional[object] = None,
        n_train: Optional[int] = None,
        seed: Optional[int] = None,
        thermal_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_id = model_id
        self.scheduler = scheduler or CurricularScheduler()
        self.grad_clip = grad_clip
        self.resilience = resilience
        self.n_train = n_train
        self.seed = seed
        tc = thermal_cfg or {}
        self.thermal_cfg = tc
        self._thermal_poll_s = float(tc.get("thermal_pause_poll_sec", 90))

        self.save_dir = Path(save_dir) / model_id
        self.ckpt_dir = self.save_dir / "checkpoints"
        self.plot_dir = self.save_dir / "plots"
        for d in (self.ckpt_dir, self.plot_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Memory / thermal managers
        self.mem = RTX4070Manager()
        _poll_iv = tc.get("thermal_poll_interval_sec")
        self.heat = HeatMonitor(
            log_file=str(self.save_dir / "heat_monitor.log"),
            pause_temp=tc.get("thermal_pause_temp"),
            resume_temp=tc.get("thermal_resume_temp"),
            check_interval_s=float(_poll_iv) if _poll_iv is not None else None,
        )
        self.heat.start()
        self.thermal_schedule = ThermalSchedule(self.scheduler, tc)

        # Gradient checkpointing
        if self.mem.should_use_grad_ckpt(self.model, force=use_grad_ckpt):
            self.mem.enable_grad_ckpt(self.model)

        # Optimizer — includes loss parameters for uncertainty weighting
        all_params = list(self.model.parameters()) + list(self.criterion.parameters())
        self.optimizer = torch.optim.AdamW(
            all_params, lr=lr, weight_decay=weight_decay
        )
        n_epochs = self.scheduler.total_epochs
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_epochs, eta_min=1e-6
        )

        # AMP scaler
        self.scaler: Optional[GradScaler] = (
            GradScaler("cuda") if self.device.type == "cuda" else None
        )

        # State
        self.epoch = 0
        self.best_val_ncc = -1.0
        self.best_epoch = 0
        self.history: Dict[str, List[float]] = {
            k: [] for k in
            ["train_loss", "val_loss", "train_ncc", "val_ncc",
             "train_rmse", "val_rmse", "phase"]
        }

        # MLflow
        self._mlflow_run_id = mlflow_run_id
        self._mlflow_run = None
        if _MLFLOW:
            self._start_mlflow()

    def _sync_synthetic_curriculum(self, epoch_idx: int) -> None:
        """Keep PDSignalDataset noise/SNR in lockstep with CurricularScheduler phases."""
        info = self.scheduler.info(epoch_idx)
        n_ep = self.scheduler.total_epochs
        for loader in (self.train_loader, self.val_loader):
            ds = getattr(loader, "dataset", None)
            fn = getattr(ds, "set_training_phase", None)
            if callable(fn):
                fn(epoch_idx, info["snr_range"], info["noise_type"], n_ep)

    # -----------------------------------------------------------------------
    # MLflow helpers
    # -----------------------------------------------------------------------

    def _start_mlflow(self) -> None:
        exp_name = f"AE-PD-Denoising/{self.model_id}"
        mlflow.set_experiment(exp_name)
        if self._mlflow_run_id:
            self._mlflow_run = mlflow.start_run(run_id=self._mlflow_run_id)
        else:
            self._mlflow_run = mlflow.start_run(run_name=self.model_id)
        # Persist run ID for crash-proof resume (optional)
        try:
            if self.resilience and self._mlflow_run:
                set_run_id = getattr(self.resilience, "set_mlflow_run_id", None)
                if callable(set_run_id):
                    set_run_id(self._mlflow_run.info.run_id)
        except Exception:
            pass
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        mlflow.log_param("model_id", self.model_id)
        mlflow.log_param("param_count_M", round(n_params, 2))
        mlflow.log_param("device", str(self.device))

    def _log_metrics(self, metrics: dict, step: int) -> None:
        if _MLFLOW and self._mlflow_run:
            mlflow.log_metrics(metrics, step=step)

    # -----------------------------------------------------------------------
    # Single train step (OOM-safe)
    # -----------------------------------------------------------------------

    def _train_step(
        self, batch: dict, current_batch_size: int
    ) -> Tuple[float, dict, int]:
        """
        Returns (loss_value, loss_components, batch_size_used).
        Handles OOM by halving batch size and retrying once.
        """
        noisy = batch["noisy"].to(self.device)
        clean = batch["clean"].to(self.device)
        mask  = batch["mask"].to(self.device)

        self.optimizer.zero_grad(set_to_none=True)

        try:
            if self.scaler:
                with autocast("cuda"):
                    pred_signal, pred_seg = self.model(noisy)
                    loss, loss_dict = self.criterion(pred_signal, clean, pred_seg, mask)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred_signal, pred_seg = self.model(noisy)
                loss, loss_dict = self.criterion(pred_signal, clean, pred_seg, mask)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            return loss.item(), loss_dict, current_batch_size

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                new_bs = self.mem.handle_oom(current_batch_size)
                return 0.0, {}, new_bs
            raise

    # -----------------------------------------------------------------------
    # Train one epoch
    # -----------------------------------------------------------------------

    def train_epoch(self) -> dict:
        self.model.train()
        total_loss = 0.0
        total_ncc  = 0.0
        total_rmse = 0.0
        n_batches  = 0

        info = self.scheduler.info(self.epoch)
        pbar = tqdm(
            self.train_loader,
            desc=f"[{self.model_id}] E{self.epoch+1} Ph{info['phase']}",
            leave=False,
        )
        current_bs = self.train_loader.batch_size or 32

        for batch in pbar:
            # Thermal gate (90 s polling until HeatMonitor clears too_hot)
            if self.heat.too_hot.is_set():
                logger.warning(
                    "GPU above thermal pause threshold — pausing training batch …"
                )
                try:
                    self.heat.wait_cool(poll_s=self._thermal_poll_s)
                except KeyboardInterrupt:
                    raise
                logger.info("Resuming after thermal pause")

            loss_val, loss_dict, current_bs = self._train_step(batch, current_bs)
            if loss_val == 0.0 and not loss_dict:
                continue  # OOM batch skipped

            noisy = batch["noisy"].to(self.device)
            clean = batch["clean"].to(self.device)
            with torch.no_grad():
                pred, _ = self.model(noisy)
                total_ncc  += _ncc(pred, clean)
                total_rmse += _rmse(pred, clean)

            total_loss += loss_val
            n_batches  += 1
            pbar.set_postfix(loss=f"{loss_val:.4f}")

        n = max(n_batches, 1)
        return {
            "train/loss": total_loss / n,
            "train/ncc":  total_ncc  / n,
            "train/rmse": total_rmse / n,
            "train/phase": info["phase"],
        }

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def validate_epoch(self) -> dict:
        self.model.eval()
        total_loss = 0.0
        total_ncc  = 0.0
        total_rmse = 0.0
        n_batches  = 0

        for batch in self.val_loader:
            noisy = batch["noisy"].to(self.device)
            clean = batch["clean"].to(self.device)
            mask  = batch["mask"].to(self.device)

            if self.scaler:
                with autocast("cuda"):
                    pred, pred_seg = self.model(noisy)
                    loss, _ = self.criterion(pred, clean, pred_seg, mask)
            else:
                pred, pred_seg = self.model(noisy)
                loss, _ = self.criterion(pred, clean, pred_seg, mask)

            total_loss += loss.item()
            total_ncc  += _ncc(pred, clean)
            total_rmse += _rmse(pred, clean)
            n_batches  += 1

        n = max(n_batches, 1)
        return {
            "val/loss": total_loss / n,
            "val/ncc":  total_ncc  / n,
            "val/rmse": total_rmse / n,
        }

    # -----------------------------------------------------------------------
    # Snapshot
    # -----------------------------------------------------------------------

    def _save_snapshot(self, epoch: int) -> None:
        """Save 3-panel denoising comparison at -15, -5, +5 dB SNR."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            self.model.eval()
            batch = next(iter(self.val_loader))
            noisy = batch["noisy"][:3].to(self.device)
            clean = batch["clean"][:3].to(self.device)
            with torch.no_grad():
                if self.scaler:
                    with autocast("cuda"):
                        pred, _ = self.model(noisy)
                else:
                    pred, _ = self.model(noisy)

            fig, axes = plt.subplots(3, 3, figsize=(15, 9))
            titles = ["Noisy", "Denoised", "Clean"]
            for row in range(3):
                for col, (sig, title) in enumerate(
                    zip([noisy[row, 0], pred[row, 0], clean[row, 0]], titles)
                ):
                    axes[row, col].plot(sig.cpu().float().numpy(), lw=0.7)
                    axes[row, col].set_title(f"Sample {row+1} — {title}", fontsize=8)
                    axes[row, col].axis("off")
            fig.suptitle(f"{self.model_id}  epoch {epoch+1}", fontsize=10)
            plt.tight_layout()
            path = self.plot_dir / f"snapshot_epoch{epoch+1:04d}.png"
            plt.savefig(path, dpi=120)
            plt.close(fig)
            if _MLFLOW and self._mlflow_run:
                mlflow.log_artifact(str(path))
        except Exception as e:
            logger.warning("Snapshot failed at epoch %d: %s", epoch + 1, e)

    # -----------------------------------------------------------------------
    # Checkpoint
    # -----------------------------------------------------------------------

    def save_checkpoint(self, name: str = "checkpoint.pt") -> Path:
        path = self.ckpt_dir / name
        torch.save({
            "epoch":            self.epoch,
            "model_state":      self.model.state_dict(),
            "optimizer_state":  self.optimizer.state_dict(),
            "scheduler_state":  self.lr_scheduler.state_dict(),
            "criterion_state":  self.criterion.state_dict(),
            "scaler_state":     self.scaler.state_dict() if self.scaler else None,
            "best_val_ncc":     self.best_val_ncc,
            "history":          self.history,
            "model_id":         self.model_id,
        }, path)
        return path

    def _save_best_config_sidecar(self) -> None:
        """Persist ablation flags + train sizing next to best.pt for reproducible benchmark loads."""
        ab = getattr(self.model, "ablation", None)
        if ab is None:
            return
        payload = {
            "use_attention_gates": bool(ab.use_attention_gates),
            "use_bigru": bool(ab.use_bigru),
            "use_swin": bool(ab.use_swin),
            "use_wavelet_pooling": bool(ab.use_wavelet_pooling),
            "use_mtl": bool(ab.use_mtl),
            "loss_config": str(ab.loss_config),
            "n_train": self.n_train,
            "seed": self.seed,
            "model_id": self.model_id,
        }
        out = self.ckpt_dir / "best_config.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Wrote %s", out)

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.lr_scheduler.load_state_dict(ckpt["scheduler_state"])
        self.criterion.load_state_dict(ckpt["criterion_state"])
        if self.scaler and ckpt.get("scaler_state"):
            self.scaler.load_state_dict(ckpt["scaler_state"])
        self.epoch         = ckpt.get("epoch", 0)
        self.best_val_ncc  = ckpt.get("best_val_ncc", -1.0)
        self.history       = ckpt.get("history", self.history)
        logger.info("Loaded checkpoint from epoch %d", self.epoch)

    # -----------------------------------------------------------------------
    # Full training loop
    # -----------------------------------------------------------------------

    def train(
        self,
        resume_from: Optional[str] = None,
        phase: Optional[int] = None,
    ) -> dict:
        """
        Run full curricular training.

        Parameters
        ----------
        resume_from : path to a checkpoint .pt file
        phase       : if 1/2/3, restrict training to that phase only
        """
        if resume_from:
            self.load_checkpoint(resume_from)

        n_epochs = self.scheduler.total_epochs
        self.mem.reset_peak_vram()

        try:
            for epoch in range(self.epoch, n_epochs):
                self.epoch = epoch
                cur_phase = self.scheduler.get_phase(epoch)

                # Skip epochs outside requested phase
                if phase is not None and cur_phase != phase:
                    continue

                self._sync_synthetic_curriculum(epoch)

                # --- train ---
                train_metrics = self.train_epoch()

                # --- validate ---
                val_metrics = self.validate_epoch()

                # --- LR step ---
                self.lr_scheduler.step()

                # --- GPU metrics ---
                gpu_metrics = self.mem.get_gpu_metrics()

                # --- combine + log ---
                all_metrics = {**train_metrics, **val_metrics, **gpu_metrics}
                self._log_metrics(all_metrics, step=epoch)

                # --- history ---
                self.history["train_loss"].append(train_metrics["train/loss"])
                self.history["val_loss"].append(val_metrics["val/loss"])
                self.history["train_ncc"].append(train_metrics["train/ncc"])
                self.history["val_ncc"].append(val_metrics["val/ncc"])
                self.history["train_rmse"].append(train_metrics["train/rmse"])
                self.history["val_rmse"].append(val_metrics["val/rmse"])
                self.history["phase"].append(cur_phase)

                logger.info(
                    "E%d Ph%d | train_loss=%.4f val_ncc=%.4f val_rmse=%.4f | "
                    "VRAM=%.2fGB GPU=%d°C",
                    epoch + 1, cur_phase,
                    train_metrics["train/loss"],
                    val_metrics["val/ncc"],
                    val_metrics["val/rmse"],
                    gpu_metrics["gpu/vram_used_gb"],
                    gpu_metrics["gpu/temp_c"],
                )

                try:
                    self.thermal_schedule.after_epoch(epoch, cur_phase)
                except KeyboardInterrupt:
                    raise

                # --- best checkpoint ---
                if val_metrics["val/ncc"] > self.best_val_ncc:
                    self.best_val_ncc = val_metrics["val/ncc"]
                    self.best_epoch = epoch
                    best_path = self.save_checkpoint("best.pt")
                    try:
                        if self.resilience:
                            self.resilience.on_checkpoint_saved(epoch, str(best_path))
                    except Exception:
                        pass
                    logger.info("  [BEST] New best NCC %.4f at epoch %d",
                                self.best_val_ncc, epoch + 1)
                    try:
                        self._save_best_config_sidecar()
                    except Exception as e:
                        logger.warning("best_config.json write failed: %s", e)

                # --- periodic checkpoint + snapshot ---
                if (epoch + 1) % SNAPSHOT_EVERY == 0:
                    self.save_checkpoint(f"checkpoint_epoch{epoch+1:04d}.pt")
                    self._save_snapshot(epoch)

                # --- resilience (session persistence + early stop) ---
                try:
                    if self.resilience:
                        decision = self.resilience.on_epoch_end(epoch, cur_phase, all_metrics)
                        if decision.get("action") == "skip_model":
                            logger.warning("Early stop requested: %s", decision.get("reason", ""))
                            break
                except Exception:
                    # Never allow persistence layer to crash training
                    pass

        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt — saving emergency checkpoint …")
            interrupt_path = self.save_checkpoint("interrupt.pt")
            try:
                if self.resilience:
                    self.resilience.on_checkpoint_saved(self.epoch, str(interrupt_path))
            except Exception:
                pass

        finally:
            self.heat.stop()
            if _MLFLOW and self._mlflow_run:
                mlflow.end_run()

        # Save history
        hist_path = self.save_dir / "history.json"
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2)

        logger.info(
            "Training complete. Best val NCC=%.4f at epoch %d.",
            self.best_val_ncc, self.best_epoch + 1
        )
        return self.history
