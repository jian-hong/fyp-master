"""
pipeline/mlflow_orchestrator.py — Resilient experiment orchestrator (single entrypoint).

Usage
-----
# Resume safely from session_state.json (recommended default):
python pipeline/mlflow_orchestrator.py --resume

# Train all models from registry (continues if session exists):
python pipeline/mlflow_orchestrator.py --run-all

# Train one model, full curriculum:
python pipeline/mlflow_orchestrator.py --model MR-TAE-FULL

# Hyperparameter override:
python pipeline/mlflow_orchestrator.py --resume --hparams config/best_hparams.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

# --------------------------------------------------------------------------
# Make project root + model package importable
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Try Model"))

from training import RTX4070Manager, Trainer
from training.curricular_scheduler import CurricularScheduler, PhaseConfig
from training.resilience import ResilienceManager, get_resume_point
from mr_tae_fusion.models.ablation_config import AblationConfig
from mr_tae_fusion.training.losses import build_loss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("orchestrator")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def load_registry(path: str = "config/model_registry.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)["models"]

def load_training_config(
    path: str = "config/training_config.yaml",
    hparams_override: str | None = None,
) -> dict:
    cfg: dict = {}
    if Path(path).exists():
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    if hparams_override and Path(hparams_override).exists():
        with open(hparams_override) as f:
            overrides = yaml.safe_load(f) or {}
        cfg.update(overrides)
        logger.info("Applied hyperparameter overrides from %s", hparams_override)
    return cfg


def apply_noise_regime(master_cfg: Any, training_cfg: dict) -> None:
    """Amplify interfering noise structures for synthetic training (never touches frozen test sets)."""
    regime = str(training_cfg.get("noise_regime", "standard")).lower().strip()
    if regime != "tough":
        return
    n = master_cfg.noise
    n.wgn_amplitude = float(min(0.22, n.wgn_amplitude * 1.22))
    n.impulse_probability = float(min(0.045, n.impulse_probability * 2.0))
    n.powerline_amplitude *= 1.38
    n.harmonic_amplitude *= 1.38
    n.narrowband_amplitude *= 1.32


def build_curricular_scheduler(training_cfg: dict) -> CurricularScheduler:
    """Match dataset difficulty phases; `tough` widens SNR and deepens Phase 3 stress."""
    p1 = int(training_cfg.get("phase1_epochs", 35))
    p2 = int(training_cfg.get("phase2_epochs", 35))
    p3 = int(training_cfg.get("phase3_epochs", 30))
    regime = str(training_cfg.get("noise_regime", "standard")).lower().strip()
    if regime == "tough":
        return CurricularScheduler(
            phases=[
                PhaseConfig(5.0, 16.0, p1, "wgn"),
                PhaseConfig(-10.0, 5.0, p2, "wgn_impulsive"),
                PhaseConfig(-28.0, -5.0, p3, "composite"),
            ]
        )
    return CurricularScheduler(phase1_epochs=p1, phase2_epochs=p2, phase3_epochs=p3)


def merge_model_flags_from_training(model_id: str, registry_entry: dict, training_cfg: dict) -> dict:
    """Merge global training YAML flags into registry entry.

    Architecture toggles (wavelet, BiGRU, Swin, attention, MTL) follow ``training_config.yaml``
    when keys are present.

    ``loss_config`` is never taken blindly from the global YAML: it is set from ``model_id``
    when the id contains ``LossB`` or ``LossC``; otherwise the registry value is kept (e.g.
    MR-TAE-FULL keeps Loss A even if training YAML says ``loss_config: B``).
    """
    keys = (
        "use_wavelet_pooling",
        "use_bigru",
        "use_swin",
        "use_attention_gates",
        "use_mtl",
    )
    out = dict(registry_entry)
    for k in keys:
        if k in training_cfg and training_cfg[k] is not None:
            out[k] = training_cfg[k]

    if "LossB" in model_id:
        out["loss_config"] = "B"
    elif "LossC" in model_id:
        out["loss_config"] = "C"
    else:
        lc = registry_entry.get("loss_config", "A")
        out["loss_config"] = str(lc).strip() if lc is not None else "A"

    return out


def build_model(model_id: str, cfg: dict) -> torch.nn.Module:
    """Construct a model from registry flags via AblationConfig."""
    from mr_tae_fusion.models.mr_tae_fusion import create_model
    from mr_tae_fusion.config import Config

    ablation = AblationConfig(
        use_wavelet_pooling=cfg.get("use_wavelet_pooling", True),
        use_bigru=cfg.get("use_bigru", True),
        use_swin=cfg.get("use_swin", True),
        use_attention_gates=cfg.get("use_attention_gates", True),
        use_mtl=cfg.get("use_mtl", True),
        loss_config=cfg.get("loss_config", "A"),
        model_id=model_id,
    )
    master_cfg = Config()
    model = create_model(master_cfg.model, ablation)
    return model


def build_criterion(ablation: AblationConfig) -> torch.nn.Module:
    """Select loss function based on loss_config flag and MTL setting."""
    return build_loss(loss_config=ablation.loss_config, use_mtl=ablation.use_mtl)


def build_loaders(
    master_cfg: Any,
    batch_size: int,
    n_train: int,
    n_val: int,
    seed: int,
    total_epochs: int,
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = False,
) -> tuple:
    from mr_tae_fusion.data.dataset import PDSignalDataset
    from torch.utils.data import DataLoader

    train_ds = PDSignalDataset(
        config=master_cfg,
        num_samples=n_train,
        mode="train",
        seed=seed,
        total_epochs=total_epochs,
    )
    val_ds = PDSignalDataset(
        config=master_cfg,
        num_samples=n_val,
        mode="val",
        seed=seed + 1,
        total_epochs=total_epochs,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers
    )
    return train_loader, val_loader


# --------------------------------------------------------------------------
# Single model run
# --------------------------------------------------------------------------

def run_model(
    model_id: str,
    cfg: dict,
    device: str = "cuda",
    results_root: str = "results",
    training_cfg: dict | None = None,
    resilience: ResilienceManager | None = None,
    resume_ckpt: str | None = None,
    mlflow_run_id: str | None = None,
) -> dict:
    logger.info("=" * 60)
    logger.info("Model: %s", model_id)
    logger.info("=" * 60)

    mem = RTX4070Manager()
    mem.cleanup_between_models()
    tcfg = training_cfg or {}

    model = build_model(model_id, cfg)
    n_params_M = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Params: %.2f M", n_params_M)

    if tcfg.get("batch_size") is not None:
        batch_size = int(tcfg["batch_size"])
    else:
        batch_size = mem.get_safe_batch_size(n_params_M)
    logger.info("Batch size: %d", batch_size)

    criterion = build_criterion(model.ablation)
    from mr_tae_fusion.config import Config as MRConfig

    master_cfg = MRConfig()
    apply_noise_regime(master_cfg, tcfg)
    curricular_scheduler = build_curricular_scheduler(tcfg)

    train_loader, val_loader = build_loaders(
        master_cfg,
        batch_size=batch_size,
        n_train=int(tcfg.get("n_train", 4000)),
        n_val=int(tcfg.get("n_val", 800)),
        seed=int(tcfg.get("seed", 42)),
        total_epochs=curricular_scheduler.total_epochs,
        num_workers=int(tcfg.get("num_workers", 0)),
        pin_memory=bool(tcfg.get("pin_memory", True)),
        persistent_workers=(
            bool(tcfg.get("persistent_workers", False))
            and int(tcfg.get("num_workers", 0)) > 0
        ),
    )
    scheduler = curricular_scheduler

    import mlflow
    if mlflow.active_run():
        mlflow.end_run()

    thermal_keys = (
        "thermal_break_phase1_sec",
        "thermal_break_phase2_entry_sec",
        "thermal_micro_break_sec",
        "thermal_micro_break_every_n_epochs",
        "thermal_micro_break_phase3_sec",
        "thermal_micro_break_every_n_epochs_phase3",
        "thermal_pause_temp",
        "thermal_resume_temp",
        "thermal_poll_interval_sec",
        "thermal_pause_poll_sec",
    )
    thermal_cfg = {k: tcfg[k] for k in thermal_keys if k in tcfg}

    trainer = Trainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=results_root,
        model_id=model_id,
        scheduler=scheduler,
        lr=float(tcfg.get("lr", 3e-4)),
        weight_decay=float(tcfg.get("weight_decay", 1e-4)),
        grad_clip=float(tcfg.get("grad_clip", 1.0)),
        mlflow_run_id=mlflow_run_id,
        resilience=resilience,
        n_train=int(tcfg.get("n_train", 4000)),
        seed=int(tcfg.get("seed", 42)),
        thermal_cfg=thermal_cfg,
    )

    history = trainer.train(resume_from=resume_ckpt, phase=None)

    best_ckpt = str(trainer.ckpt_dir / "best.pt")
    _run_benchmark(model_id, best_ckpt)

    mem.cleanup_between_models()
    
    if mlflow.active_run():
        mlflow.end_run()
        
    return history


def _run_benchmark(model_id: str, checkpoint_path: str) -> None:
    try:
        from evaluation.benchmark_runner import run_full_benchmark
        run_full_benchmark(model_id, checkpoint_path)
    except Exception as e:
        logger.warning("Benchmark failed for %s: %s", model_id, e)


# --------------------------------------------------------------------------
# Comparison report
# --------------------------------------------------------------------------

def generate_comparison_report(results_root: str = "results") -> None:
    """Aggregate all benchmark_results.json files into RESULTS_COMPARISON.md."""
    import json
    import glob

    rows = []
    for jf in sorted(glob.glob(f"{results_root}/*/benchmark_results.json")):
        try:
            with open(jf) as f:
                d = json.load(f)
            mid = Path(jf).parent.name
            rows.append((
                mid,
                d.get("efficiency", {}).get("param_count_M", "-"),
                d.get("synthetic", {}).get("ncc_mean", "-"),
                d.get("real_qlin", {}).get("ncc_mean", "-"),
                d.get("real_qlin", {}).get("rmse_mean", "-"),
                d.get("synthetic", {}).get("false_positive_rate", "-"),
                d.get("efficiency", {}).get("inference_latency_ms_b1", "-"),
            ))
        except Exception:
            pass

    lines = [
        "# Results Comparison\n",
        "| Model | Params(M) | NCC-Synth | NCC-Real | RMSE-Real | FPR | Latency(ms) |\n",
        "|---|---|---|---|---|---|---|\n",
    ]
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} |\n")

    out = Path("docs/RESULTS_COMPARISON.md")
    out.parent.mkdir(exist_ok=True)
    out.write_text("".join(lines))
    logger.info("Comparison report written to %s", out)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="AE-PD Resilient MLflow Orchestrator")
    parser.add_argument("--resume", action="store_true", help="Resume from session_state.json (recommended)")
    parser.add_argument("--run-all",  action="store_true", help="Train all registry models (continues if session exists)")
    parser.add_argument("--model",    type=str, default=None, help="Train a single model ID")
    parser.add_argument("--hparams", type=str, default=None, help="YAML overrides for training config")
    parser.add_argument("--registry", type=str, default="config/model_registry.yaml")
    parser.add_argument("--training-config", type=str, default="config/training_config.yaml")
    parser.add_argument("--results",  type=str, default=None, help="Override results root (default from training config)")
    parser.add_argument("--device",   type=str, default="cuda")
    parser.add_argument("--run-name", type=str, default=None, help="Custom run name for single model")
    args = parser.parse_args()

    registry = load_registry(args.registry)
    training_cfg = load_training_config(args.training_config, args.hparams)
    results_root = args.results or str(training_cfg.get("results_root", "results"))

    all_model_ids = list(registry.keys())
    def model_complexity(mid: str) -> int:
        e = registry[mid]
        return sum(int(e.get(k, 0)) for k in
                    ("use_bigru", "use_swin", "use_wavelet_pooling", "use_attention_gates"))
    all_model_ids.sort(key=model_complexity)

    if args.model:
        if args.model not in registry:
            logger.error("Model '%s' not in registry.", args.model)
            sys.exit(1)
        model_ids_to_run = [args.model]
    else:
        model_ids_to_run = all_model_ids

    resilience = ResilienceManager(project_root=".", model_id="")
    resilience.on_training_start(model_ids_to_run)
    resume_info = get_resume_point("session_state.json")
    skip_models = set(resume_info.get("skip_models", []))

    logger.info("Queue (%d models): %s", len(model_ids_to_run), model_ids_to_run)
    if skip_models:
        logger.info("Already done/skipped: %s", sorted(skip_models))
    if resume_info.get("resume_model"):
        logger.info("Will resume model: %s", resume_info["resume_model"])

    for model_id in model_ids_to_run:
        if model_id in skip_models:
            logger.info("Skipping %s (done/skipped)", model_id)
            continue

        resilience.model_id = model_id
        resilience.on_model_start(model_id)

        model_state = None
        try:
            model_state = resilience.load_session()
        except Exception:
            model_state = None

        resume_ckpt = None
        mlflow_run_id = None
        if args.resume and model_state:
            resume_ckpt = model_state.get("best_checkpoint")
            mlflow_run_id = model_state.get("mlflow_run_id")

        merged = merge_model_flags_from_training(model_id, registry[model_id], training_cfg)
        logger.info(
            "Model flags (merged): attn=%s swin=%s loss=%s",
            merged.get("use_attention_gates"),
            merged.get("use_swin"),
            merged.get("loss_config"),
        )
        history = run_model(
            model_id=model_id,
            cfg=merged,
            device=args.device,
            results_root=results_root,
            training_cfg=training_cfg,
            resilience=resilience,
            resume_ckpt=resume_ckpt,
            mlflow_run_id=mlflow_run_id,
        )

        try:
            final_metrics = {}
            if history.get("val_ncc"):
                final_metrics["val_ncc_best"] = max(history["val_ncc"])
            resilience.on_model_complete(model_id, final_metrics)
        except Exception:
            pass

    generate_comparison_report(results_root)


if __name__ == "__main__":
    main()
