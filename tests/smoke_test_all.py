"""
tests/smoke_test_all.py — One-command health check for the AE-PD pipeline.

Must pass before any training resumes.

Run:
    python tests/smoke_test_all.py
    Expected: 7/7 tests passed — system ready
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Try Model"))
os.chdir(PROJECT_ROOT)

import numpy as np
import torch
import yaml

(PROJECT_ROOT / "logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "smoke_test.log", mode="w"),
    ],
)
logger = logging.getLogger("smoke_test")

PASSED, FAILED = 0, 0


def _run(name: str, fn):
    global PASSED, FAILED
    logger.info("=" * 60)
    logger.info("TEST: %s", name)
    try:
        fn()
        PASSED += 1
        logger.info("  PASSED")
    except Exception as e:
        FAILED += 1
        logger.error("  FAILED — %s", e)
        traceback.print_exc()


# =====================================================================
# 1. Signal generation
# =====================================================================
def test_signal_generation():
    from mr_tae_fusion.data.pulse_generators import generate_pd_signal, verify_frequency_content
    from mr_tae_fusion.config import SignalConfig

    cfg = SignalConfig()
    signals, masks = [], []
    for i in range(10):
        sig_type = ["A", "B", "C", "D", "E", "F"][i % 6]
        t, clean, mask = generate_pd_signal(sig_type, cfg)
        signals.append(clean)
        masks.append(mask)

    batch = np.stack(signals)
    mask_batch = np.stack(masks)
    assert batch.shape == (10, 2048), f"Shape mismatch: {batch.shape}"
    assert not np.any(np.isnan(batch)), "NaN in signals"
    assert not np.any(np.isinf(batch)), "Inf in signals"

    for i in range(10):
        verify_frequency_content(signals[i], cfg.sample_rate)

    non_zero_masks = sum(1 for m in masks if m.max() > 0)
    assert non_zero_masks >= 8, f"Only {non_zero_masks}/10 windows have non-zero masks"
    logger.info("  10 windows generated, shape (10, 2048), PSD peaks in AE band")


# =====================================================================
# 2. Noise injection
# =====================================================================
def test_noise_injection():
    from mr_tae_fusion.data.noise_generators import (
        generate_wgn, generate_impulsive_noise, generate_powerline_noise,
        generate_narrowband_noise, generate_magnetostrictive_hum,
        add_noise_at_snr,
    )
    from mr_tae_fusion.data.pulse_generators import generate_pd_signal
    from mr_tae_fusion.config import SignalConfig

    cfg = SignalConfig()
    fs = cfg.sample_rate
    t = np.arange(cfg.signal_length) / fs
    _, clean, _ = generate_pd_signal("C", cfg)
    rng = np.random.default_rng(42)

    noise_fns = {
        "WGN": lambda: generate_wgn(len(clean), 0.1, rng),
        "impulsive": lambda: generate_impulsive_noise(len(clean), 0.01, rng=rng),
        "powerline": lambda: generate_powerline_noise(t),
        "NBI": lambda: generate_narrowband_noise(t, rng=rng),
        "magnetostrictive": lambda: generate_magnetostrictive_hum(t),
    }

    target_snr = 0.0
    for name, gen in noise_fns.items():
        noise = gen()
        noisy, actual_snr = add_noise_at_snr(clean, noise, target_snr)
        diff = abs(actual_snr - target_snr)
        assert diff < 3.0, f"{name}: actual SNR {actual_snr:.1f} dB, target {target_snr} dB, diff {diff:.1f}"
        logger.info("  %s: target=%.0f dB, actual=%.1f dB (diff=%.2f)", name, target_snr, actual_snr, diff)


# =====================================================================
# 3. Every model builds and forward-passes
# =====================================================================
def test_model_builds():
    from mr_tae_fusion.models.mr_tae_fusion import create_model
    from mr_tae_fusion.models.ablation_config import AblationConfig
    from mr_tae_fusion.config import Config

    with open(PROJECT_ROOT / "config" / "model_registry.yaml") as f:
        registry = yaml.safe_load(f)["models"]

    master_cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_id, flags in registry.items():
        ablation = AblationConfig(
            use_wavelet_pooling=flags.get("use_wavelet_pooling", True),
            use_bigru=flags.get("use_bigru", True),
            use_swin=flags.get("use_swin", True),
            use_attention_gates=flags.get("use_attention_gates", True),
            use_mtl=flags.get("use_mtl", True),
            loss_config=flags.get("loss_config", "A"),
            model_id=model_id,
        )
        model = create_model(master_cfg.model, ablation).to(device)
        x = torch.randn(2, 1, 2048, device=device)
        with torch.no_grad():
            denoised, seg = model(x)

        assert denoised.shape == (2, 1, 2048), f"{model_id}: denoised shape {denoised.shape}"
        assert not torch.isnan(denoised).any(), f"{model_id}: NaN in output"
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info("  %s: %.2f M params, output OK", model_id, n_params)

        del model
        torch.cuda.empty_cache()


# =====================================================================
# 4. Loss functions compute without error
# =====================================================================
def test_loss_functions():
    from mr_tae_fusion.training.losses import build_loss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for config_name, use_mtl in [("A", True), ("B", True), ("C", False), ("D", True)]:
        pred_signal = torch.randn(2, 1, 2048, device=device, requires_grad=True)
        target_signal = torch.randn(2, 1, 2048, device=device)
        pred_seg = torch.randn(2, 5, 2048, device=device, requires_grad=True)
        target_seg = torch.randint(0, 5, (2, 2048), device=device)

        loss_fn = build_loss(config_name, use_mtl).to(device)
        loss, loss_dict = loss_fn(pred_signal, target_signal, pred_seg, target_seg)
        assert loss.dim() == 0, f"Config {config_name}: loss not scalar"
        assert torch.isfinite(loss), f"Config {config_name}: loss not finite ({loss.item():.4f})"
        loss.backward()
        logger.info("  Loss config %s (MTL=%s): %.4f — backward OK", config_name, use_mtl, loss.item())


# =====================================================================
# 5. Trainer runs 2 epochs on smallest model
# =====================================================================
def test_trainer_2_epochs():
    from mr_tae_fusion.models.mr_tae_fusion import create_model
    from mr_tae_fusion.models.ablation_config import AblationConfig
    from mr_tae_fusion.config import Config
    from mr_tae_fusion.data.dataset import PDSignalDataset
    from mr_tae_fusion.training.losses import build_loss
    from training import Trainer, CurricularScheduler
    from torch.utils.data import DataLoader

    master_cfg = Config()
    ablation = AblationConfig(
        use_wavelet_pooling=True, use_bigru=True, use_swin=False,
        use_attention_gates=True, use_mtl=True, loss_config="A",
        model_id="smoke-test-noSwin",
    )
    model = create_model(master_cfg.model, ablation)
    criterion = build_loss("A", True)

    train_ds = PDSignalDataset(config=master_cfg, num_samples=16, mode="train", seed=99)
    val_ds = PDSignalDataset(config=master_cfg, num_samples=8, mode="val", seed=100)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    sched = CurricularScheduler(phase1_epochs=1, phase2_epochs=1, phase3_epochs=0)

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            model=model, criterion=criterion,
            train_loader=train_loader, val_loader=val_loader,
            device="cuda" if torch.cuda.is_available() else "cpu",
            save_dir=tmpdir, model_id="smoke-test",
            scheduler=sched, lr=3e-4, weight_decay=1e-4, grad_clip=1.0,
        )
        history = trainer.train()

    assert len(history["val_ncc"]) == 2, f"Expected 2 epochs, got {len(history['val_ncc'])}"
    assert history["val_ncc"][-1] > -1, f"val_ncc too low: {history['val_ncc'][-1]}"
    logger.info("  2-epoch train complete, val_ncc = %.4f", history["val_ncc"][-1])


# =====================================================================
# 6. Resilience system
# =====================================================================
def test_resilience():
    import tempfile, shutil
    from training.resilience import ResilienceManager

    with tempfile.TemporaryDirectory() as tmpdir:
        rm = ResilienceManager(project_root=tmpdir, model_id="test-model")
        rm.on_training_start(["test-model"])

        for epoch in range(20):
            ncc = 0.1 + epoch * 0.02
            decision = rm.on_epoch_end(epoch, phase=1, metrics={
                "val_ncc": ncc, "train_loss": 1.0 - ncc, "val_rmse": 0.5,
            })
            assert decision["action"] == "continue", \
                f"Epoch {epoch}: early stop triggered on improving model"

        for epoch in range(20, 28):
            decision = rm.on_epoch_end(epoch, phase=2, metrics={
                "val_ncc": 0.5, "train_loss": 0.5, "val_rmse": 0.5,
            })

        sess_path = Path(tmpdir) / "session_state.json"
        assert sess_path.exists(), "session_state.json not written"
        with open(sess_path) as f:
            sess = json.load(f)
        assert "test-model" in sess["models"], "Model not in session"

        prompt_path = Path(tmpdir) / "RESUME_PROMPT.md"
        assert prompt_path.exists(), "RESUME_PROMPT.md not written"
        logger.info("  Resilience system OK, session + resume prompt written")


# =====================================================================
# 7. MLflow logging
# =====================================================================
def test_mlflow():
    import mlflow

    exp_name = "smoke-test-experiment"
    mlflow.set_experiment(exp_name)
    with mlflow.start_run(run_name="smoke-test-run") as run:
        for i in range(5):
            mlflow.log_metric("smoke_metric", float(i) * 0.1, step=i)
        mlflow.log_param("smoke_param", "test_value")

    runs = mlflow.search_runs(experiment_names=[exp_name])
    assert len(runs) >= 1, f"Expected >=1 run, found {len(runs)}"
    logger.info("  MLflow: %d runs found in '%s'", len(runs), exp_name)


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    logger.info("AE-PD Smoke Test Suite")
    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    _run("1. Signal generation", test_signal_generation)
    _run("2. Noise injection", test_noise_injection)
    _run("3. Model builds & forward pass", test_model_builds)
    _run("4. Loss functions", test_loss_functions)
    _run("5. Trainer 2 epochs", test_trainer_2_epochs)
    _run("6. Resilience system", test_resilience)
    _run("7. MLflow logging", test_mlflow)

    logger.info("=" * 60)
    logger.info("RESULT: %d/%d tests passed", PASSED, PASSED + FAILED)
    if FAILED:
        logger.error("%d test(s) FAILED", FAILED)
        sys.exit(1)
    else:
        logger.info("System ready — all smoke tests passed")
