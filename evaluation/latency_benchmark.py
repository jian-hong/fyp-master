"""
CUDA-synchronized inference latency benchmark for all 13 MR-TAE variants.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Try Model"))

from evaluation.benchmark_runner import (  # noqa: E402
    _load_checkpoint_state,
    ablation_for_checkpoint,
    load_model_registry,
)
from mr_tae_fusion.config import Config  # noqa: E402
from mr_tae_fusion.models.mr_tae_fusion import create_model  # noqa: E402

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

OUT_PATH = PROJECT_ROOT / "results" / "definitive_benchmark" / "latency.json"


def load_model_from_checkpoint(model_id: str) -> torch.nn.Module:
    ckpt_path = PROJECT_ROOT / "results" / model_id / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ablation, _ = ablation_for_checkpoint(model_id, str(ckpt_path))
    model = create_model(Config().model, ablation)
    model.load_state_dict(_load_checkpoint_state(ckpt), strict=False)
    return model.to(device).eval()


def measure_latency_correct(
    model: torch.nn.Module,
    batch_size: int = 1,
    seq_len: int = 2048,
    n_warmup: int = 50,
    n_runs: int = 200,
    device: str = "cuda",
) -> float:
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval()
    x = torch.randn(batch_size, 1, seq_len, device=dev)

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()

    times: List[float] = []
    with torch.no_grad():
        for _ in range(n_runs):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    times.sort()
    trim = n_runs // 10
    trimmed = times[trim:-trim] if trim > 0 else times
    return float(sum(trimmed) / len(trimmed))


def measure_latency_b32(
    model: torch.nn.Module,
    n_warmup: int = 20,
    n_runs: int = 50,
    device: str = "cuda",
) -> float:
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval()
    x = torch.randn(32, 1, 2048, device=dev)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    times: List[float] = []
    with torch.no_grad():
        for _ in range(n_runs):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) / 32.0 * 1000.0)
    times.sort()
    trim = max(n_runs // 10, 1)
    trimmed = times[trim:-trim]
    return float(sum(trimmed) / len(trimmed))


def count_params_m(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def run_all(device: str = "cuda") -> Dict[str, Any]:
    reg = load_model_registry()
    out: Dict[str, Any] = {}
    for mid in MODELS:
        ck = PROJECT_ROOT / "results" / mid / "checkpoints" / "best.pt"
        if not ck.exists():
            print(f"[skip] {mid}: no checkpoint")
            continue
        try:
            model = load_model_from_checkpoint(mid)
            lat_b1 = measure_latency_correct(model, device=device)
            lat_b32 = measure_latency_b32(model, device=device)
            params_m = count_params_m(model)
            out[mid] = {
                "lat_ms_b1": lat_b1,
                "lat_ms_b32": lat_b32,
                "params_M": params_m,
                "registry_attn": reg.get(mid, {}).get("use_attention_gates"),
            }
            print(f"{mid}: b1={lat_b1:.3f}ms b32={lat_b32:.3f}ms params={params_m:.2f}M")
        except Exception as e:
            print(f"[error] {mid}: {e}")
            out[mid] = {"error": str(e)}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved {OUT_PATH}")
    return out


if __name__ == "__main__":
    run_all()
