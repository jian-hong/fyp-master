"""
generate_method_comparison_figures.py
======================================
Render waveform comparison figures for all 13 MR-TAE ablation variants.

FIXES vs. broken version
--------------------------
1. MODEL_PKG now correctly points to FYP_MASTER\\Try Model (not OneDrive\\Try Model).
   Root cause: mlflow_orchestrator.py uses PROJECT_ROOT / "Try Model", which is
   D:\\New folder (2)\\Data\\FYP_MASTER\\Try Model — the generated script had the
   OneDrive path which never contained the mr_tae_fusion package.
2. Path bootstrap prints what it actually added so you can diagnose future failures
   without re-reading the code.
3. CUDA fallback: if torch.cuda.is_available() is False, a clear reason is printed.
   Most likely cause: script was run without activating the FYP venv.
   Fix: activate venv first, then run.
4. Checkpoint discovery also searches NEW FOCUS/results in case you trained there.
5. All other logic (model build, inference, figure layout) is unchanged from
   the working reviewed version.

Run from ANY directory:
    # activate venv first!
    .\\venv\\Scripts\\activate           # PowerShell
    source venv/bin/activate            # bash

    python "NEW FOCUS\\generate_method_comparison_figures.py"
"""

from __future__ import annotations

import sys
import warnings
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# ① PATH BOOTSTRAP  ← ROOT CAUSE OF ORIGINAL FAILURE WAS HERE
# ---------------------------------------------------------------------------
# The mr_tae_fusion package lives at:
#   FYP_MASTER\Try Model\mr_tae_fusion\
# NOT at:
#   OneDrive_2025-12-13\Try Model\mr_tae_fusion\    ← this was the wrong path
#
# Evidence: mlflow_orchestrator.py line 34:
#   sys.path.insert(0, str(PROJECT_ROOT / "Try Model"))
# where PROJECT_ROOT = FYP_MASTER (parents[1] of pipeline/mlflow_orchestrator.py)

FYP_ROOT  = Path(r"D:\New folder (2)\Data\FYP_MASTER")
MODEL_PKG = FYP_ROOT / "Try Model"               # ← FIXED (was OneDrive_2025-12-13\Try Model)

_paths_to_add = [
    str(MODEL_PKG),
    str(FYP_ROOT / "NEW FOCUS"),
    str(FYP_ROOT),
]
for p in _paths_to_add:
    if p not in sys.path:
        sys.path.insert(0, p)

print("── Path bootstrap ──────────────────────────────────────────────────")
for p in _paths_to_add:
    pkg_exists = (Path(p) / "mr_tae_fusion").is_dir()
    print(f"  {'[mr_tae_fusion FOUND]' if pkg_exists else '[no mr_tae_fusion]':26s}  {p}")
print()

# ---------------------------------------------------------------------------
# ② CUDA DIAGNOSTIC
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Device: cuda  ({torch.cuda.get_device_name(0)})")
else:
    device = torch.device("cpu")
    print("Device: cpu")
    print("  → If you expected CUDA, make sure you activated the FYP venv before running:")
    print(r"      D:\New folder (2)\Data\FYP_MASTER\venv\Scripts\activate")
    print("  → Then verify: python -c \"import torch; print(torch.cuda.is_available())\"")
print()

# ---------------------------------------------------------------------------
# ③ ABLATION REGISTRY  (mirrors config/model_registry.yaml exactly)
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, dict] = {
    # Phase-3a: single-component removal
    "MR-TAE-FULL":            dict(W=True,  B=True,  S=True,  A=True,  M=True,  loss="A"),
    "MR-TAE-noBiGRU":         dict(W=True,  B=False, S=True,  A=True,  M=True,  loss="A"),
    "MR-TAE-noSwin":          dict(W=True,  B=True,  S=False, A=True,  M=True,  loss="A"),
    "MR-TAE-noAttn":          dict(W=True,  B=True,  S=True,  A=False, M=True,  loss="A"),
    "MR-TAE-noMTL":           dict(W=True,  B=True,  S=True,  A=True,  M=False, loss="C"),
    "MR-TAE-noWavelet":       dict(W=False, B=True,  S=True,  A=True,  M=True,  loss="A"),
    # Phase-3b: two-component removal
    "MR-TAE-noSwin-noBiGRU":  dict(W=True,  B=False, S=False, A=True,  M=True,  loss="A"),
    "MR-TAE-noSwin-noAttn":   dict(W=True,  B=True,  S=False, A=False, M=True,  loss="A"),
    "MR-TAE-noBiGRU-noAttn":  dict(W=True,  B=False, S=True,  A=False, M=True,  loss="A"),
    # Loss variants
    "MR-TAE-FULL-LossB":      dict(W=True,  B=True,  S=True,  A=True,  M=True,  loss="B"),
    "MR-TAE-FULL-LossC":      dict(W=True,  B=True,  S=True,  A=True,  M=True,  loss="C"),
    # Cross-architecture baselines
    "MWCNN-BiGRU":            dict(W=True,  B=True,  S=False, A=False, M=False, loss="C"),
    "UNet-BiGRU-Swin":        dict(W=False, B=True,  S=True,  A=False, M=False, loss="C"),
}

EVAL_SNRS = [-15, -5, 5]

# ---------------------------------------------------------------------------
# ④ HELPERS
# ---------------------------------------------------------------------------

def find_checkpoint(model_id: str) -> Path | None:
    candidates = [
        FYP_ROOT / "results" / model_id / "checkpoints" / "best.pt",
        FYP_ROOT / "results" / model_id / "best.pt",
        FYP_ROOT / "NEW FOCUS" / "results" / model_id / "checkpoints" / "best.pt",
        FYP_ROOT / "NEW FOCUS" / "results" / model_id / "best.pt",
    ]
    # Newest *.pt in the standard checkpoint folder wins
    glob_root = FYP_ROOT / "results" / model_id / "checkpoints"
    if glob_root.is_dir():
        pts = sorted(glob_root.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        candidates = pts + candidates
    for c in candidates:
        if Path(c).exists():
            return Path(c)
    return None


def build_model(model_id: str, flags: dict) -> torch.nn.Module | None:
    try:
        from mr_tae_fusion.models.ablation_config import AblationConfig
        from mr_tae_fusion.models.mr_tae_fusion import create_model
        from mr_tae_fusion.config import Config

        ablation = AblationConfig(
            use_wavelet_pooling=flags["W"],
            use_bigru=flags["B"],
            use_swin=flags["S"],
            use_attention_gates=flags["A"],
            use_mtl=flags["M"],
            loss_config=flags["loss"],
            model_id=model_id,
        )
        master_cfg = Config()
        model = create_model(master_cfg.model, ablation)
        return model
    except ImportError as exc:
        # Only print the import error once (first model); subsequent ones are the same root cause.
        print(f"  [import-error] {exc}")
        print(f"  Searched paths:")
        for p in sys.path[:6]:
            print(f"    {p}")
        return None
    except Exception as exc:
        print(f"  [error] build_model({model_id}): {exc}")
        return None


def load_model(model_id: str, flags: dict, dev: torch.device) -> torch.nn.Module | None:
    ckpt_path = find_checkpoint(model_id)
    if ckpt_path is None:
        print(f"  [skip ] {model_id}: no checkpoint in {FYP_ROOT / 'results' / model_id}")
        return None

    model = build_model(model_id, flags)
    if model is None:
        return None

    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            sd = state.get("model_state_dict", state.get("model_state", state.get("state_dict", state)))
        else:
            sd = state
        model.load_state_dict(sd, strict=False)
        model.eval().to(dev)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  [ok   ] {model_id:35s} {n_params:.2f}M params  ← {ckpt_path.relative_to(FYP_ROOT)}")
        return model
    except Exception as exc:
        print(f"  [error] {model_id}: weight loading failed – {exc}")
        return None

# ---------------------------------------------------------------------------
# ⑤ TEST-SET SAMPLING
# ---------------------------------------------------------------------------

def load_test_windows(snr_db: float, n_windows: int = 1, seed: int = 42):
    rng = np.random.default_rng(seed)
    KNOWN_TEST_PATHS = [
        FYP_ROOT / "data" / "test_synthetic_ae.pt",
        FYP_ROOT / "NEW FOCUS" / "data" / "test_synthetic_ae.pt",
    ]

    for tp in KNOWN_TEST_PATHS:
        if not tp.exists():
            continue
        dataset = torch.load(tp, map_location="cpu", weights_only=False)
        if isinstance(dataset, dict):
            noisy_all = dataset["noisy"]
            clean_all = dataset["clean"]
        elif hasattr(dataset, "tensors"):
            noisy_all, clean_all = dataset.tensors[0], dataset.tensors[1]
        else:
            continue

        # SNR-stratified selection if metadata present
        snr_key = next((k for k in ("snr_db", "snr_labels", "snr") if k in dataset), None)
        if isinstance(dataset, dict) and snr_key:
            snr_tensor = dataset[snr_key]
            mask = (snr_tensor > snr_db - 2.5) & (snr_tensor < snr_db + 2.5)
            idx = mask.nonzero(as_tuple=True)[0]
            if len(idx) >= n_windows:
                g = torch.Generator()
                g.manual_seed(seed)
                chosen = idx[torch.randperm(len(idx), generator=g)[:n_windows]]
                noisy = noisy_all[chosen].float()
                clean = clean_all[chosen].float()
                if noisy.dim() == 2:
                    noisy = noisy.unsqueeze(1)
                    clean = clean.unsqueeze(1)
                return noisy.numpy(), clean.numpy()

        # Fallback: re-corrupt at desired SNR
        total = noisy_all.shape[0]
        chosen_idx = rng.choice(total, size=n_windows, replace=False)
        clean = clean_all[chosen_idx].float()
        if clean.dim() == 2:
            clean = clean.unsqueeze(1)
        power = (clean ** 2).mean(dim=-1, keepdim=True)
        sigma = torch.sqrt(power / (10 ** (snr_db / 10))).clamp(min=1e-8)
        noisy = clean + torch.randn_like(clean) * sigma
        return noisy.numpy(), clean.numpy()

    # Gaussian scaffold — pure rendering test when no real data found
    print(f"  [warn ] No test set found. Using Gaussian scaffold for SNR={snr_db} dB.")
    L = 2048
    clean = rng.standard_normal((n_windows, 1, L)).astype(np.float32) * 0.5
    for i in range(n_windows):
        for _ in range(rng.integers(3, 8)):
            t = rng.integers(50, L - 50)
            w = rng.integers(8, 25)
            clean[i, 0, t:t+w] += rng.uniform(1.5, 4.0)
    power = (clean ** 2).mean(axis=-1, keepdims=True)
    sigma = np.sqrt(power / (10 ** (snr_db / 10))).clip(min=1e-8)
    noisy = clean + rng.standard_normal(clean.shape).astype(np.float32) * sigma
    return noisy, clean

# ---------------------------------------------------------------------------
# ⑥ METRICS
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model: torch.nn.Module, noisy: np.ndarray, dev: torch.device) -> np.ndarray:
    x = torch.from_numpy(noisy).float().to(dev)
    out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    elif isinstance(out, dict):
        out = out.get("denoised") or out.get("signal") or next(iter(out.values()))
    return out.cpu().numpy()


def ncc_batch(pred: np.ndarray, clean: np.ndarray) -> float:
    p = pred.squeeze(1)
    c = clean.squeeze(1)
    nccs = []
    for i in range(p.shape[0]):
        pp = p[i] - p[i].mean()
        cc = c[i] - c[i].mean()
        denom = np.linalg.norm(pp) * np.linalg.norm(cc) + 1e-10
        nccs.append(float(np.dot(pp, cc) / denom))
    return float(np.mean(nccs))

# ---------------------------------------------------------------------------
# ⑦ FIGURE RENDERING
# ---------------------------------------------------------------------------

COLOURS = {
    "noisy":    "#b0b0b0",
    "clean":    "#2d6a4f",
    "denoised": "#e76f51",
}


def render_figure_one_snr(
    model_results: dict[str, np.ndarray],
    noisy: np.ndarray,
    clean: np.ndarray,
    snr_db: float,
    out_path: Path,
) -> None:
    model_ids = list(model_results.keys())
    n_models = len(model_ids)
    if n_models == 0:
        print(f"  [warn ] No models loaded for SNR={snr_db} dB – skipping figure.")
        return

    fig = plt.figure(figsize=(14, max(2.5 * n_models, 4)), constrained_layout=True)
    fig.suptitle(
        f"Denoising comparison  |  Input SNR = {snr_db:+.0f} dB",
        fontsize=13, fontweight="bold",
    )
    gs = gridspec.GridSpec(n_models, 2, figure=fig, hspace=0.08, wspace=0.08)
    t = np.arange(noisy.shape[-1])

    for row, mid in enumerate(model_ids):
        den = model_results[mid]
        ncc_val = ncc_batch(den, clean)

        ax_sig = fig.add_subplot(gs[row, 0])
        ax_res = fig.add_subplot(gs[row, 1])

        # Signal overlay
        ax_sig.plot(t, noisy[0, 0], color=COLOURS["noisy"], alpha=0.55, lw=0.6,
                    label="Noisy" if row == 0 else "")
        ax_sig.plot(t, clean[0, 0], color=COLOURS["clean"], lw=1.1,
                    label="Clean GT" if row == 0 else "")
        ax_sig.plot(t, den[0, 0],   color=COLOURS["denoised"], lw=0.9, alpha=0.9,
                    label="Denoised" if row == 0 else "")
        ax_sig.set_ylabel(mid, fontsize=7, rotation=0, labelpad=80, va="center")
        ax_sig.tick_params(labelsize=6)
        ax_sig.set_xlim(0, len(t) - 1)

        # Residual
        residual = den[0, 0] - clean[0, 0]
        ax_res.plot(t, residual, color="#8338ec", lw=0.7, alpha=0.8)
        ax_res.axhline(0, color="k", lw=0.5, linestyle="--")
        ax_res.set_title(f"NCC={ncc_val:.4f}", fontsize=7, pad=2)
        ax_res.tick_params(labelsize=6)
        ax_res.set_xlim(0, len(t) - 1)

        if row == 0:
            ax_sig.legend(fontsize=6, loc="upper right", framealpha=0.6)
            ax_sig.set_title("Signal overlay (Clean / Noisy / Denoised)", fontsize=8)
            ax_res.set_title(
                f"Residual (denoised − clean)  |  NCC={ncc_val:.4f}", fontsize=7, pad=2
            )

        if row < n_models - 1:
            ax_sig.set_xticklabels([])
            ax_res.set_xticklabels([])
        else:
            ax_sig.set_xlabel("Sample index", fontsize=7)
            ax_res.set_xlabel("Sample index", fontsize=7)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig  ] saved → {out_path}")


def _render_ncc_summary(summary: dict, loaded: dict, out_dir: Path) -> None:
    snr_keys = list(summary.keys())
    model_ids = list(loaded.keys())
    x = np.arange(len(model_ids))
    width = 0.25
    colours_bar = ["#e63946", "#457b9d", "#2a9d8f"]

    fig, ax = plt.subplots(figsize=(max(12, len(model_ids) * 0.9), 5))
    for i, (sk, col) in enumerate(zip(snr_keys, colours_bar)):
        vals = [summary[sk].get(mid, 0.0) for mid in model_ids]
        ax.bar(x + i * width, vals, width, label=sk.replace("snr_", "SNR "), color=col, alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(model_ids, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("NCC")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-SNR NCC — All Ablation Variants", fontsize=12)
    ax.legend(fontsize=9)
    ax.axhline(0.9, color="k", lw=0.8, linestyle="--", alpha=0.4)
    fig.tight_layout()
    bar_path = out_dir / "ncc_summary_bar.png"
    fig.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig  ] NCC summary bar → {bar_path}")

# ---------------------------------------------------------------------------
# ⑧ MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Rendering {len(MODEL_REGISTRY)} models at SNRs {EVAL_SNRS} dB\n")

    print("Loading models …")
    loaded: dict[str, torch.nn.Module] = {}
    _import_failed = False
    for mid, flags in MODEL_REGISTRY.items():
        if _import_failed:
            # After the first import error the rest will fail identically;
            # skip loop and let the fatal message below guide the user.
            break
        m = load_model(mid, flags, device)
        if m is not None:
            loaded[mid] = m
        elif mid == list(MODEL_REGISTRY.keys())[0]:
            # Check if failure was import-related
            try:
                import mr_tae_fusion  # noqa: F401
            except ImportError:
                _import_failed = True

    print(f"\nLoaded {len(loaded)}/{len(MODEL_REGISTRY)} models.\n")

    if not loaded:
        print("[FATAL] No models could be loaded.")
        print()
        print("Most likely cause A — wrong venv (mr_tae_fusion not installed):")
        print(r"  Activate:  D:\New folder (2)\Data\FYP_MASTER\venv\Scripts\activate")
        print(r"  Verify:    python -c ""import mr_tae_fusion; print('ok')"" ")
        print()
        print("Most likely cause B — mr_tae_fusion package is missing from Try Model\\:")
        print(f"  Expected:  {MODEL_PKG / 'mr_tae_fusion' / '__init__.py'}")
        print(f"  Exists?    {(MODEL_PKG / 'mr_tae_fusion' / '__init__.py').exists()}")
        print()
        print("Most likely cause C — checkpoints not at expected location:")
        print(f"  Expected:  {FYP_ROOT / 'results' / 'MR-TAE-FULL-LossB' / 'checkpoints' / 'best.pt'}")
        return

    OUT_DIR = FYP_ROOT / "NEW FOCUS" / "results" / "method_comparison_figures"
    summary: dict = {}
    rendered_total = 0

    for snr in EVAL_SNRS:
        print(f"\n── SNR = {snr:+d} dB ──────────────────────────────────────────")
        noisy, clean = load_test_windows(snr_db=snr, n_windows=1, seed=snr + 100)

        model_results: dict[str, np.ndarray] = {}
        snr_summary: dict[str, float] = {}

        for mid, model in loaded.items():
            try:
                den = run_inference(model, noisy, device)
                ncc_val = ncc_batch(den, clean)
                model_results[mid] = den
                snr_summary[mid] = ncc_val
                print(f"    {mid:35s}  NCC={ncc_val:.4f}")
            except Exception as exc:
                print(f"    [error] {mid}: inference failed – {exc}")

        summary[f"snr_{snr:+d}dB"] = snr_summary
        rendered_total += len(model_results)

        fig_path = OUT_DIR / f"comparison_snr_{snr:+d}dB.png"
        render_figure_one_snr(model_results, noisy, clean, float(snr), fig_path)

    if summary and loaded:
        _render_ncc_summary(summary, loaded, OUT_DIR)

    json_path = OUT_DIR / "method_comparison_summary.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary JSON → {json_path}")
    print(f"Rendered {rendered_total} (model × SNR) combinations.")


if __name__ == "__main__":
    main()