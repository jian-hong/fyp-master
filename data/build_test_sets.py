"""
data/build_test_sets.py

Creates frozen evaluation test sets.

Run:
    python data/build_test_sets.py
    python data/build_test_sets.py --real-only
    python data/build_test_sets.py --real-only --max-grounded-windows 2000

Outputs:
    data/test_synthetic_ae.pt
    data/test_real_qlin.pt          — input-referenced eval (legacy)
    data/test_real_qlin_clean.pt    — [N,1,2048] z-scored clean windows (test split)
    data/test_real_qlin_noisy.pt    — [N*5,1,2048] composite noise at 5 SNRs
    data/test_real_qlin_snr_labels.pt — nominal SNR per noisy row
    data/real_noise_pool.pt
    docs/real_noise_injection_verify.png (300 DPI)
    docs/TEST_SET_HASHES.md
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import logging
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Try Model"))

from mr_tae_fusion.config import SignalConfig, NoiseConfig
from mr_tae_fusion.data.pulse_generators import generate_pd_signal
from mr_tae_fusion.data import noise_generators as noise_generators_module
from mr_tae_fusion.data.noise_generators import (
    generate_composite_noise,
    add_noise_at_snr,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_test_sets")

DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"

SNR_LEVELS = [-15, -10, -5, 0, 5]
WINDOWS_PER_SNR = 2000
TOTAL_SYNTHETIC = len(SNR_LEVELS) * WINDOWS_PER_SNR

SIGNAL_TYPES = ["A", "B", "C", "D", "E", "F"]
SEED = 42


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_synthetic_test_set() -> Path:
    """Generate 10 000 synthetic noisy/clean/mask windows at 5 SNR levels."""
    logger.info("Building synthetic test set (%d windows)...", TOTAL_SYNTHETIC)

    cfg = SignalConfig()
    noise_cfg = NoiseConfig()
    fs = cfg.sample_rate
    rng = np.random.default_rng(SEED)
    t = np.arange(cfg.signal_length) / fs

    all_noisy = np.zeros((TOTAL_SYNTHETIC, cfg.signal_length), dtype=np.float32)
    all_clean = np.zeros((TOTAL_SYNTHETIC, cfg.signal_length), dtype=np.float32)
    all_masks = np.zeros((TOTAL_SYNTHETIC, cfg.signal_length), dtype=np.int64)
    all_snr_labels = np.zeros(TOTAL_SYNTHETIC, dtype=np.float32)

    idx = 0
    for snr_db in SNR_LEVELS:
        for i in range(WINDOWS_PER_SNR):
            sig_type = SIGNAL_TYPES[i % len(SIGNAL_TYPES)]
            _, clean, mask = generate_pd_signal(sig_type, cfg)

            noise = generate_composite_noise(t, noise_cfg, rng)
            noisy, actual_snr = add_noise_at_snr(clean, noise, float(snr_db))

            all_noisy[idx] = noisy.astype(np.float32)
            all_clean[idx] = clean.astype(np.float32)
            all_masks[idx] = mask
            all_snr_labels[idx] = actual_snr
            idx += 1

        logger.info("  SNR %+d dB: %d windows generated", snr_db, WINDOWS_PER_SNR)

    out_path = DATA_DIR / "test_synthetic_ae.pt"
    torch.save({
        "noisy": torch.from_numpy(all_noisy),
        "clean": torch.from_numpy(all_clean),
        "masks": torch.from_numpy(all_masks),
        "snr_labels": torch.from_numpy(all_snr_labels),
    }, out_path)
    logger.info("Saved: %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
    return out_path


def collect_qlin_windows(
    window_len: int = 2048,
    stride: int | None = None,
    max_windows: int | None = None,
) -> tuple[np.ndarray, np.ndarray, Path | None, np.ndarray | None]:
    """Extract z-scored windows; return (windows [N,L], labels, first_mat_path, first_window).

    If ``max_windows`` is set, stop after that many windows (deterministic file order) to cap RAM.
    """
    stride = stride or window_len // 2

    qlin_dirs = [
        PROJECT_ROOT / "Datset from Q.lin",
        Path(r"D:\New folder (2)\Data\OneDrive_2025-12-13\Datset from Q.lin"),
    ]

    mat_files: list[Path] = []
    for d in qlin_dirs:
        if d.exists():
            mat_files.extend(sorted(d.glob("*.mat")))

    if not mat_files:
        logger.warning("No Q.Lin .mat files found.")
        return np.array([]), np.array([]), None, None

    try:
        import scipy.io as sio
    except ImportError:
        logger.error("scipy not installed — cannot load .mat files")
        return np.array([]), np.array([]), None, None

    all_windows: list[np.ndarray] = []
    all_labels: list[int] = []
    label_map = {"10mm": 0, "18mm": 1, "20mm": 2, "25mm": 3}

    first_mat_path: Path | None = None
    first_window: np.ndarray | None = None

    for mat_path in mat_files:
        mat_data = sio.loadmat(str(mat_path))
        data_key = None
        for k in mat_data:
            if not k.startswith("_") and isinstance(mat_data[k], np.ndarray):
                data_key = k
                break
        if data_key is None:
            logger.warning("  No array found in %s, skipping", mat_path.name)
            continue

        raw = mat_data[data_key].astype(np.float64)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)

        label_id = -1
        for tag, lid in label_map.items():
            if tag in mat_path.stem.lower().replace(".", ""):
                label_id = lid
                break

        logger.info("  %s: shape %s, label=%d", mat_path.name, raw.shape, label_id)

        for trace in raw:
            for start in range(0, len(trace) - window_len + 1, stride):
                window = trace[start:start + window_len]
                mu, sigma = window.mean(), window.std() + 1e-10
                wz = ((window - mu) / sigma).astype(np.float32)
                if first_window is None:
                    first_window = wz.copy()
                    first_mat_path = mat_path
                all_windows.append(wz)
                all_labels.append(label_id)
                if max_windows is not None and len(all_windows) >= max_windows:
                    break
            if max_windows is not None and len(all_windows) >= max_windows:
                break
        if max_windows is not None and len(all_windows) >= max_windows:
            break

    if not all_windows:
        return np.array([]), np.array([]), None, None

    return np.stack(all_windows), np.array(all_labels, dtype=np.int64), first_mat_path, first_window


def build_real_qlin_test_set(
    windows_arr: np.ndarray | None = None,
    labels_arr: np.ndarray | None = None,
    max_qlin_windows: int | None = None,
) -> Path | None:
    """Legacy input-referenced test set + noise pool (90/10 split)."""
    if windows_arr is None or labels_arr is None:
        windows_arr, labels_arr, _, _ = collect_qlin_windows(max_windows=max_qlin_windows)
    if len(windows_arr) == 0:
        logger.warning("No Q.Lin windows extracted")
        return None

    n = len(windows_arr)
    logger.info("Total Q.Lin windows: %d", n)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    n_noise = max(1, n // 10)
    noise_idx = perm[:n_noise]
    test_idx = perm[n_noise:]

    test_path = DATA_DIR / "test_real_qlin.pt"
    torch.save({
        "windows": torch.from_numpy(windows_arr[test_idx]),
        "class_labels": torch.from_numpy(labels_arr[test_idx]),
    }, test_path)
    logger.info("Saved: %s (%d windows, %.1f MB)",
                test_path, len(test_idx), test_path.stat().st_size / 1e6)

    pool_path = DATA_DIR / "real_noise_pool.pt"
    torch.save({
        "windows": torch.from_numpy(windows_arr[noise_idx]),
        "class_labels": torch.from_numpy(labels_arr[noise_idx]),
    }, pool_path)
    logger.info("Saved: %s (%d windows)", pool_path, len(noise_idx))

    return test_path


def _noise_psd_peak(noise_1d: np.ndarray, fs: float) -> tuple[float, float]:
    x = np.asarray(noise_1d, dtype=np.float64).ravel()
    x = x - x.mean()
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    psd = np.abs(spec) ** 2
    peak_i = int(np.argmax(psd))
    return float(freqs[peak_i]), float(np.sqrt(np.mean(x ** 2)))


def _measured_snr_db(clean: np.ndarray, noise: np.ndarray) -> float:
    sig_p = float(np.var(clean) + 1e-12)
    n_p = float(np.var(noise) + 1e-12)
    return float(10 * np.log10(sig_p / n_p))


def save_noise_verification_figure(
    clean_win: np.ndarray,
    source_mat: Path,
    noise_cfg: NoiseConfig,
    fs: float,
    t: np.ndarray,
) -> None:
    """5 rows × 3 panels; first window, 5 SNR levels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    noise_gen_path = Path(noise_generators_module.__file__).resolve()
    func_name = "generate_composite_noise"
    rng = np.random.default_rng(SEED)

    print(f"Noise verification: using {noise_gen_path}")
    print(f"Function: {func_name}")
    print(f"Q.Lin source: {source_mat.resolve()}")

    fig, axes = plt.subplots(len(SNR_LEVELS), 3, figsize=(14, 3 * len(SNR_LEVELS)))
    if len(SNR_LEVELS) == 1:
        axes = np.array([axes])

    seed_txt = (
        f"Q.Lin .mat: {source_mat.resolve()}\n"
        f"Noise module: mr_tae_fusion.data.noise_generators\n"
        f"File: {noise_gen_path}\n"
        f"Function: {func_name}()\n"
        f"Random seed: {SEED}"
    )
    fig.text(0.5, 0.98, seed_txt, ha="center", va="top", fontsize=8, family="monospace",
             transform=fig.transFigure, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.35))

    for row, snr_nom in enumerate(SNR_LEVELS):
        noise = generate_composite_noise(t, noise_cfg, rng)
        noisy, actual_snr = add_noise_at_snr(clean_win, noise, float(snr_nom))
        noise_only = noisy.astype(np.float64) - clean_win.astype(np.float64)

        peak_f, noise_rms = _noise_psd_peak(noise_only, fs)
        meas_snr = _measured_snr_db(clean_win, noise_only)

        axes[row, 0].plot(clean_win, lw=0.6)
        axes[row, 0].set_title(f'Clean Q.Lin — {source_mat.name}', fontsize=9)
        axes[row, 0].set_xlabel("sample")

        axes[row, 1].plot(noise_only, lw=0.6, color="C1")
        subt = (
            f'Composite noise — noise_generators.py:{func_name}()\n'
            f"RMS={noise_rms:.4f} | peak PSD freq={peak_f/1e3:.2f} kHz | "
            f"meas. SNR={meas_snr:.2f} dB"
        )
        axes[row, 1].set_title(subt, fontsize=8)

        axes[row, 2].plot(noisy, lw=0.6, color="C2")
        axes[row, 2].set_title(f"Noisy input @ {snr_nom:+d} dB (actual {actual_snr:.2f} dB)", fontsize=9)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    out_png = DOCS_DIR / "real_noise_injection_verify.png"
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved verification figure: %s", out_png)


def build_real_grounded_sets(
    max_grounded_windows: int | None = None,
    windows_arr: np.ndarray | None = None,
    labels_arr: np.ndarray | None = None,
    first_mat: Path | None = None,
    first_win: np.ndarray | None = None,
    max_qlin_windows: int | None = None,
) -> list[Path]:
    """
    Clean test windows [N,1,L], noisy [N*5,1,L], labels [N*5].
    """
    if windows_arr is None or labels_arr is None:
        windows_arr, labels_arr, first_mat, first_win = collect_qlin_windows(
            max_windows=max_qlin_windows,
        )
    if len(windows_arr) == 0:
        return []

    n = len(windows_arr)
    rng_split = np.random.default_rng(SEED)
    perm = rng_split.permutation(n)
    n_noise = max(1, n // 10)
    noise_idx = perm[:n_noise]
    test_idx = perm[n_noise:]

    if max_grounded_windows is not None:
        test_idx = test_idx[: int(max_grounded_windows)]

    clean_np = windows_arr[test_idx]
    labels_test = labels_arr[test_idx]
    n_win = len(clean_np)

    cfg = SignalConfig()
    noise_cfg = NoiseConfig()
    fs = cfg.sample_rate
    t = np.arange(cfg.signal_length) / fs
    rng = np.random.default_rng(SEED + 999)

    noisy_list: list[np.ndarray] = []
    snr_list: list[float] = []

    for i in range(n_win):
        c1 = clean_np[i]
        for snr_db in SNR_LEVELS:
            noise = generate_composite_noise(t, noise_cfg, rng)
            noisy, _actual = add_noise_at_snr(c1, noise, float(snr_db))
            noisy_list.append(noisy.astype(np.float32))
            snr_list.append(float(snr_db))

    clean_t = torch.from_numpy(clean_np).unsqueeze(1)
    noisy_t = torch.from_numpy(np.stack(noisy_list, axis=0)).unsqueeze(1)
    snr_t = torch.tensor(snr_list, dtype=torch.float32)

    paths: list[Path] = []
    p_clean = DATA_DIR / "test_real_qlin_clean.pt"
    p_noisy = DATA_DIR / "test_real_qlin_noisy.pt"
    p_snr = DATA_DIR / "test_real_qlin_snr_labels.pt"

    torch.save(
        {"clean": clean_t, "class_labels": torch.from_numpy(labels_test)},
        p_clean,
    )
    torch.save({"noisy": noisy_t}, p_noisy)
    torch.save({"snr_labels": snr_t}, p_snr)

    for p in (p_clean, p_noisy, p_snr):
        paths.append(p)
        logger.info("Saved: %s (%.2f MB)", p, p.stat().st_size / 1e6)

    if first_mat is not None and first_win is not None:
        cfg_sig = SignalConfig()
        save_noise_verification_figure(
            first_win, first_mat, noise_cfg, cfg_sig.sample_rate,
            np.arange(cfg_sig.signal_length) / cfg_sig.sample_rate,
        )

    return paths


def write_hashes(paths: list[Path]) -> None:
    DOCS_DIR.mkdir(exist_ok=True)
    lines = [
        "# Test Set SHA256 Hashes",
        "",
        "These hashes verify that test sets have not been modified after creation.",
        "Include in thesis appendix for reproducibility.",
        "",
    ]
    for p in paths:
        if p.exists():
            sha = _sha256_file(p)
            lines.append(f"**{p.name}**")
            lines.append("```")
            lines.append(sha)
            lines.append("```")
            lines.append("")
            logger.info("SHA256 %s:\n  %s", p, sha)

    hash_path = DOCS_DIR / "TEST_SET_HASHES.md"
    hash_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Hashes written to %s", hash_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-only",
        action="store_true",
        help="Rebuild real Q.Lin artifacts only; do not touch synthetic .pt",
    )
    parser.add_argument(
        "--max-grounded-windows",
        type=int,
        default=None,
        help="Cap number of clean windows for grounded set (default: all test split)",
    )
    parser.add_argument(
        "--max-qlin-windows",
        type=int,
        default=None,
        help=(
            "Stop after N Q.Lin windows when scanning .mat files (RAM cap). "
            "Default: load all windows (~780k full corpus)."
        ),
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)

    if not args.real_only:
        synth_path = DATA_DIR / "test_synthetic_ae.pt"
        if synth_path.exists():
            logger.info("Synthetic exists — skipping (delete file to rebuild)")
        else:
            build_synthetic_test_set()

    win, lab, first_mat, first_win = collect_qlin_windows(max_windows=args.max_qlin_windows)
    build_real_qlin_test_set(win, lab)
    build_real_grounded_sets(
        max_grounded_windows=args.max_grounded_windows,
        windows_arr=win,
        labels_arr=lab,
        first_mat=first_mat,
        first_win=first_win,
        max_qlin_windows=args.max_qlin_windows,
    )
    del win, lab, first_mat, first_win
    gc.collect()

    standard_names = [
        "test_synthetic_ae.pt",
        "test_real_qlin.pt",
        "real_noise_pool.pt",
        "test_real_qlin_clean.pt",
        "test_real_qlin_noisy.pt",
        "test_real_qlin_snr_labels.pt",
    ]
    to_hash = [DATA_DIR / name for name in standard_names if (DATA_DIR / name).exists()]
    write_hashes(to_hash)

    logger.info("Done.")


if __name__ == "__main__":
    main()
