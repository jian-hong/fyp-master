"""
Prompt 1: Audit mr_tae_fusion noise + pulse generators — PSD plots + assertions.

Source modules (not data/generators/ — project uses Try Model package):
  Try Model/mr_tae_fusion/data/noise_generators.py
  Try Model/mr_tae_fusion/data/pulse_generators.py

Outputs:
  docs/noise_validation.png
  docs/pulse_validation.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
MR_ROOT = PROJECT / "Try Model"
sys.path.insert(0, str(MR_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import signal as scipy_signal
except ImportError:
    scipy_signal = None

from mr_tae_fusion.config import NoiseConfig, SignalConfig  # noqa: E402
from mr_tae_fusion.data import noise_generators as ng  # noqa: E402
from mr_tae_fusion.data import pulse_generators as pg  # noqa: E402


def compute_psd(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x - np.mean(x)
    n = len(x)
    fft = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    psd = (np.abs(fft) ** 2) / max(n, 1)
    return freqs, psd


def band_power(freqs: np.ndarray, psd: np.ndarray, f_lo: float, f_hi: float) -> float:
    m = (freqs >= f_lo) & (freqs <= f_hi)
    return float(np.sum(psd[m]))


def peak_freq_near(freqs: np.ndarray, psd: np.ndarray, f_target: float, tol_hz: float) -> float:
    m = (freqs >= f_target - tol_hz) & (freqs <= f_target + tol_hz)
    if not np.any(m):
        return float("nan")
    idx = np.arange(len(freqs))[m]
    j = idx[int(np.argmax(psd[idx]))]
    return float(freqs[j])


def assert_magnetostrictive(freqs: np.ndarray, psd: np.ndarray, fs: float) -> list[str]:
    errs: list[str] = []
    tol = 8.0
    p100 = band_power(freqs, psd, 100 - tol, 100 + tol)
    p200 = band_power(freqs, psd, 200 - tol, 200 + tol)
    p_total = float(np.sum(psd)) + 1e-18
    if p100 / p_total < 0.005:
        errs.append(f"magnetostrictive: weak 100 Hz band (rel power {p100/p_total:.4f})")
    if p200 / p_total < 0.001:
        errs.append(f"magnetostrictive: weak 200 Hz band (rel power {p200/p_total:.4f})")
    return errs


def assert_powerline(
    freqs: np.ndarray,
    psd: np.ndarray,
    f0: float,
) -> list[str]:
    errs: list[str] = []
    tol = 2.5
    p_f0 = band_power(freqs, psd, f0 - tol, f0 + tol)
    p_3h = band_power(freqs, psd, 3 * f0 - tol, 3 * f0 + tol)
    p_sum = float(np.sum(psd)) + 1e-18
    if p_f0 / p_sum < 0.01:
        errs.append(f"powerline: weak fundamental {f0:g} Hz")
    if p_3h / p_sum < 0.002:
        errs.append(f"powerline: weak 3rd harmonic {3*f0:g} Hz")
    return errs


def spec_note_powerline_vs_phase0(f0: float) -> str:
    return (
        f"SPEC NOTE: Phase 0 table lists 50/60 Hz + 3rd,5th,7th harmonics; "
        f"code implements f0={f0:g} Hz + 3rd harmonic only (no 5th/7th terms)."
    )


def assert_nbi(freqs: np.ndarray, psd: np.ndarray, f_center: float) -> list[str]:
    errs: list[str] = []
    pk = peak_freq_near(freqs, psd, f_center, 2000.0)
    if not (30e3 <= pk <= 82e3):
        errs.append(
            f"NBI: dominant peak {pk/1e3:.1f} kHz not in 30-82 kHz (expected ~{f_center/1e3:.0f} kHz)"
        )
    return errs


def assert_awgn_flat(freqs: np.ndarray, psd: np.ndarray) -> list[str]:
    errs: list[str] = []
    lo, hi = 1e3, min(200e3, freqs[-1])
    m = (freqs >= lo) & (freqs <= hi)
    chunk = psd[m]
    if len(chunk) < 10:
        return errs
    med = float(np.median(chunk))
    peak_ratio = float(np.max(chunk) / (med + 1e-18))
    if peak_ratio > 500.0:
        errs.append(f"AWGN: PSD very spiky (max/median={peak_ratio:.1f}), not flat-like")
    return errs


def assert_impulse_sparse(x: np.ndarray) -> list[str]:
    errs: list[str] = []
    a = np.abs(x)
    thr = float(3 * np.std(a) + 1e-10)
    frac = float(np.mean(a > thr))
    if frac > 0.15:
        errs.append(f"impulse-like noise: {frac*100:.1f}% samples > 3σ (expect sparse spikes)")
    return errs


def run_noise_panel(
    fs: float, n: int, rng: np.random.Generator
) -> tuple[plt.Figure, list[str], list[str]]:
    t = np.arange(n) / fs
    cfg = NoiseConfig()
    all_errs: list[str] = []
    notes: list[str] = []

    # 1 AWGN (Welch for smoother flatness test)
    wgn = ng.generate_wgn(n, cfg.wgn_amplitude, rng)
    if scipy_signal is not None:
        fq, psd = scipy_signal.welch(wgn, fs, nperseg=min(n, 256), noverlap=None)
    else:
        fq, psd = compute_psd(wgn, fs)
    all_errs += [f"AWGN: {e}" for e in assert_awgn_flat(fq, psd)]

    # 2–3,5: For line spectra, generators must be evaluated on a continuous long axis.
    # Tiling 1024-sample fragments injects jumps that smear 50–200 Hz peaks.
    tile_len = int(min(fs * 2, 4e6))
    t_long = np.arange(tile_len) / fs

    # 2 Magnetostrictive (same fn as composite — 100 Hz + 200 Hz)
    mag = ng.generate_magnetostrictive_hum(t, amplitude=0.01)
    mag_long = ng.generate_magnetostrictive_hum(t_long, amplitude=0.01)
    fq2, psd2 = compute_psd(mag_long, fs)
    all_errs += assert_magnetostrictive(fq2, psd2, fs)

    # 3 Powerline
    pwr = ng.generate_powerline_noise(t, cfg.powerline_freq, cfg.powerline_amplitude, cfg.harmonic_amplitude)
    pwr_long = ng.generate_powerline_noise(
        t_long, cfg.powerline_freq, cfg.powerline_amplitude, cfg.harmonic_amplitude
    )
    fq3, psd3 = compute_psd(pwr_long, fs)
    all_errs += assert_powerline(fq3, psd3, cfg.powerline_freq)
    notes.append(spec_note_powerline_vs_phase0(cfg.powerline_freq))

    # 4 Impulsive (Bernoulli-Gaussian)
    imp = ng.generate_impulsive_noise(n, rng=rng)
    fq4, psd4 = compute_psd(imp, fs)
    all_errs += assert_impulse_sparse(imp)

    # 5 NBI
    nbi = ng.generate_narrowband_noise(t, cfg.narrowband_freq, cfg.narrowband_amplitude, rng=rng)
    nbi_long = ng.generate_narrowband_noise(
        t_long, cfg.narrowband_freq, cfg.narrowband_amplitude, rng=rng
    )
    fq5, psd5 = compute_psd(nbi_long, fs)
    all_errs += assert_nbi(fq5, psd5, cfg.narrowband_freq)

    fig, axes = plt.subplots(5, 2, figsize=(11, 12))
    titles = ["AWGN", "Magnetostrictive 100/200 Hz", "Powerline + 3rd", "Bernoulli–Gaussian impulse", "NBI"]
    series = [
        (wgn, fq, psd),
        (mag, fq2, psd2),
        (pwr, fq3, psd3),
        (imp, fq4, psd4),
        (nbi, fq5, psd5),
    ]
    for i, (title, (sig, fqx, psdx)) in enumerate(zip(titles, series)):
        axes[i, 0].plot(t * 1e3, sig, lw=0.6)
        axes[i, 0].set_ylabel("amp")
        axes[i, 0].set_title(f"{title} — time (ms)")
        if i == 4:
            axes[i, 0].set_xlabel("time (ms)")
        axes[i, 1].semilogy(fqx, psdx + 1e-20, lw=0.6)
        axes[i, 1].set_xlim(0, min(250e3, fqx[-1]))
        axes[i, 1].set_title(f"{title} — PSD")
        if i == 4:
            axes[i, 1].set_xlabel("Hz")

    fig.suptitle(f"Noise validation (fs={fs/1e6:.1f} MHz, N={n} samples)", fontsize=12)
    fig.tight_layout()
    return fig, all_errs, notes


def run_pulse_panel(fs: float, n_win: int, snr_db: float, rng: np.random.Generator) -> tuple[plt.Figure, list[str]]:
    sig_cfg = SignalConfig()
    sig_cfg.signal_length = 2048
    fs = sig_cfg.sample_rate

    psds: list[np.ndarray] = []
    freqs_ref: np.ndarray | None = None
    types = list("ABCDEF")

    for i in range(n_win):
        st = types[i % len(types)]
        t, clean, _mask = pg.generate_pd_signal(st, sig_cfg)
        noise = ng.generate_composite_noise(t, NoiseConfig(), rng)
        noisy, _ = ng.add_noise_at_snr(clean, noise, snr_db)
        fq, psd = compute_psd(noisy, fs)
        if freqs_ref is None:
            freqs_ref = fq
        psds.append(psd)

    mean_psd = np.mean(np.stack(psds, axis=0), axis=0)
    assert freqs_ref is not None

    errs: list[str] = []
    peak_idx = int(np.argmax(mean_psd))
    peak_f = float(freqs_ref[peak_idx])
    if not (20e3 <= peak_f <= 500e3):
        errs.append(f"mean pulse PSD: peak {peak_f/1e3:.1f} kHz outside AE band 20–500 kHz")

    t_show = np.arange(sig_cfg.signal_length) / fs
    _, clean_show, _ = pg.generate_pd_signal("D", sig_cfg)
    noise_show = ng.generate_composite_noise(t_show, NoiseConfig(), rng)
    noisy_show, _ = ng.add_noise_at_snr(clean_show, noise_show, snr_db)
    fq_s, psd_s = compute_psd(noisy_show, fs)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes[0, 0].plot(t_show * 1e3, clean_show, lw=0.7)
    axes[0, 0].set_title("Example clean (type D)")
    axes[0, 1].plot(t_show * 1e3, noisy_show, lw=0.7, color="C1")
    axes[0, 1].set_title(f"Noisy @ {snr_db:+.0f} dB SNR (composite noise)")
    axes[1, 0].semilogy(freqs_ref, mean_psd + 1e-20, lw=0.8)
    axes[1, 0].axvspan(20e3, 500e3, alpha=0.12, color="green")
    axes[1, 0].set_xlim(0, min(600e3, freqs_ref[-1]))
    axes[1, 0].set_title(f"Mean PSD over {n_win} pulses (all types A–F cycled)")
    axes[1, 1].semilogy(fq_s, psd_s + 1e-20, lw=0.8)
    axes[1, 1].axvspan(20e3, 500e3, alpha=0.12, color="green")
    axes[1, 1].set_title("Single-window noisy PSD (example)")
    for ax in axes.ravel():
        ax.set_xlabel("time (ms)" if ax in (axes[0, 0], axes[0, 1]) else "Hz")
    fig.suptitle("Pulse + band check (20–500 kHz)", fontsize=12)
    fig.tight_layout()
    return fig, errs


def main() -> int:
    rng = np.random.default_rng(42)
    fs = 1e6
    n = 1024

    DOCS = PROJECT / "docs"
    DOCS.mkdir(parents=True, exist_ok=True)

    fig_n, errs_n, notes_n = run_noise_panel(fs, n, rng)
    out_n = DOCS / "noise_validation.png"
    fig_n.savefig(out_n, dpi=300, bbox_inches="tight")
    plt.close(fig_n)
    print(f"Wrote {out_n}")

    fig_p, errs_p = run_pulse_panel(fs, 100, 10.0, rng)
    out_p = DOCS / "pulse_validation.png"
    fig_p.savefig(out_p, dpi=300, bbox_inches="tight")
    plt.close(fig_p)
    print(f"Wrote {out_p}")

    print("\n--- Assertion summary ---")
    if notes_n:
        print("[SPEC / implementation notes]")
        for e in notes_n:
            print(" ", e)
    if errs_n:
        print("[NOISE FAILURES]")
        for e in errs_n:
            print(" ", e)
    else:
        print("[NOISE] All automated checks passed.")

    if errs_p:
        print("[PULSE]")
        for e in errs_p:
            print(" ", e)
    else:
        print("[PULSE] All checks passed.")

    code = 1 if (errs_n or errs_p) else 0
    return code


if __name__ == "__main__":
    raise SystemExit(main())
