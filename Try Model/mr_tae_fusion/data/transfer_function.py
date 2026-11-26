"""
Acoustic transfer-function utilities for AE-PD pulse synthesis.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


def generate_tank_transfer_function(
    fs: float,
    num_reflections: Optional[int] = None,
    reflection_delays_us: Tuple[float, float] = (80.0, 400.0),
    reflection_attenuation: Tuple[float, float] = (0.15, 0.45),
    rng: Optional[np.random.Generator] = None,
    max_len: int = 512,
) -> np.ndarray:
    """
    Generate an impulse response for AE multipath reflections in a tank.

    Length is capped at *max_len* samples so that convolution with the
    signal window preserves its original length.
    """
    if rng is None:
        rng = np.random.default_rng()

    if num_reflections is None:
        num_reflections = int(rng.integers(1, 4))  # 1-3 reflections

    h_len = min(max_len, max(8, int(round(0.0005 * fs))))  # 0.5 ms cap
    h = np.zeros(h_len, dtype=np.float64)
    h[0] = 1.0  # direct path

    d_min_us, d_max_us = reflection_delays_us
    a_min, a_max = reflection_attenuation
    delays = np.sort(rng.uniform(d_min_us, d_max_us, size=num_reflections))

    for idx, delay_us in enumerate(delays):
        delay_idx = int(round(delay_us * 1e-6 * fs))
        if delay_idx >= h_len:
            continue
        # Slight attenuation decay for later reflections
        amp = rng.uniform(a_min, a_max) * (0.85 ** idx)
        h[delay_idx] += amp

    # Normalize total energy conservatively to avoid runaway amplitude
    norm = np.linalg.norm(h)
    if norm > 0:
        h = h / norm
    h[0] = max(h[0], 0.65)
    return h


def apply_acoustic_transfer_function(
    signal: np.ndarray,
    fs: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Convolve pulse with tank impulse response using same-length output."""
    h = generate_tank_transfer_function(fs=fs, rng=rng)
    return np.convolve(signal, h, mode="same")

