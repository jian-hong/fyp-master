"""
AE-accurate pulse generators for synthetic PD signal creation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt

from ..config import SignalConfig
from .transfer_function import apply_acoustic_transfer_function


def generate_dep(
    t: np.ndarray, amplitude: float, tau1: float, tau2: float, t_start: float = 0.0
) -> np.ndarray:
    """Backward-compatible DEP utility."""
    pulse = np.zeros_like(t)
    t_shifted = t - t_start
    valid = t_shifted >= 0
    t_valid = t_shifted[valid]
    pulse[valid] = amplitude * (np.exp(-t_valid / max(tau1, 1e-9)) - np.exp(-t_valid / max(tau2, 1e-9)))
    return pulse


def generate_dop(
    t: np.ndarray, amplitude: float, tau: float, fc: float, phi: float = 0.0, t_start: float = 0.0
) -> np.ndarray:
    """Backward-compatible DOP utility."""
    pulse = np.zeros_like(t)
    t_shifted = t - t_start
    valid = t_shifted >= 0
    pulse[valid] = _dop_pulse_segment(t_shifted[valid], amplitude, tau, fc, phi)
    return pulse


def _dop_pulse_segment(
    t_segment: np.ndarray, amplitude: float, tau_s: float, fc_hz: float, phi: float
) -> np.ndarray:
    envelope = np.exp(-t_segment / max(tau_s, 1e-9))
    return amplitude * envelope * np.sin(2 * np.pi * fc_hz * t_segment + phi)


def _add_event(
    signal: np.ndarray,
    mask: np.ndarray,
    t: np.ndarray,
    fs: float,
    start_time_s: float,
    duration_s: float,
    class_id: int,
    amplitude: float,
    tau_s: float,
    fc_hz: float,
    phi: float,
):
    start_idx = int(round(start_time_s * fs))
    n = int(round(duration_s * fs))
    if start_idx < 0 or start_idx >= len(signal) or n <= 3:
        return
    end_idx = min(len(signal), start_idx + n)
    local_t = t[start_idx:end_idx] - t[start_idx]
    signal[start_idx:end_idx] += _dop_pulse_segment(local_t, amplitude, tau_s, fc_hz, phi)
    mask[start_idx:end_idx] = class_id


def generate_type_a(t: np.ndarray, fs: float, config: Optional[SignalConfig] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Type A: corona-like repetitive low-amplitude bursts."""
    if config is None:
        config = SignalConfig()
    signal = np.zeros_like(t)
    mask = np.zeros_like(t, dtype=np.int64)
    cls = config.type_to_class["A"]
    n_events = int(np.random.randint(6, 16))
    for _ in range(n_events):
        _add_event(
            signal, mask, t, fs,
            start_time_s=float(np.random.uniform(t[0], t[-1] * 0.9)),
            duration_s=float(np.random.uniform(120e-6, 400e-6)),
            class_id=cls,
            amplitude=float(np.random.uniform(0.05, 0.6) * np.random.choice([-1.0, 1.0])),
            tau_s=float(np.random.uniform(120e-6, 500e-6)),
            fc_hz=float(np.random.uniform(20e3, 80e3)),
            phi=float(np.random.uniform(0, 2 * np.pi)),
        )
    return signal, mask


def generate_type_b(t: np.ndarray, fs: float, config: Optional[SignalConfig] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Type B: dense streamer-like overlapping short pulses (1.0 mm equivalent)."""
    if config is None:
        config = SignalConfig()
    signal = np.zeros_like(t)
    mask = np.zeros_like(t, dtype=np.int64)
    cls = config.type_to_class["B"]
    t_total = max(t[-1] - t[0], 1 / fs)
    event_rate_hz = float(np.random.uniform(500, 2000))
    n_events = int(max(25, event_rate_hz * t_total))
    for _ in range(n_events):
        _add_event(
            signal, mask, t, fs,
            start_time_s=float(np.random.uniform(t[0], t[-1] * 0.98)),
            duration_s=float(np.random.uniform(50e-6, 180e-6)),
            class_id=cls,
            amplitude=float(np.random.uniform(0.01, 0.08) * np.random.choice([-1.0, 1.0])),
            tau_s=float(np.random.uniform(50e-6, 200e-6)),
            fc_hz=float(np.random.uniform(80e3, 300e3)),
            phi=float(np.random.uniform(0, 2 * np.pi)),
        )
    return signal, mask


def generate_type_c(t: np.ndarray, fs: float, config: Optional[SignalConfig] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Type C: sparse surface-discharge-like impulses, higher amplitude."""
    if config is None:
        config = SignalConfig()
    signal = np.zeros_like(t)
    mask = np.zeros_like(t, dtype=np.int64)
    cls = config.type_to_class["C"]
    for _ in range(int(np.random.randint(4, 10))):
        _add_event(
            signal, mask, t, fs,
            start_time_s=float(np.random.uniform(t[0], t[-1] * 0.95)),
            duration_s=float(np.random.uniform(250e-6, 900e-6)),
            class_id=cls,
            amplitude=float(np.random.uniform(0.3, 5.0) * np.random.choice([-1.0, 1.0])),
            tau_s=float(np.random.uniform(200e-6, 1200e-6)),
            fc_hz=float(np.random.uniform(50e3, 200e3)),
            phi=float(np.random.uniform(0, 2 * np.pi)),
        )
    return signal, mask


def generate_type_d(t: np.ndarray, fs: float, config: Optional[SignalConfig] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Type D: sparse high-amplitude internal-like pulses."""
    if config is None:
        config = SignalConfig()
    signal = np.zeros_like(t)
    mask = np.zeros_like(t, dtype=np.int64)
    cls = config.type_to_class["D"]
    for _ in range(int(np.random.randint(5, 14))):
        _add_event(
            signal, mask, t, fs,
            start_time_s=float(np.random.uniform(t[0], t[-1] * 0.95)),
            duration_s=float(np.random.uniform(150e-6, 600e-6)),
            class_id=cls,
            amplitude=float(np.random.uniform(1.0, 8.0) * np.random.choice([-1.0, 1.0])),
            tau_s=float(np.random.uniform(50e-6, 300e-6)),
            fc_hz=float(np.random.uniform(100e3, 400e3)),
            phi=float(np.random.uniform(0, 2 * np.pi)),
        )
    return signal, mask


def generate_type_e(t: np.ndarray, fs: float, config: Optional[SignalConfig] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Type E: moderate-density pulses in AE-valid band."""
    if config is None:
        config = SignalConfig()
    signal = np.zeros_like(t)
    mask = np.zeros_like(t, dtype=np.int64)
    cls = config.type_to_class["E"]
    for _ in range(int(np.random.randint(10, 24))):
        _add_event(
            signal, mask, t, fs,
            start_time_s=float(np.random.uniform(t[0], t[-1] * 0.98)),
            duration_s=float(np.random.uniform(120e-6, 500e-6)),
            class_id=cls,
            amplitude=float(np.random.uniform(0.6, 4.5) * np.random.choice([-1.0, 1.0])),
            tau_s=float(np.random.uniform(100e-6, 700e-6)),
            fc_hz=float(np.random.uniform(60e3, 220e3)),
            phi=float(np.random.uniform(0, 2 * np.pi)),
        )
    return signal, mask


def generate_type_f(t: np.ndarray, fs: float, config: Optional[SignalConfig] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Type F: denser internal-like pulses with broad AE-band carriers."""
    if config is None:
        config = SignalConfig()
    signal = np.zeros_like(t)
    mask = np.zeros_like(t, dtype=np.int64)
    cls = config.type_to_class["F"]
    for _ in range(int(np.random.randint(12, 28))):
        _add_event(
            signal, mask, t, fs,
            start_time_s=float(np.random.uniform(t[0], t[-1] * 0.98)),
            duration_s=float(np.random.uniform(100e-6, 450e-6)),
            class_id=cls,
            amplitude=float(np.random.uniform(0.8, 5.2) * np.random.choice([-1.0, 1.0])),
            tau_s=float(np.random.uniform(80e-6, 500e-6)),
            fc_hz=float(np.random.uniform(80e3, 260e3)),
            phi=float(np.random.uniform(0, 2 * np.pi)),
        )
    return signal, mask


def generate_type_g(t: np.ndarray, fs: float, config: Optional[SignalConfig] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Type G (synthetic-only treeing): corrected to 80-200 kHz."""
    if config is None:
        config = SignalConfig()
    signal = np.zeros_like(t)
    mask = np.zeros_like(t, dtype=np.int64)
    cls = config.type_to_class["G"]
    for _ in range(int(np.random.randint(10, 24))):
        _add_event(
            signal, mask, t, fs,
            start_time_s=float(np.random.uniform(t[0], t[-1] * 0.97)),
            duration_s=float(np.random.uniform(60e-6, 250e-6)),
            class_id=cls,
            amplitude=float(np.random.uniform(0.2, 2.2) * np.random.choice([-1.0, 1.0])),
            tau_s=float(np.random.uniform(50e-6, 200e-6)),
            fc_hz=float(np.random.uniform(80e3, 200e3)),
            phi=float(np.random.uniform(0, 2 * np.pi)),
        )
    return signal, mask


def generate_mixed_signal(t: np.ndarray, fs: float, config: Optional[SignalConfig] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate mixed signal with multiple PD types in same trace.
    
    This represents realistic field conditions where different
    defect types coexist (e.g., corona + internal discharge).
    
    Returns:
        Tuple of (clean_signal, segmentation_mask with multiple classes)
    """
    if config is None:
        config = SignalConfig()
    
    signal = np.zeros_like(t)
    mask = np.zeros_like(t, dtype=np.int64)
    
    # Randomly select 2-4 types to combine
    all_types = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    num_types = 2 + np.random.randint(3)  # 2-4 types
    selected_types = np.random.choice(all_types, size=num_types, replace=False)
    
    generators_local = {
        'A': lambda: generate_type_a(t, fs, config),
        'B': lambda: generate_type_b(t, fs, config),
        'C': lambda: generate_type_c(t, fs, config),
        'D': lambda: generate_type_d(t, fs, config),
        'E': lambda: generate_type_e(t, fs, config),
        'F': lambda: generate_type_f(t, fs, config),
        'G': lambda: generate_type_g(t, fs, config),
    }
    
    # Generate each type with reduced amplitude and combine
    for sig_type in selected_types:
        type_signal, type_mask = generators_local[sig_type]()
        
        # Scale down each component
        scale = 0.5 + np.random.rand() * 0.5
        signal += type_signal * scale
        
        # Merge masks (later class overwrites earlier - represents overlapping)
        mask = np.where(type_mask > 0, type_mask, mask)
    
    return signal, mask


def generate_pd_signal(signal_type: str, config: Optional[SignalConfig] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic PD signal of specified type.
    
    Args:
        signal_type: One of 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'MIXED'
        config: Signal configuration
    
    Returns:
        Tuple of (time_vector, clean_signal, segmentation_mask)
    """
    if config is None:
        config = SignalConfig()
    
    fs = config.sample_rate
    n_samples = int(config.signal_length)
    t = np.arange(n_samples) / fs
    
    generators = {
        'A': lambda: generate_type_a(t, fs, config),
        'B': lambda: generate_type_b(t, fs, config),
        'C': lambda: generate_type_c(t, fs, config),
        'D': lambda: generate_type_d(t, fs, config),
        'E': lambda: generate_type_e(t, fs, config),
        'F': lambda: generate_type_f(t, fs, config),
        'G': lambda: generate_type_g(t, fs, config),
        'MIXED': lambda: generate_mixed_signal(t, fs, config),
    }
    
    if signal_type not in generators:
        raise ValueError(f"Unknown signal type: {signal_type}. Must be one of {list(generators.keys())}")
    
    signal, mask = generators[signal_type]()
    signal = apply_acoustic_transfer_function(signal, fs)
    max_amp = np.max(np.abs(signal)) + 1e-12
    signal = signal / max_amp
    return t, signal, mask


def verify_frequency_content(signal: np.ndarray, fs: float, show: bool = False):
    """
    Plot PSD and assert dominant peak lies in the AE band [20kHz, 500kHz].
    """
    x = signal - np.mean(signal)
    fft = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    psd = (np.abs(fft) ** 2) / max(len(x), 1)
    peak_freq = freqs[int(np.argmax(psd))]
    assert 20e3 <= peak_freq <= 500e3, f"Peak frequency {peak_freq:.1f} Hz outside AE band."

    if show:
        plt.figure(figsize=(8, 4))
        plt.semilogy(freqs, psd + 1e-18)
        plt.axvspan(20e3, 500e3, alpha=0.15, color="green", label="AE band")
        plt.title(f"PSD (peak={peak_freq/1e3:.1f} kHz)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power")
        plt.legend()
        plt.tight_layout()
        plt.show()


def generate_batch(
    batch_size: int, signal_types: Optional[List[str]] = None, config: Optional[SignalConfig] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a batch of synthetic PD signals.
    
    Args:
        batch_size: Number of signals to generate
        signal_types: List of types to sample from. If None, uses all types.
        config: Signal configuration
    
    Returns:
        Tuple of (clean_signals, masks, type_labels) as torch tensors
    """
    if config is None:
        config = SignalConfig()
    
    if signal_types is None:
        signal_types = ['A', 'B', 'C', 'D', 'E', 'F']
    
    signals: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    
    for i in range(batch_size):
        signal_type = signal_types[i % len(signal_types)]
        t, signal, mask = generate_pd_signal(signal_type, config)
        signals.append(signal)
        masks.append(mask)
    
    signals_tensor = torch.tensor(np.array(signals), dtype=torch.float32)
    masks_tensor = torch.tensor(np.array(masks), dtype=torch.long)
    return signals_tensor, masks_tensor
