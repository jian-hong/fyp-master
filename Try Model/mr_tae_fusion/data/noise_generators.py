"""
Noise generators for PD signal corruption.

Implements:
- White Gaussian Noise (WGN)
- Bernoulli-Gaussian impulsive noise
- Powerline interference
- Narrowband interference
- Composite noise with SNR control
"""

import numpy as np
import torch
from typing import Tuple, Optional, Union
from ..config import NoiseConfig


def generate_wgn(
    signal_length: int,
    amplitude: float = 0.08,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Generate White Gaussian Noise.
    
    Args:
        signal_length: Length of noise signal
        amplitude: Standard deviation of noise
        rng: Random number generator for reproducibility
    
    Returns:
        WGN signal
    """
    if rng is None:
        rng = np.random.default_rng()
    
    return amplitude * rng.standard_normal(signal_length)


def generate_impulsive_noise(
    signal_length: int,
    probability: float = 0.01,
    amplitude_range: Tuple[float, float] = (0.2, 1.0),
    scale: float = 0.4,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Generate Bernoulli-Gaussian impulsive noise.
    
    n_imp(t) = b(t) · g(t)
    where b(t) is Bernoulli process and g(t) is high-variance Gaussian.
    
    Args:
        signal_length: Length of noise signal
        probability: Bernoulli probability p (sparsity parameter)
        amplitude_range: (min, max) amplitude factors
        scale: Overall scaling factor
        rng: Random number generator
    
    Returns:
        Impulsive noise signal
    """
    if rng is None:
        rng = np.random.default_rng()
    
    noise = np.zeros(signal_length)
    
    # Generate Bernoulli mask
    bernoulli_mask = rng.random(signal_length) < probability
    num_impulses = np.sum(bernoulli_mask)
    
    if num_impulses > 0:
        # Generate high-variance Gaussian values at impulse locations
        amplitudes = amplitude_range[0] + (amplitude_range[1] - amplitude_range[0]) * rng.random(num_impulses)
        noise[bernoulli_mask] = scale * amplitudes * rng.choice([-1, 1], size=num_impulses)
    
    return noise


def generate_powerline_noise(
    t: np.ndarray,
    fundamental_freq: float = 50.0,
    fundamental_amplitude: float = 0.025,
    harmonic_amplitude: float = 0.015
) -> np.ndarray:
    """
    Generate powerline interference with harmonics.
    
    Args:
        t: Time vector (seconds)
        fundamental_freq: Fundamental frequency (Hz)
        fundamental_amplitude: Amplitude of fundamental
        harmonic_amplitude: Amplitude of 3rd harmonic
    
    Returns:
        Powerline interference signal
    """
    fundamental = fundamental_amplitude * np.sin(2 * np.pi * fundamental_freq * t)
    harmonic = harmonic_amplitude * np.sin(2 * np.pi * 3 * fundamental_freq * t)
    
    return fundamental + harmonic


def generate_narrowband_noise(
    t: np.ndarray,
    center_freq: float = 80e3,
    amplitude: float = 0.03,
    random_phase: bool = True,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Generate narrowband interference.
    
    Args:
        t: Time vector (seconds)
        center_freq: Center frequency (Hz)
        amplitude: Signal amplitude
        random_phase: Whether to add random phase offset
        rng: Random number generator
    
    Returns:
        Narrowband noise signal
    """
    if rng is None:
        rng = np.random.default_rng()
    
    phase = rng.random() * 2 * np.pi if random_phase else 0
    
    return amplitude * np.sin(2 * np.pi * center_freq * t + phase)


def generate_magnetostrictive_hum(
    t: np.ndarray,
    amplitude: float = 0.01
) -> np.ndarray:
    """
    Generate magnetostrictive hum (100 Hz + 200 Hz).
    """
    return (
        amplitude * np.sin(2 * np.pi * 100.0 * t) +
        (0.3 * amplitude) * np.sin(2 * np.pi * 200.0 * t)
    )


def generate_composite_noise(
    t: np.ndarray,
    config: Optional[NoiseConfig] = None,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Generate composite noise matching MATLAB simulation.
    
    Combines: WGN + powerline + narrowband + impulsive
    
    Args:
        t: Time vector (seconds)
        config: Noise configuration
        rng: Random number generator
    
    Returns:
        Composite noise signal
    """
    if config is None:
        config = NoiseConfig()
    
    if rng is None:
        rng = np.random.default_rng()
    
    signal_length = len(t)
    
    # White Gaussian Noise
    wgn = generate_wgn(signal_length, config.wgn_amplitude, rng)
    
    # Powerline interference (50 Hz + harmonics)
    powerline = generate_powerline_noise(
        t,
        config.powerline_freq,
        config.powerline_amplitude,
        config.harmonic_amplitude
    )
    
    # Narrowband interference (AE sub-MHz band)
    narrowband = generate_narrowband_noise(
        t,
        config.narrowband_freq,
        config.narrowband_amplitude,
        random_phase=True,
        rng=rng
    )
    
    # Magnetostrictive hum
    magnetostrictive = generate_magnetostrictive_hum(t, amplitude=0.01)

    # Impulsive noise (Bernoulli-Gaussian)
    impulse = np.zeros(signal_length)
    num_impulses = 15
    spike_positions = rng.choice(signal_length, size=num_impulses, replace=False)
    impulse[spike_positions] = 0.4 * (0.2 + 0.8 * rng.random(num_impulses))
    
    return wgn + powerline + narrowband + magnetostrictive + impulse


def add_noise_at_snr(
    clean_signal: np.ndarray,
    noise: np.ndarray,
    target_snr_db: float
) -> Tuple[np.ndarray, float]:
    """
    Add noise to signal at specified SNR level.
    
    Args:
        clean_signal: Clean signal
        noise: Noise signal (will be scaled)
        target_snr_db: Desired SNR in dB
    
    Returns:
        Tuple of (noisy_signal, actual_snr_db)
    """
    # Calculate current SNR
    signal_power = np.var(clean_signal) + 1e-10
    noise_power = np.var(noise) + 1e-10
    
    current_snr_db = 10 * np.log10(signal_power / noise_power)
    
    # Scale noise to achieve target SNR
    scale_factor = 10 ** ((current_snr_db - target_snr_db) / 20)
    scaled_noise = noise * scale_factor
    
    # Create noisy signal
    noisy_signal = clean_signal + scaled_noise
    
    # Verify actual SNR
    actual_noise_power = np.var(scaled_noise) + 1e-10
    actual_snr_db = 10 * np.log10(signal_power / actual_noise_power)
    
    return noisy_signal, actual_snr_db


def add_noise_at_snr_torch(
    clean_signal: torch.Tensor,
    noise: torch.Tensor,
    target_snr_db: float
) -> torch.Tensor:
    """
    Add noise to signal at specified SNR level (PyTorch version).
    
    Args:
        clean_signal: Clean signal tensor
        noise: Noise tensor (will be scaled)
        target_snr_db: Desired SNR in dB
    
    Returns:
        Noisy signal tensor
    """
    signal_power = torch.var(clean_signal) + 1e-10
    noise_power = torch.var(noise) + 1e-10
    
    current_snr_db = 10 * torch.log10(signal_power / noise_power)
    scale_factor = 10 ** ((current_snr_db - target_snr_db) / 20)
    
    return clean_signal + noise * scale_factor


class CurriculumNoiseScheduler:
    """
    Curriculum learning noise scheduler.
    
    Progressively increases noise difficulty across training phases.
    """
    
    def __init__(self, config: Optional[NoiseConfig] = None):
        if config is None:
            config = NoiseConfig()
        
        self.config = config
        self.current_phase = 1
        
    def get_snr_range(self, epoch: int, total_epochs: int) -> Tuple[float, float]:
        """
        Get SNR range for current epoch based on curriculum.
        
        Args:
            epoch: Current epoch (0-indexed)
            total_epochs: Total number of epochs
        
        Returns:
            (min_snr, max_snr) tuple
        """
        # Phase boundaries
        phase1_end = int(0.2 * total_epochs)  # 0-20%
        phase2_end = int(0.5 * total_epochs)  # 20-50%
        
        if epoch < phase1_end:
            self.current_phase = 1
            return self.config.snr_range_phase1
        elif epoch < phase2_end:
            self.current_phase = 2
            return self.config.snr_range_phase2
        else:
            self.current_phase = 3
            return self.config.snr_range_phase3
    
    def sample_snr(
        self,
        epoch: int,
        total_epochs: int,
        rng: Optional[np.random.Generator] = None
    ) -> float:
        """
        Sample a random SNR value for current training phase.
        
        Args:
            epoch: Current epoch
            total_epochs: Total epochs
            rng: Random number generator
        
        Returns:
            SNR value in dB
        """
        if rng is None:
            rng = np.random.default_rng()
        
        snr_min, snr_max = self.get_snr_range(epoch, total_epochs)
        return rng.uniform(snr_min, snr_max)
    
    def get_noise_type(self, epoch: int, total_epochs: int) -> str:
        """
        Get noise type for current training phase.
        
        Phase 1: WGN only
        Phase 2: WGN + Sparse Impulsive
        Phase 3: Full composite (including TGAN if available)
        
        Returns:
            Noise type string
        """
        phase1_end = int(0.2 * total_epochs)
        phase2_end = int(0.5 * total_epochs)
        
        if epoch < phase1_end:
            return 'wgn'
        elif epoch < phase2_end:
            return 'wgn_impulsive'
        else:
            return 'composite'
