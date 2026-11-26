"""
Distance-based PD signal generator for 4-class classification.

Simulates PD signals at different gap distances:
- Class 0: 1.0mm (10mm scale) - Very sparse PD events
- Class 1: 1.8mm (18mm scale) - Sparse PD events  
- Class 2: 2.0mm (20mm scale) - Moderate frequency PD events
- Class 3: 2.5mm (25mm scale) - Dense PD events
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class DistanceSignalConfig:
    """Configuration for distance-based signal generation."""
    sample_rate: float = 1000e6  # 1 GHz
    duration: float = 2e-6  # 2 μs
    signal_length: int = 2001
    
    # Distance class mappings
    distance_classes: Dict[int, str] = None
    
    def __post_init__(self):
        if self.distance_classes is None:
            self.distance_classes = {
                0: '1.0mm',
                1: '1.8mm', 
                2: '2.0mm',
                3: '2.5mm'
            }


def generate_distance_pd_signal(
    distance_class: int,
    config: Optional[DistanceSignalConfig] = None,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Generate a PD signal for a specific distance class.
    
    Args:
        distance_class: 0-3 corresponding to 1.0mm, 1.8mm, 2.0mm, 2.5mm
        config: Signal configuration
        rng: Random number generator
    
    Returns:
        Tuple of (time_vector, clean_signal, distance_class)
    """
    if config is None:
        config = DistanceSignalConfig()
    
    if rng is None:
        rng = np.random.default_rng()
    
    # Create time vector
    fs = config.sample_rate
    t_total = config.duration
    t = np.linspace(0, t_total, config.signal_length)
    
    # Initialize clean signal
    clean_signal = np.zeros(config.signal_length)
    
    # Distance-specific parameters
    if distance_class == 0:  # 1.0mm - Very sparse, low amplitude
        num_events = rng.integers(3, 8)
        amplitude_range = (1.0, 3.0)
        pulse_width_range = (2e-9, 8e-9)
        
    elif distance_class == 1:  # 1.8mm - Sparse, moderate amplitude
        num_events = rng.integers(10, 25)
        amplitude_range = (2.0, 5.0)
        pulse_width_range = (3e-9, 12e-9)
        
    elif distance_class == 2:  # 2.0mm - Moderate frequency
        num_events = rng.integers(25, 50)
        amplitude_range = (3.0, 7.0)
        pulse_width_range = (5e-9, 15e-9)
        
    else:  # 2.5mm - Dense, high amplitude
        num_events = rng.integers(50, 100)
        amplitude_range = (4.0, 10.0)
        pulse_width_range = (8e-9, 25e-9)
    
    # Generate PD events
    event_times = np.sort(rng.uniform(0, t_total, num_events))
    
    for event_time in event_times:
        # Find nearest sample index
        start_idx = int(event_time * fs)
        if start_idx >= config.signal_length:
            continue
        
        # Randomize pulse parameters
        amplitude = rng.uniform(*amplitude_range)
        polarity = rng.choice([-1, 1])
        pulse_width = rng.uniform(*pulse_width_range)
        pulse_samples = max(2, int(pulse_width * fs))
        
        # Choose pulse shape
        pulse_type = rng.choice(['spike', 'damped', 'oscillating'])
        
        if pulse_type == 'spike':
            # Simple spike
            if start_idx + 1 < config.signal_length:
                clean_signal[start_idx] = polarity * amplitude
                clean_signal[start_idx + 1] = -polarity * amplitude * 0.3
                
        elif pulse_type == 'damped':
            # Damped exponential pulse
            end_idx = min(start_idx + pulse_samples, config.signal_length)
            tau = pulse_width * 0.3
            for j in range(end_idx - start_idx):
                t_pulse = j / fs
                clean_signal[start_idx + j] += polarity * amplitude * np.exp(-t_pulse / tau)
                
        else:  # oscillating
            # Damped oscillation
            fc = rng.uniform(50e6, 200e6)  # Carrier frequency
            end_idx = min(start_idx + pulse_samples, config.signal_length)
            tau = pulse_width * 0.5
            for j in range(end_idx - start_idx):
                t_pulse = j / fs
                clean_signal[start_idx + j] += (
                    polarity * amplitude * 
                    np.exp(-t_pulse / tau) * 
                    np.sin(2 * np.pi * fc * t_pulse)
                )
    
    # Normalize signal amplitude
    max_amp = np.max(np.abs(clean_signal))
    if max_amp > 0:
        clean_signal = clean_signal / max_amp * amplitude_range[1]
    
    return t, clean_signal, distance_class


def apply_acoustic_gap_reflection(
    signal: np.ndarray,
    fs: float,
    gap_distance_mm: float,
    rng: np.random.Generator,
    speed_of_sound_in_air: float = 343.0  # m/s
) -> np.ndarray:
    """
    Simulates acoustic emission reflection of space energy between gaps (knocking on walls).
    Creates delayed, attenuated multipath echoes added to the original signal.
    """
    reflected = np.copy(signal)
    
    # Randomize the number of significant reflections based on gap
    num_reflections = rng.integers(1, 4)
    
    # Gap distance in meters
    d_m = gap_distance_mm / 1000.0
    
    for i in range(1, num_reflections + 1):
        # The extra distance traveled is roughly 2 * i * d_m (back and forth)
        # Adding some randomness for uneven walls/angles
        dist = 2 * i * d_m * rng.uniform(0.9, 1.2)
        
        # Delay in seconds: time = distance / speed
        time_delay = dist / speed_of_sound_in_air
        delay_samples = int(time_delay * fs)
        
        # If the delay is larger than our signal window, it wraps or cuts off. 
        # Typically the signal window is 2us. Acoustic delays are much longer (milliseconds).
        # Wait, if duration is 2us, acoustic emission is WAY too slow to show up in the same 2us window.
        # But we must simulate the *effect* in the available window, perhaps the signal is continuous.
        # For a 2us window (1GHz), any reflection from >1mm wall takes ~ 3-6us to return.
        # So we simulate microscopic gap reflections or scale the speed purely for the pulse overlap effect.
        # Let's use a microscopic echo delay directly in the nanosecond/microsecond range for electrical/acoustic transduction.
        
        # Scaling the delay so it falls within our 2us window (e.g., 50ns - 500ns delay)
        echo_delay_ns = rng.uniform(50, 500) * i
        delay_samples = int(echo_delay_ns * 1e-9 * fs)
        
        if delay_samples < len(signal):
            # Attenuation factor decreases with distance/bounces
            attenuation = rng.uniform(0.1, 0.4) / i
            
            # Shift signal and add
            echo = np.zeros_like(signal)
            echo[delay_samples:] = signal[:-delay_samples] * attenuation
            reflected += echo
            
    return reflected

def add_aepd_noise(
    signal_length: int,
    signal_power: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Generates Acoustic Emission Partial Discharge (AEPD) background noise.
    Characterized by high-frequency burst/mechanical stress vibrations.
    """
    # AEPD often manifests as colored noise (e.g., violet/blue shifted random bursts)
    # We simulate this as bandpass-filtered noise or high-frequency amplitude-modulated bursts.
    
    white_noise = rng.normal(0, np.sqrt(signal_power) * 0.5, signal_length)
    
    # Create random burst envelopes
    t_env = np.linspace(0, 1, signal_length)
    num_bursts = rng.integers(2, 6)
    envelope = np.ones(signal_length) * 0.1  # base mechanical vibration
    
    for _ in range(num_bursts):
        center = rng.uniform(0.1, 0.9)
        width = rng.uniform(0.01, 0.05)
        # Gaussian burst
        burst = np.exp(-0.5 * ((t_env - center) / width) ** 2)
        envelope += burst * rng.uniform(0.5, 2.0)
        
    return white_noise * envelope


def generate_noisy_signal(
    clean_signal: np.ndarray,
    target_snr_db: float,
    distance_class: int = 0,
    config: Optional[DistanceSignalConfig] = None,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, float]:
    """
    Add noise to achieve target SNR.
    
    Args:
        clean_signal: Clean PD signal
        target_snr_db: Target SNR in dB
        rng: Random number generator
    
    Returns:
        Tuple of (noisy_signal, actual_snr_db)
    """
    if rng is None:
        rng = np.random.default_rng()
        
    if config is None:
        config = DistanceSignalConfig()
    
    signal_power = np.mean(clean_signal ** 2)
    
    if signal_power == 0:
        # No signal, just return noise
        noise = rng.normal(0, 0.1, len(clean_signal))
        return noise, -np.inf
        
    # Apply Acoustic Gap Reflection
    # Map distance_class to approximate gap in mm
    gap_mm_map = {0: 1.0, 1: 1.8, 2: 2.0, 3: 2.5}
    gap_mm = gap_mm_map.get(distance_class, 1.0)
    
    # Add AE reflection (Multipath)
    reflected_signal = apply_acoustic_gap_reflection(
        clean_signal, config.sample_rate, gap_mm, rng
    )
    
    # Add AEPD (structural burst) noise
    aepd_noise = add_aepd_noise(len(clean_signal), signal_power, rng)
    
    # Combine signals
    composite_clean = reflected_signal + aepd_noise
    new_signal_power = np.mean(composite_clean ** 2)
    
    # Calculate required AWGN power for target SNR
    target_snr_linear = 10 ** (target_snr_db / 10)
    noise_power = new_signal_power / target_snr_linear
    noise_std = np.sqrt(noise_power)
    
    # Generate random AWGN
    awgn = rng.normal(0, noise_std, len(clean_signal))
    
    # Add impulsive noise (10% probability on random samples, representing generic EM interference)
    impulse_mask = rng.random(len(clean_signal)) < 0.01
    awgn[impulse_mask] += rng.normal(0, noise_std * 5, np.sum(impulse_mask))
    
    # Final noisy signal
    noisy_signal = composite_clean + awgn
    
    # Calculate actual SNR (against the original purely clean electrical pulse)
    actual_noise_power = np.mean((noisy_signal - clean_signal) ** 2)
    actual_snr = 10 * np.log10(signal_power / actual_noise_power) if actual_noise_power > 0 else np.inf
    
    return noisy_signal, actual_snr


def generate_batch(
    batch_size: int,
    config: Optional[DistanceSignalConfig] = None,
    snr_range: Tuple[float, float] = (-10, 10),
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Generate a batch of distance-based PD signals.
    
    Args:
        batch_size: Number of samples per class
        config: Signal configuration
        snr_range: Range of SNR values in dB
        seed: Random seed
    
    Returns:
        Dictionary with 'clean', 'noisy', 'labels', 'snr' arrays
    """
    if config is None:
        config = DistanceSignalConfig()
    
    rng = np.random.default_rng(seed)
    
    total_samples = batch_size * 4  # 4 classes
    
    clean_signals = np.zeros((total_samples, config.signal_length))
    noisy_signals = np.zeros((total_samples, config.signal_length))
    labels = np.zeros(total_samples, dtype=np.int64)
    snrs = np.zeros(total_samples)
    
    idx = 0
    for distance_class in range(4):
        for _ in range(batch_size):
            # Generate clean signal
            _, clean, _ = generate_distance_pd_signal(distance_class, config, rng)
            
            # Add noise (including acoustic AE and AEPD reflections newly implemented)
            target_snr = rng.uniform(*snr_range)
            noisy, actual_snr = generate_noisy_signal(
                clean, target_snr, distance_class=distance_class, config=config, rng=rng
            )
            
            clean_signals[idx] = clean
            noisy_signals[idx] = noisy
            labels[idx] = distance_class
            snrs[idx] = actual_snr
            idx += 1
    
    # Shuffle
    perm = rng.permutation(total_samples)
    
    return {
        'clean': clean_signals[perm],
        'noisy': noisy_signals[perm],
        'labels': labels[perm],
        'snr': snrs[perm]
    }


if __name__ == '__main__':
    # Test signal generation
    import matplotlib.pyplot as plt
    
    config = DistanceSignalConfig()
    
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    
    for i in range(4):
        t, clean, _ = generate_distance_pd_signal(i, config)
        noisy, snr = generate_noisy_signal(clean, -10, distance_class=i, config=config)
        
        axes[i, 0].plot(t * 1e6, clean)
        axes[i, 0].set_title(f'Clean - {config.distance_classes[i]}')
        axes[i, 0].set_ylabel('Amplitude')
        
        axes[i, 1].plot(t * 1e6, noisy)
        axes[i, 1].set_title(f'Noisy (-10dB) - {config.distance_classes[i]}')
    
    axes[-1, 0].set_xlabel('Time (μs)')
    axes[-1, 1].set_xlabel('Time (μs)')
    
    plt.tight_layout()
    plt.savefig('distance_signals_test.png', dpi=150)
    plt.show()
    
    print("Test complete!")
