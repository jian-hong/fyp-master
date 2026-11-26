"""Data generation and loading modules."""

from .pulse_generators import generate_dep, generate_dop, generate_pd_signal
from .noise_generators import (
    generate_wgn,
    generate_impulsive_noise,
    generate_powerline_noise,
    generate_magnetostrictive_hum,
    generate_composite_noise,
    add_noise_at_snr
)
from .transfer_function import generate_tank_transfer_function, apply_acoustic_transfer_function
from .dataset import PDSignalDataset
from .mat_loader import load_mat_data, extract_noise_segments
from .distance_signals import (
    generate_distance_pd_signal,
    generate_noisy_signal,
    generate_batch as generate_distance_batch,
    DistanceSignalConfig
)

__all__ = [
    'generate_dep',
    'generate_dop', 
    'generate_pd_signal',
    'generate_wgn',
    'generate_impulsive_noise',
    'generate_powerline_noise',
    'generate_magnetostrictive_hum',
    'generate_composite_noise',
    'add_noise_at_snr',
    'generate_tank_transfer_function',
    'apply_acoustic_transfer_function',
    'PDSignalDataset',
    'load_mat_data',
    'extract_noise_segments',
    'generate_distance_pd_signal',
    'generate_noisy_signal',
    'generate_distance_batch',
    'DistanceSignalConfig',
]
