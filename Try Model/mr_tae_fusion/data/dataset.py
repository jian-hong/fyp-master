"""
PyTorch Dataset for PD signal denoising and segmentation.

Provides curriculum learning support and real-time data augmentation.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List, Dict, Union
from pathlib import Path

from .pulse_generators import generate_pd_signal, generate_batch
from .noise_generators import (
    generate_composite_noise,
    generate_wgn,
    generate_impulsive_noise,
    add_noise_at_snr,
    CurriculumNoiseScheduler
)
from ..config import Config, SignalConfig, NoiseConfig


class PDSignalDataset(Dataset):
    """
    PyTorch Dataset for PD signal denoising and segmentation.
    
    Generates synthetic signals on-the-fly with curriculum learning support.
    
    Args:
        config: Configuration object
        num_samples: Number of samples in dataset
        signal_types: List of signal types to include
        epoch: Current training epoch (for curriculum learning)
        total_epochs: Total training epochs
        mode: 'train', 'val', or 'test'
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        num_samples: int = 1000,
        signal_types: Optional[List[str]] = None,
        epoch: int = 0,
        total_epochs: int = 100,
        mode: str = 'train',
        seed: Optional[int] = None
    ):
        if config is None:
            config = Config()
        
        self.config = config
        self.num_samples = num_samples
        self.signal_types = signal_types or ['A', 'B', 'C', 'D', 'E', 'F']
        self.epoch = epoch
        self.total_epochs = total_epochs
        self.mode = mode

        # When set, overrides internal CurriculumNoiseScheduler so generated noise tracks
        # training.curricular_scheduler.CurricularScheduler phases (trainer calls each epoch).
        self._phase_snr_range: Optional[Tuple[float, float]] = None
        self._phase_noise_type: Optional[str] = None
        
        # Initialize random generator
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Curriculum scheduler
        self.curriculum = CurriculumNoiseScheduler(config.noise)
        
        # Pre-compute signal type for each sample (for stratification)
        self._sample_types = [
            self.signal_types[i % len(self.signal_types)]
            for i in range(num_samples)
        ]
        
        # Time vector (cached) — must match signal_length exactly
        fs = config.signal.sample_rate
        self.t = np.arange(config.signal.signal_length) / fs

    def set_training_phase(
        self,
        epoch_idx: int,
        snr_range: Tuple[float, float],
        noise_type: str,
        total_epochs: int,
    ) -> None:
        """Align synthetic corruption with Trainer's CurricularScheduler for this epoch."""
        self.epoch = int(epoch_idx)
        self.total_epochs = int(total_epochs)
        lo, hi = float(snr_range[0]), float(snr_range[1])
        self._phase_snr_range = (min(lo, hi), max(lo, hi))
        self._phase_noise_type = str(noise_type)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Generate a single sample.
        
        Returns:
            Dictionary with:
            - 'noisy': Noisy signal (1, L)
            - 'clean': Clean signal (1, L)
            - 'mask': Segmentation mask (L,)
            - 'snr': Applied SNR in dB
            - 'type': Signal type index
        """
        # Get signal type for this sample
        signal_type = self._sample_types[idx]
        
        # Generate clean signal with segmentation mask
        _, clean_signal, mask = generate_pd_signal(
            signal_type, 
            self.config.signal
        )
        
        # Prefer trainer-aligned phase (see set_training_phase); else legacy scheduler
        if self._phase_snr_range is not None and self._phase_noise_type is not None:
            noise_type = self._phase_noise_type
            noise = self._generate_noise(noise_type)
            lo, hi = self._phase_snr_range
            target_snr = float(self.rng.uniform(lo, hi))
        else:
            noise_type = self.curriculum.get_noise_type(self.epoch, self.total_epochs)
            noise = self._generate_noise(noise_type)
            target_snr = self.curriculum.sample_snr(
                self.epoch,
                self.total_epochs,
                self.rng,
            )
        
        # Add noise at target SNR
        noisy_signal, actual_snr = add_noise_at_snr(clean_signal, noise, target_snr)
        
        # Convert to tensors
        noisy_tensor = torch.tensor(noisy_signal, dtype=torch.float32).unsqueeze(0)
        clean_tensor = torch.tensor(clean_signal, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(mask, dtype=torch.long)
        
        # Type index for potential use
        type_idx = self.signal_types.index(signal_type)
        
        return {
            'noisy': noisy_tensor,
            'clean': clean_tensor,
            'mask': mask_tensor,
            'snr': torch.tensor(actual_snr, dtype=torch.float32),
            'type': torch.tensor(type_idx, dtype=torch.long)
        }
    
    def _generate_noise(self, noise_type: str) -> np.ndarray:
        """Generate noise based on type and curriculum phase."""
        n = self.config.signal.signal_length
        if noise_type == 'wgn':
            return generate_wgn(n, self.config.noise.wgn_amplitude, self.rng)

        elif noise_type == 'wgn_impulsive':
            wgn = generate_wgn(n, self.config.noise.wgn_amplitude, self.rng)
            impulsive = generate_impulsive_noise(
                n, probability=self.config.noise.impulse_probability, rng=self.rng
            )
            return wgn + impulsive

        else:  # 'composite'
            return generate_composite_noise(self.t, self.config.noise, self.rng)
    
    def update_epoch(self, epoch: int):
        """Update current epoch for curriculum learning."""
        self.epoch = epoch
    
    def get_curriculum_info(self) -> Dict:
        """Get current curriculum learning state."""
        snr_range = self.curriculum.get_snr_range(self.epoch, self.total_epochs)
        noise_type = self.curriculum.get_noise_type(self.epoch, self.total_epochs)
        
        return {
            'phase': self.curriculum.current_phase,
            'snr_range': snr_range,
            'noise_type': noise_type,
            'epoch': self.epoch
        }


class RealNoiseDataset(Dataset):
    """
    Dataset for loading real noise segments from .mat files.
    
    Used for TGAN training or direct noise injection.
    """
    
    def __init__(
        self,
        noise_segments: np.ndarray,
        transform: Optional[callable] = None
    ):
        """
        Args:
            noise_segments: Array of noise segments (N x L)
            transform: Optional transform to apply
        """
        self.segments = noise_segments
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.segments)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        segment = self.segments[idx].copy()
        
        if self.transform:
            segment = self.transform(segment)
        
        return torch.tensor(segment, dtype=torch.float32)


def create_dataloaders(
    config: Config,
    train_samples: int = 3000,
    val_samples: int = 500,
    test_samples: int = 500,
    epoch: int = 0,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        config: Configuration object
        train_samples: Number of training samples
        val_samples: Number of validation samples  
        test_samples: Number of test samples
        epoch: Current epoch for curriculum
        num_workers: Number of dataloader workers
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = PDSignalDataset(
        config=config,
        num_samples=train_samples,
        epoch=epoch,
        total_epochs=config.training.total_epochs,
        mode='train',
        seed=config.training.seed
    )
    
    val_dataset = PDSignalDataset(
        config=config,
        num_samples=val_samples,
        epoch=epoch,
        total_epochs=config.training.total_epochs,
        mode='val',
        seed=config.training.seed + 1
    )
    
    test_dataset = PDSignalDataset(
        config=config,
        num_samples=test_samples,
        epoch=config.training.total_epochs - 1,  # Test at hardest difficulty
        total_epochs=config.training.total_epochs,
        mode='test',
        seed=config.training.seed + 2
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
