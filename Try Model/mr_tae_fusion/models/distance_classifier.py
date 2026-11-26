"""
Distance-based PD classifier with MR-TAE-Fusion denoising integration.

4-class classification:
- Class 0: 1.0mm gap
- Class 1: 1.8mm gap
- Class 2: 2.0mm gap
- Class 3: 2.5mm gap
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import numpy as np


class DistanceClassifier(nn.Module):
    """
    CNN-based classifier for distance-based PD signal classification.
    
    Takes denoised signals and classifies into 4 distance classes.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        signal_length: int = 2001,
        num_classes: int = 4,
        hidden_channels: list = None
    ):
        super().__init__()
        
        if hidden_channels is None:
            hidden_channels = [32, 64, 128]
        
        self.in_channels = in_channels
        self.signal_length = signal_length
        self.num_classes = num_classes
        
        # CNN feature extractor
        layers = []
        prev_channels = in_channels
        
        for i, out_channels in enumerate(hidden_channels):
            layers.extend([
                nn.Conv1d(prev_channels, out_channels, kernel_size=7, padding=3),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(0.2)
            ])
            prev_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        
        # Calculate flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, signal_length)
            feat_out = self.features(dummy)
            self.flat_size = feat_out.numel()
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.flat_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 1, signal_length)
        
        Returns:
            Logits of shape (batch, num_classes)
        """
        # Feature extraction
        features = self.features(x)
        features = features.view(features.size(0), -1)
        
        # Classification
        logits = self.classifier(features)
        
        return logits
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class labels."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)


class DistanceClassifierWithDenoising(nn.Module):
    """
    Combined model: MR-TAE-Fusion denoising + Distance classification.
    """
    
    def __init__(
        self,
        denoiser: nn.Module,
        classifier: nn.Module,
        freeze_denoiser: bool = True
    ):
        super().__init__()
        
        self.denoiser = denoiser
        self.classifier = classifier
        
        if freeze_denoiser:
            for param in self.denoiser.parameters():
                param.requires_grad = False
    
    def forward(
        self, 
        noisy_signal: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with denoising and classification.
        
        Args:
            noisy_signal: Noisy input (batch, 1, signal_length)
        
        Returns:
            Tuple of (class_logits, denoised_signal)
        """
        # Denoise
        with torch.no_grad() if not self.training else torch.enable_grad():
            denoised = self.denoiser(noisy_signal)
        
        # Handle dict output from MR-TAE-Fusion
        if isinstance(denoised, dict):
            denoised_signal = denoised.get('denoised', denoised.get('reconstruction'))
        else:
            denoised_signal = denoised
        
        # Classify
        logits = self.classifier(denoised_signal)
        
        return logits, denoised_signal
    
    def predict(self, noisy_signal: torch.Tensor) -> torch.Tensor:
        """Get predicted class labels from noisy input."""
        with torch.no_grad():
            logits, _ = self.forward(noisy_signal)
            return torch.argmax(logits, dim=1)


def create_distance_classifier(
    signal_length: int = 2001,
    pretrained_denoiser_path: Optional[str] = None,
    device: str = 'cuda'
) -> Tuple[DistanceClassifier, Optional[DistanceClassifierWithDenoising]]:
    """
    Create distance classifier, optionally with pretrained denoiser.
    
    Args:
        signal_length: Input signal length
        pretrained_denoiser_path: Path to pretrained MR-TAE-Fusion model
        device: Device to use
    
    Returns:
        Tuple of (standalone_classifier, combined_model)
    """
    # Create standalone classifier
    classifier = DistanceClassifier(
        in_channels=1,
        signal_length=signal_length,
        num_classes=4
    ).to(device)
    
    combined_model = None
    
    # Load pretrained denoiser if provided
    if pretrained_denoiser_path:
        try:
            from ..models import MRTAEFusion
            from ..config import get_config
            
            config = get_config()
            denoiser = MRTAEFusion(config.model).to(device)
            
            checkpoint = torch.load(pretrained_denoiser_path, map_location=device)
            denoiser.load_state_dict(checkpoint['model_state_dict'])
            
            combined_model = DistanceClassifierWithDenoising(
                denoiser=denoiser,
                classifier=classifier,
                freeze_denoiser=True
            ).to(device)
            
            print(f"Loaded pretrained denoiser from {pretrained_denoiser_path}")
        except Exception as e:
            print(f"Could not load denoiser: {e}")
    
    return classifier, combined_model


if __name__ == '__main__':
    # Test classifier
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    classifier = DistanceClassifier(
        in_channels=1,
        signal_length=2001,
        num_classes=4
    ).to(device)
    
    # Test forward pass
    dummy_input = torch.randn(8, 1, 2001).to(device)
    output = classifier(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Predicted classes: {classifier.predict(dummy_input)}")
    
    # Count parameters
    total_params = sum(p.numel() for p in classifier.parameters())
    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
