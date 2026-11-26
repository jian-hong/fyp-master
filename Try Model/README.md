# MR-TAE-Fusion: Multi-Task Recurrent-Transformer Autoencoder

A Hybrid Multi-Task Recurrent-Transformer Autoencoder for Partial Discharge Signal Processing at -20 dB SNR.

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
mr_tae_fusion/
├── config.py              # Hyperparameters & settings
├── data/
│   ├── pulse_generators.py    # DEP/DOP synthesis
│   ├── noise_generators.py    # Impulsive & TGAN noise
│   ├── dataset.py             # PyTorch Dataset class
│   └── mat_loader.py          # Load .mat files from Q.lin dataset
├── models/
│   ├── wavelet_layers.py      # DWT/IDWT modules (db4)
│   ├── mwcnn.py               # MWCNN encoder/decoder
│   ├── attention.py           # Attention Gates
│   ├── swin_transformer.py    # 1D Swin Transformer
│   ├── fusion_head.py         # Dual-Stream Segmentation
│   └── mr_tae_fusion.py       # Complete model
├── training/
│   ├── losses.py              # Multi-task loss with uncertainty
│   └── trainer.py             # Training loop
└── evaluation/
    ├── metrics.py             # SNR, NCC, IoU, Dice
    └── visualization.py       # Plots & analysis
```

## Key Features

- **MWCNN Backbone**: DWT/IDWT with Daubechies-4 wavelets for lossless downsampling
- **Hybrid Bottleneck**: BiGRU + 1D Swin Transformer for temporal & global context
- **Attention Gates**: Learnable skip connection filtering
- **Dual-Stream Fusion**: Joint denoising + semantic segmentation
- **Multi-Task Loss**: Uncertainty-weighted loss balancing
- **TGAN Augmentation**: Real site noise modeling from Q.lin dataset

## Class Mapping

| Simulation Type | Physics Category | Class ID |
|-----------------|------------------|----------|
| A (Sparse PD) | Corona | 1 |
| B (Spike-dense) | Surface | 2 |
| C (10mm) | Internal | 3 |
| D (18mm) | Internal | 3 |
| E (20mm) | Internal | 3 |
| F (25mm) | Internal | 3 |
| Background | Background | 0 |

## Usage

```python
from mr_tae_fusion.models import MRTAEFusion
from mr_tae_fusion.config import Config

config = Config()
model = MRTAEFusion(config)

# Forward pass
denoised, segmentation = model(noisy_signal)
```
