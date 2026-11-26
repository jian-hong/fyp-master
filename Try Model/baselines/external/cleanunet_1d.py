import sys
import os
from pathlib import Path
import torch
import torch.nn as nn

current_dir = Path(__file__).parent.resolve()
cleanunet_dir = current_dir / "CleanUNet"
if str(cleanunet_dir) not in sys.path:
    sys.path.append(str(cleanunet_dir))

from network import CleanUNet as OriginalCleanUNet

class CleanUNet1D(nn.Module):
    """
    Wrapper for NVIDIA CleanUNet operating on 1D AE signals.
    """
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # Use a lightweight-to-standard configuration to retain core features
        self.model = OriginalCleanUNet(
            channels_input=in_channels,
            channels_output=out_channels,
            channels_H=64,
            max_H=768,
            encoder_n_layers=8,
            kernel_size=4,
            stride=2,
            tsfm_n_layers=3, 
            tsfm_n_head=8,
            tsfm_d_model=512, 
            tsfm_d_inner=2048
        )
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.model(x)

if __name__ == "__main__":
    model = CleanUNet1D()
    x = torch.randn(2, 1, 2048)
    y = model(x)
    print(f"CleanUNet1D input: {x.shape}, output: {y.shape}")
