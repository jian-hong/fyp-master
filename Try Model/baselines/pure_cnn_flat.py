import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += self.shortcut(residual)
        return self.relu(x)

class PureCNNFlat(nn.Module):
    """8-layer 1D CNN, residual"""
    def __init__(self, in_channels=1, out_channels=1, features=64):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv1d(in_channels, features, kernel_size=7, padding=3),
            nn.BatchNorm1d(features),
            nn.ReLU(inplace=True)
        )
        self.layers = nn.ModuleList([ResidualBlock(features, features) for _ in range(8)])
        self.final_conv = nn.Conv1d(features, out_channels, kernel_size=1)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.initial(x)
        for layer in self.layers:
            x = layer(x)
        return self.final_conv(x)
