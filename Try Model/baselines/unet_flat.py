import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetFlat(nn.Module):
    """Plain 1D U-Net, 4 down / 4 up, no attention, no Swin, no BiGRU."""
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Encoder
        in_ch = in_channels
        for feature in features:
            self.encoder.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, feature, kernel_size=3, padding=1),
                    nn.BatchNorm1d(feature),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(feature, feature, kernel_size=3, padding=1),
                    nn.BatchNorm1d(feature),
                    nn.ReLU(inplace=True)
                )
            )
            in_ch = feature
            
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(features[-1], features[-1]*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(features[-1]*2),
            nn.ReLU(inplace=True),
            nn.Conv1d(features[-1]*2, features[-1]*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(features[-1]*2),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose1d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(
                nn.Sequential(
                    nn.Conv1d(feature * 2, feature, kernel_size=3, padding=1),
                    nn.BatchNorm1d(feature),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(feature, feature, kernel_size=3, padding=1),
                    nn.BatchNorm1d(feature),
                    nn.ReLU(inplace=True)
                )
            )
            
        self.final_conv = nn.Conv1d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        skip_connections = []
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)
            skip_connection = skip_connections[i//2]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[-1])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[i+1](concat_skip)
            
        return self.final_conv(x)
