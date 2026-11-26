"""
MR-TAE-Fusion: Complete Multi-Task Recurrent-Transformer Autoencoder.

Integrates MWCNN backbone, BiGRU-Swin bottleneck, Attention Gates,
and Dual-Stream Fusion Head for joint denoising and segmentation.

Supports ablation studies via AblationConfig: any component can be
toggled off to measure its contribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict

from .mwcnn import MWCNNEncoder, MWCNNDecoder, MWCNNDecoderBlock, ConvBlock, StandardConvEncoder
from .attention import AttentionGate, GatedSkipConnection
from .swin_transformer import SwinTransformer1D
from .fusion_head import DualStreamFusionHead, ReconstructionHead
from .ablation_config import AblationConfig
from ..config import ModelConfig


# ---------------------------------------------------------------------------
# Bottleneck variants
# ---------------------------------------------------------------------------

class HybridBottleneck(nn.Module):
    """BiGRU → Swin Transformer bottleneck (FULL model)."""

    def __init__(
        self,
        in_channels: int = 256,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        gru_dropout: float = 0.1,
        swin_dim: int = 256,
        swin_heads: int = 8,
        swin_depth: int = 2,
        swin_window: int = 32,
        swin_mlp_ratio: float = 4.0,
        swin_dropout: float = 0.1,
    ):
        super().__init__()
        self.pre_gru = nn.Conv1d(in_channels, gru_hidden * 2, kernel_size=1)
        self.gru = nn.GRU(
            input_size=gru_hidden * 2, hidden_size=gru_hidden,
            num_layers=gru_layers, batch_first=True, bidirectional=True,
            dropout=gru_dropout if gru_layers > 1 else 0,
        )
        self.gru_to_swin = nn.Conv1d(gru_hidden * 2, swin_dim, kernel_size=1)
        self.swin = SwinTransformer1D(
            dim=swin_dim, depth=swin_depth, num_heads=swin_heads,
            window_size=swin_window, mlp_ratio=swin_mlp_ratio,
            dropout=swin_dropout, attention_dropout=swin_dropout,
        )
        self.output_proj = nn.Conv1d(swin_dim, in_channels, kernel_size=1)
        self.residual = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.output_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        h = self.pre_gru(x).permute(0, 2, 1)
        h, _ = self.gru(h)
        h = h.permute(0, 2, 1)
        h = self.gru_to_swin(h)
        h = self.swin(h)
        return self.output_proj(h) + residual


class BiGRUOnlyBottleneck(nn.Module):
    """BiGRU bottleneck without Swin Transformer."""

    def __init__(self, in_channels: int = 256, gru_hidden: int = 128,
                 gru_layers: int = 2, gru_dropout: float = 0.1):
        super().__init__()
        self.pre_gru = nn.Conv1d(in_channels, gru_hidden * 2, kernel_size=1)
        self.gru = nn.GRU(
            input_size=gru_hidden * 2, hidden_size=gru_hidden,
            num_layers=gru_layers, batch_first=True, bidirectional=True,
            dropout=gru_dropout if gru_layers > 1 else 0,
        )
        self.output_proj = nn.Conv1d(gru_hidden * 2, in_channels, kernel_size=1)
        self.residual = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.output_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        h = self.pre_gru(x).permute(0, 2, 1)
        h, _ = self.gru(h)
        return self.output_proj(h.permute(0, 2, 1)) + residual


class SwinOnlyBottleneck(nn.Module):
    """Swin Transformer bottleneck without BiGRU."""

    def __init__(self, in_channels: int = 256, swin_dim: int = 256,
                 swin_heads: int = 8, swin_depth: int = 2,
                 swin_window: int = 32, swin_mlp_ratio: float = 4.0,
                 swin_dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, swin_dim, kernel_size=1)
        self.swin = SwinTransformer1D(
            dim=swin_dim, depth=swin_depth, num_heads=swin_heads,
            window_size=swin_window, mlp_ratio=swin_mlp_ratio,
            dropout=swin_dropout, attention_dropout=swin_dropout,
        )
        self.output_proj = nn.Conv1d(swin_dim, in_channels, kernel_size=1)
        self.residual = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.output_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        h = self.input_proj(x)
        h = self.swin(h)
        return self.output_proj(h) + residual


class ConvBottleneck(nn.Module):
    """Fallback conv bottleneck when both BiGRU and Swin are disabled."""

    def __init__(self, in_channels: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_channels, in_channels, kernel_size=3),
            ConvBlock(in_channels, in_channels, kernel_size=3),
        )
        self.output_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class MRTAEFusion(nn.Module):
    """
    MR-TAE-Fusion with full ablation support.

    Enabled components are controlled by ``ablation`` flags:
      use_wavelet_pooling  – MWCNN encoder/decoder vs Conv+MaxPool
      use_bigru            – BiGRU in bottleneck
      use_swin             – Swin Transformer in bottleneck
      use_attention_gates  – gated skip connections
      use_mtl              – segmentation head (multi-task learning)
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        ablation: Optional[AblationConfig] = None,
    ):
        super().__init__()
        if config is None:
            config = ModelConfig()
        if ablation is None:
            ablation = AblationConfig()

        self.config = config
        self.ablation = ablation

        # --- Input projection ---
        self.input_proj = nn.Conv1d(
            config.in_channels, config.encoder_channels[0],
            kernel_size=7, padding=3,
        )

        # --- Encoder ---
        if ablation.use_wavelet_pooling:
            self.encoder = MWCNNEncoder(
                in_channels=config.encoder_channels[0],
                channel_list=config.encoder_channels,
                wavelet="db4", kernel_size=config.encoder_kernel_size,
            )
            encoder_out_channels = config.encoder_channels[-1] * 2  # DWT doubles
        else:
            self.encoder = StandardConvEncoder(
                in_channels=config.encoder_channels[0],
                channel_list=config.encoder_channels,
                kernel_size=config.encoder_kernel_size,
            )
            encoder_out_channels = config.encoder_channels[-1]

        # --- Pre-bottleneck projection ---
        self.pre_bottleneck = nn.Conv1d(
            encoder_out_channels, config.bottleneck_channels, kernel_size=1,
        )

        # --- Bottleneck ---
        if ablation.use_bigru and ablation.use_swin:
            self.bottleneck = HybridBottleneck(
                in_channels=config.bottleneck_channels,
                gru_hidden=config.gru_hidden_size,
                gru_layers=config.gru_num_layers,
                gru_dropout=config.gru_dropout,
                swin_dim=config.swin_embed_dim,
                swin_heads=config.swin_num_heads,
                swin_depth=config.swin_depth,
                swin_window=config.swin_window_size,
                swin_mlp_ratio=config.swin_mlp_ratio,
                swin_dropout=config.swin_dropout,
            )
        elif ablation.use_bigru:
            self.bottleneck = BiGRUOnlyBottleneck(
                in_channels=config.bottleneck_channels,
                gru_hidden=config.gru_hidden_size,
                gru_layers=config.gru_num_layers,
                gru_dropout=config.gru_dropout,
            )
        elif ablation.use_swin:
            self.bottleneck = SwinOnlyBottleneck(
                in_channels=config.bottleneck_channels,
                swin_dim=config.swin_embed_dim,
                swin_heads=config.swin_num_heads,
                swin_depth=config.swin_depth,
                swin_window=config.swin_window_size,
                swin_mlp_ratio=config.swin_mlp_ratio,
                swin_dropout=config.swin_dropout,
            )
        else:
            self.bottleneck = ConvBottleneck(in_channels=config.bottleneck_channels)

        # --- Attention gates ---
        skip_channels = config.encoder_channels[::-1]
        gate_channel_list = [config.bottleneck_channels] + list(config.decoder_channels[:-1])

        if ablation.use_attention_gates:
            self.attention_gates = nn.ModuleList()
            for skip_ch, gate_ch in zip(skip_channels, gate_channel_list):
                self.attention_gates.append(
                    GatedSkipConnection(
                        gate_channels=gate_ch, skip_channels=skip_ch,
                        use_residual=True,
                        intermediate_channels=config.attention_intermediate_channels,
                    )
                )
        else:
            self.attention_gates = None

        # --- Decoder ---
        self.decoder = MWCNNDecoder(
            bottleneck_channels=config.bottleneck_channels,
            channel_list=config.decoder_channels,
            skip_channels=skip_channels,
            wavelet="db4", kernel_size=config.encoder_kernel_size,
        )

        # --- Reconstruction head ---
        self.recon_head = ReconstructionHead(
            in_channels=config.decoder_channels[-1],
            out_channels=config.in_channels,
        )

        # --- Segmentation head (multi-task) ---
        if ablation.use_mtl:
            self.seg_head = DualStreamFusionHead(
                global_channels=config.bottleneck_channels,
                local_channels=config.decoder_channels[-1],
                fusion_channels=config.fusion_channels,
                num_classes=config.num_classes,
            )
        else:
            self.seg_head = None

        self.input_length = config.signal_length

    # -----------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        input_length = x.shape[-1]

        x = self.input_proj(x)
        encoded, skip_features, skip_lengths = self.encoder(x)
        encoded = self.pre_bottleneck(encoded)
        bottleneck_features = self.bottleneck(encoded)

        reversed_skips = list(reversed(skip_features))
        reversed_lengths = list(reversed(skip_lengths))

        current = bottleneck_features
        for i, skip in enumerate(reversed_skips):
            if self.attention_gates is not None:
                gated = self.attention_gates[i](skip, current)
            else:
                gated = skip
            current = self.decoder.blocks[i](current, gated, reversed_lengths[i])

        decoded = current
        denoised = self.recon_head(decoded, input_length)

        if self.seg_head is not None:
            segmentation = self.seg_head(bottleneck_features, decoded, input_length)
        else:
            segmentation = torch.zeros(
                x.shape[0], self.config.num_classes, input_length,
                device=denoised.device, dtype=denoised.dtype,
            )

        return denoised, segmentation

    def get_denoised(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[0]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_summary(self) -> Dict:
        return {
            "total_params": self.count_parameters(),
            "encoder_params": sum(p.numel() for p in self.encoder.parameters()),
            "bottleneck_params": sum(p.numel() for p in self.bottleneck.parameters()),
            "decoder_params": sum(p.numel() for p in self.decoder.parameters()),
            "attention_params": (
                sum(p.numel() for p in self.attention_gates.parameters())
                if self.attention_gates else 0
            ),
            "seg_head_params": (
                sum(p.numel() for p in self.seg_head.parameters())
                if self.seg_head else 0
            ),
            "ablation": {
                "wavelet": self.ablation.use_wavelet_pooling,
                "bigru": self.ablation.use_bigru,
                "swin": self.ablation.use_swin,
                "attention_gates": self.ablation.use_attention_gates,
                "mtl": self.ablation.use_mtl,
                "loss_config": self.ablation.loss_config,
            },
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_model(
    config: Optional[ModelConfig] = None,
    ablation: Optional[AblationConfig] = None,
) -> MRTAEFusion:
    """Build an MR-TAE-Fusion model with optional ablation flags."""
    model = MRTAEFusion(config, ablation)

    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    return model
