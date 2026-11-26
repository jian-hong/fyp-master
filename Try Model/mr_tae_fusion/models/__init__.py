"""Model architecture modules."""

from .wavelet_layers import DWT1d, IDWT1d, get_wavelet_filters
from .mwcnn import MWCNNEncoder, MWCNNDecoder, MWCNNBlock
from .attention import AttentionGate
from .swin_transformer import SwinTransformer1D, SwinTransformerBlock
from .fusion_head import DualStreamFusionHead
from .ablation_config import AblationConfig
from .mr_tae_fusion import MRTAEFusion, create_model
from .tgan import NoiseGenerator, NoiseDiscriminator, TGAN, TGANNoiseLoader
from .distance_classifier import (
    DistanceClassifier,
    DistanceClassifierWithDenoising,
    create_distance_classifier
)

__all__ = [
    'DWT1d',
    'IDWT1d',
    'get_wavelet_filters',
    'MWCNNEncoder',
    'MWCNNDecoder',
    'MWCNNBlock',
    'AttentionGate',
    'SwinTransformer1D',
    'SwinTransformerBlock',
    'DualStreamFusionHead',
    'AblationConfig',
    'MRTAEFusion',
    'create_model',
    'NoiseGenerator',
    'NoiseDiscriminator',
    'TGAN',
    'TGANNoiseLoader',
    'DistanceClassifier',
    'DistanceClassifierWithDenoising',
    'create_distance_classifier',
]
