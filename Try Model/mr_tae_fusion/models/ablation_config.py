"""
Ablation configuration for MR-TAE model family.
"""

from dataclasses import dataclass


@dataclass
class AblationConfig:
    use_wavelet_pooling: bool = True
    use_bigru: bool = True
    use_swin: bool = True
    use_attention_gates: bool = True
    use_mtl: bool = True
    loss_config: str = "A"
    model_id: str = "MR-TAE-FULL"

