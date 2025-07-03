#!/usr/bin/env python

from dataclasses import dataclass, field
from typing import Optional, Dict, List

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.common.optim.optimizers import MultiAdamConfig


@dataclass
class NetworkConfig:
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    activate_final: bool = True
    dropout_rate: Optional[float] = None


@dataclass
class PolicyConfig:
    use_tanh_squash: bool = True
    init_final: float = 0.05
    log_std_min: float = -5.0
    log_std_max: float = 2.0


@PreTrainedConfig.register_subclass("bc")
@dataclass
class MLPConfig(PreTrainedConfig):
    """
    Behavior Cloning (BC) configuration.

    This config defines hyperparameters for behavior cloning training,
    including network architectures, normalization, and optimizer settings.
    """
    # Normalization mapping for inputs and outputs
    normalization_mapping: Dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ENV": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )
    # Dataset statistics for normalization (min/max or mean/std)
    dataset_stats: Optional[Dict[str, Dict[str, List[float]]]] = None

    # Training hyperparameters
    bc_lr: float = 3e-4
    weight_decay: float = 0.0
    batch_size: int = 256
    num_epochs: int = 50

    # Network configurations
    actor_network_kwargs: NetworkConfig = field(default_factory=NetworkConfig)
    policy_kwargs: PolicyConfig = field(default_factory=PolicyConfig)

    def get_optimizer_preset(self) -> MultiAdamConfig:
        # Single group optimizer for BC actor only
        return MultiAdamConfig(
            weight_decay=self.weight_decay,
            optimizer_groups={"actor": {"lr": self.bc_lr}},
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        # Ensure input and output features are present
        has_state = "observation.state" in self.input_features
        has_image = any(k.startswith("observation.image") for k in self.input_features)
        if not (has_state or has_image):
            raise ValueError("BCConfig requires at least 'observation.state' or image inputs.")
        if "action" not in self.output_features:
            raise ValueError("BCConfig requires 'action' in output_features.")
