#!/usr/bin/env python

from dataclasses import dataclass, field
from typing import Optional, Dict, List

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.common.optim.optimizers import MultiAdamConfig


@dataclass
class ConcurrencyConfig:
    actor: str = "threads"
    learner: str = "threads"


@dataclass
class ActorLearnerConfig:
    learner_host: str = "127.0.0.1"
    learner_port: int = 50051
    policy_parameters_push_frequency: int = 4


@dataclass
class NetworkConfig:
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    activate_final: bool = True
    dropout_rate: Optional[float] = None


@dataclass
class PolicyConfig:
    use_tanh_squash: bool = True
    init_final: float = 0.05


@PreTrainedConfig.register_subclass("mlp")
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

    # Architecture specifics
    camera_number: int = 1
    device: str = "cuda"
    storage_device: str = "cpu"
    # Set to "helper2424/resnet10" for hil serl
    vision_encoder_name: str | None = "helper2424/resnet10"
    freeze_vision_encoder: bool = True
    image_encoder_hidden_dim: int = 32
    image_embedding_pooling_dim: int = 8
    latent_dim: int = 256

    # Training parameter
    online_steps: int = 1000000
    online_env_seed: int = 10000
    buffer_capacity: int = 50000
    async_prefetch: bool = False
    online_step_before_learning: int = 100
    policy_update_freq: int = 1

    # Training hyperparameters
    bc_lr: float = 3e-4
    weight_decay: float = 0.0
    batch_size: int = 256
    num_epochs: int = 50
    grad_clip_norm: float = 40.0

    # Network configurations
    actor_network_kwargs: NetworkConfig = field(default_factory=NetworkConfig)
    policy_kwargs: PolicyConfig = field(default_factory=PolicyConfig)
    actor_learner_config: ActorLearnerConfig = field(default_factory=ActorLearnerConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)

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

    @property
    def image_features(self) -> list[str]:
        return [key for key in self.input_features if "image" in key]

    @property
    def observation_delta_indices(self) -> list:
        return None

    @property
    def action_delta_indices(self) -> list:
        return None  # SAC typically predicts one action at a time

    @property
    def reward_delta_indices(self) -> None:
        return None
