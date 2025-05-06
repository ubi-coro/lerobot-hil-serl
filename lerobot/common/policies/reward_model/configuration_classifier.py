from dataclasses import dataclass, field
from typing import List, Optional

from lerobot.common.optim.optimizers import AdamWConfig, OptimizerConfig
from lerobot.common.optim.schedulers import LRSchedulerConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode


@PreTrainedConfig.register_subclass(name="reward_classifier")
@dataclass
class RewardClassifierConfig(PreTrainedConfig):
    """Configuration for the Reward Classifier model."""

    name: str = "reward_classifier"
    num_classes: int = 2
    hidden_dim: int = 256
    dropout_rate: float = 0.1
    model_name: str = "helper2424/resnet10"
    device: str = "cpu"
    model_type: str = "cnn"  # "transformer" or "cnn"
    num_cameras: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 10
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
        }
    )
    image_keys: Optional[List[str]] = None

    @property
    def observation_delta_indices(self) -> List | None:
        return None

    @property
    def action_delta_indices(self) -> List | None:
        return None

    @property
    def reward_delta_indices(self) -> List | None:
        return None

    def get_optimizer_preset(self) -> OptimizerConfig:
        return AdamWConfig(
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            grad_clip_norm=self.grad_clip_norm,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return None

    def validate_features(self) -> None:
        """Validate feature configurations."""
        has_image = any(key.startswith("observation.image") for key in self.input_features)
        if not has_image:
            raise ValueError(
                "You must provide an image observation (key starting with 'observation.image') in the input features"
            )
