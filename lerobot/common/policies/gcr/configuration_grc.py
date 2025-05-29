from dataclasses import dataclass, field
from typing import List, Optional

from lerobot.common.optim.optimizers import AdamConfig, OptimizerConfig
from lerobot.common.optim.schedulers import LRSchedulerConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode


@PreTrainedConfig.register_subclass(name="reward_classifier")
@dataclass
class GCRConfig(PreTrainedConfig):
    """Configuration for the Reward Classifier model."""

    name: str = "grc_reward_model"
    model_name: str = "helper2424/resnet10"
    img_key: str = "observation.image"
    lang_key: str = "task"
    device: str = "cpu"
    model_id: str = "RN50"

    learning_rate: float = 1e-5
    weight_decay: float = 0.001
    discount: float = 0.98
    num_negatives: int = 0
    metric: str = "cos"
    freeze_language_encoder: bool = False
    from_scratch: bool = False

    vision_weight: float = 1.0
    lang_weight: float = 1.0
    clip_weight: float = 1.0

    contrastive_omega1: float = 1.0    # pull-together weight
    contrastive_omega2: float = 1.0,   # push-apart weight
    contrastive_mode: str = "sc",      # "sc"  = simple-contrastive   (Eq. 3)
                                       # "ic"  = InfoNCE-contrastive (Eq. 4)

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
        }
    )

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
        return AdamConfig(
            lr=self.learning_rate,
            weight_decay=self.weight_decay
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
