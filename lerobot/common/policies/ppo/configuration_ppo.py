#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Any

from lerobot.common.policies.pretrained import PreTrainedConfig


@dataclass
class PPOConfig(PreTrainedConfig):
    # Feature specifications
    input_features: Dict[str, Tuple[int, ...]]
    output_features: Dict[str, Tuple[int, ...]]
    normalization_mapping: Dict[str, Any] = field(default_factory=dict)
    dataset_stats: Optional[Dict[str, Dict[str, float]]] = None

    # Training parameters
    discount: float = 0.998
    critic_lr: float = 3e-4
    actor_lr: float = 3e-4
    state_encoder_hidden_dim: int = 256
    grad_clip_norm: float = 40.0
    gae_lambda: float = 0.95
    num_epochs: int = 4
    num_samples_per_update: int = 50

    # Encoder sharing
    shared_encoder: bool = True

    # Network architectures
    actor_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    critic_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])

    # PPO-specific hyperparameters
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.01

    # Log-std clipping bounds
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    # Initial fixed std (if desired)
    init_std: Optional[float] = None

    def __post_init__(self):
        super().__post_init__()
        # Any validation specific to SAC configuration

    def get_optimizer_preset(self) -> MultiAdamConfig:
        return MultiAdamConfig(
            weight_decay=0.0,
            optimizer_groups={
                "actor": {"lr": self.actor_lr},
                "critic": {"lr": self.critic_lr},
                "temperature": {"lr": self.temperature_lr},
            },
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        has_image = any(key.startswith("observation.image") for key in self.input_features)
        has_state = "observation.state" in self.input_features

        if not (has_state or has_image):
            raise ValueError(
                "You must provide either 'observation.state' or an image observation (key starting with 'observation.image') in the input features"
            )

        if "action" not in self.output_features:
            raise ValueError("You must provide 'action' in the output features")

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
