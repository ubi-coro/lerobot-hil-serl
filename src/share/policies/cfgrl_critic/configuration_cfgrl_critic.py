"""Configuration objects for the decoupled CFGRL critic family."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import draccus

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig, OptimizerConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig, LRSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE
from share.policies.cfgrl_common.backbones import TimmVisionBackboneConfig, VisionBackboneConfig


@dataclass
class CriticBackboneConfig:
    """Config for the XVLA-free state-action critic backbone.

    ``pool`` selects how the final action-token sequence is reduced before the
    critic head. ``mean`` is usually the safest chunk-level default.
    """

    vision_backbone: VisionBackboneConfig = field(default_factory=TimmVisionBackboneConfig)
    hidden_dim: int = 256
    num_heads: int = 4
    num_fusion_layers: int = 2
    dropout: float = 0.0
    pool: str = "mean"


@dataclass
class CriticHeadConfig(draccus.ChoiceRegistry, abc.ABC):
    """Choice-registered base class for CFGRL critic heads."""

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@CriticHeadConfig.register_subclass("scalar_flow")
@dataclass
class ScalarFlowHeadConfig(CriticHeadConfig):
    num_flow_steps: int = 8
    q_num_samples: int = 4
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.0
    clip_flow_returns: bool = True
    v_min: float = -20.0
    v_max: float = 20.0


@CriticHeadConfig.register_subclass("c51")
@dataclass
class C51HeadConfig(CriticHeadConfig):
    n_atoms: int = 101
    v_min: float = -20.0
    v_max: float = 20.0
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.0


@CriticHeadConfig.register_subclass("iqn")
@dataclass
class IQNHeadConfig(CriticHeadConfig):
    n_tau: int = 32
    tau_embed_dim: int = 64
    n_cos: int = 64
    kappa: float = 1.0
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.0


@CriticHeadConfig.register_subclass("value_flows")
@dataclass
class ValueFlowsHeadConfig(CriticHeadConfig):
    num_flow_steps: int = 10
    bcfm_lambda: float = 1.0
    dcfm_lambda: float = 1.0
    confidence_weight_temp: float = 0.3
    q_agg: str = "min"
    ret_agg: str = "min"
    clip_flow_returns: bool = True
    v_min: float = -20.0
    v_max: float = 20.0
    q_num_samples: int = 8
    hidden_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.0


@PreTrainedConfig.register_subclass("cfgrl_critic")
@dataclass
class CFGRLCriticConfig(PreTrainedConfig):
    """Configuration for the share-local chunk-aware CFGRL critic.

    Key semantics:

    - ``chunk_size`` is the action horizon evaluated by the critic. Advantage
      labels produced from this critic are only valid for that chunk length.
    - ``num_action_samples`` controls how many candidate chunks are sampled when
      estimating ``V(s)`` from an action provider.
    - ``metadata_keys`` are optional non-visual, non-proprio observation fields
      treated as extra critic context, such as task IDs or task parameters.
    """

    # Native action horizon evaluated by the critic.
    chunk_size: int = 8
    gamma: float = 0.99
    tau: float = 0.005
    # Number of sampled action chunks used when estimating V(s) from an action provider.
    num_action_samples: int = 4

    backbone: CriticBackboneConfig = field(default_factory=CriticBackboneConfig)
    head: CriticHeadConfig = field(default_factory=ScalarFlowHeadConfig)

    # Optional non-visual, non-proprio observation features fused into critic context.
    metadata_keys: list[str] = field(default_factory=list)

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
            "REWARD": NormalizationMode.IDENTITY,
        }
    )

    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4
    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 10_000

    @property
    def observation_delta_indices(self) -> list[int]:
        return [0, self.chunk_size]

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    def get_optimizer_preset(self) -> OptimizerConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self) -> LRSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
            peak_lr=self.optimizer_lr,
            decay_lr=self.optimizer_lr * 0.1,
        )

    def validate_features(self) -> None:
        """Validate required critic inputs before model construction."""

        if not self.input_features:
            raise ValueError("CFGRLCriticConfig.input_features must be provided")
        if not self.output_features or ACTION not in self.output_features:
            raise ValueError("CFGRLCriticConfig.output_features must contain 'action'")
        if self.robot_state_feature is None:
            raise ValueError(f"CFGRLCriticConfig requires '{OBS_STATE}' in input_features")
        if not self.image_features:
            raise ValueError("CFGRLCriticConfig requires at least one visual input feature")
        for key in self.metadata_keys:
            if key not in self.input_features:
                raise ValueError(f"metadata key '{key}' is not present in input_features")
