"""Configuration for the share-local visuomotor CFGRL policy."""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig, OptimizerConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig, LRSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE
from share.policies.cfgrl_common.backbones import TimmVisionBackboneConfig, VisionBackboneConfig


@PreTrainedConfig.register_subclass("cfgrl_policy")
@dataclass
class CFGRLPolicyConfig(PreTrainedConfig):
    """Configuration for the DiT-style CFGRL flow policy.

    Key semantics:

    - ``chunk_size`` is the native horizon predicted by the generative policy.
    - ``n_action_steps`` controls how many actions from a sampled chunk are
      consumed before resampling during rollout.
    - ``metadata_keys`` are optional non-visual, non-proprio observation fields
      fused into the policy context, such as task identifiers or task parameters.
    - ``previous_action_key`` optionally points to a previous action chunk or
      summary tensor that should be fused into the observation context.
    - ``condition_key`` stores CFGRL labels in the training batch. ``0`` means
      "ordinary / behavior" and ``1`` means "good / optimal". Missing labels
      imply unconditional generation.
    - ``weight_key`` stores per-sample extraction weights for CFGRL-style
      weighted imitation.
    - ``default_rollout_condition`` and ``default_guidance_scale`` are rollout
      defaults baked into checkpoints. ``None`` means unconditional rollout.
    """

    # Future-facing observation history field. The current policy uses the most
    # recent observation, but checkpoints still record the intended stack size.
    n_obs_steps: int = 1
    # Native chunk horizon predicted by the generative policy.
    chunk_size: int = 8
    # Number of actions consumed from a sampled chunk before resampling.
    n_action_steps: int = 8
    hidden_dim: int = 256
    time_embed_dim: int = 64
    num_transformer_layers: int = 2
    num_attention_heads: int = 4
    dropout: float = 0.0
    num_denoising_steps: int = 8
    time_beta_alpha: float = 1.5
    time_beta_beta: float = 1.0

    backbone: VisionBackboneConfig = field(default_factory=TimmVisionBackboneConfig)

    # Dataset field carrying CFGRL labels: 0 = ordinary/behavior, 1 = good/optimal.
    condition_key: str = "cfgrl_condition"
    # Dataset field carrying per-sample CFGRL extraction weights.
    weight_key: str = "cfgrl_weight"
    condition_dropout_p: float = 0.2
    # Rollout defaults saved in checkpoints. ``None`` means unconditional rollout.
    default_rollout_condition: int | None = None
    default_guidance_scale: float | None = None

    # Optional non-visual, non-proprio observation features fused into context,
    # such as task IDs, task embeddings, or task parameters.
    metadata_keys: list[str] = field(default_factory=list)
    # Optional observation key containing a previous action chunk or summary.
    previous_action_key: str | None = None

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4
    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 10_000

    @property
    def observation_delta_indices(self) -> list[int]:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

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
        """Validate required policy inputs before model construction."""

        if not self.input_features:
            raise ValueError("CFGRLPolicyConfig.input_features must be provided")
        if not self.output_features or ACTION not in self.output_features:
            raise ValueError("CFGRLPolicyConfig.output_features must contain 'action'")
        if self.robot_state_feature is None:
            raise ValueError(f"CFGRLPolicyConfig requires '{OBS_STATE}' in input_features")
        if not self.image_features:
            raise ValueError("CFGRLPolicyConfig requires at least one visual input feature")
        if self.n_action_steps <= 0 or self.chunk_size <= 0:
            raise ValueError("n_action_steps and chunk_size must be > 0")
        if self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps must be <= chunk_size")
        for key in self.metadata_keys:
            if key not in self.input_features:
                raise ValueError(f"metadata key '{key}' is not present in input_features")
        if self.previous_action_key is not None and self.previous_action_key not in self.input_features:
            raise ValueError(f"previous_action_key '{self.previous_action_key}' is not present in input_features")
