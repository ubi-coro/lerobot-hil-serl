from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import AdamWConfig
from lerobot.policies.xvla.configuration_xvla import XVLAConfig
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK


@dataclass
class CriticBackboneConfig:
    # Reuse XVLA latent size
    hidden_size: int = 768

    # How we fuse action-seq tokens with state tokens
    fusion: Literal["film", "cross_attn", "concat_transformer"] = "film"

    # If using concat_transformer: how many copied XVLA transformer blocks to apply
    n_reused_transformer_blocks: int = 0

    # If using cross_attn: number of cross-attn blocks
    n_cross_attn_blocks: int = 2
    n_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1

    # Pooling
    pool: Literal["mean"] = "mean"

    # Whether to include aux visual tokens from forward_vlm
    use_aux_visual_inputs: bool = True

    # Reuse XVLA action encoder’s time embedding: use t=0 for near-noise-free
    action_encoder_t: float = 0.0

    # DomainAwareLinear needs a domain_id; you want to exclude domain-id semantics => constant 0
    fixed_domain_id: int = 0

    # Add positional embeddings when using concat_transformer
    use_pos_emb: bool = True

    # Freezing knobs (independent)
    freeze_vlm: bool = True
    freeze_vlm_proj: bool = True
    freeze_aux_visual_proj: bool = True
    freeze_action_encoder: bool = False
    freeze_reused_transformer_blocks: bool = False

    # Safety: if someone configures concat_transformer but provides 0 blocks
    # we still fuse by concatenation + pooling (no blocks).
    allow_concat_without_blocks: bool = True


@PreTrainedConfig.register_subclass("xpvla")
@dataclass
class XPVLAConfig(PreTrainedConfig):
    """
    Top-level config for CFGRL-AC:
      - contains an embedded XVLA actor loaded from a base checkpoint
      - contains a critic subsystem (plug-in algorithm) trained in separate mode
      - supports policy extraction training with advantage conditioning + label dropout
      - supports inference with CFG-style guidance strength controlled via config
    """
    # ---- training / runtime mode ----
    train_mode: Literal["critic", "policy"] = "policy"

    # ---- actor backbone (X-VLA) ----
    xvla: XVLAConfig = XVLAConfig()
    xvla_pretrained_path: str = "lerobot/xvla-base"
    xvla_revision: Optional[str] = None
    xvla_strict_load: bool = False

    # ---- Observation keys ----
    advantage_key: str = "advantage_label"
    pos_adv_text: str = "Optimal: true"
    neg_adv_text: str = "Optimal: false"
    seperator_text: str = " "
    cond_tokens_key: str = field(default_factory=lambda: f"{OBS_LANGUAGE_TOKENS}_cond")
    cond_attn_key: str = field(default_factory=lambda: f"{OBS_LANGUAGE_ATTENTION_MASK}_cond")

    # --- classifier-free guidance parameters
    advantage_label_dropout_p: float = 0.2
    guidance_scale: float = 2.0

    # ---- critic configuration ----
    critic_backbone: CriticBackboneConfig = field(default_factory=CriticBackboneConfig)
    critic_type: Literal["iql", "vformer_iql", "dist_eval"] = "iql"
    critic_cfg: dict[str, Any] = field(default_factory=dict)

    def get_optimizer_preset(self) -> AdamWConfig:
        # todo: write custom optimizer for value, use xvla default for policy
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        # same as optimizer
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

    @property
    def observation_delta_indices(self) -> None:
        return [0, self.xvla.chunk_size]

    @property
    def action_delta_indices(self) -> list:
        return list(range(2 * self.xvla.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return list(range(self.xvla.chunk_size))


