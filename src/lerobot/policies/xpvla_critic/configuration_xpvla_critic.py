import abc
from dataclasses import dataclass, field

import draccus

from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim import OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies import XVLAConfig


@dataclass
class XPVLACriticBackboneConfig:
    xvla: XVLAConfig = field(default_factory=lambda: XVLAConfig(repo_id="xvla/base"))

    # How CriticBackbone fuses (state_tokens, action_tokens)
    fusion: str = "concat_transformer"  # {"film","cross_attn","concat_transformer"}

    # Concat-transformer specific: reuse N transformer blocks from XVLA (copied into critic)
    num_reused_xvla_blocks: int = 1

    # Cross-attention specific: number of small cross-attn blocks to add
    num_cross_attn_blocks: int = 2

    # Pooling
    pool: str = "mean"  # {"mean","first","last"}

    # Common MLP settings after pooling
    hidden_dim: int = 1024
    dropout: float = 0.0

    # Parameters for fusion == "cross_attn"
    n_heads: int = 8
    mlp_ratio: float = 4.0

    # Action encoder settings
    fixed_domain_id: int = 0

    # t≈0 in the XVLA timestep embedding (noise-free actions)
    action_encoder_timestep: float = 0.0

    use_pos_emb: bool = True
    use_aux_visual_inputs: bool = True

    # Caching keys
    vlm_features_key: str = "vlm_features"
    next_vlm_features_key: str = "next_vlm_features"
    policy_actions_key: str = "policy_actions"              # [B,K,H,A]
    next_policy_actions_key: str = "next_policy_actions"    # [B,K,H,A]

    freeze_vlm: bool = True
    freeze_vlm_proj: bool = False
    freeze_aux_visual_proj: bool = False
    freeze_action_encoder: bool = False
    freeze_transformer_blocks: bool = False


@dataclass
class CriticHeadConfig(draccus.ChoiceRegistry, abc.ABC):
    """Draccus choice base: policy.head.type=... selects subclass."""

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@CriticHeadConfig.register_subclass("scalar")
@dataclass
class ScalarHeadConfig(CriticHeadConfig):
    # Twin-Q head MLP
    q_hidden_dim: int = 1024
    q_num_layers: int = 2
    q_dropout: float = 0.0
    huber_delta: float = 1.0


@CriticHeadConfig.register_subclass("c51")
@dataclass
class C51HeadConfig(CriticHeadConfig):
    # C51 distributional critic
    n_atoms: int = 101
    v_min: float = -10.0
    v_max: float = 10.0

    q_hidden_dim: int = 1024
    q_num_layers: int = 2
    q_dropout: float = 0.0


@CriticHeadConfig.register_subclass("iqn")
@dataclass
class IQNHeadConfig(CriticHeadConfig):
    # IQN distributional critic
    n_tau: int = 32                 # number of quantile samples per forward
    tau_embed_dim: int = 64         # cosine embedding size
    n_cos: int = 64                 # number of cosines for embedding

    q_hidden_dim: int = 1024
    q_num_layers: int = 2
    q_dropout: float = 0.0


@CriticHeadConfig.register_subclass("value_flows")
@dataclass
class ValueFlowsHeadConfig(CriticHeadConfig):
    # Flow discretization
    num_flow_steps: int = 10

    # Loss weights
    bcfm_lambda: float = 1.0
    dcfm_lambda: float = 1.0

    # Confidence weighting
    confidence_weight_temp: float = 0.3
    q_agg: str = "min"    # {"min","mean"} for combining twin flows for std/weights and q
    ret_agg: str = "min"  # {"min","mean"} for combining next return estimates

    # Clipping of integrated returns
    clip_flow_returns: bool = True
    v_min: float = -10.0
    v_max: float = 10.0

    # Q estimation (used by forward/expectation + advantage)
    q_num_samples: int = 8

    # Vector field MLP
    hidden_dim: int = 512
    num_layers: int = 4
    dropout: float = 0.0


@PreTrainedConfig.register_subclass("xpvla_critic")
@dataclass
class XPVLACriticConfig(PreTrainedConfig):
    # TD settings
    gamma: float = 0.99
    chunk_size: int = 50
    tau: float = 0.005

    # Backbone + head configs (NO kwargs)
    backbone: XPVLACriticBackboneConfig = field(default_factory=lambda: XPVLACriticBackboneConfig())
    head: CriticHeadConfig = field(default_factory=lambda: ScalarHeadConfig())

    # Training preset
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    scheduler_warmup_steps: int = 1_000

    @property
    def observation_delta_indices(self) -> list | None:
        # Not used for critic; keep for API compatibility.
        return None

    @property
    def action_delta_indices(self) -> list | None:
        return None

    @property
    def reward_delta_indices(self) -> list | None:
        return None

    def get_optimizer_preset(self) -> OptimizerConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return CosineDecayWithWarmupSchedulerConfig(warmup_steps=self.scheduler_warmup_steps)

    def validate_features(self) -> None:
        # Enforce: if you rely on cached VLM features, VLM must be frozen.
        # (We also freeze defensively in the backbone once it sees cache keys.)
        if self.vlm_features_key or self.next_vlm_features_key:
            if not self.backbone.freeze_vlm:
                raise ValueError(
                    "XPVLACriticConfig.freeze_vlm must be True when vlm_features_key caching is enabled."
                )
