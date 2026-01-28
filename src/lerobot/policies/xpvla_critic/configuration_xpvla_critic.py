import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

import draccus

from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim import OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies import XVLAConfig
from lerobot.utils.constants import OBS_PREFIX


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
    hidden_dim: int = 1024
    num_layers: int = 2
    dropout: float = 0.0


@CriticHeadConfig.register_subclass("c51")
@dataclass
class C51HeadConfig(CriticHeadConfig):
    # C51 distributional critic
    n_atoms: int = 101
    v_min: float = -10.0
    v_max: float = 10.0

    hidden_dim: int = 1024
    num_layers: int = 2
    dropout: float = 0.0


@CriticHeadConfig.register_subclass("iqn")
@dataclass
class IQNHeadConfig(CriticHeadConfig):
    # IQN distributional critic
    n_tau: int = 32                 # number of quantile samples per forward
    tau_embed_dim: int = 64         # cosine embedding size
    n_cos: int = 64                 # number of cosines for embedding
    kappa: float = 1.0

    hidden_dim: int = 1024
    num_layers: int = 2
    dropout: float = 0.0


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

    # Backbone parameters
    backbone_pretrained_path: str = "lerobot/xvla-base"
    backbone: Any = None  # used to hold the backbone policy config
    num_reused_xvla_blocks: int = 1  # reuse N transformer blocks from XVLA (copied into critic)
    fusion: str = "concat_transformer"  # how to fuse action and vlm tokens, must be in {"film","cross_attn","concat_transformer"}
    pool: str = "mean"  # must be in {"mean","first","last"}
    hidden_dim: int = 1024  # mlp hidden dim after pooling
    dropout: float = 0.0
    num_cross_attn_blocks: int = 2  # number of small cross-attn for fusion == "cross_attn"
    n_heads: int = 8  # number of attention heads for fusion == "cross_attn"
    mlp_ratio: float = 4.0  # mlp ratio for fusion == "cross_attn"
    fixed_domain_id: int = 0  # domain id that is fed into xvla's action encoder
    action_encoder_timestep: float = 0.0  # t for the timestep embedding that is fed into xvla's action encoder
    use_pos_emb: bool = True
    use_aux_visual_inputs: bool = True

    # Head parameters
    head: CriticHeadConfig = field(default_factory=lambda: ScalarHeadConfig())

    # TD parameters
    gamma: float = 0.99
    tau: float = 0.005

    # Caching keys
    vlm_cache_key: str = f"{OBS_PREFIX}.vlm_cache"
    policy_actions_key: str = f"{OBS_PREFIX}.policy_actions"

    # Freezing parameters
    freeze_vlm: bool = True
    freeze_vlm_proj: bool = False
    freeze_aux_visual_proj: bool = False
    freeze_action_encoder: bool = False
    freeze_transformer_blocks: bool = False

    # Training parameters
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    scheduler_warmup_steps: int = 1_000

    def __post_init__(self):
        self.backbone = PreTrainedConfig.from_pretrained(self.backbone_pretrained_path)
        self.backbone.pretrained_path = Path(self.backbone_pretrained_path)

        if not isinstance(self.backbone, XVLAConfig):
            raise ValueError(f"Pretrained backbone config must be a XVLAConfig, not {type(self.backbone)}")

    @property
    def chunk_size(self) -> int:
        return self.backbone.chunk_size

    @property
    def observation_delta_indices(self) -> list | None:
        # Not used for critic; keep for API compatibility.
        return [0, self.chunk_size]

    @property
    def action_delta_indices(self) -> list | None:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> list | None:
        return list(range(self.chunk_size))

    @property
    def drop_n_last_frames(self) -> int:
        return self.chunk_size

    def get_optimizer_preset(self) -> OptimizerConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return None

    def validate_features(self) -> None:
        # Enforce: if you rely on cached VLM features, VLM must be frozen.
        # (We also freeze defensively in the backbone once it sees cache keys.)
        if self.vlm_cache_key:
            if not self.freeze_vlm:
                raise ValueError(
                    "XPVLACriticConfig.freeze_vlm must be True when vlm_cache_key caching is enabled."
                )
