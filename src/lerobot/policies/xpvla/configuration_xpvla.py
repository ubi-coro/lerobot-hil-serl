# policies/cfgrl_ac/configuration_cfgrl_ac.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.xvla.configuration_xvla import XVLAConfig
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK


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
    critic_type: Literal["iql", "vformer_iql", "dist_eval"] = "iql"
    critic_cfg: dict[str, Any] = field(default_factory=dict)
