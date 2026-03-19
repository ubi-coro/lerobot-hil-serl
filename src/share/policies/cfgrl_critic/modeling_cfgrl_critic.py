"""Chunk-aware, XVLA-independent critic family for the CFGRL stack."""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, DONE, OBS_STATE, REWARD
from share.policies.cfgrl_common.action_providers import CachedActionProvider, DatasetActionProvider
from share.policies.cfgrl_common.backbones import build_vision_backbone, should_train_backbone_outputs
from share.policies.cfgrl_common.modules import ActionChunkTokenEncoder, CrossAttentionBlock, MLP
from .configuration_cfgrl_critic import CFGRLCriticConfig, CriticBackboneConfig
from .heads.c51 import C51TwinQHead
from .heads.factory import CriticHead, TDTarget, make_critic_head
from .heads.iqn import IQNTwinQHead
from .heads.scalar_flow import ScalarFlowTwinQHead
from .heads.value_flows import ValueFlowsTwinQHead


class CriticBackbone(nn.Module):
    """Encode observation/action-chunk pairs into critic features.

    Action tokens remain the native sequence. They cross-attend to visual patch
    tokens plus compact state/metadata tokens before being pooled for the critic
    head, which keeps the critic lightweight while preserving token-level visual
    structure.
    """

    def __init__(
        self,
        config: CriticBackboneConfig,
        *,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        camera_keys: list[str],
        metadata_dims: dict[str, int],
    ) -> None:
        super().__init__()
        self.config = config
        self.camera_keys = camera_keys
        self.metadata_dims = metadata_dims
        self.vision_backbone = build_vision_backbone(config.vision_backbone)
        self.camera_proj = nn.Linear(self.vision_backbone.output_dim, config.hidden_dim)
        self.camera_token_proj = nn.Linear(self.vision_backbone.output_dim, config.hidden_dim)
        self.camera_token_embedding = nn.Embedding(max(len(camera_keys), 1), config.hidden_dim)
        self.state_proj = nn.Linear(state_dim, config.hidden_dim)
        self.metadata_proj = nn.ModuleDict(
            {key: nn.Linear(dim, config.hidden_dim) for key, dim in metadata_dims.items()}
        )
        self.action_encoder = ActionChunkTokenEncoder(
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
            chunk_size=chunk_size,
            time_embed_dim=32,
            dropout=config.dropout,
        )
        self.cross_blocks = nn.ModuleList(
            [CrossAttentionBlock(config.hidden_dim, config.num_heads, dropout=config.dropout) for _ in range(config.num_fusion_layers)]
        )
        self.post = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            MLP(config.hidden_dim, config.hidden_dim, config.hidden_dim, num_layers=2, dropout=config.dropout),
        )
        self._configure_vision_tuning()

    @property
    def out_dim(self) -> int:
        return self.config.hidden_dim

    def _prepare_image(self, image: Tensor) -> Tensor:
        if image.ndim == 5:
            image = image[:, -1]
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.shape[1] not in {1, 3} and image.shape[-1] in {1, 3}:
            image = image.permute(0, 3, 1, 2).contiguous()
        return image.float()

    def _configure_vision_tuning(self) -> None:
        """Align external projector trainability with the configured vision mode."""

        train_projectors = should_train_backbone_outputs(self.config.vision_backbone.tune_mode)
        for module in (self.camera_proj, self.camera_token_proj, self.camera_token_embedding):
            for param in module.parameters():
                param.requires_grad_(train_projectors)

    def _context_tokens(self, state: dict[str, Any]) -> list[Tensor]:
        """Build non-action context tokens for cross-attention."""

        state_tensor = state[OBS_STATE]
        if state_tensor.ndim == 3:
            state_tensor = state_tensor[:, -1]
        tokens = [self.state_proj(state_tensor.float()).unsqueeze(1)]
        for key in self.metadata_dims:
            tensor = state[key]
            if tensor.ndim > 2:
                tensor = tensor.flatten(1)
            tokens.append(self.metadata_proj[key](tensor.float()).unsqueeze(1))
        return tokens

    def encode_context_tokens(self, state: dict[str, Any]) -> Tensor:
        """Encode state and visual observations into critic context tokens."""

        obs_tokens = self._context_tokens(state)
        for camera_index, key in enumerate(self.camera_keys):
            image = self._prepare_image(state[key])
            vision_out = self.vision_backbone(image)
            camera_bias = self.camera_token_embedding.weight[camera_index].view(1, 1, -1)
            obs_tokens.append(self.camera_proj(vision_out.pooled).unsqueeze(1) + camera_bias)
            obs_tokens.append(self.camera_token_proj(vision_out.tokens) + camera_bias)
        return torch.cat(obs_tokens, dim=1)

    def forward(self, *, state: dict[str, Any], action: Tensor) -> Tensor:
        obs_tokens = self.encode_context_tokens(state)

        time = torch.zeros(action.shape[0], device=action.device, dtype=action.dtype)
        action_tokens = self.action_encoder(action, time=time, context=None)
        for block in self.cross_blocks:
            action_tokens = block(action_tokens, obs_tokens)
        if self.config.pool == "first":
            pooled = action_tokens[:, 0]
        elif self.config.pool == "last":
            pooled = action_tokens[:, -1]
        else:
            pooled = action_tokens.mean(dim=1)
        return self.post(pooled)


class CFGRLCritic(PreTrainedPolicy):
    """Decoupled chunk critic supporting scalar, discrete, and flow-value heads."""

    config_class = CFGRLCriticConfig
    name = "cfgrl_critic"

    def __init__(self, config: CFGRLCriticConfig):
        super().__init__(config)
        config.validate_features()
        self.gamma = float(config.gamma)
        self.chunk_size = int(config.chunk_size)
        self.tau = float(config.tau)
        self.action_dim = config.action_feature.shape[0]
        self.state_dim = config.robot_state_feature.shape[0]
        self.camera_keys = sorted(config.image_features.keys())
        self.metadata_dims = {key: config.input_features[key].shape[0] for key in config.metadata_keys}

        self.backbone = CriticBackbone(
            config.backbone,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            chunk_size=self.chunk_size,
            camera_keys=self.camera_keys,
            metadata_dims=self.metadata_dims,
        )
        self.head: CriticHead = make_critic_head(feat_dim=self.backbone.out_dim, config=config.head)

        self.target_backbone = copy.deepcopy(self.backbone).eval()
        self.target_head = copy.deepcopy(self.head).eval()
        for param in self.target_backbone.parameters():
            param.requires_grad_(False)
        for param in self.target_head.parameters():
            param.requires_grad_(False)

    def get_optim_params(self) -> dict:
        """Return trainable parameters for optimizer construction."""

        return {"params": [p for p in self.parameters() if p.requires_grad]}

    def reset(self) -> None:
        return None

    def forward(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor]]:
        return self.compute_loss(batch)

    def encode_state_action(
        self,
        batch: dict[str, Any],
        *,
        state_key: str = "state",
        action_key: str = ACTION,
        use_target: bool = False,
    ) -> Tensor:
        """Encode one batch of state/action chunks with either live or target weights."""

        backbone = self.target_backbone if use_target else self.backbone
        return backbone(state=batch[state_key], action=batch[action_key])

    def q_out(self, batch: dict[str, Any], *, use_target: bool = False) -> dict[str, Tensor]:
        """Return raw head outputs for the requested network."""

        feat = self.encode_state_action(batch, use_target=use_target)
        head = self.target_head if use_target else self.head
        return head(feat)

    def q(self, batch: dict[str, Any], *, use_target: bool = False) -> Tensor:
        """Return scalar critic values by applying the head expectation operator."""

        head = self.target_head if use_target else self.head
        return head.expectation(self.q_out(batch, use_target=use_target))

    def estimate_value(
        self,
        batch: dict[str, Any],
        action_provider,
        *,
        next_state: bool = False,
        use_target: bool = False,
    ) -> Tensor:
        """Estimate ``V(s)`` by averaging over action chunks from an action provider."""

        actions = action_provider.sample_next_actions(batch, self.config.num_action_samples) if next_state else action_provider.sample_current_actions(batch, self.config.num_action_samples)
        batch_size, num_samples, _, _ = actions.shape
        state_key = "next_state" if next_state else "state"
        rep_state = self._repeat_tree(batch[state_key], num_samples)
        flat_actions = actions.reshape(batch_size * num_samples, *actions.shape[2:])
        backbone = self.target_backbone if use_target else self.backbone
        head = self.target_head if use_target else self.head
        feat = backbone(state=rep_state, action=flat_actions)
        out = head(feat)
        reduced = head.reduce_over_action_samples(out, B=batch_size, K=num_samples)
        return head.expectation(reduced)

    @torch.no_grad()
    def compute_advantage(self, batch: dict[str, Any], action_provider) -> Tensor:
        """Compute a simple advantage-like quantity for policy extraction."""

        return self.q(batch) - self.estimate_value(batch, action_provider, next_state=False)

    def compute_loss(self, batch: dict[str, Any], action_provider=None) -> tuple[Tensor, dict[str, Tensor]]:
        """Build chunk TD targets and compute the active critic loss."""

        provider = action_provider or self._default_action_provider(batch)
        reward_chunk = self._accumulate_chunk_reward(batch[REWARD])
        done = self._chunk_done(batch[DONE])
        gamma_h = float(self.gamma ** self.chunk_size)

        if isinstance(self.head, ValueFlowsTwinQHead):
            return self.head.loss_from_batch(
                batch=batch,
                critic=self,
                reward_chunk=reward_chunk,
                done=done,
                gamma_H=gamma_h,
                action_provider=provider,
            )

        if isinstance(self.head, ScalarFlowTwinQHead):
            with torch.no_grad():
                next_value = self.estimate_value(batch, provider, next_state=True, use_target=True)
                target = reward_chunk + (1.0 - done) * gamma_h * next_value
            feat = self.encode_state_action(batch)
            loss = self.head.loss_from_target(feat, target)
            out = self.head(feat)
        else:
            with torch.no_grad():
                next_out = self._next_state_out(batch, provider)
                target = self.target_head.build_target(
                    reward_chunk=reward_chunk,
                    done=done,
                    gamma_H=gamma_h,
                    next_out=next_out,
                )
            out = self.q_out(batch)
            loss = self.head.loss(out, target)

        logs = {
            "critic/loss": loss.detach(),
            "critic/reward_mean": reward_chunk.mean().detach(),
            "critic/q_mean": self.head.expectation(out).mean().detach(),
        }
        return loss, logs

    @torch.no_grad()
    def soft_update_target(self) -> None:
        """Polyak-average the target backbone and head."""

        for param, target_param in zip(self.backbone.parameters(), self.target_backbone.parameters(), strict=True):
            target_param.data.mul_(1.0 - self.tau).add_(param.data, alpha=self.tau)
        self.target_head.soft_update_from(self.head, tau=self.tau)

    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Return one action chunk from the default action provider path."""

        if ACTION in batch:
            return batch[ACTION]
        provider = self._default_action_provider(batch)
        return provider.sample_current_actions(batch, 1)[:, 0]

    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Return the first action in the predicted chunk."""

        return self.predict_action_chunk(batch)[:, 0]

    def _default_action_provider(self, batch: dict[str, Any]):
        if "next_policy_action_samples" in batch or "policy_action_samples" in batch:
            return CachedActionProvider()
        return DatasetActionProvider()

    def _next_state_out(self, batch: dict[str, Any], action_provider) -> dict[str, Tensor]:
        actions = action_provider.sample_next_actions(batch, self.config.num_action_samples)
        batch_size, num_samples, _, _ = actions.shape
        rep_state = self._repeat_tree(batch["next_state"], num_samples)
        flat_actions = actions.reshape(batch_size * num_samples, *actions.shape[2:])
        feat = self.target_backbone(state=rep_state, action=flat_actions)
        out = self.target_head(feat)
        return self.target_head.reduce_over_action_samples(out, B=batch_size, K=num_samples)

    def _chunk_done(self, done: Tensor) -> Tensor:
        done = done.float()
        if done.ndim == 1:
            return done
        if done.shape[1] >= self.chunk_size:
            return done[:, self.chunk_size - 1]
        return done[:, -1]

    def _accumulate_chunk_reward(self, reward: Tensor) -> Tensor:
        """Accumulate chunk rewards with discounting inside the chunk boundary."""

        if reward.ndim == 1:
            return reward
        if reward.shape[1] >= self.chunk_size:
            reward = reward[:, : self.chunk_size]
        gammas = (self.gamma ** torch.arange(reward.shape[1], device=reward.device, dtype=reward.dtype)).view(1, -1)
        return (reward * gammas).sum(dim=1)

    @staticmethod
    def _repeat_tree(value: Any, repeats: int) -> Any:
        if torch.is_tensor(value):
            return value.repeat_interleave(repeats, dim=0)
        if isinstance(value, dict):
            return {key: CFGRLCritic._repeat_tree(item, repeats) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            repeated = [CFGRLCritic._repeat_tree(item, repeats) for item in value]
            return type(value)(repeated)
        return value
