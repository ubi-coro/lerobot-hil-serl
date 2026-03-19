"""From-scratch visuomotor CFGRL flow policy with a token-aware DiT-style head.

The policy learns a continuous-time vector field over action chunks. We use the
standard linear interpolation convention

``x_t = t * x_0 + (1 - t) * x_1``

with:

- ``x_0`` = Gaussian noise
- ``x_1`` = the target dataset action chunk

The target vector field is therefore the constant displacement

``dx_t / dt = x_0 - x_1``.

Training regresses that field from randomly sampled points along the line between
noise and data. Sampling starts from ``x_0`` at ``t = 1`` and integrates the
learned field backwards to ``t = 0`` with a negative Euler step. Because the
training target matches the derivative of the same interpolation path, the sign
convention is consistent between training and generation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from share.policies.cfgrl_common.backbones import build_vision_backbone, should_train_backbone_outputs
from share.policies.cfgrl_common.modules import (
    ActionChunkTokenEncoder,
    CrossAttentionBlock,
    DiTBlock,
    DiTFinalLayer,
    MLP,
    ObservationFusion,
    sinusoidal_time_embedding,
)
from .configuration_cfgrl_policy import CFGRLPolicyConfig


@dataclass
class ObservationEncoding:
    """Encoded observation state shared across training and rollout sampling."""

    context: Tensor
    tokens: Tensor
    camera_features: dict[str, Tensor]


class DiTFlowMatchingHead(nn.Module):
    """Flow-matching head that predicts chunk vector fields with DiT blocks.

    Action tokens are the denoising sequence. They cross-attend to visual and
    state/metadata tokens, then receive DiT-style adaLN modulation from the
    compact global conditioning vector.
    """

    def __init__(self, config: CFGRLPolicyConfig, action_dim: int) -> None:
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.chunk_encoder = ActionChunkTokenEncoder(
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
            chunk_size=config.chunk_size,
            time_embed_dim=config.time_embed_dim,
            dropout=config.dropout,
        )
        self.cond_proj = MLP(
            in_dim=config.hidden_dim + config.time_embed_dim,
            hidden_dim=config.hidden_dim,
            out_dim=config.hidden_dim,
            num_layers=2,
            dropout=config.dropout,
        )
        self.cross_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(config.hidden_dim, config.num_attention_heads, dropout=config.dropout)
                for _ in range(config.num_transformer_layers)
            ]
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_attention_heads,
                    cond_dim=config.hidden_dim,
                    dropout=config.dropout,
                )
                for _ in range(config.num_transformer_layers)
            ]
        )
        self.output = DiTFinalLayer(config.hidden_dim, action_dim, cond_dim=config.hidden_dim)

    def forward(self, noisy_actions: Tensor, time: Tensor, context: Tensor, observation_tokens: Tensor) -> Tensor:
        """Predict the continuous-time action vector field."""

        time_emb = sinusoidal_time_embedding(time, self.config.time_embed_dim)
        cond = self.cond_proj(torch.cat([context, time_emb], dim=-1))
        tokens = self.chunk_encoder(noisy_actions, time=time, context=context)
        for cross_block, block in zip(self.cross_blocks, self.blocks, strict=True):
            tokens = cross_block(tokens, observation_tokens)
            tokens = block(tokens, cond)
        return self.output(tokens, cond)


class CFGRLPolicy(PreTrainedPolicy):
    """Visuomotor CFGRL policy trained by weighted flow matching."""

    config_class = CFGRLPolicyConfig
    name = "cfgrl_policy"

    def __init__(self, config: CFGRLPolicyConfig):
        super().__init__(config)
        config.validate_features()

        self.action_dim = config.action_feature.shape[0]
        self.state_dim = config.robot_state_feature.shape[0]
        self.backbone = build_vision_backbone(config.backbone)
        self.camera_keys = sorted(config.image_features.keys())
        self.camera_proj = nn.Linear(self.backbone.output_dim, config.hidden_dim)
        self.camera_token_proj = nn.Linear(self.backbone.output_dim, config.hidden_dim)
        self.camera_token_embedding = nn.Embedding(max(len(self.camera_keys), 1), config.hidden_dim)
        self.state_proj = nn.Linear(self.state_dim, config.hidden_dim)

        metadata_dim = sum(config.input_features[key].shape[0] for key in config.metadata_keys)
        self.metadata_proj = nn.ModuleDict(
            {key: nn.Linear(config.input_features[key].shape[0], config.hidden_dim) for key in config.metadata_keys}
        )
        prev_action_dim = 0
        self.prev_action_proj: nn.Linear | None = None
        if config.previous_action_key is not None:
            prev_action_dim = int(torch.tensor(config.input_features[config.previous_action_key].shape).prod().item())
            self.prev_action_proj = nn.Linear(prev_action_dim, config.hidden_dim)

        fused_dim = len(self.camera_keys) * config.hidden_dim + self.state_dim + metadata_dim + prev_action_dim
        self.fusion = ObservationFusion(fused_dim, config.hidden_dim, dropout=config.dropout)
        self.condition_embedding = nn.Embedding(3, config.hidden_dim)
        self.head = DiTFlowMatchingHead(config, self.action_dim)
        self._time_dist = torch.distributions.Beta(config.time_beta_alpha, config.time_beta_beta)
        self._queues: dict[str, deque] | None = None
        self._configure_vision_tuning()
        self.reset()

    def get_optim_params(self) -> dict:
        """Return trainable parameters for optimizer construction."""

        return {"params": [p for p in self.parameters() if p.requires_grad]}

    def reset(self) -> None:
        """Clear rollout-time action chunk caches."""

        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}

    def _configure_vision_tuning(self) -> None:
        """Align external projector trainability with the configured backbone mode."""

        train_projectors = should_train_backbone_outputs(self.config.backbone.tune_mode)
        for module in (self.camera_proj, self.camera_token_proj, self.camera_token_embedding):
            for param in module.parameters():
                param.requires_grad_(train_projectors)

    def _extract_obs(self, batch: dict[str, Tensor] | dict) -> dict[str, Tensor]:
        """Normalize training and rollout batch layouts to a flat observation dict."""

        if OBS_STATE in batch:
            return batch  # already an observation dict
        state = batch.get("state")
        if isinstance(state, dict):
            return state
        raise KeyError("Could not infer observation dict from batch")

    def _prepare_image(self, image: Tensor) -> Tensor:
        if image.ndim == 5:
            image = image[:, -1]
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError(f"Expected image tensor with ndim 4, got {tuple(image.shape)}")
        if image.shape[1] not in {1, 3} and image.shape[-1] in {1, 3}:
            image = image.permute(0, 3, 1, 2).contiguous()
        return image.float()

    def _flatten_optional_feature(self, obs: dict[str, Tensor], key: str) -> Tensor:
        feat = obs[key]
        if feat.ndim == 3:
            feat = feat.flatten(1)
        elif feat.ndim > 3:
            feat = feat.flatten(1)
        return feat.float()

    def _raw_condition_to_embedding_index(self, condition: Tensor | None) -> Tensor:
        """Map raw CFGRL labels to embedding ids.

        Embedding index ``0`` is reserved for the unconditional branch.
        Dataset labels therefore map as:

        - ``0 -> 1`` for ordinary / behavior-conditioned generation
        - ``1 -> 2`` for good / optimal-conditioned generation
        """

        if condition is None:
            raise ValueError("condition tensor must not be None here")
        cond = condition.to(dtype=torch.long)
        if cond.ndim == 0:
            cond = cond.unsqueeze(0)
        cond = cond.clamp(min=0, max=1)
        return cond + 1

    def _sample_training_condition_indices(self, batch: dict[str, Tensor], mode: str, batch_size: int, device) -> Tensor:
        """Choose the conditioning branch used for training.

        BC is intentionally unconditional. CFGRL extraction uses provided labels
        when available, otherwise it also falls back to unconditional generation.
        Condition dropout only applies when we were going to use a labeled branch.
        """

        if mode == "bc":
            return torch.zeros(batch_size, device=device, dtype=torch.long)

        raw = batch.get(self.config.condition_key)
        if raw is None:
            return torch.zeros(batch_size, device=device, dtype=torch.long)

        raw = raw.to(device=device, dtype=torch.long).view(batch_size)
        cond_idx = self._raw_condition_to_embedding_index(raw)
        if self.training and self.config.condition_dropout_p > 0:
            drop_mask = torch.rand(batch_size, device=device) < self.config.condition_dropout_p
            cond_idx = torch.where(drop_mask, torch.zeros_like(cond_idx), cond_idx)
        return cond_idx

    def encode_observations(self, batch: dict[str, Tensor] | dict) -> ObservationEncoding:
        """Encode images, proprioception, and optional metadata into policy conditioning.

        The returned ``tokens`` keep patch/spatial structure from the vision
        backbone so action tokens can attend to visual layout directly. The
        compact ``context`` vector remains useful for timestep/condition adaLN.
        """

        obs = self._extract_obs(batch)
        state = obs[OBS_STATE]
        if state.ndim == 3:
            state = state[:, -1]

        camera_features: dict[str, Tensor] = {}
        fused_parts: list[Tensor] = []
        obs_tokens: list[Tensor] = [self.state_proj(state.float()).unsqueeze(1)]

        for camera_index, key in enumerate(self.camera_keys):
            image = self._prepare_image(obs[key])
            encoded = self.backbone(image)
            projected_pooled = self.camera_proj(encoded.pooled)
            projected_tokens = self.camera_token_proj(encoded.tokens)
            camera_bias = self.camera_token_embedding.weight[camera_index].view(1, 1, -1)
            projected_tokens = projected_tokens + camera_bias

            camera_features[key] = projected_pooled
            fused_parts.append(projected_pooled)
            obs_tokens.append(projected_tokens)

        fused_parts.append(state.float())
        for key in self.config.metadata_keys:
            flat = self._flatten_optional_feature(obs, key)
            fused_parts.append(flat)
            obs_tokens.append(self.metadata_proj[key](flat).unsqueeze(1))

        if self.config.previous_action_key is not None and self.config.previous_action_key in obs:
            prev_action = self._flatten_optional_feature(obs, self.config.previous_action_key)
            fused_parts.append(prev_action)
            if self.prev_action_proj is not None:
                obs_tokens.append(self.prev_action_proj(prev_action).unsqueeze(1))

        context = self.fusion(torch.cat(fused_parts, dim=-1))
        tokens = torch.cat(obs_tokens, dim=1)
        return ObservationEncoding(context=context, tokens=tokens, camera_features=camera_features)

    def _predict_vector_field(
        self,
        obs_encoding: ObservationEncoding,
        noisy_actions: Tensor,
        time: Tensor,
        *,
        condition_indices: Tensor,
    ) -> Tensor:
        """Apply CFGRL conditioning and predict the action vector field."""

        conditioned_context = obs_encoding.context + self.condition_embedding(condition_indices)
        return self.head(noisy_actions, time, conditioned_context, obs_encoding.tokens)

    @staticmethod
    def _build_flow_training_pair(actions: Tensor, noise: Tensor, time: Tensor) -> tuple[Tensor, Tensor]:
        """Construct the flow-matching training state ``x_t`` and its target field.

        We parameterize a straight path from data to noise:

        ``x_t = t * noise + (1 - t) * actions``

        Its derivative with respect to ``t`` is ``noise - actions``. The network
        learns that derivative, then sampling integrates the learned field from
        ``t = 1`` back to ``t = 0`` using a negative step size.
        """

        batch_size = actions.shape[0]
        mix = time.view(batch_size, 1, 1)
        x_t = mix * noise + (1.0 - mix) * actions
        target = noise - actions
        return x_t, target

    def compute_loss(
        self,
        batch: dict[str, Tensor],
        mode: Literal["bc", "cfgrl"] = "bc",
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute BC or weighted CFGRL extraction loss for one batch."""

        obs_encoding = self.encode_observations(batch)
        actions = batch[ACTION].float()
        if actions.ndim != 3:
            raise ValueError(f"Expected action chunk [B,H,A], got {tuple(actions.shape)}")

        batch_size = actions.shape[0]
        device = actions.device
        time = self._time_dist.sample((batch_size,)).to(device=device, dtype=actions.dtype)
        noise = torch.randn_like(actions)
        x_t, target = self._build_flow_training_pair(actions, noise, time)

        cond_idx = self._sample_training_condition_indices(batch, mode, batch_size, device)
        pred = self._predict_vector_field(obs_encoding, x_t, time, condition_indices=cond_idx)
        per_sample = (pred - target).pow(2).mean(dim=(1, 2))

        weights = batch.get(self.config.weight_key)
        if weights is None:
            weights = torch.ones(batch_size, device=device, dtype=per_sample.dtype)
        else:
            weights = weights.to(device=device, dtype=per_sample.dtype).view(batch_size)

        loss = (per_sample * weights).mean()
        logs = {
            "policy/loss": loss.detach(),
            "policy/mse": per_sample.mean().detach(),
            "policy/weight_mean": weights.mean().detach(),
        }
        return loss, logs

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        """Dispatch to BC or CFGRL loss depending on the batch payload."""

        mode = batch.get("mode", "cfgrl" if self.config.condition_key in batch else "bc")
        return self.compute_loss(batch, mode=mode)

    @torch.no_grad()
    def sample_actions(
        self,
        obs,
        condition: int | None = None,
        guidance_scale: float | None = None,
        num_samples: int = 1,
        noise: Tensor | None = None,
        num_steps: int | None = None,
    ) -> Tensor:
        """Sample one or more action chunks with optional CFG guidance.

        Sampling starts from Gaussian noise, corresponding to ``t = 1`` in the
        training path definition. We then integrate the learned velocity field
        backwards to ``t = 0`` with Euler updates:

        ``x <- x + dt * v_theta(x, t)``, where ``dt < 0``.

        That negative step is the only sign flip involved; the learned vector
        field itself uses the same ``noise - action`` convention as training.
        """

        obs_encoding = self.encode_observations(obs)
        batch_size = obs_encoding.context.shape[0]
        steps = int(num_steps or self.config.num_denoising_steps)
        repeated_context = obs_encoding.context.repeat_interleave(num_samples, dim=0)
        repeated_tokens = obs_encoding.tokens.repeat_interleave(num_samples, dim=0)
        repeated = ObservationEncoding(context=repeated_context, tokens=repeated_tokens, camera_features={})

        if noise is None:
            noise = torch.randn(
                batch_size * num_samples,
                self.config.chunk_size,
                self.action_dim,
                device=repeated_context.device,
                dtype=repeated_context.dtype,
            )
        else:
            if noise.ndim == 4:
                noise = noise.reshape(batch_size * num_samples, self.config.chunk_size, self.action_dim)
            elif noise.ndim == 3 and num_samples == 1:
                pass
            else:
                raise ValueError(f"Unsupported noise shape {tuple(noise.shape)}")

        x_t = noise
        unconditional_idx = torch.zeros(batch_size * num_samples, device=x_t.device, dtype=torch.long)
        if condition is None:
            conditional_idx = unconditional_idx
        else:
            conditional_idx = torch.full(
                (batch_size * num_samples,),
                1 + int(condition),
                device=x_t.device,
                dtype=torch.long,
            )

        dt = -1.0 / max(steps, 1)
        for step in range(steps):
            t_scalar = 1.0 + step * dt
            time = torch.full((batch_size * num_samples,), t_scalar, device=x_t.device, dtype=x_t.dtype)
            if condition is None:
                velocity = self._predict_vector_field(repeated, x_t, time, condition_indices=unconditional_idx)
            elif guidance_scale is None:
                velocity = self._predict_vector_field(repeated, x_t, time, condition_indices=conditional_idx)
            else:
                v_u = self._predict_vector_field(repeated, x_t, time, condition_indices=unconditional_idx)
                v_c = self._predict_vector_field(repeated, x_t, time, condition_indices=conditional_idx)
                velocity = v_u + guidance_scale * (v_c - v_u)
            x_t = x_t + dt * velocity

        if num_samples == 1:
            return x_t.view(batch_size, self.config.chunk_size, self.action_dim)
        return x_t.view(batch_size, num_samples, self.config.chunk_size, self.action_dim)

    @torch.no_grad()
    def sample_action_chunk(self, obs, **kwargs) -> Tensor:
        """Convenience wrapper returning a single action chunk per observation."""

        actions = self.sample_actions(obs, num_samples=1, **kwargs)
        if actions.ndim == 4:
            return actions[:, 0]
        return actions

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Rollout helper using config-default condition and guidance settings."""

        return self.sample_action_chunk(
            batch,
            condition=self.config.default_rollout_condition,
            guidance_scale=self.config.default_guidance_scale,
            noise=noise,
        )

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Return the next environment action from the cached rollout chunk."""

        batch = dict(batch)
        batch.pop(ACTION, None)
        if self._queues is None:
            self.reset()
        if len(self._queues[ACTION]) == 0:
            chunk = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(chunk[:, : self.config.n_action_steps].transpose(0, 1))
        return self._queues[ACTION].popleft()
