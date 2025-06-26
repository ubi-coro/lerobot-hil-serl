import math
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import MultivariateNormal

from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.policies.ppo.configuration_ppo import PPOConfig


class PPOPolicy(PreTrainedPolicy):

    config_class = PPOConfig
    name = "ppo"

    def __init__(
        self,
        config: PPOConfig | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Determine action dimension
        action_dim = config.output_features["action"].shape[0]

        # Normalization
        self._init_normalization(dataset_stats)
        # Encoders
        self._init_encoders()
        # Actor and Critic
        self._init_actor(action_dim)
        self._init_critic()

    def get_optim_params(self) -> dict:
        return {
            "actor": self.actor.parameters(),
            "critic": self.critic.parameters(),
        }

    def reset(self):
        """Reset the policy"""
        pass

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Sample an action for inference, returning action and log_prob"""
        obs_feat = None
        if self.shared_encoder:
            obs_feat = self.actor.encoder.get_cached_image_features(batch, normalize=True)

        # Compute distribution and sample
        means, log_stds = self.actor(batch, obs_feat, return_dist_params=True)
        stds = torch.exp(log_stds)
        dist = MultivariateNormal(means, torch.diag_embed(stds**2))
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs

    def forward(
        self,
        batch: dict[str, Tensor | dict[str, Tensor]],
        model: Literal["actor", "critic"] = "actor",
    ) -> dict[str, Tensor]:
        """
        Compute PPO losses for actor or critic.

        batch must contain:
          - "state": observations dict
          - "action": Tensor of actions taken
          - "return": Tensor of discounted returns
          - "advantage": Tensor of advantages (for actor)
          - "old_log_prob": Tensor of old log probabilities (for actor)
        """
        observations: dict[str, Tensor] = batch["state"]
        obs_feat: Optional[Tensor] = batch.get("observation_feature")

        # shared encoder cache
        if self.shared_encoder and obs_feat is None:
            obs_feat = self.actor.encoder.get_cached_image_features(observations, normalize=False)

        # Encode observations
        enc = self.actor.encoder(observations, cache=obs_feat, detach=self.actor.encoder_is_shared)

        if model == "critic":
            # Value loss
            values = self.critic(enc)
            returns = batch["return"]
            # MSE between value predictions and empirical returns
            loss_value = F.mse_loss(values, returns)
            return {"loss_critic": loss_value}

        if model == "actor":
            actions: Tensor = batch["action"]
            advantages: Tensor = batch["advantage"]
            old_log_probs: Tensor = batch["old_log_prob"]

            # Compute new log probs
            means, log_stds = self.actor(batch, obs_feat, return_dist_params=True)
            stds = torch.exp(log_stds)
            dist = MultivariateNormal(means, torch.diag_embed(stds**2))
            new_log_probs = dist.log_prob(actions)

            # PPO clipped surrogate objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            clip_eps = self.config.clip_epsilon
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
            loss_clip = -torch.min(surrogate1, surrogate2).mean()

            # Entropy bonus
            entropy = dist.entropy().mean()
            loss_entropy = -self.config.entropy_coeff * entropy

            loss_actor = loss_clip + loss_entropy
            return {"loss_actor": loss_actor, "entropy": entropy.item()}

        raise ValueError(f"Unknown model type: {model}")

    def _init_normalization(self, dataset_stats):
        self.normalize_inputs = nn.Identity()
        self.normalize_targets = nn.Identity()
        self.unnormalize_outputs = nn.Identity()
        if self.config.dataset_stats:
            params = _convert_normalization_params_to_tensor(self.config.dataset_stats)
            self.normalize_inputs = Normalize(
                self.config.input_features, self.config.normalization_mapping, params
            )
            stats = dataset_stats or params
            self.normalize_targets = Normalize(
                self.config.output_features, self.config.normalization_mapping, stats
            )
            self.unnormalize_outputs = Unnormalize(
                self.config.output_features, self.config.normalization_mapping, stats
            )

    def _init_encoders(self):
        self.shared_encoder = self.config.shared_encoder
        # reuse SACObservationEncoder
        from lerobot.common.policies.sac.modeling_sac import SACObservationEncoder
        self.encoder = SACObservationEncoder(self.config, self.normalize_inputs)
        if self.shared_encoder:
            self.actor_encoder = self.encoder
            self.critic_encoder = self.encoder
        else:
            self.actor_encoder = SACObservationEncoder(self.config, self.normalize_inputs)
            self.critic_encoder = SACObservationEncoder(self.config, self.normalize_inputs)

    def _init_actor(self, action_dim: int):
        """Initialize actor network to output distribution parameters"""
        # feature extractor + MLP
        hidden_dims = self.config.actor_hidden_dims
        self.actor = nn.Sequential(
            nn.Linear(self.actor_encoder.output_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.Tanh(),
            *sum(([nn.Linear(h0, h1), nn.LayerNorm(h1), nn.Tanh()] for h0, h1 in zip(hidden_dims, hidden_dims[1:])), []),
        )
        # Mean and log_std heads
        last_dim = hidden_dims[-1]
        self.actor_mean = nn.Linear(last_dim, action_dim)
        self.actor_log_std = nn.Linear(last_dim, action_dim)
        # init
        init_val = self.config.init_std
        if init_val is not None:
            nn.init.constant_(self.actor_log_std.bias, math.log(init_val))

    def _init_critic(self):
        """Initialize value function network"""
        hidden_dims = self.config.critic_hidden_dims
        self.critic = nn.Sequential(
            nn.Linear(self.critic_encoder.output_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.Tanh(),
            *sum(([nn.Linear(h0, h1), nn.LayerNorm(h1), nn.Tanh()] for h0, h1 in zip(hidden_dims, hidden_dims[1:])), []),
            nn.Linear(hidden_dims[-1], 1),
        )

    @property
    def device(self):
        return get_device_from_parameters(self)

    # Utility actor call to get distribution parameters
    def actor(self, observations: dict[str, Tensor], obs_feat: Optional[Tensor], return_dist_params: bool=False): # type: ignore
        # Encode obs
        enc = self.actor_encoder(observations, cache=obs_feat, detach=self.shared_encoder)
        # Forward through MLP
        out = enc
        for layer in self.actor:
            out = layer(out)
        means = self.actor_mean(out)
        log_stds = self.actor_log_std(out)
        log_stds = torch.clamp(log_stds, self.config.log_std_min, self.config.log_std_max)
        if return_dist_params:
            return means, log_stds
        # sampling (not used in forward loss)
        stds = torch.exp(log_stds)
        dist = MultivariateNormal(means, torch.diag_embed(stds**2))
        actions = dist.rsample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs, means


def _convert_normalization_params_to_tensor(normalization_params: dict) -> dict:
    # identical to SAC's utility
    from lerobot.common.policies.sac.modeling_sac import _convert_normalization_params_to_tensor as sac_convert
    return sac_convert(normalization_params)
