#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may not use this file except in compliance with the License.
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from collections.abc import Callable
from dataclasses import asdict
from typing import Literal

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torch.distributions import MultivariateNormal, TanhTransform, Transform, TransformedDistribution, Distribution

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.sac.configuration_sac import SACConfig, is_image_feature
from lerobot.policies.utils import get_device_from_parameters
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE

DISCRETE_DIMENSION_INDEX = -1  # Gripper is always the last dimension


class SACVLMPolicy(
    PreTrainedPolicy,
):
    config_class = SACConfig
    name = "sac_vlm"

    def __init__(
        self,
        config: SACConfig | None = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Determine action dimension and initialize all components
        continuous_action_dim = config.output_features[ACTION].shape[0]
        self._init_encoders()
        self._init_critics(continuous_action_dim)
        self._init_actor(continuous_action_dim)
        self._init_temperature()

        # Global optimization step that lives with the policy,
        # saved in the state_dict and shipped to the actor.
        # Use register_buffer so it's not part of optimizer params.
        self.register_buffer("opt_step", torch.zeros(1, dtype=torch.long))

    # --- step management (saved in state_dict) ---
    def set_opt_step(self, step: int | torch.Tensor) -> None:
        if isinstance(step, int):
            self.opt_step.fill_(step)
        else:
            # ensure correct dtype/device/shape
            self.opt_step.copy_(step.to(self.opt_step.device, dtype=self.opt_step.dtype).view(1))

    def inc_opt_step(self, n: int = 1) -> None:
        self.opt_step += n

    def get_opt_step(self) -> int:
        return int(self.opt_step.item())

    def get_optim_params(self) -> dict:
        optim_params = {
            "actor": [
                p
                for n, p in self.actor.named_parameters()
                if not n.startswith("encoder") or not self.shared_encoder
            ],
            "critic": self.critic_ensemble.parameters(),
            "temperature": self.log_alpha,
        }
        if self.config.num_discrete_actions is not None:
            optim_params["discrete_critic"] = self.discrete_critic.parameters()
        if self.config.noise_config.enable:
            optim_params["noise"] = [
                p
                for n, p in self.noise_net.named_parameters()
                if not n.startswith("encoder") or not self.shared_encoder
            ]
        return optim_params

    def reset(self):
        """Reset the policy"""
        pass

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        raise NotImplementedError("SACPolicy does not support action chunking. It returns single actions!")

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], deterministic: bool = True, use_dgn: bool = True, step: int | None = None) -> tuple[Tensor, dict]:
        """Select action for inference/evaluation"""
        if self.config.use_bc_dagger or not self.config.noise_config.enable:
            use_dgn = False

        observation_features = None
        if self.shared_encoder:
            # Cache and normalize image features
            # MODIFIED: This will now return an empty dict, which is correct.
            observation_features = self.actor.encoder.get_cached_image_features(batch)

        dist = self.actor(batch, observation_features)

        if deterministic or use_dgn:
            action = dist.mode()
            # action = dist.rsample()  # would be in line with hil-serl argmax
        else:
            action = dist.rsample()

        inference_infos = {}
        q_value, _ = self.critic_forward(
            observations=batch,
            actions=action,
            use_target=False,
            observation_features=observation_features,
        ).min(dim=0)
        inference_infos["Q-Value"] = q_value.item()

        if self.config.num_discrete_actions is not None:
            discrete_action_value = self.discrete_critic(batch, observation_features)
            discrete_action = torch.argmax(discrete_action_value, dim=-1, keepdim=True)
            action = torch.cat([action, discrete_action], dim=-1)

        if use_dgn and not deterministic:
            action, _, noise_scale = self.add_structured_noise(action, batch, observation_features, step=step)
            inference_infos["Noise Scale"] = noise_scale

        return action, inference_infos

    def critic_forward(
        self,
        observations: dict[str, Tensor],
        actions: Tensor,
        use_target: bool = False,
        observation_features: Tensor | None = None,
    ) -> Tensor:
        """Forward pass through a critic network ensemble

        Args:
            observations: Dictionary of observations
            actions: Action tensor
            use_target: If True, use target critics, otherwise use ensemble critics

        Returns:
            Tensor of Q-values from all critics
        """

        critics = self.critic_target if use_target else self.critic_ensemble
        q_values = critics(observations, actions, observation_features)
        return q_values

    def discrete_critic_forward(
        self, observations, use_target=False, observation_features=None
    ) -> torch.Tensor:
        """Forward pass through a discrete critic network

        Args:
            observations: Dictionary of observations
            use_target: If True, use target critics, otherwise use ensemble critics
            observation_features: Optional pre-computed observation features to avoid recomputing encoder output

        Returns:
            Tensor of Q-values from the discrete critic network
        """
        discrete_critic = self.discrete_critic_target if use_target else self.discrete_critic
        q_values = discrete_critic(observations, observation_features)
        return q_values

    def forward(
        self,
        batch: dict[str, Tensor | dict[str, Tensor]],
        model: Literal["actor", "critic", "temperature", "discrete_critic"] = "critic",
    ) -> dict[str, Tensor | dict]:
        """Compute the loss for the given model

        Args:
            batch: Dictionary containing:
                - action: Action tensor
                - reward: Reward tensor
                - state: Observations tensor dict (now expecting VLM embedding)
                - next_state: Next observations tensor dict (now expecting VLM embedding)
                - done: Done mask tensor
                - observation_feature: Optional pre-computed observation features (unused)
                - next_observation_feature: Optional pre-computed next observation features (unused)
            model: Which model to compute the loss for ("actor", "critic", "discrete_critic", or "temperature")

        Returns:
            The computed loss tensor
        """
        # Extract common components from batch
        actions: Tensor = batch[ACTION]
        observations: dict[str, Tensor] = batch["state"]
        # MODIFIED: observation_features is no longer used, as the VLM embedding
        # is the main input in `observations`
        observation_features: Tensor = None # batch.get("observation_feature")

        if model == "critic":
            # Extract critic-specific components
            rewards: Tensor = batch["reward"]
            next_observations: dict[str, Tensor] = batch["next_state"]
            done: Tensor = batch["done"]
            # MODIFIED: next_observation_features is no longer used
            next_observation_features: Tensor = None # batch.get("next_observation_feature")

            loss_critic, training_infos = self.compute_loss_critic(
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                done=done,
                observation_features=observation_features,
                next_observation_features=next_observation_features,
            )

            return {"loss_critic": loss_critic, "training_infos": training_infos}

        if model == "discrete_critic" and self.config.num_discrete_actions is not None:
            # Extract critic-specific components
            rewards: Tensor = batch["reward"]
            next_observations: dict[str, Tensor] = batch["next_state"]
            done: Tensor = batch["done"]
            # MODIFIED: next_observation_features is no longer used
            next_observation_features: Tensor = None # batch.get("next_observation_feature")
            complementary_info = batch.get("complementary_info")
            loss_discrete_critic = self.compute_loss_discrete_critic(
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                done=done,
                observation_features=observation_features,
                next_observation_features=next_observation_features,
                complementary_info=complementary_info,
            )
            return {"loss_discrete_critic": loss_discrete_critic}

        if model == "bc":
            bc_loss, training_infos = self.compute_loss_bc(
                observations=observations,
                actions=actions,
                observation_features=observation_features
            )
            return {"loss_bc": bc_loss, "training_infos": training_infos}

        if model == "actor":
            return {
                "loss_actor": self.compute_loss_actor(
                    observations=observations,
                    observation_features=observation_features,
                )
            }

        if model == "temperature":
            return {
                "loss_temperature": self.compute_loss_temperature(
                    observations=observations,
                    observation_features=observation_features,
                )
            }

        if model == "noise":
            noise_loss, dgn_improvement_ratio = self.compute_loss_noise(
                actions=actions,
                observations=observations,
                observation_features=observation_features,
            )
            return {"loss_noise": noise_loss, "dgn_improvement_ratio": dgn_improvement_ratio}

        raise ValueError(f"Unknown model type: {model}")

    def update_target_networks(self):
        """Update target networks with exponential moving average"""
        for target_param, param in zip(
            self.critic_target.parameters(),
            self.critic_ensemble.parameters(),
            strict=True,
        ):
            target_param.data.copy_(
                param.data * self.config.critic_target_update_weight
                + target_param.data * (1.0 - self.config.critic_target_update_weight)
            )
        if self.config.num_discrete_actions is not None:
            for target_param, param in zip(
                self.discrete_critic_target.parameters(),
                self.discrete_critic.parameters(),
                strict=True,
            ):
                target_param.data.copy_(
                    param.data * self.config.critic_target_update_weight
                    + target_param.data * (1.0 - self.config.critic_target_update_weight)
                )

    def update_temperature(self):
        self.temperature = self.log_alpha.exp().item()

    def compute_loss_critic(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        done,
        observation_features: Tensor | None = None,
        next_observation_features: Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Tensor]]:
        with torch.no_grad():
            action_dist = self.actor(next_observations, next_observation_features)
            next_action_preds = action_dist.rsample()

            # 2- compute q targets
            q_targets = self.critic_forward(
                observations=next_observations,
                actions=next_action_preds,
                use_target=True,
                observation_features=next_observation_features,
            )

            # subsample critics to prevent overfitting if use high UTD (update to date)
            # TODO: Get indices before forward pass to avoid unnecessary computation
            if self.config.num_subsample_critics is not None:
                indices = torch.randperm(self.config.num_critics)
                indices = indices[: self.config.num_subsample_critics]
                q_targets = q_targets[indices]

            # critics subsample size
            min_q, _ = q_targets.min(dim=0)  # Get values from min operation
            if self.config.use_backup_entropy:
                min_q = min_q - (self.temperature * action_dist.log_prob(action_dist.rsample()))

            td_target = rewards + (1 - done) * self.config.discount * min_q

        # 3- compute predicted qs
        if self.config.num_discrete_actions is not None:
            # NOTE: We only want to keep the continuous action part
            # In the buffer we have the full action space (continuous + discrete)
            # We need to split them before concatenating them in the critic forward
            actions: Tensor = actions[:, :DISCRETE_DIMENSION_INDEX]
        q_preds = self.critic_forward(
            observations=observations,
            actions=actions,
            use_target=False,
            observation_features=observation_features,
        )

        # 4- Calculate loss
        # Compute state-action value loss (TD loss) for all of the Q functions in the ensemble.
        td_target_duplicate = einops.repeat(td_target, "b -> e b", e=q_preds.shape[0])
        # You compute the mean loss of the batch for each critic and then to compute the final loss you sum them up
        critics_loss = (
            F.mse_loss(
                input=q_preds,
                target=td_target_duplicate,
                reduction="none",
            ).mean(dim=1)
        ).sum()

        # log key metrics to measure overestimation, stability and uncertainty
        training_infos = {
            "mean_q_ensemble_disagreement": q_preds.std(dim=0)[0].mean().item(),
            "mean_q_ensemble_var": q_preds.std(dim=1).mean().item(),
            "max_q_ensemble_range": (q_preds.max(dim=0)[0] - q_preds.min(dim=0)[0]).mean().item(),
            "mean_q": q_preds.mean().item(),
            "max_q": q_preds.max().item(),
            "min_q": q_preds.min().item()
        }

        return critics_loss, training_infos

    def compute_loss_discrete_critic(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        done,
        observation_features=None,
        next_observation_features=None,
        complementary_info=None,
    ):
        # NOTE: We only want to keep the discrete action part
        # In the buffer we have the full action space (continuous + discrete)
        # We need to split them before concatenating them in the critic forward
        actions_discrete: Tensor = actions[:, DISCRETE_DIMENSION_INDEX:].clone()
        actions_discrete = torch.round(actions_discrete)
        actions_discrete = actions_discrete.long()

        discrete_penalties: Tensor | None = None
        if complementary_info is not None:
            discrete_penalties: Tensor | None = complementary_info.get("discrete_penalty")

        with torch.no_grad():
            # For DQN, select actions using online network, evaluate with target network
            next_discrete_qs = self.discrete_critic_forward(
                next_observations, use_target=False, observation_features=next_observation_features
            )
            best_next_discrete_action = torch.argmax(next_discrete_qs, dim=-1, keepdim=True)

            # Get target Q-values from target network
            target_next_discrete_qs = self.discrete_critic_forward(
                observations=next_observations,
                use_target=True,
                observation_features=next_observation_features,
            )

            # Use gather to select Q-values for best actions
            target_next_discrete_q = torch.gather(
                target_next_discrete_qs, dim=1, index=best_next_discrete_action
            ).squeeze(-1)

            # Compute target Q-value with Bellman equation
            rewards_discrete = rewards
            if discrete_penalties is not None:
                rewards_discrete = rewards + discrete_penalties
            target_discrete_q = rewards_discrete + (1 - done) * self.config.discount * target_next_discrete_q

        # Get predicted Q-values for current observations
        predicted_discrete_qs = self.discrete_critic_forward(
            observations=observations, use_target=False, observation_features=observation_features
        )

        # Use gather to select Q-values for taken actions
        predicted_discrete_q = torch.gather(predicted_discrete_qs, dim=1, index=actions_discrete).squeeze(-1)

        # Compute MSE loss between predicted and target Q-values
        discrete_critic_loss = F.mse_loss(input=predicted_discrete_q, target=target_discrete_q)
        return discrete_critic_loss

    def compute_loss_temperature(self, observations, observation_features: Tensor | None = None) -> Tensor:
        """Compute the temperature loss"""
        # calculate temperature loss
        with torch.no_grad():
            dist = self.actor(observations, observation_features)
        temperature_loss = (-self.log_alpha.exp() * (dist.log_prob(dist.rsample()) + self.target_entropy)).mean()
        return temperature_loss

    def compute_loss_bc(
        self,
        observations,
        actions,
        observation_features: Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Tensor]]:
        dist = self.actor(observations, observation_features)
        policy_actions = dist.mode()

        if self.config.policy_kwargs.use_tanh_squash:
            actions = torch.clip(actions, -1+1e-6, 1-1e-6)

        log_probs = dist.log_prob(actions)
        mse = ((actions - policy_actions) ** 2).sum(-1).mean()
        bc_loss = -log_probs.mean()

        return bc_loss, {"mse": mse}

    def compute_loss_actor(
        self,
        observations,
        observation_features: Tensor | None = None,
    ) -> Tensor:
        dist = self.actor(observations, observation_features)
        policy_actions = dist.rsample()

        q_preds = self.critic_forward(
            observations=observations,
            actions=policy_actions,
            use_target=False,
            observation_features=observation_features,
        )
        min_q_preds = q_preds.min(dim=0)[0]

        actor_loss = ((self.temperature * dist.log_prob(policy_actions)) - min_q_preds).mean()
        return actor_loss

    def compute_loss_noise(
        self,
        actions,
        observations,
        observation_features: Tensor | None = None,
    ) -> tuple[Tensor, float]:
        with torch.no_grad():
            dist = self.actor(observations, observation_features)
            mean_actions = dist.rsample()

        noised_actions, noise_dist, _ = self.add_structured_noise(mean_actions, observations, observation_features, step=0)

        # noise_dist is in normalized action-space
        noised_actions = self.normalize_targets({"action": noised_actions})["action"]
        actions = self.normalize_targets({"action": actions})["action"]
        mean_actions = self.normalize_targets({"action": mean_actions})["action"]

        # score residuals
        residual_actions = actions - mean_actions
        noise_loss = -noise_dist.log_prob(residual_actions).mean()

        # compute improvement
        dist_to_mean = torch.norm(actions - mean_actions, dim=1)
        dist_to_sampled = torch.norm(actions - noised_actions, dim=1)
        dgn_improvement_ratio = (dist_to_sampled < dist_to_mean).float().mean().item()

        return noise_loss, dgn_improvement_ratio

    def add_structured_noise(self, mean_actions: Tensor, observations: dict[str, Tensor] , observation_features: Tensor | None = None, step: int | None = 0):
        batch_size = mean_actions.shape[0]
        action_dim = mean_actions.shape[1]

        noise_params = self.noise_net.mean_layer(
            self.noise_net.network(
                self.noise_net.encoder(
                    observations,
                    cache=observation_features,
                    detach=self.noise_net.encoder_is_shared
                )
            )
        )

        if self.config.noise_config.predict_residual:
            cov_cholemsky = noise_params[..., action_dim:]
            residual_means = noise_params[..., :action_dim]
        else:
            cov_cholemsky = noise_params
            residual_means = torch.zeros_like(mean_actions)

        # Create lower triangular matrix
        L = torch.zeros((batch_size, action_dim, action_dim), device=mean_actions.device)

        tril_indices = torch.tril_indices(row=action_dim, col=action_dim, offset=0)
        L[:, tril_indices[0], tril_indices[1]] = cov_cholemsky

        # Optionally apply softplus to diagonal elements to ensure positive definiteness
        diagonal_indices = torch.arange(action_dim)
        L[:, diagonal_indices, diagonal_indices] = torch.nn.functional.softplus(
            L[:, diagonal_indices, diagonal_indices]
        ) + 1e-6

        # Compute covariance matrix explicitly via Cholesky factorization
        cov = torch.bmm(L, L.transpose(1, 2))

        # sample from multivariate normal
        noise_dist = torch.distributions.MultivariateNormal(residual_means, covariance_matrix=cov)
        noise = noise_dist.rsample()

        # downscale noise
        effective_step = self.get_opt_step() if step is None else step
        eps = self.config.noise_config.initial_eps * np.exp(-effective_step /  self.config.noise_config.tau)
        noise = eps * noise

        # add actions and clamp accordingly
        action_normalized = self.normalize_targets({"action": mean_actions})["action"]
        action_normalized = action_normalized + noise
        action_normalized = torch.clamp(action_normalized, -1, 1)

        actions_final = self.unnormalize_outputs({"action": action_normalized})["action"]
        return actions_final, noise_dist, eps

    def _init_encoders(self):
        """Initialize shared or separate encoders for actor and critic."""
        self.shared_encoder = self.config.shared_encoder
        self.encoder_critic = SACObservationEncoder(self.config)
        self.encoder_actor = (
            self.encoder_critic if self.shared_encoder else SACObservationEncoder(self.config)
        )

        if self.config.noise_config.enable:
            self.encoder_noise = self.encoder_critic if self.shared_encoder else SACObservationEncoder(self.config)

    def _init_critics(self, continuous_action_dim):
        """Build critic ensemble, targets, and optional discrete critic."""
        heads = [
            CriticHead(
                input_dim=self.encoder_critic.output_dim + continuous_action_dim,
                **asdict(self.config.critic_network_kwargs),
            )
            for _ in range(self.config.num_critics)
        ]
        self.critic_ensemble = CriticEnsemble(encoder=self.encoder_critic, ensemble=heads)
        target_heads = [
            CriticHead(
                input_dim=self.encoder_critic.output_dim + continuous_action_dim,
                **asdict(self.config.critic_network_kwargs),
            )
            for _ in range(self.config.num_critics)
        ]
        self.critic_target = CriticEnsemble(encoder=self.encoder_critic, ensemble=target_heads)
        self.critic_target.load_state_dict(self.critic_ensemble.state_dict())

        if self.config.use_torch_compile:
            self.critic_ensemble = torch.compile(self.critic_ensemble)
            self.critic_target = torch.compile(self.critic_target)

        if self.config.num_discrete_actions is not None:
            self._init_discrete_critics()

    def _init_discrete_critics(self):
        """Build discrete discrete critic ensemble and target networks."""
        self.discrete_critic = DiscreteCritic(
            encoder=self.encoder_critic,
            input_dim=self.encoder_critic.output_dim,
            output_dim=self.config.num_discrete_actions,
            **asdict(self.config.discrete_critic_network_kwargs),
        )
        self.discrete_critic_target = DiscreteCritic(
            encoder=self.encoder_critic,
            input_dim=self.encoder_critic.output_dim,
            output_dim=self.config.num_discrete_actions,
            **asdict(self.config.discrete_critic_network_kwargs),
        )

        # TODO: (maractingi, azouitine) Compile the discrete critic
        self.discrete_critic_target.load_state_dict(self.discrete_critic.state_dict())

    def _init_actor(self, continuous_action_dim):
        """Initialize policy actor network and default target entropy."""
        # NOTE: The actor select only the continuous action part
        self.actor = Policy(
            encoder=self.encoder_actor,
            network=MLP(input_dim=self.encoder_actor.output_dim, **asdict(self.config.actor_network_kwargs)),
            action_dim=continuous_action_dim,
            encoder_is_shared=self.shared_encoder,
            **asdict(self.config.policy_kwargs),
        )

        self.target_entropy = self.config.target_entropy
        if self.target_entropy is None:
            dim = continuous_action_dim + (1 if self.config.num_discrete_actions is not None else 0)
            self.target_entropy = -dim / 2

    def _init_temperature(self):
        """Set up temperature parameter and initial log_alpha."""
        temp_init = self.config.temperature_init
        self.log_alpha = nn.Parameter(torch.tensor([math.log(temp_init)]))
        self.temperature = self.log_alpha.exp().item()

    def _init_noise(self):
        if not self.config.noise_config.enable:
            return

        action_dim = self.config.output_features["action"].shape[0]
        if self.config.num_discrete_actions is not None:
            action_dim += 1
        output_dim = action_dim * (action_dim + 1) // 2  # num of elements in lower triangular matrix

        if self.config.noise_config.predict_residual:
            output_dim += action_dim


        self.noise_net = Policy(
            encoder=self.encoder_noise,
            network=MLP(input_dim=self.encoder_actor.output_dim, **asdict(self.config.actor_network_kwargs)),
            action_dim=output_dim,
            encoder_is_shared=self.shared_encoder,
            fixed_std=torch.tensor([0.0]),
            **asdict(self.config.policy_kwargs),
        )


class SACObservationEncoder(nn.Module):
    """
    MODIFIED: Encode VLM embedding (from `OBS_STATE`) and ignore images/env_state.
    This module now acts as a simple MLP projector for the VLM embedding.
    """

    def __init__(self, config: SACConfig) -> None:
        super().__init__()
        self.config = config
        self._init_image_layers()
        self._init_state_layers()
        self._compute_output_dim()

    def _init_image_layers(self) -> None:
        # <-- MODIFIED: Force no images
        self.image_keys = []
        self.has_images = False
        # <-- END MODIFIED
        if not self.has_images:
            return

        # ... All original image encoder logic is now unreachable ...

    def _init_state_layers(self) -> None:
        # <-- MODIFIED: Force no env state, and force OBS_STATE (VLM embedding)
        self.has_env = False
        self.has_state = True
        # <-- END MODIFIED

        if self.has_env:
            # This block is now unreachable
            dim = self.config.input_features[OBS_ENV_STATE].shape[0]
            self.env_encoder = nn.Sequential(
                nn.Linear(dim, self.config.latent_dim),
                nn.LayerNorm(self.config.latent_dim),
                nn.Tanh(),
            )
        if self.has_state:
            # <-- MODIFIED: Add check to ensure VLM embedding is in config
            if OBS_STATE not in self.config.input_features:
                raise ValueError(
                    f"VLM embedding mode requires '{OBS_STATE}' in input_features."
                    " The rollout script should save the embedding under this key."
                )
            # <-- END MODIFIED

            # This now correctly uses the VLM embedding dim (e.g., 4096)
            dim = self.config.input_features[OBS_STATE].shape[0]
            # This is our MLP projector: (VLM_DIM) -> (latent_dim)
            self.state_encoder = nn.Sequential(
                nn.Linear(dim, self.config.latent_dim),
                nn.LayerNorm(self.config.latent_dim),
                nn.Tanh(),
            )

    def _compute_output_dim(self) -> None:
        out = 0
        if self.has_images: # <-- False
            out += len(self.image_keys) * self.config.latent_dim
        if self.has_env: # <-- False
            out += self.config.latent_dim
        if self.has_state: # <-- True
            out += self.config.latent_dim # This is the only part that runs
        self._out_dim = out # Output dim is correctly `latent_dim`

    def forward(
        self, obs: dict[str, Tensor], cache: dict[str, Tensor] | None = None, detach: bool = False
    ) -> Tensor:
        parts = []
        if self.has_images:
            # <-- This block is now unreachable ---
            if cache is None:
                cache = self.get_cached_image_features(obs)
            parts.append(self._encode_images(cache, detach))
            # ------------------------------------
        if self.has_env:
            # <-- This block is now unreachable ---
            parts.append(self.env_encoder(obs[OBS_ENV_STATE]))
            # ------------------------------------
        if self.has_state:
            # <-- This is the only part that runs ---
            # obs[OBS_STATE] is the VLM embedding (B, VLA_EMBED_DIM)
            # self.state_encoder is the MLP projector (VLA_EMBED_DIM -> latent_dim)
            parts.append(self.state_encoder(obs[OBS_STATE]))
            # ----------------------------------------
        if parts:
            # Returns the (B, latent_dim) projected embedding
            return torch.cat(parts, dim=-1)

        raise ValueError(
            "No parts to concatenate, you should have at least one image or environment state or state"
        )

    def get_cached_image_features(self, obs: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        MODIFIED: This now correctly does nothing and returns an empty dict
        because self.image_keys is [].
        """
        if not self.image_keys:
            return {}

        batched = torch.cat([obs[k] for k in self.image_keys], dim=0)
        out = self.image_encoder(batched)
        chunks = torch.chunk(out, len(self.image_keys), dim=0)
        return dict(zip(self.image_keys, chunks, strict=False))

    def _encode_images(self, cache: dict[str, Tensor], detach: bool) -> Tensor:
        """
        MODIFIED: This function is now unreachable because self.has_images is False.
        """
        feats = []
        for k, feat in cache.items():
            safe_key = k.replace(".", "_")
            x = self.spatial_embeddings[safe_key](feat)
            x = self.post_encoders[safe_key](x)
            if detach:
                x = x.detach()
            feats.append(x)
        return torch.cat(feats, dim=-1)

    @property
    def output_dim(self) -> int:
        return self._out_dim


class MLP(nn.Module):
    """Multi-layer perceptron builder.

    Dynamically constructs a sequence of layers based on `hidden_dims`:
      1) Linear (in_dim -> out_dim)
      2) Optional Dropout if `dropout_rate` > 0 and (not final layer or `activate_final`)
      3) LayerNorm on the output features
      4) Activation (standard for intermediate layers, `final_activation` for last layer if `activate_final`)

    Arguments:
        input_dim (int): Size of input feature dimension.
        hidden_dims (list[int]): Sizes for each hidden layer.
        activations (Callable or str): Activation to apply between layers.
        activate_final (bool): Whether to apply activation at the final layer.
        dropout_rate (Optional[float]): Dropout probability applied before normalization and activation.
        final_activation (Optional[Callable or str]): Activation for the final layer when `activate_final` is True.

    For each layer, `in_dim` is updated to the previous `out_dim`. All constructed modules are
    stored in `self.net` as an `nn.Sequential` container.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activations: Callable[[torch.Tensor], torch.Tensor] | str = nn.SiLU(),
        activate_final: bool = False,
        dropout_rate: float | None = None,
        final_activation: Callable[[torch.Tensor], torch.Tensor] | str | None = None,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        total = len(hidden_dims)

        for idx, out_dim in enumerate(hidden_dims):
            # 1) linear transform
            layers.append(nn.Linear(in_dim, out_dim))

            is_last = idx == total - 1
            # 2-4) optionally add dropout, normalization, and activation
            if not is_last or activate_final:
                if dropout_rate and dropout_rate > 0:
                    layers.append(nn.Dropout(p=dropout_rate))
                layers.append(nn.LayerNorm(out_dim))
                act_cls = final_activation if is_last and final_activation else activations
                act = act_cls if isinstance(act_cls, nn.Module) else getattr(nn, act_cls)()
                layers.append(act)

            in_dim = out_dim

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CriticHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activations: Callable[[torch.Tensor], torch.Tensor] | str = nn.SiLU(),
        activate_final: bool = False,
        dropout_rate: float | None = None,
        init_final: float | None = None,
        final_activation: Callable[[torch.Tensor], torch.Tensor] | str | None = None,
    ):
        super().__init__()
        self.net = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activations=activations,
            activate_final=activate_final,
            dropout_rate=dropout_rate,
            final_activation=final_activation,
        )
        self.output_layer = nn.Linear(in_features=hidden_dims[-1], out_features=1)
        if init_final is not None:
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            orthogonal_init()(self.output_layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.net(x))


class CriticEnsemble(nn.Module):
    """
    CriticEnsemble wraps multiple CriticHead modules into an ensemble.

    Args:
        encoder (SACObservationEncoder): encoder for observations.
        ensemble (List[CriticHead]): list of critic heads.
        init_final (float | None): optional initializer scale for final layers.

    Forward returns a tensor of shape (num_critics, batch_size) containing Q-values.
    """

    def __init__(
        self,
        encoder: SACObservationEncoder,
        ensemble: list[CriticHead],
        init_final: float | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.init_final = init_final
        self.critics = nn.ModuleList(ensemble)

    def forward(
        self,
        observations: dict[str, torch.Tensor],
        actions: torch.Tensor,
        observation_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = get_device_from_parameters(self)
        # Move each tensor in observations to device
        observations = {k: v.to(device) for k, v in observations.items() if isinstance(v, torch.Tensor)}

        # MODIFIED: `cache` is passed as None. `obs_enc` is now the
        # projected VLM embedding (B, latent_dim)
        obs_enc = self.encoder(observations, cache=observation_features)
        actions = actions.to(device)

        inputs = torch.cat([obs_enc, actions], dim=-1)

        # Loop through critics and collect outputs
        q_values = []
        for critic in self.critics:
            q_values.append(critic(inputs))

        # Stack outputs to match expected shape [num_critics, batch_size]
        q_values = torch.stack([q.squeeze(-1) for q in q_values], dim=0)
        return q_values


class DiscreteCritic(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int = 3,
        activations: Callable[[torch.Tensor], torch.Tensor] | str = nn.SiLU(),
        activate_final: bool = False,
        dropout_rate: float | None = None,
        init_final: float | None = None,
        final_activation: Callable[[torch.Tensor], torch.Tensor] | str | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.output_dim = output_dim

        self.net = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activations=activations,
            activate_final=activate_final,
            dropout_rate=dropout_rate,
            final_activation=final_activation,
        )

        self.output_layer = nn.Linear(in_features=hidden_dims[-1], out_features=self.output_dim)
        if init_final is not None:
            nn.init.uniform_(self.output_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.output_layer.bias, -init_final, init_final)
        else:
            orthogonal_init()(self.output_layer.weight)

    def forward(
        self, observations: torch.Tensor, observation_features: torch.Tensor | None = None
    ) -> torch.Tensor:
        device = get_device_from_parameters(self)
        observations = {k: v.to(device) for k, v in observations.items()}
        # MODIFIED: `cache` is passed as None. `obs_enc` is now the
        # projected VLM embedding (B, latent_dim)
        obs_enc = self.encoder(observations, cache=observation_features)
        return self.output_layer(self.net(obs_enc))


class Policy(nn.Module):
    def __init__(
        self,
        encoder: SACObservationEncoder,
        network: nn.Module,
        action_dim: int,
        std_min: float = -5,
        std_max: float = 2,
        fixed_std: torch.Tensor | None = None,
        init_final: float | None = None,
        use_tanh_squash: bool = False,
        encoder_is_shared: bool = False,
    ):
        super().__init__()
        self.encoder: SACObservationEncoder = encoder
        self.network = network
        self.action_dim = action_dim
        self.std_min = std_min
        self.std_max = std_max
        self.fixed_std = fixed_std
        self.use_tanh_squash = use_tanh_squash
        self.encoder_is_shared = encoder_is_shared

        # Find the last Linear layer's output dimension
        for layer in reversed(network.net):
            if isinstance(layer, nn.Linear):
                out_features = layer.out_features
                break
        # Mean layer
        self.mean_layer = nn.Linear(out_features, action_dim)
        if init_final is not None:
            nn.init.uniform_(self.mean_layer.weight, -init_final, init_final)
            nn.init.uniform_(self.mean_layer.bias, -init_final, init_final)
        else:
            orthogonal_init()(self.mean_layer.weight)

        # Standard deviation layer or parameter
        if fixed_std is None:
            self.std_layer = nn.Linear(out_features, action_dim)
            if init_final is not None:
                nn.init.uniform_(self.std_layer.weight, -init_final, init_final)
                nn.init.uniform_(self.std_layer.bias, -init_final, init_final)
            else:
                orthogonal_init()(self.std_layer.weight)

    def forward(
        self,
        observations: torch.Tensor,
        observation_features: torch.Tensor | None = None,
    ) -> Distribution:
        # We detach the encoder if it is shared to avoid backprop through it
        # This is important to avoid the encoder to be updated through the policy

        # MODIFIED: `cache` is passed as None. `obs_enc` is now the
        # projected VLM embedding (B, latent_dim)
        obs_enc = self.encoder(observations, cache=observation_features, detach=self.encoder_is_shared)

        # Get network outputs
        outputs = self.network(obs_enc)
        means = self.mean_layer(outputs)

        # Compute standard deviations
        if self.fixed_std is None:
            log_std = self.std_layer(outputs)
        else:
            log_std = self.fixed_std.expand_as(means).to(means.device)

        std = torch.exp(log_std)  # Match JAX "exp"
        std = torch.clamp(std, self.std_min, self.std_max)  # Match JAX default clip

        # Build transformed distribution
        dist = TanhMultivariateNormalDiag(loc=means, scale_diag=std)

        return dist

    def get_features(self, observations: torch.Tensor) -> torch.Tensor:
        """Get encoded features from observations"""
        device = get_device_from_parameters(self)
        observations = observations.to(device)
        if self.encoder is not None:
            with torch.inference_mode():
                # MODIFIED: `cache` is passed as None.
                return self.encoder(observations, cache=None)
        return observations


class DefaultImageEncoder(nn.Module):
    def __init__(self, config: SACConfig):
        super().__init__()
        # MODIFIED: This class is no longer used, but we'll
        # leave it to avoid breaking other imports.
        pass

    def forward(self, x):
        # MODIFIED: This class is no longer used.
        raise NotImplementedError("DefaultImageEncoder is disabled in VLM-embedding mode")


def freeze_image_encoder(image_encoder: nn.Module):
    """Freeze all parameters in the encoder"""
    for param in image_encoder.parameters():
        param.requires_grad = False


class PretrainedImageEncoder(nn.Module):
    def __init__(self, config: SACConfig):
        super().__init__()
        # MODIFIED: This class is no longer used.
        pass

    def _load_pretrained_vision_encoder(self, config: SACConfig):
        """Set up CNN encoder"""
        # MODIFIED: This class is no longer used.
        pass

    def forward(self, x):
        # MODIFIED: This class is no longer used.
        raise NotImplementedError("PretrainedImageEncoder is disabled in VLM-embedding mode")


def orthogonal_init():
    return lambda x: torch.nn.init.orthogonal_(x, gain=1.0)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class SpatialLearnedEmbeddings(nn.Module):
    def __init__(self, height, width, channel, num_features=8):
        """
        PyTorch implementation of learned spatial embeddings
        MODIFIED: This class is no longer used.
        """
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel
        self.num_features = num_features
        # self.kernel = nn.Parameter(torch.empty(channel, height, width, num_features))
        # nn.init.kaiming_normal_(self.kernel, mode="fan_in", nonlinearity="linear")

    def forward(self, features):
        """
        Forward pass for spatial embedding
        MODIFIED: This class is no longer used.
        """
        raise NotImplementedError("SpatialLearnedEmbeddings is disabled in VLM-embedding mode")


class RescaleFromTanh(Transform):
    def __init__(self, low: float = -1, high: float = 1):
        super().__init__()

        self.low = low

        self.high = high

    def _call(self, x):
        # Rescale from (-1, 1) to (low, high)

        return 0.5 * (x + 1.0) * (self.high - self.low) + self.low

    def _inverse(self, y):
        # Rescale from (low, high) back to (-1, 1)

        return 2.0 * (y - self.low) / (self.high - self.low) - 1.0

    def log_abs_det_jacobian(self, x, y):
        # log|d(rescale)/dx| = sum(log(0.5 * (high - low)))

        scale = 0.5 * (self.high - self.low)

        return torch.sum(torch.log(scale), dim=-1)


class TanhMultivariateNormalDiag(TransformedDistribution):
    def __init__(self, loc, scale_diag, low=None, high=None):
        base_dist = MultivariateNormal(loc, torch.diag_embed(scale_diag))

        transforms = [TanhTransform(cache_size=1)]

        if low is not None and high is not None:
            low = torch.as_tensor(low)

            high = torch.as_tensor(high)

            transforms.insert(0, RescaleFromTanh(low, high))

        super().__init__(base_dist, transforms)

    def mode(self):
        # Mode is mean of base distribution, passed through transforms

        x = self.base_dist.mean

        for transform in self.transforms:
            x = transform(x)

        return x

    def stddev(self):
        std = self.base_dist.stddev

        x = std

        for transform in self.transforms:
            x = transform(x)

        return x
