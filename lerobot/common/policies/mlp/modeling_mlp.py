#!/usr/bin/env python

# BC (Behavior Cloning) Policy based on SAC templates, without critics, trained via supervised learning.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from dataclasses import asdict
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.mlp.configuration_mlp import MLPConfig
from lerobot.common.policies.sac.configuration_sac import SACConfig
from lerobot.common.policies.sac.modeling_sac import (
    SACObservationEncoder,
    MLP,
    Policy,
    _convert_normalization_params_to_tensor
)


class MLPPolicy(PreTrainedPolicy):
    """
    Behavior Cloning policy using the same encoder and policy architecture as SAC,
    but trained via supervised MSE loss on demonstrated actions.
    """
    config_class = MLPConfig
    name = "mlp"

    def __init__(
        self,
        config: MLPConfig | None = None,
        dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Action dimension
        continuous_action_dim = config.output_features["action"].shape[0]

        # Normalization
        self._init_normalization(dataset_stats)
        # Encoders
        self._init_encoders()
        # Actor (policy network)
        self._init_actor(continuous_action_dim)

    def get_optim_params(self) -> dict:
        return {"actor": self.actor.parameters()}

    def reset(self):
        """Reset the policy"""
        pass

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Deterministic action selection (mode of the policy).
        """
        obs_feats = None

        # Encode observations
        features = self.actor.encoder(batch, cache=obs_feats, detach=False)
        net_out = self.actor.network(features)
        mean = self.actor.mean_layer(net_out)
        if self.actor.use_tanh_squash:
            action_unscaled = torch.tanh(mean)
        else:
            action_unscaled = mean
        action = self.unnormalize_outputs({"action": action_unscaled})["action"]
        return action

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        model: str = "actor"
    ) -> dict[str, torch.Tensor]:
        """
        Compute supervised loss.
        """
        if model == "actor":
            observations = batch["state"] if "state" in batch else batch["observations"]
            actions_gt = batch["action"]
            obs_feats = batch.get("observation_feature")
            loss_actor = self.compute_loss_actor(observations, actions_gt, obs_feats)
            return {"loss_actor": loss_actor}
        else:
            raise ValueError(f"Unknown model type: {model}")

    def compute_loss_actor(
        self,
        observations: dict[str, torch.Tensor],
        actions_gt: torch.Tensor,
        observation_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        MSE loss between predicted deterministic policy actions and ground truth actions.
        """
        # Normalize gt actions
        actions_gt = self.normalize_targets({"action": actions_gt})["action"]
        # Encode
        features = self.actor.encoder(observations, cache=observation_features, detach=False)
        net_out = self.actor.network(features)
        mean = self.actor.mean_layer(net_out)
        # Squash if needed
        if self.actor.use_tanh_squash:
            action_pred = torch.tanh(mean)
        else:
            action_pred = mean
        # Supervised L2 loss
        loss = F.mse_loss(action_pred, actions_gt)
        return loss

    def _init_normalization(
        self,
        dataset_stats: Optional[dict[str, dict[str, torch.Tensor]]],
    ) -> None:
        self.normalize_inputs = nn.Identity()
        self.normalize_targets = nn.Identity()
        self.unnormalize_outputs = nn.Identity()
        if self.config.dataset_stats:
            params = _convert_normalization_params_to_tensor(self.config.dataset_stats)
            self.normalize_inputs = Normalize(
                self.config.input_features,
                self.config.normalization_mapping,
                params,
            )
            stats = dataset_stats or params
            self.normalize_targets = Normalize(
                self.config.output_features,
                self.config.normalization_mapping,
                stats,
            )
            self.unnormalize_outputs = Unnormalize(
                self.config.output_features,
                self.config.normalization_mapping,
                stats,
            )

    def _init_encoders(self) -> None:
        _encoder_config = SACConfig(
            input_features=self.config.input_features,
            output_features=self.config.output_features,
            vision_encoder_name=self.config.vision_encoder_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            latent_dim=self.config.latent_dim,
        )
        self.encoder = SACObservationEncoder(_encoder_config, self.normalize_inputs)

    def _init_actor(self, continuous_action_dim: int) -> None:
        self.actor = Policy(
            encoder=self.encoder,
            network=MLP(
                input_dim=self.encoder.output_dim,
                **asdict(self.config.actor_network_kwargs)
            ),
            action_dim=continuous_action_dim,
            **asdict(self.config.policy_kwargs),
        )