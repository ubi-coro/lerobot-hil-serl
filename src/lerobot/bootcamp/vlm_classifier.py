# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import torch.nn as nn
from torch import Tensor
from collections.abc import Callable

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.sac.reward_model.configuration_classifier import RewardClassifierConfig
from lerobot.utils.constants import OBS_STATE, REWARD, ACTION # Added ACTION here for context


class ClassifierOutput:
    """Wrapper for classifier outputs with additional metadata."""

    def __init__(
        self,
        logits: Tensor,
        probabilities: Tensor | None = None,
        hidden_states: Tensor | None = None,
    ):
        self.logits = logits
        self.probabilities = probabilities
        self.hidden_states = hidden_states

    def __repr__(self):
        return (
            f"ClassifierOutput(logits={self.logits}, "
            f"probabilities={self.probabilities}, "
            f"hidden_states={self.hidden_states})"
        )


# --- VLM-EMBEDDING SPECIFIC ENCODER (MLP Projector) ---
class VLMEmbeddingEncoder(nn.Module):
    """
    Encoder specialized for VLM embeddings (OBS_STATE).
    Acts as a simple MLP projector: (VLM_EMBED_DIM) -> (latent_dim).
    """
    def __init__(self, config: RewardClassifierConfig) -> None:
        super().__init__()

        # CRITICAL: This requires OBS_STATE to be present in the config
        if OBS_STATE not in config.input_features:
            raise ValueError(
                f"VLM embedding mode requires '{OBS_STATE}' in input_features."
                " The client must save the embedding under this key."
            )

        vlm_dim = config.input_features[OBS_STATE].shape[0]
        latent_dim = config.latent_dim

        # This MLP projects the VLM embedding down to the model's latent_dim
        self.state_encoder = nn.Sequential(
            nn.Linear(vlm_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Tanh(),
        )
        self._out_dim = latent_dim

    def forward(
        self, obs: dict[str, Tensor], action: Tensor | None = None, detach: bool = False
    ) -> Tensor:
        """
        Processes the VLM embedding (OBS_STATE) and concatenates it with the action.
        """
        # 1. Encode VLM embedding
        # obs[OBS_STATE] is the VLM embedding (B, VLM_EMBED_DIM)
        obs_enc = self.state_encoder(obs[OBS_STATE])

        if detach:
            obs_enc = obs_enc.detach()

        # 2. Concatenate with action if provided (for Q-function style classifiers)
        if action is not None and ACTION in obs:
            # We assume the action is already normalized/processed outside
            return torch.cat([obs_enc, action], dim=-1)

        return obs_enc

    @property
    def output_dim(self) -> int:
        return self._out_dim

# --- REMOVED: Image-specific classes are removed or stubbed out ---
class SpatialLearnedEmbeddings(nn.Module):
    def __init__(self, height, width, channel, num_features=8):
        super().__init__()
        raise NotImplementedError("SpatialLearnedEmbeddings is disabled in VLM-embedding mode.")

    def forward(self, features):
        raise NotImplementedError("SpatialLearnedEmbeddings is disabled in VLM-embedding mode.")


# --- MAIN CLASSIFIER ---
class VLMClassifier(PreTrainedPolicy):
    """Reward classifier specialized for VLM embeddings."""

    name = "vlm_reward_classifier"
    config_class = RewardClassifierConfig

    def __init__(
        self,
        config: RewardClassifierConfig,
    ):
        super().__init__(config)
        self.config = config

        # Set up encoder (now the VLM Embedding Projector)
        self.encoder = VLMEmbeddingEncoder(self.config)

        # The input feature dim for the head is the output of the VLMEmbeddingEncoder
        self.feature_dim = self.encoder.output_dim

        self._build_classifier_head()

    # --- Setup methods (Modified/Simplified) ---
    def _freeze_encoder(self) -> None:
        """MLP encoder is not typically frozen, but we follow the original style."""
        pass # No change needed here, as we don't have a massive image encoder.

    def _build_classifier_head(self) -> None:
        """Initialize the classifier head architecture."""

        # Input dimension for the head is the projected VLM embedding feature_dim + action dim
        # We assume the action is concatenated *inside* the classifier if it's a Q-function

        # For simplicity, we assume we are building a standard MLP head that takes
        # the single projected embedding vector.
        input_dim = self.feature_dim

        # The original code supported multiple cameras; here we assume num_cameras=1
        # and that the embedding *replaces* the camera inputs.

        self.classifier_head = nn.Sequential(
            # Using latent_dim (feature_dim) as input
            nn.Linear(input_dim, self.config.hidden_dim),
            nn.Dropout(self.config.dropout_rate),
            nn.LayerNorm(self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(
                self.config.hidden_dim,
                1 if self.config.num_classes == 2 else self.config.num_classes,
            ),
        )

    def _get_encoder_output(self, obs: dict[str, Tensor], action: Tensor | None = None) -> torch.Tensor:
        """
        Extracts the VLM embedding and processes it through the MLP projector.

        Args:
            obs: The observation dictionary containing OBS_STATE (VLM embedding).
            action: Optional action tensor for concatenation (Q-function style).

        Returns:
            The projected state features (B, latent_dim).
        """
        # The VLMEmbeddingEncoder handles the encoding (z_s -> latent_dim)
        # It takes the full observation dictionary `obs`
        return self.encoder(obs, action=action)

    def extract_state_and_labels(self, batch: dict[str, Tensor]) -> tuple[dict[str, Tensor], Tensor]:
        """Extract VLM embedding tensor and label tensors from batch."""

        # We only expect the OBS_STATE key for the VLM embedding
        state = {OBS_STATE: batch[OBS_STATE]}
        labels = batch[REWARD]

        return state, labels

    def predict(self, obs: dict[str, Tensor], action: Tensor | None = None) -> ClassifierOutput:
        """Forward pass of the classifier for inference."""

        # The encoder outputs the single (B, latent_dim) feature vector
        encoder_outputs = self._get_encoder_output(obs, action)

        # CRITICAL: If the classifier is a Q-function, we must concatenate the action before the head.
        # This assumes the action is given in the input batch if required.
        # However, for simplicity and adherence to the `Classifier(s)` pattern, we assume
        # the action is passed in the `action` argument. Let's merge it here if necessary.

        # If the reward model is meant to be R(s, a), the action should be passed to the encoder.
        # If the reward model is meant to be R(s), the action should be None.

        logits = self.classifier_head(encoder_outputs)

        if self.config.num_classes == 2:
            logits = logits.squeeze(-1)
            probabilities = torch.sigmoid(logits)
        else:
            probabilities = torch.softmax(logits, dim=-1)

        return ClassifierOutput(logits=logits, probabilities=probabilities, hidden_states=encoder_outputs)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        """Standard forward pass for training compatible with train.py."""

        # Extract state (embedding) and labels
        state, labels = self.extract_state_and_labels(batch)

        # Extract action if it's a Q-function style classifier
        action = batch.get(ACTION)

        # Get predictions
        outputs = self.predict(state, action=action)

        # Calculate loss
        if self.config.num_classes == 2:
            # Binary classification
            loss = nn.functional.binary_cross_entropy_with_logits(outputs.logits, labels)
            predictions = (torch.sigmoid(outputs.logits) > 0.5).float()
        else:
            # Multi-class classification
            loss = nn.functional.cross_entropy(outputs.logits, labels.long())
            predictions = torch.argmax(outputs.logits, dim=1)

        # Calculate accuracy for logging
        correct = (predictions == labels).sum().item()
        total = labels.size(0)
        accuracy = 100 * correct / total

        # Return loss and metrics for logging
        output_dict = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }

        return loss, output_dict

    def predict_reward(self, batch: dict[str, Tensor], threshold: float = 0.5) -> Tensor:
        """Eval method. Returns predicted reward with the decision threshold as argument."""
        # Check for OBS_STATE prefix
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)

        # Extract state (embedding) from batch dict
        state = {OBS_STATE: batch[OBS_STATE]}
        action = batch.get(ACTION)

        if self.config.num_classes == 2:
            probs = self.predict(state, action=action).probabilities
            logging.debug(f"Predicted reward images: {probs}")
            return (probs > threshold).float()
        else:
            return torch.argmax(self.predict(state, action=action).probabilities, dim=1)

    def get_optim_params(self):
        """Return optimizer parameters for the policy."""
        # Now returns parameters for the MLP encoder/projector and the head
        return self.parameters()

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This method is required by PreTrainedPolicy but not used for reward classifiers.
        The reward classifier is not an actor and does not select actions.
        """
        raise NotImplementedError("Reward classifiers do not select actions")

    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This method is required by PreTrainedPolicy but not used for reward classifiers.
        The reward classifier is not an actor and does not produce action chunks.
        """
        raise NotImplementedError("Reward classifiers do not predict action chunks")

    def reset(self):
        """
        This method is required by PreTrainedPolicy but not used for reward classifiers.
        The reward classifier is not an actor and does not select actions.
        """
        pass
