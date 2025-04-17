import logging
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from lerobot.common.constants import OBS_IMAGE
from lerobot.common.policies.hilserl.classifier.configuration_classifier import (
    ClassifierConfig,
)
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ClassifierOutput:
    """Wrapper for classifier outputs with additional metadata."""

    def __init__(
        self,
        logits: Tensor,
        probabilities: Optional[Tensor] = None,
        hidden_states: Optional[Tensor] = None,
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


class Classifier(PreTrainedPolicy):
    """Image classifier built on top of a pre-trained encoder."""

    name = "hilserl_classifier"
    config_class = ClassifierConfig

    def __init__(
        self,
        config: ClassifierConfig,
        dataset_stats: Dict[str, Dict[str, Tensor]] | None = None,
    ):
        from transformers import AutoModel

        super().__init__(config)
        self.config = config

        # Initialize normalization (standardized with the policy framework)
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        # Set up encoder
        encoder = AutoModel.from_pretrained(self.config.model_name, trust_remote_code=True)
        # Extract vision model if we're given a multimodal model
        if hasattr(encoder, "vision_model"):
            logging.info("Multimodal model detected - using vision encoder only")
            self.encoder = encoder.vision_model
            self.vision_config = encoder.config.vision_config
        else:
            self.encoder = encoder
            self.vision_config = getattr(encoder, "config", None)

        # Model type from config
        self.is_cnn = self.config.model_type == "cnn"

        # For CNNs, initialize backbone
        if self.is_cnn:
            self._setup_cnn_backbone()

        self._freeze_encoder()
        self._build_classifier_head()

    def _setup_cnn_backbone(self):
        """Set up CNN encoder"""
        if hasattr(self.encoder, "fc"):
            self.feature_dim = self.encoder.fc.in_features
            self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        elif hasattr(self.encoder.config, "hidden_sizes"):
            self.feature_dim = self.encoder.config.hidden_sizes[-1]  # Last channel dimension
        else:
            raise ValueError("Unsupported CNN architecture")

    def _freeze_encoder(self) -> None:
        """Freeze the encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _build_classifier_head(self) -> None:
        """Initialize the classifier head architecture."""
        # Get input dimension based on model type
        if self.is_cnn:
            input_dim = self.feature_dim
        else:  # Transformer models
            if hasattr(self.encoder.config, "hidden_size"):
                input_dim = self.encoder.config.hidden_size
            else:
                raise ValueError("Unsupported transformer architecture since hidden_size is not found")

        self.classifier_head = nn.Sequential(
            nn.Linear(input_dim * self.config.num_cameras, self.config.hidden_dim),
            nn.Dropout(self.config.dropout_rate),
            nn.LayerNorm(self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(
                self.config.hidden_dim,
                1 if self.config.num_classes == 2 else self.config.num_classes,
            ),
        )

    def _get_encoder_output(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the appropriate output from the encoder."""
        with torch.no_grad():
            if self.is_cnn:
                # The HF ResNet applies pooling internally
                outputs = self.encoder(x)
                # Get pooled output directly
                features = outputs.pooler_output

                if features.dim() > 2:
                    features = features.squeeze(-1).squeeze(-1)
                return features
            else:  # Transformer models
                outputs = self.encoder(x)
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    return outputs.pooler_output
                return outputs.last_hidden_state[:, 0, :]

    def extract_images_and_labels(self, batch: Dict[str, Tensor]) -> Tuple[list, Tensor]:
        """Extract image tensors and label tensors from batch."""
        # Find image keys in input features
        image_keys = [key for key in self.config.input_features if key.startswith(OBS_IMAGE)]

        # Extract the images and labels
        images = [batch[key] for key in image_keys]
        labels = batch["next.reward"]

        return images, labels

    def predict(self, xs: list) -> ClassifierOutput:
        """Forward pass of the classifier for inference."""
        encoder_outputs = torch.hstack([self._get_encoder_output(x) for x in xs])
        logits = self.classifier_head(encoder_outputs)

        if self.config.num_classes == 2:
            logits = logits.squeeze(-1)
            probabilities = torch.sigmoid(logits)
        else:
            probabilities = torch.softmax(logits, dim=-1)

        return ClassifierOutput(logits=logits, probabilities=probabilities, hidden_states=encoder_outputs)

    def forward(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Standard forward pass for training compatible with train.py."""
        # Normalize inputs if needed
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)

        # Extract images and labels
        images, labels = self.extract_images_and_labels(batch)

        # Get predictions
        outputs = self.predict(images)

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

    def predict_reward(self, batch, threshold=0.6):
        """Legacy method for compatibility."""
        images, _ = self.extract_images_and_labels(batch)
        if self.config.num_classes == 2:
            probs = self.predict(images).probabilities
            logging.debug(f"Predicted reward images: {probs}")
            return (probs > threshold).float()
        else:
            return torch.argmax(self.predict(images).probabilities, dim=1)

    # Methods required by PreTrainedPolicy abstract class

    def get_optim_params(self) -> dict:
        """Return optimizer parameters for the policy."""
        return {
            "params": self.parameters(),
            "lr": getattr(self.config, "learning_rate", 1e-4),
            "weight_decay": getattr(self.config, "weight_decay", 0.01),
        }

    def reset(self):
        """Reset any stateful components (required by PreTrainedPolicy)."""
        # Classifier doesn't have stateful components that need resetting
        pass

    def select_action(self, batch: Dict[str, Tensor]) -> Tensor:
        """Return action (class prediction) based on input observation."""
        images, _ = self.extract_images_and_labels(batch)

        with torch.no_grad():
            outputs = self.predict(images)

            if self.config.num_classes == 2:
                # For binary classification return 0 or 1
                return (outputs.probabilities > 0.5).float()
            else:
                # For multi-class return the predicted class
                return torch.argmax(outputs.probabilities, dim=1)
