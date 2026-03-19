"""Vision backbone adapters for the share-local visuomotor CFGRL stack.

The policy and critic both consume a generic ``VisionBackboneOutput`` with:

- ``pooled``: a compact global summary that is useful for global conditioning.
- ``tokens``: patch/spatial tokens used by token-aware fusion blocks.

Tuning modes intentionally describe the *vision stack* contract used by the CFGRL
modules rather than only the raw encoder weights:

- ``frozen``: freeze the backbone and the lightweight projection layers attached
  directly to backbone outputs.
- ``projector_only``: freeze the backbone, but keep those external projection
  layers trainable so the policy/critic can adapt token interfaces cheaply.
- ``full``: train the backbone together with the projection layers.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import draccus
import torch
from torch import Tensor, nn

from .modules import MLP


@dataclass
class VisionBackboneOutput:
    """Standardized vision-encoder output for pooled and token-level features."""

    pooled: Tensor
    tokens: Tensor
    spatial_shape: tuple[int, int] | None = None


@dataclass
class VisionBackboneConfig(draccus.ChoiceRegistry, abc.ABC):
    """Choice-registered config for all supported CFGRL vision backbones."""

    tune_mode: str = "frozen"
    image_size: tuple[int, int] = (224, 224)
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@VisionBackboneConfig.register_subclass("mock")
@dataclass
class MockVisionBackboneConfig(VisionBackboneConfig):
    hidden_dim: int = 32
    out_dim: int = 64
    image_size: tuple[int, int] = (32, 32)
    image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_std: tuple[float, float, float] = (0.5, 0.5, 0.5)


@VisionBackboneConfig.register_subclass("timm")
@dataclass
class TimmVisionBackboneConfig(VisionBackboneConfig):
    model_name: str = "vit_base_patch14_dinov2"
    pretrained: bool = True
    out_indices: tuple[int, ...] | None = None
    image_size: tuple[int, int] = (224, 224)
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)


@VisionBackboneConfig.register_subclass("hf_auto")
@dataclass
class HFVisionBackboneConfig(VisionBackboneConfig):
    repo_id: str = "facebook/dinov2-base"
    use_auto_backbone: bool = True
    trust_remote_code: bool = False
    local_files_only: bool = False
    image_size: tuple[int, int] = (224, 224)
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)


class BaseVisionBackbone(nn.Module, abc.ABC):
    """Common interface exposed to both the policy and critic."""

    def __init__(self, config: VisionBackboneConfig) -> None:
        super().__init__()
        self.config = config

    @property
    @abc.abstractmethod
    def output_dim(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, pixel_values: Tensor) -> VisionBackboneOutput:
        raise NotImplementedError

    def configure_tuning_mode(self) -> None:
        """Apply the requested parameter-freezing strategy to encoder weights.

        ``projector_only`` intentionally behaves like ``frozen`` at the backbone
        level because the trainable projectors live in the policy/critic modules
        that consume backbone outputs.
        """

        if self.config.tune_mode == "full":
            for param in self.parameters():
                param.requires_grad_(True)
            return
        if self.config.tune_mode in {"frozen", "projector_only"}:
            for param in self.parameters():
                param.requires_grad_(False)
            return
        raise ValueError(f"Unsupported tune_mode={self.config.tune_mode}")


def should_train_backbone_outputs(tune_mode: str) -> bool:
    """Return whether modules attached directly to backbone outputs should train."""

    if tune_mode not in {"frozen", "projector_only", "full"}:
        raise ValueError(f"Unsupported tune_mode={tune_mode}")
    return tune_mode in {"projector_only", "full"}


class MockVisionBackbone(BaseVisionBackbone):
    """Tiny convolutional backbone used by CPU tests and sanity scripts."""

    def __init__(self, config: MockVisionBackboneConfig) -> None:
        super().__init__(config)
        self.conv = nn.Sequential(
            nn.Conv2d(3, config.hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(config.hidden_dim, config.hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(config.hidden_dim, config.hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.token_proj = nn.Linear(config.hidden_dim, config.out_dim)
        self.pooled_proj = MLP(config.out_dim, config.out_dim, config.out_dim, num_layers=2)
        self.configure_tuning_mode()

    @property
    def output_dim(self) -> int:
        return self.config.out_dim

    def forward(self, pixel_values: Tensor) -> VisionBackboneOutput:
        feats = self.conv(pixel_values)
        tokens = feats.flatten(2).transpose(1, 2)
        tokens = self.token_proj(tokens)
        pooled = self.pooled_proj(tokens.mean(dim=1))
        return VisionBackboneOutput(
            pooled=pooled,
            tokens=tokens,
            spatial_shape=(feats.shape[-2], feats.shape[-1]),
        )


class TimmVisionBackbone(BaseVisionBackbone):
    """Adapter around ``timm`` backbones, defaulting to DINOv2 variants."""

    def __init__(self, config: TimmVisionBackboneConfig) -> None:
        super().__init__(config)
        try:
            import timm
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError("timm is required for TimmVisionBackbone") from exc

        self.timm = timm
        self.model = timm.create_model(config.model_name, pretrained=config.pretrained, num_classes=0)
        out_dim = getattr(self.model, "num_features", None)
        if out_dim is None:
            raise ValueError(f"Could not infer output dim from timm model {config.model_name}")
        self._output_dim = int(out_dim)
        self.configure_tuning_mode()

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def _to_tokens(self, out: Any) -> tuple[Tensor, tuple[int, int] | None]:
        if isinstance(out, dict):
            if "x_norm_patchtokens" in out:
                tokens = out["x_norm_patchtokens"]
                return tokens, None
            if "last_hidden_state" in out:
                tokens = out["last_hidden_state"]
                return tokens, None
            raise ValueError(f"Unsupported timm dict output keys: {list(out.keys())}")
        if torch.is_tensor(out):
            if out.ndim == 4:
                feats = out
                tokens = feats.flatten(2).transpose(1, 2)
                return tokens, (feats.shape[-2], feats.shape[-1])
            if out.ndim == 3:
                return out, None
            if out.ndim == 2:
                return out.unsqueeze(1), None
        raise ValueError(f"Unsupported timm output shape/type: {type(out)}")

    def forward(self, pixel_values: Tensor) -> VisionBackboneOutput:
        features = self.model.forward_features(pixel_values)
        tokens, spatial_shape = self._to_tokens(features)
        pooled = tokens.mean(dim=1)
        return VisionBackboneOutput(pooled=pooled, tokens=tokens, spatial_shape=spatial_shape)


class HFVisionBackboneAdapter(BaseVisionBackbone):
    """Hugging Face backbone adapter for repo-id based visual encoders."""

    def __init__(self, config: HFVisionBackboneConfig) -> None:
        super().__init__(config)
        try:
            from transformers import AutoBackbone, AutoModel
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError("transformers is required for HFVisionBackboneAdapter") from exc

        kwargs = {
            "trust_remote_code": config.trust_remote_code,
            "local_files_only": config.local_files_only,
        }
        if config.use_auto_backbone:
            self.model = AutoBackbone.from_pretrained(config.repo_id, **kwargs)
        else:
            self.model = AutoModel.from_pretrained(config.repo_id, **kwargs)

        out_dim = getattr(self.model.config, "hidden_size", None)
        if out_dim is None:
            out_dim = getattr(self.model.config, "hidden_sizes", [None])[-1]
        if out_dim is None:
            raise ValueError(f"Could not infer output dim from HF model {config.repo_id}")
        self._output_dim = int(out_dim)
        self.configure_tuning_mode()

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, pixel_values: Tensor) -> VisionBackboneOutput:
        out = self.model(pixel_values=pixel_values)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            pooled = out.last_hidden_state.mean(dim=1)
        elif hasattr(out, "feature_maps") and out.feature_maps:
            last = out.feature_maps[-1]
            pooled = last.flatten(2).mean(dim=-1)
        else:
            raise ValueError("Unsupported HF backbone output")

        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            tokens = out.last_hidden_state
            spatial_shape = None
        elif hasattr(out, "feature_maps") and out.feature_maps:
            last = out.feature_maps[-1]
            tokens = last.flatten(2).transpose(1, 2)
            spatial_shape = (last.shape[-2], last.shape[-1])
        else:
            tokens = pooled.unsqueeze(1)
            spatial_shape = None
        return VisionBackboneOutput(pooled=pooled, tokens=tokens, spatial_shape=spatial_shape)


def build_vision_backbone(config: VisionBackboneConfig) -> BaseVisionBackbone:
    """Instantiate the configured vision backbone implementation."""

    if config.type == "mock":
        return MockVisionBackbone(config)
    if config.type == "timm":
        return TimmVisionBackbone(config)
    if config.type == "hf_auto":
        return HFVisionBackboneAdapter(config)
    raise ValueError(f"Unsupported backbone type={config.type}")
