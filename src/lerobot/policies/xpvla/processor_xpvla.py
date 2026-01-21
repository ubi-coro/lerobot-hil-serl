from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from lerobot.policies.xvla.processor_xvla import XVLAImageToFloatProcessorStep, XVLAImageNetNormalizeProcessorStep, XVLAAddDomainIdProcessorStep

from lerobot.configs.types import FeatureType, PolicyFeature, PipelineFeatureType
from lerobot.policies.xpvla.configuration_xpvla import XPVLAConfig
from lerobot.processor import PolicyProcessorPipeline, PolicyAction, RenameObservationsProcessorStep, AddBatchDimensionProcessorStep, \
    DeviceProcessorStep, NormalizerProcessorStep, UnnormalizerProcessorStep, ObservationProcessorStep, ProcessorStepRegistry, EnvTransition, \
    TransitionKey
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import (
    OBS_LANGUAGE_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK, POLICY_PREPROCESSOR_DEFAULT_NAME, POLICY_POSTPROCESSOR_DEFAULT_NAME,
)

try:
    from transformers import AutoTokenizer
    _transformers_available = True
except Exception:
    AutoTokenizer = None
    _transformers_available = False


def make_xpvla_pre_post_processors(
    config: XPVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Build the LeRobot processor pipelines for XVLA.
    """

    features = {**config.input_features, **config.output_features}
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DualTokenizerWithAdvantageProcessorStep(
            tokenizer_name=config.xvla.tokenizer_name,
            max_length=config.xvla.tokenizer_max_length,
            padding=config.xvla.pad_language_to,
            padding_side=config.xvla.tokenizer_padding_side,
            advantage_key=config.advantage_key,
            pos_adv_text=config.pos_adv_text,
            neg_adv_text=config.neg_adv_text,
            sep=config.seperator_text,
            cond_tokens_key=config.cond_tokens_key,
            cond_attn_key=config.cond_attn_key
        ),
        XVLAImageToFloatProcessorStep(),
        XVLAImageNetNormalizeProcessorStep(),
        XVLAAddDomainIdProcessorStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features=features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )



@dataclass
@ProcessorStepRegistry.register(name="dual_tokenizer_with_advantage")
class DualTokenizerWithAdvantageProcessorStep(ObservationProcessorStep):
    """
    Tokenizes task text twice:
      1) unconditional tokens -> OBS_LANGUAGE_TOKENS (+ attention mask)
      2) conditioned tokens -> cond_tokens_key (+ attention mask) if advantage label is available

    Advantage label source:
      - observation[advantage_key] (preferred)
      - complementary_data[advantage_key] (fallback)

    If the advantage label is missing:
      - require_adv_label=False: skip conditional tokenization (value training path)
      - require_adv_label=True: raise (policy improvement / rollout path)
    """

    tokenizer_name: str | None = None
    tokenizer: Any | None = None

    max_length: int = 128
    task_key: str = "task"
    advantage_key: str = "advantage_label"
    require_adv_label: bool = False
    pos_adv_text: str = "Optimal: true"
    neg_adv_text: str = "Optimal: false"
    sep: str = " "
    cond_tokens_key: str = field(default_factory=lambda: f"{OBS_LANGUAGE_TOKENS}_cond")
    cond_attn_key: str = field(default_factory=lambda: f"{OBS_LANGUAGE_ATTENTION_MASK}_cond")

    padding_side: str = "right"
    padding: str = "max_length"
    truncation: bool = True

    input_tokenizer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if not _transformers_available:
            raise ImportError(
                "transformers is required for DualTokenizerWithAdvantageProcessorStep. "
                "Install with `pip install 'lerobot[transformers-dep]'`."
            )
        if self.tokenizer is not None:
            self.input_tokenizer = self.tokenizer
        elif self.tokenizer_name is not None:
            if AutoTokenizer is None:
                raise ImportError("AutoTokenizer is not available")
            self.input_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        else:
            raise ValueError("Provide either tokenizer or tokenizer_name.")

    # --------------------
    # Helpers
    # --------------------

    @staticmethod
    def _detect_device(transition: EnvTransition) -> Optional[torch.device]:
        obs = transition.get(TransitionKey.OBSERVATION)
        if obs:
            for v in obs.values():
                if isinstance(v, torch.Tensor):
                    return v.device
        act = transition.get(TransitionKey.ACTION)
        if isinstance(act, torch.Tensor):
            return act.device
        return None

    @staticmethod
    def _infer_batch_size(observation: dict[str, Any]) -> int:
        # Find first tensor to infer batch dimension
        for v in observation.values():
            if isinstance(v, torch.Tensor):
                if v.ndim >= 1:
                    return int(v.shape[0])
        # Fallback: treat as single-sample
        return 1

    def _tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        return self.input_tokenizer(
            texts,
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
            padding_side=self.padding_side,
            return_tensors="pt",
        )

    # --------------------
    # Task extraction
    # --------------------

    def get_task_texts(self, observation: dict[str, Any], transition: EnvTransition) -> list[str]:
        comp = transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if comp is None or self.task_key not in comp:
            raise ValueError(f"Missing complementary_data['{self.task_key}'] for task tokenization.")
        task = comp[self.task_key]
        if task is None:
            raise ValueError("Task is None.")

        B = self._infer_batch_size(observation)

        # Normalize to list[str] length B
        if isinstance(task, str):
            return [task] * B

        if isinstance(task, list) and all(isinstance(t, str) for t in task):
            if len(task) == B:
                return task
            if len(task) == 1 and B > 1:
                return task * B
            raise ValueError(f"Task list length {len(task)} does not match batch size {B}.")
        raise ValueError(f"Task must be str or list[str], got: {type(task)}")

    # --------------------
    # Advantage extraction (batched)
    # --------------------

    def get_adv_labels(self, observation: dict[str, Any], transition: EnvTransition) -> Optional[list[bool]]:
        B = self._infer_batch_size(observation)

        v = None
        if self.advantage_key in observation:
            v = observation[self.advantage_key]
        else:
            comp = transition.get(TransitionKey.COMPLEMENTARY_DATA)
            if comp is not None and self.advantage_key in comp:
                v = comp[self.advantage_key]

        if v is None:
            return None

        # Convert to list[bool] length B
        if isinstance(v, bool):
            return [v] * B
        if isinstance(v, (int, float)):
            return [bool(v)] * B

        if isinstance(v, list):
            if not all(isinstance(x, (bool, int, float)) for x in v):
                raise ValueError("Advantage label list must contain bool/int/float values only.")
            if len(v) == B:
                return [bool(x) for x in v]
            if len(v) == 1 and B > 1:
                return [bool(v[0])] * B
            raise ValueError(f"Advantage label list length {len(v)} does not match batch size {B}.")

        if isinstance(v, torch.Tensor):
            # Accept shapes: (B,), (B,1), (B,...) -> squeeze to (B,)
            if v.ndim == 0:
                return [bool(v.item())] * B

            v_flat = v
            # If it's (B,1) or (B,1,1,...) squeeze trailing dims
            if v_flat.shape[0] != B:
                # Some pipelines store (1,) even if B inferred differently; attempt broadcast if scalar-ish
                if v_flat.numel() == 1:
                    return [bool(v_flat.item())] * B
                raise ValueError(f"Advantage tensor batch dim {v_flat.shape[0]} does not match inferred B={B}.")

            v_flat = v_flat.view(B, -1)
            if v_flat.shape[1] != 1:
                raise ValueError(
                    f"Advantage tensor must be scalar per sample; got shape {tuple(v.shape)}."
                )
            return [bool(x) for x in v_flat[:, 0].tolist()]

        raise ValueError(f"Cannot convert advantage labels of type {type(v)}.")

    # --------------------
    # Main step
    # --------------------

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        task_texts = self.get_task_texts(observation, self.transition)
        adv_labels = self.get_adv_labels(observation, self.transition)

        if adv_labels is None:
            if self.require_adv_label:
                raise ValueError(
                    f"Missing advantage label '{self.advantage_key}' but require_adv_label=True."
                )
        else:
            if len(adv_labels) != len(task_texts):
                raise ValueError(
                    f"Mismatch: {len(task_texts)} task texts vs {len(adv_labels)} advantage labels."
                )

        # Unconditional tokenization (always)
        tok_u = self._tokenize(task_texts)

        # Conditional tokenization (only if labels exist)
        tok_c = None
        if adv_labels is not None:
            task_texts_c = [
                f"{t}{self.sep}{self.pos_adv_text if a else self.neg_adv_text}"
                for t, a in zip(task_texts, adv_labels)
            ]
            tok_c = self._tokenize(task_texts_c)

        # Move to same device as transition tensors
        target_device = self._detect_device(self.transition)
        if target_device is not None:
            tok_u = {k: v.to(target_device) for k, v in tok_u.items()}
            if tok_c is not None:
                tok_c = {k: v.to(target_device) for k, v in tok_c.items()}

        new_obs = dict(observation)
        new_obs[OBS_LANGUAGE_TOKENS] = tok_u["input_ids"]
        new_obs[OBS_LANGUAGE_ATTENTION_MASK] = tok_u["attention_mask"].to(dtype=torch.bool)

        if tok_c is not None:
            new_obs[self.cond_tokens_key] = tok_c["input_ids"]
            new_obs[self.cond_attn_key] = tok_c["attention_mask"].to(dtype=torch.bool)

        return new_obs

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # Unconditional keys
        if OBS_LANGUAGE_TOKENS not in features[PipelineFeatureType.OBSERVATION]:
            features[PipelineFeatureType.OBSERVATION][OBS_LANGUAGE_TOKENS] = PolicyFeature(
                type=FeatureType.LANGUAGE, shape=(self.max_length,)
            )
        if OBS_LANGUAGE_ATTENTION_MASK not in features[PipelineFeatureType.OBSERVATION]:
            features[PipelineFeatureType.OBSERVATION][OBS_LANGUAGE_ATTENTION_MASK] = PolicyFeature(
                type=FeatureType.LANGUAGE, shape=(self.max_length,)
            )

        # Conditional keys (safe to always declare)
        if self.cond_tokens_key not in features[PipelineFeatureType.OBSERVATION]:
            features[PipelineFeatureType.OBSERVATION][self.cond_tokens_key] = PolicyFeature(
                type=FeatureType.LANGUAGE, shape=(self.max_length,)
            )
        if self.cond_attn_key not in features[PipelineFeatureType.OBSERVATION]:
            features[PipelineFeatureType.OBSERVATION][self.cond_attn_key] = PolicyFeature(
                type=FeatureType.LANGUAGE, shape=(self.max_length,)
            )

        return features
