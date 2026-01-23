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
    OBS_LANGUAGE_ATTENTION_MASK, POLICY_PREPROCESSOR_DEFAULT_NAME, POLICY_POSTPROCESSOR_DEFAULT_NAME, ACTION, OBS_STATE, OBS_IMAGES,
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
        B = self._infer_batch_size(observation)

        comp = transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if comp is None or self.task_key not in comp:
            raise ValueError(f"Missing complementary_data['{self.task_key}'] for task tokenization.")
        task = comp[self.task_key]
        if task is None:
            raise ValueError("Task is None.")

        # Normalize to list[str] length B
        if isinstance(task, str):
            task = [task] * B

        assert len(task) == B
        return task

    # --------------------
    # Advantage extraction (batched)
    # --------------------

    def get_adv_labels(self, observation: dict[str, Any], transition: EnvTransition) -> list[bool]:
        B = self._infer_batch_size(observation)
        v = [True] * B

        if self.advantage_key in observation:
            v = observation[self.advantage_key]
        else:
            comp = transition.get(TransitionKey.COMPLEMENTARY_DATA)
            if comp is not None and self.advantage_key in comp:
                v = comp[self.advantage_key]

        # Convert to list[bool] length B
        if isinstance(v, (int, float, bool)):
            v = [bool(v)] * B

        if isinstance(v, torch.Tensor):
            v = v.squeeze().tolist()

        assert len(v) == B
        return v

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


@dataclass
@ProcessorStepRegistry.register(name="chunked_sarsa_processor")
class ChunkedSARSAProcessorStep(ObservationProcessorStep):
    """
    Restructure a flat batch into a nested SARSA-style batch.

    Input (training-style):
      {
        OBS_STATE:              [B, 2, D]         or [B, 2, ...]
        f"{OBS_IMAGES}_camX":   [B, 2, C, H, W]   (for each cam key present)
        ACTION:                [B, 2*H, ...]      (sequence axis is dim=1)
        ... (tokens, reward, done, etc)
      }

    Output:
      {
        "state": {
            OBS_STATE:            [B, D]          (t=0)
            f"{OBS_IMAGES}_camX": [B, C, H, W]    (t=0)
            ...
        },
        "next_state": {
            OBS_STATE:            [B, D]          (t=H)
            f"{OBS_IMAGES}_camX": [B, C, H, W]    (t=H)
            ...
        },
        ACTION:                 [B, H, ...]
        f"next_{ACTION}":       [B, H, ...]
        ... (all other keys passed through unchanged)
      }

    Inference-style (single observation, no time axis, no actions):
      - If OBS_STATE / images have no time axis (dim=1 != 2), they are placed into "state"
      - "next_state" and "next_ACTION" are omitted
      - All other fields are left untouched / absent (no errors)
    """

    state_key: str = "state"
    next_state_key: str = "next_state"
    next_action_key: str = f"next_{ACTION}"

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        # If already structured, do nothing.
        if self.state_key in observation:
            return observation

        batch = dict(observation)

        # Collect observation keys to move into state/next_state
        obs_keys: list[str] = []
        if OBS_STATE in batch:
            obs_keys.append(OBS_STATE)

        for k in list(batch.keys()):
            if isinstance(k, str) and k.startswith(f"{OBS_IMAGES}_"):
                obs_keys.append(k)

        state: dict[str, Any] = {}
        next_state: dict[str, Any] = {}
        has_next = False

        for k in obs_keys:
            v = batch.get(k, None)
            if not isinstance(v, torch.Tensor):
                # If it's not a tensor, just treat it as "state" metadata.
                if v is not None:
                    state[k] = v
                batch.pop(k, None)
                continue

            # Training-style two-timestep axis: [B, 2, ...]
            if v.ndim >= 2 and v.shape[1] == 2:
                state[k] = v[:, 0]
                next_state[k] = v[:, 1]
                has_next = True
                batch.pop(k, None)
                continue

            # Occasionally you might see [B, 1, ...] (e.g., value/inference with a history dim)
            if v.ndim >= 2 and v.shape[1] == 1:
                state[k] = v[:, 0]
                batch.pop(k, None)
                continue

            # Inference-style: no explicit time axis; put it into state as-is.
            state[k] = v
            batch.pop(k, None)

        if len(state) > 0:
            batch[self.state_key] = state
        if has_next and len(next_state) > 0:
            batch[self.next_state_key] = next_state

        # Split ACTION into (action, next_action) if present and shaped like a 2H sequence.
        if ACTION in batch and isinstance(batch[ACTION], torch.Tensor):
            a: torch.Tensor = batch[ACTION]
            T = int(a.shape[1])

            H = T // 2
            batch[ACTION] = a[:, :H]
            batch[self.next_action_key] = a[:, H:]

        return batch
