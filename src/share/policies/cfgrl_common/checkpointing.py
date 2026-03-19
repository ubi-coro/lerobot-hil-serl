"""Checkpoint helpers for reproducible CFGRL policy bundles."""

from __future__ import annotations

from pathlib import Path

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import DataProcessorPipeline


def save_policy_bundle(
    policy: PreTrainedPolicy,
    save_dir: str | Path,
    *,
    preprocessor: DataProcessorPipeline | None = None,
    postprocessor: DataProcessorPipeline | None = None,
) -> Path:
    """Save a policy together with optional pre/post processors."""

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(save_path)
    if preprocessor is not None:
        preprocessor.save_pretrained(save_path)
    if postprocessor is not None:
        postprocessor.save_pretrained(save_path)
    return save_path


def load_policy_bundle(
    policy_cls: type[PreTrainedPolicy],
    save_dir: str | Path,
    *,
    config=None,
):
    """Load a saved policy bundle and its share-local processor pipelines."""

    policy = policy_cls.from_pretrained(save_dir, config=config)
    preprocessor = DataProcessorPipeline.from_pretrained(save_dir, config_filename="policy_preprocessor.json")
    postprocessor = DataProcessorPipeline.from_pretrained(save_dir, config_filename="policy_postprocessor.json")
    return policy, preprocessor, postprocessor
