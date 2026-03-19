"""Utilities for critic-driven CFGRL dataset labeling.

This module operates on *chunk-aligned* datasets, meaning each dataset row
already corresponds to one action chunk of length ``critic.config.chunk_size``.
The resulting labels are therefore only valid for that chunk length and for the
critic / baseline settings used during labeling.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import pyarrow.dataset as pa_ds
import torch

from lerobot.datasets.utils import (
    DEFAULT_DATA_PATH,
    get_hf_dataset_size_in_mb,
    get_hf_features_from_features,
    load_info,
    load_json,
    load_stats,
    update_chunk_file_indices,
    write_info,
    write_json,
    write_stats,
)
from lerobot.utils.constants import ACTION
from share.policies.cfgrl_common.action_providers import CachedActionProvider, DatasetActionProvider, PolicyActionProvider
from share.policies.cfgrl_policy import CFGRLPolicy
from .modeling_cfgrl_critic import CFGRLCritic

CFGRL_LABELING_META_PATH = "meta/cfgrl_labeling.json"


@dataclass
class CFGRLLabelingSummary:
    """Summary statistics recorded for one labeling run."""

    label_field: str
    percentile: float
    threshold: float
    chunk_size: int
    num_rows: int
    num_positive: int
    positive_fraction: float
    advantage_mean: float
    advantage_std: float
    advantage_min: float
    advantage_max: float
    provider: str
    condition: int | None
    guidance_scale: float | None


def _stack_list(values: list[Any]) -> torch.Tensor:
    """Convert one column batch from HF datasets into a torch tensor."""

    if len(values) == 0:
        return torch.empty(0)
    first = values[0]
    if torch.is_tensor(first):
        return torch.stack(values)
    if isinstance(first, np.ndarray):
        tensor = torch.from_numpy(np.stack(values))
        return tensor.float() if tensor.is_floating_point() else tensor
    if hasattr(first, "__array__"):
        tensor = torch.from_numpy(np.stack([np.asarray(item) for item in values]))
        return tensor.float() if tensor.is_floating_point() else tensor
    if isinstance(first, (list, tuple)):
        tensor = torch.as_tensor(np.asarray(values))
        return tensor.float() if tensor.is_floating_point() else tensor
    tensor = torch.as_tensor(values)
    return tensor.float() if tensor.is_floating_point() else tensor


def _column_batch_to_critic_batch(column_batch: dict[str, Any]) -> dict[str, Any]:
    """Convert flat chunk-dataset columns into the nested critic batch structure.

    Supported layouts:

    - nested dict columns named ``state`` / ``next_state``
    - flat columns prefixed with ``state.`` / ``next_state.``
    """

    batch: dict[str, Any] = {}
    state: dict[str, Any] = {}
    next_state: dict[str, Any] = {}

    for key, value in column_batch.items():
        if key == "state" and isinstance(value, dict):
            state = {sub_key: _stack_list(sub_value) for sub_key, sub_value in value.items()}
            continue
        if key == "next_state" and isinstance(value, dict):
            next_state = {sub_key: _stack_list(sub_value) for sub_key, sub_value in value.items()}
            continue
        if key.startswith("state."):
            state[key.removeprefix("state.")] = _stack_list(value)
            continue
        if key.startswith("next_state."):
            next_state[key.removeprefix("next_state.")] = _stack_list(value)
            continue
        batch[key] = _stack_list(value) if isinstance(value, list) else value

    if state:
        batch["state"] = state
    if next_state:
        batch["next_state"] = next_state
    if ACTION in batch and not torch.is_tensor(batch[ACTION]):
        batch[ACTION] = _stack_list(batch[ACTION])
    return batch


def _slice_dataset_batch(hf_dataset: datasets.Dataset, start: int, end: int) -> dict[str, Any]:
    """Read a contiguous dataset slice and convert it to a critic batch."""

    column_batch = hf_dataset[start:end]
    return _column_batch_to_critic_batch(column_batch)


def load_local_chunk_dataset(dataset_root: str | Path) -> tuple[datasets.Dataset, dict[str, Any]]:
    """Load a local chunk dataset without relying on the HF cache lock path."""

    dataset_root = Path(dataset_root)
    info = load_info(dataset_root)
    features = get_hf_features_from_features(info["features"])
    paths = sorted((dataset_root / "data").glob("*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")
    table = pa_ds.dataset(paths, format="parquet").to_table()
    table = table.cast(features.arrow_schema)
    return datasets.Dataset(table), info


def _compute_advantages(
    critic: CFGRLCritic,
    hf_dataset: datasets.Dataset,
    *,
    batch_size: int,
    action_provider,
    device: str,
) -> torch.Tensor:
    """Run the critic over the dataset and return one advantage per row."""

    critic = critic.to(device)
    critic.eval()
    chunks: list[torch.Tensor] = []
    for start in range(0, len(hf_dataset), batch_size):
        end = min(start + batch_size, len(hf_dataset))
        batch = _slice_dataset_batch(hf_dataset, start, end)
        batch = CFGRLCritic._repeat_tree(batch, 1)
        batch = _move_to_device(batch, device)
        with torch.no_grad():
            chunks.append(critic.compute_advantage(batch, action_provider).cpu())
    return torch.cat(chunks, dim=0)


def _move_to_device(value: Any, device: str) -> Any:
    """Recursively move nested tensors to a device."""

    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def summarize_advantages(
    advantages: torch.Tensor,
    *,
    percentile: float,
    chunk_size: int,
    label_field: str,
    provider: str,
    condition: int | None,
    guidance_scale: float | None,
) -> tuple[torch.Tensor, CFGRLLabelingSummary]:
    """Threshold advantages with a percentile rule and summarize the result."""

    if not 0.0 < percentile < 100.0:
        raise ValueError("percentile must lie in (0, 100)")
    threshold = float(np.percentile(advantages.numpy(), percentile))
    labels = (advantages >= threshold).to(dtype=torch.int64)
    summary = CFGRLLabelingSummary(
        label_field=label_field,
        percentile=percentile,
        threshold=threshold,
        chunk_size=chunk_size,
        num_rows=int(advantages.numel()),
        num_positive=int(labels.sum().item()),
        positive_fraction=float(labels.float().mean().item()),
        advantage_mean=float(advantages.mean().item()),
        advantage_std=float(advantages.std(unbiased=False).item()),
        advantage_min=float(advantages.min().item()),
        advantage_max=float(advantages.max().item()),
        provider=provider,
        condition=condition,
        guidance_scale=guidance_scale,
    )
    return labels, summary


def _write_chunked_dataset(
    hf_dataset: datasets.Dataset,
    dataset_root: Path,
    *,
    data_file_size_mb: float,
    chunk_folder_size: int,
) -> None:
    """Rewrite the dataset's ``data/`` directory while preserving episode integrity."""

    tmp_root = Path(tempfile.mkdtemp(prefix="cfgrl_labeling_", dir=str(dataset_root.parent)))
    try:
        dataset_size_in_mb = get_hf_dataset_size_in_mb(hf_dataset)
        if dataset_size_in_mb <= data_file_size_mb:
            path = tmp_root / DEFAULT_DATA_PATH.format(chunk_index=0, file_index=0)
            path.parent.mkdir(parents=True, exist_ok=True)
            hf_dataset.to_parquet(path)
        else:
            episode_indices = np.asarray(hf_dataset["episode_index"])
            boundaries = np.where(np.diff(episode_indices) != 0)[0] + 1
            starts = np.concatenate(([0], boundaries))
            ends = np.concatenate((boundaries, [len(hf_dataset)]))

            current_episode_idx = 0
            chunk_idx = 0
            file_idx = 0
            while current_episode_idx < len(starts):
                shard_start = starts[current_episode_idx]
                shard_end = ends[current_episode_idx]
                next_episode_idx = current_episode_idx + 1
                while next_episode_idx < len(starts):
                    candidate_end = ends[next_episode_idx]
                    candidate = hf_dataset.select(range(shard_start, candidate_end))
                    if get_hf_dataset_size_in_mb(candidate) > data_file_size_mb:
                        break
                    shard_end = candidate_end
                    next_episode_idx += 1

                shard = hf_dataset.select(range(shard_start, shard_end))
                path = tmp_root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
                path.parent.mkdir(parents=True, exist_ok=True)
                shard.to_parquet(path)
                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, chunk_folder_size)
                current_episode_idx = next_episode_idx

        target_dir = dataset_root / "data"
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.move(str(tmp_root / "data"), str(target_dir))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def _update_dataset_metadata(
    dataset_root: Path,
    *,
    label_field: str,
    summary: CFGRLLabelingSummary,
) -> None:
    """Persist label schema and labeling metadata alongside the dataset."""

    info = load_info(dataset_root)
    info["features"][label_field] = {"dtype": "int64", "shape": (1,), "names": None}
    info.setdefault("cfgrl_labeling", {})[label_field] = asdict(summary)
    write_info(info, dataset_root)

    meta_path = dataset_root / CFGRL_LABELING_META_PATH
    metadata = load_json(meta_path) if meta_path.exists() else {}
    metadata[label_field] = asdict(summary)
    write_json(metadata, meta_path)

    stats = load_stats(dataset_root) or {}
    stats[label_field] = {
        "min": [0],
        "max": [1],
        "mean": [summary.positive_fraction],
        "std": [float(np.sqrt(summary.positive_fraction * (1.0 - summary.positive_fraction)))],
        "count": [summary.num_rows],
    }
    write_stats(stats, dataset_root)


def label_chunk_dataset(
    critic: CFGRLCritic,
    dataset_root: str | Path,
    *,
    label_field: str = "cfgrl_optimal",
    percentile: float = 70.0,
    batch_size: int = 64,
    action_provider=None,
    device: str = "cpu",
    condition: int | None = None,
    guidance_scale: float | None = None,
) -> CFGRLLabelingSummary:
    """Label a chunk-aligned dataset with a critic-derived optimality bit.

    The dataset rows must already be aligned to the critic's chunk semantics and
    must expose either nested ``state`` columns or flat ``state.<feature>`` ones.
    """

    dataset_root = Path(dataset_root)
    hf_dataset, info = load_local_chunk_dataset(dataset_root)
    provider = action_provider
    provider_name = type(provider).__name__ if provider is not None else "DatasetActionProvider"
    if provider is None:
        if "policy_action_samples" in hf_dataset.column_names or "next_policy_action_samples" in hf_dataset.column_names:
            provider = CachedActionProvider()
            provider_name = type(provider).__name__
        else:
            provider = DatasetActionProvider()
    advantages = _compute_advantages(
        critic,
        hf_dataset,
        batch_size=batch_size,
        action_provider=provider,
        device=device,
    )
    labels, summary = summarize_advantages(
        advantages,
        percentile=percentile,
        chunk_size=critic.config.chunk_size,
        label_field=label_field,
        provider=provider_name,
        condition=condition,
        guidance_scale=guidance_scale,
    )

    if label_field in hf_dataset.column_names:
        hf_dataset = hf_dataset.remove_columns(label_field)
    labeled_dataset = hf_dataset.add_column(label_field, labels.tolist())
    _write_chunked_dataset(
        labeled_dataset,
        dataset_root,
        data_file_size_mb=float(info["data_files_size_in_mb"]),
        chunk_folder_size=int(info["chunks_size"]),
    )
    _update_dataset_metadata(dataset_root, label_field=label_field, summary=summary)
    return summary


def _build_action_provider(
    *,
    policy_path: str | None,
    condition: int | None,
    guidance_scale: float | None,
) -> Any | None:
    """Optionally build a policy action provider for critic-side baselines."""

    if policy_path is None:
        return None
    policy = CFGRLPolicy.from_pretrained(policy_path, local_files_only=True)
    policy.eval()
    return PolicyActionProvider(policy=policy, condition=condition, guidance_scale=guidance_scale)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for chunk-dataset labeling."""

    parser = argparse.ArgumentParser(description="Label a chunk-aligned CFGRL dataset with critic advantages.")
    parser.add_argument("--dataset-root", required=True, help="Path to the local dataset root containing meta/ and data/.")
    parser.add_argument("--critic-path", required=True, help="Path to a saved CFGRL critic checkpoint.")
    parser.add_argument("--policy-path", default=None, help="Optional reference policy checkpoint used for V(s) baselines.")
    parser.add_argument("--label-field", default="cfgrl_optimal", help="Dataset field name written for the optimality bit.")
    parser.add_argument("--percentile", type=float, default=70.0, help="Percentile threshold for positive labels, e.g. 70 = top 30%.")
    parser.add_argument("--batch-size", type=int, default=64, help="Number of chunk rows processed per critic forward pass.")
    parser.add_argument("--device", default="cpu", help="Torch device for the critic and optional policy.")
    parser.add_argument("--condition", type=int, default=None, help="Optional policy condition used when sampling baseline actions.")
    parser.add_argument("--guidance-scale", type=float, default=None, help="Optional CFG scale used when sampling baseline actions.")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for critic-driven dataset labeling."""

    args = parse_args()
    critic = CFGRLCritic.from_pretrained(args.critic_path, local_files_only=True)
    critic.eval()
    provider = _build_action_provider(
        policy_path=args.policy_path,
        condition=args.condition,
        guidance_scale=args.guidance_scale,
    )
    summary = label_chunk_dataset(
        critic,
        args.dataset_root,
        label_field=args.label_field,
        percentile=args.percentile,
        batch_size=args.batch_size,
        action_provider=provider,
        device=args.device,
        condition=args.condition,
        guidance_scale=args.guidance_scale,
    )
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
