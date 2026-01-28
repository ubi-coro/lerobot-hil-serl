#!/usr/bin/env python3
"""
Precompute XVLA VLM embeddings + policy action-chunk samples for each frame in a LeRobot dataset,
and store them into a *copied* dataset (postfixed name).

Design goals:
- LeRobot style: reuse TrainPipelineConfig + make_dataset/make_policy/make_pre_post_processors
  the same way lerobot_train.py does.
- Cache format matches XPVLA critic expectations:
    - observation.vlm_cache contains the *raw* forward_vlm output dict (vlm_features, aux_visual_inputs, ...)
    - observation.policy_actions contains [K,H,A] policy samples per frame
  (see XPVLACriticBackboneConfig keys and CriticBackbone cache semantics).

Notes:
- This script assumes a local on-disk dataset directory (not streaming).
- It creates a full copy of the dataset directory and adds new arrays/groups inside the Zarr store.
"""

import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import OBS_IMAGES, OBS_LANGUAGE_TOKENS, OBS_PREFIX
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging

try:
    import zarr
except Exception as e:
    raise ImportError("This script requires zarr. Install it (e.g. `pip install zarr`).") from e


# -------------------------
# Config
# -------------------------

@dataclass
class PrecomputeCacheConfig(TrainPipelineConfig):
    """
    Extends TrainPipelineConfig with cache-specific options.

    You should be able to pass your normal training YAML, and override only the fields below.
    """
    cache_postfix: str = "_cached"
    cache_overwrite: bool = False

    # What to cache
    cache_vlm: bool = True
    cache_policy_actions: bool = True

    # Policy action sampling
    cache_num_action_samples: int = 8  # K
    cache_action_seed: int = 0

    # Batching (for forward_vlm + action sampling)
    cache_batch_size: int = 32
    cache_num_workers: int = 4

    # Storage
    cache_dtype: str = "float16"  # float16 is usually fine for caching; use float32 if you want exactness
    cache_compressor: str = "zstd"  # zstd|blosc|none
    cache_zarr_chunksize: int = 1024  # chunk along frame dimension

    # Naming inside dataset
    vlm_cache_key: str = f"{OBS_PREFIX}.vlm_cache"
    policy_actions_key: str = f"{OBS_PREFIX}.policy_actions"


# -------------------------
# Helpers: dataset path + zarr
# -------------------------

def _infer_dataset_root_dir(dataset: Any) -> Path:
    """
    Try to infer the *local on-disk directory* that holds the dataset.
    We need this to copy the dataset and then open its Zarr store for writing.
    """
    # Common attribute patterns in LeRobot-like datasets
    for attr in ["root", "root_dir", "dataset_dir", "path", "_root", "_dataset_dir"]:
        if hasattr(dataset, attr):
            v = getattr(dataset, attr)
            if isinstance(v, (str, os.PathLike)):
                p = Path(v)
                if p.exists():
                    return p

    # Try meta if present
    meta = getattr(dataset, "meta", None)
    if meta is not None:
        for attr in ["root", "root_dir", "dataset_dir", "path"]:
            if hasattr(meta, attr):
                v = getattr(meta, attr)
                if isinstance(v, (str, os.PathLike)):
                    p = Path(v)
                    if p.exists():
                        return p

    raise RuntimeError(
        "Could not infer dataset root directory from dataset object. "
        "Please add a small adapter in _infer_dataset_root_dir() for your dataset class."
    )


def _postfix_dataset_dir(src_dir: Path, postfix: str) -> Path:
    """
    Create a target directory name by appending postfix to the last path component.
    """
    name = src_dir.name + postfix
    return src_dir.parent / name


def _copy_dataset_dir(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Target dataset dir already exists: {dst}")
        shutil.rmtree(dst)
    logging.info("Copying dataset dir: %s -> %s", src, dst)
    shutil.copytree(src, dst)


def _find_zarr_store_dir(dataset_dir: Path) -> Path:
    """
    Find a *.zarr directory inside the dataset directory.
    If there are multiple, prefer 'data.zarr' if present.
    """
    candidates = list(dataset_dir.glob("**/*.zarr"))
    if not candidates:
        raise FileNotFoundError(f"No .zarr store found under dataset dir: {dataset_dir}")
    for c in candidates:
        if c.name == "data.zarr":
            return c
    # Otherwise pick the shallowest
    candidates.sort(key=lambda p: (len(p.parts), str(p)))
    return candidates[0]


def _make_compressor(name: str):
    name = (name or "zstd").lower()
    if name == "none":
        return None
    # Zarr v2 compressor API commonly uses numcodecs
    try:
        import numcodecs
    except Exception:
        raise ImportError("numcodecs is required for compression; install with `pip install numcodecs`.")

    if name == "zstd":
        return numcodecs.Zstd(level=3)
    if name == "blosc":
        return numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.SHUFFLE)
    raise ValueError(f"Unknown compressor: {name}")


def _zarr_require_array(
    grp: zarr.Group,
    path: str,
    shape: Tuple[int, ...],
    dtype: np.dtype,
    chunks: Tuple[int, ...],
    compressor,
    overwrite: bool,
):
    if path in grp:
        if not overwrite:
            raise FileExistsError(f"Zarr array already exists and overwrite=False: {path}")
        del grp[path]
    return grp.require_dataset(
        path,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        compressor=compressor,
        overwrite=True,
    )


# -------------------------
# Helpers: policy + VLM forward
# -------------------------

@torch.no_grad()
def _forward_vlm(policy: Any, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Run XVLA forward_vlm on a *preprocessed* batch (already tokenized, images normalized, on device).
    Returns the raw dict of tensors (vlm_features, aux_visual_inputs, etc).
    """
    # Collect images like CriticBackbone does (all keys starting with OBS_IMAGES)
    pixel_values = {k: v for k, v in batch.items() if isinstance(k, str) and k.startswith(OBS_IMAGES)}
    if not pixel_values:
        raise ValueError("No image keys found in batch for forward_vlm. Expected keys starting with OBS_IMAGES.")

    input_ids = batch.get(OBS_LANGUAGE_TOKENS, None)
    if input_ids is None:
        raise ValueError(f"Missing {OBS_LANGUAGE_TOKENS} in batch for forward_vlm.")

    image_mask = batch.get("image_mask", None)
    if image_mask is None:
        # [B, num_cams] all present
        B = input_ids.shape[0]
        image_mask = torch.ones((B, len(pixel_values)), device=input_ids.device, dtype=torch.bool)

    # XVLA policies usually expose `policy.model.forward_vlm(...)`
    if not hasattr(policy, "model") or not hasattr(policy.model, "forward_vlm"):
        raise AttributeError("Policy does not expose policy.model.forward_vlm; cannot compute VLM cache.")

    enc = policy.model.forward_vlm(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_mask=image_mask,
    )
    # Keep only tensors (and detach)
    out: Dict[str, torch.Tensor] = {}
    for k, v in enc.items():
        if torch.is_tensor(v):
            out[k] = v.detach()
    return out


@torch.no_grad()
def _sample_action_chunks(policy: Any, batch: Dict[str, Any], K: int, seed: int) -> torch.Tensor:
    """
    Sample K action chunks from the policy for each element in the batch.
    Output: [B, K, H, A]
    """
    # We rely on XVLA-style internal action chunk getter (XPVLAPolicy overrides _get_action_chunk)
    if not hasattr(policy, "_get_action_chunk"):
        raise AttributeError("Policy has no _get_action_chunk(batch) method; cannot sample action chunks.")

    # Control RNG for reproducibility
    # (We only seed the per-call generator; the rest of the program remains unaffected.)
    g = torch.Generator(device=batch[OBS_LANGUAGE_TOKENS].device)
    g.manual_seed(int(seed))

    chunks = []
    for i in range(K):
        # Make sampling differ per i deterministically
        # (XVLA sampling uses torch.randn internally; setting global seed is easiest.)
        torch.manual_seed(int(seed) + i)
        a = policy._get_action_chunk(batch)  # expected [B, H, A]
        if a.ndim != 3:
            raise ValueError(f"Expected action chunk [B,H,A], got {tuple(a.shape)}")
        chunks.append(a.detach())
    return torch.stack(chunks, dim=1)  # [B,K,H,A]


def _to_numpy(x: torch.Tensor, dtype: str) -> np.ndarray:
    if dtype == "float16":
        x = x.to(torch.float16)
    elif dtype == "float32":
        x = x.to(torch.float32)
    else:
        raise ValueError(f"Unsupported cache_dtype: {dtype}")
    return x.detach().cpu().numpy()


# -------------------------
# Main
# -------------------------

@parser.wrap()
def main(cfg: PrecomputeCacheConfig):
    register_third_party_plugins()
    init_logging()
    logging.info("Precompute cache config:\n%s", cfg.to_dict() if hasattr(cfg, "to_dict") else cfg)

    cfg.validate()

    # Create dataset (must be non-streaming for copying)
    dataset = make_dataset(cfg)
    if getattr(cfg.dataset, "streaming", False):
        raise ValueError("cache script requires a local dataset (streaming=False).")

    src_dir = _infer_dataset_root_dir(dataset)
    dst_dir = _postfix_dataset_dir(src_dir, cfg.cache_postfix)
    _copy_dataset_dir(src_dir, dst_dir, overwrite=cfg.cache_overwrite)

    # Open Zarr store inside the copied dataset
    zarr_dir = _find_zarr_store_dir(dst_dir)
    logging.info("Using Zarr store: %s", zarr_dir)
    root = zarr.open_group(str(zarr_dir), mode="a")

    # Re-create policy (like lerobot_train.py) and processors
    # We need ds_meta + ds_stats to build processors
    ds_meta = getattr(dataset, "meta", None)
    if ds_meta is None:
        raise RuntimeError("Dataset has no .meta; cannot build policy/processors.")

    ds_stats = getattr(dataset, "stats", None)
    if ds_stats is None:
        # common pattern: meta.stats
        ds_stats = getattr(ds_meta, "stats", None)

    policy = make_policy(cfg=cfg.policy, ds_meta=ds_meta, rename_map=cfg.rename_map)
    policy.eval()

    # Create processors (same pattern as train.py when loading from checkpoint)
    processor_kwargs = {}
    postprocessor_kwargs = {}
    processor_kwargs["dataset_stats"] = ds_stats

    if cfg.policy.pretrained_path is not None:
        # Mirror lerobot_train behavior: override device + normalizer stats/features/norm_map
        device = cfg.policy.device if hasattr(cfg.policy, "device") else "cuda"
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device},
            "normalizer_processor": {
                "stats": ds_stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": ds_stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, _post = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    # Device
    device = torch.device(policy.config.device) if hasattr(policy, "config") and hasattr(policy.config, "device") else torch.device("cuda")
    policy.to(device)

    # Frame count
    N = int(getattr(dataset, "num_frames", len(dataset)))
    logging.info("Dataset frames: %d", N)

    # Prepare Zarr outputs
    compressor = _make_compressor(cfg.cache_compressor)
    chunk0 = min(int(cfg.cache_zarr_chunksize), N)

    # We’ll lazily allocate arrays once we know shapes (from first batch).
    vlm_arrays: Dict[str, Any] = {}
    actions_arr = None

    # Simple sequential batching over indices (keeps deterministic ordering)
    bs = int(cfg.cache_batch_size)
    torch.set_grad_enabled(False)

    # Optional: write cache metadata
    cache_info = {
        "cache_postfix": cfg.cache_postfix,
        "vlm_cache_key": cfg.vlm_cache_key,
        "policy_actions_key": cfg.policy_actions_key,
        "num_action_samples": int(cfg.cache_num_action_samples),
        "cache_dtype": cfg.cache_dtype,
    }
    (dst_dir / "cache_info.json").write_text(json.dumps(cache_info, indent=2))

    for start in range(0, N, bs):
        end = min(N, start + bs)
        idxs = list(range(start, end))

        # Collate a batch manually (dataset[i] expected to return dict[str, Tensor/np/...])
        samples = [dataset[i] for i in idxs]

        # Minimal collation: stack torch tensors; keep non-tensors as list
        batch: Dict[str, Any] = {}
        keys = set()
        for s in samples:
            keys |= set(s.keys())
        for k in keys:
            vals = [s.get(k, None) for s in samples]
            if all(torch.is_tensor(v) for v in vals if v is not None):
                # stack with None -> raise (schema mismatch)
                if any(v is None for v in vals):
                    raise ValueError(f"Missing key {k} in some samples during collation.")
                batch[k] = torch.stack(vals, dim=0)
            else:
                batch[k] = vals

        # Preprocess (tokenize + image normalization + device + normalization)
        batch = preprocessor(batch)

        # Compute caches
        vlm_dict: Optional[Dict[str, torch.Tensor]] = None
        if cfg.cache_vlm:
            vlm_dict = _forward_vlm(policy, batch)

        actions: Optional[torch.Tensor] = None
        if cfg.cache_policy_actions:
            actions = _sample_action_chunks(
                policy, batch, K=int(cfg.cache_num_action_samples), seed=int(cfg.cache_action_seed) + start
            )

        # Lazily allocate Zarr arrays on first iteration
        if start == 0:
            if cfg.cache_vlm and vlm_dict is not None:
                # Create a group for the vlm_cache dict
                grp_vlm = root.require_group(cfg.vlm_cache_key.replace(".", "/"))
                for name, t in vlm_dict.items():
                    np_t = _to_numpy(t, cfg.cache_dtype)
                    shape = (N,) + np_t.shape[1:]  # [N, ...]
                    chunks = (chunk0,) + np_t.shape[1:]
                    vlm_arrays[name] = _zarr_require_array(
                        grp_vlm,
                        name,
                        shape=shape,
                        dtype=np_t.dtype,
                        chunks=chunks,
                        compressor=compressor,
                        overwrite=cfg.cache_overwrite,
                    )
                    logging.info("Alloc VLM cache array: %s/%s shape=%s dtype=%s", grp_vlm.path, name, shape, np_t.dtype)

            if cfg.cache_policy_actions and actions is not None:
                grp_obs = root.require_group(OBS_PREFIX)
                np_a = _to_numpy(actions, cfg.cache_dtype)  # [B,K,H,A]
                shape = (N,) + np_a.shape[1:]  # [N,K,H,A]
                chunks = (chunk0,) + np_a.shape[1:]
                key = cfg.policy_actions_key.split(".", 1)[1] if cfg.policy_actions_key.startswith(f"{OBS_PREFIX}.") else cfg.policy_actions_key
                actions_arr = _zarr_require_array(
                    grp_obs,
                    key,
                    shape=shape,
                    dtype=np_a.dtype,
                    chunks=chunks,
                    compressor=compressor,
                    overwrite=cfg.cache_overwrite,
                )
                logging.info("Alloc policy_actions array: %s/%s shape=%s dtype=%s", grp_obs.path, key, shape, np_a.dtype)

        # Write slices
        sl = slice(start, end)
        if cfg.cache_vlm and vlm_dict is not None:
            for name, t in vlm_dict.items():
                vlm_arrays[name][sl] = _to_numpy(t, cfg.cache_dtype)

        if cfg.cache_policy_actions and actions is not None:
            assert actions_arr is not None
            actions_arr[sl] = _to_numpy(actions, cfg.cache_dtype)

        if (start // bs) % 20 == 0 or end == N:
            logging.info("Cached frames %d..%d / %d", start, end, N)

    logging.info("Done. Cached dataset written to: %s", dst_dir)


if __name__ == "__main__":
    main()
