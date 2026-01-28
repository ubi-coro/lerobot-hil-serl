#!/usr/bin/env python3
"""
One-time conversion script to:
1) Load a pretrained XVLA checkpoint (Hub repo id or local folder).
2) Overwrite its input feature spec + image view config using a target dataset's features.
3) Rebuild processors with dataset stats (normalizer fix).
4) Re-save the checkpoint as an XPVLA policy checkpoint.

Key feature:
- Force a specific camera key to be the FIRST image view (index 0), so it gets fused with language.
  This is done by:
    (a) putting that key first in cfg.image_features (if present), and
    (b) reordering cfg.input_features so that key is inserted first among image features.

Config:
- Uses draccus via lerobot.configs.parser.wrap() pattern (same style as your cache script),
  and subclasses TrainPipelineConfig to reuse dataset/policy args consistently.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import init_logging

logger = logging.getLogger("convert_xvla_checkpoint")


# -------------------------
# Config
# -------------------------

@dataclass
class ConvertXVLAConfig(TrainPipelineConfig):
    """
    Extends TrainPipelineConfig with conversion-specific options.

    You should be able to pass your normal training YAML (dataset/policy/etc),
    and override only these fields on the command line.

    Example:
      python lerobot/scripts/convert_xvla_checkpoint.py \
        --config path/to/train.yaml \
        pretrained_id=org/xvla-checkpoint \
        out_dir=/tmp/xpvla_converted \
        fused_camera_key=observation.rgb.cam_global \
        overwrite=true
    """
    pretrained_id: str = ""
    out_dir: str = ""
    overwrite: bool = False
    verbose: bool = False

    # Optional overrides
    force_num_views: int | None = None

    # Camera that should be view-0 / fused with language.
    # Must match the dataset/policy image key exactly (the key used in batch dicts).
    fused_camera_key: str = ""


# -------------------------
# Helpers
# -------------------------

def _safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _infer_present_image_keys_from_features(features: dict[str, Any]) -> list[str]:
    """
    Conservative heuristic, identical spirit to your original script.
    Replace with your dataset meta's explicit image feature list if available.
    """
    keys = list(features.keys())
    image_like: list[str] = []
    for k in keys:
        lk = k.lower()
        if ("rgb" in lk or "image" in lk or "camera" in lk) and ("observation" in lk or "obs" in lk):
            image_like.append(k)
    return image_like


def _reorder_image_keys(image_keys: list[str], fused_key: str) -> list[str]:
    """
    Ensure fused_key is the first element in image_keys (if provided and present).
    """
    if not fused_key:
        return image_keys
    if fused_key not in image_keys:
        logger.warning(
            "Requested fused_camera_key=%s not found among inferred image keys (%d). "
            "Leaving image order unchanged.",
            fused_key,
            len(image_keys),
        )
        return image_keys
    return [fused_key] + [k for k in image_keys if k != fused_key]


def _reorder_input_features_for_fused_camera(
    input_features: dict[str, Any],
    image_keys_ordered: list[str],
) -> dict[str, Any]:
    """
    Return a NEW dict with insertion order:
      1) ordered image keys (subset of input_features)
      2) all remaining keys in original order
    """
    out: dict[str, Any] = {}

    # 1) image keys in desired order
    for k in image_keys_ordered:
        if k in input_features:
            out[k] = input_features[k]

    # 2) rest in original order
    for k, v in input_features.items():
        if k not in out:
            out[k] = v

    return out


def _overwrite_xvla_config_from_dataset(
    cfg: PreTrainedConfig,
    ds_meta: Any,
    *,
    force_num_views: int | None,
    fused_camera_key: str,
) -> PreTrainedConfig:
    """
    Overwrite policy feature spec based on dataset features and enforce image ordering.
    """
    # 1) Policy input/output features
    features = dataset_to_policy_features(ds_meta.features)
    cfg.output_features = {k: ft for k, ft in features.items() if getattr(ft, "type", None).name == "ACTION"}
    cfg.input_features = {k: ft for k, ft in features.items() if k not in cfg.output_features}

    # 2) XVLA-specific image config knobs (best-effort; only if present on cfg)
    image_keys = _infer_present_image_keys_from_features(cfg.input_features)
    image_keys = _reorder_image_keys(image_keys, fused_camera_key)

    # Make sure the fused camera is also first in input_features insertion order (if present)
    cfg.input_features = _reorder_input_features_for_fused_camera(cfg.input_features, image_keys)

    if hasattr(cfg, "image_features"):
        # XVLA typically expects an ordered iterable of keys.
        try:
            cfg.image_features = image_keys
        except Exception:
            cfg.image_features = {k: cfg.input_features[k] for k in image_keys if k in cfg.input_features}

    if hasattr(cfg, "num_image_views"):
        if force_num_views is not None:
            cfg.num_image_views = int(force_num_views)
        else:
            cfg.num_image_views = max(1, len(image_keys))

    # 3) Log sequence safety for aux views
    if hasattr(cfg, "max_len_seq"):
        num_views = getattr(cfg, "num_image_views", None) or max(1, len(image_keys))
        logger.info(
            "XVLA config after overwrite: fused_camera_key=%s | %d image keys | num_image_views=%s | max_len_seq=%s. "
            "Remember aux_visual_inputs length grows with (num_views-1).",
            fused_camera_key or "<unset>",
            len(image_keys),
            str(num_views),
            str(getattr(cfg, "max_len_seq", None)),
        )

    return cfg


def _copy_weights(src_policy: torch.nn.Module, dst_policy: torch.nn.Module) -> None:
    src_sd = src_policy.state_dict()
    missing, unexpected = dst_policy.load_state_dict(src_sd, strict=False)

    if missing:
        logger.warning("Missing keys when loading into destination policy (%d): %s", len(missing), missing[:20])
    if unexpected:
        logger.warning("Unexpected keys when loading into destination policy (%d): %s", len(unexpected), unexpected[:20])

    if len(missing) > 0:
        logger.warning("Destination policy did not receive all parameters. Verify XPVLA wrapper compatibility.")


# -------------------------
# Main
# -------------------------

@parser.wrap()
def main(cfg: ConvertXVLAConfig) -> None:
    register_third_party_plugins()

    # Logging
    init_logging()
    if cfg.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not cfg.pretrained_id:
        raise ValueError("cfg.pretrained_id is required.")
    if not cfg.out_dir:
        raise ValueError("cfg.out_dir is required.")
    if not cfg.fused_camera_key:
        logger.warning(
            "cfg.fused_camera_key is empty. No camera will be forced to view-0. "
            "If you rely on 'view-0 fused with language', set fused_camera_key explicitly."
        )

    out_dir = Path(cfg.out_dir)
    if out_dir.exists():
        if not cfg.overwrite:
            raise FileExistsError(f"{out_dir} exists. Set overwrite=true to replace it.")
        _safe_rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Load dataset (meta + stats)
    # ------------------------------------------------------------
    logger.info("Creating dataset via make_dataset(...) to fetch meta/stats.")
    dataset = make_dataset(cfg)
    ds_meta = getattr(dataset, "meta", None)
    if ds_meta is None:
        raise RuntimeError("Dataset has no .meta; cannot derive features/stats.")
    ds_stats = getattr(dataset, "stats", None)
    if ds_stats is None:
        ds_stats = getattr(ds_meta, "stats", None)
        if ds_stats is None:
            logger.warning("Dataset has no stats (dataset.stats or dataset.meta.stats). Processor rebuild may be incomplete.")

    # ------------------------------------------------------------
    # Load pretrained XVLA config and overwrite from dataset
    # ------------------------------------------------------------
    logger.info("Loading pretrained config from: %s", cfg.pretrained_id)
    base_cfg = PreTrainedConfig.from_pretrained(cfg.pretrained_id)

    # Prefer device from TrainPipelineConfig policy if present
    if hasattr(cfg, "policy") and hasattr(cfg.policy, "device") and cfg.policy.device is not None:
        base_cfg.device = cfg.policy.device

    base_cfg = _overwrite_xvla_config_from_dataset(
        base_cfg,
        ds_meta,
        force_num_views=cfg.force_num_views,
        fused_camera_key=cfg.fused_camera_key,
    )

    # Important: instantiate from weights
    base_cfg.pretrained_path = cfg.pretrained_id

    # ------------------------------------------------------------
    # Instantiate XVLA policy (with overwritten feature spec)
    # ------------------------------------------------------------
    logger.info("Instantiating XVLA policy from pretrained weights with overwritten feature spec.")
    xvla_policy = make_policy(base_cfg, ds_meta=ds_meta)

    # ------------------------------------------------------------
    # Build processors from dataset stats (normalizer fix)
    # ------------------------------------------------------------
    logger.info("Building pre/post processors using dataset stats (normalizer fix).")
    preproc, postproc = make_pre_post_processors(
        policy_cfg=base_cfg,
        pretrained_path=None,  # force rebuild rather than loading checkpoint processors
        dataset_stats=ds_stats,
        dataset_meta=ds_meta,
    )

    # ------------------------------------------------------------
    # Map XVLA -> XPVLA config and policy
    # ------------------------------------------------------------
    logger.info("Mapping XVLA config -> XPVLA policy config.")
    try:
        from lerobot.policies.xpvla_policy.configuration_xpvla_policy import XPVLAPolicyConfig
    except Exception as e:
        raise ImportError("Could not import XPVLAPolicyConfig. Fix the import path to your XPVLA policy config.") from e

    # Serialize config
    if hasattr(base_cfg, "model_dump"):
        cfg_dict = base_cfg.model_dump()
    elif hasattr(base_cfg, "to_dict"):
        cfg_dict = base_cfg.to_dict()
    else:
        cfg_path = Path(cfg.pretrained_id) / "config.json"
        if cfg_path.exists():
            cfg_dict = json.loads(cfg_path.read_text())
        else:
            raise RuntimeError("Cannot serialize base_cfg. Implement a serializer for your config class.")

    cfg_dict["type"] = "xpvla_policy"
    cfg_dict["pretrained_path"] = None
    cfg_dict["device"] = getattr(base_cfg, "device", None)

    # Construct XPVLA config (filter unknown keys if needed)
    try:
        xpvla_cfg = XPVLAPolicyConfig(**cfg_dict)
    except TypeError:
        allowed = set(getattr(XPVLAPolicyConfig, "__annotations__", {}).keys())
        filtered = {k: v for k, v in cfg_dict.items() if k in allowed}
        xpvla_cfg = XPVLAPolicyConfig(**filtered)

    # Ensure dataset-derived feature spec and image ordering are carried over
    xpvla_cfg.input_features = base_cfg.input_features
    xpvla_cfg.output_features = base_cfg.output_features
    if hasattr(base_cfg, "image_features") and hasattr(xpvla_cfg, "image_features"):
        xpvla_cfg.image_features = getattr(base_cfg, "image_features")
    if hasattr(base_cfg, "num_image_views") and hasattr(xpvla_cfg, "num_image_views"):
        xpvla_cfg.num_image_views = getattr(base_cfg, "num_image_views")

    logger.info("Instantiating fresh XPVLA policy and copying weights from XVLA.")
    xpvla_policy = make_policy(xpvla_cfg, ds_meta=ds_meta)
    _copy_weights(xvla_policy, xpvla_policy)

    # ------------------------------------------------------------
    # Save checkpoint folder
    # ------------------------------------------------------------
    logger.info("Saving converted policy to: %s", str(out_dir))
    xpvla_policy.save_pretrained(out_dir)

    logger.info("Saving rebuilt processors to: %s", str(out_dir))
    preproc.save_pretrained(out_dir, config_filename="preprocessor.json")
    postproc.save_pretrained(out_dir, config_filename="postprocessor.json")

    manifest = {
        "source_pretrained_id": cfg.pretrained_id,
        "dataset_repo_id": getattr(getattr(cfg, "dataset", None), "repo_id", None),
        "dataset_root": getattr(getattr(cfg, "dataset", None), "root", None),
        "force_num_views": cfg.force_num_views,
        "device": getattr(base_cfg, "device", None),
        "fused_camera_key": cfg.fused_camera_key,
        "notes": (
            "Converted XVLA->XPVLA, overwrote features+image config from dataset, "
            "reordered image keys so fused_camera_key is view-0, rebuilt processors with dataset stats."
        ),
    }
    (out_dir / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2))

    logger.info("Done.")


if __name__ == "__main__":
    main()
