from typing import Any

import numpy as np
import torch

from lerobot.configs.types import PolicyFeature, FeatureType, PipelineFeatureType
from lerobot.datasets.factory import IMAGENET_STATS
from lerobot.datasets.pipeline_features import create_initial_features, strip_prefix, PREFIXES_TO_STRIP
from lerobot.envs.robot_env import RobotEnv
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.sac.configuration_sac import is_image_feature

from lerobot.processor import DataProcessorPipeline
from lerobot.utils.constants import ACTION, OBS_STATE, OBS_IMAGES, REWARD, DONE


def get_pipeline_dataset_features(
    env: RobotEnv,
    env_processor: DataProcessorPipeline,
    action_dim: int,
    use_video: bool = True
) -> dict[str, dict]:

    # build initial features from gym spaces
    initial_obs_features = {}
    for name, space in env.observation_space.items():
        initial_obs_features[name] = PolicyFeature(
            type=FeatureType.VISUAL if len(space.shape) == 3 else FeatureType.STATE,
            shape=space.shape,
        )

    # process features with respective pipeline
    obs_features = env_processor.transform_features(create_initial_features(observation=initial_obs_features))[PipelineFeatureType.OBSERVATION]

    # from pipeline features to huggingface features
    features = {
        ACTION: {"dtype": "float32", "shape": (action_dim,)},
        OBS_STATE: {"dtype": "float32", "shape": obs_features[OBS_STATE].shape},
        REWARD: {"dtype": "float32", "shape": (1,), "names": None},
        DONE: {"dtype": "bool", "shape": (1,), "names": None}
    }

    # add visual features
    for key, ft in obs_features.items():
        if ft.type == FeatureType.VISUAL:
            key = strip_prefix(key, PREFIXES_TO_STRIP)
            features[f"{OBS_IMAGES}.{key}"] = {
                "dtype": "video" if use_video else "image",
                "shape": ft.shape,
                "names": ["channels", "height", "width"],
            }

    return features


# todo: finalize this and move to policy factory
def make_rl_policy(policy_cfg: Any | None, env_cfg: Any):
    if policy_cfg is None:
        return None

    from lerobot.policies.factory import make_policy
    from lerobot.policies.sac.modeling_sac import SACPolicy

    policy = make_policy(policy_cfg, env_cfg=env_cfg)

    if not isinstance(policy, SACPolicy):
        raise ValueError("Only sac supported atm")

    return policy


def make_processors_from_stats(cfg, device, stats: dict):
    if hasattr(cfg.env, "stats") and cfg.env.stats is not None:
        for key, env_stats in cfg.env.stats.items():
            stats[key].update({metric: np.array(env_stats[metric]) for metric in env_stats})

    if cfg.dataset.use_imagenet_stats:
        for img_key in [k for k in cfg.policy.input_features if is_image_feature(k)]:
            for stats_type, _stats in IMAGENET_STATS.items():
                stats[img_key][stats_type] = torch.tensor(_stats, dtype=torch.float32)

    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        # Only provide dataset_stats when not resuming from saved processor state
        processor_kwargs["dataset_stats"] = stats

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": stats,
                "features": {**cfg.policy.input_features, **cfg.policy.output_features},
                "norm_map": cfg.policy.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": stats,
                "features": cfg.policy.output_features,
                "norm_map": cfg.policy.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    return preprocessor, postprocessor


