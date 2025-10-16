from copy import deepcopy

import gymnasium

from lerobot.configs.types import PolicyFeature, FeatureType
from lerobot.envs.robot_env import RobotEnv
from lerobot.processor import DataProcessorPipeline
from lerobot.utils.constants import ACTION


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
        ACTION: {
            "dtype": "float32",
            "shape": (action_dim,)
        },
        OBS_STATE: {
            "dtype": "float32",
            "shape": (obs_features[OBS_STATE].shape,)
        }
    }

    # add visual features
    for key, ft in obs_features:
        if ft.type == FeatureType.VISUAL:
            key = strip_prefix(key, PREFIXES_TO_STRIP)
            features[f"{OBS_IMAGES}.{key}"] = {
                "dtype": "video" if use_video else "image",
                "shape": ft.shape,
                "names": ["height", "width", "channels"],
            }

    return features

