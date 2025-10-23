import types
import typing
from typing import get_origin, get_args

from lerobot.configs.types import PolicyFeature, FeatureType, PipelineFeatureType
from lerobot.datasets.pipeline_features import create_initial_features, strip_prefix, PREFIXES_TO_STRIP
from lerobot.envs.robot_env import RobotEnv
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


def is_union_with_dict(field_type) -> bool:
    origin = get_origin(field_type)
    if origin is types.UnionType or origin is typing.Union:
        return any(get_origin(arg) is dict for arg in get_args(field_type))
    return False

