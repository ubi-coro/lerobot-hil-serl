import argparse
import time
from contextlib import nullcontext
from dataclasses import dataclass

import einops
import hydra
import numpy as np
import gymnasium as gym
import torch
import logging
import sys
import time
from threading import Lock
from typing import Annotated, Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import torch
import torchvision.transforms.functional as F  # noqa: N812

from lerobot.common.envs.configs import EnvConfig
from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.robot_devices.control_utils import (
    busy_wait,
    is_headless,
    reset_follower_position,
)
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.common.utils.utils import log_say
from lerobot.configs import parser
from lerobot.scripts.server.kinematics import RobotKinematics
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.server.maniskill_manipulator import make_maniskill
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.control_configs import ControlConfig
from lerobot.common.envs.configs import EnvConfig

@dataclass
class RecordControlConfig:
    repo_id: str
    root: str
    task: str
    env: EnvConfig

    fps: int = 10
    num_episodes: int = 20
    control_time_s: int = 10


@parser.wrap()
def record(cfg: RecordControlConfig):
    """
    Record a dataset of robot interactions using either a policy or teleop.

    cfg.
        env: The environment to record from
        repo_id: Repository ID for dataset storage
        root: Local root directory for dataset (optional)
        num_episodes: Number of episodes to record
        control_time_s: Maximum episode length in seconds
        fps: Frames per second for recording
        push_to_hub: Whether to push dataset to Hugging Face Hub
        task_description: Description of the task being recorded
        policy: Optional policy to generate actions (if None, uses teleop)
    """


    env = make_maniskill(
        cfg.env,
        n_envs=1
    )

    # Setup initial action (zero action if using teleop)
    dummy_action = env.action_space.sample()
    dummy_action = (dummy_action[0] * 0.0, True)
    action = dummy_action

    dummy_obs, info = env.reset()

    # Configure dataset features based on environment spaces
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": dummy_obs["observation.state"].squeeze().cpu().numpy().shape,
            "names": None,
        },
        "action": {
            "dtype": "float32",
            "shape": env.action_space[0].shape,
            "names": None,
        },
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
    }

    # Add image features
    for key in dummy_obs:
        if "image" in key:
            features[key] = {
                "dtype": "video",
                "shape": dummy_obs[key].squeeze().cpu().numpy().shape,
                "names": None,
            }

    # Create dataset
    dataset = LeRobotDataset.create(
        cfg.repo_id,
        cfg.fps,
        root=cfg.root,
        use_videos=True,
        image_writer_threads=4,
        image_writer_processes=0,
        features=features,
    )

    # Record episodes
    episode_index = 0
    while episode_index < cfg.num_episodes:
        obs, _ = env.reset()
        start_episode_t = time.perf_counter()
        log_say(f"Recording episode {episode_index}", play_sounds=True)

        # Run episode steps
        while time.perf_counter() - start_episode_t < cfg.control_time_s:
            start_loop_t = time.perf_counter()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            env.render(          )

            # Check if episode needs to be rerecorded
            if info.get("rerecord_episode", False):
                break

            # For teleop, get action from intervention
            if "is_intervention" in info and info["is_intervention"]:
                recorded_action = info["action_intervention"].cpu().float()
            else:
                recorded_action = action[0]

            recorded_action = {
                "action": recorded_action
            }

            # Process observation for dataset
            obs = {k: v.cpu().squeeze(0).float() for k, v in obs.items()}

            # Add frame to dataset
            frame = {**obs, **recorded_action}
            frame["next.reward"] = np.array([reward], dtype=np.float32)
            frame["next.done"] = np.array([terminated or truncated], dtype=bool)
            frame["task"] = cfg.task
            dataset.add_frame(frame)

            # Maintain consistent timing
            if cfg.fps:
                dt_s = time.perf_counter() - start_loop_t
                busy_wait(1 / cfg.fps - dt_s)

            if terminated or truncated:
                break

        # Handle episode recording
        if info.get("rerecord_episode", False):
            dataset.clear_episode_buffer()
            logging.info(f"Re-recording episode {episode_index}")
            continue

        dataset.save_episode()
        episode_index += 1

        time.sleep(0.5)

if __name__ == "__main__":
    record()