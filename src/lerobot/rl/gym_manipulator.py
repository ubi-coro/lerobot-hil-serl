# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import (
    DataProcessorPipeline,
    EnvTransition,
    TransitionKey,
    create_transition, make_default_processors,
)
from lerobot.processor.hil_processor import TELEOP_ACTION_KEY
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
    so100_follower,
)
from lerobot.robots.robot import Robot
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.constants import ACTION, DONE, OBS_IMAGES, OBS_STATE, REWARD
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

logging.basicConfig(level=logging.INFO)


@dataclass
class DatasetConfig:
    """Configuration for dataset creation and management."""

    repo_id: str
    task: str
    root: str | None = None
    num_episodes_to_record: int = 5
    replay_episode: int | None = None
    push_to_hub: bool = False


@dataclass
class HILSerlRobotEnvConfig:
    pass


@dataclass
class GymManipulatorConfig:
    """Main configuration for gym manipulator environment."""

    env: HILSerlRobotEnvConfig
    dataset: DatasetConfig
    mode: str | None = None  # Either "record", "replay", None
    device: str = "cpu"


def reset_follower_position(robot_arm: Robot, target_position: np.ndarray) -> None:
    """Reset robot arm to target position using smooth trajectory."""
    current_position_dict = robot_arm.bus.sync_read("Present_Position")
    current_position = np.array(
        [current_position_dict[name] for name in current_position_dict], dtype=np.float32
    )
    trajectory = torch.from_numpy(
        np.linspace(current_position, target_position, 50)
    )  # NOTE: 30 is just an arbitrary number
    for pose in trajectory:
        action_dict = dict(zip(current_position_dict, pose, strict=False))
        robot_arm.bus.sync_write("Goal_Position", action_dict)
        precise_sleep(0.015)


class RobotEnv(gym.Env):
    """Gym environment for robotic control with human intervention support."""

    def __init__(
        self,
        robot,
        use_gripper: bool = False,
        display_cameras: bool = False,
        reset_pose: list[float] | None = None,
        reset_time_s: float = 5.0,
    ) -> None:
        """Initialize robot environment with configuration options.

        Args:
            robot: Robot interface for hardware communication.
            use_gripper: Whether to include gripper in action space.
            display_cameras: Whether to show camera feeds during execution.
            reset_pose: Joint positions for environment reset.
            reset_time_s: Time to wait during reset.
        """
        super().__init__()

        self.robot = robot
        self.display_cameras = display_cameras

        # Connect to the robot if not already connected.
        if not self.robot.is_connected:
            self.robot.connect()

        # Episode tracking.
        self.current_step = 0
        self.episode_data = None

        self._joint_names = [f"{key}.pos" for key in self.robot.bus.motors]
        self._image_keys = self.robot.cameras.keys()

        self.reset_pose = reset_pose
        self.reset_time_s = reset_time_s

        self.use_gripper = use_gripper

        self._joint_names = list(self.robot.bus.motors.keys())
        self._raw_joint_positions = None

        self._setup_spaces()

    def _get_observation(self) -> dict[str, Any]:
        """Get current robot observation including joint positions and camera images."""
        obs_dict = self.robot.get_observation()
        raw_joint_joint_position = {f"{name}.pos": obs_dict[f"{name}.pos"] for name in self._joint_names}
        joint_positions = np.array([raw_joint_joint_position[f"{name}.pos"] for name in self._joint_names])

        images = {key: obs_dict[key] for key in self._image_keys}

        return {"agent_pos": joint_positions, "pixels": images, **raw_joint_joint_position}

    def _setup_spaces(self) -> None:
        """Configure observation and action spaces based on robot capabilities."""
        current_observation = self._get_observation()

        observation_spaces = {}

        # Define observation spaces for images and other states.
        if current_observation is not None and "pixels" in current_observation:
            prefix = OBS_IMAGES
            observation_spaces = {
                f"{prefix}.{key}": gym.spaces.Box(
                    low=0, high=255, shape=current_observation["pixels"][key].shape, dtype=np.uint8
                )
                for key in current_observation["pixels"]
            }

        if current_observation is not None:
            agent_pos = current_observation["agent_pos"]
            observation_spaces[OBS_STATE] = gym.spaces.Box(
                low=0,
                high=10,
                shape=agent_pos.shape,
                dtype=np.float32,
            )

        self.observation_space = gym.spaces.Dict(observation_spaces)

        # Define the action space for joint positions along with setting an intervention flag.
        action_dim = 3
        bounds = {}
        bounds["min"] = -np.ones(action_dim)
        bounds["max"] = np.ones(action_dim)

        if self.use_gripper:
            action_dim += 1
            bounds["min"] = np.concatenate([bounds["min"], [0]])
            bounds["max"] = np.concatenate([bounds["max"], [2]])

        self.action_space = gym.spaces.Box(
            low=bounds["min"],
            high=bounds["max"],
            shape=(action_dim,),
            dtype=np.float32,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Additional reset options.

        Returns:
            Tuple of (observation, info) dictionaries.
        """
        # Reset the robot
        # self.robot.reset()
        start_time = time.perf_counter()
        if self.reset_pose is not None:
            log_say("Reset the environment.", play_sounds=True)
            reset_follower_position(self.robot, np.array(self.reset_pose))
            log_say("Reset the environment done.", play_sounds=True)

        precise_sleep(self.reset_time_s - (time.perf_counter() - start_time))

        super().reset(seed=seed, options=options)

        # Reset episode tracking variables.
        self.current_step = 0
        self.episode_data = None
        obs = self._get_observation()
        self._raw_joint_positions = {f"{key}.pos": obs[f"{key}.pos"] for key in self._joint_names}
        return obs, {TeleopEvents.IS_INTERVENTION: False}

    def step(self, action) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Execute one environment step with given action."""
        joint_targets_dict = {f"{key}.pos": action[i] for i, key in enumerate(self.robot.bus.motors.keys())}

        self.robot.send_action(joint_targets_dict)

        obs = self._get_observation()

        self._raw_joint_positions = {f"{key}.pos": obs[f"{key}.pos"] for key in self._joint_names}

        if self.display_cameras:
            self.render()

        self.current_step += 1

        reward = 0.0
        terminated = False
        truncated = False

        return (
            obs,
            reward,
            terminated,
            truncated,
            {TeleopEvents.IS_INTERVENTION: False},
        )

    def render(self) -> None:
        """Display robot camera feeds."""
        import cv2

        current_observation = self._get_observation()
        if current_observation is not None:
            image_keys = [key for key in current_observation if "image" in key]

            for key in image_keys:
                cv2.imshow(key, cv2.cvtColor(current_observation[key].numpy(), cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

    def close(self) -> None:
        """Close environment and disconnect robot."""
        if self.robot.is_connected:
            self.robot.disconnect()

    def get_raw_joint_positions(self) -> dict[str, float]:
        """Get raw joint positions."""
        return self._raw_joint_positions


def make_robot_env(cfg: HILSerlRobotEnvConfig) -> tuple[gym.Env, Any]:
    """Create robot environment from configuration.

    Args:
        cfg: Environment configuration.

    Returns:
        Tuple of (gym environment, teleoperator device).
    """
    # Check if this is a GymHIL simulation environment
    if cfg.name == "gym_hil":
        assert cfg.robot is None and cfg.teleop is None, "GymHIL environment does not support robot or teleop"
        import gym_hil  # noqa: F401

        # Extract gripper settings with defaults
        use_gripper = cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else True
        gripper_penalty = cfg.processor.gripper.gripper_penalty if cfg.processor.gripper is not None else 0.0

        env = gym.make(
            f"gym_hil/{cfg.task}",
            image_obs=True,
            render_mode="human",
            use_gripper=use_gripper,
            gripper_penalty=gripper_penalty,
        )

        return env, None


def step_env_and_process_transition(
    env: gym.Env,
    action: torch.Tensor,
    env_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    action_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    info: dict,
    exit_early_on_intervention_end: bool = False
) -> EnvTransition:
    """
    Execute one step with processor pipeline.

    Args:
        env: The robot environment
        action: Action to execute
        env_processor: Environment processor
        action_processor: Action processor
        info: info dict
        exit_early_on_intervention_end: whether to abort early when an intervention ends

    Returns:
        Processed transition with updated state.
    """
    # Process action
    action_transition = create_transition(action=action, info=info)
    processed_action_transition = action_processor(action_transition)

    if exit_early_on_intervention_end and processed_action_transition[TransitionKey.INFO].get(TeleopEvents.INTERVENTION_COMPLETED, False):
        return processed_action_transition

    # Step env
    obs, reward, terminated, truncated, info = env.step(processed_action_transition[TransitionKey.ACTION])

    # Read out info and possibly overwrite action
    complementary_data = processed_action_transition[TransitionKey.COMPLEMENTARY_DATA].copy()
    info.update(processed_action_transition[TransitionKey.INFO].copy())

    # Determine which action to store (either action that went in, or teleop action that was written as complementary data)
    if info.get(TeleopEvents.IS_INTERVENTION, False) and TELEOP_ACTION_KEY in complementary_data:
        action_to_record = complementary_data[TELEOP_ACTION_KEY]
    else:
        action_to_record = action_transition[TransitionKey.ACTION]

    # Create and process transition
    new_transition = create_transition(
        observation=obs,
        action=action_to_record,
        reward=reward + processed_action_transition[TransitionKey.REWARD],
        done=terminated or processed_action_transition[TransitionKey.DONE],
        truncated=truncated or processed_action_transition[TransitionKey.TRUNCATED],
        info=info,
        complementary_data=complementary_data,
    )
    new_transition = env_processor(new_transition)

    return new_transition


def control_loop(
    env: gym.Env,
    env_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    action_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    teleop_device: Teleoperator,
    cfg: GymManipulatorConfig,
) -> None:
    """Main control loop for robot environment interaction.
    if cfg.mode == "record": then a dataset will be created and recorded

    Args:
     env: The robot environment
     env_processor: Environment processor
     action_processor: Action processor
     teleop_device: Teleoperator device
     cfg: gym_manipulator configuration
    """
    dt = 1.0 / cfg.env.fps

    print(f"Starting control loop at {cfg.env.fps} FPS")
    print("Controls:")
    print("- Use gamepad/teleop device for intervention")
    print("- When not intervening, robot will stay still")
    print("- Press Ctrl+C to exit")

    # Reset environment and processors
    obs, info = env.reset()
    complementary_data = (
        {"raw_joint_positions": info.pop("raw_joint_positions")} if "raw_joint_positions" in info else {}
    )
    env_processor.reset()
    action_processor.reset()

    # Process initial observation
    transition = create_transition(observation=obs, info=info, complementary_data=complementary_data)
    transition = env_processor(data=transition)

    # Determine if gripper is used
    use_gripper = cfg.env.processor.gripper.use_gripper if cfg.env.processor.gripper is not None else True

    dataset = None
    if cfg.mode == "record":
        action_features = teleop_device.action_features
        features = {
            ACTION: action_features,
            REWARD: {"dtype": "float32", "shape": (1,), "names": None},
            DONE: {"dtype": "bool", "shape": (1,), "names": None},
        }
        if use_gripper:
            features["complementary_info.discrete_penalty"] = {
                "dtype": "float32",
                "shape": (1,),
                "names": ["discrete_penalty"],
            }

        for key, value in transition[TransitionKey.OBSERVATION].items():
            if key == OBS_STATE:
                features[key] = {
                    "dtype": "float32",
                    "shape": value.squeeze(0).shape,
                    "names": None,
                }
            if "image" in key:
                features[key] = {
                    "dtype": "video",
                    "shape": value.squeeze(0).shape,
                    "names": ["channels", "height", "width"],
                }

        # Create dataset
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.env.fps,
            root=cfg.dataset.root,
            use_videos=True,
            image_writer_threads=4,
            image_writer_processes=0,
            features=features,
        )

    episode_idx = 0
    episode_step = 0
    episode_start_time = time.perf_counter()

    while episode_idx < cfg.dataset.num_episodes_to_record:
        step_start_time = time.perf_counter()

        # Create a neutral action (no movement)
        neutral_action = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        if use_gripper:
            neutral_action = torch.cat([neutral_action, torch.tensor([1.0])])  # Gripper stay

        # Use the new step function
        transition = step_env_and_process_transition(
            env=env,
            transition=transition,
            action=neutral_action,
            env_processor=env_processor,
            action_processor=action_processor,
        )
        terminated = transition.get(TransitionKey.DONE, False)
        truncated = transition.get(TransitionKey.TRUNCATED, False)

        if cfg.mode == "record":
            observations = {
                k: v.squeeze(0).cpu()
                for k, v in transition[TransitionKey.OBSERVATION].items()
                if isinstance(v, torch.Tensor)
            }
            # Use teleop_action if available, otherwise use the action from the transition
            action_to_record = transition[TransitionKey.COMPLEMENTARY_DATA].get(
                "teleop_action", transition[TransitionKey.ACTION]
            )
            frame = {
                **observations,
                ACTION: action_to_record.cpu(),
                REWARD: np.array([transition[TransitionKey.REWARD]], dtype=np.float32),
                DONE: np.array([terminated or truncated], dtype=bool),
            }
            if use_gripper:
                discrete_penalty = transition[TransitionKey.COMPLEMENTARY_DATA].get("discrete_penalty", 0.0)
                frame["complementary_info.discrete_penalty"] = np.array([discrete_penalty], dtype=np.float32)

            if dataset is not None:
                frame["task"] = cfg.dataset.task
                dataset.add_frame(frame)

        episode_step += 1

        # Handle episode termination
        if terminated or truncated:
            episode_time = time.perf_counter() - episode_start_time
            logging.info(
                f"Episode ended after {episode_step} steps in {episode_time:.1f}s with reward {transition[TransitionKey.REWARD]}"
            )
            episode_step = 0
            episode_idx += 1

            if dataset is not None:
                if transition[TransitionKey.INFO].get(TeleopEvents.RERECORD_EPISODE, False):
                    logging.info(f"Re-recording episode {episode_idx}")
                    dataset.clear_episode_buffer()
                    episode_idx -= 1
                else:
                    logging.info(f"Saving episode {episode_idx}")
                    dataset.save_episode()

            # Reset for new episode
            obs, info = env.reset()
            env_processor.reset()
            action_processor.reset()

            transition = create_transition(observation=obs, info=info)
            transition = env_processor(transition)

        # Maintain fps timing
        precise_sleep(dt - (time.perf_counter() - step_start_time))

    if dataset is not None and cfg.dataset.push_to_hub:
        logging.info("Pushing dataset to hub")
        dataset.push_to_hub()


def replay_trajectory(
    env: gym.Env, action_processor: DataProcessorPipeline, cfg: GymManipulatorConfig
) -> None:
    """Replay recorded trajectory on robot environment."""
    assert cfg.dataset.replay_episode is not None, "Replay episode must be provided for replay"

    dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=[cfg.dataset.replay_episode],
        download_videos=False,
    )
    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == cfg.dataset.replay_episode)
    actions = episode_frames.select_columns(ACTION)

    _, info = env.reset()

    for action_data in actions:
        start_time = time.perf_counter()
        transition = create_transition(
            observation=env.get_raw_joint_positions() if hasattr(env, "get_raw_joint_positions") else {},
            action=action_data[ACTION],
        )
        transition = action_processor(transition)
        env.step(transition[TransitionKey.ACTION])
        precise_sleep(1 / cfg.env.fps - (time.perf_counter() - start_time))


@parser.wrap()
def main(cfg: GymManipulatorConfig) -> None:
    """Main entry point for gym manipulator script."""
    env, teleop_device = make_robot_env(cfg.env)
    env_processor, action_processor = make_default_processors(env, teleop_device, cfg.env, cfg.device)

    print("Environment observation space:", env.observation_space)
    print("Environment action space:", env.action_space)
    print("Environment processor:", env_processor)
    print("Action processor:", action_processor)

    if cfg.mode == "replay":
        replay_trajectory(env, action_processor, cfg)
        exit()

    control_loop(env, env_processor, action_processor, teleop_device, cfg)


if __name__ == "__main__":
    main()
