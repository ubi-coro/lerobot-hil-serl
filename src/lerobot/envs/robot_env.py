import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from lerobot.configs.types import PolicyFeature, FeatureType
from lerobot.processor.hil_processor import GRIPPER_KEY
from lerobot.robots import Robot
from lerobot.teleoperators import TeleopEvents
from lerobot.utils.constants import OBS_STATE, OBS_IMAGES
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say


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
        #robot_arm.bus.sync_write("Goal_Position", action_dict)
        busy_wait(0.015)


class RobotEnv(gym.Env):
    """Gym environment for robotic control with human intervention support."""

    def __init__(
        self,
        robot_dict: dict[str, Robot],
        use_gripper: bool = dict[str, bool],
        reset_pose: dict[str, list] | None = None,
        reset_time_s: dict[str, float] | None = None,
        display_cameras: bool = False,
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

        self.robot_dict = robot_dict
        self.use_gripper = use_gripper
        self.display_cameras = display_cameras

        if reset_pose is None:
            reset_pose = {name: None for name in robot_dict}
        if reset_time_s is None:
            reset_pose = {name: 5.0 for name in robot_dict}

        # Episode tracking.
        self.current_step = 0
        self.episode_data = None

        self._joint_names_list = []
        self._joint_names_dict = {}
        self._image_keys = []
        for name, robot in self.robot_dict.items():
            ft = robot._motors_ft
            if f"{GRIPPER_KEY}.pos" in ft and not self.use_gripper[name]:
                ft.pop(f"{GRIPPER_KEY}.pos")

            self._joint_names_list.extend([f"{name}.{key}" for key in ft])
            self._joint_names_dict[name] = ft
            self._image_keys.extend(robot._cameras_ft.keys())

        self.reset_pose = reset_pose
        self.reset_time_s = reset_time_s

        self._raw_joint_positions = None

        self._setup_spaces()

    @property
    def action_features(self) -> PolicyFeature:
        """
        A dictionary describing the structure and types of the actions expected by the robot. Its structure
        (keys) should match the structure of what is passed to :pymeth:`send_action`. Values for the dict
        should be the type of the value if it's a simple value, e.g. `float` for single proprioceptive value
        (a joint's goal position/velocity)

        Note: this property should be able to be called regardless of whether the robot is connected or not.
        """
        return {key: float for key in self._joint_names_list}


    @property
    def observation_features(self) -> dict:
        """
        A dictionary describing the structure and types of the actions expected by the robot. Its structure
        (keys) should match the structure of what is passed to :pymeth:`send_action`. Values for the dict
        should be the type of the value if it's a simple value, e.g. `float` for single proprioceptive value
        (a joint's goal position/velocity)

        Note: this property should be able to be called regardless of whether the robot is connected or not.
        """
        ft = {"agent_pos": float}
        for name in self.robot_dict:
            ft |= {f"{name}.{key}": float for key in self.robot_dict[name]._motors_ft}

        if self._image_keys:
            ft["pixels"] = {}
            for name in self.robot_dict:
                ft["pixels"] |= self.robot_dict[name]._cameras_ft

        return ft


    def _get_observation(self) -> dict[str, Any]:
        """Get current robot observation including joint positions and camera images."""
        raw_joint_joint_position = {}
        images = {}
        for name in self.robot_dict:
            obs_dict = self.robot_dict[name].get_observation()
            raw_joint_joint_position |= {f"{name}.{key}": obs_dict[key] for key in self._joint_names_dict[name]}
            images |= {key: obs_dict[key] for key in self.robot_dict[name]._cameras_ft}

        obs = {"agent_pos": np.array(list(raw_joint_joint_position.values())), **raw_joint_joint_position}
        if self._image_keys:
            obs["pixels"] = images

        return obs

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
        action_dim = len(self._joint_names_list)
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
        for name, robot in self.robot_dict.items():

            if self.reset_pose.get(name, None) is not None:
                log_say(f"Reset the environment of the {name} robot.", play_sounds=True)
                reset_follower_position(robot, np.array(self.reset_pose[name]))
                log_say("Reset the environment done.", play_sounds=True)

            busy_wait(self.reset_time_s.get(name, 1.0) - (time.perf_counter() - start_time))

        super().reset(seed=seed, options=options)

        # Reset episode tracking variables.
        self.current_step = 0
        self.episode_data = None
        obs = self._get_observation()
        self._raw_joint_positions = {key: obs[key] for key in self._joint_names_list}
        return obs, {TeleopEvents.IS_INTERVENTION: False}

    def step(self, action) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Execute one environment step with given action."""

        idx_start = 0
        for name, robot in self.robot_dict.items():
            target_joint_positions = {key: action[idx_start + i] for i, key in enumerate(self._joint_names_dict[name])}
            robot.send_action(target_joint_positions)
            idx_start += len(self._joint_names_dict[name])

        obs = self._get_observation()

        self._raw_joint_positions = {key: obs[key] for key in self._joint_names_list}

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
        for robot in self.robot_dict.values():
            robot.disconnect()

    def get_raw_joint_positions(self) -> dict[str, float]:
        """Get raw joint positions."""
        return self._raw_joint_positions
