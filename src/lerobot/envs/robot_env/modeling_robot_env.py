import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from lerobot.cameras import Camera
from lerobot.configs.types import PolicyFeature, FeatureType
from lerobot.datasets.pipeline_features import create_initial_features
from lerobot.envs.configs import HILSerlProcessorConfig
from lerobot.envs.factory import RobotEnvInterface
from lerobot.processor.hil_processor import GRIPPER_KEY
from lerobot.robots import Robot
from lerobot.teleoperators import TeleopEvents
from lerobot.utils.constants import ACTION
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


class RobotEnv(RobotEnvInterface):
    """Gym environment for robotic control with human intervention support."""

    def __init__(
        self,
        robot_dict: dict[str, Robot],
        cameras: dict[str, Camera] | None = None,
        processor: HILSerlProcessorConfig | None = None
    ) -> None:
        """Initialize robot environment with configuration options.

        Args:
            robot: Robot interface for hardware communication.
            use_gripper: Whether to include gripper in action space.
            display_cameras: Whether to show camera feeds during execution.
            reset_pose: Joint positions for environment reset.
            reset_time_s: Time to wait during reset.
        """
        super().__init__(robot_dict=robot_dict, cameras=cameras, processor=processor)

        self.use_gripper = self.processor.gripper.use_gripper
        self.reset_pose = self.processor.reset.fixed_reset_joint_positions
        self.reset_time_s = self.processor.reset.reset_time_s
        self.display_cameras = self.processor.display_cameras

        # Episode tracking.
        self.current_step = 0
        self.episode_data = None

        self._raw_joint_positions = None
        self._joint_names_list = []
        self._joint_names_dict = {}
        for name, robot in self.robot_dict.items():
            ft = robot._motors_ft
            if f"{GRIPPER_KEY}.pos" in ft and not self.use_gripper[name]:
                ft.pop(f"{GRIPPER_KEY}.pos")

            self._joint_names_list.extend([f"{name}.{key}" for key in ft])
            self._joint_names_dict[name] = ft

        self._setup_spaces()

    @staticmethod
    def get_features_from_cfg(cfg: 'RobotEnvConfig'):
        # calculate state == action dim
        gripper = cfg.processor.gripper.use_gripper
        state_dim = 6 * len(gripper) + sum(gripper.values())

        # action features
        action_ft = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(state_dim, ))}

        # obs features
        obs_ft = {"agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))}
        for cam_name, cam_cfg in cfg.cameras.items():
            obs_ft[f"pixels.{cam_name}"] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(cam_cfg.height, cam_cfg.width, 3)
            )

        return create_initial_features(observation=obs_ft, action=action_ft)

    def _get_observation(self) -> dict[str, Any]:
        """Get current robot observation including joint positions and camera images."""

        # Capture state info from robots
        raw_joint_joint_position = {}
        for name in self.robot_dict:
            obs_dict = self.robot_dict[name].get_observation()
            raw_joint_joint_position |= {f"{name}.{key}": obs_dict[key] for key in self._joint_names_dict[name]}

        obs = {"agent_pos": np.array(list(raw_joint_joint_position.values())), **raw_joint_joint_position}

        # Capture images (time each camera individually)
        cam_timings_ms: dict[str, float] = {}
        if self.cameras:
            pixels: dict[str, Any] = {}
            cam_start_total = time.perf_counter()

            for cam_key, cam in self.cameras.items():
                t_cam0 = time.perf_counter()
                pixels[cam_key] = cam.async_read()
                t_cam1 = time.perf_counter()
                cam_timings_ms[cam_key] = (t_cam1 - t_cam0) * 1000.0

            cam_end_total = time.perf_counter()
            obs["pixels"] = pixels

        return obs

    def _setup_spaces(self) -> None:
        """Configure observation and action spaces based on robot capabilities."""
        current_observation = self._get_observation()

        observation_spaces = {}

        # Define observation spaces for images and other states.
        if current_observation is not None and "pixels" in current_observation:
            observation_spaces = {
                f"pixels.{key}": gym.spaces.Box(
                    low=0, high=255, shape=current_observation["pixels"][key].shape, dtype=np.uint8
                )
                for key in current_observation["pixels"]
            }

        if current_observation is not None:
            observation_spaces["agent_pos"] = gym.spaces.Box(
                low=0,
                high=10,
                shape=current_observation["agent_pos"].shape,
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

                busy_wait(self.reset_time_s.get(name, 5.0) - (time.perf_counter() - start_time))

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
