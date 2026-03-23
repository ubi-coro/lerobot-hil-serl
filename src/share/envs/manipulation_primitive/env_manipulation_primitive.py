from typing import SupportsFloat, Any

import gymnasium
import numpy as np
import torch
from gymnasium.core import ActType, ObsType

from lerobot.teleoperators import TeleopEvents
from lerobot.utils.constants import OBS_IMAGES
from lerobot.cameras import Camera
from lerobot.robots import Robot

from .task_frame import TaskFrame
from ..utils import check_task_frame_robot


class ManipulationPrimitive(gymnasium.Env):
    """Minimal gym env wiring robots, cameras, and task-frame commands."""

    def __init__(
        self,
        task_frame: dict[str, TaskFrame],
        robot_dict: dict[str, Robot],
        cameras: dict[str, Camera],
        display_cameras: bool = False
    ):
        """Initialize robot/camera handles and action slicing metadata."""
        self.robot_dict = robot_dict
        self.task_frame = task_frame
        self.cameras = cameras
        self.display_cameras = display_cameras

        self.current_step = 0
        self._motor_keys: set[str] = set()
        self._is_task_frame_robot: dict[str, bool] = check_task_frame_robot(robot_dict)
        for name, robot in self.robot_dict.items():
            if self._is_task_frame_robot[name]:
                robot.set_task_frame(self.task_frame[name])
            self._motor_keys.update([f"{name}.{key}" for key in robot._motors_ft])

    def step(self, action: dict[str, dict[str, float]]) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Apply a per-robot action dict and return fresh observations."""
        for name, robot in self.robot_dict.items():
            if self._is_task_frame_robot[name]:
                robot.set_task_frame(self.task_frame[name])
            robot.send_action(action.get(name, {}))

        obs = self._get_observation()

        if self.display_cameras:
            self.render()

        self.current_step += 1
        reward = 0.0
        terminated = False
        truncated = False
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset counters and resend configured task frames when supported."""
        super().reset(seed=seed, options=options)

        # Manipulation primitives do not execute an autonomous reset trajectory by default.
        # We only re-send the configured task frame to robots that support task-frame commands.
        for name, robot in self.robot_dict.items():
            if self._is_task_frame_robot.get(name, False):
                robot.set_task_frame(self.task_frame[name])

        self.current_step = 0
        obs = self._get_observation()
        return obs, self._get_info()

    def render(self) -> None:
        """Display camera observations using OpenCV windows."""
        import cv2
        current_observation = self._get_observation()
        if current_observation is not None and "pixels" in current_observation:
            for key, img in current_observation["pixels"].items():
                cv2.imshow(key, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

    def close(self) -> None:
        """Disconnect any connected robots."""
        for robot_dict in self.robot_dict.values():
            if robot_dict.is_connected:
                robot_dict.disconnect()

    def _get_observation(self):
        """Collect camera pixels and robot observations into one dict."""
        obs_dict = {}

        for cam_key, cam in self.cameras.items():
            obs_dict[f"{OBS_IMAGES}.{cam_key}"] = cam.async_read()

        for name in self.robot_dict:
            robot_dict_obs_dict = self.robot_dict[name].get_observation()
            obs_dict |= {f"{name}.{key}": robot_dict_obs_dict[key] for key in robot_dict_obs_dict}

        return obs_dict

    @staticmethod
    def _get_info():
        """Return default per-step info payload."""
        return {TeleopEvents.IS_INTERVENTION: False}
