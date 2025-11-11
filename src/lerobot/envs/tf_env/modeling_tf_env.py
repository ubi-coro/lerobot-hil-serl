import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any

from torch import Tensor

from lerobot.cameras import Camera
from lerobot.configs.types import PolicyFeature, FeatureType
from lerobot.datasets.pipeline_features import create_initial_features
from lerobot.envs.factory import RobotEnvInterface
from lerobot.envs.robot_env.configuration_robot_env import HILSerlProcessorConfig
from lerobot.robots.ur import TF_UR
from lerobot.robots.ur.tf_controller import TaskFrameCommand, AxisMode
from lerobot.teleoperators import TeleopEvents
from lerobot.utils.constants import ACTION
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say


class TaskFrameEnv(RobotEnvInterface):
    """Multi-robot Gym environment for task-frame based control with intervention support."""

    def __init__(
        self,
        robot_dict: dict[str, "TF_UR"],
        cameras: dict[str, Camera] | None = None,
        processor: HILSerlProcessorConfig | None = None
    ) -> None:
        super().__init__(robot_dict=robot_dict, cameras=cameras, processor=processor)

        self.task_frame = self.processor.task_frame.command
        self.control_mask = {name: np.asarray(msk).astype(bool) for name, msk in self.processor.task_frame.control_mask.items()}
        self.use_gripper = self.processor.gripper.use_gripper
        self.reset_pose = self.processor.reset.fixed_reset_joint_positions
        self.reset_time_s = self.processor.reset.reset_time_s
        self.display_cameras = self.processor.display_cameras

        assert all([self.robot_dict[name].config.use_gripper or not self.use_gripper[name] for name in self.robot_dict]), "To use a gripper, the robot must have one!"

        # build reset task frame commands
        self.reset_task_frame: dict[str, TaskFrameCommand | None] = {}
        for name, robot in self.robot_dict.items():
            if self.reset_pose[name] is None:
                self.reset_task_frame[name] = None
            else:
                # copy task frame command and overwrite target
                base = self.task_frame[name]
                self.reset_task_frame[name] = TaskFrameCommand(
                    mode=[AxisMode.POS] * 6,
                    target=self.reset_pose[name],
                    kp=base.kp,
                    kd=base.kd,
                    T_WF=base.T_WF,
                    max_pose_rpy=base.max_pose_rpy,
                    min_pose_rpy=base.min_pose_rpy,
                )

            if not robot.is_connected:
                robot.connect()

            # remove cameras to only use "ours"
            robot.cameras = {}

        self.current_step = 0
        self.episode_data = None

        # Collect motor and image keys
        self._motor_keys: set[str] = set()
        for name, robot in self.robot_dict.items():
            self._motor_keys.update([f"{name}.{key}" for key in robot._motors_ft])

        #self._setup_spaces()

    @staticmethod
    def get_features_from_cfg(cfg: 'TFRobotEnvConfig'):
        # action features
        masks = cfg.processor.task_frame.control_mask
        gripper = cfg.processor.gripper.use_gripper
        action_dim = sum([sum(m) for m in masks.values()]) + sum(gripper.values())
        action_ft = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}

        # obs features
        obs_ft = {}
        for name in cfg.robot:
            for i, ax in enumerate(["x", "y", "z", "wx", "wy", "wz"]):
                obs_ft[f"{name}.{ax}.ee_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(1,))
                obs_ft[f"{name}.{ax}.ee_vel"] = PolicyFeature(type=FeatureType.STATE, shape=(1,))
                obs_ft[f"{name}.{ax}.ee_wrench"] = PolicyFeature(type=FeatureType.STATE, shape=(1,))

            for i, joint_name in enumerate(TF_UR.joint_names):
                obs_ft[f"{name}.{joint_name}.q_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(1,))

            if cfg.processor.gripper.use_gripper[name]:
                obs_ft[f"{name}.gripper.pos"] = PolicyFeature(type=FeatureType.STATE, shape=(1,))

        for cam_name, cam_cfg in cfg.cameras.items():
            # Match your env's observation dict convention: "pixels.<cam>"
            obs_ft[f"pixels.{cam_name}"] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(cam_cfg.height, cam_cfg.width, 3),
            )

        return create_initial_features(observation=obs_ft, action=action_ft)

    def _setup_spaces(self) -> None:
        """Configure observation and action spaces for all robots."""
        # peek one observation
        any_robot = next(iter(self.robot_dict.values()))
        current_observation = any_robot.get_observation()

        obs_spaces = {}
        for motor_key in self._motor_keys:
            obs_spaces[motor_key] = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        if self.cameras:
            pixel_spaces = {}
            for image_key in self.cameras:
                pixel_spaces[image_key] = spaces.Box(
                    low=0, high=255, shape=current_observation["pixels"][image_key].shape, dtype=np.uint8
                )
            obs_spaces["pixels"] = gym.spaces.Dict(pixel_spaces)

        self.observation_space = gym.spaces.Dict(obs_spaces)

        # Action space (combined size of all robots' control masks and grippers)
        total_dims = 0
        for name in self.robot_dict:
            total_dims += sum(self.control_mask[name]) + int(self.use_gripper[name])
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(total_dims,), dtype=np.float32)

    def _get_ee_pos(self):
        obs = self.robot.get_observation()
        return np.array([obs[f"{name}.{ax}.ee_pos"] for ax in ["x", "y", "z", "wx", "wy", "wz"] for name in self.robot_dict])

    def _get_observation(self):
        obs_dict = {}
        if self.cameras:
            obs_dict["pixels"] = {}

        for cam_key, cam in self.cameras.items():
            obs_dict["pixels"][cam_key] = cam.async_read()

        for name in self.robot_dict:
            robot_obs_dict = self.robot_dict[name].get_observation()
            obs_dict |= {f"{name}.{key}": robot_obs_dict[key] for key in robot_obs_dict}

            if not self.use_gripper[name]:
                obs_dict.pop(f"{name}.gripper.pos", None)

        return obs_dict

    def _get_info(self):
        return {TeleopEvents.IS_INTERVENTION: False}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Reset all robots to initial state."""
        for name, robot in self.robot_dict.items():
            robot.task_frame = self.task_frame[name]

            if self.reset_pose[name] is not None:
                log_say(f"Resetting robot {name}", play_sounds=True)
                current_pos = self._get_ee_pos(name)
                trajectory = np.linspace(current_pos, self.reset_pose[name], 50)

                for pose in trajectory:
                    tf_cmd = self.reset_task_frame[name]
                    tf_cmd.target = pose
                    robot.send_action(tf_cmd.to_robot_action())
                    busy_wait(self.reset_time_s[name] * 0.8 / 50)

                log_say(f"Robot {name} reset done.", play_sounds=True)

        super().reset(seed=seed, options=options)

        self.current_step = 0
        self.episode_data = None
        return self._get_observation(), self._get_info()

    def step(self, action: np.ndarray | Tensor):
        """Step all robots with a flat action vector split per robot."""
        if isinstance(action, Tensor):
            action = action.detach().cpu().numpy()

        offset = 0
        for name, robot in self.robot_dict.items():
            n_ctrl = sum(self.control_mask[name])
            n_total = n_ctrl + int(self.use_gripper[name])

            robot_action_vec = action[offset:offset+n_total]
            offset += n_total

            tf_cmd = self.task_frame[name]
            target = np.array(tf_cmd.target)
            if self.use_gripper[name]:
                target[self.control_mask[name]] = robot_action_vec[:-1]
                robot.send_gripper_action(robot_action_vec[-1])
            else:
                target[self.control_mask[name]] = robot_action_vec

            tf_cmd.target = target.tolist()
            robot.controller.send_cmd(tf_cmd)

        obs = self._get_observation()

        if self.display_cameras:
            self.render()

        self.current_step += 1
        reward = 0.0
        terminated = False
        truncated = False
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        import cv2
        current_observation = self._get_observation()
        if current_observation is not None and "pixels" in current_observation:
            for key, img in current_observation["pixels"].items():
                cv2.imshow(key, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

    def close(self) -> None:
        for robot in self.robot_dict.values():
            if robot.is_connected:
                robot.disconnect()