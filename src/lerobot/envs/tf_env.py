import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any

from lerobot.robots.ur.tff_controller import TaskFrameCommand, AxisMode
from lerobot.teleoperators import TeleopEvents
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say


class TaskFrameEnv(gym.Env):
    """Multi-robot Gym environment for task-frame based control with intervention support."""

    def __init__(
        self,
        robot_dict: dict[str, "TF_UR"],
        task_frame: dict[str, TaskFrameCommand],
        control_mask: dict[str, list[bool]],
        use_gripper: dict[str, bool],
        reset_pose: dict[str, list[float]],
        reset_time_s: dict[str, float],
        display_cameras: bool = False,
    ) -> None:
        super().__init__()

        self.robot_dict = robot_dict
        self.task_frame = task_frame
        self.control_mask = control_mask
        self.use_gripper = use_gripper
        self.reset_pose = reset_pose
        self.reset_time_s = reset_time_s
        self.display_cameras = display_cameras

        # build reset task frame commands
        self.reset_task_frame: dict[str, TaskFrameCommand] = {}
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

        self.current_step = 0
        self.episode_data = None

        # Collect motor and image keys
        self._motor_keys: set[str] = set()
        self._image_keys: set[str] = set()
        for name, robot in self.robot_dict.items():
            self._motor_keys.update([f"{name}.{key}" for key in robot._motors_ft])
            self._image_keys.update([f"{name}.{key}" for key in robot._cameras_ft])

        self._setup_spaces()

    def _setup_spaces(self) -> None:
        """Configure observation and action spaces for all robots."""
        # peek one observation
        any_robot = next(iter(self.robot_dict.values()))
        current_observation = any_robot.get_observation()

        obs_spaces = {}
        for motor_key in self._motor_keys:
            obs_spaces[motor_key] = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        if self._image_keys:
            pixel_spaces = {}
            for image_key in self._image_keys:
                pixel_spaces[image_key] = spaces.Box(
                    low=0, high=255, shape=current_observation[image_key].shape, dtype=np.uint8
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
        for name in self.robot_dict:
            robot_obs_dict = self.robot_dict[name].get_observation()
            obs_dict |= {f"{name}.{key}": robot_obs_dict[key] for key in robot_obs_dict}

        if self._image_keys:
            obs_dict["pixels"] = {}
            for image_key in self._image_keys:
                obs_dict["pixels"][image_key] = obs_dict[image_key]
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

    def step(self, action: np.ndarray):
        """Step all robots with a flat action vector split per robot."""
        offset = 0
        for name, robot in self.robot_dict.items():
            n_ctrl = sum(self.control_mask[name])
            n_total = n_ctrl + int(self.use_gripper[name])

            robot_action_vec = action[offset:offset+n_total]
            offset += n_total

            if self.use_gripper[name]:
                tf_cmd = self.task_frame[name]
                tf_cmd.target[self.control_mask[name]] = robot_action_vec[:-1]
                robot_action = tf_cmd.to_robot_action()
                robot_action["gripper.pos"] = robot_action_vec[-1]
            else:
                tf_cmd = self.task_frame[name]
                tf_cmd.target[self.control_mask[name]] = robot_action_vec
                robot_action = tf_cmd.to_robot_action()

            robot.send_action(robot_action)

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



if __name__ == "__main__":
    from lerobot.robots.ur import URConfig, TF_UR
    from lerobot.envs.configs import TFHilSerlRobotEnvConfig
    from lerobot.teleoperators.spacemouse import SpacemouseConfig
    from lerobot.teleoperators.keyboard import KeyboardEndEffectorTeleop, KeyboardEndEffectorTeleopConfig
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.processor import create_transition
    from lerobot.rl.gym_manipulator import make_robot_env, step_env_and_process_transition
    import torch

    env_config = TFHilSerlRobotEnvConfig(
        robot=URConfig(mock=True, model="ur3e", robot_ip="127.0.0.1"),#, cameras={"main": OpenCVCameraConfig(index_or_path=4)}),
        teleop=KeyboardEndEffectorTeleopConfig(),
    )

    online_env, env_processor, action_processor = make_robot_env(cfg=env_config)

    obs, info = online_env.reset()
    env_processor.reset()
    action_processor.reset()

    # Process initial observation
    transition = create_transition(observation=obs, info=info)
    transition = env_processor(transition)

    while True:
        action = torch.tensor([0] * 6).type(torch.float32)

        new_transition = step_env_and_process_transition(
            env=online_env,
            transition=transition,
            action=action,
            env_processor=env_processor,
            action_processor=action_processor,
        )

        print(new_transition)

        transition = new_transition

        busy_wait(0.2)
