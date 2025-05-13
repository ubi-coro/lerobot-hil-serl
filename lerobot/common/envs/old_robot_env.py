import argparse
from collections import deque

import matplotlib.pyplot as plt
import logging
import time
from scipy.spatial.transform import Rotation as R
from threading import Lock
from typing import Annotated, Any, Dict, Tuple
import gymnasium as gym
import numpy as np
import torch
import torchvision.transforms.functional as F  # noqa: N812

from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.robot_devices.control_utils import (
    busy_wait,
    is_headless,
    reset_follower_position,
    reset_leader_position
)
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.utils.utils import init_hydra_config, log_say

from lerobot.scripts.server.kinematics import MRKinematics, RobotKinematics
from lerobot.common.robot_devices.motors.dynamixel import TorqueMode

logging.basicConfig(level=logging.INFO)


class HILSerlRobotEnv(gym.Env):
    """
    Gym-compatible environment for evaluating robotic control policies with integrated human intervention.

    This environment wraps a robot interface to provide a consistent API for policy evaluation. It supports both relative (delta)
    and absolute joint position commands and automatically configures its observation and action spaces based on the robot's
    sensors and configuration.

    The environment can switch between executing actions from a policy or using teleoperated actions (human intervention) during
    each step. When teleoperation is used, the override action is captured and returned in the `info` dict along with a flag
    `is_intervention`.
    """

    def __init__(
        self,
        robot,
        ee_action_space_params,
        display_cameras: bool = False,
        display_lag: bool = True
    ):
        """
        Initialize the HILSerlRobotEnv environment.

        The environment is set up with a robot interface, which is used to capture observations and send joint commands. The setup
        supports both relative (delta) adjustments and absolute joint positions for controlling the robot.

        Args:
            robot: The robot interface object used to connect and interact with the physical robot.
            use_delta_action_space (bool): If True, uses a delta (relative) action space for joint control. Otherwise, absolute
                joint positions are used.
            delta (float or None): A scaling factor for the relative adjustments applied to joint positions. Should be a value between
                0 and 1 when using a delta action space.
            display_cameras (bool): If True, the robot's camera feeds will be displayed during execution.
        """
        super().__init__()

        self.robot = robot
        self.display_cameras = display_cameras

        # Initialize kinematics instance for the appropriate robot type
        if robot.config.robot_type.lower().startswith('aloha'):
            kinematics = MRKinematics(robot.config.follower_model)
            self.preprocess_q_dot = kinematics.apply_joint_correction  # removes shadows and converts to radian for mr
        else:
            kinematics = RobotKinematics(getattr(robot.config, "robot_type", "so100"))
            self.preprocess_q_dot = lambda q_dot: q_dot
        self.fk = kinematics.fk_gripper_tip
        self.ik = kinematics.ik
        self.jacobian = kinematics.compute_jacobian

        # Connect to the robot if not already connected.
        if not self.robot.is_connected:
            self.robot.connect()

        self.curr_q = None
        self.curr_tcp_pose = None
        self.curr_tcp_vel = None
        self.next_tcp_pose = None
        self._update()

        # Episode tracking.
        self.current_step = 0
        self.episode_data = None

        self.ee_action_space_params = ee_action_space_params
        self.action_scale = np.array(
            3 * [self.ee_action_space_params["xyz_step_size"]] +
            3 * [self.ee_action_space_params["rot_step_size"]] +
            1 * [self.ee_action_space_params["gripper_step_size"]]
        )

        # Dynamically configure the observation and action spaces.
        self._setup_spaces()

        # Optionally track position lag
        self.display_lag = display_lag
        if self.display_lag:
            self.current_history = deque(maxlen=100)
            self.target_history = deque(maxlen=100)

            plt.ion()
            f1, self.pos_axs = plt.subplots(3, 1, figsize=(8, 6))
            f2, self.rot_axs = plt.subplots(3, 1, figsize=(8, 6))

        # todo: calc this in the reset wrapper and use it when no reset pose is passed (+ absorb reset wrapper)
        reset_pose = np.eye(4)
        reset_pose[:3, :2] = (self.xyz_bounding_box.high[:2] + self.xyz_bounding_box.low[:2]) / 2
        reset_pose[:3, 2] = self.xyz_bounding_box.high[2]
        reset_pose[:3, :3] = R.from_euler(
            "xyz", (self.rpy_bounding_box.high + self.rpy_bounding_box.low) / 2
        ).as_matrix()
        reset_joints = self.ik(current_joint_state=self.curr_q, desired_ee_pose=reset_pose, fk_func=self.fk, gripper_pos=0)


    def _setup_spaces(self):
        """
        Dynamically configure the observation and action spaces based on the robot's capabilities.

        Observation Space:
            - For keys with "image": A Box space with pixel values ranging from 0 to 255.
            - For non-image keys: A nested Dict space is created under 'observation.state' with a suitable range.

        Action Space:
            - The action space is defined as a Tuple where:
                • The first element is a Box space representing joint position commands. It is defined as relative (delta)
                  or absolute, based on the configuration.
                • ThE SECONd element is a Discrete space (with 2 values) serving as a flag for intervention (teleoperation).
        """


        # Define observation spaces for images and other states.
        observation_spaces = {}
        ignore_keys = ["observation.state", "observation.joint_vel"]
        for key, feature in self.robot.features.items():
            if key in ignore_keys:
                continue

            if "image" in key:
                observation_spaces[key] = gym.spaces.Box(low=0, high=255, shape=feature["shape"], dtype=np.uint8)
            else:
                observation_spaces[key] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=feature["shape"], dtype=np.float32)

        observation_spaces["observation.tcp_pose"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        observation_spaces["observation.tcp_vel"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(observation_spaces)

        action_space_robot = gym.spaces.Box(
            low=np.array([-1] * 7),
            high=np.array([1] * 7),
            shape=(7,),
            dtype=np.float32,
            )
        self.action_space = gym.spaces.Tuple(
            (
                action_space_robot,
                gym.spaces.Discrete(2),
            ),
        )

        # boundary box
        lower_bounds = np.array(self.ee_action_space_params["bounds"]["min"])
        upper_bounds = np.array(self.ee_action_space_params["bounds"]["max"])
        self.xyz_bounding_box = gym.spaces.Box(
            low=lower_bounds[:3],
            high=upper_bounds[:3],
            dtype=np.float64,
        )
        self.rpy_bounding_box = gym.spaces.Box(
            low=lower_bounds[3:],
            high=upper_bounds[3:],
            dtype=np.float64,
        )

    def reset(
        self, seed=None, options=None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment to its initial state.
        This method resets the step counter and clears any episodic data.

        Args:
            seed (Optional[int]): A seed for random number generation to ensure reproducibility.
            options (Optional[dict]): Additional options to influence the reset behavior.

        Returns:
            A tuple containing:
                - observation (dict): The initial sensor observation.
                - info (dict): A dictionary with supplementary information, including the key "initial_position".
        """
        super().reset(seed=seed, options=options)

        # Capture the initial observation.
        obs = self.robot.capture_observation()
        self._update(obs=obs)

        # Reset episode tracking variables.
        self.current_step = 0
        self.episode_data = None

        return obs, {}

    def step(
        self, action: Tuple[np.ndarray, bool]
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute a single step within the environment using the specified action.

        The provided action is a tuple comprised of:
            • A policy action that specifies end-effector deltas
            • A boolean flag indicating whether teleoperation (human intervention) should be used for this step.

        Behavior:
            - When the intervention flag is False, the environment processes and sends the policy action to the robot.
            - When True, a teleoperation step is executed. If using a delta action space, an absolute teleop action is converted
              to relative change based on the current joint positions.

        Args:
            action (tuple): A tuple with two elements:
                - policy_action (np.ndarray or torch.Tensor): The commanded end-effector delta positions.
                - intervention_bool (bool): True if the human operator intervenes by providing a teleoperation input.

        Returns:
            tuple: A tuple containing:
                - observation (dict): The new sensor observation after taking the step.
                - reward (float): The step reward (default is 0.0 within this wrapper).
                - terminated (bool): True if the episode has reached a terminal state.
                - truncated (bool): True if the episode was truncated (e.g., time constraints).
                - info (dict): Additional debugging information including:
                    ◦ "action_intervention": The teleop action if intervention was used.
                    ◦ "is_intervention": Flag indicating whether teleoperation was employed.
        """
        obs = self.robot.capture_observation()
        self._update(obs=obs)

        # policy_action looks like (dx, dy, dz, dp, dr, dy, gripper)
        delta_action, intervention_bool = action

        teleop_action = None
        if intervention_bool:
            # Capture leader joints, ignore max_relative_target, last joint is gripper
            laeder_joint_positions = self.robot.leader_arms["main"].read("Present_Position")
            teleop_action = self._compute_delta_action(laeder_joint_positions)
            delta_action = teleop_action.copy()

        delta_action = np.clip(delta_action, self.action_space[0].low[0], self.action_space[0].high[0])

        # from [-1, 1] to step size
        delta_action *= self.action_scale

        # Gripper from [-gripper_step_size, gripper_step_size] -> [0, 2 * gripper_step_size]
        delta_action[6] += self.ee_action_space_params["gripper_step_size"]

        delta_action[6] = 0.0

        if isinstance(delta_action, torch.Tensor):
            delta_action = delta_action.cpu().numpy()

        # Update next pose
        self.next_tcp_pose = self.curr_tcp_pose.copy()
        self.next_tcp_pose[:3, 3] = self.next_tcp_pose[:3, 3] + delta_action[:3]

        # Get orientation from action
        self.next_tcp_pose[:3, :3] = (
            R.from_euler("xyz", delta_action[3:6], degrees=False)
            * R.from_matrix(self.curr_tcp_pose[:3, :3])
        ).as_matrix()

        desired_tcp_pose = self._clip_safety_box(self.next_tcp_pose)
        target_joint_positions = self.ik(
            current_joint_state=self.curr_q,
            desired_ee_pose=desired_tcp_pose,
            fk_func=self.fk,
            gripper_pos=delta_action[6]
        )
        self.robot.send_action(torch.from_numpy(target_joint_positions))

        self.current_step += 1

        reward = 0.0
        terminated = False
        truncated = False

        if self.display_lag:
            self.current_history.append(self.curr_tcp_pose)
            self.target_history.append(desired_tcp_pose)

        return (
            obs,
            reward,
            terminated,
            truncated,
            {
                "action_intervention": teleop_action,
                "is_intervention": teleop_action is not None,
            },
        )

    def render(self):
        """
        Render the current state of the environment by displaying the robot's camera feeds.
        """
        import cv2

        if self.display_cameras:
            observation = self.robot.capture_observation()
            image_keys = [key for key in observation if "image" in key]

            for key in image_keys:
                cv2.imshow(key, cv2.cvtColor(observation[key].numpy(), cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

        if self.display_lag:
            labels = ['X', 'Y', 'Z']
            if len(self.current_history) > 0:
                current_arr = np.array(self.current_history)
                target_arr = np.array(self.target_history)
                for i in range(3):
                    self.pos_axs[i].clear()
                    self.pos_axs[i].plot(current_arr[:, i, 3], label='Current')
                    self.pos_axs[i].plot(target_arr[:, i, 3], label='Target')
                    self.pos_axs[i].set_ylabel(labels[i])
                    self.pos_axs[i].axhline(y=self.xyz_bounding_box.low[i], color='k', linestyle='--')
                    self.pos_axs[i].axhline(y=self.xyz_bounding_box.high[i], color='k', linestyle='--')
                    self.pos_axs[i].legend()
                self.pos_axs[2].set_xlabel('Timesteps')

            labels = ['RX', 'RY', 'RZ']
            if len(self.current_history) > 0:
                current_arr = R.from_matrix(np.array(self.current_history)[:, :3, :3]).as_euler("xyz")
                target_arr = R.from_matrix(np.array(self.target_history)[:, :3, :3]).as_euler("xyz")
                for i in range(3):
                    self.rot_axs[i].clear()
                    self.rot_axs[i].plot(current_arr[:, i], label='Current')
                    self.rot_axs[i].plot(target_arr[:, i], label='Target')
                    self.rot_axs[i].set_ylabel(labels[i])
                    self.rot_axs[i].axhline(y=self.rpy_bounding_box.low[i], color='k', linestyle='--')
                    self.rot_axs[i].axhline(y=self.rpy_bounding_box.high[i], color='k', linestyle='--')
                    self.rot_axs[i].legend()
                self.rot_axs[2].set_xlabel('Timesteps')
                plt.pause(0.003)

    def close(self):
        """
        Close the environment and clean up resources by disconnecting the robot.

        If the robot is currently connected, this method properly terminates the connection to ensure that all
        associated resources are released.
        """
        if self.robot.is_connected:
            self.robot.disconnect()
        plt.ioff()

    def _update(self, obs=None):
        if obs is None:
            obs = self.robot.capture_observation()

        self.curr_q = obs['observation.state'].detach().cpu().numpy()
        self.curr_tcp_pose = self.fk(self.curr_q)

        q_dot = self.preprocess_q_dot(obs['observation.joint_vel'].detach().cpu().numpy())
        self.curr_tcp_vel = self.jacobian(self.curr_q, fk_func=self.fk) @ q_dot


    def _clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """Clip the pose to be within the safety box."""
        pose[:3, 3] = np.clip(
            pose[:3, 3], self.xyz_bounding_box.low, self.xyz_bounding_box.high
        )
        euler = R.from_matrix(pose[:3, :3]).as_euler("xyz")

        # Clip first euler angle separately due to discontinuity from pi to -pi
        sign = np.sign(euler[0])
        euler[0] = sign * (
            np.clip(
                np.abs(euler[0]),
                self.rpy_bounding_box.low[0],
                self.rpy_bounding_box.high[0],
            )
        )

        euler[1:] = np.clip(
            euler[1:], self.rpy_bounding_box.low[1:], self.rpy_bounding_box.high[1:]
        )
        pose[:3, :3] = R.from_euler("xyz", euler).as_matrix()
        return pose

    def _compute_delta_action(self, leader_joint_positions: np.ndarray) -> np.ndarray:
        """
        Compute the delta action needed to move from the current end-effector pose to the desired pose.

        Args:
            current_pose (np.ndarray): 4x4 homogeneous transformation of the current pose.
            desired_pose (np.ndarray): 4x4 homogeneous transformation of the desired pose.

        Returns:
            np.ndarray: A 7-element vector containing:
                [dx, dy, dz, droll, dpitch, dyaw, gripper]
                where the first 3 elements are the translation differences,
                the next 3 are the Euler angle differences (in radians),
                and the last element is the gripper action.
        """
        desired_pose = self.fk(leader_joint_positions)

        # Compute the position difference
        delta_pos = desired_pose[:3, 3] - self.curr_tcp_pose[:3, 3]

        # Compute the rotation difference
        relative_rot = R.from_matrix(desired_pose[:3, :3]) * R.from_matrix(self.curr_tcp_pose[:3, :3]).inv()
        # Convert the relative rotation to Euler angles (radians) using the 'xyz' convention
        delta_rpy = relative_rot.as_euler('xyz', degrees=False)

        # Concatenate into a single delta action vector
        gripper_pos = leader_joint_positions[-1]
        delta_action = np.concatenate([delta_pos, delta_rpy, [gripper_pos]])

        # Gripper from [0, 2 * gripper_step_size] -> [-gripper_step_size, gripper_step_size]
        delta_action[6] -= self.ee_action_space_params["gripper_step_size"]

        # Map back to [-1, 1]
        delta_action /= self.action_scale

        # Respect bounds while preserving directions
        delta_action[:3] = self._clip_delta_teleop_action_component(delta_action[:3])
        delta_action[3:6] = self._clip_delta_teleop_action_component(delta_action[3:6])
        delta_action[6] = np.clip(delta_action[6], -1.0, 1.0)

        return delta_action

    def _clip_delta_teleop_action_component(self, action_component):
        max_val = np.max(np.abs(action_component))
        if max_val > 1.0:
            action_component /= max_val
        return action_component


class ActionRepeatWrapper(gym.Wrapper):
    def __init__(self, env, nb_repeat: int = 1):
        super().__init__(env)
        self.nb_repeat = nb_repeat

    def step(self, action):
        for _ in range(self.nb_repeat):
            obs, reward, done, truncated, info = self.env.step(action)
            if done or truncated:
                break
        return obs, reward, done, truncated, info


class RewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_classifier, device: torch.device = "cuda"):
        """
        Wrapper to add reward prediction to the environment, it use a trained classifer.

        Args:
            env: The environment to wrap
            reward_classifier: The reward classifier model
            device: The device to run the model on
        """
        self.env = env

        # NOTE: We got 15% speedup by compiling the model
        self.reward_classifier = torch.compile(reward_classifier)

        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

    def step(self, action):
        observation, _, terminated, truncated, info = self.env.step(action)
        images = [
            observation[key].to(self.device, non_blocking=self.device.type == "cuda")
            for key in observation
            if "image" in key
        ]
        start_time = time.perf_counter()
        with torch.inference_mode():
            reward = (
                self.reward_classifier.predict_reward(images, threshold=0.8)
                if self.reward_classifier is not None
                else 0.0
            )
        info["Reward classifer frequency"] = 1 / (time.perf_counter() - start_time)

        if reward == 1.0:
            terminated = True
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)


class JointMaskingActionSpace(gym.Wrapper):
    def __init__(self, env, mask):
        """
        Wrapper to mask out dimensions of the action space.

        Args:
            env: The environment to wrap
            mask: Binary mask array where 0 indicates dimensions to remove
        """
        super().__init__(env)

        # Validate mask matches action space

        # Keep only dimensions where mask is 1
        self.active_dims = np.where(mask)[0]

        if isinstance(env.action_space, gym.spaces.Box):
            if len(mask) != env.action_space.shape[0]:
                raise ValueError("Mask length must match action space dimensions")
            low = env.action_space.low[self.active_dims]
            high = env.action_space.high[self.active_dims]
            self.action_space = gym.spaces.Box(
                low=low, high=high, dtype=env.action_space.dtype
            )

        if isinstance(env.action_space, gym.spaces.Tuple):
            if len(mask) != env.action_space[0].shape[0]:
                raise ValueError("Mask length must match action space 0 dimensions")

            low = env.action_space[0].low[self.active_dims]
            high = env.action_space[0].high[self.active_dims]
            action_space_masked = gym.spaces.Box(
                low=low, high=high, dtype=env.action_space[0].dtype
            )
            self.action_space = gym.spaces.Tuple(
                (action_space_masked, env.action_space[1])
            )
            # Create new action space with masked dimensions

    def action(self, action):
        """
        Convert masked action back to full action space.

        Args:
            action: Action in masked space. For Tuple spaces, the first element is masked.

        Returns:
            Action in original space with masked dims set to 0.
        """

        # Determine whether we are handling a Tuple space or a Box.
        if isinstance(self.env.action_space, gym.spaces.Tuple):
            # Extract the masked component from the tuple.
            masked_action = action[0] if isinstance(action, tuple) else action
            # Create a full action for the Box element.
            full_box_action = np.zeros(
                self.env.action_space[0].shape, dtype=self.env.action_space[0].dtype
            )
            full_box_action[self.active_dims] = masked_action
            # Return a tuple with the reconstructed Box action and the unchanged remainder.
            return (full_box_action, action[1])
        else:
            # For Box action spaces.
            masked_action = action if not isinstance(action, tuple) else action[0]
            full_action = np.zeros(
                self.env.action_space.shape, dtype=self.env.action_space.dtype
            )
            full_action[self.active_dims] = masked_action
            return full_action

    def step(self, action):
        action = self.action(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        if "action_intervention" in info and info["action_intervention"] is not None:
            if info["action_intervention"].dim() == 1:
                info["action_intervention"] = info["action_intervention"][
                    self.active_dims
                ]
            else:
                info["action_intervention"] = info["action_intervention"][
                    :, self.active_dims
                ]
        return obs, reward, terminated, truncated, info


class TimeLimitWrapper(gym.Wrapper):
    def __init__(self, env, control_time_s, fps):
        self.env = env
        self.control_time_s = control_time_s
        self.fps = fps

        self.last_timestamp = 0.0
        self.episode_time_in_s = 0.0

        self.max_episode_steps = int(self.control_time_s * self.fps)

        self.current_step = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        time_since_last_step = time.perf_counter() - self.last_timestamp
        self.episode_time_in_s += time_since_last_step
        self.last_timestamp = time.perf_counter()
        self.current_step += 1
        # check if last timestep took more time than the expected fps
        if 1.0 / time_since_last_step < self.fps:
            logging.debug(f"Current timestep exceeded expected fps {self.fps}")

        if self.current_step >= self.max_episode_steps:
            terminated = True
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.episode_time_in_s = 0.0
        self.last_timestamp = time.perf_counter()
        self.current_step = 0
        return self.env.reset(seed=seed, options=options)


class ImageCropResizeWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        crop_params_dict: Dict[str, Annotated[Tuple[int], 4]],
        resize_size=None,
    ):
        super().__init__(env)
        self.env = env
        self.crop_params_dict = crop_params_dict
        print(f"obs_keys , {self.env.observation_space}")
        print(f"crop params dict {crop_params_dict.keys()}")
        for key_crop in crop_params_dict:
            if key_crop not in self.env.observation_space.keys():  # noqa: SIM118
                raise ValueError(f"Key {key_crop} not in observation space")
        for key in crop_params_dict:
            top, left, height, width = crop_params_dict[key]
            new_shape = (top + height, left + width)
            self.observation_space[key] = gym.spaces.Box(
                low=0, high=255, shape=new_shape
            )

        self.resize_size = resize_size
        if self.resize_size is None:
            self.resize_size = (128, 128)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        for k in self.crop_params_dict:
            device = obs[k].device
            if obs[k].dim() >= 3:
                # Reshape to combine height and width dimensions for easier calculation
                batch_size = obs[k].size(0)
                channels = obs[k].size(1)
                flattened_spatial_dims = obs[k].view(batch_size, channels, -1)

                # Calculate standard deviation across spatial dimensions (H, W)
                std_per_channel = torch.std(flattened_spatial_dims, dim=2)

                # If any channel has std=0, all pixels in that channel have the same value
                if (std_per_channel <= 0.02).any():
                    logging.warning(
                        f"Potential hardware issue detected: All pixels have the same value in observation {k}"
                    )
            # Check for NaNs before processing
            if torch.isnan(obs[k]).any():
                logging.error(
                    f"NaN values detected in observation {k} before crop and resize"
                )

            if device == torch.device("mps:0"):
                obs[k] = obs[k].cpu()

            obs[k] = F.crop(obs[k], *self.crop_params_dict[k])
            obs[k] = F.resize(obs[k], self.resize_size)

            # Check for NaNs after processing
            if torch.isnan(obs[k]).any():
                logging.error(
                    f"NaN values detected in observation {k} after crop and resize"
                )

            obs[k] = obs[k].to(device)

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        for k in self.crop_params_dict:
            device = obs[k].device
            if device == torch.device("mps:0"):
                obs[k] = obs[k].cpu()
            obs[k] = F.crop(obs[k], *self.crop_params_dict[k])
            obs[k] = F.resize(obs[k], self.resize_size)
            obs[k] = obs[k].to(device)
        return obs, info


class ConvertToLeRobotObservation(gym.ObservationWrapper):
    def __init__(self, env, device):
        super().__init__(env)

        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

    def observation(self, observation):
        observation = preprocess_observation(observation)

        observation = {
            key: observation[key].to(
                self.device, non_blocking=self.device.type == "cuda"
            )
            for key in observation
        }
        observation = {
            k: torch.tensor(v, device=self.device) for k, v in observation.items()
        }
        return observation


class KeyboardInterfaceWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.listener = None
        self.events = {
            "exit_early": False,
            "pause_policy": False,
            "reset_env": False,
            "human_intervention_step": False,
            "episode_success": False,
            "leader_synced": False,
        }
        self.event_lock = Lock()  # Thread-safe access to events
        self.robot = self.unwrapped.robot
        self.leader_synced = False
        self._init_keyboard_listener()

    def _init_keyboard_listener(self):
        """Initialize keyboard listener if not in headless mode"""

        if is_headless():
            logging.warning(
                "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
            )
            return
        try:
            from pynput import keyboard

            def on_press(key):
                with self.event_lock:
                    try:
                        if key == keyboard.Key.right or key == keyboard.Key.esc:
                            print("Right arrow key pressed. Exiting loop...")
                            self.events["exit_early"] = True
                            return
                        if hasattr(key, "char") and key.char == "s":
                            print("Key 's' pressed. Episode success triggered.")
                            self.events["episode_success"] = True
                            return
                        if key == keyboard.Key.space and not self.events["exit_early"]:
                            if not self.events["pause_policy"]:
                                print(
                                    "Space key pressed. Human intervention required.\n"
                                    "Place the leader in similar pose to the follower and press space again."
                                )
                                self.events["pause_policy"] = True
                                log_say(
                                    "Human intervention stage. Get ready to take over.",
                                    play_sounds=True,
                                )
                                return
                            if (
                                self.events["pause_policy"]
                                and not self.events["human_intervention_step"]
                            ):
                                self.events["human_intervention_step"] = True
                                print("Space key pressed. Human intervention starting.")
                                log_say(
                                    "Starting human intervention.", play_sounds=True
                                )
                                return
                            if (
                                self.events["pause_policy"]
                                and self.events["human_intervention_step"]
                            ):
                                self.events["pause_policy"] = False
                                self.events["human_intervention_step"] = False
                                self.events["leader_synced"] = False
                                print("Space key pressed for a third time.")
                                log_say(
                                    "Continuing with policy actions.", play_sounds=True
                                )
                                return
                    except Exception as e:
                        print(f"Error handling key press: {e}")

            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.start()
        except ImportError:
            logging.warning(
                "Could not import pynput. Keyboard interface will not be available."
            )
            self.listener = None

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict]:
        is_intervention = False
        terminated_by_keyboard = False

        # Extract policy_action if needed
        if isinstance(self.env.action_space, gym.spaces.Tuple):
            policy_action = action[0]

        # Check the event flags without holding the lock for too long.
        with self.event_lock:
            if self.events["exit_early"]:
                terminated_by_keyboard = True
            pause_policy = self.events["pause_policy"]

        if pause_policy:

            # Move the leader arm to the follower position
            if not self.events["leader_synced"]:
                robot.leader_arms["main"].write("Torque_Enable", TorqueMode.ENABLED.value)
                reset_leader_position(robot, robot.follower_arms["main"].read("Present_Position"))

            # Now, wait for human_intervention_step without holding the lock
            while True:
                with self.event_lock:
                    if self.events["human_intervention_step"]:
                        is_intervention = True
                        break
                time.sleep(0.1)  # Check more frequently if desired

            # Torque the leader back off again
            if not self.events["leader_synced"]:
                robot.leader_arms["main"].write("Torque_Enable", TorqueMode.DISABLED.value)
                self.events["leader_synced"] = True

        # Execute the step in the underlying environment
        obs, reward, terminated, truncated, info = self.env.step(
            (policy_action, is_intervention)
        )

        # Override reward and termination if episode success event triggered
        with self.event_lock:
            if self.events["episode_success"]:
                reward = 1
                terminated_by_keyboard = True

        return obs, reward, terminated or terminated_by_keyboard, truncated, info

    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """
        Reset the environment and clear any pending events
        """
        with self.event_lock:
            self.events = {k: False for k in self.events}
        return self.env.reset(**kwargs)

    def close(self):
        """
        Properly clean up the keyboard listener when the environment is closed
        """
        if self.listener is not None:
            self.listener.stop()
        super().close()


class ResetWrapper(gym.Wrapper):
    def __init__(
        self,
        env: HILSerlRobotEnv,
        reset_pose: np.ndarray | None = None,
        reset_time_s: float = 5,
    ):
        super().__init__(env)
        self.reset_time_s = reset_time_s
        self.reset_pose = reset_pose
        self.robot = self.unwrapped.robot

    def reset(self, *, seed=None, options=None):
        if self.reset_pose is not None:
            start_time = time.perf_counter()
            log_say("Reset the environment.", play_sounds=True)
            reset_follower_position(self.robot, self.reset_pose)
            busy_wait(self.reset_time_s - (time.perf_counter() - start_time))
            log_say("Reset the environment done.", play_sounds=True)
        else:
            log_say(
                f"Manually reset the environment for {self.reset_time_s} seconds.",
                play_sounds=True,
            )
            start_time = time.perf_counter()
            while time.perf_counter() - start_time < self.reset_time_s:
                self.robot.teleop_step()

            log_say("Manual reseting of the environment done.", play_sounds=True)
        return super().reset(seed=seed, options=options)


class BatchCompatibleWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(
        self, observation: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        for key in observation:
            if "image" in key and observation[key].dim() == 3:
                observation[key] = observation[key].unsqueeze(0)
            if "state" in key and observation[key].dim() == 1:
                observation[key] = observation[key].unsqueeze(0)
            if "velocity" in key and observation[key].dim() == 1:
                observation[key] = observation[key].unsqueeze(0)
        return observation


class ActionScaleWrapper(gym.ActionWrapper):
    def __init__(self, env, ee_action_space_params=None):
        super().__init__(env)
        assert (
            ee_action_space_params is not None
        ), "TODO: method implemented for ee action space only so far"
        self.scale_vector = np.array(
            [
                [
                    ee_action_space_params.x_step_size,
                    ee_action_space_params.y_step_size,
                    ee_action_space_params.z_step_size,
                ]
            ]
        )

    def action(self, action):
        is_intervention = False
        if isinstance(action, tuple):
            action, is_intervention = action

        return action * self.scale_vector, is_intervention


def make_robot_env(
    robot,
    reward_classifier,
    cfg,
    n_envs: int = 1,
) -> gym.vector.VectorEnv:
    """
    Factory function to create a vectorized robot environment.

    Args:
        robot: Robot instance to control
        reward_classifier: Classifier model for computing rewards
        cfg: Configuration object containing environment parameters
        n_envs: Number of environments to create in parallel. Defaults to 1.

    Returns:
        A vectorized gym environment with all the necessary wrappers applied.
    """
    from omegaconf import OmegaConf

    if "maniskill" in cfg.env.name:
        from lerobot.scripts.server.maniskill_manipulator import make_maniskill

        logging.warning("WE SHOULD REMOVE THE MANISKILL BEFORE THE MERGE INTO MAIN")
        env = make_maniskill(
            cfg=cfg,
            n_envs=1,
        )
        return env

    # Create base environment
    env = HILSerlRobotEnv(
        robot=robot,
        display_cameras=cfg.env.wrapper.display_cameras,
        display_lag=cfg.env.wrapper.display_lag,
        ee_action_space_params=cfg.env.wrapper.ee_action_space_params
    )

    env = ConvertToLeRobotObservation(env=env, device=cfg.env.device)

    if cfg.env.wrapper.crop_params_dict is not None:
        env = ImageCropResizeWrapper(
            env=env,
            crop_params_dict=dict(cfg.env.wrapper.crop_params_dict),
            resize_size=list(cfg.env.wrapper.resize_size),
        )

    # Add reward computation and control wrappers
    # env = RewardWrapper(env=env, reward_classifier=reward_classifier, device=cfg.device)
    env = TimeLimitWrapper(env=env, control_time_s=cfg.env.wrapper.control_time_s, fps=cfg.fps)
    env = KeyboardInterfaceWrapper(env=env)
    env = ResetWrapper(
        env=env,
        reset_pose=list(cfg.env.wrapper.fixed_reset_joint_positions),
        reset_time_s=cfg.env.wrapper.reset_time_s,
    )
    env = BatchCompatibleWrapper(env=env)
    return env


def get_classifier(pretrained_path, config_path, device="mps"):
    if pretrained_path is None or config_path is None:
        return None

    from lerobot.common.policies.factory import _policy_cfg_from_hydra_cfg
    from lerobot.common.policies.hilserl.classifier.configuration_classifier import (
        ClassifierConfig,
    )
    from lerobot.common.policies.hilserl.classifier.modeling_classifier import (
        Classifier,
    )

    cfg = init_hydra_config(config_path)

    classifier_config = _policy_cfg_from_hydra_cfg(ClassifierConfig, cfg)
    classifier_config.num_cameras = len(
        cfg.training.image_keys
    )  # TODO automate these paths
    model = Classifier(classifier_config)
    model.load_state_dict(Classifier.from_pretrained(pretrained_path).state_dict())
    model = model.to(device)
    return model


def record_dataset(
    env,
    repo_id,
    root=None,
    num_episodes=1,
    control_time_s=20,
    fps=30,
    push_to_hub=True,
    task_description="",
    policy=None,
):
    """
    Record a dataset of robot interactions using either a policy or teleop.

    Args:
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
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    # Setup initial action (zero action if using teleop)
    dummy_action = env.action_space.sample()
    dummy_action = (torch.from_numpy(dummy_action[0] * 0.0), False)
    action = dummy_action

    # Configure dataset features based on environment spaces
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": env.observation_space["observation.state"].shape,
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
    for key in env.observation_space:
        if "image" in key:
            features[key] = {
                "dtype": "video",
                "shape": env.observation_space[key].shape,
                "names": None,
            }

    # Create dataset
    dataset = LeRobotDataset.create(
        repo_id,
        fps,
        root=root,
        use_videos=True,
        image_writer_threads=4,
        image_writer_processes=0,
        features=features,
    )

    # Record episodes
    episode_index = 0
    while episode_index < num_episodes:
        obs, _ = env.reset()
        start_episode_t = time.perf_counter()
        log_say(f"Recording episode {episode_index}", play_sounds=True)

        # Run episode steps
        while time.perf_counter() - start_episode_t < control_time_s:
            start_loop_t = time.perf_counter()

            # Get action from policy if available
            if policy is not None:
                action = policy.select_action(obs)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Check if episode needs to be rerecorded
            if info.get("rerecord_episode", False):
                break

            # For teleop, get action from intervention
            if policy is None:
                action = {
                    "action": info["action_intervention"].cpu().squeeze(0).float()
                }

            # Process observation for dataset
            obs = {k: v.cpu().squeeze(0).float() for k, v in obs.items()}

            # Add frame to dataset
            frame = {**obs, **action}
            frame["next.reward"] = reward
            frame["next.done"] = terminated or truncated
            dataset.add_frame(frame)

            # Maintain consistent timing
            if fps:
                dt_s = time.perf_counter() - start_loop_t
                busy_wait(1 / fps - dt_s)

            if terminated or truncated:
                break

        # Handle episode recording
        if info.get("rerecord_episode", False):
            dataset.clear_episode_buffer()
            logging.info(f"Re-recording episode {episode_index}")
            continue

        dataset.save_episode(task_description)
        episode_index += 1

    # Finalize dataset
    dataset.consolidate(run_compute_stats=True)
    if push_to_hub:
        dataset.push_to_hub(repo_id)


def replay_episode(env, repo_id, root=None, episode=0):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    local_files_only = root is not None
    dataset = LeRobotDataset(
        repo_id, root=root, episodes=[episode], local_files_only=local_files_only
    )
    env.reset()

    actions = dataset.hf_dataset.select_columns("action")

    for idx in range(dataset.num_frames):
        start_episode_t = time.perf_counter()

        action = actions[idx]["action"][:4]
        env.step((action, False))
        # env.step((action / env.unwrapped.delta, False))

        dt_s = time.perf_counter() - start_episode_t
        busy_wait(1 / 10 - dt_s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=int, default=30, help="control frequency")
    parser.add_argument(
        "--robot-path",
        type=str,
        default="lerobot/configs/robot/koch.yaml",
        help="Path to robot yaml file used to instantiate the robot using `make_robot` factory function.",
    )
    parser.add_argument(
        "--robot-overrides",
        type=str,
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )
    parser.add_argument(
        "-p",
        "--pretrained-policy-name-or-path",
        help=(
            "Either the repo ID of a model hosted on the Hub or a path to a directory containing weights "
            "saved using `Policy.save_pretrained`. If not provided, the policy is initialized from scratch "
        ),
    )
    parser.add_argument(
        "--display-cameras",
        help=("Whether to display the camera feed while the rollout is happening"),
    )
    parser.add_argument(
        "--reward-classifier-pretrained-path",
        type=str,
        default=None,
        help="Path to the pretrained classifier weights.",
    )
    parser.add_argument(
        "--reward-classifier-config-file",
        type=str,
        default=None,
        help="Path to a yaml config file that is necessary to build the reward classifier model.",
    )
    parser.add_argument(
        "--env-path", type=str, default=None, help="Path to the env yaml file"
    )
    parser.add_argument(
        "--env-overrides",
        type=str,
        default=None,
        help="Overrides for the env yaml file",
    )
    parser.add_argument(
        "--control-time-s",
        type=float,
        default=60,
        help="Maximum episode length in seconds",
    )
    parser.add_argument(
        "--reset-follower-pos",
        type=int,
        default=1,
        help="Reset follower between episodes",
    )
    parser.add_argument(
        "--replay-repo-id",
        type=str,
        default=None,
        help="Repo ID of the episode to replay",
    )
    parser.add_argument(
        "--dataset-root", type=str, default=None, help="Root of the dataset to replay"
    )
    parser.add_argument(
        "--replay-episode", type=int, default=0, help="Episode to replay"
    )
    parser.add_argument(
        "--record-repo-id",
        type=str,
        default=None,
        help="Repo ID of the dataset to record",
    )
    parser.add_argument(
        "--record-num-episodes",
        type=int,
        default=1,
        help="Number of episodes to record",
    )
    parser.add_argument(
        "--record-episode-task",
        type=str,
        default="",
        help="Single line description of the task to record",
    )

    args = parser.parse_args()

    robot_cfg = init_hydra_config(args.robot_path, args.robot_overrides)
    robot = make_robot(robot_cfg)

    reward_classifier = None
    #reward_classifier = get_classifier(
    #    args.reward_classifier_pretrained_path, args.reward_classifier_config_file
    #)
    user_relative_joint_positions = True

    cfg = init_hydra_config(args.env_path, args.env_overrides)
    env = make_robot_env(
        robot,
        reward_classifier,
        cfg,  # .wrapper,
    )

    if args.record_repo_id is not None:
        policy = None
        if args.pretrained_policy_name_or_path is not None:
            from lerobot.common.policies.sac.modeling_sac import SACPolicy

            policy = SACPolicy.from_pretrained(args.pretrained_policy_name_or_path)
            policy.to(cfg.device)
            policy.eval()

        record_dataset(
            env,
            args.record_repo_id,
            root=args.dataset_root,
            num_episodes=args.record_num_episodes,
            fps=args.fps,
            task_description=args.record_episode_task,
            policy=policy,
        )
        exit()

    if args.replay_repo_id is not None:
        replay_episode(
            env,
            args.replay_repo_id,
            root=args.dataset_root,
            episode=args.replay_episode,
        )
        exit()

    # tune pid gains
    robot.follower_arms["main"].write("Position_I_Gain", [50] * 9)

    env.reset()

    # Retrieve the robot's action space for joint commands.
    action_space_robot = env.action_space.spaces[0]

    # Initialize the smoothed action as a random sample.
    smoothed_action = action_space_robot.sample()

    # Smoothing coefficient (alpha) defines how much of the new random sample to mix in.
    # A value close to 0 makes the trajectory very smooth (slow to change), while a value close to 1 is less smooth.
    alpha = 0.25

    num_episode = 0
    sucesses = []
    while num_episode < 20:
        start_loop_s = time.perf_counter()
        # Sample a new random action from the robot's action space.
        new_random_action = action_space_robot.sample()
        # Update the smoothed action using an exponential moving average.
        smoothed_action = alpha * new_random_action + (1 - alpha) * smoothed_action

        # Execute the step: wrap the NumPy action in a torch tensor.
        obs, reward, terminated, truncated, info = env.step(
            (torch.from_numpy(smoothed_action), False)
        )
        if terminated or truncated:
            sucesses.append(reward)
            env.reset()
            num_episode += 1

        env.render()

        dt_s = time.perf_counter() - start_loop_s
        busy_wait(1 / args.fps - dt_s)

    logging.info(f"Success after 20 steps {sucesses}")
    logging.info(f"success rate {sum(sucesses)/ len(sucesses)}")
