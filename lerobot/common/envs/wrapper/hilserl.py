import logging
import sys
import time
from collections import deque
from threading import Lock
from typing import Annotated, Any, Dict, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
import torchvision.transforms.functional as F  # noqa: N812
from scipy.spatial.transform import Rotation as R

from lerobot.common.robot_devices.control_utils import (
    busy_wait,
    is_headless,
    reset_follower_position,
)
from lerobot.common.utils.utils import log_say
from lerobot.scripts.server.kinematics import get_kinematics

logging.basicConfig(level=logging.INFO)
MAX_GRIPPER_COMMAND = 40


class AddJointVelocityToObservation(gym.ObservationWrapper):
    def __init__(self, env, joint_velocity_limits=100.0, fps=30, num_dof=6):
        super().__init__(env)

        # Extend observation space to include joint velocities
        old_low = self.observation_space["observation.state"].low
        old_high = self.observation_space["observation.state"].high
        old_shape = self.observation_space["observation.state"].shape

        self.last_joint_positions = np.zeros(num_dof)

        new_low = np.concatenate([old_low, np.ones(num_dof) * -joint_velocity_limits])
        new_high = np.concatenate([old_high, np.ones(num_dof) * joint_velocity_limits])

        new_shape = (old_shape[0] + num_dof,)

        self.observation_space["observation.state"] = gym.spaces.Box(
            low=new_low,
            high=new_high,
            shape=new_shape,
            dtype=np.float32,
        )

        self.dt = 1.0 / fps

    def observation(self, observation):
        joint_velocities = (observation["observation.state"] - self.last_joint_positions) / self.dt
        self.last_joint_positions = observation["observation.state"].clone()
        observation["observation.state"] = torch.cat(
            [observation["observation.state"], joint_velocities], dim=-1
        )
        return observation


class AddCurrentToObservation(gym.ObservationWrapper):
    def __init__(self, env, max_current=500, num_dof=6):
        super().__init__(env)

        # Extend observation space to include joint velocities
        old_low = self.observation_space["observation.state"].low
        old_high = self.observation_space["observation.state"].high
        old_shape = self.observation_space["observation.state"].shape

        new_low = np.concatenate([old_low, np.zeros(num_dof)])
        new_high = np.concatenate([old_high, np.ones(num_dof) * max_current])

        new_shape = (old_shape[0] + num_dof,)

        self.observation_space["observation.state"] = gym.spaces.Box(
            low=new_low,
            high=new_high,
            shape=new_shape,
            dtype=np.float32,
        )

    def observation(self, observation):
        present_current = (
            self.unwrapped.robot.follower_arms["main"].read("Present_Current").astype(np.float32)
        )
        observation["observation.state"] = torch.cat(
            [observation["observation.state"], torch.from_numpy(present_current)], dim=-1
        )
        return observation


class RewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_classifier, device: torch.device = "cuda"):
        """
        Wrapper to add reward prediction to the environment, it use a trained classifier.

        cfg.
            env: The environment to wrap
            reward_classifier: The reward classifier model
            device: The device to run the model on
        """
        self.env = env

        self.device = device

        self.reward_classifier = torch.compile(reward_classifier)
        self.reward_classifier.to(self.device)

    def step(self, action):
        observation, _, terminated, truncated, info = self.env.step(action)
        images = {
            key: observation[key].to(self.device, non_blocking=self.device.type == "cuda")
            for key in observation
            if "image" in key
        }
        start_time = time.perf_counter()
        with torch.inference_mode():
            success = (
                self.reward_classifier.predict_reward(images, threshold=0.8)
                if self.reward_classifier is not None
                else 0.0
            )
        info["Reward classifier frequency"] = 1 / (time.perf_counter() - start_time)

        if success == 1.0:
            terminated = True
            reward = 1.0

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)


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
            new_shape = (3, resize_size[0], resize_size[1])
            self.observation_space[key] = gym.spaces.Box(low=0, high=255, shape=new_shape)

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
                # If any channel has std=0, all pixels in that channel have the same value
                # This is helpful if one camera mistakenly covered or the image is black
                std_per_channel = torch.std(flattened_spatial_dims, dim=2)
                if (std_per_channel <= 0.02).any():
                    logging.warning(
                        f"Potential hardware issue detected: All pixels have the same value in observation {k}"
                    )

            if device == torch.device("mps:0"):
                obs[k] = obs[k].cpu()

            obs[k] = F.crop(obs[k], *self.crop_params_dict[k])
            obs[k] = F.resize(obs[k], self.resize_size)
            # TODO (michel-aractingi): Bug in resize, it returns values outside [0, 1]
            obs[k] = obs[k].clamp(0.0, 1.0)
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
            obs[k] = obs[k].clamp(0.0, 1.0)
            obs[k] = obs[k].to(device)
        return obs, info


class ConvertToLeRobotObservation(gym.ObservationWrapper):
    def __init__(self, env, device: str = "cpu"):
        super().__init__(env)

        self.device = torch.device(device)

    def observation(self, observation):
        for key in observation:
            observation[key] = observation[key].float()
            if "image" in key:
                observation[key] = observation[key].permute(2, 0, 1)
                observation[key] /= 255.0
        observation = {
            key: observation[key].to(self.device, non_blocking=self.device.type == "cuda")
            for key in observation
        }

        return observation


class ResetWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        reset_pose: np.ndarray | None = None,
        reset_time_s: float = 5,
    ):
        super().__init__(env)
        self.reset_time_s = reset_time_s
        self.reset_pose = reset_pose
        self.robot = self.unwrapped.robot

    def reset(self, *, seed=None, options=None):
        start_time = time.perf_counter()
        if self.reset_pose is not None:
            log_say("Reset the environment.", play_sounds=True)
            reset_follower_position(self.robot.follower_arms["main"], self.reset_pose)
            log_say("Reset the environment done.", play_sounds=True)

            if len(self.robot.leader_arms) > 0:
                self.robot.leader_arms["main"].write("Torque_Enable", 1)
                log_say("Reset the leader robot.", play_sounds=True)
                reset_follower_position(self.robot.leader_arms["main"], self.reset_pose)
                log_say("Reset the leader robot done.", play_sounds=True)
        else:
            log_say(
                f"Manually reset the environment for {self.reset_time_s} seconds.",
                play_sounds=True,
            )
            start_time = time.perf_counter()
            while time.perf_counter() - start_time < self.reset_time_s:
                self.robot.teleop_step()

            log_say("Manual reset of the environment done.", play_sounds=True)

        busy_wait(self.reset_time_s - (time.perf_counter() - start_time))

        return super().reset(seed=seed, options=options)


class BatchCompatibleWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for key in observation:
            if "image" in key and observation[key].dim() == 3:
                observation[key] = observation[key].unsqueeze(0)
            if "state" in key and observation[key].dim() == 1:
                observation[key] = observation[key].unsqueeze(0)
            if "velocity" in key and observation[key].dim() == 1:
                observation[key] = observation[key].unsqueeze(0)
        return observation


class GripperPenaltyWrapper(gym.RewardWrapper):
    def __init__(self, env, penalty: float = -0.1):
        super().__init__(env)
        self.penalty = penalty
        self.last_gripper_state = None

    def reward(self, reward, action):
        gripper_state_normalized = self.last_gripper_state / MAX_GRIPPER_COMMAND

        action_normalized = action - 1.0  # action / MAX_GRIPPER_COMMAND

        gripper_penalty_bool = (gripper_state_normalized < 0.5 and action_normalized > 0.5) or (
            gripper_state_normalized > 0.75 and action_normalized < -0.5
        )

        return reward + self.penalty * int(gripper_penalty_bool)

    def step(self, action):
        self.last_gripper_state = self.unwrapped.robot.follower_arms["main"].read("Present_Position")[-1]
        gripper_action = action[-1]
        obs, reward, terminated, truncated, info = self.env.step(action)
        gripper_penalty = self.reward(reward, gripper_action)

        info["discrete_penalty"] = gripper_penalty

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.last_gripper_state = None
        obs, info = super().reset(**kwargs)
        info["gripper_penalty"] = 0.0
        return obs, info


class GripperActionWrapper(gym.ActionWrapper):
    def __init__(self, env, quantization_threshold: float = 0.2, gripper_sleep: float = 0.0):
        super().__init__(env)
        self.quantization_threshold = quantization_threshold
        self.gripper_sleep = gripper_sleep
        self.last_gripper_action_time = 0.0
        self.last_gripper_action = None

    def action(self, action):
        if self.gripper_sleep > 0.0:
            if (
                self.last_gripper_action is not None
                and time.perf_counter() - self.last_gripper_action_time < self.gripper_sleep
            ):
                action[-1] = self.last_gripper_action
            else:
                self.last_gripper_action_time = time.perf_counter()
                self.last_gripper_action = action[-1]

        gripper_command = action[-1]
        # Gripper actions are between 0, 2
        # we want to quantize them to -1, 0 or 1
        gripper_command = gripper_command - 1.0

        if self.quantization_threshold is not None:
            # Quantize gripper command to -1, 0 or 1
            gripper_command = (
                np.sign(gripper_command) if abs(gripper_command) > self.quantization_threshold else 0.0
            )
        gripper_command = gripper_command * MAX_GRIPPER_COMMAND
        gripper_state = self.unwrapped.robot.follower_arms["main"].read("Present_Position")[-1]
        gripper_action = np.clip(gripper_state + gripper_command, 0, MAX_GRIPPER_COMMAND)
        action[-1] = gripper_action.item()
        return action

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.last_gripper_action_time = 0.0
        self.last_gripper_action = None
        return obs, info


class EEActionWrapper(gym.ActionWrapper):
    def __init__(self, env, ee_action_space_params=None, use_gripper=False):
        super().__init__(env)
        self.ee_action_space_params = ee_action_space_params
        self.use_gripper = use_gripper

        # Initialize kinematics instance for the appropriate robot type
        self.kinematics = get_kinematics(env.unwrapped.robot.config, robot_type="follower")
        self.fk_function = self.kinematics.fk_gripper_tip

        action_space_bounds = np.array(
            [
                ee_action_space_params.x_step_size,
                ee_action_space_params.y_step_size,
                ee_action_space_params.z_step_size,
            ]
        )
        if self.use_gripper:
            # gripper actions open at 2.0, and closed at 0.0
            min_action_space_bounds = np.concatenate([-action_space_bounds, [0.0]])
            max_action_space_bounds = np.concatenate([action_space_bounds, [2.0]])
        else:
            min_action_space_bounds = -action_space_bounds
            max_action_space_bounds = action_space_bounds

        self.action_space = gym.spaces.Box(
            low=min_action_space_bounds,
            high=max_action_space_bounds,
            shape=(3 + int(self.use_gripper),),
            dtype=np.float32,
        )

        self.bounds = ee_action_space_params.bounds

    def action(self, action):
        desired_ee_pos = np.eye(4)

        if self.use_gripper:
            gripper_command = action[-1]
            action = action[:-1]

        current_joint_pos = self.unwrapped.robot.follower_arms["main"].read("Present_Position")
        current_ee_pos = self.fk_function(current_joint_pos)
        desired_ee_pos[:3, 3] = np.clip(
            current_ee_pos[:3, 3] + action,
            self.bounds["min"],
            self.bounds["max"],
        )
        target_joint_pos = self.kinematics.ik(
            current_joint_pos,
            desired_ee_pos,
            position_only=True,
            fk_func=self.fk_function,
        )
        if self.use_gripper:
            target_joint_pos[-1] = gripper_command

        return target_joint_pos


class EEObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, ee_pose_limits):
        super().__init__(env)

        # Extend observation space to include end effector pose
        prev_space = self.observation_space["observation.state"]

        self.observation_space["observation.state"] = gym.spaces.Box(
            low=np.concatenate([prev_space.low, ee_pose_limits["min"]]),
            high=np.concatenate([prev_space.high, ee_pose_limits["max"]]),
            shape=(prev_space.shape[0] + 3,),
            dtype=np.float32,
        )

        # Initialize kinematics instance for the appropriate robot type
        robot_type = getattr(env.unwrapped.robot.config, "robot_type", "so100")
        self.kinematics = get_kinematics(env.unwrapped.robot.config, robot_type="follower")
        self.fk_function = self.kinematics.fk_gripper_tip

    def observation(self, observation):
        current_joint_pos = self.unwrapped.robot.follower_arms["main"].read("Present_Position")
        current_ee_pos = self.fk_function(current_joint_pos)
        observation["observation.state"] = torch.cat(
            [
                observation["observation.state"],
                torch.from_numpy(current_ee_pos[:3, 3]),
            ],
            dim=-1,
        )
        return observation


###########################################################
# Wrappers related to human intervention and input devices
###########################################################


class BaseLeaderControlWrapper(gym.Wrapper):
    """Base class for leader-follower robot control wrappers."""

    def __init__(
        self, env, use_geared_leader_arm: bool = False, ee_action_space_params=None, use_gripper=False
    ):
        super().__init__(env)
        self.robot_leader = env.unwrapped.robot.leader_arms["main"]
        self.robot_follower = env.unwrapped.robot.follower_arms["main"]
        self.use_geared_leader_arm = use_geared_leader_arm
        self.ee_action_space_params = ee_action_space_params
        self.use_ee_action_space = ee_action_space_params is not None
        self.use_gripper: bool = use_gripper

        # Set up keyboard event tracking
        self._init_keyboard_events()
        self.event_lock = Lock()  # Thread-safe access to events

        # Initialize robot control
        self.kinematics = get_kinematics(env.unwrapped.robot.config, robot_type="leader")
        self.prev_leader_ee = None
        self.prev_leader_pos = None
        self.leader_torque_enabled = True

        # Configure leader arm
        # NOTE: Lower the gains of leader arm for automatic take-over
        # With lower gains we can manually move the leader arm without risk of injury to ourselves or the robot
        # With higher gains, it would be dangerous and difficult to modify the leader's pose while torque is enabled
        # Default value for P_coeff is 32
        self.robot_leader.write("Torque_Enable", 1)
        self.robot_leader.write("P_Coefficient", 4)
        self.robot_leader.write("I_Coefficient", 0)
        self.robot_leader.write("D_Coefficient", 4)

        self._init_keyboard_listener()

    def _init_keyboard_events(self):
        """Initialize the keyboard events dictionary - override in subclasses."""
        self.keyboard_events = {
            "episode_success": False,
            "episode_end": False,
            "rerecord_episode": False,
        }

    def _handle_key_press(self, key, keyboard):
        """Handle key presses - override in subclasses for additional keys."""
        try:
            if key == keyboard.Key.esc:
                self.keyboard_events["episode_end"] = True
                return
            if key == keyboard.Key.left:
                self.keyboard_events["rerecord_episode"] = True
                return
            if hasattr(key, "char") and key.char == "s":
                logging.info("Key 's' pressed. Episode success triggered.")
                self.keyboard_events["episode_success"] = True
                return
        except Exception as e:
            logging.error(f"Error handling key press: {e}")

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
                    self._handle_key_press(key, keyboard)

            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.start()

        except ImportError:
            logging.warning("Could not import pynput. Keyboard interface will not be available.")
            self.listener = None

    def _check_intervention(self):
        """Check if intervention is needed - override in subclasses."""
        return False

    def _handle_intervention(self, action):
        """Process actions during intervention mode."""
        if self.leader_torque_enabled:
            self.robot_leader.write("Torque_Enable", 0)
            self.leader_torque_enabled = False

        leader_pos = self.robot_leader.read("Present_Position")
        follower_pos = self.robot_follower.read("Present_Position")

        # [:3, 3] Last column of the transformation matrix corresponds to the xyz translation
        leader_ee = self.kinematics.fk_gripper_tip(leader_pos)[:3, 3]
        follower_ee = self.kinematics.fk_gripper_tip(follower_pos)[:3, 3]

        if self.prev_leader_ee is None:
            self.prev_leader_ee = leader_ee

        # NOTE: Using the leader's position delta for teleoperation is too noisy
        # Instead, we move the follower to match the leader's absolute position,
        # and record the leader's position changes as the intervention action
        action = leader_ee - follower_ee
        action_intervention = leader_ee - self.prev_leader_ee
        self.prev_leader_ee = leader_ee

        if self.use_gripper:
            # Get gripper action delta based on leader pose
            leader_gripper = leader_pos[-1]
            follower_gripper = follower_pos[-1]
            gripper_delta = leader_gripper - follower_gripper

            # Normalize by max angle and quantize to {0,1,2}
            normalized_delta = gripper_delta / MAX_GRIPPER_COMMAND
            if normalized_delta > 0.3:
                gripper_action = 2
            elif normalized_delta < -0.3:
                gripper_action = 0
            else:
                gripper_action = 1

            action = np.append(action, gripper_action)
            action_intervention = np.append(action_intervention, gripper_delta)

        return action, action_intervention

    def _handle_leader_teleoperation(self):
        """Handle leader teleoperation (non-intervention) operation."""
        if not self.leader_torque_enabled:
            self.robot_leader.write("Torque_Enable", 1)
            self.leader_torque_enabled = True

        follower_pos = self.robot_follower.read("Present_Position")
        self.robot_leader.write("Goal_Position", follower_pos)

    def step(self, action):
        """Execute environment step with possible intervention."""
        is_intervention = self._check_intervention()
        action_intervention = None

        # NOTE:
        if is_intervention:
            action, action_intervention = self._handle_intervention(action)
        else:
            self._handle_leader_teleoperation()

        # NOTE:
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Add intervention info
        info["is_intervention"] = is_intervention
        info["action_intervention"] = action_intervention if is_intervention else None

        # Check for success or manual termination
        success = self.keyboard_events["episode_success"]
        terminated = terminated or self.keyboard_events["episode_end"] or success

        if success:
            reward = 1.0
            logging.info("Episode ended successfully with reward 1.0")

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment and internal state."""
        self.prev_leader_ee = None
        self.prev_leader_pos = None
        self.keyboard_events = dict.fromkeys(self.keyboard_events, False)
        return super().reset(**kwargs)

    def close(self):
        """Clean up resources."""
        if hasattr(self, "listener") and self.listener is not None:
            self.listener.stop()
        return self.env.close()


class GearedLeaderControlWrapper(BaseLeaderControlWrapper):
    """Wrapper that enables manual intervention via keyboard."""

    def _init_keyboard_events(self):
        """Initialize keyboard events including human intervention flag."""
        super()._init_keyboard_events()
        self.keyboard_events["human_intervention_step"] = False

    def _handle_key_press(self, key, keyboard):
        """Handle key presses including space for intervention toggle."""
        super()._handle_key_press(key, keyboard)
        if key == keyboard.Key.space:
            if not self.keyboard_events["human_intervention_step"]:
                logging.info(
                    "Space key pressed. Human intervention required.\n"
                    "Place the leader in similar pose to the follower and press space again."
                )
                self.keyboard_events["human_intervention_step"] = True
                log_say("Human intervention step.", play_sounds=True)
            else:
                self.keyboard_events["human_intervention_step"] = False
                logging.info("Space key pressed for a second time.\nContinuing with policy actions.")
                log_say("Continuing with policy actions.", play_sounds=True)

    def _check_intervention(self):
        """Check if human intervention is active."""
        return self.keyboard_events["human_intervention_step"]


class GearedLeaderAutomaticControlWrapper(BaseLeaderControlWrapper):
    """Wrapper with automatic intervention based on error thresholds."""

    def __init__(
        self,
        env,
        ee_action_space_params=None,
        use_gripper=False,
        intervention_threshold=1.7,
        release_threshold=0.01,
        queue_size=10,
    ):
        super().__init__(env, ee_action_space_params=ee_action_space_params, use_gripper=use_gripper)

        # Error tracking parameters
        self.intervention_threshold = intervention_threshold  # Threshold to trigger intervention
        self.release_threshold = release_threshold  # Threshold to release intervention
        self.queue_size = queue_size  # Number of error measurements to keep

        # Error tracking variables
        self.error_queue = deque(maxlen=self.queue_size)
        self.error_over_time_queue = deque(maxlen=self.queue_size)
        self.previous_error = 0.0
        self.is_intervention_active = False
        self.start_time = time.perf_counter()

    def _check_intervention(self):
        """Determine if intervention should occur based on leader-follower error."""
        # Skip intervention logic for the first few steps to collect data
        if time.perf_counter() - self.start_time < 1.0:  # Wait 1 second before enabling
            return False

        # Get current positions
        leader_positions = self.robot_leader.read("Present_Position")
        follower_positions = self.robot_follower.read("Present_Position")

        # Calculate error and error rate
        error = np.linalg.norm(leader_positions - follower_positions)
        error_over_time = np.abs(error - self.previous_error)

        # Add to queue for running average
        self.error_queue.append(error)
        self.error_over_time_queue.append(error_over_time)

        # Update previous error
        self.previous_error = error

        # Calculate averages if we have enough data
        if len(self.error_over_time_queue) >= self.queue_size:
            avg_error_over_time = np.mean(self.error_over_time_queue)

            # Debug info
            if self.is_intervention_active:
                logging.debug(f"Error rate during intervention: {avg_error_over_time:.4f}")

            # Determine if intervention should start or stop
            if not self.is_intervention_active and avg_error_over_time > self.intervention_threshold:
                # Transition to intervention mode
                self.is_intervention_active = True
                logging.info(f"Starting automatic intervention: error rate {avg_error_over_time:.4f}")

            elif self.is_intervention_active and avg_error_over_time < self.release_threshold:
                # End intervention mode
                self.is_intervention_active = False
                logging.info(f"Ending automatic intervention: error rate {avg_error_over_time:.4f}")

        return self.is_intervention_active

    def reset(self, **kwargs):
        """Reset error tracking on environment reset."""
        self.error_queue.clear()
        self.error_over_time_queue.clear()
        self.previous_error = 0.0
        self.is_intervention_active = False
        self.start_time = time.perf_counter()
        return super().reset(**kwargs)


class GamepadControlWrapper(gym.Wrapper):
    """
    Wrapper that allows controlling a gym environment with a gamepad.

    This wrapper intercepts the step method and allows human input via gamepad
    to override the agent's actions when desired.
    """

    def __init__(
        self,
        env,
        x_step_size=1.0,
        y_step_size=1.0,
        z_step_size=1.0,
        use_gripper=False,
        auto_reset=False,
        input_threshold=0.001,
    ):
        """
        Initialize the gamepad controller wrapper.

        cfg.
            env: The environment to wrap
            x_step_size: Base movement step size for X axis in meters
            y_step_size: Base movement step size for Y axis in meters
            z_step_size: Base movement step size for Z axis in meters
            vendor_id: USB vendor ID of the gamepad (default: Logitech)
            product_id: USB product ID of the gamepad (default: RumblePad 2)
            auto_reset: Whether to auto reset the environment when episode ends
            input_threshold: Minimum movement delta to consider as active input
        """
        super().__init__(env)
        from lerobot.scripts.server.end_effector_control_utils import (
            GamepadController,
            GamepadControllerHID,
        )

        # use HidApi for macos
        if sys.platform == "darwin":
            self.controller = GamepadControllerHID(
                x_step_size=x_step_size,
                y_step_size=y_step_size,
                z_step_size=z_step_size,
            )
        else:
            self.controller = GamepadController(
                x_step_size=x_step_size,
                y_step_size=y_step_size,
                z_step_size=z_step_size,
            )
        self.auto_reset = auto_reset
        self.use_gripper = use_gripper
        self.input_threshold = input_threshold
        self.controller.start()

        logging.info("Gamepad control wrapper initialized")
        print("Gamepad controls:")
        print("  Left analog stick: Move in X-Y plane")
        print("  Right analog stick: Move in Z axis (up/down)")
        print("  X/Square button: End episode (FAILURE)")
        print("  Y/Triangle button: End episode (SUCCESS)")
        print("  B/Circle button: Exit program")

    def get_gamepad_action(
        self,
    ) -> Tuple[bool, np.ndarray, bool, bool, bool]:
        """
        Get the current action from the gamepad if any input is active.

        Returns:
            Tuple of (is_active, action, terminate_episode, success)
        """
        # Update the controller to get fresh inputs
        self.controller.update()

        # Get movement deltas from the controller
        delta_x, delta_y, delta_z = self.controller.get_deltas()

        intervention_is_active = self.controller.should_intervene()

        # Create action from gamepad input
        gamepad_action = np.array([delta_x, delta_y, delta_z], dtype=np.float32)

        if self.use_gripper:
            gripper_command = self.controller.gripper_command()
            if gripper_command == "open":
                gamepad_action = np.concatenate([gamepad_action, [2.0]])
            elif gripper_command == "close":
                gamepad_action = np.concatenate([gamepad_action, [0.0]])
            else:
                gamepad_action = np.concatenate([gamepad_action, [1.0]])

        # Check episode ending buttons
        # We'll rely on controller.get_episode_end_status() which returns "success", "failure", or None
        episode_end_status = self.controller.get_episode_end_status()
        terminate_episode = episode_end_status is not None
        success = episode_end_status == "success"
        rerecord_episode = episode_end_status == "rerecord_episode"

        return (
            intervention_is_active,
            gamepad_action,
            terminate_episode,
            success,
            rerecord_episode,
        )

    def step(self, action):
        """
        Step the environment, using gamepad input to override actions when active.

        cfg.
            action: Original action from agent

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Get gamepad state and action
        (
            is_intervention,
            gamepad_action,
            terminate_episode,
            success,
            rerecord_episode,
        ) = self.get_gamepad_action()

        # Update episode ending state if requested
        if terminate_episode:
            logging.info(f"Episode manually ended: {'SUCCESS' if success else 'FAILURE'}")

        # Only override the action if gamepad is active
        action = gamepad_action if is_intervention else action

        # Step the environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Add episode ending if requested via gamepad
        terminated = terminated or truncated or terminate_episode

        if success:
            reward = 1.0
            logging.info("Episode ended successfully with reward 1.0")

        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)

        info["is_intervention"] = is_intervention
        info["action_intervention"] = action
        info["rerecord_episode"] = rerecord_episode

        # If episode ended, reset the state
        if terminated or truncated:
            # Add success/failure information to info dict
            info["next.success"] = success

            # Auto reset if configured
            if self.auto_reset:
                obs, reset_info = self.reset()
                info.update(reset_info)

        return obs, reward, terminated, truncated, info

    def close(self):
        """Clean up resources when environment closes."""
        # Stop the controller
        if hasattr(self, "controller"):
            self.controller.stop()

        # Call the parent close method
        return self.env.close()


class TorchBox(gym.spaces.Box):
    """A version of gym.spaces.Box that handles PyTorch tensors.

    This class extends gym.spaces.Box to work with PyTorch tensors,
    providing compatibility between NumPy arrays and PyTorch tensors.
    """

    def __init__(
        self,
        low: float | Sequence[float] | np.ndarray,
        high: float | Sequence[float] | np.ndarray,
        shape: Sequence[int] | None = None,
        np_dtype: np.dtype | type = np.float32,
        torch_dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        seed: int | np.random.Generator | None = None,
    ) -> None:
        super().__init__(low, high, shape=shape, dtype=np_dtype, seed=seed)
        self.torch_dtype = torch_dtype
        self.device = device

    def sample(self) -> torch.Tensor:
        arr = super().sample()
        return torch.as_tensor(arr, dtype=self.torch_dtype, device=self.device)

    def contains(self, x: torch.Tensor) -> bool:
        # Move to CPU/numpy and cast to the internal dtype
        arr = x.detach().cpu().numpy().astype(self.dtype, copy=False)
        return super().contains(arr)

    def seed(self, seed: int | np.random.Generator | None = None):
        super().seed(seed)
        return [seed]

    def __repr__(self) -> str:
        return (
            f"TorchBox({self.low_repr}, {self.high_repr}, {self.shape}, "
            f"np={self.dtype.name}, torch={self.torch_dtype}, device={self.device})"
        )


class TorchActionWrapper(gym.Wrapper):
    """
    The goal of this wrapper is to change the action_space.sample()
    to torch tensors.
    """

    def __init__(self, env: gym.Env, device: str):
        super().__init__(env)
        self.action_space = TorchBox(
            low=env.action_space.low,
            high=env.action_space.high,
            shape=env.action_space.shape,
            torch_dtype=torch.float32,
            device=torch.device("cpu"),
        )

    def step(self, action: torch.Tensor):
        if action.dim() == 2:
            action = action.squeeze(0)
        action = action.detach().cpu().numpy()
        return self.env.step(action)


class StabilizingActionMaskingWrapper(gym.ActionWrapper):
    """
    A wrapper that:
    1. Restricts motion to a single axis (e.g., 'x').
    2. Stabilizes all other axes using a proportional controller.
    3. Expects a scalar action input corresponding to the selected axis.
    """

    def __init__(
        self,
        env,
        ax: list | float | None = None,
        ref_pose: np.ndarray = None,
        kp_pos: float = 0.1,
        kp_rot: float = 0.1,
    ):
        super().__init__(env)

        if ax is None:
            ax = 0
        if not isinstance(ax, list):
            ax = [ax]
        self.axes = ax

        self.kp_pos = kp_pos
        self.kp_rot = kp_rot

        # Reference pose: 3D position + 3D rotation as quaternion
        self.ref_pose = ref_pose  # np.ndarray with shape (7,)

        # Action space becomes 1D scalar for the selected axis
        low = [env.action_space[0].low[..., ax] for ax in self.axes]
        high = [env.action_space[0].high[..., ax] for ax in self.axes]
        self.action_space = gym.spaces.Tuple(
            spaces=(
                gym.spaces.Box(
                    low=np.array(low),
                    high=np.array(high),
                    shape=(len(self.axes),), dtype=np.float32
                ),
                gym.spaces.Discrete(2)  # keep teleop flag
            )
        )

    def step(self, action):
        if isinstance(action, tuple):
            action, telop = action
        else:
            if action.ndim > 1:
                action = action.squeeze(0)
            telop = 0

        # Initialize full action vector
        full_action = np.zeros_like(self.env.action_space[0].low).squeeze()

        if self.ref_pose is None:
            # Get current pose (position + quaternion) from environment
            current_pose = self.env.unwrapped.agent.tcp.pose.raw_pose.squeeze().numpy()
            self.ref_pose = current_pose.copy()

        # Decompose current and reference pose
        current_pos = self.env.unwrapped.agent.tcp.pose.p.squeeze().numpy()
        current_quat = self.env.unwrapped.agent.tcp.pose.q.squeeze().numpy()
        ref_pos = self.ref_pose[:3]
        ref_quat = self.ref_pose[3:]

        # --- Positional control ---
        delta_pos = np.zeros(3)

        # Stabilize other axes
        pos_error = ref_pos - current_pos
        pos_correction = self.kp_pos * pos_error
        delta_pos += pos_correction


        # --- Rotational control ---
        current_r = R.from_quat(current_quat)
        ref_r = R.from_quat(ref_quat)
        delta_r = (ref_r * current_r.inv()).as_rotvec()  # axis-angle difference
        delta_rot = self.kp_rot * delta_r

        full_action[:3] = delta_pos
        # full_action[3:6] = delta_rot

        for ax in self.axes:
            full_action[ax] = action[ax]

        return self.env.step((full_action, telop))