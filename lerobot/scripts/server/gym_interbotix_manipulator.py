import argparse
import logging
import time
from copy import copy
from threading import Lock
from typing import Annotated, Any, Callable, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torchvision.transforms.functional as F  # noqa: N812
from scipy.spatial.transform import Rotation as R

from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.robot_devices.control_utils import busy_wait, is_headless
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.robots.utils import construct_adjoint_matrix, construct_homogeneous_matrix
from lerobot.common.utils.utils import init_hydra_config, log_say

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

    Now works with ee poses and multiple arms at the same time
    """

    def __init__(
        self,
        robot,
        use_delta_action_space: bool = True,
        delta_action_scale: list[float] | None = None,
        display_cameras: bool = False,
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

        assert not use_delta_action_space or delta_action_scale is not None, "Specify delta_action_scale if a delta action space is used"
        assert delta_action_scale is None or len(delta_action_scale) == 3, "delta_action_scale must be None or a 3-sequence"

        self.robot = robot
        self.display_cameras = display_cameras

        # Connect to the robot if not already connected.
        if not self.robot.is_connected:
            self.robot.connect()

        self.init_pose = robot.capture_observation()['observation.state']
        self.current_pose = copy(self.init_pose)
        self.next_pose = copy(self.init_pose)
        self.num_follower = len(self.init_pose) / 7  # (x,y,z,r,p,y,gripper)

        # Episode tracking.
        self.current_step = 0
        self.episode_data = None

        self.delta_action_scale = delta_action_scale
        self.delta = np.array(self.num_follower * [delta_action_scale[0]] * 3 + delta_action_scale[1] * 3 + [delta_action_scale[2]])
        self.use_delta_action_space = use_delta_action_space

        # Retrieve the size of the joint position interval bound.
        self.relative_bound_high = torch.Tensor(self.robot.config.abs_pose_limit_high)
        self.relative_bound_low = torch.Tensor(self.robot.config.abs_pose_limit_low)
        self.relative_bounds_size = self.relative_bound_high - self.relative_bound_low
        self.delta_relative_bounds_size = self.relative_bounds_size * self.delta

        self.robot.config.max_relative_target = self.delta_relative_bounds_size.float()

        # Dynamically configure the observation and action spaces.
        self._setup_spaces()

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
                • The second element is a Discrete space (with 2 values) serving as a flag for intervention (teleoperation).
        """
        example_obs = self.robot.capture_observation()

        # Define observation spaces for images and other states.
        image_keys = [key for key in example_obs if "image" in key]
        state_keys = [key for key in example_obs if "image" not in key]
        observation_spaces = {
            key: gym.spaces.Box(
                low=0, high=255, shape=example_obs[key].shape, dtype=np.uint8
            )
            for key in image_keys
        }
        observation_spaces["observation.state"] = gym.spaces.Dict(
            {
                key: gym.spaces.Box(
                    low=0, high=10, shape=example_obs[key].shape, dtype=np.float32
                )
                for key in state_keys
            }
        )

        self.observation_space = gym.spaces.Dict(observation_spaces)

        # Define the action space for joint positions along with setting an intervention flag.
        action_dim = len(self.robot.capture_observation()["observation.space"])
        if self.use_delta_action_space:
            action_space_robot = gym.spaces.Box(
                low=-self.relative_bounds_size.cpu().numpy(),
                high=self.relative_bounds_size.cpu().numpy(),
                shape=(action_dim,),
                dtype=np.float32,
            )
        else:
            action_space_robot = gym.spaces.Box(
                low=self.robot.relative_bound_low.cpu().numpy(),
                high=self.robot.relative_bound_high.cpu().numpy(),
                shape=(action_dim,),
                dtype=np.float32,
            )

        self.action_space = gym.spaces.Tuple(
            (
                action_space_robot,
                gym.spaces.Discrete(2),
            ),
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
        observation = self.robot.capture_observation()

        # Reset episode tracking variables.
        self.current_step = 0
        self.episode_data = None
        self.current_pose = observation['observation.state']

        return observation, {"initial_position": self.init_pose}

    def step(
        self, action: Tuple[np.ndarray, bool]
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute a single step within the environment using the specified action.

        The provided action is a tuple comprised of:
            • A policy action (joint position commands) that may be either in absolute values or as a delta.
            • A boolean flag indicating whether teleoperation (human intervention) should be used for this step.

        Behavior:
            - When the intervention flag is False, the environment processes and sends the policy action to the robot.
            - When True, a teleoperation step is executed. If using a delta action space, an absolute teleop action is converted
              to relative change based on the current joint positions.

        Args:
            action (tuple): A tuple with two elements:
                - policy_action (np.ndarray or torch.Tensor): The commanded joint positions.
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
        policy_action, intervention_bool = action
        teleop_action = None


        if isinstance(policy_action, torch.Tensor):
            policy_action = policy_action.cpu().numpy()
            policy_action = np.clip(policy_action, self.action_space[0].low, self.action_space[0].high)

        if not intervention_bool:
            self.next_pose = copy(self.current_pose)

            if self.use_delta_action_space:
                self.next_pose = self.apply_delta_action(policy_action)
            else:
                self.next_pose = policy_action

            self.robot.send_action(torch.from_numpy(self.next_pose))
            observation = self.robot.capture_observation()
        else:
            observation, teleop_action = self.robot.teleop_step(record_data=True)
            teleop_action = teleop_action["action"]  # Convert tensor to appropriate format

            # When applying the delta action space, convert teleop absolute values to relative differences.
            if self.use_delta_action_space:
                teleop_action = self.revert_delta_action(teleop_action)

            # NOTE: To mimic the shape of a neural network output, we add a batch dimension to the teleop action.
            if teleop_action.dim() == 1:
                teleop_action = teleop_action.unsqueeze(0)

        # self.render()

        self.current_step += 1

        reward = 0.0
        terminated = False
        truncated = False

        return (
            observation,
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

        observation = self.robot.capture_observation()
        image_keys = [key for key in observation if "image" in key]

        for key in image_keys:
            cv2.imshow(key, cv2.cvtColor(observation[key].numpy(), cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

    def close(self):
        """
        Close the environment and clean up resources by disconnecting the robot.

        If the robot is currently connected, this method properly terminates the connection to ensure that all
        associated resources are released.
        """
        if self.robot.is_connected:
            self.robot.disconnect()

    def revert_delta_action(self, action):
        """
        Convert an absolute teleop action back into a delta action relative to the current pose.
        """
        from_idx = 0
        to_idx = 0
        delta_action = []
        for i in range(len(self.num_follower)):
            to_idx += 7
            single_action = action[from_idx:to_idx]
            from_idx = to_idx

            single_action = torch.from_numpy(single_action) if not isinstance(action, torch.Tensor) else action

            # Compute the delta for position
            delta_pos = (single_action[:3] - self.current_pose[:3]) / self.delta_action_scale[0]

            # Compute the delta for rotation using quaternions
            current_rot = R.from_euler("xyz", self.current_pose[3:6])
            target_rot = R.from_euler("xyz", single_action[3:6])
            delta_rot = (target_rot * current_rot.inv()).as_euler("xyz") / self.delta_action_scale[1]

            # Compute the delta for gripper
            delta_gripper = single_action[6] / self.delta_action_scale[2]

            # Concat to 7 element action
            single_delta_action = torch.concat([torch.tensor(delta_pos), torch.tensor(delta_rot), torch.tensor([delta_gripper])])

            # Compare against bounds and clamp if necessary
            if torch.any(single_delta_action < -self.relative_bounds_size) or torch.any(single_delta_action > self.relative_bounds_size):
                logging.debug(
                        f"Relative teleop delta exceeded for arm no. {i}, bounds {self.relative_bounds_size}, teleop_action {single_delta_action}\n"
                        f"lower bounds condition {single_delta_action < -self.relative_bounds_size}\n"
                        f"upper bounds condition {single_delta_action > self.relative_bounds_size}"
                    )

                single_delta_action = torch.clamp(
                        single_delta_action,
                        -self.relative_bounds_size,
                        self.relative_bounds_size,
                    )

            # Concatenate to form the final delta action
            delta_action.append(single_delta_action)

        delta_action = torch.cat(delta_action)
        return delta_action

    def apply_delta_action(self, action):
        from_idx = 0
        next_pose = copy(self.current_pose)
        for _ in range(len(self.num_follower)):
            next_pose = copy(self.current_pose)
            next_pose[from_idx:from_idx+3] = self.current_pose[from_idx:from_idx+3] + action[from_idx:from_idx+3] * self.delta_action_scale[0]
            next_pose[from_idx+3:from_idx+6] = (
                R.from_euler("xyz", action[from_idx+3:from_idx+6] * self.delta_action_scale[1])
                * R.from_euler("xyz", self.current_pose[from_idx+3:from_idx+6])
            )
            next_pose[from_idx+6] = action[from_idx+6] * self.delta_action_scale[2]
        return next_pose


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
        Wrapper to add reward prediction to the environment, it uses a trained classifer.

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


class MaskingActionSpace(gym.Wrapper):
    def __init__(self, env, mask):
        """
        Wrapper to mask out dimensions of the action space.

        Args:
            env: The environment to wrap
            mask: Binary mask array where 0 indicates dimensions to remove
        """
        super().__init__(env)

        # Keep only dimensions where mask is 1
        self.active_dims = np.where(mask)[0]

        # Create new action space with masked dimensions
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

    def mask_action(self, action):
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
        action = self.mask_action(action)
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

        if self.episode_time_in_s > self.control_time_s:
            # if self.current_step >= self.max_episode_steps:
            # Terminated = True
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
        }
        self.event_lock = Lock()  # Thread-safe access to events
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
            # Now, wait for human_intervention_step without holding the lock
            while True:
                with self.event_lock:
                    if self.events["human_intervention_step"]:
                        is_intervention = True
                        break
                time.sleep(0.1)  # Check more frequently if desired

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
        reset_fn: Optional[Callable[[], None]] = None,
        reset_time_s: float = 5,
    ):
        super().__init__(env)
        self.reset_fn = reset_fn
        self.reset_time_s = reset_time_s

        self.robot = self.unwrapped.robot
        self.init_pos = self.unwrapped.init_pose

    def reset(self, *, seed=None, options=None):
        if self.reset_fn is not None:
            self.reset_fn(self.env)
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
        return observation


class RelativeFrameWrapper(gym.ObservationWrapper):
    """
    This wrapper transforms the observation and action to be expressed in the end-effector frame.
    Optionally, it can transform the tcp_pose into a relative frame defined as the reset pose.

    This wrapper is expected to be used on top of the base Franka environment, which has the following
    observation space:
    {
        "state": spaces.Dict(
            {
                "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,)), # xyz + quat
                ......
            }
        ),
        ......
    }, and at least 6 DoF action space with (x, y, z, rx, ry, rz, ...)
    """

    def __init__(self, env, include_relative_pose=True):
        super().__init__(env)
        self.adjoint_matrix = np.zeros((6, 6))

        self.include_relative_pose = include_relative_pose
        if self.include_relative_pose:
            # Homogeneous transformation matrix from reset pose's relative frame to base frame
            self.T_r_o_inv = np.zeros((4, 4))

    def step(self, action: np.ndarray):
        # action is assumed to be (x, y, z, rx, ry, rz, gripper)
        # Transform action from end-effector frame to base frame
        transformed_action = self.transform_action(action)
        obs, reward, done, truncated, info = self.env.step(transformed_action)
        info['original_state_obs'] = copy.deepcopy({"tcp_pose": obs["observation.tcp_pose"], "tcp_vel": obs["observation.tcp_vel"]})

        # this is to convert the spacemouse intervention action
        if info["is_intervention"]:
            info["action_intervention"] = self.transform_action_inv(info["action_intervention"])

        # Update adjoint matrix
        self.adjoint_matrix = construct_adjoint_matrix(obs["observation.tcp_pose"])

        # Transform observation to spatial frame
        transformed_obs = self.transform_observation(obs)
        return transformed_obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['original_state_obs'] = copy.deepcopy({"tcp_pose": obs["observation.tcp_pose"], "tcp_vel": obs["observation.tcp_vel"]})

        # Update adjoint matrix
        self.adjoint_matrix = construct_adjoint_matrix(obs["observation.tcp_pose"])
        if self.include_relative_pose:
            # Update transformation matrix from the reset pose's relative frame to base frame
            self.T_r_o_inv = np.linalg.inv(
                construct_homogeneous_matrix(obs["observation.tcp_pose"])
            )

        # Transform observation to spatial frame
        return self.transform_observation(obs), info

    def transform_observation(self, obs):
        """
        Transform observations from spatial(base) frame into body(end-effector) frame
        using the adjoint matrix
        """
        adjoint_inv = np.linalg.inv(self.adjoint_matrix)
        obs["observation.tcp_vel"] = adjoint_inv @ obs["observation.tcp_vel"]

        if self.include_relative_pose:
            T_b_o = construct_homogeneous_matrix(obs["observation.tcp_vel"])
            T_b_r = self.T_r_o_inv @ T_b_o

            # Reconstruct transformed tcp_pose vector
            p_b_r = T_b_r[:3, 3]
            theta_b_r = R.from_matrix(T_b_r[:3, :3]).as_euler("xyz")
            obs["observation.tcp_vel"] = np.concatenate((p_b_r, theta_b_r))

        return obs

    def transform_action(self, action: np.ndarray):
        """
        Transform action from body(end-effector) frame into into spatial(base) frame
        using the adjoint matrix.
        """
        action = np.array(action)  # in case action is a jax read-only array
        action[:6] = self.adjoint_matrix @ action[:6]
        return action

    def transform_action_inv(self, action: np.ndarray):
        """
        Transform action from spatial(base) frame into body(end-effector) frame
        using the adjoint matrix.
        """
        action = np.array(action)
        action[:6] = np.linalg.inv(self.adjoint_matrix) @ action[:6]
        return action


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
    # Create base environment
    env = HILSerlRobotEnv(
        robot=robot,
        display_cameras=cfg.env.wrapper.display_cameras,
        delta=cfg.env.wrapper.delta_action,
        use_delta_action_space=cfg.env.wrapper.use_relative_joint_positions,
    )
    
    # transform to relative frame
    env = RelativeFrameWrapper(env)

    # Add observation and image processing
    env = ConvertToLeRobotObservation(env=env, device=cfg.device)
    if cfg.env.wrapper.crop_params_dict is not None:
        env = ImageCropResizeWrapper(
            env=env,
            crop_params_dict=cfg.env.wrapper.crop_params_dict,
            resize_size=cfg.env.wrapper.resize_size,
        )

    # Add reward computation and control wrappers
    env = RewardWrapper(env=env, reward_classifier=reward_classifier, device=cfg.device)
    env = TimeLimitWrapper(
        env=env, control_time_s=cfg.env.wrapper.control_time_s, fps=cfg.fps
    )
    env = KeyboardInterfaceWrapper(env=env)
    env = ResetWrapper(
        env=env, reset_fn=None, reset_time_s=cfg.env.wrapper.reset_time_s
    )
    env = MaskingActionSpace(
        env=env, mask=cfg.env.wrapper.joint_masking_action_space
    )
    env = BatchCompatibleWrapper(env=env)

    return env

    # batched version of the env that returns an observation of shape (b, c)


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


def replay_episode(env, repo_id, root=None, episode=0):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    local_files_only = root is not None
    dataset = LeRobotDataset(
        repo_id, root=root, episodes=[episode], local_files_only=local_files_only
    )
    actions = dataset.hf_dataset.select_columns("action")

    for idx in range(dataset.num_frames):
        start_episode_t = time.perf_counter()

        action = actions[idx]["action"][:4]
        print(action)
        env.step((action / env.unwrapped.delta, False))

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
            "(useful for debugging). This argument is mutually exclusive with `--config`."
        ),
    )
    parser.add_argument(
        "--config",
        help=(
            "Path to a yaml config you want to use for initializing a policy from scratch (useful for "
            "debugging). This argument is mutually exclusive with `--pretrained-policy-name-or-path` (`-p`)."
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
        default=20,
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
        "--replay-root", type=str, default=None, help="Root of the dataset to replay"
    )
    parser.add_argument(
        "--replay-episode", type=int, default=0, help="Episode to replay"
    )
    args = parser.parse_args()

    robot_cfg = init_hydra_config(args.robot_path, args.robot_overrides)
    robot = make_robot(robot_cfg)

    reward_classifier = get_classifier(
        args.reward_classifier_pretrained_path, args.reward_classifier_config_file
    )
    user_relative_joint_positions = True

    cfg = init_hydra_config(args.env_path, args.env_overrides)
    env = make_robot_env(
        robot,
        reward_classifier,
        cfg.env,  # .wrapper,
    )

    env.reset()

    if args.replay_repo_id is not None:
        replay_episode(
            env, args.replay_repo_id, root=args.replay_root, episode=args.replay_episode
        )
        exit()

    # Retrieve the robot's action space for joint commands.
    action_space_robot = env.action_space.spaces[0]

    # Initialize the smoothed action as a random sample.
    smoothed_action = action_space_robot.sample()

    # Smoothing coefficient (alpha) defines how much of the new random sample to mix in.
    # A value close to 0 makes the trajectory very smooth (slow to change), while a value close to 1 is less smooth.
    alpha = 0.4

    while True:
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
            env.reset()

        dt_s = time.perf_counter() - start_loop_s
        busy_wait(1 / args.fps - dt_s)
