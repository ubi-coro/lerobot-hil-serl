from typing import Any

import einops
import gymnasium as gym
import numpy as np
import torch
from pynput import keyboard


def preprocess_maniskill_observation(
    observations: dict[str, np.ndarray],
) -> dict[str, torch.Tensor]:
    """Convert environment observation to LeRobot format observation.
    Args:
        observation: Dictionary of observation batches from a Gym vector environment.
    Returns:
        Dictionary of observation batches with keys renamed to LeRobot format and values as tensors.
    """
    # map to expected inputs for the policy
    return_observations = {}
    # TODO: You have to merge all tensors from agent key and extra key
    # You don't keep sensor param key in the observation
    # And you keep sensor data rgb
    q_pos = observations["agent"]["qpos"]
    q_vel = observations["agent"]["qvel"]
    tcp_pos = observations["extra"]["tcp_pose"]

    imgs = {f"observation.{name}": observations["sensor_data"][name]["rgb"] for name in observations["sensor_data"]}
    is_multi_cam =  len(imgs) > 1

    for name in imgs:
        _, h, w, c = imgs[name].shape
        assert c < h and c < w, f"expect channel last images, but instead got {imgs[name].shape=}"

        # sanity check that images are uint8
        assert imgs[name].dtype == torch.uint8, f"expect torch.uint8, but instead {imgs[name].dtype=}"

        # convert to channel first of type float32 in range [0,1]
        imgs[name] = einops.rearrange(imgs[name], "b h w c -> b c h w").contiguous()
        imgs[name] = imgs[name].type(torch.float32)
        imgs[name] /= 255

    #state = torch.cat([q_pos, q_vel, tcp_pos], dim=-1)

    if is_multi_cam:
        return_observations = imgs
    else:
        return_observations["observation.image"] = imgs["base_camera"]
    return_observations["observation.state"] = tcp_pos[:, :3]
    return return_observations


class ManiSkillObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, device: torch.device = "cuda"):
        super().__init__(env)
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

    def observation(self, observation):
        observation = preprocess_maniskill_observation(observation)
        observation = {k: v.to(self.device) for k, v in observation.items()}
        return observation


class ManiSkillCompat(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        new_action_space_shape = env.action_space.shape[-1]
        new_low = np.squeeze(env.action_space.low, axis=0)
        new_high = np.squeeze(env.action_space.high, axis=0)
        self.action_space = gym.spaces.Box(low=new_low, high=new_high, shape=(new_action_space_shape,))

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        options = {}
        return super().reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = reward.item()
        terminated = terminated.item()
        truncated = truncated.item()
        return obs, reward, terminated, truncated, info


class ManiSkillActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Tuple(spaces=(env.action_space, gym.spaces.Discrete(2)))

    def action(self, action):
        action, telop = action
        return action


class ManiSkillMultiplyActionWrapper(gym.Wrapper):
    def __init__(self, env, multiply_factor: float = 1):
        super().__init__(env)
        self.multiply_factor = multiply_factor
        action_space_agent: gym.spaces.Box = env.action_space[0]
        action_space_agent.low = action_space_agent.low * multiply_factor
        action_space_agent.high = action_space_agent.high * multiply_factor
        self.action_space = gym.spaces.Tuple(spaces=(action_space_agent, gym.spaces.Discrete(2)))

    def step(self, action):
        if isinstance(action, tuple):
            action, telop = action
        else:
            telop = 0
        action = action / self.multiply_factor
        obs, reward, terminated, truncated, info = self.env.step((action, telop))
        return obs, reward, terminated, truncated, info


class BatchCompatibleWrapper(gym.ObservationWrapper):
    """Ensures observations are batch-compatible by adding a batch dimension if necessary."""

    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for key in observation:
            if "image" in key and observation[key].dim() == 3:
                observation[key] = observation[key].unsqueeze(0)
            if "state" in key and observation[key].dim() == 1:
                observation[key] = observation[key].unsqueeze(0)
        return observation


class KeyboardControlWrapper(gym.Wrapper):
    """
    A wrapper for controlling the robot's end-effector using the keyboard.
    Arrow keys are used to move the end-effector along the X-axis.
    """

    def __init__(self, env, ax: list | float | None = None):
        super().__init__(env)

        if ax is None:
            ax = 0
        if not isinstance(ax, list):
            ax = [ax]
        assert len(ax) <= 3, "Can only control three axes with KeyboardControlWrapper"
        self.axes = ax

        self.action = np.array([0.0, 0.0, 0.0])  # Default action value
        self.done = False
        self.intervention = False

        listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        listener.start()

    def _on_press(self, key):
        try:
            if key == keyboard.Key.right:
                self.action[0] = -0.03  # Move right (X-axis)
            elif key == keyboard.Key.left:
                self.action[0] = 0.03  # Move left (X-axis)
            elif key == keyboard.Key.up:
                self.action[1] = -0.03  # Move up (Y-axis)
            elif key == keyboard.Key.down:
                self.action[1] = 0.03  # Move down (Y-axis)
            elif key == keyboard.Key.page_up:
                self.action[2] = 0.03  # Move up (Z-axis) with Shift
            elif key == keyboard.Key.page_down:
                self.action[2] = -0.03  # Move down (Z-axis) with Shift
            elif key == keyboard.Key.space:
                self.intervention = True
            elif key == keyboard.Key.esc:
                self.done = True  # Exit the loop
        except AttributeError:
            pass

    def _on_release(self, key):
        if key in [keyboard.Key.right, keyboard.Key.left]:
            self.action[0] = 0.0  # Stop movement along X-axis
        if key in [keyboard.Key.up, keyboard.Key.down]:
            self.action[1] = 0.0  # Stop movement along Y-axis
        if key in [keyboard.Key.page_up, keyboard.Key.page_down]:
            self.action[2] = 0.0  # Stop movement along Z-axis
        if key == keyboard.Key.space:
            self.intervention = False

    def step(self, action):
        """
        Run the environment loop with keyboard control.
        """
        if isinstance(action, tuple):
            action, telop = action
        else:
            telop = self.intervention

        if telop:
            action = np.zeros_like(action)
            for i, ax in enumerate(self.axes):
                action[ax] = self.action[i]

        obs, reward, done, vec, info = self.env.step(action)

        if telop:
            info["is_intervention"] = bool(telop)
            info["action_intervention"] = torch.from_numpy(action)

        return obs, reward, self.done or done, vec, info