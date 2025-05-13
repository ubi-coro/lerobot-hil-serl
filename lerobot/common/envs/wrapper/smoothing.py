import gymnasium as gym
import numpy as np
import torch
from sympy.physics.units import action


class SmoothActionWrapper(gym.Wrapper):
    """
    Clamps per-step delta actions to limit sudden direction changes,
    and appends the previous delta action to the observation.

    To be used after ManiSkillObservationWrapper.
    """
    def __init__(self,
                 env,
                 device,
                 smoothing_range_factor: float = 0.3,
                 smoothing_penalty: float = 0.0,
                 use_gripper: bool = False):
        super().__init__(env)

        self.device = device
        self.use_gripper = use_gripper

        # get action space bounds and compute the max delta as a fraction of high - low
        action_space = self.action_space[0] if isinstance(self.action_space, gym.spaces.Tuple) else self.action_space
        self.action_dim = action_space.shape
        self.smoothing_penalty = smoothing_penalty
        self.max_delta = smoothing_range_factor * action_space.high
        self.min_delta = smoothing_range_factor * action_space.low
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)

        # extend observation space to include end effector pose
        prev_obs_space = self.observation_space["observation.state"]
        self.observation_space["observation.state"] = gym.spaces.Box(
            low=np.concatenate([prev_obs_space.low, action_space.low]),
            high=np.concatenate([prev_obs_space.high, action_space.high]),
            shape=(prev_obs_space.shape[0] + action_space.shape[0],),
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_action[:] = 0.0
        obs = self._append_prev_action(obs)
        return obs, info

    def step(self, action):
        # Clamp the delta action
        delta_action = action - self.prev_action
        exceeds_limits = any(delta_action > self.max_delta) or any(delta_action < self.min_delta)
        if exceeds_limits:
            delta_action = np.clip(delta_action, self.min_delta, self.max_delta)

        smoothed_action = self.prev_action + delta_action

        if self.use_gripper:
            smoothed_action[-1] = action[-1]

        self.prev_action = smoothed_action.copy()

        obs, reward, terminated, truncated, info = self.env.step(smoothed_action)
        obs = self._append_prev_action(obs)
        if exceeds_limits:
            reward -= self.smoothing_penalty
        return obs, reward, terminated, truncated, info

    def _append_prev_action(self, obs):
        # Append previous delta action to observation.state
        if "observation.state" in obs:
            state_tensor = obs["observation.state"]
            if not torch.is_tensor(state_tensor):
                state_tensor = torch.tensor(state_tensor, dtype=torch.float32)
            prev_tensor = torch.tensor(self.prev_action, dtype=torch.float32).to(self.device)
            obs["observation.state"] = torch.cat([state_tensor, prev_tensor], dim=-1)
        return obs