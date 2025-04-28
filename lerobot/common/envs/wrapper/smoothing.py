import gymnasium as gym

class SmoothActionWrapper(gym.Wrapper):
    """
    Clamps per-step delta actions to limit sudden direction changes,
    and appends the previous delta action to the observation.

    To be used after ManiSkillObservationWrapper.
    """
    def __init__(self, env, device, max_delta_range_factor: float = 0.3):
        super().__init__(env)
        self.device = device

        action_space = self.env.action_space[0] \
            if isinstance(self.env.action_space, gym.spaces.Tuple) \
            else self.env.action_space
        self.action_dim = action_space.shape
        self.max_delta = max_delta_range_factor * action_space.high
        self.min_delta = max_delta_range_factor * action_space.low

        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_action[:] = 0.0
        obs = self._append_prev_action(obs)
        return obs, info

    def step(self, action):
        if isinstance(action, tuple):
            action, telop = action
        else:
            telop = 0

        # Clamp the delta action
        delta_action = action - self.prev_action
        exceeds_limits = any(delta_action > self.max_delta) or any(delta_action < self.min_delta)
        if exceeds_limits:
            print(f"Clamp {delta_action} to {self.max_delta}")
            delta_action = np.clip(delta_action, self.min_delta, self.max_delta)
        smoothed_action = self.prev_action + delta_action
        self.prev_action = smoothed_action.copy()

        obs, reward, terminated, truncated, info = self.env.step((smoothed_action, telop))
        obs = self._append_prev_action(obs)
        if exceeds_limits:
            reward -= 0.05
        return obs, reward, terminated, truncated, info

    def _append_prev_action(self, obs):
        # Append previous delta action to observation.state
        if "observation.state" in obs:
            state_tensor = obs["observation.state"]
            if not torch.is_tensor(state_tensor):
                state_tensor = torch.tensor(state_tensor, dtype=torch.float32)
            prev_tensor = torch.tensor(self.prev_action[np.newaxis, :], dtype=torch.float32).to(self.device)
            obs["observation.state"] = torch.cat([state_tensor, prev_tensor], dim=-1)
        return obs