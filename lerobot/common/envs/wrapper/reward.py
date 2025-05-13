import gymnasium as gym
import numpy as np
import torch


class SuccessRepeatWrapper(gym.Wrapper):
    # only sets terminated if success (reward=1.0) for num_repeats times
    # also adds the current number of success to the input to keep the mdp markovian

    def __init__(self, env, num_repeats: int = 3):
        super().__init__(env)
        self.num_repeats = num_repeats
        self.num_success = 0

        prev_space = self.observation_space["observation.state"]
        self.observation_space["observation.state"] = gym.spaces.Box(
            low=np.concatenate([prev_space.low, np.array([0])]),
            high=np.concatenate([prev_space.high, np.array([num_repeats])]),
            shape=(prev_space.shape[0] + 1,),
            dtype=np.float32,
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if reward > 0.0:
            self.num_success += 1

        terminated = (self.num_success + 1) >= self.num_repeats
        obs = self._append_success_cnt(obs)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment and internal state."""
        self.num_success = 0
        obs, info = super().reset(**kwargs)
        obs = self._append_success_cnt       (obs)
        return obs, info

    def _append_success_cnt(self, obs):
        # concatenate the number of success with the state
        obs["observation.state"] = torch.cat((
            obs["observation.state"],
            torch.tensor([self.num_success]).to(obs["observation.state"].device)
        ))
        return obs