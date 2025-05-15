import time

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


class RewardClassifierWrapper(gym.Wrapper):
    def __init__(self, env, reward_classifier, device: torch.device = "cuda"):
        """
        Wrapper to add reward prediction to the environment, it use a trained classifier.

        cfg.
            env: The environment to wrap
            reward_classifier: The reward classifier model
            device: The device to run the model on
        """
        super().__init__(env)

        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.reward_classifier = torch.compile(reward_classifier)
        self.reward_classifier.to(self.device)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        images = {
            key: observation[key].to(self.device, non_blocking=self.device.type == "cuda").unsqueeze(0)
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
