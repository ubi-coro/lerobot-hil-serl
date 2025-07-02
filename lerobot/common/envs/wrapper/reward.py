import time
from typing import Dict, Optional, Sequence

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


class AxisDistanceRewardWrapper(gym.Wrapper):
    """
    Computes an additional reward based on the negative distance between a robot's actual
    module pose and a target value along a chosen axis.

    For each robot *with a specified target* in `targets`:
      r_i(t) = - (pose_i[axis] - target_i)

    The total extra reward is the sum of these per-robot rewards, optionally scaled and clipped.
    This wrapper adds the extra reward to the reward output from the underlying environment.

    Args:
        env: The gym environment to wrap.
        targets: Dict of robot_name -> target value along the selected axis.
                 Only robots present in this dict will be used to compute the extra reward.
        axis: Index of the axis to monitor (0=X, 1=Y, 2=Z, 3=Rx, ...). Default is 2.
        scale: Optional scalar multiplier for the extra reward (default 1.0).
        clip: Optional (min, max) tuple to clip the extra reward.
    """
    def __init__(
        self,
        env: gym.Env,
        targets: Dict[str, float],
        axis: int = 2,
        scale: float = 1.0,
        clip: Optional[tuple[float, float]] = None,
        terminate_on_success: bool = True,
        normalization_range: Optional[Sequence[float]] = None
    ):
        super().__init__(env)
        self.targets = targets
        self.axis = axis
        self.scale = scale
        self.clip = clip
        self.terminate_on_success = terminate_on_success
        self.normalization_range = normalization_range

        # Only consider controllers/robots that exist in the wrapped env.
        self.robot_names = list(env.unwrapped.robot.controllers.keys())

    def compute_extra_reward(self) -> tuple[float, bool]:
        """
        Computes extra reward only for robots that have an entry in self.targets.
        Per robot, extra reward is computed as:
            r = - (actual_pose[axis] - target)
        Returns the summed, scaled, and optionally clipped extra reward, along with a per-robot reward dict.
        """
        rewards = {}
        successes = {}
        for name in self.robot_names:
            if name not in self.targets:
                # Skip this robot if no target is set.
                continue
            ctrl = self.env.unwrapped.robot.controllers[name]
            pose = ctrl.get_robot_state()["ActualTCPPose"]
            actual = pose[self.axis]
            target = self.targets[name]

            reward = -(target - actual)

            if self.normalization_range is not None:
                reward = reward / (self.normalization_range[1] - self.normalization_range[0]) + self.normalization_range[0]

            rewards[name] = reward
            successes[name] = actual > target

        extra_reward = sum(rewards.values()) * self.scale
        success = all(list(successes.values()))

        if self.clip is not None:
            extra_reward = np.clip(extra_reward, *self.clip)

        return float(extra_reward), success

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        extra_reward, success = self.compute_extra_reward()
        total_reward = reward + extra_reward

        info["success"] = success
        if self.terminate_on_success:
            terminated = terminated | success

        return obs, total_reward, terminated, truncated, info
