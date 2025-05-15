import time

import gymnasium as gym
import numpy as np

MAX_GRIPPER_COMMAND = 80


class GripperPenaltyWrapper(gym.RewardWrapper):
    def __init__(self, env, penalty: float = -0.05):
        super().__init__(env)
        self.penalty = penalty
        self.last_gripper_state = None

    def reward(self, reward, action, info):
        gripper_state_normalized = self.last_gripper_state / MAX_GRIPPER_COMMAND
        action_normalized = action - 1.0

        gripper_penalty_bool = (
            (gripper_state_normalized < 0.5 and action_normalized > 0.5) or
            (gripper_state_normalized > 0.75 and action_normalized < -0.5)
        )

        gripper_penalty = self.penalty * int(gripper_penalty_bool)
        info["gripper_penalty"] = gripper_penalty
        return reward + gripper_penalty, info

    def step(self, action):
        self.last_gripper_state = self.unwrapped.robot.follower_arms["main"].read("Present_Position")[-1]
        gripper_action = action[-1]
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward, info = self.reward(reward, gripper_action, info)
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
        self.gripper_action = None

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

        # Update reference gripper position
        if self.gripper_action is None:
            self.gripper_action = self.unwrapped.robot.follower_arms["main"].read("Present_Position")[-1]
        self.gripper_action = np.clip( self.gripper_action + gripper_command, 0, MAX_GRIPPER_COMMAND)

        action[-1] = self.gripper_action
        return action

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.last_gripper_action_time = 0.0
        self.last_gripper_action = None
        self.gripper_action = None
        return obs, info
