import gymnasium as gym
import numpy as np
import torch

from lerobot.scripts.server.kinematics import get_kinematics


class EEActionWrapper(gym.ActionWrapper):
    def __init__(self, env, ee_action_space_params=None, use_gripper=False):
        super().__init__(env)
        self.ee_action_space_params = ee_action_space_params
        self.use_gripper = use_gripper

        # Initialize kinematics instance for the appropriate robot type
        self.kinematics = get_kinematics(env.unwrapped.robot.config, robot_type="follower")
        self.fk_function = self.kinematics.fk_gripper_tip
        self._target_ee_pos = None

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

        if self.use_gripper:
            gripper_command = action[-1]
            action = action[:-1]

        current_joint_pos = self.unwrapped.robot.follower_arms["main"].read("Present_Position")
        if self._target_ee_pos is None:
            self._target_ee_pos = self.fk_function(current_joint_pos)

        # increment the internal target EE position
        self._target_ee_pos[:3, 3] = np.clip(
            self._target_ee_pos[:3, 3] + action,
            self.bounds["min"],
            self.bounds["max"],
        )

        target_joint_pos = self.kinematics.ik(
            current_joint_pos,
            self._target_ee_pos,
            position_only=True,
            fk_func=self.fk_function,
        )

        if self.use_gripper:
            target_joint_pos[-1] = gripper_command

        return target_joint_pos

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._target_ee_pos = None
        return obs, info

    @property
    def target_ee_pos(self):
        return self._target_ee_pos


class EEObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, ee_pose_limits):
        super().__init__(env)

        # Extend observation space to include end effector pose
        prev_space = self.observation_space["observation.state"]

        #self.observation_space["observation.state"] = gym.spaces.Box(
        #    low=np.concatenate([prev_space.low, ee_pose_limits["min"]]),
        #    high=np.concatenate([prev_space.high, ee_pose_limits["max"]]),
        #    shape=(prev_space.shape[0] + 3,),
        #    dtype=np.float32,
        #)
        self.observation_space["observation.state"] = gym.spaces.Box(
            low=np.array(ee_pose_limits["min"]),
            high=np.array(ee_pose_limits["max"]),
            shape=(3,),
            dtype=np.float32,
        )

        # Initialize kinematics instance for the appropriate robot type
        self.kinematics = get_kinematics(env.unwrapped.robot.config, robot_type="follower")
        self.fk_function = self.kinematics.fk_gripper_tip

    def observation(self, observation):
        current_joint_pos = self.unwrapped.robot.follower_arms["main"].read("Present_Position")
        current_ee_pos = self.fk_function(current_joint_pos)
        #observation["observation.state"] = torch.cat(
        #    [
        #        observation["observation.state"],
        #        torch.from_numpy(current_ee_pos[:3, 3]),
        #    ],
        #    dim=-1,
        #)
        observation["observation.state"] = torch.from_numpy(current_ee_pos[:3, 3])
        return observation
