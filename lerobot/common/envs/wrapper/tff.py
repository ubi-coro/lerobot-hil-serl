from typing import Dict, Sequence

import gymnasium as gym
import numpy as np
import torch

from lerobot.common.robot_devices.motors.rtde_tff_controller import TaskFrameCommand


class StaticTaskFrameWrapper(gym.Wrapper):
    """
    Wraps a URGymEnv to:
      1) Send one static TaskFrameCommand per robot at reset.
      2) Expose only selected axes of the 6 (or 7 with gripper) as the action space.

    Args:
      env:           a URGymEnv instance.
      static_tffs:   dict robot_name -> TaskFrameCommand (defines a fixed T_WF/mode/kp/kd).
      action_indices: dict robot_name -> list of ints in [0..n_axes-1] to expose.
    """

    def __init__(
        self,
        env: gym.Env,
        static_tffs: Dict[str, TaskFrameCommand],
        action_indices: Dict[str, Sequence[int]],
    ):
        super().__init__(env)
        self.static_tffs = static_tffs
        self.action_indices = action_indices

        # build new action space: sum of lengths of each robot's action_indices
        dims = [len(idxs) for idxs in action_indices.values()]
        self.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(sum(dims),), dtype=np.float32
        )

        # create an index map so we know where each robot's slice lives
        offset = 0
        self.slice_map: Dict[str, slice] = {}
        for name, idxs in action_indices.items():
            lin, rin = offset, offset + len(idxs)
            self.slice_map[name] = slice(lin, rin)
            offset = rin

    def reset(self, **kwargs):
        # send static TFF to each controller
        for name, tff in self.static_tffs.items():
            ctrl = self.env.unwrapped.robot.controllers[name]
            ctrl.send_cmd(tff)

        # reset underlying env
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        action: low-dim np.ndarray of shape (sum len(action_indices),)
        We expand it to the full action vector, inserting zeros for unexposed dims.
        """
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy().ravel()

        # overwrite static axes with tff targets
        for name in self.action_indices:
            idc = self.action_indices[name]
            slc = self.slice_map[name]
            target = self.static_tffs[name].target
            target[idc] = action[slc]  # overwrite only dims specified by action_indices
            action[slc] = target

        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info