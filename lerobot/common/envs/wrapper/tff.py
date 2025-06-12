import time
from copy import copy
from typing import Dict, Sequence, Optional, Literal

import gymnasium as gym
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from lerobot.common.robot_devices.motors.rtde_tff_controller import TaskFrameCommand, AxisMode
from lerobot.common.robot_devices.utils import busy_wait


class StaticTaskFrameActionWrapper(gym.Wrapper):
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
        self.robot_names = list(env.unwrapped.robot.controllers.keys())

        default_tff = TaskFrameCommand(
            target=np.full(6, 0.0),
            mode=[AxisMode.IMPEDANCE_VEL] * 6,
            kp=np.array([2500, 2500, 2500, 100, 100, 100]),
            kd=np.full(6, 0.2),
            T_WF=np.eye(4)
        )

        # create an index map so we know where each robot's slice lives
        offset = 0
        self.static_tffs = {}
        self.gripper_enabled = {}
        self.action_indices = {}
        self.slice_map: Dict[str, slice] = {}
        for name in self.robot_names:
            self.static_tffs[name] = static_tffs.get(name, copy(default_tff))

            self.gripper_enabled[name] = env.unwrapped.robot.controllers[name].config.use_gripper
            num_actions = 7 if self.gripper_enabled[name] else 6
            self.action_indices[name] = np.array(action_indices.get(name, [1] * num_actions), dtype=bool)
            assert len(self.action_indices[name]) == num_actions

            lin, rin = offset, offset + sum(self.action_indices[name])
            self.slice_map[name] = slice(lin, rin)
            offset = rin

        # build new action space: sum of lengths of each robot's action_indices
        dims = [sum(idc) for idc in action_indices.values()]
        self.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(sum(dims),), dtype=np.float32
        )

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
        # overwrite static axes with tff targets
        full_action = []
        for name in self.robot_names:
            idc = self.action_indices[name]
            slc = self.slice_map[name]
            target = torch.Tensor(self.static_tffs[name].target)

            if self.gripper_enabled[name]:
                gripper_value = action[slc][-1] if idc[-1] else 0.0
                target = torch.append(target, gripper_value)

            target[idc] = action[slc]
            full_action.append(target)

        obs, reward, terminated, truncated, info = self.env.step(torch.cat(full_action))
        return obs, reward, terminated, truncated, info


class StaticTaskFrameResetWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        static_tffs: Dict[str, TaskFrameCommand],
        reset_pos: Dict[str, Sequence[float]],
        reset_kp: Optional[Dict[str, Sequence[float]]] = None,
        reset_kd: Optional[Dict[str, Sequence[float]]] = None,
        noise_std: Optional[Dict[str, Sequence[float]]] = None,
        noise_dist: Literal["normal", "uniform"] = "uniform",
        safe_reset: bool = True,
        threshold: float = 0.005,  # 5 mm / ~0.3 deg
        timeout: float = 5.0,  # seconds per wait
    ):
        super().__init__(env)
        self.robot_names = list(env.unwrapped.robot.controllers.keys())
        self.safe_reset = safe_reset
        self.threshold = threshold
        self.timeout = timeout
        self.noise_std = noise_std or {}
        self.noise_dist = noise_dist

        default_tff = TaskFrameCommand(
            target=np.full(6, 0.0),
            mode=[AxisMode.IMPEDANCE_VEL] * 6,
            kp=np.array([2500, 2500, 2500, 100, 100, 100]),
            kd=np.full(6, 0.2),
            T_WF=np.eye(4)
        )

        # create task frame commands for resetting the environment
        reset_kp = reset_kp or {}
        reset_kd = reset_kd or {}
        self.static_tffs = {}
        self.reset_tffs = {}
        for name in self.robot_names:
            self.static_tffs[name] = static_tffs.get(name, copy(default_tff))
            self.reset_tffs[name] = TaskFrameCommand(
                mode=[AxisMode.POS] * 6,
                target=np.array(reset_pos.get(name)),  # no default, should error on missing robot
                kp=reset_kp.get(name, self.static_tffs[name].kp),
                kd=reset_kd.get(name, self.static_tffs[name].kd),
                T_WF = self.static_tffs[name].T_WF
            )

            # correctly shape noise std
            if name in self.noise_std:
                noise_std = np.atleast_1d(self.noise_std[name]).astype(np.float64)
                target_len = self.reset_tffs[name].target.shape[0]

                if noise_std.shape == ():  # scalar
                    noise_std = np.full(target_len, noise_std.item())
                elif noise_std.shape[0] != target_len:
                    raise ValueError(
                        f"Noise shape mismatch for {name}: expected {target_len} values, got {noise_std.shape[0]}")
                self.noise_std[name] = noise_std

    def wait_until_reached(self, name: str, target: np.ndarray):
        """Waits until the robot's TCP pose is close to target or timeout."""
        start_time = time.time()
        ctrl = self.env.unwrapped.robot.controllers[name]
        while True:
            current = np.array(ctrl.get_robot_state()["ActualTCPPose"])

            # split into translation + rotation‐vector
            pos_curr, rot_curr = current[:3], current[3:6]
            pos_tgt, rot_tgt = target[:3], target[3:6]

            pos_err = np.linalg.norm(pos_curr - pos_tgt)
            r_curr = R.from_rotvec(rot_curr)
            r_tgt = R.from_rotvec(rot_tgt)
            r_err = r_curr.inv() * r_tgt
            rot_err = np.linalg.norm(r_err.as_rotvec())

            error = np.sqrt(pos_err ** 2 + rot_err ** 2)
            if error < self.threshold:
                break

            if time.time() - start_time > self.timeout:
                print(f"[WARN] {name} did not reach target within {self.timeout}s "
                      f"(final error={error:.4f}, pos={pos_err:.4f}, rot={rot_err:.4f})")
                break
            time.sleep(0.01)

    def reset(self, **kwargs):
        for name in self.robot_names:
            base_cmd = copy(self.reset_tffs[name])
            ctrl = self.env.unwrapped.robot.controllers[name]

            if self.safe_reset:
                ctrl.send_cmd(base_cmd)
                self.wait_until_reached(name, base_cmd.target)

            # Apply noise if provided
            if name in self.noise_std:
                noisy_cmd = copy(base_cmd)
                if self.noise_dist == "uniform":
                    noisy_cmd.target += np.random.uniform(-self.noise_std[name], self.noise_std[name])
                elif self.noise_dist == "normal":
                    noisy_cmd.target += np.random.normal(0.0, self.noise_std[name])
                ctrl.send_cmd(noisy_cmd)
                self.wait_until_reached(name, noisy_cmd.target)

            elif not self.safe_reset:
                ctrl.send_cmd(base_cmd)
                self.wait_until_reached(name, base_cmd.target)

        return self.env.reset(**kwargs)

