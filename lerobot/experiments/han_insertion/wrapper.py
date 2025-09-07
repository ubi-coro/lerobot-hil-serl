import copy
import time

import numpy as np

from lerobot.common.envs.wrapper.tff import StaticTaskFrameActionWrapper, StaticTaskFrameResetWrapper
from lerobot.common.robot_devices.motors.rtde_tff_controller import TaskFrameCommand, AxisMode, RTDETFFController


class TrapezoidResetWrapper(StaticTaskFrameResetWrapper):
    def __init__(self, c_offset_min_std_rad: float, c_offset_max_std_rad: float, x_offset_mean_mm, **kwargs):
        super().__init__(**kwargs)
        self.c_offset_min_std_rad = c_offset_min_std_rad
        self.c_offset_max_std_rad = c_offset_max_std_rad
        self.x_offset_mean_mm = x_offset_mean_mm / 1000
        assert self.safe_reset

    def reset(self, **kwargs):
        base_cmd = copy.copy(self.reset_tffs["main"])
        ctrl: RTDETFFController = self.env.unwrapped.robot.controllers["main"]

        ctrl.send_cmd(base_cmd)
        self.wait_until_reached("main", base_cmd.target)

        # first sample noise as usual
        noisy_cmd = copy.deepcopy(base_cmd)
        if self.noise_dist == "uniform":
            factor = np.sqrt(12) / 2
            noisy_cmd.target += np.random.uniform(-factor * self.noise_std["main"], factor * self.noise_std["main"])
        else:
            noisy_cmd.target += np.random.normal(0.0, self.noise_std["main"])

        # shift x to the right
        noisy_cmd.target[0] += self.x_offset_mean_mm

        # interpolate c-axis limits and resample uniformly
        min_x = base_cmd.min_pose_rpy[0]
        max_x = base_cmd.max_pose_rpy[0]
        x = float(np.clip(noisy_cmd.target[0], min_x, max_x))

        min_c = self.c_offset_min_std_rad
        max_c = self.c_offset_max_std_rad

        c_std_norm = (x - min_x) / (max_x - min_x)
        c_std = (max_c - min_c) * c_std_norm + min_c
        c_lim = c_std * np.sqrt(12) / 2
        noisy_cmd.target[5] = np.random.uniform(-c_lim, c_lim)

        # bound noisy target
        noisy_cmd.target = np.clip(
            noisy_cmd.target,
            base_cmd.min_pose_rpy,
            base_cmd.max_pose_rpy
        )

        ctrl.send_cmd(noisy_cmd)
        self.wait_until_reached("main", noisy_cmd.target)

        time.sleep(0.03)
        ctrl.zero_ft()
        time.sleep(0.03)

        return self.env.reset(**kwargs)
