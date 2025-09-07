import logging
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from termcolor import colored

import lerobot.experiments
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.control_utils import busy_wait
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.common.utils.utils import log_say
from lerobot.configs import parser
from lerobot.scripts.server.mp_nets import MPNetConfig, reset_mp_net


@dataclass
class RecordConfig:
    env: MPNetConfig


@parser.wrap()
def record_dataset(cfg: RecordConfig):
    mp_net = cfg.env
    robot = make_robot_from_config(mp_net.robot)

    # Go through each primitive and setup their datasets, policies and transition functions
    ctrl_states_hist = []
    current_primitive = mp_net.primitives[mp_net.start_primitive]

    # full reset at the beginning of each sequence
    env = current_primitive.make(mp_net, robot=robot)
    reset_mp_net(env, mp_net)

    # Run episode steps
    while True:
        start_loop_t = time.perf_counter()
        prev_primitive = current_primitive

        # Sample action
        action = env.action_space.sample()

        # read low-level robot states
        ctrl_states = env.unwrapped.robot.controllers["main"].get_all_robot_states()
        ctrl_states_hist.append(ctrl_states)

        # Step environment
        obs, _, terminated, truncated, _ = env.step(torch.zeros_like(action))

        # Check stop triggered by transition function
        done = (terminated or truncated)  # and info.get("success", False)
        current_primitive = mp_net.check_transitions(current_primitive, obs, done)

        # If primitive changed, close old env and make new env
        if prev_primitive != current_primitive:
            if current_primitive.is_terminal:
                break

            env = current_primitive.make(mp_net, robot=robot)
            env.reset()

        # Maintain consistent timing
        if mp_net.fps:
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / mp_net.fps - dt_s)

    robot.disconnect()
    env.close()

    # save results
    measured_forces = []
    ctrl_forces = []
    timestamps = []
    for ctrl_states in ctrl_states_hist:
        measured_forces.append(ctrl_states["ActualTCPForce"][:, 2])
        ctrl_forces.append(ctrl_states["SetTCPForce"][:, 2])
        timestamps.append(ctrl_states["timestamp"])

    measured_forces = np.concatenate(measured_forces)
    ctrl_forces = np.concatenate(ctrl_forces)
    timestamps = np.concatenate(timestamps)

    data = {
        "axis": 2,
        "measured_forces": measured_forces,
        "ctrl_forces": ctrl_forces,
        "timestamps": timestamps,
        "F_max": robot.config.follower_arms["main"].wrench_limits[2],
        "s_min": robot.config.follower_arms["main"].contact_limit_scale_min[2],
        "f_star": robot.config.follower_arms["main"].contact_desired_wrench[2],
    }

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "contact_forces.pkl"), 'wb') as f:
        pickle.dump(data, f)

    # visualize results
    timestamps = timestamps - timestamps[0]
    plt.figure()
    plt.scatter(timestamps, ctrl_forces, label="$F_{ctrl}$", s=4)
    plt.scatter(timestamps, measured_forces, label="$F_{meas}$", s=4)
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join("results", "contact_forces.pdf"))
    plt.show()


if __name__ == "__main__":
    record_dataset()
