import logging
import os
import pickle
import time
from dataclasses import dataclass, field
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
    root: str = "results/contact_forces_gain_1.0.pkl"


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
            dt_load = time.perf_counter() - start_loop_t
            busy_wait(1 / mp_net.fps - dt_load)
            dt_loop = time.perf_counter() - start_loop_t
            #logging.info(
            #    f"dt_loop: {dt_loop * 1000:5.2f}ms ({1 / dt_loop:3.1f}hz), "
            #    f"dt_load: {dt_load * 1000:5.2f}ms ({1 / dt_load:3.1f}hz)"
            #)

    robot.disconnect()
    env.close()

    # save results
    measured_forces = []
    filtered_forces = []
    ctrl_forces = []
    timestamps = []
    for ctrl_states in ctrl_states_hist:
        measured_forces.append(ctrl_states["ActualTCPForce"][:, 2])
        filtered_forces.append(ctrl_states["ActualTCPForceFiltered"][:, 2])
        ctrl_forces.append(ctrl_states["SetTCPForce"][:, 2])
        timestamps.append(ctrl_states["timestamp"])

    measured_forces = np.concatenate(measured_forces)
    ctrl_forces = np.concatenate(ctrl_forces)
    timestamps = np.concatenate(timestamps)

    data = {
        "fps": mp_net.fps,
        "axis": 2,
        "measured_forces": measured_forces,
        "filtered_forces": filtered_forces,
        "ctrl_forces": ctrl_forces,
        "timestamps": timestamps,
        "F_max": robot.config.follower_arms["main"].wrench_limits[2],
        "s_min": robot.config.follower_arms["main"].compliance_adaptive_limit_min[2],
        "f_star": robot.config.follower_arms["main"].compliance_desired_wrench[2],
    }

    os.makedirs("results", exist_ok=True)
    with open(cfg.root, 'wb') as f:
        pickle.dump(data, f)

    # visualize results
    timestamps = timestamps - timestamps[0]
    plt.figure()
    plt.plot(timestamps, -ctrl_forces, label="$-F_{ctrl}$", color="orange")
    plt.plot(timestamps, measured_forces, label="$F_{meas}$", color="blue")
    plt.axhline(y=5.0, color="k", linestyle="--", label="$F^*$")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(cfg.root.replace("pkl", "pdf")))
    plt.show()


if __name__ == "__main__":
    record_dataset()
