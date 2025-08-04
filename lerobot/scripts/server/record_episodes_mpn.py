import logging
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

import lerobot.experiments
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.envs.wrapper.tff import StaticTaskFrameResetWrapper
from lerobot.common.robot_devices.control_utils import busy_wait
from lerobot.common.utils.utils import log_say
from lerobot.configs import parser
from lerobot.scripts.server.mp_nets import MPNetConfig, init_datasets


def reset_mp_net(env, cfg: MPNetConfig):
    reset_env = StaticTaskFrameResetWrapper(
        env,
        static_tffs=cfg.reset.static_tffs or {},
        reset_pos=cfg.reset.reset_pos,
        reset_kp=cfg.reset.reset_kp,
        reset_kd=cfg.reset.reset_kd,
        noise_std=cfg.reset.noise_std,
        noise_dist=cfg.reset.noise_dist,
        safe_reset=cfg.reset.safe_reset,
        threshold=cfg.reset.threshold,
        timeout=cfg.reset.timeout
    )

    obs, info = reset_env.reset()
    return obs, info


@parser.wrap()
def record_dataset(cfg: MPNetConfig):
    # Go through each primitive and setup their datasets, policies and transition functions
    datasets, episode = init_datasets(cfg)

    # Record episodes
    while episode < cfg.num_episodes:
        log_say(f"Recording episode {episode}", play_sounds=True)
        current_primitive = cfg.primitives[cfg.start_primitive]

        # full reset at the beginning of each sequence
        env = current_primitive.make()
        obs, info = reset_mp_net(env, cfg)

        # Run episode steps
        while True:
            start_loop_t = time.perf_counter()
            prev_primitive = current_primitive

            # Check end of sequence
            if current_primitive.is_terminal:
                episode += 1
                break

            # Sample action
            if current_primitive.has_policy:
                action = current_primitive.policy.sample_action(obs)
            else:
                action = env.action_space.sample()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Check stop triggered by transition function
            for primitive_name, stop_condition in current_primitive.transitions.items():
                if stop_condition(obs):
                    current_primitive = cfg.primitives[primitive_name]

            # Check stop triggered by spacemouse
            if terminated or truncated:
                assert len(current_primitive.transitions) == 1, "Spacemouse transition is ambiguous, only one transition per primitive at the moment."
                current_primitive = cfg.primitives[list(current_primitive.transitions)[0]]

            # If primitive changed, close old env and make new env
            if prev_primitive != current_primitive:
                env.close()
                env = current_primitive.make()

                if prev_primitive.is_adaptive:
                    datasets[prev_primitive].save_episode()

            # Store frame
            if not current_primitive.is_adaptive:
                continue

            # Process info
            if info["is_intervention"]:
                # For teleop, get action from intervention
                recorded_action = {
                    "action": info["action_intervention"] if current_primitive.has_policy.has_policy else action
                }
            else:
                recorded_action = {
                    "action": action
                }

            # Process observation for dataset
            obs = {k: v.cpu().squeeze(0).float() for k, v in obs.items()}

            # Add frame to dataset
            frame = {**obs, **recorded_action}
            frame["next.reward"] = np.array([reward], dtype=np.float32)
            frame["next.done"] = np.array([terminated or truncated], dtype=bool)
            frame["task"] = cfg.task
            datasets[current_primitive].add_frame(frame)

            # Check if episode needs to be rerecorded
            if info.get("rerecord_episode", False):
                break

            # Maintain consistent timing
            if cfg.fps:
                dt_s = time.perf_counter() - start_loop_t
                busy_wait(1 / cfg.fps - dt_s)

        # Handle episode recording
        if info.get("rerecord_episode", False):
            for ds in datasets.values():
                ds.clear_episode_buffer()
            logging.info(f"Re-recording episode {episode}")

    env.close()

    # Finalize dataset
    if cfg.push_to_hub:
        for ds in datasets.values():
            ds.push_to_hub()


if __name__ == "__main__":
    record_dataset()
