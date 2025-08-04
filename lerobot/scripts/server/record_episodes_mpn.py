import logging
import time
from pathlib import Path
from typing import Tuple, Dict

import numpy as np

import lerobot.experiments
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.control_utils import busy_wait
from lerobot.common.utils.utils import log_say
from lerobot.configs import parser
from lerobot.scripts.server.mp_nets import MPNetConfig, reset_mp_net


def init_datasets(cfg: MPNetConfig) -> Tuple[Dict[str, LeRobotDataset], int]:
    datasets = {}
    min_episode = float('inf')
    for name, primitive in cfg.primitives.values():
        # Configure dataset features based on environment spaces
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": primitive.features["observation.state"].shape,
                "names": None,
            },
            "action": {
                "dtype": "float32",
                "shape": primitive.features["action"].shape,
                "names": None,
            },
            "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
            "next.done": {"dtype": "bool", "shape": (1,), "names": None},
        }

        # Add image features
        for key in primitive.features:
            if "image" in key:
                features[key] = {
                    "dtype": "video",
                    "shape": primitive.features[key].shape,
                    "names": None,
                }

        # Create dataset
        dataset_root = Path(cfg.dataset_root) / name
        repo_id = cfg.repo_id + f"-{name}"
        if cfg.resume:
            datasets[primitive.id] = LeRobotDataset(cfg.repo_id, root=dataset_root)
            datasets[primitive.id].start_image_writer(
                num_processes=2,
                num_threads=4 * len(cfg.robot.cameras),
            )
        else:
            datasets[primitive.id] = LeRobotDataset.create(
                cfg.fps,
                repo_id,
                root=dataset_root,
                use_videos=True,
                image_writer_threads=4 * len(cfg.robot.cameras),
                image_writer_processes=2,
                features=features,
            )

        # Update min_episode
        if datasets[primitive.id].num_episodes < min_episode:
            min_episode = datasets[primitive.id].num_episodes

    return datasets, min_episode


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
