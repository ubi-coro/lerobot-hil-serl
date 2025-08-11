import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
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


def init_datasets(cfg: MPNetConfig) -> Tuple[Dict[str, LeRobotDataset], int]:
    datasets = {}
    min_episode = float('inf')
    for name, primitive in cfg.primitives.items():
        if not primitive.is_adaptive:
            continue

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
            datasets[primitive.id] = LeRobotDataset(repo_id, root=dataset_root)
            datasets[primitive.id].start_image_writer(
                num_processes=2,
                num_threads=4 * len(cfg.robot.cameras),
            )
        else:
            datasets[primitive.id] = LeRobotDataset.create(
                repo_id,
                cfg.fps,
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
def record_dataset(cfg: RecordConfig):
    mp_net = cfg.env
    policies = mp_net.make_policies()
    robot = make_robot_from_config(mp_net.robot)
    step_counter = mp_net.get_step_counter()

    # Go through each primitive and setup their datasets, policies and transition functions
    datasets, episode = init_datasets(mp_net)
    episode = 0

    # Record episodes
    while episode < mp_net.num_episodes:
        log_say(f"Recording episode {episode}", play_sounds=True)
        current_primitive = mp_net.primitives[mp_net.start_primitive]
        policy = policies.get(current_primitive.id, None)
        sum_reward = 0.0

        # full reset at the beginning of each sequence
        env = current_primitive.make(mp_net, robot=robot)
        obs, info = reset_mp_net(env, mp_net)

        # Run episode steps
        while True:
            start_loop_t = time.perf_counter()
            prev_primitive = current_primitive

            # Sample action
            if policy is not None:
                action = policy.select_action(obs, add_structured_noise=False)
            else:
                action = env.action_space.sample()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            sum_reward += float(reward)

            # Increment that primitive's step counter
            step_counter.increment(current_primitive.id)

            # If nothing is saved, the rest of the loop can be skipped
            if current_primitive.is_adaptive:

                # Process info dict
                if info.get("is_intervention", False):
                    # For teleop, get action from intervention
                    recorded_action = {"action": info["action_intervention"]}
                else:
                    recorded_action = {"action": action}

                # Process observation for dataset
                obs = {k: v.cpu().squeeze(0).float() for k, v in obs.items()}

                # Add frame to dataset
                frame = {**obs, **recorded_action}
                frame["next.reward"] = np.array([reward], dtype=np.float32)
                frame["next.done"] = np.array([terminated or truncated], dtype=bool)
                frame["task"] = mp_net.task
                datasets[current_primitive.id].add_frame(frame)

                # Check if episode needs to be rerecorded
                if info.get("rerecord_episode", False):
                    break

            # Check stop triggered by transition function
            done = (terminated or truncated)  # and info.get("success", False)
            current_primitive = mp_net.check_transitions(current_primitive, obs, done)

            # If primitive changed, close old env and make new env
            if prev_primitive != current_primitive:
                if prev_primitive.is_adaptive:
                    datasets[prev_primitive.id].save_episode()
                    logging.info(
                        f"Finished {episode} episode for {prev_primitive.id} primitive (Demo), "
                        f"episode reward: {sum_reward}, "
                        f"successful? {['no', 'yes'][int(info.get('success', False))]}, "
                        f"local step: {step_counter[prev_primitive.id]}, "
                        f"global step: {step_counter.global_step}"
                    )
                else:
                    logging.info(
                        f"Finished {episode} episode for {prev_primitive.id} primitive (Demo), "
                    )

                logging.info(f"Now transition to  {current_primitive.id} primitive")
                env.close()

                if current_primitive.is_terminal:
                    episode += 1
                    break

                sum_reward = 0.0

                env = current_primitive.make(mp_net, robot=robot)

            # Maintain consistent timing
            if mp_net.fps:
                dt_s = time.perf_counter() - start_loop_t
                busy_wait(1 / mp_net.fps - dt_s)

        # Handle episode recording
        if info.get("rerecord_episode", False):
            for ds in datasets.values():
                ds.clear_episode_buffer()
            logging.info(f"Re-recording episode {episode}")

    robot.disconnect()
    env.close()

    # Finalize dataset
    if mp_net.push_to_hub:
        for ds in datasets.values():
            ds.push_to_hub()


if __name__ == "__main__":
    record_dataset()
