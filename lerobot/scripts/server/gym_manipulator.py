import logging
import time

import numpy as np

from lerobot.common.envs.configs import PushCubeRobotEnvConfig
from lerobot.common.robot_devices.control_utils import busy_wait
from lerobot.common.utils.utils import log_say


###########################################################
# Record and replay functions
###########################################################


def record_dataset(env, policy, cfg):
    """
    Record a dataset of robot interactions using either a policy or teleop.

    cfg.
        env: The environment to record from
        repo_id: Repository ID for dataset storage
        root: Local root directory for dataset (optional)
        num_episodes: Number of episodes to record
        control_time_s: Maximum episode length in seconds
        fps: Frames per second for recording
        push_to_hub: Whether to push dataset to Hugging Face Hub
        task_description: Description of the task being recorded
        policy: Optional policy to generate actions (if None, uses teleop)
    """
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    # Setup initial action (zero action if using teleop)

    # Configure dataset features based on environment spaces
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": env.observation_space["observation.state"].shape,
            "names": None,
        },
        "action": {
            "dtype": "float32",
            "shape": env.action_space.shape,
            "names": None,
        },
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
    }

    # Add image features
    for key in env.observation_space:
        if "image" in key:
            features[key] = {
                "dtype": "video",
                "shape": env.observation_space[key].shape,
                "names": None,
            }

    # Create dataset
    if cfg.resume:
        dataset = LeRobotDataset(cfg.repo_id, root=cfg.dataset_root)
        dataset.start_image_writer(
            num_processes=2,
            num_threads=4 * len(cfg.robot.cameras),
        )
    else:
        dataset = LeRobotDataset.create(
            cfg.repo_id,
            cfg.fps,
            root=cfg.dataset_root,
            use_videos=True,
            image_writer_threads=4 * len(cfg.robot.cameras),
            image_writer_processes=2,
            features=features,
        )

    # Record episodes
    while dataset.num_episodes < cfg.num_episodes:
        obs, _ = env.reset()
        busy_wait(0.1)

        start_episode_t = time.perf_counter()
        log_say(f"Recording episode {dataset.num_episodes}", play_sounds=True)

        # Run episode steps
        while time.perf_counter() - start_episode_t < cfg.wrapper.control_time_s:
            start_loop_t = time.perf_counter()

            # Get action from policy if available
            if cfg.pretrained_policy_name_or_path is not None:
                action = policy.select_action(obs)
            else:
                action = env.action_space.sample() #* 0.0

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Check if episode needs to be rerecorded
            if info.get("rerecord_episode", False):
                break

            #if not info["is_intervention"]:
            #    continue

            if info["is_intervention"]:
                # For teleop, get action from intervention
                recorded_action = {
                    "action": info["action_intervention"] if policy is None else action
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
            dataset.add_frame(frame)

            # Maintain consistent timing
            if cfg.fps:
                dt_s = time.perf_counter() - start_loop_t
                busy_wait(1 / cfg.fps - dt_s)

            if terminated or truncated:
                break

        # Handle episode recording
        if info.get("rerecord_episode", False):
            dataset.clear_episode_buffer()
            logging.info(f"Re-recording episode {dataset.num_episodes}")
            continue

        dataset.save_episode()

    # Finalize dataset
    # dataset.consolidate(run_compute_stats=True)
    if cfg.push_to_hub:
        dataset.push_to_hub()


def replay_episode(env, cfg):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.dataset_root, episodes=[cfg.episode])
    env.reset()

    actions = dataset.hf_dataset.select_columns("action")

    for idx in range(dataset.num_frames):
        start_episode_t = time.perf_counter()

        action = actions[idx]["action"]
        env.step(action)

        dt_s = time.perf_counter() - start_episode_t
        busy_wait(1 / 10 - dt_s)


def main():
    from lerobot.experiments import RealGraspCubeEnvConfig

    cfg = RealGraspCubeEnvConfig(
        mode="record",
        resume=False,
    )
    env = cfg.make()

    if cfg.mode == "record":
        policy = None
        if cfg.pretrained_policy_name_or_path is not None:
            from lerobot.common.policies.sac.modeling_sac import SACPolicy

            policy = SACPolicy.from_pretrained(cfg.pretrained_policy_name_or_path)
            policy.to(cfg.device)
            policy.eval()

        record_dataset(
            env,
            policy=policy,
            cfg=cfg,
        )
        exit()

    if cfg.mode == "replay":
        replay_episode(
            env,
            cfg=cfg,
        )
        exit()

    env.reset()

    # Initialize the smoothed action as a random sample.
    smoothed_action = env.action_space.sample()

    # Smoothing coefficient (alpha) defines how much of the new random sample to mix in.
    # A value close to 0 makes the trajectory very smooth (slow to change), while a value close to 1 is less smooth.
    alpha = 1.0

    num_episode = 0
    successes = []
    while num_episode < 20:
        start_loop_s = time.perf_counter()
        # Sample a new random action from the robot's action space.
        new_random_action = env.action_space.sample()
        # Update the smoothed action using an exponential moving average.
        smoothed_action = alpha * new_random_action + (1 - alpha) * smoothed_action

        # Execute the step: wrap the NumPy action in a torch tensor.
        obs, reward, terminated, truncated, info = env.step(smoothed_action)
        if terminated or truncated:
            successes.append(reward)
            env.reset()
            num_episode += 1

        dt_s = time.perf_counter() - start_loop_s
        busy_wait(1 / cfg.fps - dt_s)

    logging.info(f"Success after 20 steps {successes}")
    logging.info(f"success rate {sum(successes) / len(successes)}")


if __name__ == "__main__":
    main()
