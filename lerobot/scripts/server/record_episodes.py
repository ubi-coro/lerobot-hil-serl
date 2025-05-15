import logging
import time
from dataclasses import dataclass

import numpy as np

import lerobot.experiments
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.envs import EnvConfig
from lerobot.common.envs.configs import HILSerlRobotEnvConfig
from lerobot.common.robot_devices.control_utils import busy_wait
from lerobot.common.utils.utils import log_say
from lerobot.configs import parser


@dataclass
class RecordControlConfig:
    mode: str = "record"
    env: EnvConfig = HILSerlRobotEnvConfig()


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
                action = env.action_space.sample() * 0.0

                if hasattr(cfg, "wrapper") and cfg.wrapper.use_gripper:
                    action[-1] = 1.0  # neutral gripper action

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Check if episode needs to be rerecorded
            if info.get("rerecord_episode", False):
                break

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


@parser.wrap()
def main(cfg: RecordControlConfig):
    env = cfg.env.make()

    if cfg.mode == "record":
        policy = None
        if cfg.env.pretrained_policy_name_or_path is not None:
            from lerobot.common.policies.sac.modeling_sac import SACPolicy

            policy = SACPolicy.from_pretrained(cfg.env.pretrained_policy_name_or_path)
            policy.to(cfg.env.device)
            policy.eval()

        record_dataset(
            env,
            policy=policy,
            cfg=cfg.env,
        )
        exit()

    if cfg.mode == "replay":
        replay_episode(
            env,
            cfg=cfg.env,
        )
        exit()

    env.reset()


if __name__ == "__main__":
    main()
