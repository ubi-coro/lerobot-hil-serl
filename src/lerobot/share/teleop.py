





"""Simple real-time teleoperation using LeRobot environment configurations."""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import torch

from lerobot.configs import parser
from lerobot.envs import EnvConfig
from lerobot.processor import TransitionKey, create_transition
from lerobot.processor.hil_processor import TELEOP_ACTION_KEY
from lerobot.teleoperators import TeleopEvents
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from lerobot.utils.utils import init_logging

import experiments


@dataclass
class TeleopConfig:
    env: EnvConfig
    display_data: bool = False
    control_time_s: float | None = None


@parser.wrap()
def teleop(cfg: TeleopConfig):
    env_cfg: EnvConfig = cfg.env
    
    # Disable time limit for teleoperation (no need for episode progress bar)
    if hasattr(env_cfg, "processor") and hasattr(env_cfg.processor, "control_time_s"):
        env_cfg.processor.control_time_s = None
    
    # But allow override if user explicitly wants a time limit
    if cfg.control_time_s is not None:
        if hasattr(env_cfg, "processor") and hasattr(env_cfg.processor, "control_time_s"):
            env_cfg.processor.control_time_s = cfg.control_time_s

    init_logging()
    logging.info("Starting teleop with config:")
    logging.info(pformat(asdict(env_cfg)))

    env, env_processor, action_processor = env_cfg.make(device="cpu")

    if cfg.display_data:
        init_rerun(session_name="teleoperation")

    obs, info = env.reset()
    env_processor.reset()
    action_processor.reset()

    transition = env_processor(create_transition(observation=obs, info=info))
    action_processor.reset()

    fps = getattr(env_cfg, "fps", 30)
    action_dim = getattr(env_cfg, "action_dim", None)
    if action_dim is None:
        action_dim = int(env.action_space.shape[0])

    logging.info("Running teleoperation loop...")

    start_time = time.perf_counter()
    try:
        while True:
            loop_start = time.perf_counter()

            prev_transition = transition
            info = {TeleopEvents.IS_INTERVENTION: True}

            action = torch.zeros(action_dim, dtype=torch.float32)
            action_transition = create_transition(action=action, info=info)
            processed_action_transition = action_processor(action_transition)

            obs, reward, terminated, truncated, step_info = env.step(
                processed_action_transition[TransitionKey.ACTION]
            )

            complementary_data_raw = processed_action_transition[TransitionKey.COMPLEMENTARY_DATA]
            complementary_data = dict(complementary_data_raw) if complementary_data_raw is not None else {}
            action_info = processed_action_transition[TransitionKey.INFO] or {}
            step_info.update(action_info)

            if step_info.get(TeleopEvents.IS_INTERVENTION, False) and TELEOP_ACTION_KEY in complementary_data:
                action_to_record = complementary_data[TELEOP_ACTION_KEY]
            else:
                action_to_record = processed_action_transition[TransitionKey.ACTION]

            transition = create_transition(
                observation=obs,
                action=action_to_record,
                reward=reward + processed_action_transition[TransitionKey.REWARD],
                done=terminated or processed_action_transition[TransitionKey.DONE],
                truncated=truncated or processed_action_transition[TransitionKey.TRUNCATED],
                info=step_info,
                complementary_data=complementary_data,
            )
            transition = env_processor(transition)
            action_processor.reset()

            if cfg.display_data:
                dataset_observation = {
                    k: v.squeeze().cpu()
                    for k, v in prev_transition[TransitionKey.OBSERVATION].items()
                    if isinstance(v, torch.Tensor)
                }
                rerun_obs = {k: v.numpy() for k, v in dataset_observation.items()}
                action_np = transition[TransitionKey.ACTION].detach().cpu().numpy()
                log_rerun_data(observation=rerun_obs, action=action_np)

            if terminated or truncated:
                logging.info("Environment signalled termination. Resetting.")
                obs, info = env.reset()
                env_processor.reset()
                action_processor.reset()
                transition = env_processor(create_transition(observation=obs, info=info))
                continue

            if step_info.get(TeleopEvents.TERMINATE_EPISODE, False) or step_info.get(
                TeleopEvents.RERECORD_EPISODE, False
            ):
                action_processor.reset()

            if cfg.control_time_s is not None and (time.perf_counter() - start_time) >= cfg.control_time_s:
                logging.info("Control time reached. Stopping teleoperation loop.")
                break

            elapsed = time.perf_counter() - loop_start
            busy_wait(max(0.0, 1 / fps - elapsed))

    except KeyboardInterrupt:
        logging.info("Teleoperation stopped by user.")
    finally:
        env.close()


def main():
    teleop()


if __name__ == "__main__":
    main()
