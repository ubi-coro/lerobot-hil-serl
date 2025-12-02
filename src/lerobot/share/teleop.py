





"""
Simple real-time teleoperation using LeRobot environments.

Run example:
python teleop_run.py \
    --env.robot.type=viperx_follower \
    --env.robot.port=/dev/ttyUSB0 \
    --env.teleop.type=viperx_leader \
    --env.teleop.port=/dev/ttyUSB1 \
    --display_data=true
"""

import time
import logging
from pprint import pformat
from dataclasses import asdict, dataclass

import torch
from lerobot.configs import parser
from lerobot.envs import EnvConfig
from lerobot.envs.robot_env import RobotEnv
from lerobot.processor import create_transition
from lerobot.rl.gym_manipulator import step_env_and_process_transition
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from lerobot.utils.utils import init_logging

import experiments


@dataclass
class TeleopConfig:
    env: EnvConfig


@parser.wrap()
def teleop(cfg: TeleopConfig):
    cfg: EnvConfig = cfg.env
    init_logging()
    logging.info("Starting teleop with config:")
    logging.info(pformat(asdict(cfg)))

    # Initialize environment (robot + teleop pipelines)
    env, env_processor, action_processor = cfg.make()

    # Optional visualization
    #init_rerun(session_name="teleoperation")

    obs, info = env.reset()
    env_processor.reset()
    action_processor.reset()

    transition = create_transition(observation=obs, info=info)
    transition = env_processor(data=transition)

    fps = 30
    logging.info("Running teleoperation loop...")

    try:
        while True:
            start_t = time.perf_counter()

            # Get teleop actions (processed)
            transition["action"] = torch.tensor([0.0] * 7, dtype=torch.float32)

            transition = step_env_and_process_transition(
                env=env,
                transition=transition,
                action=torch.tensor([0.0] * 7, dtype=torch.float32),
                env_processor=env_processor,
                action_processor=action_processor,
            )

            # Maintain loop rate
            dt_s = time.perf_counter() - start_t
            precise_sleep(1 / fps - dt_s)

    except KeyboardInterrupt:
        logging.info("Teleoperation stopped.")
    finally:
        env.close()


def main():
    teleop()


if __name__ == "__main__":
    main()
