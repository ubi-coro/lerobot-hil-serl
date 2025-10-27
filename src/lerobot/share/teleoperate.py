# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Teleoperate a robot using experiment-based configurations with calibration migration.

This script provides teleoperation with the same configuration system as recording,
including support for calibration migration for compatibility with new v3 calibration files.

Example:

```shell
lerobot-teleoperate \
    --env.type=aloha_bimanual_lemgo_v2 \
    --display_data=true
```

If you need to override specific parameters:
```shell
lerobot-teleoperate \
    --env.type=aloha_bimanual_lemgo_v2 \
    --env.robot.left.port=/dev/ttyDXL_follower_left \
    --env.robot.right.port=/dev/ttyDXL_follower_right \
    --env.teleop.left.port=/dev/ttyDXL_leader_left \
    --env.teleop.right.port=/dev/ttyDXL_leader_right \
    --display_data=true
```
"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import rerun as rr

from lerobot.configs import parser
from lerobot.envs.configs import EnvConfig
from lerobot.envs.robot_env import RobotEnv
from lerobot.processor import (
    create_transition,
    TransitionKey,
    RobotProcessorPipeline,
)
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from lerobot.utils.transition import Transition


@dataclass
class TeleoperateConfig:
    env: EnvConfig
    # Limit the maximum frames per second.
    fps: int = 30
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False


def teleop_loop(
    env: RobotEnv,
    fps: int,
    action_dim: int,
    action_processor: RobotProcessorPipeline[Transition, Transition],
    env_processor: RobotProcessorPipeline[Transition, Transition],
    display_data: bool = False,
    duration: float | None = None,
):
    """
    This function continuously reads actions from a teleoperation device through the environment
    processors, processes them through the calibration migration pipeline, sends them to the robot,
    and optionally displays the robot's state.

    Args:
        env: The robot environment containing robot, teleoperator, and cameras.
        fps: The target frequency for the control loop in frames per second.
        action_dim: The dimensionality of the action space.
        action_processor: Pipeline to process actions before sending to robot.
        env_processor: Pipeline to process environment observations.
        display_data: If True, fetches robot observations and displays them in console and Rerun.
        duration: The maximum duration of the teleoperation loop in seconds. If None, runs indefinitely.
    """

    obs, info = env.reset()
    env_processor.reset()
    action_processor.reset()

    # Process initial observation
    transition = create_transition(observation=obs, info=info)
    transition = env_processor(data=transition)

    start = time.perf_counter()
    
    # Calculate display length for formatting
    robot_action_features = []
    if hasattr(env.robot, 'action_features'):
        robot_action_features = env.robot.action_features
    elif isinstance(env.robot, dict):
        # Bimanual case
        for name, robot in env.robot.items():
            robot_action_features.extend([f"{name}.{feat}" for feat in robot.action_features])
    
    display_len = max(len(key) for key in robot_action_features) if robot_action_features else 20

    while True:
        loop_start = time.perf_counter()

        # Create dummy action (will be overridden by teleop through action_processor)
        import torch
        action = torch.zeros(action_dim, dtype=torch.float32)
        
        # Create action transition and process it (teleop intervention happens here)
        action_transition = create_transition(action=action, info={})
        processed_action_transition = action_processor(action_transition)
        
        # Check if episode should end
        if processed_action_transition.get(TransitionKey.DONE, False):
            logging.info("Episode ended by user")
            return

        # Step environment with processed action
        obs, reward, terminated, truncated, info = env.step(
            processed_action_transition[TransitionKey.ACTION]
        )

        # Process observation
        transition = create_transition(
            observation=obs,
            action=processed_action_transition[TransitionKey.ACTION],
            reward=reward,
            done=terminated,
            truncated=truncated,
            info=info,
        )
        transition = env_processor(transition)

        if display_data:
            # Log to Rerun
            rerun_obs = {
                k: v.numpy() if hasattr(v, 'numpy') else v
                for k, v in transition[TransitionKey.OBSERVATION].items()
            }
            log_rerun_data(
                observation=rerun_obs,
                action=transition[TransitionKey.ACTION],
            )

            # Display action values in console
            print("\n" + "-" * (display_len + 10))
            print(f"{'ACTION':<{display_len}} | {'VALUE':>7}")
            
            action_tensor = transition[TransitionKey.ACTION]
            if hasattr(action_tensor, 'tolist'):
                action_values = action_tensor.tolist()
            else:
                action_values = list(action_tensor)
            
            for i, (name, value) in enumerate(zip(robot_action_features, action_values)):
                print(f"{name:<{display_len}} | {value:>7.2f}")
            
            move_cursor_up(len(robot_action_features) + 5)

        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)
        loop_s = time.perf_counter() - loop_start
        print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

        if duration is not None and time.perf_counter() - start >= duration:
            return


@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    
    if cfg.display_data:
        init_rerun(session_name="teleoperation")

    # Create environment with processors from config
    env, env_processor, action_processor = cfg.env.make(device="cpu")

    try:
        teleop_loop(
            env=env,
            fps=cfg.fps,
            action_dim=cfg.env.action_dim,
            action_processor=action_processor,
            env_processor=env_processor,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s,
        )
    except KeyboardInterrupt:
        logging.info("Teleoperation interrupted by user")
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        env.close()


def main():
    import experiments  # noqa: F401 - registers experiment configs
    teleoperate()


if __name__ == "__main__":
    main()
