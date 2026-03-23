import logging
import time

import torch

from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.processor import TransitionKey
from lerobot.teleoperators import TeleopEvents
from lerobot.teleoperators.spacemouse import SpacemouseConfig
from lerobot.utils.robot_utils import precise_sleep
from share.envs.manipulation_primitive_net.transitions import OnSuccess, OnTimeLimit
from share.envs.manipulation_primitive.task_frame import TaskFrame, PolicyMode, ControlMode
from share.envs.manipulation_primitive.config_manipulation_primitive import ManipulationPrimitiveConfig, EventConfig, \
    ManipulationPrimitiveProcessorConfig, GripperConfig
from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import ManipulationPrimitiveNetConfig
from share.envs.manipulation_primitive_net.env_manipulation_primitive_net import ManipulationPrimitiveNet
from share.robots.lerobot_robot_ur.lerobot_robot_urV2 import URV2Config
from share.teleoperators.lerobot_teleoperator_delta_keyboard.lerobot_teleoperator_delta_keyboard import \
    KeyboardVelocityTeleopConfig

_event = EventConfig(
    foot_switch_mapping={
        (TeleopEvents.SUCCESS,): {"device": 4, "toggle": False}
    }
)

_gripper = GripperConfig(
    enable=True,
    discretize=True
)

reset_cfg = ManipulationPrimitiveConfig(
    task_frame=TaskFrame(
        target=[0.0] * 6,
        policy_mode=[PolicyMode.RELATIVE] * 6,
        control_mode=[ControlMode.POS] * 6,
        kp=[2500, 2500, 2500, 100, 100, 100],
        kd=[960, 960, 320, 6, 6, 6]
    ),
    processor=ManipulationPrimitiveProcessorConfig(events=_event)
)

learn_cfg = ManipulationPrimitiveConfig(
    task_frame=TaskFrame(
        target=[0.0] * 6,
        policy_mode=[PolicyMode.RELATIVE] * 3 + [None] * 3,
        control_mode=[ControlMode.POS] * 3 + [ControlMode.VEL] * 3,
        kp=[2500, 2500, 2500, 100, 100, 100],
        kd=[960, 960, 320, 6, 6, 6]
    ),
    processor=ManipulationPrimitiveProcessorConfig(events=_event, gripper=_gripper)
)

# Define the Net with Diverse Transitions
net_cfg = ManipulationPrimitiveNetConfig(
    fps=30,
    start_primitive="reset",
    reset_primitive="reset",
    primitives={
        "reset": reset_cfg,
        "learn": learn_cfg,
    },
    transitions=[
        OnSuccess(source="reset", target="learn"),
        OnSuccess(source="learn", target="reset"),
        #OnTimeLimit(source="learn", target="reset", max_steps=1000)
    ],
    robot=URV2Config(
        robot_ip="172.22.22.2",
        frequency=500,
        soft_real_time=True,
        rt_core=3,
        use_gripper=True
    ),
    teleop=SpacemouseConfig(action_scale=[0.5, 0.5, 0.5, 0.7, 0.7, 0.7]),
    cameras={
        "main": RealSenseCameraConfig(serial_number_or_name="352122273250")
    }
)

def run_demo():
    net = ManipulationPrimitiveNet(net_cfg)
    net.reset()

    print(f"--- ROLLOUT START: {net.active_primitive} ---")

    # Run until we hit the terminal stage
    for i in range(100_000):
        start_loop_t = time.perf_counter()

        # Dummy action tensor matching the search primitive's 3 adaptive dimensions
        action = torch.randn(net.action_dim)

        if net._active == "learn":
            x = 5

        transition = net.step(action)

        if transition[TransitionKey.DONE]:
            net.reset()

        dt_load = time.perf_counter() - start_loop_t
        precise_sleep(1 / net.config.fps - dt_load)
        print(
            f"Running [{net._active}], "
            f"dt_load: {dt_load * 1000:5.2f}ms ({1 / dt_load:3.1f}hz)"
            f"{transition[TransitionKey.ACTION]}"
        )

    # The "Two Processor Step Reset" happens here
    print("\n--- TRIGGERING SYSTEM RESET ---")
    net.reset()


if __name__ == "__main__":
    run_demo()
