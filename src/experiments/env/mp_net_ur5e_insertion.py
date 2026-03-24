from dataclasses import dataclass

import numpy as np
from pynput import keyboard

from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.envs import EnvConfig
from lerobot.teleoperators import TeleopEvents
from lerobot.teleoperators.spacemouse import SpacemouseConfig
from share.envs.manipulation_primitive_net.transitions import OnSuccess
from share.envs.manipulation_primitive.task_frame import TaskFrame, PolicyMode, ControlMode
from share.envs.manipulation_primitive.config_manipulation_primitive import ManipulationPrimitiveConfig, EventConfig, \
    ManipulationPrimitiveProcessorConfig, GripperConfig, ObservationConfig
from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import ManipulationPrimitiveNetConfig
from share.robots.lerobot_robot_ur.lerobot_robot_urV2 import URV2Config


mp = ManipulationPrimitiveConfig(
    task_frame=TaskFrame(
        target=[0.0, 0.0, 0.0, -float(np.pi), 0.0, float(np.pi) / 2],
        policy_mode=[PolicyMode.RELATIVE] * 3 + [None] * 2 + [PolicyMode.RELATIVE],
        control_mode=[ControlMode.POS] * 6,
        kp=[2500, 2500, 2500, 100, 100, 100],
        kd=[960, 960, 320, 6, 6, 6]
    ),
    processor=ManipulationPrimitiveProcessorConfig(
        gripper=GripperConfig(
            enable=True,
            discretize=True,
            max_pos=0.718,
            min_pos=0.6
        ),
        observation=ObservationConfig(
            add_joint_position_to_observation=False,
            add_ee_pos_to_observation=False,
            add_ee_velocity_to_observation=True,
            add_ee_wrench_to_observation=True
        ),
        events=EventConfig(
            foot_switch_mapping={
                (TeleopEvents.SUCCESS,): {"device": 4, "toggle": False}
            },
            key_mapping={
                TeleopEvents.RERECORD_EPISODE: keyboard.Key.left,
                TeleopEvents.STOP_RECORDING: keyboard.Key.right
            }
        )
    )
)

@dataclass
@EnvConfig.register_subclass("mp_net_ur5e_insertion")
class MPNetUR5eInsertionConfig(ManipulationPrimitiveNetConfig):
    def __post_init__(self):
        self.fps = 30
        self.start_primitive = "reset"
        self.reset_primitive = "reset"
        self.primitives = {
            "reset": mp,
            "learn": mp,
        }
        self.transitions = [
            OnSuccess(source="reset", target="learn"),
            OnSuccess(source="learn", target="reset"),
            # OnTimeLimit(source="learn", target="reset", max_steps=1000)
        ]
        self.robot = URV2Config(
            robot_ip="172.22.22.2",
            frequency=500,
            soft_real_time=True,
            rt_core=3,
            use_gripper=True,
            gripper_vel=0.2,
            wrench_limits=[20.0, 20.0, 20.0, 5.0, 5.0, 5.0],
            compliance_safety_enable=[True] * 3 + [False] * 3,
            compliance_desired_wrench=[5.0] * 6,
            compliance_adaptive_limit_min=[0.12] * 6
        )
        self.teleop=SpacemouseConfig(action_scale=[0.2, 0.2, 0.2, 0.7, 0.7, 0.7])
        self.cameras={
            "wrist": RealSenseCameraConfig(serial_number_or_name="352122273250"),
            "low": RealSenseCameraConfig(serial_number_or_name="352122271573")
        }
        super().__post_init__()