from dataclasses import dataclass
from typing import Literal

from pynput import keyboard

from lerobot.envs import TFRobotEnvConfig
from lerobot.envs.configs import EnvConfig
from lerobot.robots.ur import URConfig
from lerobot.robots.ur.tf_controller import TaskFrameCommand, AxisMode
from lerobot.teleoperators import TeleopEvents
from lerobot.teleoperators.spacemouse import SpacemouseConfig

@dataclass
@EnvConfig.register_subclass("ur3e_spacemouse")
class UR3eSpacemouseEnvConfig(TFRobotEnvConfig):
    fps: int = 30

    # velocities
    v_ee = 0.15
    omega_ee = 0.5
    v_gripper = 1.0

    controller: Literal["velocity", "impedance", "leaky_impedance"] = "leaky_impedance"

    def __post_init__(self):
        self.processor.control_time_s = 3600.0

        self.processor.gripper.use_gripper = True
        self.processor.reset.terminate_on_success = True
        self.processor.reset.teleop_on_reset = True
        self.processor.reset.reset_time_s = 15.0

        self.processor.events.key_mapping = {
            TeleopEvents.TERMINATE_EPISODE: keyboard.Key.right,
            TeleopEvents.RERECORD_EPISODE: keyboard.Key.left,
            TeleopEvents.SUCCESS: keyboard.Key.enter,
            TeleopEvents.STOP_RECORDING: keyboard.Key.down
        }

        kwargs = {}
        if self.controller != "leaky_impedance":  # disable automatic leaking
            kwargs["leak_rate_pos"] = 0.0
            kwargs["leak_rate_rot"] = 0.0

        # device configs
        self.robot = URConfig(
            model="ur3e",
            frequency=125.0,
            robot_ip="172.22.22.2",
            use_gripper=True,
            gripper_vel=self.v_gripper,
            soft_real_time=False,
            verbose=True,
            wrench_limits=[300.0, 300.0, 300.0, 20.0, 20.0, 20.0],
            **kwargs
        )

        control_mode = AxisMode.PURE_VEL if self.controller == "velocity" else AxisMode.IMPEDANCE_VEL
        control_dependent_action_scaling = 2.0 if self.controller == "velocity" else 1.0
        action_scale = 3 * [control_dependent_action_scaling * self.v_ee] + 3 * [control_dependent_action_scaling * self.omega_ee]
        self.teleop = SpacemouseConfig(
            action_scale=action_scale,
            gripper_close_button_idx=0,
            gripper_open_button_idx=1
        )

        # task frame configuration
        self.processor.task_frame.command = TaskFrameCommand(
                T_WF=6 * [0.0],
                target=6 * [0.0],
                mode=6 * [control_mode],
                kp=[5000, 5000, 1000, 100, 100, 20],
                kd=[480, 480, 96, 6, 6, 1.2],
                max_pose_rpy=6 * [float("inf")],
                min_pose_rpy=6 * [-float("inf")]
            )

        super().__post_init__()


