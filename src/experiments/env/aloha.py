import time
from dataclasses import dataclass, field
from pathlib import Path

from pynput import keyboard

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.envs import RobotEnvConfig
from lerobot.envs.configs import EnvConfig
from lerobot.robots import RobotConfig
from lerobot.robots.viperx.viperx import ViperXConfig
from lerobot.teleoperators import TeleopEvents, TeleoperatorConfig
from lerobot.teleoperators.widowx.widowx import WidowXConfig


@dataclass
@EnvConfig.register_subclass("aloha_bimanual")
class AlohaBimanualEnvConfig(RobotEnvConfig):
    benchmark: bool = False

    def __post_init__(self):
        self.robot = {
            "left": ViperXConfig(port="/dev/ttyDXL_follower_left", id="left", max_relative_target=None),
            "right": ViperXConfig(port="/dev/ttyDXL_follower_right", id="right", max_relative_target=None),
        }
        self.teleop = {
            "left": WidowXConfig(port="/dev/ttyDXL_leader_left", id="left"),
            "right": WidowXConfig(port="/dev/ttyDXL_leader_right", id="right")
        }
        self.cameras = {
            "cam_low": OpenCVCameraConfig(
                index_or_path=Path("/dev/CAM_LOW"),
                fps=30,
                width=640,
                height=480,
                fourcc="MJPG"
            ),
            "cam_top": OpenCVCameraConfig(
                index_or_path=Path("/dev/CAM_HIGH"),
                fps=30,
                width=640,
                height=480,
                fourcc="MJPG"
            ),
            "cam_right_wrist": RealSenseCameraConfig(
                serial_number_or_name="218622272856",
                fps=30,
                width=640,
                height=480,
            ),
            "cam_left_wrist": RealSenseCameraConfig(
                serial_number_or_name="218722270675",
                fps=30,
                width=640,
                height=480,
            )
        }

        self.processor.gripper.use_gripper = True
        self.processor.reset.terminate_on_success = True
        self.processor.events.foot_switch_mapping = {
            (TeleopEvents.SUCCESS,): {"device": 2, "toggle": False},
            (TeleopEvents.IS_INTERVENTION,): {"device": 7, "toggle": True},
        }
        self.processor.events.key_mapping = {
            TeleopEvents.RERECORD_EPISODE: keyboard.Key.left
        }

        if self.benchmark:
            self.processor.hooks.time_env_processor = True
            self.processor.hooks.time_action_processor = True

        super().__post_init__()


@dataclass
@EnvConfig.register_subclass("aloha_bimanual_safe")
class AlohaBimanualSafeEnvConfig(AlohaBimanualEnvConfig):
    max_relative_target: float = 0.25

    def __post_init__(self):
        super().__post_init__()

        for name in self.robot:
            self.robot[name].max_relative_target = self.max_relative_target

        for name in self.teleop:
            self.teleop[name].max_relative_target = self.max_relative_target


@dataclass
@EnvConfig.register_subclass("aloha_single")
class AlohaSingleEnvConfig(RobotEnvConfig):
    teleop: TeleoperatorConfig = WidowXConfig(port="/dev/ttyDXL_leader_left", id="left")
    robot: RobotConfig = ViperXConfig(port="/dev/ttyDXL_follower_left", id="left")
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "cam_low": OpenCVCameraConfig(
                index_or_path=Path("/dev/CAM_LOW"),
                fps=30,
                width=640,
                height=480,
            ),
            "cam_top": OpenCVCameraConfig(
                index_or_path=Path("/dev/CAM_HIGH"),
                fps=30,
                width=640,
                height=480,
            ),
            "cam_right_wrist": RealSenseCameraConfig(
                serial_number_or_name="218622272856",
                fps=30,
                width=640,
                height=480,
            ),
            "cam_left_wrist": RealSenseCameraConfig(
                serial_number_or_name="218722270675",
                fps=30,
                width=640,
                height=480,
            )
        }
    )

    def __post_init__(self):
        self.processor.gripper.use_gripper = True
        self.processor.events.foot_switch_mapping = {
            (TeleopEvents.TERMINATE_EPISODE,): {"device": 3, "toggle": False},
            (TeleopEvents.IS_INTERVENTION, ): {"device": 6, "toggle": True},
        }
