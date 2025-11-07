from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from pynput import keyboard

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.envs.configs import HilSerlRobotEnvConfig, EnvConfig
from lerobot.envs.robot_env import RobotEnv
from lerobot.processor.migrate_calibration_processor import MigrateCalibrationObsProcessorStep, MigrateInterventionActionProcessorStep
from lerobot.teleoperators import TeleopEvents
from tests.mocks.mock_robot import MockRobotConfig
from tests.mocks.mock_teleop import MockTeleopConfig


@dataclass
@EnvConfig.register_subclass("mock")
class MockEnvConfig(HilSerlRobotEnvConfig):
    use_aloha_cameras: bool = False

    def __post_init__(self):
        self.robot = {
            "left": MockRobotConfig(n_motors=6),
            "right": MockRobotConfig(n_motors=6)
        }
        self.teleop = {
            "left": MockTeleopConfig(n_motors=6),
            "right": MockTeleopConfig(n_motors=6)
        }

        if self.use_aloha_cameras:
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

        self.processor.hooks.time_action_processor = True
        self.processor.hooks.time_env_processor = True
        self.processor.hooks.log_every = 1
        self.processor.gripper.use_gripper = False
        self.processor.reset.terminate_on_success = True
        self.processor.reset.teleop_on_reset = True
        self.processor.reset.reset_time_s = 10.0
        #self.processor.control_time_s = 10.0
        self.processor.events.key_mapping = {
            TeleopEvents.RERECORD_EPISODE: keyboard.Key.left,
            TeleopEvents.TERMINATE_EPISODE: keyboard.Key.right
        }

        super().__post_init__()

