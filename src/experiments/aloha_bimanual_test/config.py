from dataclasses import dataclass
from pathlib import Path

from pynput import keyboard

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.envs.configs import HilSerlRobotEnvConfig, EnvConfig
from lerobot.robots.viperx import ViperXConfig
from lerobot.share.configs import DatasetRecordConfig
from lerobot.teleoperators import TeleopEvents
from lerobot.teleoperators.widowx import WidowXConfig


@dataclass
@EnvConfig.register_subclass("aloha_bimanual")
class AlohaBimanualEnvConfig(HilSerlRobotEnvConfig):

    def __post_init__(self):
        self.robot = {
            "left": ViperXConfig(port="/dev/ttyDXL_follower_left", id="left"),
            "right": ViperXConfig(port="/dev/ttyDXL_follower_right", id="right")
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

        self.processor.gripper.use_gripper = True
        self.processor.reset.terminate_on_success = True
        self.processor.events.foot_switch_mapping = {
            (TeleopEvents.SUCCESS,): {"device": 3, "toggle": False},
            (TeleopEvents.IS_INTERVENTION,): {"device": 6, "toggle": True},
        }
        self.processor.events.key_mapping = {
            TeleopEvents.RERECORD_EPISODE: keyboard.Key.left
        }


@dataclass
@DatasetRecordConfig.register_subclass("aloha_bimanual")
class AlohaBimanualDatasetConfig(DatasetRecordConfig):
    repo_id: str = "test/aloha_bimanual"
    single_task: str = "Fold the hoodie"
    root: str = "/media/nvme1/jstranghoener/lerobot/data/test/aloha_bimanual"

