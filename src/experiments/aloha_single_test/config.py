from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.envs.configs import HilSerlRobotEnvConfig, EnvConfig
from lerobot.robots import RobotConfig
from lerobot.robots.viperx import ViperXConfig
from lerobot.share.configs import DatasetRecordConfig
from lerobot.teleoperators import TeleoperatorConfig, TeleopEvents
from lerobot.teleoperators.widowx import WidowXConfig


@dataclass
@EnvConfig.register_subclass("aloha_single")
class AlohaSingleEnvConfig(HilSerlRobotEnvConfig):
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


@dataclass
@DatasetRecordConfig.register_subclass("aloha_single")
class AlohaSingleDatasetConfig(DatasetRecordConfig):
    repo_id: str = "test/aloha_single"
    single_task: str = "test"
    root: str = "/media/nvme1/jstranghoener/lerobot/data/test/aloha_single"



