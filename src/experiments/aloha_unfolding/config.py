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
@DatasetRecordConfig.register_subclass("aloha_unfolding")
class AlohaUnfoldingDatasetConfig(DatasetRecordConfig):
    repo_id: str = "hoodie_unfolding/hoodie_unfolding_interactive_311025_act_281025"
    single_task: str = "Unfold the hoodie"
    root: str = "/media/nvme1/jstranghoener/lerobot/data/test/hoodie_unfolding/hoodie_unfolding_interactive_311025_act_281025"
    video_encoding_batch_size = 32
