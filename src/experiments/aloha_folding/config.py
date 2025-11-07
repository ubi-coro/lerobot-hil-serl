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
@DatasetRecordConfig.register_subclass("aloha_folding")
class AlohaFoldingDatasetConfig(DatasetRecordConfig):
    repo_id: str = "hoodie_folding/base"
    single_task: str = "Fold the hoodie"
    root: str = "/media/nvme1/jstranghoener/lerobot/data/jannick-st/hoodie_folding_v3/base"
    num_episodes: int = 30
    episode_time_s: int = 5.0
    reset_time_s: int = 5.0
    push_to_hub: bool = False
    video_encoding_batch_size: int = 30  # write to disk every 30 episodes
    num_image_writer_processes: int = 1
    num_image_writer_threads_per_camera: int = 4

