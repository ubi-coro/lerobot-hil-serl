
from dataclasses import dataclass

from lerobot.share.configs import DatasetRecordConfig


@dataclass
@DatasetRecordConfig.register_subclass("aloha_cable")
class AlohaCableDatasetConfig(DatasetRecordConfig):
    repo_id: str = "cable_v3/test"
    single_task: str = "Insert the cable"
    root: str = "/media/nvme1/jstranghoener/lerobot/data/jannick-st/cable_v3/test"
    num_episodes: int = 30
    episode_time_s: int = 80.0
    reset_time_s: int = 10.0
    push_to_hub: bool = False
    video_encoding_batch_size: int = 0  # write to disk every 30 episodes
    num_image_writer_processes: int = 1
    num_image_writer_threads_per_camera: int = 4

