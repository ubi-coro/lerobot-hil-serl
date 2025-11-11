from dataclasses import dataclass

from lerobot.share.configs import DatasetRecordConfig


@dataclass
@DatasetRecordConfig.register_subclass("aloha_unfolding")
class AlohaUnfoldingDatasetConfig(DatasetRecordConfig):
    repo_id: str = "hoodie_unfolding_v3/base"
    single_task: str = "Unfold the hoodie"
    root: str = "/media/nvme1/jstranghoener/lerobot/data/jannick-st/hoodie_unfolding_v3/base"
    num_episodes: int = 30
    episode_time_s: int = 5.0
    reset_time_s: int = 5.0
    push_to_hub: bool = False
    video_encoding_batch_size: int = 30  # write to disk every 30 episodes
    num_image_writer_processes: int = 1
    num_image_writer_threads_per_camera: int = 4

