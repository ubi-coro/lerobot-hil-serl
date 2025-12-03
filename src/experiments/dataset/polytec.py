from dataclasses import dataclass

from lerobot.share.configs import DatasetRecordConfig


@dataclass
@DatasetRecordConfig.register_subclass("polytec")
class PolytecDatasetConfig(DatasetRecordConfig):
    repo_id: str = "polytec/base"
    single_task: str = "Insert the seal"
    root: str = "/home/jannick/data/polytec/base"
    num_episodes: int = 30
    push_to_hub: bool = False
    video_encoding_batch_size: int = 1
    num_image_writer_processes: int = 1
    num_image_writer_threads_per_camera: int = 4

