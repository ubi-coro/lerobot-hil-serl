from dataclasses import dataclass

from lerobot.share.configs import DatasetRecordConfig


@dataclass
@DatasetRecordConfig.register_subclass("aloha_folding")
class AlohaFoldingDatasetConfig(DatasetRecordConfig):
    repo_id: str = "hoodie_folding_v3/hoodie_folding_eval"
    single_task: str = "Fold the hoodie"
    root: str = "/media/nvme1/jstranghoener/lerobot/data/jannick-st/hoodie_folding_v3/hoodie_folding_eval"
    num_episodes: int = 50
    episode_time_s: int = 90.0
    reset_time_s: int = 15.0
    push_to_hub: bool = False

