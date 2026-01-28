from dataclasses import dataclass

from lerobot.share.configs import DatasetRecordConfig


@dataclass
@DatasetRecordConfig.register_subclass("aloha_unfolding")
class AlohaUnfoldingDatasetConfig(DatasetRecordConfig):
    repo_id: str = ("hoodie_unfolding_v3/hoodie_unfolding_pi0-150126_eval_280126_2")
    single_task: str = "Unfold the hoodie"
    root: str = "/media/nvme1/jstranghoener/lerobot/data/jannick-st/hoodie_unfolding_v3/hoodie_unfolding_pi0-150126_eval_280126_2"
    num_episodes: int = 50
    episode_time_s: int = 90.0
    reset_time_s: int = 15.0
    push_to_hub: bool = False

