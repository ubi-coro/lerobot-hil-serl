from dataclasses import dataclass

from lerobot.share.configs import DatasetRecordConfig


@dataclass
@DatasetRecordConfig.register_subclass("aloha_bimanual_lemgo_v2")
class AlohaBimanualDatasetConfigLemgoV2(DatasetRecordConfig):
    repo_id: str = "local/20251024_hoodie_folding_lemgo"
    single_task: str = "Hoodie Folding Lemgo"
    root: str = "/media/jannick/DATA/aloha_data_lerobot/jannick-st/eval_20251024_hoodie_folding_lemgo"
