from dataclasses import dataclass, field
from datetime import time, datetime

from lerobot.share.configs import DatasetRecordConfig


@dataclass
@DatasetRecordConfig.register_subclass("test")
class DatasetTestConfig(DatasetRecordConfig):
    repo_id: str = "test/test"
    single_task: str = "test"
    root: str = field(default_factory=lambda: "/home/jannick/data/test/"  + datetime.now().strftime("%Y%m%d-%H%M%S"))



