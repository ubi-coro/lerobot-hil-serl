from dataclasses import dataclass

from lerobot.envs.configs import HilSerlRobotEnvConfig, EnvConfig
from lerobot.robots import RobotConfig
from lerobot.robots.viperx import ViperXConfig
from lerobot.share.record import DatasetRecordConfig
from lerobot.teleoperators import TeleoperatorConfig
from lerobot.teleoperators.widowx import WidowXConfig


@dataclass
@EnvConfig.register_subclass("aloha_single")
class AlohaSingleEnvConfig(HilSerlRobotEnvConfig):
    teleop: TeleoperatorConfig = WidowXConfig(port="/dev/ttyDXL_leader_left", id="left")
    robot: RobotConfig = ViperXConfig(port="/dev/ttyDXL_follower_left", id="left")

    def __post_init__(self):
        self.processor.gripper.use_gripper = True


@dataclass
@DatasetRecordConfig.register_subclass("aloha_single")
class AlohaSingleDatasetConfig(DatasetRecordConfig):
    repo_id: str = "test/aloha_single"
    single_task: str = "test"
    root: str = "/media/nvme1/jstranghoener/lerobot/data/test/aloha_single"#


if __name__ == "__main__":
    from lerobot.share.record import RecordConfig, record
    record(RecordConfig(env=AlohaSingleEnvConfig(), dataset=AlohaSingleDatasetConfig()))


