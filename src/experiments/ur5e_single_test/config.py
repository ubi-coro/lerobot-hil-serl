from dataclasses import dataclass

from lerobot.envs.configs import EnvConfig, TFHilSerlRobotEnvConfig
from lerobot.robots.ur import URConfig
from lerobot.share.configs import DatasetRecordConfig
from lerobot.teleoperators.spacemouse import SpacemouseConfig


@dataclass
@EnvConfig.register_subclass("ur5e_single")
class UR5eSingleEnvConfig(TFHilSerlRobotEnvConfig):

    def __post_init__(self):

        self.robot = URConfig(
            model="ur5e",
            robot_ip="192.168.1.10",
            use_gripper=True,
            soft_real_time=True,
            verbose=True,
            control_frequency=30.0,
            wrench_limits=[100.0, 100.0, 100.0, 30.0, 30.0, 30.0]
        )
        self.teleop = SpacemouseConfig()

        self.processor.task_frame.action_scale = [3.0, 3.0, 3.0, 1.5, 1.5, 1.5, 1.0]

        self.processor.hooks.time_action_processor = False
        self.processor.hooks.time_env_processor = False
        self.processor.hooks.log_every = 1
        self.processor.gripper.use_gripper = True
        self.processor.reset.terminate_on_success = True
        self.processor.reset.teleop_on_reset = True
        self.processor.reset.reset_time_s = 5.0

        super().__post_init__()


@dataclass
@DatasetRecordConfig.register_subclass("ur5e_single")
class UR5eSingleDatasetConfigV2(DatasetRecordConfig):
    repo_id: str = "test/ur5e_single"
    single_task: str = "test"
    root: str = "/media/nvme1/jstranghoener/lerobot/data/test/ur5e_single"

