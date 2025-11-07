from dataclasses import dataclass

from lerobot.envs.configs import EnvConfig, TFHilSerlRobotEnvConfig
from lerobot.robots.ur import URConfig
from lerobot.share.configs import DatasetRecordConfig
from lerobot.teleoperators.spacemouse import SpacemouseConfig


@dataclass
@EnvConfig.register_subclass("ur5e_single")
class UR5eSingleEnvConfig(TFHilSerlRobotEnvConfig):

    def __post_init__(self):

        self.robot = {
            "left": URConfig(
                model="ur5e",
                robot_ip="192.168.1.10",
                use_gripper=True,
                gripper_soft_real_time=False,
                soft_real_time=True,
                rt_core=3,
                verbose=True,
                wrench_limits=[100.0, 100.0, 100.0, 20.0, 20.0, 20.0]
            ),
            "right": URConfig(
                model="ur5e",
                robot_ip="192.168.1.11",
                use_gripper=True,
                gripper_soft_real_time=False,
                soft_real_time=True,
                rt_core=4,
                verbose=True,
                wrench_limits=[100.0, 100.0, 100.0, 20.0, 20.0, 20.0]
            ),
        }
        self.teleop = {
            "left": SpacemouseConfig(path="/dev/hidraw2"),
            "right": SpacemouseConfig(path="/dev/hidraw5")
        }
        self.processor.task_frame.action_scale = [5.0, 5.0, 5.0, 1.5, 1.5, 1.5, 0.3,
                                                  5.0, 5.0, 5.0, 1.5, 1.5, 1.5, 0.3]

        self.processor.hooks.time_action_processor = False
        self.processor.hooks.time_env_processor = False
        self.processor.hooks.log_every = 1
        self.processor.gripper.use_gripper = True
        self.processor.gripper.offset = 0.7
        self.processor.reset.terminate_on_success = True
        self.processor.reset.teleop_on_reset = True
        self.processor.reset.reset_time_s = 5.0

        super().__post_init__()