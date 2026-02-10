from dataclasses import dataclass, field
from typing import Type, Tuple

from pynput import keyboard

from lerobot.cameras.mujoco import MujocoCameraConfig
from lerobot.envs import RobotEnvConfig
from lerobot.envs.configs import EnvConfig
from lerobot.envs.factory import RobotEnvInterface
from lerobot.processor import DataProcessorPipeline
from lerobot.robots.viperx.config_viperx_sim_twin import ViperXSimTwinConfig
from lerobot.sim.configs import AlohaSimConfig, SimConfig
from lerobot.sim.mujoco_utils.sim_singleton import SimManager
from lerobot.sim.sim_robot_env import SimRobotEnv
from lerobot.robots.viperx import SimViperXConfig, ViperXConfig
from lerobot.teleoperators import TeleopEvents
from tests.mocks.mock_teleop import MockTeleopConfig


@dataclass
@EnvConfig.register_subclass("aloha_sim_twin")
class AlohaTwinEnvConfig(RobotEnvConfig):
    benchmark: bool = False
    sim: SimConfig = field(default_factory=AlohaSimConfig)  # TODO(jzilke)

    def __post_init__(self):
        self.kinematics_solver = None
        self.robot = {
            "left": ViperXSimTwinConfig(
                sim_config=SimViperXConfig(id="left"),
                real_config=ViperXConfig(port="/dev/ttyDXL_follower_left", id="left", max_relative_target=None),
            ),
            "right": ViperXSimTwinConfig(
                sim_config=SimViperXConfig(id="right"),
                real_config=ViperXConfig(port="/dev/ttyDXL_follower_right", id="right", max_relative_target=None),
            ),
        }
        self.teleop = {
            "left": MockTeleopConfig(id="left", n_motors=7),
            "right": MockTeleopConfig(id="right", n_motors=7)
        }
        self.cameras = {
            "cam_low": MujocoCameraConfig(
                mujoco_id="worms_eye_cam",
                fps=30,
                width=640,
                height=480,
            ),
            "cam_top": MujocoCameraConfig(
                mujoco_id="overhead_cam",
                fps=30,
                width=640,
                height=480,
            ),
            "cam_right_wrist": MujocoCameraConfig(
                mujoco_id="wrist_cam_right",
                fps=30,
                width=640,
                height=480,
            ),
            "cam_left_wrist": MujocoCameraConfig(
                mujoco_id="wrist_cam_left",
                fps=30,
                width=640,
                height=480,
            )
        }
        self.processor.gripper.use_gripper = True
        self.processor.reset.terminate_on_success = True
        self.processor.events.foot_switch_mapping = {
            # (TeleopEvents.SUCCESS,): {"device": 24, "toggle": False},
            # (TeleopEvents.IS_INTERVENTION,): {"device": 25, "toggle": True},
        }
        self.processor.events.key_mapping = {
            TeleopEvents.RERECORD_EPISODE: keyboard.Key.left,
            TeleopEvents.STOP_RECORDING: keyboard.Key.down
        }

        if self.benchmark:
            self.processor.hooks.time_env_processor = True
            self.processor.hooks.time_action_processor = True

        super().__post_init__()

    @property
    def env_cls(self) -> Type[RobotEnvInterface]:
        return SimRobotEnv

    def make(self, device: str = "cpu") -> Tuple[RobotEnvInterface, DataProcessorPipeline, DataProcessorPipeline]:
        SimManager.init(self.sim)
        env, env_processor, action_processor = super().make(device=device)
        return env, env_processor, action_processor
