import abc
from dataclasses import dataclass, field
from typing import List, Tuple, Type, Any

from pynput import keyboard

import draccus

from lerobot.cameras import CameraConfig, Camera
from lerobot.envs import EnvConfig, RobotEnvConfig, RobotEnv
from lerobot.envs.configs import HILSerlProcessorConfig
from lerobot.envs.factory import RobotEnvInterface
from lerobot.processor import DataProcessorPipeline
from lerobot.robots import Robot
from lerobot.robots.viperx import ViperXConfig
from lerobot.sim.configuration_mujococamera import MujocoCameraConfig
from lerobot.sim.mujoco_utils.sim_singleton import SimManager
from lerobot.sim.sim_viperx import SimViperXConfig
from lerobot.teleoperators import TeleopEvents

from tests.mocks.mock_robot import MockRobot, MockRobotConfig
from tests.mocks.mock_teleop import MockTeleopConfig


@dataclass
class SimConfig(draccus.ChoiceRegistry, abc.ABC):
    env: str
    viewer: str
    image_keys: List[str]
    calibration_dir: str


    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@SimConfig.register_subclass("aloha")
@dataclass
class AlohaSimConfig(SimConfig):
    env: str = "aloha"
    viewer: str = "mujoco"
    image_keys: List[str] = field(default_factory=lambda: [
        "wrist_cam_right",
        "wrist_cam_left",
        "teleoperator_pov",
        "overhead_cam",
    ])
    calibration_dir: str = ".cache/calibration/aloha_sim"



class SimRobotEnv(RobotEnv):
    def __init__(
            self,
            robot_dict: dict[str, Robot],
            cameras: dict[str, Camera] | None = None,
            processor: HILSerlProcessorConfig | None = None
    ) -> None:
        self.sim = SimManager.get()
        super().__init__(robot_dict, cameras, processor)

    def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        _ = self.sim.reset(seed=seed, options=options)
        obs, events = super().reset(seed=seed, options=options)
        return obs, events

    def _send_actions(self, action):
        super()._send_actions(action)
        self.sim.step()

    def _setup_spaces(self) -> None:
        self.sim.action_order = self._joint_names_list
        self.sim.reset()
        super()._setup_spaces()

@dataclass
@EnvConfig.register_subclass("sim_aloha")
class SimAlohaEnvConfig(RobotEnvConfig):
    benchmark: bool = False
    sim: SimConfig = field(default_factory=AlohaSimConfig)  # TODO(jzilke)

    def __post_init__(self):
        self.kinematics_solver = None
        self.robot = {
            "left": MockRobotConfig(n_motors=7),
            "right": SimViperXConfig(id="right")
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
            (TeleopEvents.SUCCESS,): {"device": 24, "toggle": False},
            (TeleopEvents.IS_INTERVENTION,): {"device": 25, "toggle": True},
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

