import abc
from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Tuple, Type, Any

from pynput import keyboard

import draccus
import numpy as np

from experiments import AlohaBimanualEnvConfigV2, AlohaBimanualEnvConfig
from lerobot.cameras import CameraConfig, Camera
from lerobot.configs.types import PipelineFeatureType
from lerobot.envs import EnvConfig, RobotEnvConfig, RobotEnv
from lerobot.envs.configs import HILSerlProcessorConfig
from lerobot.envs.factory import RobotEnvInterface, make_env_config, make_env
from lerobot.processor import DataProcessorPipeline
from lerobot.processor.migrate_calibration_processor import MigrateCalibrationObsProcessorStep
from lerobot.robots import RobotConfig, Robot
from lerobot.robots.viperx import ViperXConfig
from lerobot.sim.configuration_mujococamera import MujocoCameraConfig
from lerobot.sim.mujoco_utils.mujoco_wrapper import init_with_existing_sim
from lerobot.sim.mujoco_utils.sim_singleton import init_sim, get_sim
from lerobot.sim.sim_viperx import SimViperXConfig
from lerobot.teleoperators import TeleopEvents
from lerobot.teleoperators.widowx import WidowXConfig
from lerobot.utils.constants import ACTION
from tests.mocks.mock_teleop import MockTeleopConfig


@dataclass
class SimConfig(draccus.ChoiceRegistry, abc.ABC):
    env: str
    task_name: str
    simulated_arms: List[str]
    calibration_dir: str
    viewer: str
    image_keys: List[str]

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@SimConfig.register_subclass("aloha")
@dataclass
class AlohaSimConfig(SimConfig):
    env: str = "aloha"
    task_name: str = "stacking"
    viewer: str = "mujoco"
    image_keys: List[str] = field(default_factory=lambda: [
        "wrist_cam_right",
        "wrist_cam_left",
        "teleoperator_pov",
        "overhead_cam",
        "worms_eye_cam",
    ])
    simulated_arms: List[str] = field(default_factory=lambda: [
        "left_follower",
        "right_follower"])
    calibration_dir: str = ".cache/calibration/aloha_sim"



@CameraConfig.register_subclass("mujoco_sim_cam")
@dataclass
class SimCameraConfig(CameraConfig):
    name = ""
    fps: int | None = 30
    width: int | None = 640
    height: int | None = 480


@dataclass
class SimConfig(draccus.ChoiceRegistry, abc.ABC):
    env: str
    task_name: str
    simulated_arms: List[str]
    calibration_dir: str
    viewer: str
    image_keys: List[str]


    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@SimConfig.register_subclass("aloha")
@dataclass
class AlohaSimConfig(SimConfig):
    env: str = "aloha"
    task_name: str = "stacking"
    viewer: str = "mujoco"
    image_keys: List[str] = field(default_factory=lambda: [
        "wrist_cam_right",
        "wrist_cam_left",
        "teleoperator_pov",
        "overhead_cam",
    ])
    simulated_arms: List[str] = field(default_factory=lambda: [
        "left",
        "right"])
    calibration_dir: str = ".cache/calibration/aloha_sim"



class SimRobotEnv(RobotEnv):
    def __init__(
            self,
            robot_dict: dict[str, Robot],
            cameras: dict[str, Camera] | None = None,
            processor: HILSerlProcessorConfig | None = None
    ) -> None:
        super().__init__(robot_dict, cameras, processor)
        self.sim = get_sim()

    def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        obs_sim, events_sim = self.sim.reset(seed=seed, options=options)
        obs, events = super().reset(seed=seed, options=options)
        return obs, events

    def step(self, action):
        obs, reward, terminated, truncated, events = super().step(action)
        _obs = self.sim.step(action)
        return obs, reward, terminated, truncated, events

@dataclass
@EnvConfig.register_subclass("sim_aloha")
class SimAlohaEnvConfig(RobotEnvConfig):
    benchmark: bool = False
    sim: SimConfig = field(default_factory=AlohaSimConfig)  # TODO(jzilke)

    def __post_init__(self):
        self._init_sim()

        self.kinematics_solver = None
        self.robot = {
            "left": SimViperXConfig(id="left", action_idx=(0,7)),
            "right": SimViperXConfig(id="right", action_idx=(7,14))
        }
        self.teleop = {
            "left": WidowXConfig(port="/dev/ttyDXL_leader_left", id="left"),
            "right": WidowXConfig(port="/dev/ttyDXL_leader_right", id="right")
        }
        self.cameras = {
            "cam_left_wrist": MujocoCameraConfig(
            )
        }

        self.processor.gripper.use_gripper = True
        self.processor.reset.terminate_on_success = True
        self.processor.events.foot_switch_mapping = {
            (TeleopEvents.SUCCESS,): {"device": 2, "toggle": False},
            (TeleopEvents.IS_INTERVENTION,): {"device": 7, "toggle": True},
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

    def _init_sim(self):
        env_cfg = make_env_config(self.sim.type)
        gym_env = make_env(env_cfg)
        gym_env = gym_env.get(self.sim.type).get(0)
        init_sim(gym_env)
