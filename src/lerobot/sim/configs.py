import abc
from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Tuple

from pynput import keyboard

import draccus
import numpy as np

from experiments import AlohaBimanualEnvConfigV2
from lerobot.cameras import CameraConfig
from lerobot.configs.types import PipelineFeatureType
from lerobot.envs import EnvConfig, RobotEnvConfig
from lerobot.envs.factory import RobotEnvInterface, make_env_config, make_env
from lerobot.processor import DataProcessorPipeline
from lerobot.processor.migrate_calibration_processor import MigrateCalibrationObsProcessorStep
from lerobot.robots import RobotConfig
from lerobot.sim.configuration_mujococamera import MujocoCameraConfig
from lerobot.sim.mujoco_utils.mujoco_wrapper import init_with_existing_sim
from lerobot.sim.sim_viperx import SimViperXConfig
from lerobot.teleoperators import TeleopEvents
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
    viewer: str = "both"
    image_keys: List[str] = field(default_factory=lambda: [
        "wrist_cam_right",
        "wrist_cam_left",
        "teleoperator_pov",
        "overhead_cam",
    ])
    simulated_arms: List[str] = field(default_factory=lambda: [
        "left_follower",
        "right_follower"])
    calibration_dir: str = ".cache/calibration/aloha_sim"



@dataclass
class SimRobotEnvConfig(RobotEnvConfig):
    sim: SimConfig | None = None # TODO(jzilke)
    robot: RobotConfig | dict[str, RobotConfig] | None = None

    def make(self, device: str = "cpu") -> Tuple[RobotEnvInterface, DataProcessorPipeline, DataProcessorPipeline]:
        gym_env = self._make_sim()
        env, env_processor, action_processor = super().make(device)
        # gym_env.envs[0].unwrapped._env._task.get_observation(gym_env.envs[0].unwrapped._env.physics)
        return env, env_processor, action_processor

    def _make_sim(self):
        env_cfg = make_env_config(self.type) # TODO(jzilke)
        gym_env = make_env(env_cfg)

        gym_env = gym_env.get('aloha').get(0) # TODO(jzilke)
        gym_env.reset()
        gym_env.step(actions=np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))

        physics = gym_env.envs[0].unwrapped._env.physics
        model = physics.model.ptr
        data = physics.data.ptr
        init_with_existing_sim(model, data)

        return gym_env



@dataclass
@EnvConfig.register_subclass("sim_aloha")
class SimAlohaEnvConfig(AlohaBimanualEnvConfigV2):
    def __post_init__(self):
        self.kinematics_solver = None
        self.robot = {
            "left": SimViperXConfig(port="/dev/ttyDXL_follower_left", id="left"),
            "right": SimViperXConfig(port="/dev/ttyDXL_follower_right", id="right")
        }
        self.teleop = {
            "left": MockTeleopConfig(n_motors=6),
            "right": MockTeleopConfig(n_motors=6)       }
        self.cameras = {
            "cam_left_wrist": MujocoCameraConfig(
            )
        }

        self.processor.hooks.time_action_processor = False
        self.processor.hooks.time_env_processor = False
        self.processor.hooks.log_every = 1
        self.processor.gripper.use_gripper = True
        self.processor.reset.terminate_on_success = True
        self.processor.reset.teleop_on_reset = True
        self.processor.reset.reset_time_s = 10.0
        #self.processor.control_time_s = 10.0
        self.processor.events.foot_switch_mapping = {
            # (TeleopEvents.SUCCESS,): {"device": 3, "toggle": False},
            # (TeleopEvents.IS_INTERVENTION,): {"device": 6, "toggle": True},
        }
        self.processor.events.key_mapping = {
            TeleopEvents.RERECORD_EPISODE: keyboard.Key.left
        }