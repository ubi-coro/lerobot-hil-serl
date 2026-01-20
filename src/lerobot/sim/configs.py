import abc
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import draccus
import numpy as np

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.envs import EnvConfig, RobotEnvConfig
from lerobot.envs.factory import RobotEnvInterface, make_env_config, make_env
from lerobot.processor import DataProcessorPipeline
from lerobot.robots import RobotConfig
from lerobot.robots.viperx import ViperXConfig
from lerobot.sim.mujoco_utils.mujoco_wrapper import init_with_existing_sim
from lerobot.teleoperators.widowx import WidowXConfig
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
class SimRobotConfig(RobotConfig):
    pass

@SimRobotConfig.register_subclass("sim_viperx")
@dataclass
class SimViperXConfig(SimRobotConfig):
    port: str  # Port to connect to the arm

    disable_torque_on_disconnect: bool = False

    # /!\ FOR SAFETY, READ THIS /!\
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    # For Aloha, for every goal position request, motor rotations are capped at 5 degrees by default.
    # When you feel more confident with teleoperation or running the policy, you can extend
    # this safety limit and even removing it by setting it to `null`.
    max_relative_target: float | None = 5.0

    # The duration of the velocity-based time profile
    # Higher values lead to smoother motions, but increase lag.
    moving_time: float = 0.1

    # cameras
    cameras: dict[str, SimCameraConfig] = field(default_factory=dict)

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
    robot: SimRobotConfig | dict[str, SimRobotConfig] | None = None

    def make(self, device: str = "cpu") -> Tuple[RobotEnvInterface, DataProcessorPipeline, DataProcessorPipeline]:
        gym_env = self._make_sim()
        env, env_processor, action_processor = super().make(device)
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
class SimAlohaEnvConfig(SimRobotEnvConfig):
    def __post_init__(self):
        self.robot = {
            "left": SimViperXConfig(port="/dev/ttyDXL_follower_left", id="left"),
            "right": SimViperXConfig(port="/dev/ttyDXL_follower_right", id="right")
        }
        self.teleop = {
            "left": MockTeleopConfig(n_motors=6),
            "right": MockTeleopConfig(n_motors=6)       }
        self.cameras = {
            "cam_left_wrist": SimCameraConfig(
            )
        }

    def make_env_processor(self, device, env: RobotEnvInterface | None = None) -> DataProcessorPipeline:
        return DataProcessorPipeline()

    def make_action_processor(self, teleoperators, device) -> DataProcessorPipeline:
        action_pipeline_steps = []
        return DataProcessorPipeline(steps=action_pipeline_steps)