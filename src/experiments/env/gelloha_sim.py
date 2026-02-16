from dataclasses import dataclass, field
from typing import Type, Tuple

from lerobot.teleoperators.gello.config_gello import GellohaConfig
from pynput import keyboard

from lerobot.cameras.mujoco import MujocoCameraConfig
from lerobot.envs import RobotEnvConfig
from lerobot.envs.configs import EnvConfig
from lerobot.envs.factory import RobotEnvInterface
from lerobot.processor import DataProcessorPipeline
from lerobot.sim.configs import AlohaSimConfig, SimConfig
from lerobot.sim.mujoco_utils.sim_singleton import SimManager
from lerobot.sim.sim_robot_env import SimRobotEnv
from lerobot.robots.viperx import SimViperXConfig
from lerobot.teleoperators import TeleopEvents
from tests.mocks.mock_teleop import MockTeleopConfig

from lerobot.motors import MotorCalibration, MotorNormMode, Motor


@dataclass
@EnvConfig.register_subclass("gelloha_sim")
class GellohaSimEnvConfig(RobotEnvConfig):
    benchmark: bool = False
    sim: SimConfig = field(default_factory=AlohaSimConfig)  # TODO(jzilke)

    def __post_init__(self):
        self.kinematics_solver = None
        self.robot = {
            "left": SimViperXConfig(id="left"),
            "right": SimViperXConfig(id="right")
        }
        self.teleop = {
            "left": GellohaConfig(
                id="left",
                port="/dev/ttyUSB1",
                motors={
                    "waist": Motor(1, "xl330-m288", MotorNormMode.RADIANS),
                    "shoulder": Motor(2, "xl330-m288", MotorNormMode.RADIANS),
                    "elbow": Motor(3, "xl330-m288", MotorNormMode.RADIANS),
                    "forearm_roll": Motor(4, "xl330-m288", MotorNormMode.RADIANS),
                    "wrist_angle": Motor(5, "xl330-m288", MotorNormMode.RADIANS),
                    "wrist_rotate": Motor(6, "xl330-m288", MotorNormMode.RADIANS),
                    "gripper": Motor(7, "xl330-m077", MotorNormMode.RADIANS),
                },
                default_calibration={
                    "waist": MotorCalibration(id=1, drive_mode=0, homing_offset=1024, range_min=0, range_max=4095),
                    "shoulder": MotorCalibration(id=2, drive_mode=1, homing_offset=1024, range_min=0, range_max=4095),
                    "elbow": MotorCalibration(id=3, drive_mode=1, homing_offset=-3072, range_min=0, range_max=4095),
                    "forearm_roll": MotorCalibration(id=4, drive_mode=0, homing_offset=1024, range_min=0, range_max=4095),
                    "wrist_angle": MotorCalibration(id=5, drive_mode=0, homing_offset=-1024, range_min=0, range_max=4095),
                    "wrist_rotate": MotorCalibration(id=6, drive_mode=0, homing_offset=1024, range_min=0, range_max=4095),
                    "gripper": MotorCalibration(id=7, drive_mode=0, homing_offset=0, range_min=1500, range_max=3072),
                }

            ),
            # "right": GellohaConfig(id="right", port="/dev/ttyUSB0")
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
