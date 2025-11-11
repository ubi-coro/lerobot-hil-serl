from dataclasses import dataclass
from functools import cached_property

from pynput import keyboard

from experiments import AlohaBimanualEnvConfigV2
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.envs.configs import EnvConfig
from lerobot.envs.robot_env import RobotEnv
from lerobot.processor.migrate_calibration_processor import MigrateCalibrationObsProcessorStep, MigrateInterventionActionProcessorStep
from lerobot.robots.viperx import ViperXConfig
from lerobot.teleoperators import TeleopEvents
from lerobot.teleoperators.widowx import WidowXConfig


@dataclass
@EnvConfig.register_subclass("aloha_bimanual_lemgo_v2")
class AlohaBimanualEnvConfigLemgoV2(AlohaBimanualEnvConfigV2):

    def __post_init__(self):
        self.robot = {
            "left": ViperXConfig(port="/dev/ttyDXL_follower_left", id="left_v3"),
            "right": ViperXConfig(port="/dev/ttyDXL_follower_right", id="right_v3")
        }
        self.teleop = {
            "left": WidowXConfig(port="/dev/ttyDXL_leader_left", id="left_v3", use_aloha2_gripper_servo=True),
            "right": WidowXConfig(port="/dev/ttyDXL_leader_right", id="right_v3", use_aloha2_gripper_servo=True)
        }
        self.cameras = {
            "cam_low": RealSenseCameraConfig(
                serial_number_or_name="130322272007",
                fps=30,
                width=640,
                height=480,
            ),
            "cam_top": RealSenseCameraConfig(
                serial_number_or_name="218722270994",
                fps=30,
                width=640,
                height=480,
            ),
            "cam_right_wrist": RealSenseCameraConfig(
                serial_number_or_name="130322274116",
                fps=30,
                width=640,
                height=480,
            ),
            "cam_left_wrist": RealSenseCameraConfig(
                serial_number_or_name="218622276088",
                fps=30,
                width=640,
                height=480,
            )
        }

        self.processor.hooks.time_action_processor = False
        self.processor.hooks.time_env_processor = False
        self.processor.hooks.log_every = 1
        self.processor.control_time_s = 60
        self.processor.gripper.use_gripper = True
        self.processor.reset.terminate_on_success = True
        self.processor.reset.teleop_on_reset = True
        self.processor.reset.reset_time_s = 5.0
        self.processor.events.foot_switch_mapping = {
            (TeleopEvents.SUCCESS,): {"device": 3, "toggle": False},
            (TeleopEvents.IS_INTERVENTION,): {"device": 6, "toggle": True},
        }
        self.processor.events.key_mapping = {
            TeleopEvents.RERECORD_EPISODE: keyboard.Key.left
        }
