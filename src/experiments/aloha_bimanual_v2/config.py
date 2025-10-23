from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from pynput import keyboard

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.envs.configs import HilSerlRobotEnvConfig, EnvConfig
from lerobot.envs.robot_env import RobotEnv
from lerobot.processor.migrate_calibration_processor import MigrateCalibrationObsProcessorStep, MigrateInterventionActionProcessorStep
from lerobot.robots.viperx import ViperXConfig
from lerobot.share.configs import DatasetRecordConfig
from lerobot.teleoperators import TeleopEvents
from lerobot.teleoperators.widowx import WidowXConfig


@dataclass
@EnvConfig.register_subclass("aloha_bimanual_v2")
class AlohaBimanualEnvConfigV2(HilSerlRobotEnvConfig):

    def __post_init__(self):
        self.robot = {
            "left": ViperXConfig(port="/dev/ttyDXL_follower_left", id="left"),
            "right": ViperXConfig(port="/dev/ttyDXL_follower_right", id="right")
        }
        self.teleop = {
            "left": WidowXConfig(port="/dev/ttyDXL_leader_left", id="left"),
            "right": WidowXConfig(port="/dev/ttyDXL_leader_right", id="right")
        }
        self.cameras = {
            "cam_low": OpenCVCameraConfig(
                index_or_path=Path("/dev/CAM_LOW"),
                fps=30,
                width=640,
                height=480,
            ),
            "cam_top": OpenCVCameraConfig(
                index_or_path=Path("/dev/CAM_HIGH"),
                fps=30,
                width=640,
                height=480,
            ),
            "cam_right_wrist": RealSenseCameraConfig(
                serial_number_or_name="218622272856",
                fps=30,
                width=640,
                height=480,
            ),
            "cam_left_wrist": RealSenseCameraConfig(
                serial_number_or_name="218722270675",
                fps=30,
                width=640,
                height=480,
            )
        }

        self.processor.hooks.time_action_processor = False
        self.processor.hooks.time_env_processor = False
        self.processor.hooks.log_every = 1
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

    @cached_property
    def action_dim(self):
        return super().action_dim + 2 * len(self.robot)  # shadows

    # def obs processors
    def _processors(self, env: RobotEnv, teleoperators, device):
        env_processor, action_processor = super()._processors(env, teleoperators, device)

        env_processor.steps.append(
            MigrateCalibrationObsProcessorStep(num_robots=len(self.robot))
        )

        # redo step hooks
        if self.processor.hooks.time_env_processor:
            from lerobot.utils.control_utils import make_step_timing_hooks
            before_step_hooks, after_step_hooks = make_step_timing_hooks(
                pipeline_steps=env_processor.steps,  # your DataProcessorPipeline instance
                label="env",
                log_every=self.processor.hooks.log_every,  # emit every 10 passes; set to 1 for every pass
                ema_alpha=0.2,
                also_print=False,
            )
            env_processor.before_step_hooks = before_step_hooks
            env_processor.after_step_hooks = after_step_hooks

        action_processor.steps.pop(-1)
        action_processor.steps.append(
            MigrateInterventionActionProcessorStep(
                teleoperators=teleoperators,
                use_gripper=self.processor.gripper.use_gripper,
                terminate_on_success=self.processor.reset.terminate_on_success
            )
        )

        return env_processor, action_processor


@dataclass
@DatasetRecordConfig.register_subclass("aloha_bimanual_v2")
class AlohaBimanualDatasetConfigV2(DatasetRecordConfig):
    repo_id: str = "test/aloha_bimanual_v2"
    single_task: str = "test"
    root: str = "/media/nvme1/jstranghoener/lerobot/data/test/aloha_bimanual_v2"

