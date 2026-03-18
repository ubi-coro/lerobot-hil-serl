from dataclasses import dataclass
from pathlib import Path

from pynput import keyboard

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.envs.configs import EnvConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.robots.viperx.viperx import ViperXConfig
from lerobot.teleoperators import TeleopEvents
from lerobot.teleoperators.widowx.widowx import WidowXConfig
from share.envs.manipulation_primitive.config_manipulation_primitive import (
    ManipulationPrimitiveConfig,
    ManipulationPrimitiveProcessorConfig,
    GripperConfig,
    EventConfig
)
from share.envs.manipulation_primitive.task_frame import TaskFrame, ControlSpace, PolicyMode, ControlMode
from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import ManipulationPrimitiveNetConfig
from share.envs.manipulation_primitive_net.transitions import OnSuccess


@dataclass
@EnvConfig.register_subclass("aloha_looped_folding")
class AlohaLoopedFoldingEnvConfig(ManipulationPrimitiveNetConfig):
    fps: int = 30
    max_relative_target: float = 0.3  # radians per step
    benchmark: bool = False
    start_primitive = "manual_reset"
    reset_primitive = "manual_reset"

    def __post_init__(self):
        self.robot = {
            "left": ViperXConfig(port="/dev/ttyDXL_follower_left", id="left", max_relative_target=self.max_relative_target),
            "right": ViperXConfig(port="/dev/ttyDXL_follower_right", id="right", max_relative_target=self.max_relative_target),
        }
        self.teleop = {
            "left": WidowXConfig(port="/dev/ttyDXL_leader_left", id="left", max_relative_target=self.max_relative_target),
            "right": WidowXConfig(port="/dev/ttyDXL_leader_right", id="right", max_relative_target=self.max_relative_target)
        }
        self.cameras = {
            "cam_low": OpenCVCameraConfig(
                index_or_path=Path("/dev/CAM_LOW"),
                fps=30,
                width=640,
                height=480,
                fourcc="MJPG"
            ),
            "cam_top": OpenCVCameraConfig(
                index_or_path=Path("/dev/CAM_HIGH"),
                fps=30,
                width=640,
                height=480,
                fourcc="MJPG"
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

        _task_frame = TaskFrame(
            space=ControlSpace.JOINT,
            policy_mode=7 * [PolicyMode.ABSOLUTE],
            control_mode=7 * [ControlMode.POS],
            target=7 * [0.0]
        )
        _processor = ManipulationPrimitiveProcessorConfig(
            gripper=GripperConfig(
                enable=True,
                discretize=False
            ),
            events=EventConfig(
                key_mapping={
                    TeleopEvents.RERECORD_EPISODE: keyboard.Key.left,
                    TeleopEvents.STOP_RECORDING: keyboard.Key.down
                },
                foot_switch_mapping={
                    (TeleopEvents.SUCCESS,): {"device": 3, "toggle": False},
                    (TeleopEvents.IS_INTERVENTION,): {"device": 8, "toggle": True},
                }
            )
        )

        if self.benchmark:
            _processor.hooks.time_env_processor = True
            _processor.hooks.time_action_processor = True

        self.primitives = {
            "manual_reset": ManipulationPrimitiveConfig(
                task_frame=_task_frame,
                processor=_processor,
            ),
            "folding": ManipulationPrimitiveConfig(
                task_frame=_task_frame,
                processor=_processor,
                policy=ACTConfig(pretrained_path="/vol/coro/jstranghoener/lerobot_volume/models/hoodie_folding/base/act-081225-v3/checkpoints/0700000/pretrained_model/"),
                policy_overwrites={"temporal_ensemble_coeff": 0.01, "n_action_steps": 1}
            ),
            "unfolding": ManipulationPrimitiveConfig(
                task_frame=_task_frame,
                processor=_processor,
                policy=ACTConfig(pretrained_path="/vol/coro/jstranghoener/lerobot_volume/models/hoodie_unfolding/base/act-240226/checkpoints/0400000/pretrained_model/"),
                policy_overwrites={"temporal_ensemble_coeff": 0.01, "n_action_steps": 1}
            ),
        }

        self.transitions = [
            OnSuccess(source="manual_reset", target="unfolding"),
            OnSuccess(source="folding", target="unfolding"),
            OnSuccess(source="unfolding", target="folding"),
        ]

        super().__post_init__()