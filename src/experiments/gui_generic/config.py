# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Generic GUI Environment Configuration

This config serves as a template for GUI-based robot control.
Hardware parameters (ports, camera IDs, etc.) are injected at runtime
from the GUI's hardware profile selection.

Usage:
    - GUI loads this config
    - Injects hardware parameters from hardware_profiles.toml
    - Calls env.make() with injected parameters
"""

from dataclasses import dataclass
from functools import cached_property
from typing import Any

from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.envs.configs import HilSerlRobotEnvConfig, EnvConfig
from lerobot.envs.robot_env import RobotEnv
from lerobot.robots.viperx import ViperXConfig
from lerobot.share.configs import DatasetRecordConfig
from lerobot.teleoperators.widowx import WidowXConfig


@dataclass
@EnvConfig.register_subclass("gui_aloha_bimanual")
class GuiAlohaBimanualEnvConfig(HilSerlRobotEnvConfig):
    """
    Generic bimanual ALOHA configuration for GUI usage.
    
    Hardware parameters should be set via set_hardware_config() before calling make().
    """
    
    # These will be overridden by GUI
    _robot_left_port: str | None = None
    _robot_right_port: str | None = None
    _teleop_left_port: str | None = None
    _teleop_right_port: str | None = None
    _camera_configs: dict[str, dict[str, Any]] | None = None
    
    def set_hardware_config(
        self,
        robot_left_port: str,
        robot_right_port: str,
        teleop_left_port: str,
        teleop_right_port: str,
        cameras: dict[str, dict[str, Any]],
        robot_id: str = "gui_session",
        teleop_id: str = "gui_session",
        calibration_dir: str | None = None
    ):
        """
        Inject hardware parameters from GUI before calling make().
        
        Args:
            robot_left_port: Serial port for left follower arm
            robot_right_port: Serial port for right follower arm
            teleop_left_port: Serial port for left leader arm
            teleop_right_port: Serial port for right leader arm
            cameras: Dictionary of camera configurations
            robot_id: Robot identifier for calibration
            teleop_id: Teleoperator identifier for calibration
            calibration_dir: Base calibration directory
        """
        self._robot_left_port = robot_left_port
        self._robot_right_port = robot_right_port
        self._teleop_left_port = teleop_left_port
        self._teleop_right_port = teleop_right_port
        self._camera_configs = cameras
        self._robot_id = robot_id
        self._teleop_id = teleop_id
        self._calibration_dir = calibration_dir or "/home/jannick/.cache/huggingface/lerobot/calibration"

    def __post_init__(self):
        # Default configuration - will be overridden by set_hardware_config()
        if self._robot_left_port is None:
            # Set defaults (can be overridden before make())
            self._robot_left_port = "/dev/ttyDXL_follower_left"
            self._robot_right_port = "/dev/ttyDXL_follower_right"
            self._teleop_left_port = "/dev/ttyDXL_leader_left"
            self._teleop_right_port = "/dev/ttyDXL_leader_right"
            self._robot_id = "gui_default"
            self._teleop_id = "gui_default"
            self._calibration_dir = "/home/jannick/.cache/huggingface/lerobot/calibration"
            self._camera_configs = {}
        
        # Build robot configuration
        self.robot = {
            "left": ViperXConfig(
                port=self._robot_left_port,
                id=f"{self._robot_id}_left",
                calibration_dir=f"{self._calibration_dir}/robots/bi_viperx"
            ),
            "right": ViperXConfig(
                port=self._robot_right_port,
                id=f"{self._robot_id}_right",
                calibration_dir=f"{self._calibration_dir}/robots/bi_viperx"
            )
        }
        
        # Build teleop configuration
        self.teleop = {
            "left": WidowXConfig(
                port=self._teleop_left_port,
                id=f"{self._teleop_id}_left",
                calibration_dir=f"{self._calibration_dir}/teleoperators/bi_widowx",
                use_aloha2_gripper_servo=True
            ),
            "right": WidowXConfig(
                port=self._teleop_right_port,
                id=f"{self._teleop_id}_right",
                calibration_dir=f"{self._calibration_dir}/teleoperators/bi_widowx",
                use_aloha2_gripper_servo=True
            )
        }
        
        # Build camera configuration
        self.cameras = {}
        if self._camera_configs:
            for cam_name, cam_cfg in self._camera_configs.items():
                self.cameras[cam_name] = RealSenseCameraConfig(
                    serial_number_or_name=cam_cfg.get("serial_number_or_name"),
                    fps=cam_cfg.get("fps", 30),
                    width=cam_cfg.get("width", 640),
                    height=cam_cfg.get("height", 480),
                )
        
        # Processor configuration - reasonable defaults for GUI usage
        self.processor.hooks.time_action_processor = False
        self.processor.hooks.time_env_processor = False
        self.processor.hooks.log_every = 10
        self.processor.control_time_s = 60  # 60 seconds per episode
        self.processor.gripper.use_gripper = True
        self.processor.reset.terminate_on_success = False  # GUI users manually stop
        self.processor.reset.teleop_on_reset = True
        self.processor.reset.reset_time_s = 5.0
        
        # Event mappings - can be customized per session
        # self.processor.events.foot_switch_mapping = {...}
        # self.processor.events.key_mapping = {...}

    @cached_property
    def action_dim(self):
        # 6 DOF per arm + 1 gripper per arm = 14 total
        return 14


@dataclass
@DatasetRecordConfig.register_subclass("gui_session")
class GuiSessionDatasetConfig(DatasetRecordConfig):
    """
    Generic dataset configuration for GUI recording sessions.
    Should be overridden with specific values from GUI.
    """
    repo_id: str = "local/gui_recording_session"
    single_task: str = "GUI Recording Session"
    root: str = "/tmp/lerobot_gui_recordings"  # Default, should be overridden
    num_episodes: int = 10
    fps: int = 30


@dataclass
@EnvConfig.register_subclass("gui_aloha_single_left")
class GuiAlohaSingleLeftEnvConfig(HilSerlRobotEnvConfig):
    """Single left arm configuration for GUI."""
    
    _robot_port: str | None = None
    _teleop_port: str | None = None
    _camera_configs: dict[str, dict[str, Any]] | None = None
    
    def set_hardware_config(
        self,
        robot_port: str,
        teleop_port: str,
        cameras: dict[str, dict[str, Any]],
        robot_id: str = "gui_left",
        teleop_id: str = "gui_left",
        calibration_dir: str | None = None
    ):
        self._robot_port = robot_port
        self._teleop_port = teleop_port
        self._camera_configs = cameras
        self._robot_id = robot_id
        self._teleop_id = teleop_id
        self._calibration_dir = calibration_dir or "/home/jannick/.cache/huggingface/lerobot/calibration"
    
    def __post_init__(self):
        if self._robot_port is None:
            self._robot_port = "/dev/ttyDXL_follower_left"
            self._teleop_port = "/dev/ttyDXL_leader_left"
            self._robot_id = "gui_left"
            self._teleop_id = "gui_left"
            self._calibration_dir = "/home/jannick/.cache/huggingface/lerobot/calibration"
            self._camera_configs = {}
        
        self.robot = ViperXConfig(
            port=self._robot_port,
            id=self._robot_id,
            calibration_dir=f"{self._calibration_dir}/robots/viperx"
        )
        
        self.teleop = WidowXConfig(
            port=self._teleop_port,
            id=self._teleop_id,
            calibration_dir=f"{self._calibration_dir}/teleoperators/widowx",
            use_aloha2_gripper_servo=True
        )
        
        self.cameras = {}
        if self._camera_configs:
            for cam_name, cam_cfg in self._camera_configs.items():
                self.cameras[cam_name] = RealSenseCameraConfig(
                    serial_number_or_name=cam_cfg.get("serial_number_or_name"),
                    fps=cam_cfg.get("fps", 30),
                    width=cam_cfg.get("width", 640),
                    height=cam_cfg.get("height", 480),
                )
        
        # Processor defaults
        self.processor.gripper.use_gripper = True
        self.processor.reset.teleop_on_reset = True
        self.processor.reset.reset_time_s = 5.0
        self.processor.control_time_s = 60

    @cached_property
    def action_dim(self):
        return 7  # 6 DOF + 1 gripper


@dataclass
@EnvConfig.register_subclass("gui_aloha_single_right")
class GuiAlohaSingleRightEnvConfig(GuiAlohaSingleLeftEnvConfig):
    """Single right arm configuration for GUI."""
    
    def __post_init__(self):
        if self._robot_port is None:
            self._robot_port = "/dev/ttyDXL_follower_right"
            self._teleop_port = "/dev/ttyDXL_leader_right"
            self._robot_id = "gui_right"
            self._teleop_id = "gui_right"
            self._calibration_dir = "/home/jannick/.cache/huggingface/lerobot/calibration"
            self._camera_configs = {}
        
        # Call parent to build config with right arm defaults
        super().__post_init__()
