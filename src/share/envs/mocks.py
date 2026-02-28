import numpy as np
import torch
from typing import Any
from pathlib import Path
from lerobot.robots import Robot, RobotConfig
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig
from lerobot.processor import RobotAction, RobotObservation
from share.envs.manipulation_primitive.task_frame import TaskFrame


class MockRobot(Robot):
    config_class = RobotConfig
    name = "mock_robot"

    def __init__(self, name="mock_robot", is_task_frame=True):
        # Initialize with a dummy config
        cfg = RobotConfig(id=name)
        super().__init__(cfg)
        self._is_task_frame = is_task_frame
        self.current_joints = np.zeros(6)
        # Mock bus for configuration validation (looks for motors dict)
        self.bus = type("MockBus", (), {"motors": {f"joint_{i+1}": None for i in range(6)}})
        self.current_frame = TaskFrame()

    @property
    def observation_features(self) -> dict:
        return {f"joint_{i+1}.pos": float for i in range(6)}

    @property
    def action_features(self) -> dict:
        return {f"joint_{i+1}.pos": float for i in range(6)}

    @property
    def is_connected(self) -> bool: return True
    @property
    def is_calibrated(self) -> bool: return True

    def connect(self, calibrate: bool = True): pass
    def disconnect(self): pass
    def calibrate(self): pass
    def configure(self): pass

    def get_observation(self) -> RobotObservation:
        return {f"joint_{i+1}.pos": float(self.current_joints[i]) for i in range(6)}

    def send_action(self, action: RobotAction) -> RobotAction:
        for i in range(6):
            key = f"joint_{i+1}.pos"
            if key in action: self.current_joints[i] = action[key]
        return action

    def set_task_frame(self, frame):
        if not self._is_task_frame:
            raise AttributeError("Hardware does not support task frames.")
        self.current_frame = frame


class MockTeleoperator(Teleoperator):
    config_class = TeleoperatorConfig
    name = "mock_teleop"

    def __init__(self, name="mock_teleop", is_delta=True):
        cfg = TeleoperatorConfig(id=name)
        super().__init__(cfg)
        self._is_delta = is_delta
        if is_delta:
            self._features = {f"delta_{ax}": float for ax in ["x", "y", "z", "rx", "ry", "rz"]}
        else:
            self._features = {f"joint_{i+1}.pos": float for i in range(6)}

    @property
    def action_features(self) -> dict: return self._features
    @property
    def is_connected(self) -> bool: return True
    @property
    def is_calibrated(self) -> bool: return True
    @property
    def feedback_features(self) -> dict: return {}
    def connect(self): pass
    def disconnect(self): pass
    def calibrate(self): pass
    def configure(self): pass
    def send_feedback(self): pass
    def get_action(self) -> RobotAction: return {key: 0.25 for key in self._features}

class MockKinematicsSolver:
    """Mock that mimics RobotKinematics but uses simple identity/vector math."""
    def forward_kinematics(self, joints: np.ndarray | dict) -> np.ndarray:
        if isinstance(joints, dict):
            return np.array([joints.get(f"joint_{i+1}.pos", 0.0) for i in range(6)])
        return joints[:6]

    def inverse_kinematics(self, current_joint_pos: np.ndarray, desired_ee_pose: np.ndarray, **kwargs) -> np.ndarray:
        # In mock, pose == joints.
        # Round-trip check: IK(FK(q)) -> q
        return desired_ee_pose
