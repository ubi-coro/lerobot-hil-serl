"""Broad release-readiness smoke coverage for the share env stack.

Redundancy note: parts of this file overlap with newer focused unit tests in
the same directory. The overlapping smoke checks are intentionally skipped and
should be considered candidates for manual deletion later.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
import torch

from lerobot.processor.core import TransitionKey
from lerobot.processor.hil_processor import TELEOP_ACTION_KEY
from lerobot.teleoperators import TeleopEvents
from share.envs.manipulation_primitive.env_manipulation_primitive import ManipulationPrimitive
from share.envs.manipulation_primitive.processor_steps import (
    InterventionActionProcessorStep,
    MatchTeleopToPolicyActionProcessorStep,
    ToJointActionProcessorStep,
)
from share.envs.manipulation_primitive.task_frame import ControlMode, ControlSpace, PolicyMode, TaskFrame
import mock_pipeline_entities as mocks

@dataclass
class _DummyRobot:
    _action_features: dict[str, type] = field(
        default_factory=lambda: {
            "joint_1.pos": float,
            "joint_2.pos": float,
            "joint_3.pos": float,
        }
    )
    _motors_ft: dict[str, type] = field(default_factory=lambda: {"joint_1.pos": float, "joint_2.pos": float, "joint_3.pos": float})
    is_connected: bool = False
    last_action: np.ndarray | dict | None = None

    @property
    def action_features(self) -> dict[str, type]:
        return self._action_features

    def send_action(self, action):
        self.last_action = action

    def get_observation(self):
        return {
            "x.ee_pos": 0.0,
            "y.ee_pos": 0.0,
            "z.ee_pos": 0.0,
            "wx.ee_pos": 0.0,
            "wy.ee_pos": 0.0,
            "wz.ee_pos": 0.0,
        }

    def disconnect(self):
        self.is_connected = False


@dataclass
class _DummyTaskFrameRobot(_DummyRobot):
    last_task_frame: TaskFrame | None = None
    current_frame: TaskFrame = field(default_factory=lambda: TaskFrame(control_mode=[ControlMode.VEL] * 6))

    @property
    def action_features(self) -> dict[str, type]:
        frame = self.last_task_frame or self.current_frame
        return {key: float for key in frame.action_feature_keys().keys()}

    def set_task_frame(self, command: TaskFrame):
        self.last_task_frame = command


def _transition(action, observation=None, info=None, complementary_data=None):
    return {
        TransitionKey.OBSERVATION: observation or {},
        TransitionKey.ACTION: action,
        TransitionKey.REWARD: 0.0,
        TransitionKey.DONE: False,
        TransitionKey.TRUNCATED: False,
        TransitionKey.INFO: info or {},
        TransitionKey.COMPLEMENTARY_DATA: complementary_data or {},
    }


def test_env_reset_returns_obs_info():
    frame = TaskFrame(target=[0.0] * 6, control_mode=[ControlMode.POS] * 6)
    robot = _DummyTaskFrameRobot()
    env = ManipulationPrimitive(task_frame={"arm": frame}, robot_dict={"arm": robot}, cameras={})

    obs, info = env.reset()

    assert isinstance(obs, dict)
    assert info == {TeleopEvents.IS_INTERVENTION: False}
    assert robot.last_task_frame is frame
    assert env.current_step == 0


def test_env_step_after_reset_smoke():
    frame = TaskFrame(
        target=[0.0] * 6,
        control_mode=[ControlMode.VEL, ControlMode.POS, ControlMode.FORCE, ControlMode.POS, ControlMode.POS, ControlMode.VEL],
    )
    robot = _DummyTaskFrameRobot()
    env = ManipulationPrimitive(task_frame={"arm": frame}, robot_dict={"arm": robot}, cameras={})

    env.reset()
    obs, reward, terminated, truncated, info = env.step(
        np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
    )

    assert isinstance(obs, dict)
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info == {TeleopEvents.IS_INTERVENTION: False}
    assert robot.last_action == {
        "x.vel": pytest.approx(0.1),
        "y.pos": pytest.approx(0.2),
        "z.wrench": pytest.approx(0.3),
        "wx.pos": pytest.approx(0.4),
        "wy.pos": pytest.approx(0.5),
        "wz.vel": pytest.approx(0.6),
    }


def test_task_frame_serialization_deserialization_compatibility_for_target_limits():
    frame = TaskFrame(
        target=[0.0] * 6,
        control_mode=[ControlMode.POS] * 6,
        min_pose=[-1.0] * 6,
        max_pose=[1.0] * 6,
    )

    raw = frame.to_dict()
    decoded = TaskFrame.from_dict(raw)
    assert decoded.min_target == [-1.0] * 6
    assert decoded.max_target == [1.0] * 6

    legacy_raw = {
        "space": int(ControlSpace.TASK),
        "origin": [0.0] * 6,
        "target": [0.0] * 6,
        "policy_mode": [int(PolicyMode.ABSOLUTE)] * 6,
        "control_mode": [int(ControlMode.POS)] * 6,
        "min_pose": [-2.0] * 6,
        "max_pose": [2.0] * 6,
    }
    decoded_legacy = TaskFrame.from_dict(legacy_raw)
    assert decoded_legacy.min_target == [-2.0] * 6
    assert decoded_legacy.max_target == [2.0] * 6


@pytest.mark.parametrize(
    "teleop,space,policy_mode,requires_kinematics",
    [
        (mocks.MockDeltaTeleoperator(), ControlSpace.TASK, PolicyMode.RELATIVE, False),
        (mocks.MockAbsoluteJointTeleoperator(), ControlSpace.TASK, PolicyMode.ABSOLUTE, True),
        (mocks.MockDeltaTeleoperator(), ControlSpace.JOINT, PolicyMode.ABSOLUTE, True),
    ],
)
@pytest.mark.skip(reason="Redundant with focused MatchTeleopToPolicyActionProcessorStep tests; manual deletion candidate.")
def test_compatibility_matrix_pipeline_branching(teleop, space, policy_mode, requires_kinematics):
    frame = TaskFrame(
        target=[0.0] * 6,
        space=space,
        policy_mode=[policy_mode, None, None, None, None, None],
        control_mode=[ControlMode.POS] * 6,
    )

    match_step = MatchTeleopToPolicyActionProcessorStep(
        teleoperators={"arm": teleop},
        task_frame={"arm": frame},
        kinematics={"arm": mocks.MockKinematicsSolver()} if requires_kinematics else {},
    )

    teleop_action = {"delta_x": 0.1} if isinstance(teleop, mocks.MockDeltaTeleoperator) else {"joint_1.pos": 1.0, "joint_2.pos": 2.0, "joint_3.pos": 3.0}

    tr = _transition(
        torch.tensor([0.0]),
        complementary_data={TELEOP_ACTION_KEY: {"arm": teleop_action}},
    )
    out = match_step(tr)
    converted = out[TransitionKey.COMPLEMENTARY_DATA][TELEOP_ACTION_KEY]["arm"]
    assert isinstance(converted, torch.Tensor)
    assert converted.numel() == frame.policy_action_dim


@pytest.mark.skip(reason="Redundant end-to-end smoke coverage; superseded by focused intervention/to-joint tests.")
def test_end_to_end_pipeline_smoke_with_single_step_call():
    frame = TaskFrame(
        target=[0.0] * 6,
        policy_mode=[PolicyMode.RELATIVE, None, None, None, None, None],
        control_mode=[ControlMode.POS] * 6,
        min_pose=[-2.0] * 6,
        max_pose=[2.0] * 6,
    )

    env = ManipulationPrimitive(task_frame={"arm": frame}, robot_dict={"arm": _DummyRobot()}, cameras={})
    env.reset()

    match = MatchTeleopToPolicyActionProcessorStep(
        teleoperators={"arm": mocks.MockDeltaTeleoperator()},
        task_frame={"arm": frame},
    )
    intervention = InterventionActionProcessorStep(task_frame={"arm": frame})
    to_joint = ToJointActionProcessorStep(
        is_task_frame_robot={"arm": False},
        task_frame={"arm": frame},
        kinematics={"arm": mocks.MockKinematicsSolver()},
        joint_names={"arm": ["joint_1", "joint_2", "joint_3"]},
    )

    tr = _transition(
        torch.tensor([0.0]),
        observation={
            "arm.x.ee_pos": 0.2,
            "arm.y.ee_pos": 0.0,
            "arm.z.ee_pos": 0.0,
            "arm.wx.ee_pos": 0.0,
            "arm.wy.ee_pos": 0.0,
            "arm.wz.ee_pos": 0.0,
        },
        info={TeleopEvents.IS_INTERVENTION: True},
        complementary_data={TELEOP_ACTION_KEY: {"arm": {"delta_x": 0.1}}},
    )

    processed = to_joint(intervention(match(tr)))
    low_level_action = np.array([processed[TransitionKey.ACTION][f"joint_{i}.pos"] for i in [1, 2, 3]], dtype=np.float32)

    obs, reward, terminated, truncated, info = env.step(low_level_action)

    assert isinstance(obs, dict)
    assert reward == 0.0
    assert terminated is False and truncated is False
    assert info == {TeleopEvents.IS_INTERVENTION: False}
