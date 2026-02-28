from __future__ import annotations

import pytest
import torch

from lerobot.processor.core import TransitionKey
from share.envs.manipulation_primitive.processor_steps import (
    JointsToEEObservation,
    RelativeFrameActionProcessor,
    RelativeFrameObservationProcessor,
    RobotActionToPolicyActionProcessorStep,
)
from tests.share.envs.mock_pipeline_entities import MockComplexKinematicsSolver


def _transition(action=None, observation=None):
    return {
        TransitionKey.OBSERVATION: observation or {},
        TransitionKey.ACTION: action,
        TransitionKey.REWARD: 0.0,
        TransitionKey.DONE: False,
        TransitionKey.TRUNCATED: False,
        TransitionKey.INFO: {},
        TransitionKey.COMPLEMENTARY_DATA: {},
    }


def test_joints_to_ee_observation_adds_expected_ee_pose_keys():
    solver = MockComplexKinematicsSolver(joint_names=["joint_1", "joint_2", "joint_3"])
    step = JointsToEEObservation(kinematics={"arm": solver}, motor_names={"arm": ["joint_1", "joint_2", "joint_3"]})

    observation = {
        "arm.joint_1.pos": 0.35,
        "arm.joint_2.pos": -0.25,
        "arm.joint_3.pos": 0.55,
    }

    out = step(_transition(observation=observation))
    obs_out = out[TransitionKey.OBSERVATION]

    assert obs_out["arm.x.ee_pos"] == pytest.approx(0.5 * 0.35 + 0.2 * -0.25 - 0.1 * 0.55)
    assert obs_out["arm.y.ee_pos"] == pytest.approx(-0.3 * 0.35 + 0.4 * -0.25 + 0.2 * 0.55)
    assert obs_out["arm.z.ee_pos"] == pytest.approx(0.35 - 0.25 + 0.55)
    assert obs_out["arm.wx.ee_pos"] == pytest.approx(0.1 * 0.35)
    assert obs_out["arm.wy.ee_pos"] == pytest.approx(-0.05 * -0.25)
    assert obs_out["arm.wz.ee_pos"] == pytest.approx(0.2 * 0.55)


def test_joints_to_ee_observation_raises_on_missing_joint_key():
    step = JointsToEEObservation(
        kinematics={"arm": MockComplexKinematicsSolver()},
        motor_names={"arm": ["joint_1", "joint_2", "joint_3"]},
    )

    with pytest.raises(ValueError, match="Missing joint observation key 'arm.joint_3.pos'"):
        step(_transition(observation={"arm.joint_1.pos": 0.1, "arm.joint_2.pos": 0.2}))


def test_relative_frame_observation_processor_tracks_per_robot_reference():
    step = RelativeFrameObservationProcessor(enable={"arm": True, "other": False})

    first = _transition(
        observation={
            "arm.x.ee_pos": 1.0,
            "arm.y.ee_pos": 2.0,
            "arm.z.ee_pos": 3.0,
            "arm.wx.ee_pos": 0.1,
            "arm.wy.ee_pos": 0.2,
            "arm.wz.ee_pos": 0.3,
            "other.x.ee_pos": 5.0,
            "other.y.ee_pos": 6.0,
            "other.z.ee_pos": 7.0,
            "other.wx.ee_pos": 0.5,
            "other.wy.ee_pos": 0.6,
            "other.wz.ee_pos": 0.7,
        }
    )
    out1 = step(first)[TransitionKey.OBSERVATION]
    assert out1["arm.x.ee_pos"] == pytest.approx(0.0)
    assert out1["arm.wz.ee_pos"] == pytest.approx(0.0)
    assert out1["other.x.ee_pos"] == pytest.approx(5.0)

    second = _transition(
        observation={
            "arm.x.ee_pos": 1.5,
            "arm.y.ee_pos": 1.0,
            "arm.z.ee_pos": 4.0,
            "arm.wx.ee_pos": 0.2,
            "arm.wy.ee_pos": -0.2,
            "arm.wz.ee_pos": 0.4,
        }
    )
    out2 = step(second)[TransitionKey.OBSERVATION]
    assert out2["arm.x.ee_pos"] == pytest.approx(0.5)
    assert out2["arm.y.ee_pos"] == pytest.approx(-1.0)
    assert out2["arm.z.ee_pos"] == pytest.approx(1.0)
    assert out2["arm.wx.ee_pos"] == pytest.approx(0.1)
    assert out2["arm.wy.ee_pos"] == pytest.approx(-0.4)
    assert out2["arm.wz.ee_pos"] == pytest.approx(0.1)


def test_relative_frame_observation_processor_reset_reinitializes_reference():
    step = RelativeFrameObservationProcessor(enable=True)

    step(
        _transition(
            observation={
                "arm.x.ee_pos": 1.0,
                "arm.y.ee_pos": 2.0,
                "arm.z.ee_pos": 3.0,
                "arm.wx.ee_pos": 0.1,
                "arm.wy.ee_pos": 0.2,
                "arm.wz.ee_pos": 0.3,
            }
        )
    )
    step.reset()
    out = step(
        _transition(
            observation={
                "arm.x.ee_pos": -2.0,
                "arm.y.ee_pos": -3.0,
                "arm.z.ee_pos": -4.0,
                "arm.wx.ee_pos": -0.1,
                "arm.wy.ee_pos": -0.2,
                "arm.wz.ee_pos": -0.3,
            }
        )
    )[TransitionKey.OBSERVATION]

    assert out["arm.x.ee_pos"] == pytest.approx(0.0)
    assert out["arm.wz.ee_pos"] == pytest.approx(0.0)


def test_relative_frame_action_processor_transforms_kinematic_axes_only():
    step = RelativeFrameActionProcessor(enable={"arm": True})
    action = {
        "joint_1.pos": 0.1,
        "joint_2.pos": -0.2,
        "gripper.pos": 0.75,
    }
    out = step(_transition(action=action))[TransitionKey.ACTION]
    assert out["joint_1.pos"] == pytest.approx(0.1)
    assert out["joint_2.pos"] == pytest.approx(-0.2)
    assert out["gripper.pos"] == pytest.approx(0.75)


def test_relative_frame_action_processor_is_noop_when_disabled():
    step = RelativeFrameActionProcessor(enable=False)
    action = {"joint_1.pos": 0.2}
    out = step(_transition(action=action))[TransitionKey.ACTION]
    assert out == action


def test_robot_action_to_policy_action_processor_stable_joint_order():
    step = RobotActionToPolicyActionProcessorStep(
        motor_names={"arm": ["joint_1", "joint_2"], "wrist": ["joint_3"]}
    )
    action = {
        "joint_3.pos": 3.0,
        "joint_2.pos": 2.0,
        "joint_1.pos": 1.0,
    }
    out = step(_transition(action=action))[TransitionKey.ACTION]
    assert isinstance(out, torch.Tensor)
    torch.testing.assert_close(out, torch.tensor([1.0, 2.0, 3.0]))


def test_robot_action_to_policy_action_processor_missing_joint_key_error():
    step = RobotActionToPolicyActionProcessorStep(motor_names={"arm": ["joint_1", "joint_2"]})
    with pytest.raises(ValueError, match="missing expected keys"):
        step(_transition(action={"joint_1.pos": 1.0}))


def test_robot_action_to_policy_action_processor_extra_joint_key_error():
    step = RobotActionToPolicyActionProcessorStep(motor_names={"arm": ["joint_1"]})
    with pytest.raises(ValueError, match="unexpected keys"):
        step(_transition(action={"joint_1.pos": 1.0, "joint_2.pos": 2.0}))
