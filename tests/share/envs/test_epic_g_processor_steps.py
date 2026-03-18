"""Focused tests for observation-side processor steps.

This file covers FK observation enrichment, relative EE-frame transforms,
action flattening, and observation-state feature inference.
"""

from __future__ import annotations

import pytest
import torch

from lerobot.processor.core import TransitionKey
from share.envs.manipulation_primitive.processor_steps import (
    JointsToEEObservation,
    RelativeFrameActionProcessor,
    RelativeFrameObservationProcessor,
    RobotActionToPolicyActionProcessorStep,
    VanillaMPObservationProcessorStep,
    _euler_xyz_from_rotation,
    _rotation_from_extrinsic_xyz,
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
    """Observation enrichment: FK should append deterministic EE pose channels."""
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
    """Observation enrichment: missing joint inputs should fail with a clear error."""
    step = JointsToEEObservation(
        kinematics={"arm": MockComplexKinematicsSolver()},
        motor_names={"arm": ["joint_1", "joint_2", "joint_3"]},
    )

    with pytest.raises(ValueError, match="Missing joint observation key 'arm.joint_3.pos'"):
        step(_transition(observation={"arm.joint_1.pos": 0.1, "arm.joint_2.pos": 0.2}))


def test_relative_frame_observation_processor_tracks_per_robot_reference():
    """Relative observation frame: positions subtract and orientations compose on SO(3)."""
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
    expected_orientation = _euler_xyz_from_rotation(
        _rotation_from_extrinsic_xyz(0.2, -0.2, 0.4) *
        _rotation_from_extrinsic_xyz(0.1, 0.2, 0.3).inv()
    )
    assert out2["arm.wx.ee_pos"] == pytest.approx(expected_orientation[0])
    assert out2["arm.wy.ee_pos"] == pytest.approx(expected_orientation[1])
    assert out2["arm.wz.ee_pos"] == pytest.approx(expected_orientation[2])


def test_relative_frame_observation_processor_reset_reinitializes_reference():
    """Relative observation frame: reset should clear the stored per-episode reference pose."""
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
    """Relative action frame: the current implementation is an identity on numeric values."""
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
    """Relative action frame: disabling the step should leave actions untouched."""
    step = RelativeFrameActionProcessor(enable=False)
    action = {"joint_1.pos": 0.2}
    out = step(_transition(action=action))[TransitionKey.ACTION]
    assert out == action


def test_robot_action_to_policy_action_processor_stable_joint_order():
    """Action flattening: per-robot action tensors should flatten in insertion order."""
    step = RobotActionToPolicyActionProcessorStep()
    action = {
        "arm": torch.tensor([1.0, 2.0]),
        "wrist": torch.tensor([3.0]),
    }
    out = step(_transition(action=action))[TransitionKey.ACTION]
    assert isinstance(out, torch.Tensor)
    torch.testing.assert_close(out, torch.tensor([1.0, 2.0, 3.0]))


def test_robot_action_to_policy_action_processor_passes_non_dict_actions_through():
    """Action flattening: non-dict inputs should be returned unchanged."""
    step = RobotActionToPolicyActionProcessorStep()
    action = torch.tensor([1.0, 2.0])
    out = step(_transition(action=action))[TransitionKey.ACTION]
    torch.testing.assert_close(out, action)


def test_vanilla_mp_observation_processor_collects_modalities_and_images():
    """Observation assembly: enabled modalities should populate state and normalize images."""
    step = VanillaMPObservationProcessorStep(
        gripper_enable={"arm": True},
        add_joint_position_to_observation={"arm": True},
        add_joint_velocity_to_observation={"arm": True},
        add_current_to_observation={"arm": True},
        add_ee_pos_to_observation={"arm": True},
        add_ee_velocity_to_observation={"arm": True},
        add_ee_wrench_to_observation={"arm": True},
    )

    observation = {
        "arm.joint_1.pos": 1.0,
        "arm.joint_2.pos": 2.0,
        "arm.joint_1.current": 0.1,
        "arm.joint_2.current": 0.2,
        "arm.x.ee_pos": 0.01,
        "arm.y.ee_pos": 0.02,
        "arm.z.ee_pos": 0.03,
        "arm.wx.ee_pos": 0.04,
        "arm.wy.ee_pos": 0.05,
        "arm.wz.ee_pos": 0.06,
        "arm.x.ee_wrench": 1.0,
        "arm.y.ee_wrench": 2.0,
        "arm.z.ee_wrench": 3.0,
        "arm.wx.ee_wrench": 4.0,
        "arm.wy.ee_wrench": 5.0,
        "arm.wz.ee_wrench": 6.0,
        "arm.gripper.pos": 0.9,
        "observation.images.cam": torch.full((8, 8, 3), 255, dtype=torch.uint8),
    }

    first = step(_transition(observation=observation))[TransitionKey.OBSERVATION]
    assert first["observation.images.cam"].shape == (3, 8, 8)
    assert first["observation.images.cam"].dtype == torch.float32

    state = first["observation.state"]
    # joint pos(2) + joint vel(differentiated -> 2 zeros) + current(2) + ee_pos(6) + ee_vel(diff -> 6 zeros) + ee_wrench(6) + gripper(1)
    assert state.shape == (25,)
    torch.testing.assert_close(state[2:4], torch.zeros(2))
    torch.testing.assert_close(state[12:18], torch.zeros(6))


def test_vanilla_mp_observation_processor_transform_features_counts_enabled_modalities():
    """Feature inference: state width should include fallback velocity channels when direct ones are absent."""
    step = VanillaMPObservationProcessorStep(
        gripper_enable={"arm": True},
        add_joint_position_to_observation={"arm": True},
        add_joint_velocity_to_observation={"arm": True},
        add_current_to_observation={"arm": False},
        add_ee_pos_to_observation={"arm": True},
        add_ee_velocity_to_observation={"arm": True},
        add_ee_wrench_to_observation={"arm": False},
    )

    from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature

    features = {
        PipelineFeatureType.OBSERVATION: {
            "arm.joint_1.pos": PolicyFeature(type=FeatureType.STATE, shape=(1,)),
            "arm.joint_2.pos": PolicyFeature(type=FeatureType.STATE, shape=(1,)),
            "arm.x.ee_pos": PolicyFeature(type=FeatureType.STATE, shape=(1,)),
            "arm.y.ee_pos": PolicyFeature(type=FeatureType.STATE, shape=(1,)),
            "arm.z.ee_pos": PolicyFeature(type=FeatureType.STATE, shape=(1,)),
            "arm.wx.ee_pos": PolicyFeature(type=FeatureType.STATE, shape=(1,)),
            "arm.wy.ee_pos": PolicyFeature(type=FeatureType.STATE, shape=(1,)),
            "arm.wz.ee_pos": PolicyFeature(type=FeatureType.STATE, shape=(1,)),
            "arm.gripper.pos": PolicyFeature(type=FeatureType.STATE, shape=(1,)),
        },
        PipelineFeatureType.ACTION: {},
    }

    out = step.transform_features(features)
    assert out[PipelineFeatureType.OBSERVATION]["observation.state"].shape == (17,)


def test_vanilla_mp_observation_processor_supports_axis_selection_and_frame_stacking():
    """Observation assembly: selected EE axes should be kept and stacked inside the shared vanilla processor."""

    step = VanillaMPObservationProcessorStep(
        add_joint_position_to_observation={"arm": False},
        add_joint_velocity_to_observation={"arm": False},
        add_current_to_observation={"arm": False},
        add_ee_pos_to_observation={"arm": True},
        ee_pos_axes={"arm": ["z"]},
        add_ee_velocity_to_observation={"arm": True},
        ee_velocity_axes={"arm": ["x", "y", "z", "wx", "wy", "wz"]},
        add_ee_wrench_to_observation={"arm": True},
        ee_wrench_axes={"arm": ["x", "y", "z"]},
        stack_frames=2,
    )

    observation = {
        "arm.x.ee_pos": 0.01,
        "arm.y.ee_pos": 0.02,
        "arm.z.ee_pos": 0.03,
        "arm.wx.ee_pos": 0.04,
        "arm.wy.ee_pos": 0.05,
        "arm.wz.ee_pos": 0.06,
        "arm.x.ee_wrench": 1.0,
        "arm.y.ee_wrench": 2.0,
        "arm.z.ee_wrench": 3.0,
    }

    first = step(_transition(observation=observation))[TransitionKey.OBSERVATION]["observation.state"]
    second_obs = dict(observation)
    second_obs["arm.z.ee_pos"] = 0.05
    second = step(_transition(observation=second_obs))[TransitionKey.OBSERVATION]["observation.state"]

    assert first.shape == (20,)
    torch.testing.assert_close(first[:10], first[10:])
    torch.testing.assert_close(second[-10:], torch.tensor([0.05, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]))
