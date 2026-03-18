"""Focused tests for teleop-to-policy action normalization.

Each test documents one mapping rule in ``MatchTeleopToPolicyActionProcessorStep``.
"""

import torch

from lerobot.processor.core import TransitionKey
from lerobot.processor.hil_processor import TELEOP_ACTION_KEY
from share.envs.manipulation_primitive.processor_steps import MatchTeleopToPolicyActionProcessorStep
from share.envs.manipulation_primitive.task_frame import ControlMode, ControlSpace, PolicyMode, TaskFrame
from tests.share.envs.mock_pipeline_entities import (
    MockAbsoluteJointTeleoperator,
    MockComplexKinematicsSolver,
    MockComplexObservationRobot,
    MockDeltaTeleoperator,
    MockKinematicsSolver,
    MockVelocityDeltaTeleoperator,
)


def _transition_with_teleop_action(robot_name: str, action: dict[str, float]):
    return {
        TransitionKey.OBSERVATION: {},
        TransitionKey.ACTION: torch.zeros(1),
        TransitionKey.REWARD: 0.0,
        TransitionKey.DONE: False,
        TransitionKey.TRUNCATED: False,
        TransitionKey.INFO: {},
        TransitionKey.COMPLEMENTARY_DATA: {TELEOP_ACTION_KEY: {robot_name: action}},
    }


def test_delta_teleop_maps_differential_targets_directly():
    """Delta teleop mapping: differential task-frame axes should pass through directly."""
    step = MatchTeleopToPolicyActionProcessorStep(
        teleoperators={"arm": MockDeltaTeleoperator()},
        task_frame={
            "arm": TaskFrame(
                policy_mode=[PolicyMode.ABSOLUTE, PolicyMode.ABSOLUTE, None, None, None, None],
                control_mode=[ControlMode.VEL, ControlMode.FORCE, ControlMode.POS, ControlMode.POS, ControlMode.POS, ControlMode.POS],
                target=[0.0] * 6,
            )
        },
    )

    tr = _transition_with_teleop_action("arm", {"delta_x": 0.4, "delta_y": -0.2})
    out = step(tr)
    converted = out[TransitionKey.COMPLEMENTARY_DATA][TELEOP_ACTION_KEY]["arm"]

    assert torch.allclose(converted, torch.tensor([0.4, -0.2]))


def test_extract_delta_action_accepts_velocity_style_keys():
    """Delta extraction: SpaceMouse-style Cartesian velocity keys should map to the 6D delta vector."""
    deltas = MatchTeleopToPolicyActionProcessorStep._extract_delta_action(
        {
            "x.vel": 0.1,
            "y.vel": -0.2,
            "z.vel": 0.3,
            "wx.vel": -0.4,
            "wy.vel": 0.5,
            "wz.vel": -0.6,
        }
    )

    assert deltas == [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]


def test_extract_delta_action_defaults_missing_axes_to_zero_for_velocity_keys():
    """Delta extraction: missing Cartesian velocity axes should default to zero."""
    deltas = MatchTeleopToPolicyActionProcessorStep._extract_delta_action(
        {
            "x.vel": 0.25,
            "wz.vel": -0.75,
        }
    )

    assert deltas == [0.25, 0.0, 0.0, 0.0, 0.0, -0.75]


def test_delta_teleop_absolute_pos_integration_respects_virtual_reference_flag():
    """Delta integration: absolute POS targets should honor virtual-reference accumulation settings."""
    frame = TaskFrame(
        policy_mode=[PolicyMode.ABSOLUTE, None, None, None, None, None],
        control_mode=[ControlMode.POS] * 6,
        target=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )

    with_virtual = MatchTeleopToPolicyActionProcessorStep(
        teleoperators={"arm": MockDeltaTeleoperator()},
        task_frame={"arm": frame},
        use_virtual_reference=True,
    )
    out1 = with_virtual(_transition_with_teleop_action("arm", {"delta_x": 0.1}))
    out2 = with_virtual(_transition_with_teleop_action("arm", {"delta_x": 0.1}))
    v1 = out1[TransitionKey.COMPLEMENTARY_DATA][TELEOP_ACTION_KEY]["arm"]
    v2 = out2[TransitionKey.COMPLEMENTARY_DATA][TELEOP_ACTION_KEY]["arm"]

    assert torch.allclose(v1, torch.tensor([1.1]))
    assert torch.allclose(v2, torch.tensor([1.2]))

    no_virtual = MatchTeleopToPolicyActionProcessorStep(
        teleoperators={"arm": MockDeltaTeleoperator()},
        task_frame={"arm": frame},
        use_virtual_reference=False,
    )
    out3 = no_virtual(_transition_with_teleop_action("arm", {"delta_x": 0.1}))
    out4 = no_virtual(_transition_with_teleop_action("arm", {"delta_x": 0.1}))
    nv1 = out3[TransitionKey.COMPLEMENTARY_DATA][TELEOP_ACTION_KEY]["arm"]
    nv2 = out4[TransitionKey.COMPLEMENTARY_DATA][TELEOP_ACTION_KEY]["arm"]

    assert torch.allclose(nv1, torch.tensor([1.1]))
    assert torch.allclose(nv2, torch.tensor([1.1]))


def test_velocity_style_delta_teleop_maps_differential_targets_directly():
    """Delta teleop mapping: velocity-style EE teleops should match differential task-frame semantics."""
    step = MatchTeleopToPolicyActionProcessorStep(
        teleoperators={"arm": MockVelocityDeltaTeleoperator()},
        task_frame={
            "arm": TaskFrame(
                policy_mode=[PolicyMode.ABSOLUTE, PolicyMode.ABSOLUTE, None, None, None, None],
                control_mode=[ControlMode.VEL, ControlMode.FORCE, ControlMode.POS, ControlMode.POS, ControlMode.POS, ControlMode.POS],
                target=[0.0] * 6,
            )
        },
    )

    tr = _transition_with_teleop_action("arm", {"x.vel": 0.4, "y.vel": -0.2})
    out = step(tr)
    converted = out[TransitionKey.COMPLEMENTARY_DATA][TELEOP_ACTION_KEY]["arm"]

    assert torch.allclose(converted, torch.tensor([0.4, -0.2]))


def test_absolute_joint_teleop_uses_fk_and_relative_modes():
    """Absolute-joint mapping: FK-based task poses should support relative policy channels."""
    step = MatchTeleopToPolicyActionProcessorStep(
        teleoperators={"arm": MockAbsoluteJointTeleoperator()},
        task_frame={
            "arm": TaskFrame(
                policy_mode=[PolicyMode.RELATIVE, None, None, None, None, None],
                control_mode=[ControlMode.POS] * 6,
                target=[0.0] * 6,
                space=ControlSpace.TASK,
            )
        },
        kinematics={"arm": MockKinematicsSolver()},
    )

    first = step(
        _transition_with_teleop_action(
            "arm", {"joint_1.pos": 1.0, "joint_2.pos": 2.0, "joint_3.pos": 3.0}
        )
    )
    second = step(
        _transition_with_teleop_action(
            "arm", {"joint_1.pos": 1.5, "joint_2.pos": 2.0, "joint_3.pos": 3.0}
        )
    )

    first_val = first[TransitionKey.COMPLEMENTARY_DATA][TELEOP_ACTION_KEY]["arm"]
    second_val = second[TransitionKey.COMPLEMENTARY_DATA][TELEOP_ACTION_KEY]["arm"]

    assert torch.allclose(first_val, torch.tensor([0.0]))
    assert torch.allclose(second_val, torch.tensor([0.5]))


def test_match_step_uses_complex_fk_for_relative_kinematic_channels():
    """Absolute-joint mapping: richer FK models should propagate the correct relative Cartesian deltas."""
    robot = MockComplexObservationRobot()
    obs = robot.get_observation(prefix="arm")

    step = MatchTeleopToPolicyActionProcessorStep(
        teleoperators={"arm": MockAbsoluteJointTeleoperator()},
        task_frame={
            "arm": TaskFrame(
                policy_mode=[PolicyMode.RELATIVE, PolicyMode.RELATIVE, None, None, None, None],
                control_mode=[ControlMode.POS] * 6,
                target=[0.0] * 6,
                space=ControlSpace.TASK,
            )
        },
        kinematics={"arm": MockComplexKinematicsSolver()},
    )

    joint_action_1 = {"joint_1.pos": obs["arm.joint_1.pos"], "joint_2.pos": obs["arm.joint_2.pos"], "joint_3.pos": obs["arm.joint_3.pos"]}
    out1 = step(_transition_with_teleop_action("arm", joint_action_1))
    val1 = out1[TransitionKey.COMPLEMENTARY_DATA][TELEOP_ACTION_KEY]["arm"]
    assert torch.allclose(val1, torch.tensor([0.0, 0.0]))

    joint_action_2 = {
        "joint_1.pos": joint_action_1["joint_1.pos"] + 0.1,
        "joint_2.pos": joint_action_1["joint_2.pos"] - 0.05,
        "joint_3.pos": joint_action_1["joint_3.pos"] + 0.02,
    }
    out2 = step(_transition_with_teleop_action("arm", joint_action_2))
    val2 = out2[TransitionKey.COMPLEMENTARY_DATA][TELEOP_ACTION_KEY]["arm"]

    # Expected deltas under MockComplexKinematicsSolver.forward_kinematics.
    expected_dx = 0.5 * 0.1 + 0.2 * (-0.05) - 0.1 * 0.02
    expected_dy = -0.3 * 0.1 + 0.4 * (-0.05) + 0.2 * 0.02
    assert torch.allclose(val2, torch.tensor([expected_dx, expected_dy], dtype=torch.float32), atol=1e-6)
