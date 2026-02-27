import torch

from lerobot.processor.core import TransitionKey
from lerobot.processor.hil_processor import TELEOP_ACTION_KEY
from share.envs.manipulation_primitive.processor_steps import MatchTeleopToPolicyActionProcessorStep
from share.envs.manipulation_primitive.task_frame import ControlMode, ControlSpace, PolicyMode, TaskFrame
from tests.share.envs.mock_pipeline_entities import (
    MockAbsoluteJointTeleoperator,
    MockDeltaTeleoperator,
    MockKinematicsSolver,
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


def test_delta_teleop_absolute_pos_integration_respects_virtual_reference_flag():
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


def test_absolute_joint_teleop_uses_fk_and_relative_modes():
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
