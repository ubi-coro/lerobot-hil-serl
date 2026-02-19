import pytest

from share_rl.primitives.config import AxisMode, ControlSpace, Origin, TaskFrameCommand


def test_task_frame_requires_transform_matrix_in_task_space() -> None:
    with pytest.raises(ValueError, match="domain_context"):
        TaskFrameCommand(
            space=ControlSpace.TASK,
            origin=Origin.ABSOLUTE,
            domain_context=None,
            target=[0.0] * 6,
            policy_indices=[True] * 6,
            mode=[AxisMode.STIFF_POS] * 6,
        )


def test_joint_space_disallows_non_position_modes() -> None:
    with pytest.raises(ValueError, match="space == JOINT"):
        TaskFrameCommand(
            space=ControlSpace.JOINT,
            origin=Origin.ABSOLUTE,
            domain_context=None,
            target=[0.0] * 6,
            policy_indices=[True] * 6,
            mode=[AxisMode.STIFF_POS] * 5 + [AxisMode.STIFF_VEL],
        )


def test_serialization_round_trip() -> None:
    command = TaskFrameCommand(
        space=ControlSpace.TASK,
        origin=Origin.ABSOLUTE,
        domain_context=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        target=[0.0] * 6,
        policy_indices=[False, True, False, True, False, True],
        mode=[AxisMode.STIFF_POS] * 6,
    )

    restored = TaskFrameCommand.from_dict(command.to_dict())

    assert restored.space == ControlSpace.TASK
    assert restored.learnable_axis_indices == [1, 3, 5]
    assert restored.is_adaptive is True
