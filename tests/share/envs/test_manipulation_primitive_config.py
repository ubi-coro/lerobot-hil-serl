"""Validation tests for ``ManipulationPrimitiveConfig`` teleoperator compatibility rules."""

import pytest

from share.envs.manipulation_primitive.config_manipulation_primitive import ManipulationPrimitiveConfig
from share.envs.manipulation_primitive.task_frame import ControlMode, ControlSpace, PolicyMode, TaskFrame
from tests.share.envs.mock_pipeline_entities import (
    MockAbsoluteJointTeleoperator,
    MockTaskFrameRobot,
    MockVelocityDeltaTeleoperator,
)


def test_validate_allows_velocity_style_delta_teleop_for_adaptive_vel_force():
    """Config validation: SpaceMouse-style EE velocity teleops should satisfy adaptive VEL/FORCE rules."""
    config = ManipulationPrimitiveConfig(
        task_frame=TaskFrame(
            target=[0.0] * 6,
            space=ControlSpace.TASK,
            policy_mode=[PolicyMode.ABSOLUTE, None, None, None, None, None],
            control_mode=[ControlMode.VEL, ControlMode.POS, ControlMode.POS, ControlMode.POS, ControlMode.POS, ControlMode.POS],
        ),
    )

    config.validate(
        robot_dict={"arm": MockTaskFrameRobot()},
        teleop_dict={"arm": MockVelocityDeltaTeleoperator()},
    )


def test_validate_rejects_absolute_joint_teleop_for_adaptive_vel_force():
    """Config validation: absolute-joint leaders should still fail adaptive VEL/FORCE task-frame configs."""
    config = ManipulationPrimitiveConfig(
        task_frame=TaskFrame(
            target=[0.0] * 6,
            space=ControlSpace.TASK,
            policy_mode=[PolicyMode.ABSOLUTE, None, None, None, None, None],
            control_mode=[ControlMode.VEL, ControlMode.POS, ControlMode.POS, ControlMode.POS, ControlMode.POS, ControlMode.POS],
        ),
    )

    with pytest.raises(ValueError, match="require a delta teleoperator"):
        config.validate(
            robot_dict={"arm": MockTaskFrameRobot()},
            teleop_dict={"arm": MockAbsoluteJointTeleoperator()},
        )
