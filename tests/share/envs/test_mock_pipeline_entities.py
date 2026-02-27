import pytest

from share.envs.utils import check_delta_teleoperator, check_task_frame_robot

from tests.share.envs.mock_pipeline_entities import (
    MockAbsoluteJointTeleoperator,
    MockDeltaTeleoperator,
    MockJointOnlyRobot,
    MockKinematicsSolver,
    MockTaskFrameRobot,
)


def test_mock_robots_cover_task_frame_and_joint_only_modalities():
    robot_dict = {
        "task": MockTaskFrameRobot(),
        "joint": MockJointOnlyRobot(),
    }

    result = check_task_frame_robot(robot_dict)

    assert result == {"task": True, "joint": False}


def test_mock_teleoperators_cover_delta_and_absolute_joint_modalities():
    teleop_dict = {
        "delta": MockDeltaTeleoperator(),
        "absolute": MockAbsoluteJointTeleoperator(),
    }

    result = check_delta_teleoperator(teleop_dict)

    assert result == {"delta": True, "absolute": False}


def test_mock_kinematics_solver_is_deterministic_for_fk_and_ik():
    solver = MockKinematicsSolver()
    joints = {"joint_1": 0.2, "joint_2": 0.3, "joint_3": -0.1}

    pose = solver.forward_kinematics(joints)
    roundtrip_joints = solver.inverse_kinematics(pose)

    assert pose == pytest.approx([0.4, 0.2, -0.1, 0.04, 0.02, -0.01])
    assert roundtrip_joints == pytest.approx(joints)
