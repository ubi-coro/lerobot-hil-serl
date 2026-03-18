import pytest

from share.envs.utils import check_delta_teleoperator, check_task_frame_robot

from tests.share.envs.mock_pipeline_entities import (
    MockAbsoluteJointTeleoperator,
    MockComplexKinematicsSolver,
    MockComplexObservationRobot,
    MockDeltaTeleoperator,
    MockGamepadStyleDeltaTeleoperator,
    MockJointOnlyRobot,
    MockKeyboardStyleDeltaTeleoperator,
    MockKinematicsSolver,
    MockPhoneLikeTeleoperator,
    MockTaskFrameRobot,
    MockVelocityDeltaTeleoperator,
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
        "keyboard": MockKeyboardStyleDeltaTeleoperator(),
        "gamepad": MockGamepadStyleDeltaTeleoperator(),
        "spacemouse": MockVelocityDeltaTeleoperator(),
        "absolute": MockAbsoluteJointTeleoperator(),
        "phone": MockPhoneLikeTeleoperator(),
    }

    result = check_delta_teleoperator(teleop_dict)

    assert result == {
        "delta": True,
        "keyboard": True,
        "gamepad": True,
        "spacemouse": True,
        "absolute": False,
        "phone": False,
    }


def test_mock_kinematics_solver_is_deterministic_for_fk_and_ik():
    solver = MockKinematicsSolver()
    joints = {"joint_1": 0.2, "joint_2": 0.3, "joint_3": -0.1}

    pose = solver.forward_kinematics(joints)
    roundtrip_joints = solver.inverse_kinematics(pose)

    assert pose == pytest.approx([0.4, 0.2, -0.1, 0.04, 0.02, -0.01])
    assert roundtrip_joints == pytest.approx(joints)


def test_complex_mock_robot_observation_matches_complex_fk_mapping():
    robot = MockComplexObservationRobot()
    solver = MockComplexKinematicsSolver()

    obs = robot.get_observation(prefix="arm")
    joints = {
        "joint_1": obs["arm.joint_1.pos"],
        "joint_2": obs["arm.joint_2.pos"],
        "joint_3": obs["arm.joint_3.pos"],
    }

    pose = solver.forward_kinematics(joints)
    assert obs["arm.x.ee_pos"] == pytest.approx(pose[0])
    assert obs["arm.y.ee_pos"] == pytest.approx(pose[1])
    assert obs["arm.z.ee_pos"] == pytest.approx(pose[2])
    assert obs["arm.wx.ee_pos"] == pytest.approx(pose[3])
    assert obs["arm.wy.ee_pos"] == pytest.approx(pose[4])
    assert obs["arm.wz.ee_pos"] == pytest.approx(pose[5])
