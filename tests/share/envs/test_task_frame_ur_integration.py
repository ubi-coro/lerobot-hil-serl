"""Focused tests for shared TaskFrame <-> UR command integration.

These tests cover the shared core behavior used by task-frame robots:

- serializable ``TaskFrame`` conversion to the UR controller command,
- and the UR robot's ability to accept the shared task-frame spec directly.
"""

from lerobot.robots.ur.tf_controller import AxisMode, TaskFrameCommand
from lerobot.robots.ur.tf_ur import TF_UR
from share.envs.manipulation_primitive.task_frame import ControlMode, ControlSpace, TaskFrame


def test_task_frame_to_task_frame_command_preserves_modes_targets_and_gains():
    """Task-frame conversion should preserve the low-level UR command semantics."""

    frame = TaskFrame(
        space=ControlSpace.TASK,
        origin=[0.1, 0.2, 0.3, 0.0, 0.1, 0.2],
        target=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        control_mode=[
            ControlMode.POS,
            ControlMode.VEL,
            ControlMode.FORCE,
            ControlMode.POS,
            ControlMode.VEL,
            ControlMode.FORCE,
        ],
        kp=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        kd=[6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        min_pose=[-1.0] * 6,
        max_pose=[1.0] * 6,
    )

    command = frame.to_task_frame_command()

    assert isinstance(command, TaskFrameCommand)
    assert command.T_WF == [0.1, 0.2, 0.3, 0.0, 0.1, 0.2]
    assert command.target == [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    assert command.mode == [
        AxisMode.POS,
        AxisMode.PURE_VEL,
        AxisMode.FORCE,
        AxisMode.POS,
        AxisMode.PURE_VEL,
        AxisMode.FORCE,
    ]
    assert command.kp == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert command.kd == [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]


def test_tf_ur_set_task_frame_accepts_shared_task_frame_spec():
    """The UR robot should accept the shared TaskFrame and convert it internally."""

    robot = object.__new__(TF_UR)
    robot.task_frame = TaskFrameCommand.make_default_cmd()
    robot.gripper = None

    frame = TaskFrame(
        space=ControlSpace.TASK,
        target=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        control_mode=[
            ControlMode.POS,
            ControlMode.VEL,
            ControlMode.FORCE,
            ControlMode.POS,
            ControlMode.VEL,
            ControlMode.FORCE,
        ],
        kp=[1.0] * 6,
        kd=[2.0] * 6,
    )

    robot.set_task_frame(frame)

    assert isinstance(robot.task_frame, TaskFrameCommand)
    assert robot.task_frame.mode == [
        AxisMode.POS,
        AxisMode.PURE_VEL,
        AxisMode.FORCE,
        AxisMode.POS,
        AxisMode.PURE_VEL,
        AxisMode.FORCE,
    ]
    assert robot.action_features == {
        "x.pos": float,
        "y.vel": float,
        "z.wrench": float,
        "wx.pos": float,
        "wy.vel": float,
        "wz.wrench": float,
    }
