import pytest
import torch

from lerobot.processor.core import TransitionKey
from lerobot.processor.hil_processor import TELEOP_ACTION_KEY
from lerobot.teleoperators import TeleopEvents
from share.envs.manipulation_primitive.processor_steps import (
    InterventionActionProcessorStep,
    MatchTeleopToPolicyActionProcessorStep,
    ToJointActionProcessorStep,
)
from share.envs.manipulation_primitive.task_frame import ControlMode, PolicyMode, TaskFrame
from tests.share.envs.mock_pipeline_entities import MockDeltaTeleoperator, MockKinematicsSolver


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


def test_to_joint_integrates_relative_task_frame_action_and_clamps_limits():
    frame = TaskFrame(
        target=[0.0] * 6,
        policy_mode=[PolicyMode.RELATIVE, None, None, None, None, None],
        control_mode=[ControlMode.POS] * 6,
        min_pose=[-0.2] * 6,
        max_pose=[0.2] * 6,
    )
    step = ToJointActionProcessorStep(
        is_task_frame_robot={"arm": False},
        task_frame={"arm": frame},
        kinematics={"arm": MockKinematicsSolver()},
        joint_names={"arm": ["joint_1", "joint_2", "joint_3"]},
        use_virtual_reference=True,
    )

    out1 = step(
        _transition(
            {"arm": torch.tensor([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])},
            observation={
                "arm.x.ee_pos": 0.1,
                "arm.y.ee_pos": 0.0,
                "arm.z.ee_pos": 0.0,
                "arm.wx.ee_pos": 0.0,
                "arm.wy.ee_pos": 0.0,
                "arm.wz.ee_pos": 0.0,
            },
        )
    )
    assert out1[TransitionKey.ACTION]["joint_2.pos"] == pytest.approx(0.2)

    out2 = step(_transition({"arm": torch.tensor([-0.1, 0.0, 0.0, 0.0, 0.0, 0.0])}))
    assert out2[TransitionKey.ACTION]["joint_2.pos"] == pytest.approx(0.1)


def test_processor_chain_teleop_to_task_frame_to_joint_action():
    frame = TaskFrame(
        target=[0.0] * 6,
        policy_mode=[PolicyMode.RELATIVE, None, None, None, None, None],
        control_mode=[ControlMode.POS] * 6,
        min_pose=[-2.0] * 6,
        max_pose=[2.0] * 6,
    )

    match = MatchTeleopToPolicyActionProcessorStep(
        teleoperators={"arm": MockDeltaTeleoperator()},
        task_frame={"arm": frame},
    )
    intervention = InterventionActionProcessorStep(task_frame={"arm": frame})
    to_joint = ToJointActionProcessorStep(
        is_task_frame_robot={"arm": False},
        task_frame={"arm": frame},
        kinematics={"arm": MockKinematicsSolver()},
        joint_names={"arm": ["joint_1", "joint_2", "joint_3"]},
        use_virtual_reference=False,
    )

    tr = _transition(
        torch.tensor([0.0]),
        observation={
            "arm.x.ee_pos": 0.5,
            "arm.y.ee_pos": 0.0,
            "arm.z.ee_pos": 0.0,
            "arm.wx.ee_pos": 0.0,
            "arm.wy.ee_pos": 0.0,
            "arm.wz.ee_pos": 0.0,
        },
        info={TeleopEvents.IS_INTERVENTION: True},
        complementary_data={TELEOP_ACTION_KEY: {"arm": {"delta_x": 0.1}}},
    )

    out = to_joint(intervention(match(tr)))

    assert out[TransitionKey.ACTION]["joint_1.pos"] == pytest.approx(0.0)
    assert out[TransitionKey.ACTION]["joint_2.pos"] == pytest.approx(0.6)
    assert out[TransitionKey.ACTION]["joint_3.pos"] == pytest.approx(0.0)
