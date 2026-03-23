import torch

from lerobot.processor.core import TransitionKey
from lerobot.processor.hil_processor import TELEOP_ACTION_KEY
from share.envs.manipulation_primitive.env_manipulation_primitive import ManipulationPrimitive
from share.envs.manipulation_primitive.processor_steps import (
    InterventionActionProcessorStep,
    ToNestedActionProcessorStep,
)
from lerobot.teleoperators import TeleopEvents
from share.envs.manipulation_primitive.task_frame import ControlMode, PolicyMode, TaskFrame
from share.envs.mocks import MockRobot


def test_to_nested_action_processor_step_splits_policy_tensor_with_gripper_key():
    frame = TaskFrame(
        target=[0.0] * 6,
        policy_mode=[PolicyMode.ABSOLUTE, None, None, None, None, None],
        control_mode=[ControlMode.POS] * 6,
    )
    step = ToNestedActionProcessorStep(task_frame={"arm": frame}, gripper_enable={"arm": True})

    transition = {TransitionKey.ACTION: torch.tensor([0.25, 0.9])}
    out = step(transition)

    assert torch.isclose(out[TransitionKey.ACTION]["arm"]["x.pos"], torch.tensor(0.25))
    assert torch.isclose(out[TransitionKey.ACTION]["arm"]["gripper.pos"], torch.tensor(0.9))


def test_intervention_action_processor_merges_gripper_override_and_keeps_flat_record_tensor():
    frame = TaskFrame(
        target=[0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        policy_mode=[PolicyMode.ABSOLUTE, None, None, None, None, None],
        control_mode=[ControlMode.POS] * 6,
    )
    step = InterventionActionProcessorStep(
        teleoperators={},
        task_frame={"arm": frame},
        gripper_enable={"arm": True},
        gripper_static_pos={"arm": 0.0},
    )

    transition = {
        TransitionKey.ACTION: {"arm": {"x.pos": torch.tensor(0.25), "gripper.pos": torch.tensor(0.1)}},
        TransitionKey.INFO: {TeleopEvents.IS_INTERVENTION: True},
        TransitionKey.COMPLEMENTARY_DATA: {
            TELEOP_ACTION_KEY: {"arm": {"x.pos": 0.25, "gripper.pos": 0.9}}
        },
    }
    out = step(transition)

    assert out[TransitionKey.ACTION]["arm"]["x.pos"] == 0.25
    assert out[TransitionKey.ACTION]["arm"]["gripper.pos"] == 0.9
    assert torch.allclose(out[TransitionKey.COMPLEMENTARY_DATA][TELEOP_ACTION_KEY], torch.tensor([0.25, 0.9]))


def test_manipulation_primitive_step_accepts_per_robot_action_dict():
    robot = MockRobot(name="arm", is_task_frame=True)
    env = ManipulationPrimitive(task_frame={"arm": TaskFrame()}, robot_dict={"arm": robot}, cameras={})

    env.step({"arm": {f"joint_{i + 1}.pos": float(i + 1) for i in range(6)}})

    assert robot.current_joints.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
