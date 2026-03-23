"""Focused tests for intervention-side action projection.

Each test covers one part of the teleop/policy merge logic used in the action
pipeline.
"""

import math

import pytest
import torch

from lerobot.processor.core import TransitionKey
from lerobot.processor.hil_processor import TELEOP_ACTION_KEY
from lerobot.teleoperators import TeleopEvents
from share.envs.manipulation_primitive.processor_steps import (
    InterventionActionProcessorStep,
    _rotation_from_extrinsic_xyz,
)
from share.envs.manipulation_primitive.task_frame import ControlMode, PolicyMode, TaskFrame


def _base_transition(action, info: dict | None = None, complementary_data: dict | None = None):
    return {
        TransitionKey.OBSERVATION: {},
        TransitionKey.ACTION: action,
        TransitionKey.REWARD: 0.0,
        TransitionKey.DONE: False,
        TransitionKey.TRUNCATED: False,
        TransitionKey.INFO: info or {},
        TransitionKey.COMPLEMENTARY_DATA: complementary_data or {},
    }


def test_intervention_action_processor_projects_and_merges_task_frame_targets():
    """Projection: learnable axes should be projected while static axes keep configured targets."""
    frame = TaskFrame(
        target=[1.0, 2.0, 3.0, 0.1, 0.2, 0.3],
        policy_mode=[PolicyMode.ABSOLUTE, None, PolicyMode.RELATIVE, PolicyMode.ABSOLUTE, None, None],
        control_mode=[ControlMode.VEL, ControlMode.POS, ControlMode.POS, ControlMode.POS, ControlMode.POS, ControlMode.POS],
        min_pose=[-0.5] * 6,
        max_pose=[0.5] * 6,
    )
    step = InterventionActionProcessorStep(task_frame={"arm": frame})

    transition = _base_transition({"arm": {"x.vel": torch.tensor(2.0), "z.pos": torch.tensor(-0.5), "wx.pos.cos": torch.tensor(0.0), "wx.pos.sin": torch.tensor(1.0)}})
    out = step(transition)

    action = out[TransitionKey.ACTION]["arm"]
    assert action["x.vel"] == pytest.approx(math.tanh(2.0) * 0.5, abs=1e-6)
    assert action["y.pos"] == pytest.approx(2.0, abs=1e-6)
    assert action["z.pos"] == pytest.approx(-0.5, abs=1e-6)
    assert action["wx.pos"] == pytest.approx(math.pi / 2, abs=1e-6)
    assert action["wy.pos"] == pytest.approx(0.2, abs=1e-6)
    assert action["wz.pos"] == pytest.approx(0.3, abs=1e-6)


def test_intervention_action_processor_prefers_teleop_during_intervention_and_marks_completion():
    """Teleop override: intervention actions should win and then emit a completion marker on release."""
    frame = TaskFrame(
        target=[0.0] * 6,
        policy_mode=[PolicyMode.ABSOLUTE, None, None, None, None, None],
        control_mode=[ControlMode.POS] * 6,
    )
    step = InterventionActionProcessorStep(task_frame={"arm": frame})

    first = step(
        _base_transition(
            {"arm": {"x.pos": torch.tensor(0.1)}},
            info={TeleopEvents.IS_INTERVENTION: True},
            complementary_data={TELEOP_ACTION_KEY: {"arm": {"x.pos": 0.4}}},
        )
    )
    assert first[TransitionKey.ACTION]["arm"]["x.pos"] == pytest.approx(0.4, abs=1e-6)

    second = step(
        _base_transition(
            {"arm": {"x.pos": torch.tensor(0.2)}},
            info={TeleopEvents.IS_INTERVENTION: False},
            complementary_data={},
        )
    )
    assert second[TransitionKey.INFO][TeleopEvents.INTERVENTION_COMPLETED] is True


def test_intervention_action_processor_decodes_so3_6d_representation():
    """Rotation manifold decoding: SO(3) 6D actions should decode back to Euler task-frame targets."""
    expected_euler = [0.2, -0.3, 0.4]
    matrix = _rotation_from_extrinsic_xyz(*expected_euler).as_matrix()
    encoded = torch.tensor(
        [matrix[0][0], matrix[1][0], matrix[2][0], matrix[0][1], matrix[1][1], matrix[2][1]],
        dtype=torch.float32,
    )

    frame = TaskFrame(
        target=[0.0] * 6,
        policy_mode=[None, None, None, PolicyMode.ABSOLUTE, PolicyMode.ABSOLUTE, PolicyMode.ABSOLUTE],
        control_mode=[ControlMode.POS] * 6,
    )
    step = InterventionActionProcessorStep(task_frame={"arm": frame})

    out = step(_base_transition({"arm": {
        "rotation.so3.a1.x": encoded[0],
        "rotation.so3.a1.y": encoded[1],
        "rotation.so3.a1.z": encoded[2],
        "rotation.so3.a2.x": encoded[3],
        "rotation.so3.a2.y": encoded[4],
        "rotation.so3.a2.z": encoded[5],
    }}))
    action = out[TransitionKey.ACTION]["arm"]

    assert action["wx.pos"] == pytest.approx(expected_euler[0], abs=1e-5)
    assert action["wy.pos"] == pytest.approx(expected_euler[1], abs=1e-5)
    assert action["wz.pos"] == pytest.approx(expected_euler[2], abs=1e-5)
