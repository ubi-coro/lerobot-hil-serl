from dataclasses import dataclass

from share_rl.primitives.config import (
    AxisMode,
    ControlSpace,
    Origin,
    PrimitiveGraphConfig,
    PrimitiveGraphNodeConfig,
    RobotPrimitiveConfig,
    TaskFrameCommand,
)
from share_rl.primitives.runtime import PrimitiveGraphEnv


@dataclass
class DummyPrimitive:
    transition: str | None = None

    def reset(self) -> dict:
        return {"obs": 0}

    def step(self, action: dict) -> tuple[dict, float, bool, dict]:
        info = {}
        if self.transition:
            info["transition"] = self.transition
        return {"obs": action.get("obs", 1)}, 1.0, False, info


def _command() -> TaskFrameCommand:
    return TaskFrameCommand(
        space=ControlSpace.JOINT,
        origin=Origin.ABSOLUTE,
        domain_context=None,
        target=[0.0] * 6,
        policy_indices=[False] * 6,
        mode=[AxisMode.STIFF_POS] * 6,
    )


def test_graph_env_adds_primitive_id_and_transitions() -> None:
    cfg = PrimitiveGraphConfig(
        start_primitive_id="reach",
        nodes=[
            PrimitiveGraphNodeConfig(
                primitive_id="reach",
                primitive_type="reach",
                robot_primitives={"arm": RobotPrimitiveConfig(robot_id="arm", task_frame=_command())},
                transitions={"grasped": "lift"},
            ),
            PrimitiveGraphNodeConfig(
                primitive_id="lift",
                primitive_type="lift",
                robot_primitives={"arm": RobotPrimitiveConfig(robot_id="arm", task_frame=_command())},
            ),
        ],
    )

    env = PrimitiveGraphEnv(
        config=cfg,
        primitives={"reach": DummyPrimitive(transition="grasped"), "lift": DummyPrimitive()},
    )

    reset_obs = env.reset()
    assert reset_obs["primitive_id"] == "reach"

    step_obs, _, _, info = env.step({})
    assert step_obs["primitive_id"] == "lift"
    assert info["active_primitive_id"] == "lift"
