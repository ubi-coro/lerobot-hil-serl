from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from lerobot.envs.configs import EnvConfig
from share.envs.manipulation_primitive.config_manipulation_primitive import ManipulationPrimitiveConfig
from share.envs.manipulation_primitive.task_frame import ControlMode, PolicyMode, TaskFrame
from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import ManipulationPrimitiveNetConfig
from share.envs.manipulation_primitive_net.env_manipulation_primitive_net import ManipulationPrimitiveNet
from share.envs.manipulation_primitive_net.transitions import (
    AlwaysTransition,
    MP_Transition,
    ObservationThresholdTransition,
    TimeLimitTransition,
)
from share.envs.mocks import MockKinematicsSolver, MockRobot, MockTeleoperator


class RandomObservationRobot(MockRobot):
    """Mock robot that emits noisy joint and end-effector observations."""

    def __init__(self, name: str = "random_bot"):
        super().__init__(name=name, is_task_frame=True)
        self.current_joints = np.zeros(6)

    @property
    def _motors_ft(self):
        return {f"joint_{i + 1}.pos": float for i in range(6)}

    def get_observation(self) -> dict[str, Any]:
        obs = {
            f"joint_{i + 1}.pos": float(self.current_joints[i] + np.random.normal(0, 1.0))
            for i in range(6)
        }
        for i, ax in enumerate(["x", "y", "z", "wx", "wy", "wz"]):
            obs[f"{ax}.ee_pos"] = float(self.current_joints[i] + np.random.normal(0, 1.0))
        return obs


def _full_task_frame(target: list[float], policy_mode: list[PolicyMode | None]) -> TaskFrame:
    return TaskFrame(
        target=target,
        policy_mode=policy_mode,
        control_mode=[ControlMode.POS] * 6,
    )


def _build_primitives() -> dict[str, ManipulationPrimitiveConfig]:
    return {
        "search": ManipulationPrimitiveConfig(
            task_frame={
                "random_bot": _full_task_frame(
                    target=[0.4, 0.0, 0.4, 0.0, 0.0, 0.0],
                    policy_mode=[PolicyMode.RELATIVE, PolicyMode.RELATIVE, PolicyMode.RELATIVE, None, None, None],
                )
            }
        ),
        "final_stage": ManipulationPrimitiveConfig(
            task_frame={
                "random_bot": _full_task_frame(
                    target=[0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
                    policy_mode=[None, None, None, None, None, None],
                )
            },
            is_terminal=True,
        ),
        "retract": ManipulationPrimitiveConfig(
            task_frame={
                "random_bot": _full_task_frame(
                    target=[0.4, 0.0, 0.6, 0.0, 0.0, 0.0],
                    policy_mode=[None, None, None, None, None, None],
                )
            }
        ),
        "home": ManipulationPrimitiveConfig(
            task_frame={
                "random_bot": _full_task_frame(
                    target=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    policy_mode=[None, None, None, None, None, None],
                )
            }
        ),
    }


def _build_transitions():
    return [
        (
            "search",
            "final_stage",
            ObservationThresholdTransition(obs_key="random_bot.x.ee_pos", threshold=0.45, operator="ge"),
        ),
        ("final_stage", "retract", AlwaysTransition()),
        ("retract", "home", TimeLimitTransition(max_steps=5)),
        ("home", "search", TimeLimitTransition(max_steps=5)),
    ]


@dataclass
@EnvConfig.register_subclass("demo_mp_net")
class DemoManipulationPrimitiveNetConfig(ManipulationPrimitiveNetConfig):
    start_primitive: str = "search"
    reset_primitive: str = "final_stage"
    primitives: dict[str, ManipulationPrimitiveConfig] = field(default_factory=_build_primitives)
    transitions: list[tuple[str, str, MP_Transition]] = field(default_factory=_build_transitions)


class DemoManipulationPrimitiveNet(ManipulationPrimitiveNet):
    """MP-Net runtime that wires mock robot, teleop, and mock kinematics."""

    def connect(self):
        robot = RandomObservationRobot("random_bot")
        solver = MockKinematicsSolver()
        for primitive in self.config.primitives.values():
            primitive._kinematics_solver = {"random_bot": solver}
        return {"random_bot": robot}, {"random_bot": MockTeleoperator("random_bot", is_delta=True)}, {}


def run_demo(steps: int = 100):
    net = DemoManipulationPrimitiveNet(DemoManipulationPrimitiveNetConfig())
    _, info = net.reset()
    print(f"--- ROLLOUT START: {info['active_primitive']} ---")

    for i in range(steps):
        action = torch.randn(3)
        _, _, terminated, _, info = net.step(action)
        active = info["active_primitive"]
        if terminated:
            print(f"Step {i}: [{active}] Terminal condition met!")
            break
        print(f"Step {i}: Running [{active}]...")

    print("\n--- TRIGGERING SYSTEM RESET ---")
    _, reset_info = net.reset()
    print(f"Final Destination: {reset_info['active_primitive']}")
    print(f"Reset Path Steps Taken: {reset_info.get('reset_transition_steps', 0)}")
    print("Reset Sequence: final_stage -> retract -> home -> search")


if __name__ == "__main__":
    run_demo()
