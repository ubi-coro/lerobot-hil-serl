import torch
import numpy as np
import time
from typing import Any

from share.envs.mocks import MockRobot, MockTeleoperator, MockKinematicsSolver
from share.envs.manipulation_primitive.task_frame import TaskFrame, ControlSpace, PolicyMode, ControlMode
from share.envs.manipulation_primitive.config_manipulation_primitive import ManipulationPrimitiveConfig
from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import ManipulationPrimitiveNetConfig
from share.envs.manipulation_primitive_net.env_manipulation_primitive_net import ManipulationPrimitiveNet
from share.envs.manipulation_primitive_net.transitions import (
    ObservationThresholdTransition,
    TimeLimitTransition,
    TransitionOutcome, AlwaysTransition
)

class RandomObservationRobot(MockRobot):
    """
    Subclass of MockRobot that provides stochastic data for transitions.
    Note: We provide '_motors_ft' as a property mapping to ensure compatibility
    with the ManipulationPrimitive initialization logic.
    """
    def __init__(self, name="random_bot"):
        super().__init__(name=name, is_task_frame=True)
        self.current_joints = np.zeros(6)

    @property
    def _motors_ft(self):
        # Maps internal LeRobot feature expectations to this mock
        return {f"joint_{i+1}.pos": float for i in range(6)}

    def get_observation(self) -> dict[str, Any]:
        # Return random poses centered around current state
        obs = {f"joint_{i+1}.pos": float(self.current_joints[i] + np.random.normal(0, 1.0)) for i in range(6)}
        # Simulate End-Effector Cartesian positions for transition thresholding
        for i, ax in enumerate(["x", "y", "z", "wx", "wy", "wz"]):
            obs[f"{ax}.ee_pos"] = float(self.current_joints[i] + np.random.normal(0, 1.0))
        return obs

# Initialize shared mock infrastructure
solver = MockKinematicsSolver()
robot = RandomObservationRobot("mock_robot")
teleop = MockTeleoperator("mock_robot", is_delta=True)


# 1. Search (Start State)
search_cfg = ManipulationPrimitiveConfig(
    task_frame={"random_bot": TaskFrame(
        target=[0.4, 0.0, 0.4, 0, 0, 0],
        policy_mode=[PolicyMode.RELATIVE] * 3 + [None] * 3,
        control_mode=[ControlMode.POS] * 6
    )}
)

# 2. Final Stage (Terminal)
final_success_cfg = ManipulationPrimitiveConfig(
    task_frame={"random_bot": TaskFrame(target=[0.5, 0.5, 0.5, 0, 0, 0])},
    is_terminal=True
)

# 3. Reset Step 1: Retract
retract_cfg = ManipulationPrimitiveConfig(
    task_frame={"random_bot": TaskFrame(target=[0.4, 0.0, 0.6, 0, 0, 0])},
)

# 4. Reset Step 2: Home
home_cfg = ManipulationPrimitiveConfig(
    task_frame={"random_bot": TaskFrame(target=[0.0, 0.0, 0.0, 0, 0, 0])},
)

# Define the Net with Diverse Transitions
net_config = ManipulationPrimitiveNetConfig(
    start_primitive="search",
    reset_primitive="final_stage",
    primitives={
        "search": search_cfg,
        "final_stage": final_success_cfg,
        "retract": retract_cfg,
        "home": home_cfg,
    },
    transitions=[
        # Trigger: If robot 'finds' target (EE-X > 0.45), go to final success
        ("search", "final_stage", ObservationThresholdTransition(
            obs_key="random_bot.x.ee_pos", threshold=0.45, operator="ge"
        )),

        # Reset Path Logic:
        # Final stage links to Retract upon completion
        ("final_stage", "retract", AlwaysTransition()),

        # Retract runs for 1 step then links to Home
        ("retract", "home", TimeLimitTransition(max_steps=5)),

        # Home links back to Start (Search)
        ("home", "search", TimeLimitTransition(max_steps=5)),
    ],
)


# Demo wrapper to inject mocks
class DemoNet(ManipulationPrimitiveNet):
    def connect(self):
        bot = RandomObservationRobot("random_bot")
        from share.envs.mocks import MockTeleoperator
        return {"random_bot": bot}, {"random_bot": MockTeleoperator("random_bot")}, {}


def run_demo():
    net = DemoNet(net_config)
    obs, info = net.reset()

    print(f"--- ROLLOUT START: {info['active_primitive']} ---")

    # Run until we hit the terminal stage
    for i in range(100):
        # Dummy action tensor matching the search primitive's 3 adaptive dimensions
        action = torch.randn(3)

        obs, reward, terminated, truncated, info = net.step(action)

        active = info['active_primitive']
        if terminated:
            print(f"Step {i}: [{active}] Terminal condition met!")
            break
        print(f"Step {i}: Running [{active}]...")

    # The "Two Processor Step Reset" happens here
    print("\n--- TRIGGERING SYSTEM RESET ---")
    new_obs, reset_info = net.reset()

    print(f"Final Destination: {reset_info['active_primitive']}")
    print(f"Reset Path Steps Taken: {reset_info.get('reset_transition_steps', 0)}")
    print(f"Reset Sequence: final_stage -> retract -> home -> search")


if __name__ == "__main__":
    run_demo()
