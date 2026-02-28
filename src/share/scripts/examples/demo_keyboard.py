import torch
import numpy as np
from share.envs.manipulation_primitive.task_frame import TaskFrame, ControlSpace, PolicyMode, ControlMode
from share.envs.manipulation_primitive.config_manipulation_primitive import ManipulationPrimitiveConfig
from share.envs.mocks import MockRobot, MockTeleoperator, MockKinematicsSolver
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.processor.hil_processor import TELEOP_ACTION_KEY
from lerobot.teleoperators import TeleopEvents

def highlight_pipeline_usages():
    solver = MockKinematicsSolver()

    # ---------------------------------------------------------
    # USAGE 1: Keyboard (Delta) -> Joint-Only Robot (e.g., ViperX)
    # ---------------------------------------------------------
    print("\n--- USAGE 1: Delta Teleop to Joint-Only Robot ---")
    tf_joint_only = TaskFrame(
        target=[0.4, 0.0, 0.2, 0, 0, 0],
        space=ControlSpace.TASK,
        policy_mode=[PolicyMode.RELATIVE] * 3 + [None] * 3,
        control_mode=[ControlMode.POS] * 6
    )

    config1 = ManipulationPrimitiveConfig(task_frame={"viper": tf_joint_only})
    config1._kinematics_solver = {"viper": solver}
    config1._joint_names = {"viper": [f"joint_{i+1}" for i in range(6)]}

    robot_joint = MockRobot("viper", is_task_frame=False)
    # The Keyboard teleoperator is a delta teleoperator
    teleop_delta = MockTeleoperator("viper", is_delta=True)

    _, _, action_proc1 = config1.make({"viper": robot_joint}, {"viper": teleop_delta}, {})

    transition1 = EnvTransition(
        observation={"viper.x.ee_pos": 0.4, "viper.y.ee_pos": 0.0, "viper.z.ee_pos": 0.2},
        action=torch.zeros(3),
        info={TeleopEvents.IS_INTERVENTION: True},
        complementary_data={TELEOP_ACTION_KEY: {"viper": {"delta_x": 0.05, "delta_y": 0, "delta_z": 0}}}
    )

    out1 = action_proc1(transition1)
    # Expected: x.pos becomes 0.45, IK maps that to joint_1.pos = 0.45
    print(f"Joint Commands (Integrated X: 0.4 + 0.05): {out1[TransitionKey.ACTION]}")

    # ---------------------------------------------------------
    # USAGE 2: Absolute Teleop (Leader Arm) -> Task-Frame Robot (e.g., UR5e)
    # ---------------------------------------------------------
    print("\n--- USAGE 2: Absolute Teleop to Task-Frame Robot ---")
    tf_task_native = TaskFrame(
        target=[0.5, 0.0, 0.3, 0, 0, 0],
        space=ControlSpace.TASK,
        policy_mode=[PolicyMode.RELATIVE] * 3 + [None] * 3,
        control_mode=[ControlMode.POS] * 6
    )

    config2 = ManipulationPrimitiveConfig(task_frame={"ur5": tf_task_native})
    config2._kinematics_solver = {"ur5": solver}

    robot_tf = MockRobot("ur5", is_task_frame=True)
    teleop_abs = MockTeleoperator("ur5", is_delta=False)

    _, _, action_proc2 = config2.make({"ur5": robot_tf}, {"ur5": teleop_abs}, {})

    transition2 = EnvTransition(
        observation={},
        action=torch.zeros(3),
        info={TeleopEvents.IS_INTERVENTION: True},
        complementary_data={TELEOP_ACTION_KEY: {"ur5": [0.6, 0.0, 0.3, 0, 0, 0]}}
    )
    # MatchTeleopToPolicy is the 4th step in the action pipeline
    action_proc2.steps[3]._prev_fk_pose["ur5"] = [0.5, 0.0, 0.3, 0, 0, 0]

    out2 = action_proc2(transition2)
    print(f"Task Frame Commands (Differentiated Delta X: 0.1): {out2[TransitionKey.ACTION]['ur5']}")

    # ---------------------------------------------------------
    # USAGE 3: Force/Velocity Control (Adaptive Admittance)
    # ---------------------------------------------------------
    print("\n--- USAGE 3: Force/Velocity Adaptive Mode Validation ---")
    tf_force = TaskFrame(
        target=[0, 0, 0, 0, 0, 0],
        space=ControlSpace.TASK,
        policy_mode=[None, None, PolicyMode.ABSOLUTE, None, None, None],
        control_mode=[ControlMode.POS, ControlMode.POS, ControlMode.FORCE,
                      ControlMode.POS, ControlMode.POS, ControlMode.POS]
    )

    config3 = ManipulationPrimitiveConfig(task_frame={"arm": tf_force})
    try:
        config3.validate({"arm": robot_tf}, {"arm": teleop_abs})
    except ValueError as e:
        print(f"Validation correctly caught error (ENV-101): {e}")


if __name__ == "__main__":
    highlight_pipeline_usages()
