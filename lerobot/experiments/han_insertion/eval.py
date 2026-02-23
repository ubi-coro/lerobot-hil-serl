from dataclasses import dataclass, field
from typing import Literal

from numpy.f2py.crackfortran import verbose

from lerobot.common.robot_devices.cameras.configs import IntelRealSenseCameraConfig
from lerobot.common.robot_devices.motors.configs import URArmConfig
from lerobot.common.robot_devices.robots.configs import URConfig
from lerobot.experiments import HAN_Insertion_RLPD_Sparse_RotAB_NoPriors
from lerobot.experiments.han_insertion.base import InsertionPrimitive, HAN_Insertion
from lerobot.scripts.server.mp_nets import MPNetConfig, MPConfig


@MPNetConfig.register_subclass("han_insertion_eval_z_force")
@dataclass
class HAN_Insertion_Eval_Z_Force(HAN_Insertion):
    fps: float = 50.0
    x_offset_std_mm: float = 0.0
    y_offset_std_mm: float = 0.0
    c_offset_max_std_rad: float = 0.0
    c_offset_min_std_rad: float = 0.0

    adaptive: bool = True
    force_mode_gain: float = 1.0

    start_primitive: str = "insert"
    primitives: dict[str, MPConfig] = field(default_factory=lambda: {
        "insert": InsertionPrimitive(
            sparse_reward=True,
            transitions={"terminal": "terminal_false"}  # automatically after timeout
        ),
        "terminal": MPConfig(is_terminal=True),
    })

    def __post_init__(self):
        self.robot.follower_arms["main"].compliance_safety_enable = [False] * 6

        super().__post_init__()

        self.robot.follower_arms["main"].wrench_limits[2] = 30.0
        self.primitives["insert"].tff["main"].target[2] = 20.0
        self.primitives["insert"].wrapper.control_time_s = 5.0

        if self.adaptive:
            self.robot.follower_arms["main"].compliance_safety_enable[2] = True
            self.robot.follower_arms["main"].compliance_desired_wrench[2] = 5.0
            self.robot.follower_arms["main"].compliance_adaptive_limit_min[2] = 0.11
            self.robot.follower_arms["main"].force_mode_gain_scaling = self.force_mode_gain

            # recalculate adaptive compliance params
            self.robot.follower_arms["main"].verbose=True
            self.robot.follower_arms["main"].compliance_adaptive_limit_theta = None
            self.robot.follower_arms["main"].__post_init__()


@MPNetConfig.register_subclass("han_insertion_eval_adaptive_limits")
@dataclass
class HAN_Insertion_Eval_Adaptive_Limits(HAN_Insertion):
    """
    Same as HAN_Insertion, but robot's out ring buffer is larger for evaluation purposes.
    This variant should be used with scripts/server/eval/record_forces.py
    """
    primitives: dict[str, MPConfig] = field(default_factory=lambda: {
        "press": MPConfig(
            transitions={"insert": "start_insertion"},
            tff=InsertionPrimitive().tff
        ),
        "insert": InsertionPrimitive(
            sparse_reward=True,
            transitions={"terminal": "terminal_false"}  # automatically after timeout
        ),
        "terminal": MPConfig(is_terminal=True),
    })

    root: str = "/home/jannick/data/paper/hil-amp/eval_adaptive_limits"


@MPNetConfig.register_subclass("han_insertion_static_limits")
@dataclass
class HAN_Insertion_Eval_Static_Limits(HAN_Insertion_Eval_Adaptive_Limits):
    root: str = "/home/jannick/data/paper/hil-amp/eval_static_limits"

    def __post_init__(self):
        super().__post_init__()
        self.robot.follower_arms["main"].compliance_safety_enable=[False] * 6
        self.robot.follower_arms["main"].wrench_limits = [4.0, 4.0, 30.0, 15.0, 15.0, 0.5]


@MPNetConfig.register_subclass("han_insertion_eval_hilserl_limits")
@dataclass
class HAN_Insertion_Eval_HILSERL_Limits(HAN_Insertion_Eval_Adaptive_Limits):
    """
    Same as HAN_Insertion, but robot's out ring buffer is larger for evaluation purposes.
    This variant should be used with scripts/server/eval/record_forces.py
    """
    root: str = "/home/jannick/data/paper/hil-amp/eval_hilserl_limits"

    def __post_init__(self):
        super().__post_init__()
        self.robot.follower_arms["main"].compliance_safety_mode = "reference_limits"
        self.reset.timeout = 15.0  # resets are slow too


@MPNetConfig.register_subclass("han_insertion_eval_random_policy")
@dataclass
class HAN_Insertion_Eval_Random_Policy(HAN_Insertion_Eval_Adaptive_Limits):
    """
    Same as HAN_Insertion, but robot's out ring buffer is larger for evaluation purposes.
    This variant should be used with scripts/server/eval/record_forces.py
    """
    root: str = "/home/jannick/data/paper/hil-amp/eval_random_policy"
    num_episodes: int = 50


@MPNetConfig.register_subclass("han_insertion_eval_6d")
@dataclass
class HAN_Insertion_Eval_6D(HAN_Insertion_RLPD_Sparse_RotAB_NoPriors):
    root = "/home/jannick/data/paper/hil-amp/eval_6d"

    compliance_safety_mode: Literal["adaptive_wrench_limits", "reference_limits"] = "reference_limits"

    def __post_init__(self):
        super().__post_init__()

        if self.compliance_safety_mode == "reference_limits":
            self.root += "_hilserl_limits"
        elif self.compliance_safety_mode == "adaptive_wrench_limits":
            self.root += "_adaptive_limits"

        # enable adaptive compliance for all axes, recalculate parameters
        self.robot.follower_arms["main"].verbose=True
        #self.robot.follower_arms["main"].wrench_limits = [30.0, 30.0, 30.0, 10.0, 10.0, 10.0]

        self.robot.follower_arms["main"].compliance_adaptive_limit_theta = None
        self.robot.follower_arms["main"].compliance_safety_mode = self.compliance_safety_mode
        self.robot.follower_arms["main"].compliance_safety_enable = [True] * 6
        self.robot.follower_arms["main"].compliance_desired_wrench = [4.0, 4.0, 5.0, 0.5, 0.5, 0.5]
        self.robot.follower_arms["main"].compliance_adaptive_limit_min = [0.09, 0.09, 0.11, 0.04, 0.04, 0.04]
        self.robot.follower_arms["main"].__post_init__()

        self.primitives["insert"].wrapper.spacemouse_action_scale["main"] = [0.02, -0.02, -0.05, 0.1, -0.1, -0.75]

