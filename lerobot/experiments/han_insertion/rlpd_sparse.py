from dataclasses import dataclass, field

import numpy as np

from lerobot.common.robot_devices.motors.rtde_tff_controller import AxisMode
from lerobot.experiments.han_insertion.base import InsertionPrimitive, HAN_Insertion
from lerobot.scripts.server.mp_nets import MPNetConfig, MPConfig


@MPNetConfig.register_subclass("han_insertion_rlpd_sparse")
@dataclass
class HAN_Insertion_RLPD_Sparse(HAN_Insertion):
    root: str = "/home/jannick/data/paper/hil-amp/rlpd_reward_sparse_cam_lab_terminate_early_init_large_demos_itv_3"

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


@MPNetConfig.register_subclass("han_insertion_rlpd_sparse_no_vision")
@dataclass
class HAN_Insertion_RLPD_Sparse_NoVision(HAN_Insertion):
    root: str = "/home/jannick/data/paper/hil-amp/rlpd_reward_sparse_cam_toWindow_terminate_early_init_large_demos_itv_no_vision_1"

    use_vision: bool = False

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


@MPNetConfig.register_subclass("han_insertion_rlpd_sparse_dgn")
@dataclass
class HAN_Insertion_RLPD_Sparse_DGN(HAN_Insertion):
    root: str = "/home/jannick/data/paper/hil-amp/rlpd_reward_sparse_cam_toWindow_terminate_early_init_large_demos_itv_dgn_0"

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

    def __post_init__(self):
        self.primitives["insert"].policy.config.noise_config.enable = True
        super().__post_init__()


@MPNetConfig.register_subclass("han_insertion_rlpd_sparse_rotab")
@dataclass
class HAN_Insertion_RLPD_Sparse_RotAB(HAN_Insertion):
    # learns z velocity as well, start immediately after reset, as opposed to after contact

    ab_offset_max_std_rad: float = 0.07
    root: str = "/home/jannick/data/paper/hil-amp/rlpd_reward_sparse_cam_lab_terminate_early_init_large_demos_itv_rotab_0"

    start_primitive: str = "press"
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

    def __post_init__(self):
        super().__post_init__()

        self.reset.noise_std["main"][3] = self.ab_offset_max_std_rad
        self.reset.noise_std["main"][4] = self.ab_offset_max_std_rad

        # pose limits
        if self.reset.noise_dist == "normal":
            ab_limit = 3 * self.ab_offset_max_std_rad
        else:  # self.wrapper.noise_dist = "uniform"
            ab_limit = np.sqrt(12) / 2 * self.ab_offset_max_std_rad

        # increase reset position height
        self.reset.pos["main"][2] -= np.sin(ab_limit) * 0.02  # 1.5 cm diameter of connector plus safety

        for primitive in self.primitives.values():
            if "main" not in primitive.tff:
                continue
            primitive.tff["main"].max_pose_rpy[3] = ab_limit
            primitive.tff["main"].max_pose_rpy[4] = ab_limit
            primitive.tff["main"].min_pose_rpy[3] = -ab_limit
            primitive.tff["main"].min_pose_rpy[4] = -ab_limit

        self.primitives["insert"].__post_init__()


@MPNetConfig.register_subclass("han_insertion_rlpd_sparse_rotab_no_priors")
@dataclass
class HAN_Insertion_RLPD_Sparse_RotAB_NoPriors(HAN_Insertion_RLPD_Sparse_RotAB):
    # learns z velocity as well, start immediately after reset, as opposed to after contact

    root: str = "/home/jannick/data/paper/hil-amp/rlpd_reward_sparse_cam_lab_terminate_early_init_large_demos_itv_rotab_no_priors_1"

    start_primitive: str = "insert"
    primitives: dict[str, MPConfig] = field(default_factory=lambda: {
        "insert": InsertionPrimitive(
            sparse_reward=True,
            transitions={"terminal": "terminal_false"}  # automatically after timeout
        ),
        "terminal": MPConfig(is_terminal=True),
    })

    def __post_init__(self):
        # controller
        self.primitives["insert"].tff["main"].target = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.primitives["insert"].tff["main"].mode = 3 * [AxisMode.PURE_VEL] + 2 * [AxisMode.IMPEDANCE_VEL] + [AxisMode.PURE_VEL]

        # policy
        self.primitives["insert"].policy.indices["main"] = [1, 1, 1, 1, 1, 1]
        self.primitives["insert"].policy.config.dataset_stats["action"] = {
            "min": [-0.02, -0.02, -0.05, -0.1, -0.1, -0.75],
            "max": [0.02, 0.02, 0.05, 0.1, 0.1, 0.75]
        }

        # interface
        self.primitives["insert"].wrapper.control_time_s += 2.0  # account for approach
        self.primitives["insert"].wrapper.spacemouse_action_scale["main"] = [0.02, -0.02, -0.05, 0.1, -0.1, -0.75]

        # enable adaptive compliance for all axes, recalculate parameters
        self.robot.follower_arms["main"].compliance_adaptive_limit_theta = None
        self.robot.follower_arms["main"].compliance_safety_enable = [True, True, False, True, True, True]
        self.robot.follower_arms["main"].__post_init__()

        super().__post_init__()


@MPNetConfig.register_subclass("han_insertion_rlpd_sparse_large_connector")
@dataclass
class HAN_Insertion_RLPD_Sparse_Large_Connector(HAN_Insertion_RLPD_Sparse):
    def __post_init__(self):
        super().__post_init__()
        self.primitives["insert"].reward_axis_targets["main"] = 0.028

