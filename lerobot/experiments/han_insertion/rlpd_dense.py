from dataclasses import dataclass, field

from lerobot.common.robot_devices.cameras.configs import IntelRealSenseCameraConfig
from lerobot.common.robot_devices.motors.configs import URArmConfig
from lerobot.common.robot_devices.motors.rtde_tff_controller import AxisMode
from lerobot.common.robot_devices.robots.configs import URConfig
from lerobot.experiments.han_insertion.base import InsertionPrimitive, HAN_Insertion
from lerobot.scripts.server.mp_nets import MPNetConfig, MPConfig


@MPNetConfig.register_subclass("han_insertion_rlpd_dense")
@dataclass
class HAN_Insertion_RLPD_Dense(HAN_Insertion):
    root: str = "/home/jannick/data/paper/hil-amp/rlpd_reward_dense_cam_toWindow_terminate_early_init_large_demos_itv_0"

    primitives: dict[str, MPConfig] = field(default_factory=lambda: {
        "press": MPConfig(
            transitions={"insert": "start_insertion"},
            tff=InsertionPrimitive().tff
        ),
        "insert": InsertionPrimitive(
            sparse_reward=False,
            transitions={"terminal": "terminal_false"}  # automatically after timeout
        ),
        "terminal": MPConfig(is_terminal=True),
    })


@MPNetConfig.register_subclass("han_insertion_rlpd_dense_no_priors")
@dataclass
class HAN_Insertion_RLPD_Dense_NoPriors(HAN_Insertion):
    # learns z velocity as well, start immediately after reset, as opposed to after contact

    root: str = "/home/jannick/data/paper/hil-amp/rlpd_reward_dense_cam_toWindow_terminate_early_init_large_no_priors_1"

    start_primitive: str = "insert"
    primitives: dict[str, MPConfig] = field(default_factory=lambda: {
        "insert": InsertionPrimitive(
            sparse_reward=False,
            transitions={"terminal": "terminal_false"}  # automatically after timeout
        ),
        "terminal": MPConfig(is_terminal=True),
    })

    robot: URConfig = URConfig(
        follower_arms={
            "main": URArmConfig(
                robot_ip="172.22.22.2",
                frequency=500,
                payload_mass=1.080,
                payload_cog=[-0.000, 0.000, 0.071],
                soft_real_time=True,
                rt_core=3,
                get_max_k=10,
                use_gripper=False,
                speed_limits=[15.0, 15.0, 15.0, 0.40, 0.40, 1.0],
                wrench_limits=[30.0, 30.0, 30.0, 15.0, 15.0, 10.0],
                enable_contact_aware_force_scaling=[True, True, True, False, False, True],
                contact_desired_wrench=[4.0, 4.0, 5.0, 0, 0, 0.5],
                contact_limit_scale_min=[0.09, 0.09, 0.12, 0, 0, 0.04],
                debug=False,
                debug_axis=0,
                mock=False
            )
        },
        cameras={
            "main": IntelRealSenseCameraConfig(
                serial_number=218622271373,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    def __post_init__(self):

        # controller
        self.primitives["insert"].tff["main"].target = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.primitives["insert"].tff["main"].mode = 3 * [AxisMode.PURE_VEL] + 2 * [AxisMode.POS] + [AxisMode.PURE_VEL]

        # policy
        self.primitives["insert"].policy.indices["main"] = [1, 1, 1, 0, 0, 1]
        self.primitives["insert"].policy.config.dataset_stats["action"] = {
            "min": [-0.02, -0.02, -0.02, -0.75],
            "max": [0.02, 0.02, 0.02, 0.75]
        }

        # interface
        self.primitives["insert"].wrapper.spacemouse_action_scale["main"] = [0.02, -0.02, -0.02, 0, 0, -0.75]

        super().__post_init__()
        self.primitives["insert"].__post_init__()


@MPNetConfig.register_subclass("han_insertion_rlpd_dense_dgn")
@dataclass
class HAN_Insertion_RLPD_Dense_DGN(HAN_Insertion):
    root: str = "/home/jannick/data/paper/hil-amp/rlpd_reward_dense_cam_toWindow_terminate_early_init_large_demos_itv_dgn_0"

    primitives: dict[str, MPConfig] = field(default_factory=lambda: {
        "press": MPConfig(
            transitions={"insert": "start_insertion"},
            tff=InsertionPrimitive().tff
        ),
        "insert": InsertionPrimitive(
            sparse_reward=False,
            transitions={"terminal": "terminal_false"}  # automatically after timeout
        ),
        "terminal": MPConfig(is_terminal=True),
    })

    def __post_init__(self):
        self.primitives["insert"].policy.config.noise_config.enable = True
        super().__post_init__()
