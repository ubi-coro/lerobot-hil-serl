from dataclasses import dataclass, field

from numpy.f2py.crackfortran import verbose

from lerobot.common.robot_devices.cameras.configs import IntelRealSenseCameraConfig
from lerobot.common.robot_devices.motors.configs import URArmConfig
from lerobot.common.robot_devices.robots.configs import URConfig
from lerobot.experiments.han_insertion.base import InsertionPrimitive, HAN_Insertion
from lerobot.scripts.server.mp_nets import MPNetConfig, MPConfig


@MPNetConfig.register_subclass("han_insertion_record_forces")
@dataclass
class HAN_Insertion_Record_Forces(HAN_Insertion):
    """
    Same as HAN_Insertion, but robot's out ring buffer is larger for evaluation purposes.
    This variant should be used with scripts/server/eval/record_forces.py
    """
    root: str = "/home/jannick/data/paper/hil-amp/force_profile"

    robot: URConfig = URConfig(
        follower_arms={
            "main": URArmConfig(
                robot_ip="172.22.22.2",
                frequency=500,
                payload_mass=1.080,
                payload_cog=[-0.000, 0.000, 0.071],
                soft_real_time=True,
                rt_core=3,
                get_max_k=500,  # up from 10
                use_gripper=False,
                speed_limits=[15.0, 15.0, 15.0, 0.40, 0.40, 1.0],
                wrench_limits=[30.0, 30.0, 15.0, 15.0, 15.0, 10.0],
                enable_contact_aware_force_scaling=[True, True, True, False, False, True],
                contact_desired_wrench=[4.0, 4.0, 2.0, 0, 0, 0.5],
                contact_limit_scale_min=[0.09, 0.09, 0.11, 0, 0, 0.04],
                debug=True,
                debug_axis=2,
                mock=False,
                verbose=True
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


@MPNetConfig.register_subclass("han_insertion_static_limits")
@dataclass
class HAN_Insertion_Static_Limits(HAN_Insertion):
    """

    """
    root: str = "/home/jannick/data/paper/hil-amp/eval_static_limits"

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
                wrench_limits=[4.0, 4.0, 30.0, 15.0, 15.0, 0.5],
                enable_contact_aware_force_scaling=[False, False, False, False, False, False],
                contact_desired_wrench=[0.0, 0.0, 0, 0, 0, 0.0],
                contact_limit_scale_min=[0.0, 0.0, 0, 0, 0, 0.0],
                debug=False,
                debug_axis=0,
                mock=False,
                verbose=False
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


@MPNetConfig.register_subclass("han_insertion_adaptive_limits")
@dataclass
class HAN_Insertion_Adaptive_Limits(HAN_Insertion):
    """
    Same as HAN_Insertion, but robot's out ring buffer is larger for evaluation purposes.
    This variant should be used with scripts/server/eval/record_forces.py
    """
    root: str = "/home/jannick/data/paper/hil-amp/eval_adaptive_limits"


@MPNetConfig.register_subclass("han_insertion_random_policy")
@dataclass
class HAN_Insertion_Random_Policy(HAN_Insertion):
    """
    Same as HAN_Insertion, but robot's out ring buffer is larger for evaluation purposes.
    This variant should be used with scripts/server/eval/record_forces.py
    """
    root: str = "/home/jannick/data/paper/hil-amp/eval_random_policy"
    num_episodes: int = 50

