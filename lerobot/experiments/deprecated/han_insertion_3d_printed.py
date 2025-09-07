import copy
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Callable

import draccus
import numpy as np

from lerobot.common.constants import ACTION, OBS_ROBOT, OBS_IMAGE
from lerobot.common.envs.ur_env import UREnv
from lerobot.common.envs.wrapper.hilserl import (
    ConvertToLeRobotObservation,
    ImageCropResizeWrapper,
    TorchActionWrapper,
    BatchCompatibleWrapper,
    TimeLimitWrapper
)
from lerobot.common.envs.wrapper.reward import AxisDistanceRewardWrapper
from lerobot.common.envs.wrapper.spacemouse import SpaceMouseInterventionWrapper
from lerobot.common.envs.wrapper.tff import StaticTaskFrameActionWrapper
from lerobot.common.policies.reward_model.configuration_classifier import RewardClassifierConfig
from lerobot.common.policies.sac.configuration_sac import SACConfig, DataGuidedNoiseConfig
from lerobot.common.robot_devices.cameras.configs import IntelRealSenseCameraConfig
from lerobot.common.robot_devices.motors.configs import URArmConfig
from lerobot.common.robot_devices.motors.rtde_tff_controller import TaskFrameCommand, AxisMode
from lerobot.common.robot_devices.robots.configs import URConfig
from lerobot.common.robot_devices.robots.ur import UR
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.scripts.server.mp_nets import (
    MPNetConfig,
    MPConfig,
    ResetConfig,
    WrapperConfig,
    AMPObsWrapper, PolicyConfig
)


@dataclass
class InsertionPrimitive(MPConfig):
    # MP parameters
    is_terminal: bool = False
    tff: Dict[str, TaskFrameCommand] = field(default_factory=lambda: {
            "main": TaskFrameCommand(
                T_WF=[0.04659, -0.33303, 0.120, 0.0, float(np.pi), 0.0],
                target=[0.0, 0.0, 7.0, 0.0, -0.0, 0.0],
                mode=2 * [AxisMode.PURE_VEL] + [AxisMode.FORCE] + 2 * [AxisMode.POS] + [AxisMode.PURE_VEL],
                kp=[2500, 2500, 2500, 100, 100, 100],
                kd=[960, 960, 320, 6, 6, 6]
            )
        })

    # Reward parameters
    reward_axis_targets: Optional[Dict[str, float]] = field(default_factory=lambda: {
        "main": 0.015  # -> max_depth += 5%: 0.01575
    })
    reward_axis: int = 2  # z
    reward_scale: float = 1.0
    reward_clip: Optional[tuple[float, float]] = None
    reward_terminate_on_success: bool = False

    policy: PolicyConfig = PolicyConfig(
        indices= {"main": [1, 1, 0, 0, 0, 1]},
        config=SACConfig(
            online_steps=10000000,
            training_starts=60,
            online_buffer_capacity=25000,
            offline_buffer_capacity=10000,
            camera_number=1,
            cta_ratio=2,
            storage_device="cuda",
            shared_encoder=True,
            num_critics=3,
            target_entropy=-1.5,
            use_backup_entropy=False,
            freeze_vision_encoder=False,
            noise_config=DataGuidedNoiseConfig(enable=False),
            dataset_stats={
                "observation.image.main": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
                # [f_x, f_y, f_z, a_0, a_1, a_2, (t_0, t_1, t_2,) (p_x, p_y,) p_z]
                # f_xy: 2 * contact_desired_wrench -> [3.0, 3.0, 0, 0, 0, 0.4],
                # f_z: 2 * tff["main].target[2]
                # a: spacemouse_action_scale
                # t_xy: wrench_limits,
                # t_z: 2 * contact_desired_wrench -> [3.0, 3.0, 0, 0, 0, 0.4]
                # p workspace
                "observation.state": {
                    "min": [-6.0, -6.0, -13.0, -0.02, -0.02, -0.75, -2.0, -2.0, -0.8, -0.005],
                    "max": [6.0,  6.0,  3.0,   0.02,   0.02,  0.75,  2.0,  2.0,  0.8,  0.01575]
                },
                "action": {
                    "min": [-0.02, -0.02, -0.75],
                    "max": [0.02, 0.02, 0.75]
                },
            }
        ),
        pretrained_policy_name_or_path=None
    )

    wrapper: WrapperConfig = WrapperConfig(
        control_time_s=7.0,
        crop_params_dict={"observation.image.main": (150, 260, 150, 170)},
        crop_resize_size=(128, 128),
        spacemouse_devices={"main": "SpaceMouse Compact"},
        spacemouse_action_scale={"main": [-0.02, 0.02, 0, 0, 0, -0.75]},
        spacemouse_intercept_with_button=True
    )

    features: dict[str, PolicyFeature] = field(default_factory=lambda: {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(0,)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(0,)),
        "observation.image.main": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128))
    })
    features_map: dict[str, str] = field(default_factory=lambda: {
        "action": ACTION,
        "observation.state": OBS_ROBOT,
        "observation.image.main": f"{OBS_IMAGE}.main",
    })

    def make(self, mp_net: MPNetConfig, robot: Optional[UR] = None):
        if robot is None:
            robot = make_robot_from_config(mp_net.robot)

        env = UREnv(
            robot=robot,
            display_cameras=mp_net.display_cameras
        )

        env = StaticTaskFrameActionWrapper(
            env,
            static_tffs=self.tff,
            action_bounds=self.policy.action_bounds,
            action_indices=self.policy.indices,
            device=mp_net.device
        )

        if self.wrapper.control_time_s is not None:
            env = TimeLimitWrapper(env, fps=mp_net.fps, control_time_s=self.wrapper.control_time_s)

        # SpaceMouse Intervention
        if (
            self.is_adaptive and
            self.wrapper.spacemouse_devices and
            self.wrapper.spacemouse_action_scale
        ):
            env = SpaceMouseInterventionWrapper(
                env,
                devices=self.wrapper.spacemouse_devices,
                action_indices=self.policy.indices,
                action_scale=self.wrapper.spacemouse_action_scale,
                intercept_with_button=self.wrapper.spacemouse_intercept_with_button,
                device=mp_net.device
            )

        if self.reward_axis_targets:
            env = AxisDistanceRewardWrapper(
                env,
                targets=self.reward_axis_targets,
                axis=self.reward_axis,
                scale=self.reward_scale,
                clip=self.reward_clip,
                terminate_on_success=self.reward_terminate_on_success,
                normalization_range=[0.0, self.reward_axis_targets["main"]]
            )

        env = ConvertToLeRobotObservation(env, device=mp_net.device)

        if self.wrapper.crop_params_dict is not None:
            env = ImageCropResizeWrapper(
                env=env,
                crop_params_dict=self.wrapper.crop_params_dict,
                resize_size=self.wrapper.crop_resize_size,
            )

        env = AMPObsWrapper(
            env,
            use_xy_position=getattr(mp_net, "use_xy_position", False),
            use_torque=getattr(mp_net, "use_torque", True),
            device=mp_net.device
        )

        env = BatchCompatibleWrapper(env=env)
        env = TorchActionWrapper(env, device=mp_net.device)

        # set tff manually
        controllers = env.unwrapped.robot.controllers
        for name, tff in self.tff.items():
            controllers[name].send_cmd(tff)

        return env

    def start_insertion(self, obs):
        state = obs["observation.state"].cpu().numpy()
        curr_z = state[0, -1]
        curr_force = abs(state[0, 2])
        max_z = self.reward_axis_targets["main"]
        max_force = self.tff["main"].target[2]
        return (curr_force > max_force) or (curr_z > max_z)


@MPNetConfig.register_subclass("ur3_han_insertion_3d_printed")
@dataclass
class UR3_HAN_Insertion_3d_Printed(MPNetConfig):
    start_primitive: str = "press"
    primitives: dict[str, MPConfig] = field(default_factory=lambda: {
        "press": MPConfig(
            transitions={"insert": "start_insertion"},
            tff=InsertionPrimitive().tff
        ),
        "insert": InsertionPrimitive(
            transitions={"terminal": "terminal_false"}  # automatically after timeout
        ),
        "terminal": MPConfig(is_terminal=True),
    })

    display_cameras: bool = False
    fps: int = 10
    resume: bool = False
    repo_id: str = "jannick-st/ur3-han-insertion-3d-printed-offline-demos"
    dataset_root: str = "/home/jannick/data/jannick-st/ur3-han-insertion-3d-printed/offline-demos"
    task: str = ""
    num_episodes: int = 10
    episode: int = 0
    device: str = "cuda"
    storage_device: str = "cuda"
    push_to_hub: bool = False
    seed: int = 42

    # Insertion parameters
    xy_offset_std_mm: float = 1.5
    c_offset_std_rad: float = 0.3
    use_xy_position: bool = False
    use_torque: bool = True
    use_vision: bool = True

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
                wrench_limits=[30.0, 30.0, 30.0, 15.0, 15.0, 5.0],
                enable_contact_aware_force_scaling=[True, True, False, False, False, True],
                contact_desired_wrench=[3.0, 3.0, 0, 0, 0, 0.4],
                contact_limit_scale_min=[0.09, 0.09, 0, 0, 0, 0.06],
                debug=False,
                debug_axis=0
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

    reset: ResetConfig = ResetConfig(
        pos={
            "main": [0.0] * 6  # origin in task frame
        },
        noise_dist="normal",
        noise_std={
            "main": [0] * 6  # set in __post_init__
        },
        safe_reset=True
    )

    def __post_init__(self):
        # build initial pos ranges and pose limits from reset pose and offsets
        reset_pos = np.array(self.reset.pos["main"])
        max_pose = reset_pos.copy()
        min_pose = reset_pos.copy()

        if self.reset.noise_dist == "normal":
            xy_limit = 3 * self.xy_offset_std_mm / 1000.0
            c_limit = 3 * self.c_offset_std_rad
        else:  # self.wrapper.noise_dist = "uniform"
            xy_limit = np.sqrt(12) / 2000 * self.xy_offset_std_mm
            c_limit = np.sqrt(12) / 2 * self.c_offset_std_rad

        # x, y axes
        max_pose[:2] += xy_limit
        min_pose[:2] -= xy_limit

        # z axis
        z_range = self.primitives["insert"].reward_axis_targets["main"] - reset_pos[2]
        max_pose[2] = self.primitives["insert"].reward_axis_targets["main"] + z_range * 0.05
        min_pose[2] = reset_pos[2] - z_range * 0.05

        # a, b axes
        max_pose[3:5] += 1e-3
        min_pose[3:5] -= 1e-3

        # c axis
        max_pose[5] += c_limit
        min_pose[5] -= c_limit

        # build noise level from pose limits
        self.reset.noise_std["main"][0] = self.xy_offset_std_mm / 1000.0
        self.reset.noise_std["main"][1] = self.xy_offset_std_mm / 1000.0
        self.reset.noise_std["main"][-1] = self.c_offset_std_rad

        # store max poses as list
        for primitive in self.primitives.values():
            if "main" not in primitive.tff:
                continue
            primitive.tff["main"].max_pose_rpy = max_pose.tolist()
            primitive.tff["main"].min_pose_rpy = min_pose.tolist()

        # compute state, action dim for insertion primitive
        # noinspection PyTypeChecker
        p: InsertionPrimitive = self.primitives["insert"]
        action_dim = sum(p.policy.indices["main"])
        p.features["action"].shape = (action_dim,)

        state_dim = 4 + action_dim
        if self.use_xy_position:
            state_dim += 2
        if self.use_torque:
            state_dim += 3
        p.features["observation.state"].shape = (state_dim,)

        # handle vision
        if not self.use_vision:
            self.repo_id += "-no-vision"
            self.dataset_root += "-no-vision"
            self.robot.cameras = {}
            p.wrapper.crop_params_dict = None
            del p.features["observation.image.main"]
            del p.features_map["observation.image.main"]

        if self.robot.follower_arms["main"].verbose:
            print("Min pose:", min_pose)
            print("Max pose:", max_pose)

        super().__post_init__()

    @property
    def condition_registry(self) -> dict[str, Callable | RewardClassifierConfig]:
        return {
            "start_insertion": self.primitives["insert"].start_insertion,
            "terminal_false": lambda obs:False
        }


if __name__ == "__main__":
    cfg = UR3_HAN_Insertion_3d_Printed()
    cfg.robot.follower_arms["main"].verbose = True
    cfg.__post_init__()

    with (open("../temp", "w") as f, draccus.config_type("json"), ):
        draccus.dump(cfg, f, indent=4)

