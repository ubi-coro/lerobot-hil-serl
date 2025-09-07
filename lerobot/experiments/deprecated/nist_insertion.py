import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import gymnasium
import numpy as np
import torch
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

from lerobot.common.constants import ACTION, OBS_ROBOT, OBS_IMAGE
from lerobot.common.envs.configs import TaskFrameWrapperConfig, EnvConfig, UREnvConfig
from lerobot.common.envs.ur_env import UREnv
from lerobot.common.envs.wrapper.hilserl import ConvertToLeRobotObservation, TorchActionWrapper, TimeLimitWrapper, \
    ImageCropResizeWrapper, BatchCompatibleWrapper
from lerobot.common.envs.wrapper.reward import AxisDistanceRewardWrapper
from lerobot.common.envs.wrapper.spacemouse import SpaceMouseInterventionWrapper
from lerobot.common.envs.wrapper.tff import StaticTaskFrameResetWrapper, StaticTaskFrameActionWrapper
from lerobot.common.policies.mlp.configuration_mlp import MLPConfig
from lerobot.common.policies.sac.configuration_sac import SACConfig
from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
from lerobot.common.robot_devices.motors.configs import URArmConfig
from lerobot.common.robot_devices.motors.rtde_tff_controller import TaskFrameCommand, AxisMode
from lerobot.common.robot_devices.robots.configs import URConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature


# -----------------
# Environments
# -----------------
@EnvConfig.register_subclass("ur3_nist_insertion_xyc_small")
@dataclass
class UR3_NIST_Insertion_XYC_Small(UREnvConfig):
    num_episodes: int = 20
    repo_id: str = "jannick-st/ur3-nist-insertion-xyc-small-offline-demos"
    dataset_root: str = "/home/jannick/data/jannick-st/ur3-nist-insertion/offline-demos-xyc-small"
    task: str = "Push in the peg"
    resume: bool = False
    fps: int = 10
    display_cameras: bool = False
    push_to_hub: bool = False

    explore: bool = False
    xy_offset_std_mm: float = 1.0
    c_offset_std_rad: float = 0.1
    depth_to_tighten_c_bounds: Optional[float] = 0.008
    use_xy_position: bool = False
    use_torque: bool = True
    use_vision: bool = True

    robot: URConfig = URConfig(
        follower_arms={
            "main": URArmConfig(
                robot_ip="172.22.22.2",
                soft_real_time=True,
                use_gripper=False,
                speed_limits=[0.40, 0.40, 0.40, 4.0, 4.0, 4.0],
                payload_mass=0.925,
                payload_cog=[0.0, 0.0, 0.058]
            )
        },
        cameras={
            "main": OpenCVCameraConfig(
                camera_index=0,
                fps=30,
                width=640,
                height=480,
                focus=100
            ),
        }
    )
    wrapper: TaskFrameWrapperConfig = TaskFrameWrapperConfig(
        control_time_s=6.0,
        crop_params_dict={
            "observation.image.main": (150, 260, 150, 170)
        },
        crop_resize_size=(128, 128),
        static_tffs={
            "main": TaskFrameCommand(
                T_WF=[0.1800, -0.3977, 0.11, 2.221, -2.221, 0.0],
                target=[0.0, 0.0, 4.0, 0.0, -0.0, 0.0],
                mode=2 * [AxisMode.IMPEDANCE_VEL] + [AxisMode.FORCE] + 3 * [AxisMode.IMPEDANCE_VEL],
                kp=[3000, 3000, 3000, 100, 100, 100],
                kd=[150, 150, 300, 6, 6, 6]
            )
        },
        action_indices={
            "main": [1, 1, 0, 0, 0, 1]
        },
        reset_pos={
            "main": [0.0] * 6  # origin in task frame
        },
        noise_dist="normal",
        noise_std={
            "main": [0] * 6  # set in __post_init__
        },
        safe_reset=True,
        threshold=0.0008,
        timeout=3.0,
        spacemouse_devices={
            "main": "SpaceMouse Compact"
        },
        spacemouse_action_scale={
            "main": [0.02, -0.02, 0, 0, 0, -1.0]
        },
        spacemouse_intercept_with_button=True,
        reward_axis_targets={
            "main": 0.013
        },
        reward_axis=2,
        reward_scale=1.0,
        reward_clip=None,
        reward_terminate_on_success=False
    )

    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(100000,)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(100000,)),
            "observation.image.main": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128))
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "observation.state": OBS_ROBOT,
            "observation.image.main": f"{OBS_IMAGE}.main",
        }
    )

    def __post_init__(self):
        self.wrapper.action_bounds = {}
        for name in self.robot.follower_arms:
            # build pose limits from reset pose and xy_offset
            reset_pos = np.array(self.wrapper.reset_pos["main"])
            max_pose = reset_pos.copy()
            min_pose = reset_pos.copy()

            if self.wrapper.noise_dist == "normal":
                xy_limit = 3 * self.xy_offset_std_mm / 1000.0
                c_limit = 3 * self.c_offset_std_rad
            else:  # self.wrapper.noise_dist = "uniform"
                xy_limit = np.sqrt(12) / 2000 * self.xy_offset_std_mm
                c_limit = np.sqrt(12) / 2 * self.c_offset_std_rad

            # x, y axes
            max_pose[:2] += xy_limit
            min_pose[:2] -= xy_limit

            # z axis
            z_range = self.wrapper.reward_axis_targets["main"] - reset_pos[2]
            max_pose[2] = self.wrapper.reward_axis_targets["main"] + z_range * 0.05
            min_pose[2] = reset_pos[2] - z_range * 0.05

            # a, b axes
            max_pose[3:5] += 1e-3
            min_pose[3:5] -= 1e-3

            # c axis
            max_pose[5] += c_limit
            min_pose[5] -= c_limit

            # store as list of floats
            self.wrapper.static_tffs[name].max_pose_rpy = max_pose.tolist()
            self.wrapper.static_tffs[name].min_pose_rpy = min_pose.tolist()

            # build noise level from pose limits
            self.wrapper.noise_std[name][0] = self.xy_offset_std_mm / 1000.0
            self.wrapper.noise_std[name][1] = self.xy_offset_std_mm / 1000.0
            self.wrapper.noise_std[name][-1] = self.c_offset_std_rad

            # build action space bounds from space mouse scaling factors
            action_scale = self.wrapper.spacemouse_action_scale[name]
            action_indices = self.wrapper.action_indices[name]
            self.wrapper.action_bounds[name] = {
                "min": [-abs(s) for i, s in enumerate(action_scale) if action_indices[i]],
                "max": [abs(s) for i, s in enumerate(action_scale) if action_indices[i]]
            }

        # compute state, action dim
        action_dim = sum(self.wrapper.action_indices["main"])
        self.features["action"].shape = (action_dim,)

        state_dim = 4 + action_dim
        if self.use_xy_position:
            state_dim += 2
        if self.use_torque:
            state_dim += 3
        self.features["observation.state"].shape = (state_dim, )

        # handle vision
        if not self.use_vision:
            self.repo_id += "-no-vision"
            self.dataset_root += "-no-vision"
            self.robot.cameras = {}
            self.wrapper.crop_params_dict = None
            del self.features["observation.image.main"]
            del self.features_map["observation.image.main"]


    def make(self):
        env = UREnv(
            robot=make_robot_from_config(self.robot),
            display_cameras=self.display_cameras
        )

        # Static Action
        if self.wrapper.static_tffs and self.wrapper.action_indices:
            env = StaticTaskFrameActionWrapper(
                env,
                static_tffs=self.wrapper.static_tffs,
                action_bounds=self.wrapper.action_bounds,
                action_indices=self.wrapper.action_indices,
                device=self.device
            )

        if self.depth_to_tighten_c_bounds is not None:
            env = TightenXYCBoundsWrapper(
                env,
                static_tffs=self.wrapper.static_tffs,
                threshold=self.depth_to_tighten_c_bounds,
                pos_eps=self.wrapper.static_tffs["main"].max_pose_rpy[0] / 2.0
            )

        # Static Reset
        if self.wrapper.reset_pos:
            env = StaticTaskFrameResetWrapper(
                env,
                static_tffs=self.wrapper.static_tffs or {},
                reset_pos=self.wrapper.reset_pos,
                reset_kp=self.wrapper.reset_kp,
                reset_kd=self.wrapper.reset_kd,
                noise_std=self.wrapper.noise_std,
                noise_dist=self.wrapper.noise_dist,
                safe_reset=self.wrapper.safe_reset,
                threshold=self.wrapper.threshold,
                timeout=self.wrapper.timeout
            )

        env = AwaitForceResetWrapper(
            env,
            threshold=1.1 * self.wrapper.static_tffs["main"].target[2],
            axis=2,
            timeout=self.wrapper.timeout
        )

        env = TimeLimitWrapper(env, fps=self.fps, control_time_s=self.wrapper.control_time_s)

        # SpaceMouse Intervention
        if (
            self.wrapper.spacemouse_devices and
            self.wrapper.action_indices and
            self.wrapper.spacemouse_action_scale
        ):
            env = SpaceMouseInterventionWrapper(
                env,
                devices=self.wrapper.spacemouse_devices,
                action_indices=self.wrapper.action_indices,
                action_scale=self.wrapper.spacemouse_action_scale,
                intercept_with_button=self.wrapper.spacemouse_intercept_with_button,
                device=self.device
            )

        # Axis-distance Reward
        if self.wrapper.reward_axis_targets:
            env = AxisDistanceRewardWrapper(
                env,
                targets=self.wrapper.reward_axis_targets,
                axis=self.wrapper.reward_axis,
                scale=self.wrapper.reward_scale,
                clip=self.wrapper.reward_clip,
                terminate_on_success=self.wrapper.reward_terminate_on_success,
                normalization_range=[0.0, self.wrapper.reward_axis_targets["main"]]
            )

        env = ConvertToLeRobotObservation(env, device=self.device)

        if self.wrapper.crop_params_dict is not None:
            env = ImageCropResizeWrapper(
                env=env,
                crop_params_dict=self.wrapper.crop_params_dict,
                resize_size=self.wrapper.crop_resize_size,
            )

        env = AMPObsWrapper(
            env,
            use_xy_position=self.use_xy_position,
            use_torque=self.use_torque,
            device=self.device

        )

        env = BatchCompatibleWrapper(env=env)
        env = TorchActionWrapper(env, device=self.device)

        return env


@EnvConfig.register_subclass("ur3_nist_insertion_xyc_medium")
@dataclass
class UR3_NIST_Insertion_XYC_Medium(UR3_NIST_Insertion_XYC_Small):
    repo_id: str = "jannick-st/ur3-nist-insertion-xyc-medium-offline-demos"
    dataset_root: str = "/home/jannick/data/jannick-st/ur3-nist-insertion/offline-demos-xyc-medium"
    xy_offset_std_mm: float = 3.0
    c_offset_std_rad: float = 0.15


@EnvConfig.register_subclass("ur3_nist_insertion_xyc_large")
@dataclass
class UR3_NIST_Insertion_XYC_Large(UR3_NIST_Insertion_XYC_Small):
    repo_id: str = "jannick-st/ur3-nist-insertion-xyc-large-offline-demos"
    dataset_root: str = "/home/jannick/data/jannick-st/ur3-nist-insertion/offline-demos-xyc-large"
    xy_offset_std_mm: float = 6.0
    c_offset_std_rad: float = 0.3


@EnvConfig.register_subclass("ur3_nist_insertion_xy_small")
@dataclass
class UR3_NIST_Insertion_XY_Small(UR3_NIST_Insertion_XYC_Small):
    repo_id: str = "jannick-st/ur3-nist-insertion-xy-small-offline-demos"
    dataset_root: str = "/home/jannick/data/jannick-st/ur3-nist-insertion/offline-demos-xy-small"
    c_offset_std_rad: float = 0.0
    depth_to_tighten_c_bounds: Optional[float] = None

    def __post_init__(self):
        self.wrapper.action_indices["main"] = [1, 1, 0, 0, 0, 0]
        self.wrapper.spacemouse_action_scale["main"] = [0.02, -0.02, 0, 0, 0, 0.0]
        UR3_NIST_Insertion_XYC_Small.__post_init__(self)


@EnvConfig.register_subclass("ur3_nist_insertion_xy_medium")
@dataclass
class UR3_NIST_Insertion_XY_Medium(UR3_NIST_Insertion_XY_Small):
    repo_id: str = "jannick-st/ur3-nist-insertion-xy-medium-offline-demos"
    dataset_root: str = "/home/jannick/data/jannick-st/ur3-nist-insertion/offline-demos-xy-medium"
    xy_offset_std_mm: float = 3.0


@EnvConfig.register_subclass("ur3_nist_insertion_xy_large")
@dataclass
class UR3_NIST_Insertion_XY_Large(UR3_NIST_Insertion_XY_Small):
    repo_id: str = "jannick-st/ur3-nist-insertion-xy-large-offline-demos"
    dataset_root: str = "/home/jannick/data/jannick-st/ur3-nist-insertion/offline-demos-xy-large"
    xy_offset_std_mm: float = 6.0


# -----------------
# Policies
# -----------------
@PreTrainedConfig.register_subclass("sac_nist_insertion_xyc")
@dataclass
class SAC_NIST_Insertion_XYC(SACConfig):
    # tuning recipe:
    # try to hit 30 fps
    # keep ratio of utd_ratio:num_critics at 3:2, increase as much as possible
    # freeze and share the encoder
    # if possible, use "cuda" as the storage device, decrease buffer sizes accordingly
    # run the actor on cpu for maximum throughput

    online_steps: int = 10000000
    training_starts: int = 60
    online_buffer_capacity: int = 25000
    offline_buffer_capacity: int = 10000
    camera_number: int = 1  # also affects fps linearly, resolution affects quadratically
    cta_ratio: int = 2  # affects fps linearly, hil-serl default is 2
    storage_device: str = "cuda"  # destabilizes fps, sometimes cuts 10 fps
    shared_encoder: bool = True  # does not affect fps much
    num_critics: int = 3  # affects fps sub-linearly, hil-serl default is 2
    target_entropy: float = -1.5  # -dim(A) / 2
    use_backup_entropy: bool = False  # td backup the entropy too -> more stable in my experience if entropy only affects actor loss
    freeze_vision_encoder: bool = False  # cuts ~10 fps for one camera

    dataset_stats: dict[str, dict[str, list[float]]] | None = field(
        default_factory=lambda: {
            "observation.image.main": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "observation.state": {
                "min": [-10.0, -10.0, -10.0, -2.0, -2.0, -2.0, -0.00042, -0.02, -0.02, -1.0],
                "max": [10.0, 10.0, 10.0, 2.0, 2.0, 2.0, 0.01442, 0.02, 0.02, 1.0]
            },
            "action": {
                "min": [-0.02, -0.02, -1.0],
                "max": [0.02, 0.02, 1.0]
            },
        }
    )


@PreTrainedConfig.register_subclass("sac_nist_insertion_xy")
@dataclass
class SAC_NIST_Insertion_XY(SAC_NIST_Insertion_XYC):
    dataset_stats: dict[str, dict[str, list[float]]] | None = field(
        default_factory=lambda: {
            "observation.image.main": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "observation.state": {
                "min": [-10.0, -10.0, -10.0, -2.0, -2.0, -2.0, -0.00042, -0.02, -0.02],
                "max": [10.0, 10.0, 10.0, 2.0, 2.0, 2.0, 0.01442, 0.02, 0.02]
            },
            "action": {
                "min": [-0.02, -0.02],
                "max": [0.02, 0.02]
            },
        }
    )


@PreTrainedConfig.register_subclass("mlp_nist_insertion_xyc")
@dataclass
class DAgger_NIST_Insertion_XYC(MLPConfig):
    # tuning recipe:
    # try to hit 30 fps
    # keep ratio of utd_ratio:num_critics at 3:2, increase as much as possible
    # freeze and share the encoder
    # if possible, use "cuda" as the storage device, decrease buffer sizes accordingly
    # run the actor on cpu for maximum throughput

    online_steps: int = 10000000
    online_step_before_learning: int = 60
    buffer_capacity: int = 25000
    camera_number: int = 1  # also affects fps linearly, resolution affects quadratically
    storage_device: str = "cuda"  # destabilizes fps, sometimes cuts 10 fps
    freeze_vision_encoder: bool = False  # cuts ~10 fps for one camera

    dataset_stats: dict[str, dict[str, list[float]]] | None = field(
        default_factory=lambda: {
            "observation.image.main": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "observation.state": {
                "min": [-10.0, -10.0, -10.0, -2.0, -2.0, -2.0, -0.00042, -0.02, -0.02, -1.0],
                "max": [10.0, 10.0, 10.0, 2.0, 2.0, 2.0, 0.01442, 0.02, 0.02, 1.0]
            },
            "action": {
                "min": [-0.02, -0.02, -1.0],
                "max": [0.02, 0.02, 1.0]
            },
        }
    )

# -----------------
# Wrapper
# -----------------


class AwaitForceResetWrapper(gymnasium.Wrapper):
    def __init__(self, env, threshold: float = 1.0, axis: int = 2, timeout: float = 5.0):
        super().__init__(env)
        self.threshold = threshold
        self.axis = axis
        self.timeout = timeout

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.env.unwrapped.robot.controllers["main"].zero_ft()

        start_time = time.time()
        while True:
            wrench = self.env.unwrapped.robot.capture_observation()["observation.main_eef_wrench"]
            if abs(wrench[self.axis]) >= abs(self.threshold):
                break

            if time.time() - start_time > self.timeout:
                print(f"[WARN] Did not reach target force {self.threshold:.1f}N"
                      f"within {self.timeout}s, last measured contact force was {wrench[self.axis]:.1f}N")
                break

            time.sleep(0.01)

        return obs, info


class TightenXYCBoundsWrapper(gymnasium.Wrapper):
    def __init__(
        self,
        env,
        static_tffs,
        threshold=0.0078,
        pos_eps=1e-2,
        rot_eps=1e-3
    ):
        super().__init__(env)
        self.static_tffs = static_tffs
        self.threshold = threshold
        self.pos_eps = pos_eps
        self.rot_eps = rot_eps

        self._tightened = False
        self._initial_max_pose_rpy = static_tffs["main"].max_pose_rpy
        self._initial_min_pose_rpy = static_tffs["main"].min_pose_rpy

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        pose = obs["observation.main_eef_pos"].detach().cpu().numpy()
        if not self._tightened and pose[2] > self.threshold:
            logging.info("Tighten bounds")
            new_min_pose_rpy = self._initial_min_pose_rpy.copy()
            new_max_pose_rpy = self._initial_max_pose_rpy.copy()

            for idx in [0, 1]:
                new_min_pose_rpy[idx] = pose[idx] - self.pos_eps
                new_max_pose_rpy[idx] = pose[idx] + self.pos_eps

            rpy = R.from_rotvec(pose[3:6]).as_euler('xyz', degrees=False)
            for idx in [5]:
                new_min_pose_rpy[idx] = rpy[idx - 3] - self.rot_eps
                new_max_pose_rpy[idx] = rpy[idx - 3] + self.rot_eps

            ctrl = self.env.unwrapped.robot.controllers["main"]
            ctrl.send_cmd(TaskFrameCommand(
                min_pose_rpy=new_min_pose_rpy,
                max_pose_rpy=new_max_pose_rpy
            ))
            self._tightened = True

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._tightened = False
        return self.env.reset(**kwargs)


if __name__ == "__main__":
    # print policy bounds
    cfg = UR3_NIST_Insertion_XYC_Small()

    min_pose = cfg.wrapper.static_tffs["main"].min_pose_rpy
    max_pose = cfg.wrapper.static_tffs["main"].max_pose_rpy

    if cfg.use_torque:
        min_bounds = [-10.0] * 3 + [-2.0] * 3
        max_bounds = [10.0] * 3 + [2.0] * 3
    else:
        min_bounds = [-10.0] * 3
        max_bounds = [10.0] * 3

    if cfg.use_xy_position:
        min_bounds = min_bounds + min_pose[:3] + [-s for s in cfg.wrapper.spacemouse_action_scale["main"][:2]]
        max_bounds = max_bounds + max_pose[:3] + [s for s in cfg.wrapper.spacemouse_action_scale["main"][:2]]
    else:
        min_bounds = min_bounds + [min_pose[2]] + [-s for s in cfg.wrapper.spacemouse_action_scale["main"][:2]]
        max_bounds = max_bounds + [max_pose[2]] + [s for s in cfg.wrapper.spacemouse_action_scale["main"][:2]]

    print("=== policy input normalization parameters")
    print("- observation.state")
    print("    min:", min_bounds)
    print("    max:", max_bounds)

