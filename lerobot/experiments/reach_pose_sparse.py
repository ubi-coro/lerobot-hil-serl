from dataclasses import dataclass, field

import gymnasium as gym

from lerobot.common.constants import ACTION, OBS_ROBOT, OBS_IMAGE
from lerobot.common.envs.configs import (
    EnvConfig,
    HILSerlRobotEnvConfig,
    EnvWrapperConfig,
    EEActionSpaceConfig
)
from lerobot.common.envs.wrapper.reward import SuccessRepeatWrapper
from lerobot.common.policies.sac.configuration_sac import SACConfig
from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig, IntelRealSenseCameraConfig
from lerobot.common.robot_devices.robots.configs import AlohaRobotConfig, RobotConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature


class TimePenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty_per_step=0.01):
        super().__init__(env)
        self.penalty_per_step = penalty_per_step

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        reward -= self.penalty_per_step

        return obs, reward, terminated, truncated, info


class TargetPoseWrapper(gym.Wrapper):
    def __init__(self, env, target_pose, axis=0):
        super().__init__(env)
        self.target_pose = target_pose
        self.axis = axis

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        ee_pose = obs["observation.state"][-3:].cpu().numpy()

        if ee_pose[self.axis] > self.target_pose[self.axis]:
            reward = 1.0
            terminated = True

        return obs, reward, terminated, truncated, info


@EnvConfig.register_subclass("real_reach_pose_sparse")
@dataclass
class RealReachPoseSparseEnvConfig(HILSerlRobotEnvConfig):
    repo_id: str = "jannick-st/reach-pose-sparse-offline-demos-repeat"
    dataset_root: str = "/media/nvme1/jstranghoener/lerobot/data/jannick-st/reach-pose-sparse-repeat/offline-demos"
    task: str = "Reach the target pose"
    num_episodes: int = 20  # only for record mode
    episode: int = 0
    device: str = "cuda"
    push_to_hub: bool = False
    fps: int = 10

    target_ee_pos: list[float] = field(default_factory=lambda: [0.31, 0.0, 0.15])
    axis: int = 0
    penalty_per_step: float = 0.005
    num_success_repeats: int = 3
    vision_only: bool = False
    state_only: bool = False

    wrapper: EnvWrapperConfig = EnvWrapperConfig(
        display_cameras=True,
        control_time_s=10.0,
        add_ee_pose_to_observation=True,
        fixed_reset_joint_positions=[ 0.0,  -24.609375,   -24.433594,    52.558594,    52.822266, -0.43945312,  56.953125,    -2.8125,       4.6242776 ],
        smoothing_range_factor=0.3,
        ee_action_space_params=EEActionSpaceConfig(
            x_step_size=0.02,
            y_step_size=0.02,
            z_step_size=0.02,
            bounds={
                "max": [0.32, 0.15, 0.25],
                "min": [0.16, -0.15, 0.09]
            },
            control_mode="leader"
        ),
        crop_params_dict={
            "observation.images.cam_top": (57, 23, 401, 295),
        #    #"observation.images.cam_left_wrist": (25, 86, 450, 547)
        },
        resize_size=(64, 64)
    )

    robot: RobotConfig = field(default_factory=lambda: AlohaRobotConfig(
        cameras={
            "cam_top": OpenCVCameraConfig(
                camera_index="/dev/CAM_HIGH",
                fps=30,
                width=640,
                height=480,
            ),
            #"cam_left_wrist": IntelRealSenseCameraConfig(
            #    serial_number=218722270675,
            #    fps=30,
            #    width=640,
            #    height=480,
            #)
        },
        calibration_dir="/home/jstranghoener/PycharmProjects/lerobot-hil-serl/.cache/calibration/aloha_default"
    )
                               )
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(3,)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(15,)),
            #"observation.images.cam_left_wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
            "observation.images.cam_top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64))
        }
    )

    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "observation.state": OBS_ROBOT,
            #"observation.images.cam_left_wrist": f"{OBS_IMAGE}s.cam_left_wrist",
            "observation.images.cam_top": f"{OBS_IMAGE}s.cam_top"
        }
    )

    def __post_init__(self):
        if self.vision_only and self.state_only:
            raise ValueError("vision_only and state_only cannot be both True")

        if self.vision_only:
            del self.features["observation.state"]
            del self.features_map["observation.state"]

        if self.state_only:
            del self.features["observation.images.cam_left_wrist"]
            del self.features["observation.images.cam_top"]
            del self.features_map["observation.images.cam_left_wrist"]
            del self.features_map["observation.images.cam_top"]
            self.robot.cameras = dict()

        if self.mode == "record":
            self.crop_params_dict = dict()

    def make(self):
        import lerobot.common.envs.wrapper.hilserl as wrapper
        from lerobot.common.envs.manipulator_env import RobotEnv
        from lerobot.common.envs.wrapper.smoothing import SmoothActionWrapper
        from lerobot.common.robot_devices.robots.utils import make_robot_from_config

        robot = make_robot_from_config(self.robot)
        # Create base environment
        env = RobotEnv(
            robot=robot,
            display_cameras=self.wrapper.display_cameras,
        )

        # Add observation and image processing
        if self.wrapper.add_joint_velocity_to_observation:
            env = wrapper.AddJointVelocityToObservation(env=env, fps=self.fps)
        if self.wrapper.add_current_to_observation:
            env = wrapper.AddCurrentToObservation(env=env)
        if self.wrapper.add_ee_pose_to_observation:
            env = wrapper.EEObservationWrapper(env=env, ee_pose_limits=self.wrapper.ee_action_space_params.bounds)

        env = wrapper.ConvertToLeRobotObservation(env=env, device=self.device)

        if self.wrapper.crop_params_dict is not None:
            env = wrapper.ImageCropResizeWrapper(
                env=env,
                crop_params_dict=self.wrapper.crop_params_dict,
                resize_size=self.wrapper.resize_size,
            )

        # reward and termination
        env = TargetPoseWrapper(env, self.target_ee_pos, axis=self.axis)
        env = TimePenaltyWrapper(env, penalty_per_step=self.penalty_per_step)
        env = SuccessRepeatWrapper(env, num_repeats=self.num_success_repeats)
        env = wrapper.TimeLimitWrapper(env=env, control_time_s=self.wrapper.control_time_s, fps=self.fps)

        env = wrapper.EEActionWrapper(
            env=env,
            ee_action_space_params=self.wrapper.ee_action_space_params,
            use_gripper=self.wrapper.use_gripper,
        )

        if self.wrapper.smoothing_range_factor is not None:
            env = SmoothActionWrapper(env, smoothing_range_factor=self.wrapper.smoothing_range_factor,
                                      device=self.device)

        if self.wrapper.ee_action_space_params.control_mode == "gamepad":
            env = wrapper.GamepadControlWrapper(
                env=env,
                x_step_size=self.wrapper.ee_action_space_params.x_step_size,
                y_step_size=self.wrapper.ee_action_space_params.y_step_size,
                z_step_size=self.wrapper.ee_action_space_params.z_step_size,
                use_gripper=self.wrapper.use_gripper,
            )
        elif self.wrapper.ee_action_space_params.control_mode == "leader":
            env = wrapper.GearedLeaderControlWrapper(
                env=env,
                ee_action_space_params=self.wrapper.ee_action_space_params,
                use_gripper=self.wrapper.use_gripper,
            )
        elif self.wrapper.ee_action_space_params.control_mode == "leader_automatic":
            env = wrapper.GearedLeaderAutomaticControlWrapper(
                env=env,
                ee_action_space_params=self.wrapper.ee_action_space_params,
                use_gripper=self.wrapper.use_gripper,
            )
        else:
            raise ValueError(f"Invalid control mode: {self.wrapper.ee_action_space_params.control_mode}")

        env = wrapper.ResetWrapper(
            env=env,
            reset_pose=self.wrapper.fixed_reset_joint_positions,
            reset_time_s=self.wrapper.reset_time_s,
        )
        env = wrapper.BatchCompatibleWrapper(env=env)
        env = wrapper.TorchActionWrapper(env=env, device=self.device)

        return env


@PreTrainedConfig.register_subclass("sac_real_reach_pose_sparse")
@dataclass
class SACRealReachPoseSparseConfig(SACConfig):
    # tuning recipe:
    # try to hit 30 fps
    # keep ratio of utd_ratio:num_critics at 3:2, increase as much as possible
    # freeze and share the encoder
    # if possible, use "cuda" as the storage device

    online_step_before_learning: int = 20
    camera_number: int = 1  # also affects fps linearly, resolution affects quadratically
    utd_ratio: int = 6  # affects fps linearly
    storage_device: str = "cuda"  # destabilizes fps, sometimes cuts 10 fps
    shared_encoder: bool = True  # does not affect fps much
    num_critics: int = 4  # affects fps sub-linearly
    target_entropy: float = -1.5
    use_backup_entropy: bool = False  # as per lil'km
    freeze_vision_encoder: bool = True

    online_buffer_capacity: int = 10000
    offline_buffer_capacity: int = 10000

    dataset_stats: dict[str, dict[str, list[float]]] | None = field(
        default_factory=lambda: {
            "observation.state": {
                "min": [0.16, -0.15, 0.08, 0.0, -0.02, -0.02, -0.02],
                "max": [0.32, 0.15, 0.25, 3.0, 0.02, 0.02, 0.02],
            },
            "observation.images.cam_top": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "observation.images.cam_left_wrist": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "action": {
                "min": [-0.02, -0.02, -0.02],
                "max": [0.02, 0.02, 0.02],
            },
        }
    )
