from dataclasses import dataclass, field
from typing import Optional

from lerobot.common.constants import ACTION, OBS_ROBOT, OBS_IMAGE
from lerobot.common.envs.configs import (
    EnvConfig,
    HILSerlRobotEnvConfig,
    EnvWrapperConfig,
    EEActionSpaceConfig
)
from lerobot.common.policies.sac.configuration_sac import SACConfig
from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig, IntelRealSenseCameraConfig
from lerobot.common.robot_devices.robots.configs import AlohaRobotConfig, RobotConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature


@EnvConfig.register_subclass("real_grasp_cube")
@dataclass
class RealGraspCubeEnvConfig(HILSerlRobotEnvConfig):
    repo_id: str = "jannick-st/grasp-cube-offline-demos"
    dataset_root: str = "/media/nvme1/jstranghoener/lerobot/data/jannick-st/grasp-cube/offline-demos"
    task: str = "Grasp the cube"
    num_episodes: int = 30  # only for record mode
    episode: int = 0
    device: str = "cuda"
    push_to_hub: bool = False
    fps: int = 10

    pretrained_policy_name_or_path: Optional[str] = None
    reward_classifier_pretrained_path: Optional[str] = "/media/nvme1/jstranghoener/lerobot/models/jannick-st/grasp-cube/classifier-150525/checkpoints/005940/pretrained_model/"

    wrapper: EnvWrapperConfig = EnvWrapperConfig(
        display_cameras=True,
        num_success_repeats=4,
        control_time_s=15.0,
        add_ee_pose_to_observation=True,
        keep_joints_in_observation=False,
        use_gripper=True,
        fixed_reset_joint_positions=[0.52734375, -8.349609, -8.0859375, 68.203125, 68.55469, -1.5820312, -27.773438, -2.9003906, 90.0],
        smoothing_range_factor=0.4,
        ee_action_space_params=EEActionSpaceConfig(
            x_step_size=0.015,
            y_step_size=0.015,
            z_step_size=0.015,
            bounds={
                "max": [0.36, 0.12, 0.10],
                "min": [0.25, -0.12, 0.07]
            },
            control_mode="leader"
        ),
        crop_params_dict={
            "observation.images.cam_left_wrist": (
                2,
                133,
                384,
                446
            ),
            "observation.images.cam_low": (
                185,
                306,
                293,
                331
            )
        },
        resize_size=(64, 64),
        foot_switches={
            "episode_success": {"device": 18, "toggle": False},
            "human_intervention_step": {"device": 21, "toggle": True}
        }
    )

    robot: RobotConfig = field(default_factory=lambda: AlohaRobotConfig(
            cameras={
                "cam_low": OpenCVCameraConfig(
                    camera_index="/dev/CAM_LOW",
                    fps=30,
                    width=640,
                    height=480,
                ),
                "cam_left_wrist": IntelRealSenseCameraConfig(
                    serial_number=218722270675,
                    fps=30,
                    width=640,
                    height=480,
                )
            },
            calibration_dir="/home/jstranghoener/PycharmProjects/lerobot-hil-serl/.cache/calibration/aloha_default",
            max_relative_target=100
        )
    )

    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            "observation.images.cam_low": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            "observation.images.cam_left_wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64))
        }
    )

    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "observation.state": OBS_ROBOT,
            "observation.images.cam_low": f"{OBS_IMAGE}s.cam_low",
            "observation.images.cam_left_wrist": f"{OBS_IMAGE}s.cam_left_wrist"
        }
    )




@PreTrainedConfig.register_subclass("sac_real_grasp_cube")
@dataclass
class SACRealGraspCubeConfig(SACConfig):
    # tuning recipe:
    # try to hit 30 fps
    # keep ratio of utd_ratio:num_critics at 3:2, increase as much as possible
    # freeze and share the encoder
    # if possible, use "cuda" as the storage device, decrease buffer sizes accordingly
    # run the actor on cpu for maximum throughput

    training_starts: int = 50
    online_buffer_capacity: int = 10000
    offline_buffer_capacity: int = 10000
    camera_number: int = 2  # also affects fps linearly, resolution affects quadratically
    cta_ratio: int = 2  # affects fps linearly, hil-serl default is 2
    storage_device: str = "cuda"  # destabilizes fps, sometimes cuts 10 fps
    shared_encoder: bool = True  # does not affect fps much
    num_critics: int = 2  # affects fps sub-linearly, hil-serl default is 2
    target_entropy: float = -2.0  # -dim(A) / 2
    use_backup_entropy: bool = False  # td backup the entropy too -> more stable in my experience if entropy only affects actor loss
    freeze_vision_encoder: bool = True  # cuts ~10 fps for one camera

    dataset_stats: dict[str, dict[str, list[float]]] | None = field(
        default_factory=lambda: {
            "observation.state": {
                "min": [0.36, 0.12, 0.10, 100.0, -0.02, -0.02, -0.02, 0.0],
                "max": [0.25, -0.12, 0.07, 0.0, 0.02, 0.02, 0.02, 4.0]
            },
            "action": {
                "min": [-0.02, -0.02, -0.02, 0.0],
                "max": [0.02, 0.02, 0.02, 2.0]
            },

            # cams use ImageNet stats
            "observation.images.cam_low": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "observation.images.cam_left_wrist": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        }
    )