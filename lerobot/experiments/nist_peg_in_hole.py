from dataclasses import dataclass, field

import gymnasium
import numpy as np
import torch
from gymnasium import spaces

from lerobot.common.constants import ACTION, OBS_ROBOT
from lerobot.common.envs.configs import TaskFrameWrapperConfig, EnvConfig, UREnvConfig
from lerobot.common.policies.sac.configuration_sac import SACConfig
from lerobot.common.robot_devices.motors.configs import URArmConfig
from lerobot.common.robot_devices.motors.rtde_tff_controller import TaskFrameCommand, AxisMode
from lerobot.common.robot_devices.robots.configs import URConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature


@EnvConfig.register_subclass("ur3_nist_peg_in_hole")
@dataclass
class URPegInHoleConfig(UREnvConfig):
    num_episodes: int = 20
    repo_id: str = "jannick-st/nist-peg-in-hole-offline-demos"
    dataset_root: str = "/home/jannick/data/jannick-st/nist-peg-in-hole/offline-demos"
    task: str = "Push in the peg"
    resume: bool = False
    fps: int = 10
    display_cameras: bool = False
    push_to_hub: bool = False

    explore: bool = False
    xy_offset_limit_mm: float = 3.0

    robot: URConfig = URConfig(
        follower_arms={
            "main": URArmConfig(
                robot_ip="172.22.22.2",
                soft_real_time=True,
                use_gripper=False,
                wrench_limits=[4.0, 4.0, 4.0, 0.4, 0.4, 0.4]
            )
        }
    )
    wrapper: TaskFrameWrapperConfig = TaskFrameWrapperConfig(
        control_time_s=5.0,
        static_tffs={
            "main": TaskFrameCommand(
                T_WF=np.eye(4).tolist(),
                target=[0.0, 0.0, -(9.82 * 0.925 + 2.0), 2.221, -2.221, 0.0],
                mode=2 * [AxisMode.IMPEDANCE_VEL] + [AxisMode.FORCE] + 3 * [AxisMode.POS],
                kp=[3000, 3000, 3000, 200, 200, 200],
                kd=[150, 150, 300, 8, 8, 8]
            )
        },
        action_indices={
            "main": [1, 1, 0, 0, 0, 0]
        },
        reset_pos={
            "main": [0.1812, -0.3752, 0.112, 2.221, -2.221, 0.0]
        },
        noise_dist="uniform",
        noise_std={},  # set in post_init
        safe_reset=True,
        threshold=0.0008,
        timeout=1.0,
        spacemouse_devices={
            "main": "SpaceMouse Compact"
        },
        spacemouse_action_scale={
            "main": [1 / 25] * 6
        },
        reward_axis_targets={
            "main": 0.096
        },
        reward_axis=2,
        reward_scale=1.0,
        reward_clip=None,
        reward_terminate_on_success=False
    )

    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "observation.state": OBS_ROBOT,
        }
    )

    def __post_init__(self):
        for name in self.robot.follower_arms:
            reset_pos = np.array(self.wrapper.reset_pos["main"])
            max_pose = reset_pos.copy()
            min_pose = reset_pos.copy()

            max_pose[:2] += self.xy_offset_limit_mm / 1000.0
            min_pose[:2] -= self.xy_offset_limit_mm / 1000.0

            z_range = reset_pos[2] - self.wrapper.reward_axis_targets["main"]
            max_pose[2] = reset_pos[2] + z_range * 0.03
            min_pose[2] = self.wrapper.reward_axis_targets["main"] - z_range * 0.03

            max_pose[3:] -= 1e-3
            min_pose[3:] += 1e-3
            max_pose[4:] += 1e-3
            min_pose[4:] -= 1e-3

            self.robot.follower_arms[name].max_pose = max_pose.tolist()
            self.robot.follower_arms[name].min_pose = min_pose.tolist()

            self.wrapper.noise_std[name] = (max_pose - min_pose) / 2.0
            self.wrapper.noise_std[name][2:] = 0.0
            self.wrapper.noise_std[name] = self.wrapper.noise_std[name].tolist()

        if self.explore:
            self.wrapper.control_time_s = 300
            self.wrapper.static_tffs["main"].target[:3] = [0.0, 0.0, 0.0]
            self.wrapper.static_tffs["main"].mode[:3] = 3 * [AxisMode.IMPEDANCE_VEL]
            self.wrapper.action_indices["main"] = [1, 1, 1, 0, 0, 0]

    def make(self):
        env = super().make()
        return AMPObsWrapper(env)


@PreTrainedConfig.register_subclass("sac_nist_peg_in_hole")
@dataclass
class SACPegInHoleConfig(SACConfig):
    # tuning recipe:
    # try to hit 30 fps
    # keep ratio of utd_ratio:num_critics at 3:2, increase as much as possible
    # freeze and share the encoder
    # if possible, use "cuda" as the storage device, decrease buffer sizes accordingly
    # run the actor on cpu for maximum throughput

    online_step_before_learning: int = 50
    online_buffer_capacity: int = 10000
    offline_buffer_capacity: int = 10000
    camera_number: int = 0  # also affects fps linearly, resolution affects quadratically
    utd_ratio: int = 3  # affects fps linearly, hil-serl default is 2
    storage_device: str = "cuda"  # destabilizes fps, sometimes cuts 10 fps
    shared_encoder: bool = True  # does not affect fps much
    num_critics: int = 3  # affects fps sub-linearly, hil-serl default is 2
    target_entropy: float = -1.0  # -dim(A) / 2
    use_backup_entropy: bool = False  # td backup the entropy too -> more stable in my experience if entropy only affects actor loss
    freeze_vision_encoder: bool = True  # cuts ~10 fps for one camera

    dataset_stats: dict[str, dict[str, list[float]]] | None = field(
        default_factory=lambda: {
            "observation.state": {
                "min": [-3.0, -3.0, -3.0, 0.098, -1/25, -1/25],
                "max": [3.0, 3.0, 20.0, 0.115, 1/25, 1/25]
            },
            "action": {
                "min": [-1/25, -1/25],
                "max": [1/25, 1/25]
            },
        }
    )


class AMPObsWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Dict({
            "observation.state": spaces.Box(
                low=np.full(6, -np.inf),
                high=np.full(6, np.inf),
                shape=(6, ),
                dtype=np.uint8)
        })

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if info["is_intervention"]:
            action_to_append = info["action_intervention"]
        else:
            action_to_append = action

        new_obs = {
            "observation.state": torch.tensor([
                    obs["observation.main_eef_wrench"][0],
                    obs["observation.main_eef_wrench"][1],
                    obs["observation.main_eef_wrench"][2],
                    obs["observation.main_eef_pos"][2],
                    action_to_append[0],
                    action_to_append[1]
                ])
        }

        return new_obs, reward, terminated, truncated, info


