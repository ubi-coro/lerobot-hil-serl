import time
from dataclasses import dataclass, field

import gymnasium
import numpy as np
import torch
from gymnasium import spaces

from lerobot.common.constants import ACTION, OBS_ROBOT
from lerobot.common.envs.configs import TaskFrameWrapperConfig, EnvConfig, UREnvConfig
from lerobot.common.envs.ur_env import UREnv
from lerobot.common.envs.wrapper.hilserl import ConvertToLeRobotObservation, TorchActionWrapper, TimeLimitWrapper
from lerobot.common.envs.wrapper.reward import AxisDistanceRewardWrapper
from lerobot.common.envs.wrapper.spacemouse import SpaceMouseInterventionWrapper
from lerobot.common.envs.wrapper.tff import StaticTaskFrameResetWrapper, StaticTaskFrameActionWrapper
from lerobot.common.policies.sac.configuration_sac import SACConfig
from lerobot.common.robot_devices.motors.configs import URArmConfig
from lerobot.common.robot_devices.motors.rtde_tff_controller import TaskFrameCommand, AxisMode
from lerobot.common.robot_devices.robots.configs import URConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature


@EnvConfig.register_subclass("ur3_nist_peg_in_hole")
@dataclass
class URPegInHoleConfig(UREnvConfig):
    num_episodes: int = 10
    repo_id: str = "jannick-st/nist-peg-in-hole-offline-demos"
    dataset_root: str = "/home/jannick/data/jannick-st/nist-peg-in-hole/offline-demos"
    task: str = "Push in the peg"
    resume: bool = False
    fps: int = 10
    display_cameras: bool = False
    push_to_hub: bool = False

    explore: bool = False
    xy_offset_limit_mm: float = 4.0
    use_xy_position: bool = False
    use_torque: bool = True

    robot: URConfig = URConfig(
        follower_arms={
            "main": URArmConfig(
                robot_ip="172.22.22.2",
                soft_real_time=True,
                use_gripper=False,
                wrench_limits=[4.0, 4.0, 4.0, 0.4, 0.4, 0.4],
                payload_mass=0.925,
                payload_cog=[0.0, 0.0, 0.058]
            )
        }
    )
    wrapper: TaskFrameWrapperConfig = TaskFrameWrapperConfig(
        control_time_s=5.0,
        static_tffs={
            "main": TaskFrameCommand(
                T_WF=np.eye(4).tolist(),
                target=[0.0, 0.0, -4.0, 2.221, -2.221, 0.0],
                mode=2 * [AxisMode.IMPEDANCE_VEL] + [AxisMode.FORCE] + 3 * [AxisMode.POS],
                kp=[3000, 3000, 3000, 300, 300, 300],
                kd=[150, 150, 300, 12, 12, 12]
            )
        },
        action_indices={
            "main": [1, 1, 0, 0, 0, 0]
        },
        reset_pos={
            "main": [0.1809, -0.3746, 0.11, 2.221, -2.221, 0.0]
        },
        noise_dist="uniform",
        noise_std={},  # set in post_init
        safe_reset=True,
        threshold=0.0008,
        timeout=1.5,
        spacemouse_devices={
            "main": "SpaceMouse Compact"
        },
        spacemouse_action_scale={
            "main": [1 / 25] * 6
        },
        spacemouse_intercept_with_button=True,
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
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "observation.state": OBS_ROBOT,
        }
    )

    def __post_init__(self):
        self.wrapper.action_bounds = {}
        for name in self.robot.follower_arms:

            # build pose limits from reset pose and xy_offset
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

            # build noise level from pose limits
            self.wrapper.noise_std[name] = (max_pose - min_pose) / 2.0
            self.wrapper.noise_std[name][2:] = 0.0
            self.wrapper.noise_std[name] = self.wrapper.noise_std[name].tolist()

            # build action space bounds from space mouse scaling factors
            action_scale = self.wrapper.spacemouse_action_scale[name]
            action_indices = self.wrapper.action_indices[name]
            self.wrapper.action_bounds[name] = {
                "min": [-s for i, s in enumerate(action_scale) if action_indices[i]],
                "max": [s for i, s in enumerate(action_scale) if action_indices[i]]
            }

        if self.explore:
            self.wrapper.control_time_s = 300
            self.wrapper.static_tffs["main"].target[:3] = [0.0, 0.0, 0.0]
            self.wrapper.static_tffs["main"].mode[:3] = 3 * [AxisMode.IMPEDANCE_VEL]
            self.wrapper.action_indices["main"] = [1, 1, 1, 0, 0, 0]

        state_dim = 6
        if self.use_xy_position:
            state_dim += 2
        if self.use_torque:
            state_dim += 3
        self.features["observation.state"].shape = (state_dim, )


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
            threshold=2.0,
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
                terminate_on_success=self.wrapper.reward_terminate_on_success
            )

        env = ConvertToLeRobotObservation(env, device=self.device)
        env = AMPObsWrapper(env, use_xy_position=self.use_xy_position, device=self.device)
        env = TorchActionWrapper(env, device=self.device)

        return env


@PreTrainedConfig.register_subclass("sac_nist_peg_in_hole")
@dataclass
class SACPegInHoleConfig(SACConfig):
    # tuning recipe:
    # try to hit 30 fps
    # keep ratio of utd_ratio:num_critics at 3:2, increase as much as possible
    # freeze and share the encoder
    # if possible, use "cuda" as the storage device, decrease buffer sizes accordingly
    # run the actor on cpu for maximum throughput

    online_steps: int = 10000000
    online_step_before_learning: int = 30
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
                "min": [-8.0, -8.0, -8.0, 0.1769, -0.3786, 0.09558, -0.04, -0.0],
                "max": [8.0, 8.0, 8.0, 0.1849, -0.3706, 0.11042, 0.04, 0.04]
            },
            "action": {
                "min": [-1/25, -1/25],
                "max": [1/25, 1/25]
            },
        }
    )


class AMPObsWrapper(gymnasium.Wrapper):
    def __init__(self,
                 env,
                 use_xy_position: bool = False,
                 use_torque: bool = False,
                 device: str = "cuda"):
        super().__init__(env)
        self.device = device
        self.prev_action = np.zeros(env.action_space.shape, dtype=np.float32)
        self.use_xy_position = use_xy_position
        self.use_torque = use_torque
        
        num_actions = 8 if use_xy_position else 6
        self.observation_space = spaces.Dict({
            "observation.state": spaces.Box(
                low=np.full(num_actions, -np.inf),
                high=np.full(num_actions, np.inf),
                shape=(num_actions, ),
                dtype=np.uint8)
        })

    def _obs(self, obs):
        new_obs = [
            obs["observation.main_eef_wrench"][0],
            obs["observation.main_eef_wrench"][1],
            obs["observation.main_eef_wrench"][2]
        ]

        if self.use_torque:
            new_obs.extend([
                obs["observation.main_eef_wrench"][3],
                obs["observation.main_eef_wrench"][4],
                obs["observation.main_eef_wrench"][5]
            ])

        if self.use_xy_position:
            new_obs.extend([
                obs["observation.main_eef_pos"][0],
                obs["observation.main_eef_pos"][1],
            ])

        new_obs.extend([
            obs["observation.main_eef_pos"][2],
            self.prev_action[0],
            self.prev_action[1]
        ])

        return {
            "observation.state": torch.tensor(new_obs).to(device=self.device)
        }

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if info["is_intervention"]:
            self.prev_action = info["action_intervention"]
        else:
            self.prev_action = action

        new_obs = self._obs(obs)

        return new_obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_action[:] = 0.0
        obs = self._obs(obs)
        return obs, info


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


if __name__ == "__main__":
    # print policy bounds
    cfg = URPegInHoleConfig()

    min_pose = cfg.robot.follower_arms["main"].min_pose
    max_pose = cfg.robot.follower_arms["main"].max_pose
    if cfg.use_xy_position:
        min_bounds = [-8.0] * 3 + min_pose[:3] + [-s for s in cfg.wrapper.spacemouse_action_scale["main"][:2]]
        max_bounds = [8.0] * 3 + max_pose[:3] + [s for s in cfg.wrapper.spacemouse_action_scale["main"][:2]]
    else:
        min_bounds = [-8.0] * 3 + [min_pose[2]] + [-s for s in cfg.wrapper.spacemouse_action_scale["main"][:2]]
        max_bounds = [8.0] * 3 + [max_pose[2]] + [s for s in cfg.wrapper.spacemouse_action_scale["main"][:2]]
    print("=== policy input normalization parameters")
    print("- observation.state")
    print("    min:", min_bounds)
    print("    max:", max_bounds)

