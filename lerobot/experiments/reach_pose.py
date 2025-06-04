from dataclasses import dataclass, field

import gymnasium as gym

from lerobot.common.constants import ACTION, OBS_ENV, OBS_IMAGE, OBS_IMAGES, OBS_ROBOT
from lerobot.common.envs.configs import (
    EnvConfig,
    HILSerlRobotEnvConfig,
    EnvWrapperConfig,
    EEActionSpaceConfig
)
from lerobot.common.policies.sac.configuration_sac import SACConfig
from lerobot.common.robot_devices.robots.configs import AlohaRobotConfig, RobotConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature


class TargetPoseWrapper(gym.Wrapper):
    def __init__(self, env, target_pose):
        super().__init__(env)
        self.target_pose = target_pose

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # pose is the
        ee_pose = obs["observation.state"][-3:].cpu().numpy()

        reward = float(-((ee_pose - self.target_pose)**2).sum())
        print("Reward:", reward)

        return obs, reward, terminated, truncated, info


@EnvConfig.register_subclass("real_reach_pose")
@dataclass
class RealReachPoseEnvConfig(HILSerlRobotEnvConfig):
    repo_id: str = "jannick-st/reach-pose-offline-demos"
    dataset_root: str = "/media/nvme1/jstranghoener/lerobot/data/jannick-st/reach-pose/offline-demos"
    target_ee_pos: list[float] = field(default_factory=lambda: [0.27, 0.0, 0.15          ])

    robot: RobotConfig = AlohaRobotConfig(
        cameras=dict(),
        calibration_dir="/home/jstranghoener/PycharmProjects/lerobot-hil-serl/.cache/calibration/aloha_default"
    )
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(3,)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(15,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "observation.state": OBS_ROBOT,
        }
    )
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
                "min": [0.16, -0.15, 0.08]
            },
            control_mode="leader"
        )
    )
    task: str = "Reach the target pose"
    num_episodes: int = 40  # only for record mode
    episode: int = 0
    device: str = "cuda"
    push_to_hub: bool = False
    fps: int = 10

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

        env = TargetPoseWrapper(env, self.target_ee_pos)

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


@PreTrainedConfig.register_subclass("sac_real_reach_pose")
@dataclass
class SACRealReachPoseConfig(SACConfig):
    online_step_before_learning: int = 300
    camera_number: int = 0
    utd_ratio: int = 3
    storage_device: str = "cpu"
    shared_encoder: bool = False
    num_critics: int = 4
    dataset_stats: dict[str, dict[str, list[float]]] | None = field(
        default_factory=lambda: {
            "observation.state": {
                "min": [-23.115234, -71.71875, -71.80664, 25.136719, 25.3125, -20,
                        -48.95508, -55.283203, 15.317919, 0.16, -0.15, 0.08, -0.02, -0.02, -0.02],
                "max": [49.04297, -15.644531, -15.46875, 78.57422, 78.66211, 20,
                        48.95508, 30.058594, 15.510597, 0.32, 0.15, 0.25, 0.02, 0.02, 0.02],
            },
            "action": {
                "min": [-0.02, -0.02, -0.02],
                "max": [0.02, 0.02, 0.02],
            },
        }
    )
