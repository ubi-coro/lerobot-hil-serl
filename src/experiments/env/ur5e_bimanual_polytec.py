from dataclasses import dataclass
from typing import Literal, Any

import numpy as np
import torch
from pynput import keyboard
from scipy.spatial.transform import Rotation as R

from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.envs import TFRobotEnvConfig, TaskFrameEnv
from lerobot.envs.configs import EnvConfig, ImagePreprocessingConfig
from lerobot.envs.factory import RobotEnvInterface
from lerobot.processor import DataProcessorPipeline, ProcessorStepRegistry, ObservationProcessorStep, TransitionKey
from lerobot.robots.ur import URConfig
from lerobot.robots.ur.tf_controller import TaskFrameCommand, AxisMode
from lerobot.teleoperators import TeleopEvents
from lerobot.teleoperators.spacemouse import SpacemouseConfig
from lerobot.utils.constants import OBS_STATE, ACTION


@ProcessorStepRegistry.register("bimanual_gap_constraint_processor")
@dataclass
class BimanualGapConstraintObsProcessor(ObservationProcessorStep):
    """
    Crops and/or resizes image observations.

    This step iterates through all image keys in an observation dictionary and applies
    the specified transformations. It handles device placement, moving tensors to the
    CPU if necessary for operations not supported on certain accelerators like MPS.

    Attributes:
        crop_params_dict: A dictionary mapping image keys to cropping parameters
                          (top, left, height, width).
        resize_size: A tuple (height, width) to resize all images to.
    """

    ee_min_gap_m: float
    base_lr_offset_m: float
    base_lr_offset_axis: Literal["x", "y", "z"] | int
    env: TaskFrameEnv | None = None

    def __post_init__(self):
        if isinstance(self.base_lr_offset_axis, int):
            self._base_lr_offset_axis = ["x", "y", "z"][self.base_lr_offset_axis]
            self._base_lr_offset_idx = self.base_lr_offset_axis

        else:
            self._base_lr_offset_axis = self.base_lr_offset_axis
            self._base_lr_offset_idx = ["x", "y", "z"].index(self.base_lr_offset_axis)

        if self.env is not None:
            self._initial_bounds = self.env.task_frame["right"].min_pose_rpy

    def observation(self, observation: dict) -> dict:
        """
        Applies cropping and resizing to all images in the observation dictionary.

        Args:
            observation: The observation dictionary, potentially containing image tensors.

        Returns:
            A new observation dictionary with transformed images.
        """
        new_observation = dict(observation)

        # compute distance
        raw_obs = self.transition[TransitionKey.COMPLEMENTARY_DATA]["raw_observation"]
        left_pos = raw_obs[f"left.{self._base_lr_offset_axis}.ee_pos"]
        right_pos = raw_obs[f"right.{self._base_lr_offset_axis}.ee_pos"]
        dist = right_pos - left_pos + self.base_lr_offset_m

        # add distance to state
        state = observation[OBS_STATE]
        dist_t = torch.tensor([dist], dtype=state.dtype, device=state.device)
        new_observation[OBS_STATE] = observation[OBS_STATE] = torch.cat([state, dist_t])

        # move bounds
        new_min_bound = left_pos + self.ee_min_gap_m - self.base_lr_offset_m
        self.env.task_frame["right"].min_pose_rpy[self._base_lr_offset_idx] = new_min_bound

        return new_observation

    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the step for serialization.

        Returns:
            A dictionary with the crop parameters and resize dimensions.
        """
        return {
            "ee_min_gap_m": self.ee_min_gap_m,
            "base_lr_offset_m": self.base_lr_offset_m,
            "base_lr_offset_axis": self.base_lr_offset_axis,
        }

    def reset(self) -> None:
        self.env.task_frame["right"].min_pose_rpy = self._initial_bounds

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Updates the image feature shapes in the policy features dictionary if resizing is applied.

        Args:
            features: The policy features dictionary.

        Returns:
            The updated policy features dictionary with new image shapes.
        """
        state_ft = features[PipelineFeatureType.OBSERVATION][OBS_STATE]

        features[PipelineFeatureType.OBSERVATION][OBS_STATE] = PolicyFeature(
            type=state_ft.type, shape=(state_ft.shape[0] + 1,)
        )

        return features



@dataclass
@EnvConfig.register_subclass("ur5e_bimanual_polytec")
class UR5eBimanualPolytecEnvConfig(TFRobotEnvConfig):
    fps: int = 20

    # 7 dof problem
    # left arm: xyz translation + c rotation + gripper
    # right arm: xyz translation

    # workspace
    left_min_x_pos: float = 0.4590
    left_max_x_pos: float = 0.5523
    left_min_y_pos: float = -0.235
    left_max_y_pos: float = -0.1851
    left_min_z_pos: float = 0.2340
    left_max_z_pos: float = 0.25
    max_c_rot_deg: float = 30.0
    min_c_rot_deg: float = -5.0

    right_y_rot_deg: float = -30.0  # fixed position
    right_max_x_pos: float = -0.3760
    right_min_z_pos: float = 0.2275
    right_max_z_pos: float = 0.25
    right_min_y_pos: float = -0.235
    right_max_y_pos: float = -0.1851

    # velocities
    v_ee = 0.15
    omega_ee = 1.0
    v_gripper = 0.3

    # parameters for bimanual setup + collision avoidance
    # assumes offset in one axis only and prevents the left robot
    # from colliding with the right robot on that axis
    ee_min_gap_m: float = 0.02
    base_lr_offset_axis: Literal["x", "y", "z"] | int = "x"
    base_lr_offset_m: float = 1.0


    def __post_init__(self):
        self.processor.control_time_s = 20.0

        self.processor.observation.add_ee_velocity_to_observation = True
        self.processor.observation.add_ee_wrench_to_observation = True
        self.processor.observation.ee_pos_mask = {
            "left": [1, 1, 1, 0, 0, 1],
            "right": [1, 1, 1, 0, 0, 0],
        }

        # (top, left, height, width) crop
        #self.processor.image_preprocessing = ImagePreprocessingConfig(
        #    crop_params_dict={'observation.images.left_wrist': (220, 257, 170, 312)},
        #    resize_size=(128, 256)
        #)
        self.processor.image_preprocessing = None


        self.processor.gripper.use_gripper = {"left": True, "right": False}
        self.processor.gripper.max_pos = 1.0
        self.processor.gripper.min_pos = 0.8

        self.processor.reset.terminate_on_success = True
        self.processor.reset.teleop_on_reset = True
        self.processor.reset.reset_time_s = 10.0

        self.processor.events.key_mapping = {
            TeleopEvents.TERMINATE_EPISODE: keyboard.Key.right,
            TeleopEvents.RERECORD_EPISODE: keyboard.Key.left
        }

        self.processor.hooks.time_action_processor = False
        self.processor.hooks.time_env_processor = False
        self.processor.hooks.log_every = 1

        # calculate fixed target orientations for both arms
        left_rot = R.from_rotvec([np.pi / np.sqrt(2), np.pi / np.sqrt(2), 0.0], degrees=False)
        right_rot = R.from_rotvec([np.pi, 0.0, 0.0], degrees=False)
        right_rot *= R.from_euler('y', self.right_y_rot_deg, degrees=True)

        # calculate bounds
        deg2rad = lambda deg: deg / 180.0 * np.pi
        min_pose_rpy_left = [
            self.left_min_x_pos,
            self.left_min_y_pos,
            self.left_min_z_pos,
            -float("inf"),
            -float("inf"),
            deg2rad(self.min_c_rot_deg) + left_rot.as_euler("xyz", degrees=False)[2]
        ]

        max_pose_rpy_left = [
            self.left_max_x_pos,
            self.left_max_y_pos,
            self.left_max_z_pos,
            float("inf"),
            float("inf"),
            deg2rad(self.max_c_rot_deg) + left_rot.as_euler("xyz", degrees=False)[2]
        ]

        min_pose_rpy_right = [
            -float("inf"),
            self.right_min_y_pos,
            self.right_min_z_pos,
            -float("inf"),
            -float("inf"),
            -float("inf")
        ]

        max_pose_rpy_right = [
            self.right_max_x_pos,
            self.right_max_y_pos,
            self.right_max_z_pos,
            float("inf"),
            float("inf"),
            float("inf")
        ]

        # device configs
        self.robot = {
            "left": URConfig(
                model="ur5e",
                robot_ip="192.168.1.10",
                use_gripper=True,
                gripper_soft_real_time=True,
                gripper_rt_core=5,
                gripper_vel=self.v_gripper,
                soft_real_time=True,
                rt_core=3,
                verbose=True,
                wrench_limits=[300.0, 300.0, 300.0, 20.0, 20.0, 20.0]
            ),
            "right": URConfig(
                model="ur5e",
                robot_ip="192.168.1.11",
                use_gripper=False,
                soft_real_time=True,
                rt_core=4,
                verbose=True,
                wrench_limits=[300.0, 300.0, 300.0, 20.0, 20.0, 20.0]
            ),
        }

        action_scale = 3 * [self.v_ee] + 3 * [self.omega_ee]
        self.teleop = {
            "left": SpacemouseConfig(
                path="/dev/hidraw0",
                action_scale=action_scale,
                gripper_close_button_idx=0,
                gripper_open_button_idx=1
            ),
            "right": SpacemouseConfig(
                path="/dev/hidraw7",
                action_scale=action_scale,
                button_mapping={
                    0: {
                        "event": TeleopEvents.IS_INTERVENTION,
                        "toggle": True
                    },
                    1: {
                        "event": TeleopEvents.SUCCESS,
                        "toggle": False
                    },
                }
            )
        }
        self.cameras = {
            "left_wrist": RealSenseCameraConfig(
                serial_number_or_name="218622271373",
                fps=30,
                width=640,
                height=480
            )
        }
        self.processor.events.key_mapping = {
            TeleopEvents.STOP_RECORDING: keyboard.Key.down
        }

        # task frame configuration
        self.processor.task_frame.command = {
            "left": TaskFrameCommand(
                T_WF=[0.0] * 6,
                target=3 * [0.0] + left_rot.as_rotvec().tolist(),
                mode=3 * [AxisMode.PURE_VEL] + 2 * [AxisMode.POS] + [AxisMode.PURE_VEL],
                kp=[5000, 5000, 5000, 100, 100, 100],
                kd=[480, 480, 480, 6, 6, 6],
                max_pose_rpy=max_pose_rpy_left,
                min_pose_rpy=min_pose_rpy_left
            ),
            "right": TaskFrameCommand(
                T_WF=[0.0] * 6,
                target=3 * [0.0] + right_rot.as_rotvec().tolist(),
                mode=3 * [AxisMode.PURE_VEL] + 3 * [AxisMode.POS],
                kp=[5000, 5000, 5000, 100, 100, 100],
                kd=[480, 480, 480, 6, 6, 6],
                max_pose_rpy=max_pose_rpy_right,
                min_pose_rpy=min_pose_rpy_right
            )
        }

        # 1 if mode[i] != POS
        self.processor.task_frame.control_mask = {
            "left": [1, 1, 1, 0, 0, 1],
            "right": [1, 1, 1, 0, 0, 0]
        }

        # Hard-code stats for online learning without offline datasets
        self.stats = {
            ACTION: {
                "min": [-1] * 8,
                "max": [1] * 8
            }
        }

        super().__post_init__()

    def make_env_processor(self, device, env: RobotEnvInterface | None = None) -> DataProcessorPipeline:
        env_processor = super().make_env_processor(device=device, env=env)
        env_processor.steps.insert(1, BimanualGapConstraintObsProcessor(
            env=env,
            ee_min_gap_m=self.ee_min_gap_m,
            base_lr_offset_m=self.base_lr_offset_m,
            base_lr_offset_axis=self.base_lr_offset_axis
        ))
        return env_processor


@dataclass
@EnvConfig.register_subclass("ur5e_single_polytec")
class UR5eSinglePolytecEnvConfig(TFRobotEnvConfig):
    # 5 dof problem
    # xyz translation + c rotation + gripper

    # obs space is 17 dimensional
    # 0.  x/pos
    # 1.  x/vel
    # 2.  x/wrench
    # 3.  y/pos
    # 4.  y/vel
    # 5.  y/wrench
    # 6.  z/pos
    # 7.  z/vel
    # 8.  z/wrench
    # 9.  a/vel
    # 10. a/wrench
    # 11. b/vel
    # 12. b/wrench
    # 13. c/pos
    # 14. c/vel
    # 15. c/wrench
    # 16. gripper/pos

    fps: int = 20

    # workspace
    left_min_x_pos: float = 0.4590
    left_max_x_pos: float = 0.5523
    left_min_y_pos: float = -0.235
    left_max_y_pos: float = -0.1851
    left_min_z_pos: float = 0.2340
    left_max_z_pos: float = 0.25
    max_c_rot_deg: float = 30.0
    min_c_rot_deg: float = -5.0

    # action scaling
    v_ee = 0.15
    omega_ee = 1.5
    v_gripper = 0.3

    def __post_init__(self):
        self.processor.control_time_s = 45.0

        self.processor.observation.add_ee_velocity_to_observation = True
        self.processor.observation.add_ee_wrench_to_observation = True
        self.processor.observation.ee_pos_mask = [1, 1, 1, 0, 0, 1]

        # (top, left, height, width) crop
        #self.processor.image_preprocessing = ImagePreprocessingConfig(
        #    crop_params_dict={'observation.images.left_wrist': (220, 257, 170, 312)},
        #    resize_size=(128, 256)
        #)
        self.processor.image_preprocessing = None

        self.processor.gripper.use_gripper = True
        self.processor.gripper.max_pos = 1.0
        self.processor.gripper.min_pos = 0.8

        self.processor.reset.terminate_on_success = True
        self.processor.reset.teleop_on_reset = True
        self.processor.reset.reset_time_s = 10.0

        self.processor.events.key_mapping = {
            TeleopEvents.SUCCESS: keyboard.Key.right,
            TeleopEvents.RERECORD_EPISODE: keyboard.Key.left,
            TeleopEvents.STOP_RECORDING: keyboard.Key.down,
            TeleopEvents.IS_INTERVENTION: keyboard.Key.up
        }

        self.processor.hooks.time_action_processor = False
        self.processor.hooks.time_env_processor = False
        self.processor.hooks.log_every = 1

        # calculate fixed target orientations for both arms
        left_rot = R.from_rotvec([np.pi / np.sqrt(2), np.pi / np.sqrt(2), 0.0], degrees=False)

        # calculate bounds
        deg2rad = lambda deg: deg / 180.0 * np.pi
        min_pose_rpy_left = [
            self.left_min_x_pos,
            self.left_min_y_pos,
            self.left_min_z_pos,
            -float("inf"),
            -float("inf"),
            deg2rad(self.min_c_rot_deg) + left_rot.as_euler("xyz", degrees=False)[2]
        ]

        max_pose_rpy_left = [
            self.left_max_x_pos,
            self.left_max_y_pos,
            self.left_max_z_pos,
            float("inf"),
            float("inf"),
            deg2rad(self.max_c_rot_deg) + left_rot.as_euler("xyz", degrees=False)[2]
        ]

        # device configs
        self.robot = {
            "left": URConfig(
                model="ur5e",
                robot_ip="192.168.1.10",
                use_gripper=True,
                gripper_soft_real_time=True,
                gripper_rt_core=5,
                gripper_vel=self.v_gripper,
                soft_real_time=True,
                rt_core=3,
                verbose=True,
                wrench_limits=[300.0, 300.0, 300.0, 20.0, 20.0, 20.0]
            )
        }
        self.teleop = {
            "left": SpacemouseConfig(
                path="/dev/hidraw0",
                action_scale=3 * [self.v_ee] + 3 * [self.omega_ee],
                gripper_close_button_idx=0,
                gripper_open_button_idx=1
            ),
        }
        self.cameras = {
            "left_wrist": RealSenseCameraConfig(
                serial_number_or_name="218622271373",
                fps=30,
                width=640,
                height=480
            )
        }
        self.processor.events.foot_switch_mapping = {
            (TeleopEvents.IS_INTERVENTION,): {"device": 21, "toggle": True},
        }


        # task frame configuration
        self.processor.task_frame.command = {
            "left": TaskFrameCommand(
                T_WF=[0.0] * 6,
                target=3 * [0.0] + left_rot.as_rotvec().tolist(),
                mode=3 * [AxisMode.PURE_VEL] + 2 * [AxisMode.POS] + [AxisMode.PURE_VEL],
                kp=[5000, 5000, 5000, 100, 100, 100],
                kd=[480, 480, 480, 6, 6, 6],
                max_pose_rpy=max_pose_rpy_left,
                min_pose_rpy=min_pose_rpy_left
            )
        }
        # 1 if mode[i] != POS
        self.processor.task_frame.control_mask = [1, 1, 1, 0, 0, 1]
        self.processor.task_frame.action_scale = 3 * [self.v_ee] + [self.omega_ee] + [1.0]

        # Hard-code stats for online learning without offline datasets
        self.stats = {
            ACTION: {
                "min": [-1] * 5,
                "max": [1] * 5
            }
        }

        super().__post_init__()



