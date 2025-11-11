from dataclasses import dataclass
from typing import Literal, Any

import numpy as np
import torch
from pynput import keyboard
from scipy.spatial.transform import Rotation as R

from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.envs import TFRobotEnvConfig, TaskFrameEnv
from lerobot.envs.configs import EnvConfig
from lerobot.envs.factory import RobotEnvInterface
from lerobot.processor import DataProcessorPipeline, ProcessorStepRegistry, ObservationProcessorStep, TransitionKey
from lerobot.robots.ur import URConfig
from lerobot.robots.ur.tf_controller import TaskFrameCommand, AxisMode
from lerobot.teleoperators import TeleopEvents
from lerobot.teleoperators.spacemouse import SpacemouseConfig
from lerobot.utils.constants import OBS_STATE


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

        # add to state
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

    left_x_rot_deg: float = 0.0
    left_max_x_pos: float = 0.593
    left_min_z_pos: float = 0.221
    left_max_z_pos: float = 0.3

    right_x_rot_deg: float = -15.0
    right_min_x_pos: float = -0.386
    right_min_z_pos: float = 0.22546
    right_max_z_pos: float = 0.3

    v_ee = 0.25
    omega_ee = 1.0
    v_gripper = 0.3
    max_gripper_pos: float = 0.2

    ee_min_gap_m: float = 0.05
    base_lr_offset_m: float = 1.0
    base_lr_offset_axis: Literal["x", "y", "z"] | int = "x"


    def __post_init__(self):

        # calculate fixed target poses for both arms
        left_rot = R.from_rotvec([np.pi / np.sqrt(2), np.pi / np.sqrt(2), 0.0], degrees=False) * R.from_euler('x', self.left_x_rot_deg, degrees=True)
        left_target = 3 * [0.0] + left_rot.as_rotvec().tolist()

        right_rot = R.from_rotvec([np.pi / np.sqrt(2), -np.pi / np.sqrt(2), 0.0], degrees=False) * R.from_euler('x', self.right_x_rot_deg, degrees=True)
        right_target = 3 * [0.0] + right_rot.as_rotvec().tolist()

        max_pose_rpy_left = np.full(6, np.inf)
        min_pose_rpy_left = np.full(6, -np.inf)
        max_pose_rpy_left[0] = self.left_max_x_pos
        max_pose_rpy_left[2] = self.left_max_z_pos
        min_pose_rpy_left[2] = self.left_min_z_pos

        max_pose_rpy_right = np.full(6, np.inf)
        min_pose_rpy_right = np.full(6, -np.inf)
        max_pose_rpy_right[2] = self.right_max_z_pos
        min_pose_rpy_right[0] = self.right_min_x_pos
        min_pose_rpy_right[2] = self.right_min_z_pos

        self.robot = {
            "left": URConfig(
                model="ur5e",
                robot_ip="192.168.1.10",
                use_gripper=False,
                gripper_soft_real_time=False,
                gripper_vel=self.v_gripper,
                soft_real_time=True,
                rt_core=3,
                verbose=True,
                wrench_limits=[100.0, 100.0, 100.0, 20.0, 20.0, 20.0]
            ),
            "right": URConfig(
                model="ur5e",
                robot_ip="192.168.1.11",
                use_gripper=False,
                gripper_soft_real_time=False,
                gripper_vel=self.v_gripper,
                soft_real_time=True,
                rt_core=4,
                verbose=True,
                wrench_limits=[300.0, 300.0, 300.0, 20.0, 20.0, 20.0]
            ),
        }
        self.teleop = {
            "left": SpacemouseConfig(path="/dev/hidraw2"),
            "right": SpacemouseConfig(path="/dev/hidraw5")
        }
        self.cameras = {
            "left_wrist": RealSenseCameraConfig(
                serial_number_or_name="218622271373",
                fps=30,
                width=640,
                height=480
            )
        }

        self.processor.task_frame.command = {
            "left": TaskFrameCommand(
                T_WF=[0.0] * 6,
                target=left_target,
                mode=3 * [AxisMode.PURE_VEL] + 2 * [AxisMode.POS] + [AxisMode.PURE_VEL],
                kp=[2500, 2500, 2500, 100, 100, 100],
                kd=[480, 480, 480, 6, 6, 6],
                max_pose_rpy=max_pose_rpy_left.tolist(),
                min_pose_rpy=min_pose_rpy_left.tolist()
            ),
            "right": TaskFrameCommand(
                T_WF=[0.0] * 6,
                target=right_target,
                mode=3 * [AxisMode.PURE_VEL] + 3 * [AxisMode.POS],
                kp=[2500, 2500, 2500, 100, 100, 100],
                kd=[480, 480, 480, 6, 6, 6],
                max_pose_rpy=max_pose_rpy_right.tolist(),
                min_pose_rpy=min_pose_rpy_right.tolist()
            )
        }
        self.processor.task_frame.control_mask = {
            "left": 3 * [1] + 2 * [0] + [1],
            "right": 3 * [1] + 3 * [0]
        }

        #self.processor.task_frame.action_scale = 3 * [self.v_ee] + [self.omega_ee] + [self.max_gripper_pos] + 3 * [self.v_ee]
        self.processor.task_frame.action_scale = 3 * [self.v_ee] + [self.omega_ee] + 3 * [self.v_ee]
        self.processor.hooks.time_action_processor = False
        self.processor.hooks.time_env_processor = False
        self.processor.hooks.log_every = 1
        self.processor.gripper.use_gripper = {"left": False, "right": False}
        self.processor.gripper.offset = 1 - self.max_gripper_pos
        self.processor.control_time_s = 3600.0  # 1h for debugging
        self.processor.reset.terminate_on_success = True
        self.processor.reset.teleop_on_reset = False
        self.processor.reset.reset_time_s = 5.0
        self.processor.events.key_mapping = {
            TeleopEvents.TERMINATE_EPISODE: keyboard.Key.right
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



