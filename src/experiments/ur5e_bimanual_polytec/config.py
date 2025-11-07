from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

from lerobot.envs.configs import EnvConfig, TFHilSerlRobotEnvConfig
from lerobot.robots.ur import URConfig
from lerobot.robots.ur.tff_controller import TaskFrameCommand, AxisMode
from lerobot.share.configs import DatasetRecordConfig
from lerobot.teleoperators.spacemouse import SpacemouseConfig


@dataclass
@EnvConfig.register_subclass("ur5e_bimanual_polytec")
class UR5eBimanualPolytecEnvConfig(TFHilSerlRobotEnvConfig):

    left_x_rot_deg: float = -5.0
    left_max_x_pos: float = 0.593
    left_min_z_pos: float = 0.221
    left_max_z_pos: float = 0.3

    right_x_rot_deg: float = -12.0
    right_min_x_pos: float = -0.386
    right_min_z_pos: float = 0.22546
    right_max_z_pos: float = 0.3

    ee_vel = 0.1
    max_gripper_pos: float = 0.2
    gripper_vel: float = 0.3




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
                use_gripper=True,
                gripper_soft_real_time=False,
                gripper_vel=self.gripper_vel,
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
                gripper_vel=self.gripper_vel,
                soft_real_time=True,
                rt_core=4,
                verbose=True,
                wrench_limits=[300.0, 300.0, 300.0, 20.0, 20.0, 20.0]
            ),
        }
        self.teleop = {
            "left": SpacemouseConfig(path="/dev/hidraw3"),
            "right": SpacemouseConfig(path="/dev/hidraw6")
        }

        self.processor.task_frame.command = {
            "left": TaskFrameCommand(
                T_WF=[0.0] * 6,
                target=left_target,
                mode=3 * [AxisMode.PURE_VEL] + 3 * [AxisMode.POS],
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
            "left": 3 * [1] + 3 * [0],
            "right": 3 * [1] + 3 * [0]
        }

        self.processor.task_frame.action_scale = 3 * [self.ee_vel] + [self.max_gripper_pos] + 3 * [self.ee_vel]

        self.processor.hooks.time_action_processor = False
        self.processor.hooks.time_env_processor = False
        self.processor.hooks.log_every = 1
        self.processor.gripper.use_gripper = {"left": True, "right": False}
        self.processor.gripper.offset = 1 - self.max_gripper_pos
        self.processor.reset.terminate_on_success = True
        self.processor.reset.teleop_on_reset = True
        self.processor.reset.reset_time_s = 5.0

        super().__post_init__()


@dataclass
@DatasetRecordConfig.register_subclass("ur5e_single")
class UR5eSingleDatasetConfigV2(DatasetRecordConfig):
    repo_id: str = "test/ur5e_single"
    single_task: str = "test"
    root: str = "/media/nvme1/jstranghoener/lerobot/data/test/ur5e_single"

