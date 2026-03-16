# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging

import numpy as np
import modern_robotics as mr
from lerobot.model.kinematics import RobotKinematics


def get_kinematics(
        robot_name: str,
        urdf_path: str,
        target_frame_name: str = "gripper_frame_link",
        joint_names: list[str] | None = None
    ) -> RobotKinematics:
    if robot_name in MRKinematics.ROBOT_DESC:
        return MRKinematics(robot_name)
    else:
        return RobotKinematics(urdf_path, target_frame_name, joint_names)


class MRKinematics(RobotKinematics):
    ROBOT_DESC = {
        "vx300s": {
            "M": np.array([[1.0, 0.0, 0.0, 0.536494],
                           [0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.42705],
                           [0.0, 0.0, 0.0, 1.0]]),
            "Slist": np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0, -0.12705, 0.0, 0.0],
                               [0.0, 1.0, 0.0, -0.42705, 0.0, 0.05955],
                               [1.0, 0.0, 0.0, 0.0, 0.42705, 0.0],
                               [0.0, 1.0, 0.0, -0.42705, 0.0, 0.35955],
                               [1.0, 0.0, 0.0, 0.0, 0.42705, 0.0]]).T
        },
        "wx250s": {
            "M": np.array([[1.0, 0.0, 0.0, 0.458325],
                           [0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.36065],
                           [0.0, 0.0, 0.0, 1.0]]),
            "Slist": np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0, -0.11065, 0.0, 0.0],
                               [0.0, 1.0, 0.0, -0.36065, 0.0, 0.04975],
                               [1.0, 0.0, 0.0, 0.0, 0.36065, 0.0],
                               [0.0, 1.0, 0.0, -0.36065, 0.0, 0.29975],
                               [1.0, 0.0, 0.0, 0.0, 0.36065, 0.0]]).T
        },
        "vx300s-bota": {
            "M": np.array([[1.0, 0.0, 0.0, 0.576694],  # +40.2 mm due to bota extension
                           [0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.42705],
                           [0.0, 0.0, 0.0, 1.0]]),
            "Slist": np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0, -0.12705, 0.0, 0.0],
                               [0.0, 1.0, 0.0, -0.42705, 0.0, 0.05955],
                               [1.0, 0.0, 0.0, 0.0, 0.42705, 0.0],
                               [0.0, 1.0, 0.0, -0.42705, 0.0, 0.35955],
                               [1.0, 0.0, 0.0, 0.0, 0.42705, 0.0]]).T  # never forget this tranpose
        }
    }

    def __init__(self, robot_model):
        # can also be initialized via "robot_type", which defaults to the standard aloha follower
        if robot_model == "aloha":
            robot_model = "vx300s"

        assert robot_model in self.ROBOT_DESC, f"MRKinematics.__init__: Unkown robot_type {robot_model}"
        self.gripper_desc = self.ROBOT_DESC[robot_model]
        self.shadow_mask = np.array([0, 1, 0, 1, 0, 0, 0, 0]).astype(bool)

        if robot_model + '-tip' in self.ROBOT_DESC:
            self.gripper_tip_desc = self.ROBOT_DESC[robot_model + '-tip']
        else:
            self.gripper_tip_desc = self.ROBOT_DESC[robot_model]

    def apply_joint_correction(self, robot_pos_deg):
        # filter shadows and gripper
        robot_pos_deg = robot_pos_deg[:len(self.shadow_mask)]
        robot_pos_deg = robot_pos_deg[~self.shadow_mask]

        # modern_robotics fk needs radians
        rotated_pos_rad = robot_pos_deg / 180.0 * np.pi

        return rotated_pos_rad

    def revert_joint_correction(self, rotated_pos_rad):
        """
        Inverts apply_joint_correction by reinserting shadow joints using the
        next real joint's value, except the last (gripper) which remains 0.

        Args:
            rotated_pos_rad (np.ndarray): Filtered joint array (radians), as output by apply_joint_correction.

        Returns:
            robot_pos_deg (np.ndarray): Full joint array with shadows (and gripper) in degrees.
        """
        # Convert radians to degrees
        robot_pos_deg_filtered = rotated_pos_rad * 180.0 / np.pi

        # Initialize full array without gripper
        full_length = len(self.shadow_mask)
        robot_pos_deg_full = np.zeros(full_length, dtype=np.float32)

        # Fill real (non-shadow) joint values
        real_indices = np.where(~self.shadow_mask)[0]
        robot_pos_deg_full[real_indices] = robot_pos_deg_filtered

        # Fill shadows from the *next* real joint
        for i in range(full_length):
            if self.shadow_mask[i]:
                robot_pos_deg_full[i] = robot_pos_deg_full[i + 1]

        return robot_pos_deg_full

    def forward_kinematics(self, joint_pos_deg):
        """Forward kinematics for the gripper frame."""
        return mr.FKinSpace(
            self.gripper_desc["M"],
            self.gripper_desc["Slist"],
            self.apply_joint_correction(joint_pos_deg)
        )

    def inverse_kinematics(
        self,
        current_joint_pos: np.ndarray,
        desired_ee_pose: np.ndarray,
        position_weight: float = 1e-6,
        orientation_weight: float = 1e-6
    ):
        joint_states, success = mr.IKinSpace(
            Slist=self.gripper_desc["Slist"],
            M=self.gripper_desc["M"],
            T=desired_ee_pose,
            thetalist0=self.apply_joint_correction(current_joint_pos),
            ev=position_weight,
            eomg=orientation_weight,
        )

        if success:
            joint_states = self.revert_joint_correction(joint_states)
        else:
            joint_states = current_joint_pos
            logging.info('No valid pose could be found. Will return current position')

        return joint_states

    def compute_jacobian(self, current_joint_state, fk_func=None):
        if fk_func is None:
            fk_func = self.fk_gripper

        if fk_func == self.fk_gripper:
            desc = self.gripper_desc
        elif fk_func == self.fk_gripper_tip:
            desc = self.gripper_tip_desc
        else:
            raise ValueError("MRKinematics.ik: Unknown fk_func")

        return mr.JacobianSpace(Slist=desc["Slist"], thetalist=self.apply_joint_correction(current_joint_state))
