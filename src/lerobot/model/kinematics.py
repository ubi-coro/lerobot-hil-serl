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


class RobotKinematics:
    """Robot kinematics using placo library for forward and inverse kinematics."""

    def __init__(
        self,
        urdf_path: str,
        target_frame_name: str = "gripper_frame_link",
        joint_names: list[str] = None,
    ):
        """
        Initialize placo-based kinematics solver.

        Args:
            urdf_path: Path to the robot URDF file
            target_frame_name: Name of the end-effector frame in the URDF
            joint_names: List of joint names to use for the kinematics solver
        """
        try:
            import placo
        except ImportError as e:
            raise ImportError(
                "placo is required for RobotKinematics. "
                "Please install the optional dependencies of `kinematics` in the package."
            ) from e

        self.robot = placo.RobotWrapper(urdf_path)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)  # Fix the base

        self.target_frame_name = target_frame_name

        # Set joint names
        self.joint_names = list(self.robot.joint_names()) if joint_names is None else joint_names

        # Initialize frame task for IK
        self.tip_frame = self.solver.add_frame_task(self.target_frame_name, np.eye(4))

    def forward_kinematics(self, joint_pos_deg):
        """
        Compute forward kinematics for given joint configuration given the target frame name in the constructor.

        Args:
            joint_pos_deg: Joint positions in degrees (numpy array)

        Returns:
            4x4 transformation matrix of the end-effector pose
        """

        # Convert degrees to radians
        joint_pos_rad = np.deg2rad(joint_pos_deg[: len(self.joint_names)])

        # Update joint positions in placo robot
        for i, joint_name in enumerate(self.joint_names):
            self.robot.set_joint(joint_name, joint_pos_rad[i])

        # Update kinematics
        self.robot.update_kinematics()

        # Get the transformation matrix
        return self.robot.get_T_world_frame(self.target_frame_name)

    def inverse_kinematics(
        self, current_joint_pos, desired_ee_pose, position_weight=1.0, orientation_weight=0.01
    ):
        """
        Compute inverse kinematics using placo solver.

        Args:
            current_joint_pos: Current joint positions in degrees (used as initial guess)
            desired_ee_pose: Target end-effector pose as a 4x4 transformation matrix
            position_weight: Weight for position constraint in IK
            orientation_weight: Weight for orientation constraint in IK, set to 0.0 to only constrain position

        Returns:
            Joint positions in degrees that achieve the desired end-effector pose
        """

        # Convert current joint positions to radians for initial guess
        current_joint_rad = np.deg2rad(current_joint_pos[: len(self.joint_names)])

        # Set current joint positions as initial guess
        for i, joint_name in enumerate(self.joint_names):
            self.robot.set_joint(joint_name, current_joint_rad[i])

        # Update the target pose for the frame task
        self.tip_frame.T_world_frame = desired_ee_pose

        # Configure the task based on position_only flag
        self.tip_frame.configure(self.target_frame_name, "soft", position_weight, orientation_weight)

        # Solve IK
        self.solver.solve(True)
        self.robot.update_kinematics()

        # Extract joint positions
        joint_pos_rad = []
        for joint_name in self.joint_names:
            joint = self.robot.get_joint(joint_name)
            joint_pos_rad.append(joint)

        # Convert back to degrees
        joint_pos_deg = np.rad2deg(joint_pos_rad)

        # Preserve gripper position if present in current_joint_pos
        if len(current_joint_pos) > len(self.joint_names):
            result = np.zeros_like(current_joint_pos)
            result[: len(self.joint_names)] = joint_pos_deg
            result[len(self.joint_names) :] = current_joint_pos[len(self.joint_names) :]
            return result
        else:
            return joint_pos_deg


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

        assert robot_model in self.ROBOT_DESC, f"MRKinematics.__init__: Unkown robot_type {robot_type}"
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

    def fk_gripper(self, robot_pos_deg):
        """Forward kinematics for the gripper frame."""
        return mr.FKinSpace(
            self.gripper_desc["M"],
            self.gripper_desc["Slist"],
            self.apply_joint_correction(robot_pos_deg)
        )

    def fk_gripper_tip(self, robot_pos_deg):
        """Forward kinematics for the gripper tip frame."""
        return mr.FKinSpace(
            self.gripper_tip_desc["M"],
            self.gripper_tip_desc["Slist"],
            self.apply_joint_correction(robot_pos_deg)
        )

    def ik(self, current_joint_state, desired_ee_pose, position_only=True, gripper_pos=None, fk_func=None):
        if fk_func is None:
            fk_func = self.fk_gripper

        if gripper_pos is None and len(current_joint_state) > len(self.shadow_mask):
            gripper_pos = current_joint_state[len(self.shadow_mask)]

        if fk_func == self.fk_gripper:
            desc = self.gripper_desc
        elif fk_func == self.fk_gripper_tip:
            desc = self.gripper_tip_desc
        else:
            raise ValueError("MRKinematics.ik: Unknown fk_func")

        joint_states, success = mr.IKinSpace(
            Slist=desc["Slist"],
            M=desc["M"],
            T=desired_ee_pose,
            thetalist0=self.apply_joint_correction(current_joint_state),
            ev=1e-6,
            eomg=1e10 if position_only else 1e-6,
        )

        if success:
            joint_states = self.revert_joint_correction(joint_states)
            if gripper_pos is not None:
                joint_states = np.append(joint_states, gripper_pos)
        else:
            joint_states = current_joint_state
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
