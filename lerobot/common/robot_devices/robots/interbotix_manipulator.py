"""Contains logic to instantiate a robot, read information from its motors and cameras,
and send orders to its motors.
"""
# TODO(rcadene, aliberts): reorganize the codebase into one file per robot, with the associated
# calibration procedure, to make it easy for people to add their own robot.

import json
import logging
import time
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from lerobot.common.robot_devices.cameras.utils import Camera
from lerobot.common.robot_devices.motors.utils import MotorsBus
from lerobot.common.robot_devices.robots.utils import get_arm_id
from lerobot.common.robot_devices.utils import RobotDeviceNotConnectedError
from lerobot.common.robot_devices.motors.interbotix import InterbotixBus
from lerobot.common.robot_devices.robots.manipulator import ensure_safe_goal_position, ManipulatorRobot


@dataclass
class InterbotixManipulatorRobotConfig:
    """
    Example of usage:
    ```python
    InterbotixManipulatorRobotConfig()
    ```
    """

    # Define all components of the robot
    robot_type: str = "aloha"
    leader_arms: dict[str, MotorsBus] = field(default_factory=lambda: {})
    follower_arms: dict[str, MotorsBus] = field(default_factory=lambda: {})
    cameras: dict[str, Camera] = field(default_factory=lambda: {})
    botas: dict[str, object] = field(default_factory=lambda: {})

    # Optionally limit the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length
    # as the number of motors in your follower arms (assumes all follower arms have the same number of
    # motors).
    max_relative_target: list[float] | float | None = None

    # The duration of the velocity-based time profile
    # Higher values lead to smoother motions, but increase lag.
    # Only applicable to aloha
    moving_time: float = 0.1
    accel_time: float = 0.0

    # Optionally set the leader arm in torque mode with the gripper motor set to this angle. This makes it
    # possible to squeeze the gripper and have it spring back to an open position on its own. If None, the
    # gripper is not put in torque mode.
    gripper_open_degree: float | None = None

    abs_pose_limit_low: list[float] = field(default_factory=[-np.inf] * 7)
    abs_pose_limit_high: list[float] = field(default_factory=[np.inf] * 7)

    def __setattr__(self, prop: str, val):
        if (
            prop == "max_relative_target"
            and val is not None
            and isinstance(val, Sequence)
        ):
            for name in self.follower_arms:
                if len(self.follower_arms[name].motors) != len(val):
                    raise ValueError(
                        f"len(max_relative_target)={len(val)} but the follower arm with name {name} has "
                        f"{len(self.follower_arms[name].motors)} motors. Please make sure that the "
                        f"`max_relative_target` list has as many parameters as there are motors per arm. "
                        "Note: This feature does not yet work with robots where different follower arms have "
                        "different numbers of motors."
                    )
        if prop in ("abs_pose_limit_low", "abs_pose_limit_high") and val is not None:
            for key in val:
                val[key] = torch.tensor(val[key])
        super().__setattr__(prop, val)

    def __post_init__(self):
        if self.robot_type not in ["koch", "koch_bimanual", "aloha", "so100", "moss"]:
            raise ValueError(
                f"Provided robot type ({self.robot_type}) is not supported."
            )


class InterbotixManipulatorRobot(ManipulatorRobot):
    """
    same as ManipulatorRobot, but expects follower arms to each be a lerobot.common.robot_devices.motors.interbotix.InterbotixBus
    """
    # TODO(rcadene): Implement force feedback
    """This class allows to control any manipulator robot of various number of motors.

    Non exaustive list of robots:
    - [Koch v1.0](https://github.com/AlexanderKoch-Koch/low_cost_robot), with and without the wrist-to-elbow expansion, developed
    by Alexander Koch from [Tau Robotics](https://tau-robotics.com)
    - [Koch v1.1](https://github.com/jess-moss/koch-v1-1) developed by Jess Moss
    - [Aloha](https://www.trossenrobotics.com/aloha-kits) developed by Trossen Robotics

    Example of highest frequency teleoperation without camera:
    ```python
    # Defines how to communicate with the motors of the leader and follower arms
    leader_arms = {
        "main": DynamixelMotorsBus(
            port="/dev/tty.usbmodem575E0031751",
            motors={
                # name: (index, model)
                "shoulder_pan": (1, "xl330-m077"),
                "shoulder_lift": (2, "xl330-m077"),
                "elbow_flex": (3, "xl330-m077"),
                "wrist_flex": (4, "xl330-m077"),
                "wrist_roll": (5, "xl330-m077"),
                "gripper": (6, "xl330-m077"),
            },
        ),
    }
    follower_arms = {
        "main": DynamixelMotorsBus(
            port="/dev/tty.usbmodem575E0032081",
            motors={
                # name: (index, model)
                "shoulder_pan": (1, "xl430-w250"),
                "shoulder_lift": (2, "xl430-w250"),
                "elbow_flex": (3, "xl330-m288"),
                "wrist_flex": (4, "xl330-m288"),
                "wrist_roll": (5, "xl330-m288"),
                "gripper": (6, "xl330-m288"),
            },
        ),
    }
    robot = ManipulatorRobot(
        robot_type="koch",
        calibration_dir=".cache/calibration/koch",
        leader_arms=leader_arms,
        follower_arms=follower_arms,
    )

    # Connect motors buses and cameras if any (Required)
    robot.connect()

    while True:
        robot.teleop_step()
    ```

    Example of highest frequency data collection without camera:
    ```python
    # Assumes leader and follower arms have been instantiated already (see first example)
    robot = ManipulatorRobot(
        robot_type="koch",
        calibration_dir=".cache/calibration/koch",
        leader_arms=leader_arms,
        follower_arms=follower_arms,
    )
    robot.connect()
    while True:
        observation, action = robot.teleop_step(record_data=True)
    ```

    Example of highest frequency data collection with cameras:
    ```python
    # Defines how to communicate with 2 cameras connected to the computer.
    # Here, the webcam of the laptop and the phone (connected in USB to the laptop)
    # can be reached respectively using the camera indices 0 and 1. These indices can be
    # arbitrary. See the documentation of `OpenCVCamera` to find your own camera indices.
    cameras = {
        "laptop": OpenCVCamera(camera_index=0, fps=30, width=640, height=480),
        "phone": OpenCVCamera(camera_index=1, fps=30, width=640, height=480),
    }

    # Assumes leader and follower arms have been instantiated already (see first example)
    robot = ManipulatorRobot(
        robot_type="koch",
        calibration_dir=".cache/calibration/koch",
        leader_arms=leader_arms,
        follower_arms=follower_arms,
        cameras=cameras,
    )
    robot.connect()
    while True:
        observation, action = robot.teleop_step(record_data=True)
    ```

    Example of controlling the robot with a policy (without running multiple policies in parallel to ensure highest frequency):
    ```python
    # Assumes leader and follower arms + cameras have been instantiated already (see previous example)
    robot = ManipulatorRobot(
        robot_type="koch",
        calibration_dir=".cache/calibration/koch",
        leader_arms=leader_arms,
        follower_arms=follower_arms,
        cameras=cameras,
    )
    robot.connect()
    while True:
        # Uses the follower arms and cameras to capture an observation
        observation = robot.capture_observation()

        # Assumes a policy has been instantiated
        with torch.inference_mode():
            action = policy.select_action(observation)

        # Orders the robot to move
        robot.send_action(action)
    ```

    Example of disconnecting which is not mandatory since we disconnect when the object is deleted:
    ```python
    robot.disconnect()
    ```
    """
    def __init__(
        self,
        config: InterbotixManipulatorRobotConfig | None = None,
        calibration_dir: Path = ".cache/calibration/koch",
        **kwargs,
    ):
        if config is None:
            config = InterbotixManipulatorRobotConfig()
        # Overwrite config arguments using kwargs
        self.config = replace(config, **kwargs)
        self.calibration_dir = Path(calibration_dir)

        self.robot_type = self.config.robot_type
        self.leader_arms = self.config.leader_arms
        self.follower_arms = self.config.follower_arms
        self.cameras = self.config.cameras
        self.is_connected = False
        self.logs = {}
        self.previous_ik_solution: dict = {name: None for name in self.follower_arms}

    def get_motor_names(self, arm: dict[str, MotorsBus]) -> list:
        return [f"{arm}_{motor}" for arm, bus in arm.items() for motor in bus.motor_names]

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    @property
    def motor_features(self) -> dict:
        action_names = self.get_motor_names(self.leader_arms)
        state_names = self.get_motor_names(self.leader_arms)
        return {
            "action": {
                "dtype": "float32",
                "shape": (len(action_names),),
                "names": action_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_names),),
                "names": state_names,
            },
        }

    @property
    def features(self):
        return {**self.motor_features, **self.camera_features}

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    @property
    def available_arms(self):
        available_arms = []
        for name in self.follower_arms:
            arm_id = get_arm_id(name, "follower")
            available_arms.append(arm_id)
        for name in self.leader_arms:
            arm_id = get_arm_id(name, "leader")
            available_arms.append(arm_id)
        return available_arms

    def connect(self):
        if not all([isinstance(arm, InterbotixBus) for arm in list(self.follower_arms.values()) + list(self.leader_arms.values())]):
            raise ValueError(
                "All arms must be an instance of lerobot.common.robot_devices.motors.interbotix.InterbotixBus"
            )

        if not self.leader_arms and not self.follower_arms and not self.cameras:
            raise ValueError(
                "ManipulatorRobot doesn't have any device to connect. See example of usage in docstring of the class."
            )

        # We assume that at connection time, arms are in a rest position, and torque can
        # be safely disabled to run calibration and/or set robot preset configurations.
        for name in self.follower_arms:
            self.follower_arms[name].torque_off()
        for name in self.leader_arms:
            self.leader_arms[name].torque_off()

        self.activate_calibration()

        # Set robot preset (e.g. torque in leader gripper for Koch v1.1)
        self.set_aloha_robot_preset()

        # Enable torque on all motors of the follower arms
        for name in self.follower_arms:
            print(f"Activating torque on {name} follower arm.")
            self.follower_arms[name].torque_enable()

        if self.config.gripper_open_degree is not None:
            # Set the leader arm in torque mode with the gripper motor set to an angle. This makes it possible
            # to squeeze the gripper and have it spring back to an open position on its own.
            for name in self.leader_arms:
                self.leader_arms[name].bot.core.robot_torque_enable('single', 'gripper', True)
                self.leader_arms[name].set_gripper_position(self.config.gripper_open_degree)

        # Check both arms can be read
        for name in self.follower_arms:
            self.follower_arms[name].read("Present_Position")
        for name in self.leader_arms:
            self.leader_arms[name].read("Present_Position")

        # Connect the cameras
        for name in self.cameras:
            self.cameras[name].connect()

        self.is_connected = True

    def set_aloha_robot_preset(self):
        for name in self.leader_arms:
            core = self.leader_arms[name].bot.core
            core.robot_reboot_motors('single', 'gripper', True)
            core.robot_set_operating_modes('group', 'arm', 'position')
            core.robot_set_operating_modes('single', 'gripper', 'current_based_position')
            core.robot_set_motor_registers('group', 'arm', 'Position_P_Gain', 800)
            core.robot_set_motor_registers('group', 'arm', 'Position_I_Gain', 0)

        for name in self.leader_arms:
            core = self.leader_arms[name].bot.core
            core.robot_reboot_motors('single', 'gripper', True)
            core.robot_set_operating_modes('group', 'arm', 'position')
            core.robot_set_operating_modes('single', 'gripper', 'current_based_position')
            core.robot_set_motor_registers('group', 'arm', 'Position_P_Gain', 800)
            core.robot_set_motor_registers('group', 'arm', 'Position_I_Gain', 0)

    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        # Prepare to assign the position of the leader to the follower
        leader_pos = {}
        for name in self.leader_arms:
            before_lread_t = time.perf_counter()
            leader_pos[name] = self.leader_arms[name].get_joint_positions()
            leader_pos[name] = torch.from_numpy(leader_pos[name])
            self.logs[f"read_leader_{name}_pos_dt_s"] = (time.perf_counter() - before_lread_t)

        # Read follower position
        follower_ee_pose = {}
        for name in self.follower_arms:
            before_fread_t = time.perf_counter()
            follower_ee_pose[name] = self.follower_arms[name].get_ee_pose()
            follower_ee_pose[name] = torch.from_numpy(follower_ee_pose[name])
            self.logs[f"read_follower_{name}_pos_dt_s"] = (time.perf_counter() - before_fread_t)

        # Read follower velocity
        follower_ee_vel = {}
        for name in self.follower_arms:
            before_fread_t = time.perf_counter()
            follower_ee_vel[name] = self.follower_arms[name].get_ee_velocity()
            follower_ee_vel[name] = torch.from_numpy(follower_ee_vel[name])
            self.logs[f"read_follower_{name}_vel_dt_s"] = (time.perf_counter() - before_fread_t)

        # Send goal position to the follower
        follower_ee_goal_pose = {}
        for name in self.follower_arms:
            before_fwrite_t = time.perf_counter()
            goal_pos = leader_pos[name]
            goal_ee_pose = self.follower_arms[name].fk(goal_pos)  # leader joints -> follower ee pose

            # If specified, clip the goal positions within predefined bounds specified in the config of the robot
            if self.config.abs_pose_limit_low is not None and self.config.abs_pose_limit_high is not None:
                goal_ee_pose = torch.clamp(
                    goal_ee_pose,
                    self.config.abs_pose_limit_low,
                    self.config.abs_pose_limit_high
                )

            # Cap goal position when too far away from present position.
            # Slower fps expected due to reading from the follower.
            if self.config.max_relative_target is not None:
                goal_ee_pose = ensure_safe_goal_position(
                    goal_ee_pose, follower_ee_pose, self.config.max_relative_target
                )

            goal_ee_pose = goal_ee_pose.numpy().astype(np.int32)
            follower_ee_goal_pose[name] = goal_ee_pose
            ik_solution = self.follower_arms[name].set_ee_pose(goal_ee_pose, initial_guess=self.previous_ik_solution[name])
            self.previous_ik_solution[name] = ik_solution
            self.logs[f"write_follower_{name}_goal_pos_dt_s"] = (
                time.perf_counter() - before_fwrite_t
            )

        # Early exit when recording data is not requested
        if not record_data:
            return

        # Create action by concatenating follower goal position
        action_dict = {}
        action = []
        for name in self.follower_arms:
            if name in follower_ee_goal_pose:
                action.append(self.follower_ee_goal_pose[name])
        action = torch.cat(action)
        action_dict["action"] = action

        obs_dict = self.capture_observation(follower_ee_pose=follower_ee_pose, follower_ee_vel=follower_ee_vel)

        return obs_dict, action_dict

    def capture_observation(
            self,
            follower_ee_pose: dict[str, torch.Tensor] | None = None,
            follower_ee_vel: dict[str, torch.Tensor] | None = None
    ):
        """The returned observations do not have a batch dimension."""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        if follower_ee_pose is None:
            # Read follower position
            follower_ee_pose = {}
            for name in self.follower_arms:
                before_fread_t = time.perf_counter()
                follower_ee_pose[name] = self.follower_arms[name].get_ee_pose()
                follower_ee_pose[name] = torch.from_numpy(follower_ee_pose[name])
                self.logs[f"read_follower_{name}_pos_dt_s"] = (
                    time.perf_counter() - before_fread_t
                )

        if follower_ee_vel is None:
            # Read follower velocity
            follower_ee_vel = {}
            for name in self.follower_arms:
                before_fread_t = time.perf_counter()
                follower_ee_vel[name] = self.follower_arms[name].get_ee_velocity()
                follower_ee_vel[name] = torch.from_numpy(follower_ee_vel[name])
                self.logs[f"read_follower_{name}_vel_dt_s"] = (
                    time.perf_counter() - before_fread_t
                )

        # Create state by concatenating follower current position
        tcp_pose = []
        tcp_vel = []
        for name in self.follower_arms:
            if name in follower_ee_pose:
                tcp_pose.append(follower_ee_pose[name])
            if name in follower_ee_vel:
                tcp_vel.append(follower_ee_vel[name])
        tcp_pose = torch.cat(tcp_pose)
        tcp_vel = torch.cat(tcp_vel)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs[
                "delta_timestamp_s"
            ]
            self.logs[f"async_read_camera_{name}_dt_s"] = (
                time.perf_counter() - before_camread_t
            )

        # Populate output dictionnaries and format to pytorch
        obs_dict = {}
        obs_dict["observation.tcp_pose"] = tcp_pose
        obs_dict["observation.tcp_vel"] = tcp_vel
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]
        return obs_dict

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        """Command the follower arms to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Args:
            action: tensor containing the concatenated goal positions for the follower arms.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        from_idx = 0
        to_idx = 0
        action_sent = []
        for name in self.follower_arms:
            # Get goal position of each follower arm by splitting the action vector
            # x,y,z, r, p, y, gripper
            to_idx += 7
            goal_ee_pose = action[from_idx:to_idx]  # absolute, not delta values
            from_idx = to_idx

            # If specified, clip the goal positions within predefined bounds specified in the config of the robot
            if self.config.abs_pose_limit_low is not None and self.config.abs_pose_limit_high is not None:
                goal_ee_pose = torch.clamp(
                    goal_ee_pose,
                    self.config.abs_pose_limit_low,
                    self.config.abs_pose_limit_high
                )

            # Cap goal position when too far away from present position.
            # Slower fps expected due to reading from the follower.
            if self.config.max_relative_target is not None:
                goal_ee_pose = ensure_safe_goal_position(
                    goal_ee_pose, self.follower_arms[name].get_ee_pose(), self.config.max_relative_target
                )

            # Save tensor to concat and return
            action_sent.append(goal_ee_pose)

            # Send goal position to each follower
            goal_ee_pose = goal_ee_pose.numpy().astype(np.int32)

            ik_solution = self.follower_arms[name].set_ee_pose(goal_ee_pose, initial_guess=self.previous_ik_solution[name])
            self.previous_ik_solution[name] = ik_solution

        return torch.cat(action_sent)

    def print_logs(self):
        pass
        # TODO(aliberts): move robot-specific logs logic here

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()` before disconnecting."
            )

        for name in self.follower_arms:
            self.follower_arms[name].disconnect()

        for name in self.leader_arms:
            self.leader_arms[name].disconnect()

        for name in self.cameras:
            self.cameras[name].disconnect()

        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()

