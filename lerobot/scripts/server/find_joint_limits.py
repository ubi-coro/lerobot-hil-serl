import argparse
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from lerobot.common.robot_devices.control_utils import is_headless
from lerobot.common.robot_devices.robots.configs import RobotConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.configs import parser
from lerobot.scripts.server.kinematics import RobotKinematics, MRKinematics

follower_port = "/dev/tty.usbmodem58760431631"
leader_port = "/dev/tty.usbmodem58760433331"


def find_joint_bounds(
    robot,
    control_time_s=30,
    display_cameras=False,
):
    if not robot.is_connected:
        robot.connect()

    start_episode_t = time.perf_counter()
    pos_list = []
    while True:
        observation, action = robot.teleop_step(record_data=True)

        # Wait for 5 seconds to stabilize the robot initial position
        if time.perf_counter() - start_episode_t < 5:
            continue

        pos_list.append(robot.follower_arms["main"].read("Present_Position"))

        if display_cameras and not is_headless():
            image_keys = [key for key in observation if "image" in key]
            for key in image_keys:
                cv2.imshow(
                    key, cv2.cvtColor(observation[key].numpy(), cv2.COLOR_RGB2BGR)
                )
            cv2.waitKey(1)

        if time.perf_counter() - start_episode_t > control_time_s:
            max = np.max(np.stack(pos_list), 0)
            min = np.min(np.stack(pos_list), 0)
            print(f"Max angle position per joint {max}")
            print(f"Min angle position per joint {min}")
            break


def find_ee_bounds(
    robot,
    control_time_s=30,
    display_cameras=False,
):
    if not robot.is_connected:
        robot.connect()

    if robot.config.robot_type.lower().startswith('aloha'):
        kinematics = MRKinematics(robot.config.follower_model)
    else:
        robot_type = getattr(robot.config, "robot_type", "so100")
        kinematics = RobotKinematics(robot_type)

    start_episode_t = time.perf_counter()
    ee_list = []
    while True:
        observation, action = robot.teleop_step(record_data=True)

        # Wait for 5 seconds to stabilize the robot initial position
        if time.perf_counter() - start_episode_t < 5:
            continue

        joint_positions = robot.follower_arms["main"].read("Present_Position")
        print(f"Joint positions: {joint_positions}")

        ee_pose = kinematics.fk_gripper_tip(joint_positions)
        xyz = ee_pose[:3, 3]
        rpy = R.from_matrix(ee_pose[:3, :3]).as_euler("xyz")
        ee_list.append(np.concatenate([xyz, rpy]))

        if display_cameras and not is_headless():
            image_keys = [key for key in observation if "image" in key]
            for key in image_keys:
                cv2.imshow(
                    key, cv2.cvtColor(observation[key].numpy(), cv2.COLOR_RGB2BGR)
                )
            cv2.waitKey(1)

        if time.perf_counter() - start_episode_t > control_time_s:
            max = np.max(np.stack(ee_list), 0)
            min = np.min(np.stack(ee_list), 0)
            print(f"Max ee position {max}")
            print(f"Min ee position {min}")
            break


def make_robot(robot_type="so100"):
    """
    Create a robot instance using the appropriate robot config class.

    Args:
        robot_type: Robot type string (e.g., "so100", "koch", "aloha")

    Returns:
        Robot instance
    """

    # Get the appropriate robot config class based on robot_type
    robot_config = RobotConfig.get_choice_class(robot_type)(mock=False)
    robot_config.leader_arms["main"].port = leader_port
    robot_config.follower_arms["main"].port = follower_port

    return make_robot_from_config(robot_config)


if __name__ == "__main__":
    # Create argparse for script-specific arguments
    parser = argparse.ArgumentParser(add_help=False)  # Set add_help=False to avoid conflict
    parser.add_argument(
        "--mode",
        type=str,
        default="joint",
        choices=["joint", "ee"],
        help="Mode to run the script in. Can be 'joint' or 'ee'.",
    )
    parser.add_argument(
        "--control-time-s",
        type=int,
        default=30,
        help="Time step to use for control.",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="so100",
        help="Robot type (so100, koch, aloha, etc.)",
    )

    # Only parse known args, leaving robot config args for Hydra if used
    args = parser.parse_args()

    # Create robot with the appropriate config
    robot = make_robot(args.robot_type)

    if args.mode == "joint":
        find_joint_bounds(robot, args.control_time_s)
    elif args.mode == "ee":
        find_ee_bounds(robot, args.control_time_s)

    if robot.is_connected:
        robot.disconnect()
