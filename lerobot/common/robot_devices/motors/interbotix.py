import modern_robotics as mr
import numpy as np
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_xs_modules.xs_robot.gravity_compensation import InterbotixGravityCompensationInterface

from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
from lerobot.common.robot_devices.motors.utils import MotorsBus

INTERBOTIX_MOTORMODELS = {
    'vx300s': [
        'xm540-w270',
        'xm540-w270',
        'xm540-w270',
        'xm540-w270',
        'xm540-w270',
        'xm540-w270',
        'xm540-w270',
        'xm430-w350',
        'xm430-w350',

    ],
    'wx250s': [
        'xm430-w350',
        'xm430-w350',
        'xm430-w350',
        'xm430-w350',
        'xm430-w350',
        'xm430-w350',
        'xm430-w350',
        'xl430-w250',
        'xc430-w150'
    ]
}


class InterbotixBus(MotorsBus):
    ARM_GROUP = set(['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate'])

    def __init__(
            self,
            robot_model,
            robot_name,
            node,
            moving_time=2.0,
            accel_time=0.3,
            gripper_pressure=0.5,
            gripper_pressure_lower_limit=150,
            gripper_pressure_upper_limit=350
    ):

        self.bot = InterbotixManipulatorXS(
            robot_model=robot_model,
            group_name=robot_name,
            gripper_name='gripper',
            robot_name='follower_left',
            node=node,
            iterative_update_fk=False,
            gripper_pressure=gripper_pressure,
            gripper_pressure_lower_limit=gripper_pressure_lower_limit,
            gripper_pressure_upper_limit=gripper_pressure_upper_limit
        )
        self.set_trajectory_time(moving_time, accel_time)
        self.bus = DynamixelMotorsBus("", list(self.ARM_GROUP) + ["gripper"], mock=True)
        self._torque_on = False

    ### interface
    @property
    def motor_names(self):
        return self.manipulator.group_info.joint_names + ["gripper"]

    @property
    def motor_models(self) -> list[str]:
        return INTERBOTIX_MOTORMODELS[self.bot.arm.robot_model]

    def connect(self):
        ...

    def disconnect(self):
        ...

    def read(self, data_name, joint_names: str | list[str] | None = None):
        if data_name == "Present_Position":
            return self.get_joint_positions()
        else:
            use_group_cmd, remaining_joint_names = self._use_group_cmd(joint_names)

            values = []
            if use_group_cmd:
                values = self.manipulator.dxl.robot_get_motor_registers("group", "arm", data_name)
            for joint_name in remaining_joint_names:
                values.append(self.manipulator.dxl.robot_get_motor_registers("single", joint_name, data_name))

            return np.array(values)

    def write(self):
        pass

    ### calibration
    def set_calibration(self, calibration):
        self.bus.set_calibration(calibration)

    def apply_calibration(self, values, joint_names=None, is_velocity: bool = False):
        if joint_names is None:
            joint_names = self.joint_names

        assert len(values) == len(joint_names), \
            f"InterbotixBus: number of values ({len(values)}) must match number of joints ({len(joint_names)})"

        return self.bus.apply_calibration_autocorrect(self, values, joint_names, is_velocity)

    def revert_calibration(self, values, joint_names=None, is_velocity: bool = False):
        if joint_names is None:
            joint_names = self.joint_names

        assert len(values) == len(joint_names), \
            f"InterbotixBus: number of values ({len(values)}) must match number of joints ({len(joint_names)})"

        return self.bus.revert_calibration(self, values, joint_names, is_velocity)

    def set_trajectory_time(self, moving_time, accel_time):
        self.bot.arm.set_trajectory_time(moving_time, accel_time)

    ### joints
    def get_joint_positions(self):
        qpos = self.bot.get_joint_positions()
        qpos = self.apply_calibration(qpos, joint_names=self.ARM_GROUP)
        gripper_pos = self.get_gripper_position()
        return qpos + [gripper_pos]

    def get_joint_velocities(self):
        qvel = self.bot.get_joint_velocities()
        qvel = self.apply_calibration(qvel, joint_names=self.ARM_GROUP, is_velocity=True)
        gripper_vel = self.get_gripper_velocity()
        return qvel + [gripper_vel]

    def set_joint_positions(self, goal_pos):
        goal_pos = self.apply_calibration(values=goal_pos[:6], joint_names=self.ARM_GROUP)
        self.bot.set_joint_positions(goal_pos, blocking=False)
        self.set_gripper_position(goal_pos[6])

    ### gripper
    def get_gripper_position(self):
        gripper_pos = self.bot.core.robot.robot_get_single_joint_state("gripper").position
        gripper_pos = self.apply_calibration([gripper_pos], joint_names=["gripper"])[0]
        return gripper_pos

    def get_gripper_velocity(self):
        gripper_vel = self.bot.core.robot.robot_get_single_joint_state("gripper").velocity
        gripper_vel = self.apply_calibration([gripper_vel], joint_names=["gripper"], is_velocity=True)[0]
        return gripper_vel

    def set_gripper_position(self, goal_pos):
        gripper_pos = self.revert_calibration([goal_pos], joint_names=["gripper"])[0]
        self.bot.core.write_joint_command(name="gripper", cmd=gripper_pos)

    ### ee pose
    def get_ee_pose(self):
        ee_pose = self.bot.get_ee_pose()
        gripper_pos = self.get_gripper_position()
        return ee_pose + [gripper_pos]

    def get_ee_velocity(self):
        """
        Compute the end-effector velocity given joint velocities.

        :param arm: Instance of InterbotixArmXSInterface
        :return: A 6x1 numpy array representing the end-effector velocity [linear; angular] in the space frame.
        """
        # Get joint positions and velocities
        qpos = self.bot.get_joint_positions()
        qvel = self.bot.get_joint_velocities()

        # Compute the space Jacobian
        J_space = mr.JacobianSpace(self.bot.robot_des.Slist, qpos)

        # Convert joint velocities to end-effector velocity
        ee_vel = np.dot(J_space, qvel)
        gripper_vel = self.get_gripper_velocity()
        return ee_vel.tolist()  + [gripper_vel]

    def set_ee_pose(self, goal_pose: list, initial_guess: list = None, blocking: bool = False):
        """
            :param goal_pose: a list with seven elements corresponding to [x, y, z, pitch, roll, yaw, gripper_pos] in the space frame
        """
        x, y, z, roll, pitch, yaw = goal_pose[:6]
        theta_list, success = self.bot.set_ee_pose_components(x, y, z, roll, pitch, yaw, initial_guess=initial_guess[:6], blocking=blocking)
        self.set_gripper_position(goal_pose[6])

        if not success:
            ...
        return theta_list

    def fk(self, joint_positions):
        return mr.FKinSpace(self.bot.arm.robot_des.M, self.bot.arm.robot_des.Slist, joint_positions[:6])

    ### utility
    def torque_off(self):
        self.bot.core.robot_torque_enable('group', 'arm', False)
        self.bot.core.robot_torque_enable('single', 'gripper', False)

    def torque_on(self):
        self.bot.core.robot_torque_enable('group', 'arm', True)
        self.bot.core.robot_torque_enable('single', 'gripper', True)

    def enable_gravity_compensation(self):
        gravity_compensation = InterbotixGravityCompensationInterface(self.bot.core)
        gravity_compensation.enable()

    def disable_gravity_compensation(self):
        gravity_compensation = InterbotixGravityCompensationInterface(self.bot.core)
        gravity_compensation.disable()

    def _use_group_cmd(self, joint_names=None):
        # if joint_names is a proper subset of all joints in the arm,
        # returns (true, remaining_joint_names)
        # otherwise returns (false, joint_names)
        if joint_names is None:
            joint_names = self.joint_names

        _joint_names = set(joint_names)
        if _joint_names.issubset(self.ARM_GROUP):
            return True, list(_joint_names.difference(self.ARM_GROUP))
        else:
            return False, joint_names
