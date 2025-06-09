import time

import rtde_control

from lerobot.common.robot_devices.motors.rtde_robotiq_preamble import ROBOTIQ_PREAMBLE


class RobotiqGripper(object):
    """
    RobotiqGripper is a class for controlling a robotiq gripper using the
    ur_rtde robot interface.

    Attributes:
        rtde_c (rtde_control.RTDEControlInterface): The interface to use for the communication
    """

    def __init__(self, rtde_c):
        """
        The constructor for RobotiqGripper class.

        Parameters:
           rtde_c (rtde_control.RTDEControlInterface): The interface to use for the communication
        """
        self.rtde_c = rtde_c

    def call(self, script_name, script_function):
        return self.rtde_c.sendCustomScriptFunction(
            "ROBOTIQ_" + script_name,
            ROBOTIQ_PREAMBLE + script_function
        )

    def activate(self):
        """
        Activates the gripper. Currently the activation will take 5 seconds.

        Returns:
            True if the command succeeded, otherwise it returns False
        """
        ret = self.call("ACTIVATE", "rq_activate()")
        time.sleep(5)  # HACK
        return ret

    def set_speed(self, speed):
        """
        Set the speed of the gripper.

        Parameters:
            speed (int): speed as a percentage [0-100]

        Returns:
            True if the command succeeded, otherwise it returns False
        """
        return self.call("SET_SPEED", "rq_set_speed_norm(" + str(speed) + ")")

    def set_force(self, force):
        """
        Set the force of the gripper.

        Parameters:
            force (int): force as a percentage [0-100]

        Returns:
            True if the command succeeded, otherwise it returns False
        """
        return self.call("SET_FORCE", "rq_set_force_norm(" + str(force) + ")")

    def move(self, pos_in_mm, wait: bool = False):
        """
        Move the gripper to a specified position in (mm).

        Parameters:
            pos_in_mm (int): position in millimeters.
            wait (bool): await finished or fire-and-forget

        Returns:
            True if the command succeeded, otherwise it returns False
        """
        if wait:
            return self.call("MOVE", "rq_move_and_wait_mm(" + str(pos_in_mm) + ")")
        else:
            return self.call("MOVE", "rq_move_mm(" + str(pos_in_mm) + ")")

    def get_pos(self):
        """
        Returns the current position of the robot in (mm).
        """
        return self.call("READ", "rq_current_pos_mm()")

    def open(self, wait: bool = False):
        """
        Open the gripper.

        Parameters:
            wait (bool): await finished or fire-and-forget

        Returns:
            True if the command succeeded, otherwise it returns False
        """
        if wait:
            return self.call("OPEN", "rq_open_and_wait()")
        else:
            return self.call("OPEN", "rq_open()")

    def close(self, wait: bool = False):
        """
        Close the gripper.

        Parameters:
            wait (bool): await finished or fire-and-forget

        Returns:
            True if the command succeeded, otherwise it returns False
        """
        if wait:
            self.call("CLOSE", "rq_close_and_wait()")
        else:
            self.call("CLOSE", "rq_close()")
