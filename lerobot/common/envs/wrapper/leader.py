import logging
import sys
import time
from collections import deque
from threading import Lock
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch

from lerobot.common.robot_devices.control_utils import (
    is_headless,
    FootSwitchHandler,
)
from lerobot.common.utils.utils import log_say
from lerobot.scripts.server.kinematics import get_kinematics
logging.basicConfig(level=logging.INFO)


###########################################################
# Wrappers related to human intervention and input devices
###########################################################


class BaseLeaderControlWrapper(gym.Wrapper):
    """Base class for leader-follower robot control wrappers."""

    def __init__(
        self, env, use_geared_leader_arm: bool = False, ee_action_space_params=None, use_gripper=False, foot_switches: dict[str, dict] | None = None
    ):
        super().__init__(env)
        self.robot_leader = env.unwrapped.robot.leader_arms["main"]
        self.robot_follower = env.unwrapped.robot.follower_arms["main"]
        self.use_geared_leader_arm = use_geared_leader_arm
        self.ee_action_space_params = ee_action_space_params
        self.use_ee_action_space = ee_action_space_params is not None
        self.use_gripper: bool = use_gripper
        self.use_target_ee_pos = hasattr(self, "target_ee_pos")
        self.last_gripper_action: int = 1
        self.foot_switch_threads = dict()
        self._block_interventions = False

        if self.use_ee_action_space:
            self.action_bounds = np.array([
                self.ee_action_space_params.x_step_size,
                self.ee_action_space_params.y_step_size,
                self.ee_action_space_params.z_step_size,
            ], dtype=np.float32)

        # Set up keyboard event tracking
        self.keyboard_events = dict()
        self._init_keyboard_events()
        self._init_foot_switches(foot_switches)
        self.event_lock = Lock()  # Thread-safe access to events

        # Initialize robot control
        self.leader_kinematics = get_kinematics(env.unwrapped.robot.config, robot_type="leader")
        self.follower_kinematics = get_kinematics(env.unwrapped.robot.config, robot_type="follower")
        self.prev_leader_ee = None
        self.prev_leader_pos = None
        self.leader_torque_enabled = True

        # Configure leader arm
        # NOTE: Lower the gains of leader arm for automatic take-over
        # With lower gains we can manually move the leader arm without risk of injury to ourselves or the robot
        # With higher gains, it would be dangerous and difficult to modify the leader's pose while torque is enabled
        # Default value for P_coeff is 32
        xm_motors = [
            "waist",
            "shoulder",
            "shoulder_shadow",
            "elbow",
            "elbow_shadow",
            "forearm_roll",
            "wrist_angle",
        ]
        self.robot_leader.write("Torque_Enable", 1)
        self.robot_leader.write("Position_P_Gain", 200, motor_names=xm_motors)
        self.robot_leader.write("Position_I_Gain", 0, motor_names=xm_motors)
        self.robot_leader.write("Position_D_Gain", 0, motor_names=xm_motors)

        self._init_keyboard_listener()

    @property
    def block_interventions(self):
        return self._block_interventions

    @block_interventions.setter
    def block_interventions(self, val: bool):
        self._block_interventions = val

    def _init_keyboard_events(self):
        """Initialize the keyboard events dictionary - override in subclasses."""
        self.keyboard_events = {
            "episode_success": False,
            "episode_end": False,
            "rerecord_episode": False,
            "manual_intervention": False
        }

    def _handle_key_press(self, key, keyboard):
        """Handle key presses - override in subclasses for additional keys."""
        try:
            if key == keyboard.Key.esc and "episode_end":
                self.keyboard_events["episode_end"] = True
                return
            if key == keyboard.Key.left:
                self.keyboard_events["rerecord_episode"] = True
                print("Rerecord Episode")
                return
            if hasattr(key, "char") and key.char == "s":
                logging.info("Key 's' pressed. Episode success triggered.")
                self.keyboard_events["episode_success"] = True
                return
            if key == keyboard.Key.space:
                self.keyboard_events["manual_intervention"] = not self.keyboard_events["manual_intervention"]
                return

        except Exception as e:
            logging.error(f"Error handling key press: {e}")

    def _init_keyboard_listener(self):
        """Initialize keyboard listener if not in headless mode"""
        if is_headless():
            logging.warning(
                "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
            )
            return
        try:
            from pynput import keyboard

            def on_press(key):
                with self.event_lock:
                    self._handle_key_press(key, keyboard)

            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.start()

        except ImportError:
            logging.warning("Could not import pynput. Keyboard interface will not be available.")
            self.listener = None

    def _init_foot_switches(self, foot_switches: dict[str, str] | None) -> None:
        if foot_switches is None:
            foot_switches = dict()

        for event_name in foot_switches:
            assert event_name in self.keyboard_events
            fs_params = foot_switches[event_name]
            self.foot_switch_threads[event_name] = FootSwitchHandler(
                device_path=f'/dev/input/event{fs_params["device"]}',
                toggle=bool(fs_params["toggle"]),
                event_name=event_name
            )
            self.foot_switch_threads[event_name].start()

    def _check_intervention(self):
        """Check if intervention is needed - override in subclasses."""
        return False

    def _handle_intervention(self, action):
        """Process actions during intervention mode."""
        if self.leader_torque_enabled:
            self.robot_leader.write("Torque_Enable", 0)
            self.leader_torque_enabled = False

        follower_pos = self.robot_follower.read("Present_Position")
        leader_pos = self.robot_leader.read("Present_Position")
        leader_ee = self.follower_kinematics.fk_gripper_tip(leader_pos)[:3, 3]

        # if the env has a EEActionWrapper, we use the internal target value as a reference
        if self.use_target_ee_pos and self.target_ee_pos is not None:
            follower_ee = self.target_ee_pos[:3, 3]
        else:
            follower_ee = self.follower_kinematics.fk_gripper_tip(follower_pos)[:3, 3]


        if self.prev_leader_ee is None:
            self.prev_leader_ee = leader_ee

        action = leader_ee - follower_ee
        action = np.clip(
            action,
            -self.action_bounds,
            self.action_bounds
        )

        self.prev_leader_ee = leader_ee

        if self.use_gripper:
            leader_gripper = leader_pos[-1]
            if leader_gripper < 15:
                gripper_action = 0
            elif leader_gripper > 85:
                gripper_action = 2
            else:
                gripper_action = 1

            # only update the last gripper action when it flips from 0 to 2 or vice versa
            # this way we only sent gripper commands once, even when we go from 0 -> 1 -> 0 or 2 -> 1 -> 2
            if gripper_action == 0 and self.last_gripper_action == 0:
                actual_gripper_action = 1
            elif gripper_action == 2 and self.last_gripper_action == 2:
                actual_gripper_action = 1
            else:
                actual_gripper_action = gripper_action
            self.last_gripper_action = gripper_action if gripper_action in [0, 2] else self.last_gripper_action

            action = np.append(action, actual_gripper_action)

        return action, torch.Tensor(action)

    def _handle_leader_teleoperation(self):
        """Handle leader teleoperation (non-intervention) operation."""
        if not self.leader_torque_enabled:
            self.robot_leader.write("Torque_Enable", 1)
            self.leader_torque_enabled = True

        follower_pos = self.robot_follower.read("Present_Position")
        leader_pos = self.robot_leader.read("Present_Position")

        target_joint_pos = self.leader_kinematics.ik(
            leader_pos,
            self.follower_kinematics.fk_gripper_tip(follower_pos),
            position_only=True,
            fk_func=self.leader_kinematics.fk_gripper_tip,
            gripper_pos=follower_pos[-1]
        )

        self.robot_leader.write("Goal_Position", target_joint_pos)

    def _handle_events(self):
        for handler in self.foot_switch_threads.values():
            self.keyboard_events.update(handler.keyboard_events)

    def step(self, action):
        """Execute environment step with possible intervention."""
        self._handle_events()
        is_intervention = self._check_intervention()
        action_intervention = None

        # NOTE:
        if is_intervention and not self._block_interventions:
            action, action_intervention = self._handle_intervention(action)
        else:
            self._handle_leader_teleoperation()

        # NOTE:
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Add intervention info
        info["is_intervention"] = is_intervention
        if is_intervention:
            info["action_intervention"] = action_intervention
        info["rerecord_episode"] = self.keyboard_events["rerecord_episode"]

        # Check for success or manual termination
        success = self.keyboard_events["episode_success"]
        terminated = (
                terminated or
                self.keyboard_events["episode_end"] or
                self.keyboard_events["rerecord_episode"] or
                success
        )

        if success:
            reward = 1.0
            logging.info("Episode ended successfully with reward 1.0")

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment and internal state."""
        self.prev_leader_ee = None
        self.prev_leader_pos = None
        self.last_gripper_action = 1
        self.keyboard_events = dict.fromkeys(self.keyboard_events, False)
        for handler in self.foot_switch_threads.values():
            handler.reset()
        return super().reset(**kwargs)

    def close(self):
        """Clean up resources."""
        if hasattr(self, "listener") and self.listener is not None:
            self.listener.stop()
        for handler in self.foot_switch_threads.values():
            handler.stop()
        return self.env.close()


class GearedLeaderControlWrapper(BaseLeaderControlWrapper):
    """Wrapper that enables manual intervention via keyboard."""

    def _init_keyboard_events(self):
        """Initialize keyboard events including human intervention flag."""
        super()._init_keyboard_events()
        self.keyboard_events["human_intervention_step"] = False

    def _handle_key_press(self, key, keyboard):
        """Handle key presses including space for intervention toggle."""
        super()._handle_key_press(key, keyboard)
        if key == keyboard.Key.space:
            if not self.keyboard_events["human_intervention_step"]:
                logging.info(
                    "Space key pressed. Human intervention required.\n"
                    "Place the leader in similar pose to the follower and press space again."
                )
                self.keyboard_events["human_intervention_step"] = True
                log_say("Human intervention step.", play_sounds=True)
            else:
                self.keyboard_events["human_intervention_step"] = False
                logging.info("Space key pressed for a second time.\nContinuing with policy actions.")
                log_say("Continuing with policy actions.", play_sounds=True)

    def _check_intervention(self):
        """Check if human intervention is active."""
        return self.keyboard_events["human_intervention_step"]


class GearedLeaderAutomaticControlWrapper(BaseLeaderControlWrapper):
    """Wrapper with automatic intervention based on error thresholds."""

    def __init__(
        self,
        env,
        ee_action_space_params=None,
        use_gripper=False,
        intervention_threshold=0.034,
        release_threshold=0.01,
        queue_size=10,
    ):
        super().__init__(env, ee_action_space_params=ee_action_space_params, use_gripper=use_gripper)

        # Error tracking parameters
        self.intervention_threshold = intervention_threshold  # Threshold to trigger intervention
        self.release_threshold = release_threshold  # Threshold to release intervention
        self.queue_size = queue_size  # Number of error measurements to keep

        # Error tracking variables
        self.error_queue = deque(maxlen=self.queue_size)
        self.error_over_time_queue = deque(maxlen=self.queue_size)
        self.previous_error = 0.0
        self.is_intervention_active = False
        self.start_time = time.perf_counter()

    def _check_intervention(self):
        """Determine if intervention should occur based on leader-follower error."""
        # Skip intervention logic for the first few steps to collect data
        if time.perf_counter() - self.start_time < 1.0:  # Wait 1 second before enabling
            return False

        # Get current positions
        leader_positions = self.robot_leader.read("Present_Position")
        follower_positions = self.robot_follower.read("Present_Position")
        leader_ee = self.leader_kinematics.fk_gripper_tip(leader_positions)[:3, 3]
        follower_ee = self.follower_kinematics.fk_gripper_tip(follower_positions)[:3, 3]

        # Calculate error and error rate
        error = np.linalg.norm(leader_ee - follower_ee)
        error_over_time = np.abs(error - self.previous_error)

        # Add to queue for running average
        self.error_queue.append(error)
        self.error_over_time_queue.append(error_over_time)

        # Update previous error
        self.previous_error = error

        # Calculate averages if we have enough data
        if len(self.error_over_time_queue) >= self.queue_size:
            avg_error_over_time = np.mean(self.error_queue)

            # Debug info
            if self.is_intervention_active:
                logging.debug(f"Error rate during intervention: {avg_error_over_time:.4f}")

            # Determine if intervention should start or stop
            if not self.is_intervention_active and avg_error_over_time > self.intervention_threshold:
                # Transition to intervention mode
                self.is_intervention_active = True
                logging.info(f"Starting automatic intervention: error rate {avg_error_over_time:.4f}")

            elif self.is_intervention_active and avg_error_over_time < self.release_threshold:
                # End intervention mode
                self.is_intervention_active = False
                logging.info(f"Ending automatic intervention: error rate {avg_error_over_time:.4f}")

        return self.is_intervention_active or self.keyboard_events["manual_intervention"]

    def reset(self, **kwargs):
        """Reset error tracking on environment reset."""
        self.error_queue.clear()
        self.error_over_time_queue.clear()
        self.previous_error = 0.0
        self.is_intervention_active = False
        self.start_time = time.perf_counter()
        return super().reset(**kwargs)


class GamepadControlWrapper(gym.Wrapper):
    """
    Wrapper that allows controlling a gym environment with a gamepad.

    This wrapper intercepts the step method and allows human input via gamepad
    to override the agent's actions when desired.
    """

    def __init__(
        self,
        env,
        x_step_size=1.0,
        y_step_size=1.0,
        z_step_size=1.0,
        use_gripper=False,
        auto_reset=False,
        input_threshold=0.001,
    ):
        """
        Initialize the gamepad controller wrapper.

        cfg.
            env: The environment to wrap
            x_step_size: Base movement step size for X axis in meters
            y_step_size: Base movement step size for Y axis in meters
            z_step_size: Base movement step size for Z axis in meters
            vendor_id: USB vendor ID of the gamepad (default: Logitech)
            product_id: USB product ID of the gamepad (default: RumblePad 2)
            auto_reset: Whether to auto reset the environment when episode ends
            input_threshold: Minimum movement delta to consider as active input
        """
        super().__init__(env)
        from lerobot.scripts.server.end_effector_control_utils import (
            GamepadController,
            GamepadControllerHID,
        )

        # use HidApi for macos
        if sys.platform == "darwin":
            self.controller = GamepadControllerHID(
                x_step_size=x_step_size,
                y_step_size=y_step_size,
                z_step_size=z_step_size,
            )
        else:
            self.controller = GamepadController(
                x_step_size=x_step_size,
                y_step_size=y_step_size,
                z_step_size=z_step_size,
            )
        self.auto_reset = auto_reset
        self.use_gripper = use_gripper
        self.input_threshold = input_threshold
        self.controller.start()

        logging.info("Gamepad control wrapper initialized")
        print("Gamepad controls:")
        print("  Left analog stick: Move in X-Y plane")
        print("  Right analog stick: Move in Z axis (up/down)")
        print("  X/Square button: End episode (FAILURE)")
        print("  Y/Triangle button: End episode (SUCCESS)")
        print("  B/Circle button: Exit program")

    def get_gamepad_action(
        self,
    ) -> Tuple[bool, np.ndarray, bool, bool, bool]:
        """
        Get the current action from the gamepad if any input is active.

        Returns:
            Tuple of (is_active, action, terminate_episode, success)
        """
        # Update the controller to get fresh inputs
        self.controller.update()

        # Get movement deltas from the controller
        delta_x, delta_y, delta_z = self.controller.get_deltas()

        intervention_is_active = self.controller.should_intervene()

        # Create action from gamepad input
        gamepad_action = np.array([delta_x, delta_y, delta_z], dtype=np.float32)

        if self.use_gripper:
            gripper_command = self.controller.gripper_command()
            if gripper_command == "open":
                gamepad_action = np.concatenate([gamepad_action, [2.0]])
            elif gripper_command == "close":
                gamepad_action = np.concatenate([gamepad_action, [0.0]])
            else:
                gamepad_action = np.concatenate([gamepad_action, [1.0]])

        # Check episode ending buttons
        # We'll rely on controller.get_episode_end_status() which returns "success", "failure", or None
        episode_end_status = self.controller.get_episode_end_status()
        terminate_episode = episode_end_status is not None
        success = episode_end_status == "success"
        rerecord_episode = episode_end_status == "rerecord_episode"

        return (
            intervention_is_active,
            gamepad_action,
            terminate_episode,
            success,
            rerecord_episode,
        )

    def step(self, action):
        """
        Step the environment, using gamepad input to override actions when active.

        cfg.
            action: Original action from agent

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Get gamepad state and action
        (
            is_intervention,
            gamepad_action,
            terminate_episode,
            success,
            rerecord_episode,
        ) = self.get_gamepad_action()

        # Update episode ending state if requested
        if terminate_episode:
            logging.info(f"Episode manually ended: {'SUCCESS' if success else 'FAILURE'}")

        # Only override the action if gamepad is active
        action = gamepad_action if is_intervention else action

        # Step the environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Add episode ending if requested via gamepad
        terminated = terminated or truncated or terminate_episode

        if success:
            reward = 1.0
            logging.info("Episode ended successfully with reward 1.0")

        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)

        info["is_intervention"] = is_intervention
        info["action_intervention"] = action
        info["rerecord_episode"] = rerecord_episode

        # If episode ended, reset the state
        if terminated or truncated:
            # Add success/failure information to info dict
            info["next.success"] = success

            # Auto reset if configured
            if self.auto_reset:
                obs, reset_info = self.reset()
                info.update(reset_info)

        return obs, reward, terminated, truncated, info

    def close(self):
        """Clean up resources when environment closes."""
        # Stop the controller
        if hasattr(self, "controller"):
            self.controller.stop()

        # Call the parent close method
        return self.env.close()


class SpacemouseControlWrapper(gym.Wrapper):
    pass
