import logging
import multiprocessing
import time
from typing import Any

import numpy as np

from lerobot.processor.hil_processor import HasTeleopEvents
from lerobot.teleoperators import Teleoperator, TeleopEvents
from lerobot.teleoperators.spacemouse import pyspacemouse
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

logger = logging.getLogger(__name__)


class SpaceMouse(Teleoperator, HasTeleopEvents):
    """
    Teleoperator interface for 3Dconnexion SpaceMouse devices.

    This class wraps the `pyspacemouse` driver and exposes a consistent
    teleoperation API for use within the LeRobot framework. It runs a separate
    background process to continuously read 6-DoF motion and button states from
    the device, making the latest data available through a
    `multiprocessing.Manager` dictionary.

    **Features**
    -----------
    - Provides velocity-based 6-DoF control (`x`, `y`, `z`, `wx`, `wy`, `wz`).
    - Supports configurable gripper control (binary or continuous modes).
    - Supports generic teleoperation events via `button_mapping`, where each
      button can trigger a *toggle* event (press once → ON, press again → OFF).
    - Non-blocking multiprocessing reader ensures stable performance and avoids
      input lag.
    - Safe connect/disconnect flow with graceful shutdown of the reader process.

    **Usage**
    ---------
    After instantiating and calling :meth:`connect`, the main training or control
    loop may repeatedly call:
        - :meth:`get_action` to obtain the latest motion/gripper command.
        - :meth:`get_teleop_events` to obtain the current state of mapped toggle
          events.

    **Multiprocessing Architecture**
    --------------------------------
    - A subprocess (`SpaceMouseReader`) performs HID reads at ~500Hz.
    - Data is stored in a shared Manager dict:
        - `"action"`: 6-element float list of velocities.
        - `"buttons"`: list of raw button states from the device.
    - The parent process never communicates directly with `pyspacemouse`,
      ensuring thread/process safety.

    **Gripper Control**
    -------------------
    If configured, gripper actions are derived from the designated button indices:
        - Continuous mode: buttons apply incremental open/close.
        - Binary mode: directly set gripper to open/closed states.

    **Teleop Events**
    -----------------
    Teleoperation events are configured via `config.button_mapping`, which maps
    event names to button indices. Events use a *toggle* behavior:
        - Event becomes True on a rising edge (0→1).
        - Event stays True until the next press toggles it back to False.

    **Errors**
    ----------
    - Raises :class:`DeviceAlreadyConnectedError` if `connect()` is called while
      the device is already running.
    - Raises :class:`DeviceNotConnectedError` on invalid disconnect attempts.

    Parameters
    ----------
    config : SpaceMouseConfig
        Configuration object specifying device selection, button bindings,
        gripper behavior, and other teleoperation options.

    Notes
    -----
    - The class does not currently implement haptic/force feedback.
    - The action and event structures follow the conventions of the LeRobot
      teleoperator interface to ensure compatibility with the rest of the
      framework.
    """

    def __init__(self, config: 'SpaceMouseConfig'):
        self.id = config.id
        self.config = config

        pyspacemouse.open(device=config.device, path=config.path)
        self.process = None
        self.stop_event = multiprocessing.Event()

        # Manager to handle shared state between processes
        self.manager = multiprocessing.Manager()
        self.latest_data = self.manager.dict()
        self.latest_data["action"] = [0.0] * 6
        self.latest_data["gripper_pos"] = [0] * 4
        self._last_gripper_pos = config.initial_gripper_pos
        self._has_gripper = self.config.gripper_close_button_idx is not None and self.config.gripper_open_button_idx is not None
        self._event_states: dict[str, bool] = {
            data["event"]: False for data in self.config.button_mapping.values()
        }
        self._prev_button_states: dict[str, int] = {
            data["event"]: 0 for data in self.config.button_mapping.values()
        }


    @property
    def action_features(self) -> dict[str, type]:
        return {f"{ax}.vel": float for ax in ["x", "y", "z", "wx", "wy", "wz"]}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.process is not None and self.process.is_alive()

    def connect(self, _: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.process = multiprocessing.Process(
            target=self._read_spacemouse, name="SpaceMouseReader"
        )
        self.process.daemon = True
        self.process.start()

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()

        # write velocity action dict
        latest_action = self.latest_data.get("action", [0.0] * 6)
        action = {
            f"{ax}.vel": latest_action[i] * self.config.action_scale[i]
            for i, ax in enumerate(["x", "y", "z", "wx", "wy", "wz"])
        }

        # handle gripper action
        if self._has_gripper:
            latest_buttons = self.latest_data.get("buttons", [0] * 2)
            close_gripper = latest_buttons[self.config.gripper_close_button_idx]
            open_gripper = latest_buttons[self.config.gripper_open_button_idx]

            action["gripper.pos"] = 0.0
            if open_gripper:
                action["gripper.pos"] = np.random.uniform(-1, -0.9)
            elif close_gripper:
                action["gripper.pos"] = np.random.uniform(0.9, 1.0)

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def get_teleop_events(self) -> dict[str, Any]:
        events: dict[str, Any] = {}

        buttons = self.latest_data.get("buttons", [0] * 2)

        for index, data in self.config.button_mapping.items():
            event = data["event"]
            toggle = data["toggle"]

            # safely read button state
            current = int(buttons[index]) if index < len(buttons) else 0

            if toggle:
                prev = self._prev_button_states.get(event, 0)

                # Rising edge: button just pressed -> toggle event state
                if current == 1 and prev == 0:
                    self._event_states[event] = not self._event_states.get(event, False)

            else:
                self._event_states[event] = current

            # Update previous state
            self._prev_button_states[event] = current

            # Expose current toggle state
            events[event] = self._event_states[event]

        return events

    def _read_spacemouse(self):
        while not self.stop_event.is_set():
            state = pyspacemouse.read_all()
            action = [0.0] * 6
            buttons = [0, 0, 0, 0]

            if len(state) == 2:
                action = [
                    -state[0].y, state[0].x, state[0].z,
                    -state[0].roll, -state[0].pitch, -state[0].yaw,
                    -state[1].y, state[1].x, state[1].z,
                    -state[1].roll, -state[1].pitch, -state[1].yaw
                ]
                buttons = state[0].buttons + state[1].buttons
            elif len(state) == 1:
                action = [
                    -state[0].y, state[0].x, state[0].z,
                    -state[0].roll, -state[0].pitch, -state[0].yaw
                ]
                buttons = state[0].buttons

            try:
                # If the manager/pipe is gone during shutdown, just exit quietly
                self.latest_data["action"] = action
                self.latest_data["buttons"] = buttons
            except (BrokenPipeError, EOFError, ConnectionResetError, OSError):
                break

            # be nice to a CPU core :)
            time.sleep(0.002)

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            DeviceNotConnectedError(f"{self} is not connected.")

        # Request the reader to stop
        self.stop_event.set()
        # Closing the HID usually unblocks read_all() if it’s waiting
        try:
            pyspacemouse.close()
        except Exception:
            pass

        # Give the process a moment to exit cleanly
        self.process.join(timeout=0.5)
        if self.process.is_alive():
            # Fall back to hard kill if needed
            self.process.terminate()
            self.process.join(timeout=0.2)

        # Now it’s safe to tear down the manager/IPC
        try:
            self.manager.shutdown()
        except Exception:
            pass

        logger.info(f"{self} disconnected.")
