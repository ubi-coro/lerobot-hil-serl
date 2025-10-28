import logging
import multiprocessing
import time
from typing import Any

from lerobot.processor.hil_processor import HasTeleopEvents
from lerobot.teleoperators import Teleoperator, TeleopEvents
from lerobot.teleoperators.spacemouse import pyspacemouse
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

logger = logging.getLogger(__name__)


class SpaceMouse(Teleoperator, HasTeleopEvents):
    def __init__(self, config: 'SpaceMouseConfig'):
        self.id = config.id
        self.config = config

        pyspacemouse.open(device=config.device)
        self.process = None
        self.stop_event = multiprocessing.Event()

        # Manager to handle shared state between processes
        self.manager = multiprocessing.Manager()
        self.latest_data = self.manager.dict()
        self.latest_data["action"] = [0.0] * 6
        self.latest_data["gripper_pos"] = [0] * 4
        self._last_gripper_pos = config.initial_gripper_pos

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
        action = {f"{ax}.vel": latest_action[i] for i, ax in enumerate(["x", "y", "z", "wx", "wy", "wz"])}

        # handle gripper action
        latest_buttons = self.latest_data.get("buttons", [0] * 4)
        close_gripper = latest_buttons[self.config.gripper_close_button_idx]
        open_gripper = latest_buttons[self.config.gripper_open_button_idx]

        if self.config.gripper_continuous:
            if open_gripper:
                self._last_gripper_pos = max([0.0, self._last_gripper_pos - self.config.gripper_gain])
            elif close_gripper:
                self._last_gripper_pos = min([1.0, self._last_gripper_pos + self.config.gripper_gain])
        else:
            if open_gripper:
                self._last_gripper_pos = 0.0
            elif close_gripper:
                self._last_gripper_pos = 1.0
        action["gripper.pos"] = self._last_gripper_pos

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def get_teleop_events(self) -> dict[str, Any]:
        buttons = self.latest_data.get("buttons", [0, 0, 0, 0])
        events = {TeleopEvents.IS_INTERVENTION: bool(buttons[0]) | bool(buttons[1])}
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
