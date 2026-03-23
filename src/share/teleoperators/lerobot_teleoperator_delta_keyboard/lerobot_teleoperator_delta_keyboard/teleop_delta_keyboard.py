#!/usr/bin/env python

import logging
import os
import sys
import time
from queue import Queue
from typing import Any

from lerobot.processor import RobotAction
from lerobot.processor.hil_processor import HasTeleopEvents
from lerobot.teleoperators import TeleopEvents, Teleoperator
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config_delta_keyboard import (
    KeyboardAxisBinding,
    KeyboardVelocityTeleopConfig,
)

PYNPUT_AVAILABLE = True
try:
    if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
        logging.info("No DISPLAY set. Skipping pynput import.")
        raise ImportError("pynput blocked intentionally due to no display.")

    from pynput import keyboard
except ImportError:
    keyboard = None
    PYNPUT_AVAILABLE = False
except Exception as e:
    keyboard = None
    PYNPUT_AVAILABLE = False
    logging.info(f"Could not import pynput: {e}")


logger = logging.getLogger(__name__)


class KeyboardVelocityTeleop(Teleoperator, HasTeleopEvents):
    """
    Configurable keyboard teleoperator returning velocity commands, e.g.:

        {
            "x.vel": 0.1,
            "y.vel": 0.0,
            "z.vel": -0.1,
            "rx.vel": 0.0,
            "ry.vel": 0.5,
            "rz.vel": 0.0,
        }

    Key bindings and axis scales are defined in the config.
    """

    config_class = KeyboardVelocityTeleopConfig
    name = "keyboard_velocity"

    AXES = ("x", "y", "z", "rx", "ry", "rz")

    def __init__(self, config: KeyboardVelocityTeleopConfig):
        super().__init__(config)
        self.config = config

        self.event_queue = Queue()
        self.current_pressed: dict[str, bool] = {}
        self.listener = None
        self.logs: dict[str, float] = {}

        self._event_states: dict[str, bool] = {
            event_name: False for event_name in self.config.event_bindings
        }
        self._prev_event_key_state: dict[str, bool] = {
            event_name: False for event_name in self.config.event_bindings
        }

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{axis}.vel": float for axis in self.AXES}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return PYNPUT_AVAILABLE and isinstance(self.listener, keyboard.Listener) and self.listener.is_alive()

    @property
    def is_calibrated(self) -> bool:
        return True

    @check_if_already_connected
    def connect(self) -> None:
        if PYNPUT_AVAILABLE:
            logger.info("pynput is available - enabling local keyboard listener.")
            self.listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            self.listener.start()
        else:
            logger.info("pynput not available - skipping local keyboard listener.")
            self.listener = None

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def _normalize_key(self, key) -> str | None:
        """
        Convert pynput key objects into stable config-friendly string tokens.
        """
        if key is None:
            return None

        # Character key
        if hasattr(key, "char") and key.char is not None:
            return str(key.char).lower()

        # Special key
        if keyboard is not None and isinstance(key, keyboard.Key):
            return key.name.lower()

        # Fallback
        try:
            return str(key).lower()
        except Exception:
            return None

    def _on_press(self, key) -> None:
        token = self._normalize_key(key)
        if token is not None:
            self.event_queue.put((token, True))

    def _on_release(self, key) -> None:
        token = self._normalize_key(key)
        if token is not None:
            self.event_queue.put((token, False))

            if self.config.escape_disconnects and token == "esc":
                logger.info("ESC pressed, disconnecting.")
                self.disconnect()

    def _drain_pressed_keys(self) -> None:
        while not self.event_queue.empty():
            token, is_pressed = self.event_queue.get_nowait()
            if is_pressed:
                self.current_pressed[token] = True
            else:
                self.current_pressed.pop(token, None)

    def _axis_value(self, binding: KeyboardAxisBinding) -> float:
        if not binding.enabled:
            return 0.0

        value = 0.0
        if binding.pos_key is not None and self.current_pressed.get(binding.pos_key, False):
            value += 1.0
        if binding.neg_key is not None and self.current_pressed.get(binding.neg_key, False):
            value -= 1.0

        return value * binding.scale

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        start = time.perf_counter()

        self._drain_pressed_keys()

        action = {
            "x.vel": self._axis_value(self.config.x),
            "y.vel": self._axis_value(self.config.y),
            "z.vel": self._axis_value(self.config.z),
            "rx.vel": self._axis_value(self.config.rx),
            "ry.vel": self._axis_value(self.config.ry),
            "rz.vel": self._axis_value(self.config.rz),
        }

        self.logs["read_pos_dt_s"] = time.perf_counter() - start
        return action

    def get_teleop_events(self) -> dict[str, Any]:
        """
        Optional event interface.

        Configurable events are defined in config.event_bindings.
        If include_intervention_event=True, IS_INTERVENTION is True whenever any motion axis is active.
        """
        self._drain_pressed_keys()

        events: dict[str, Any] = {}

        for event_name, binding in self.config.event_bindings.items():
            current_pressed = self.current_pressed.get(binding.key, False)
            prev_pressed = self._prev_event_key_state.get(event_name, False)

            if binding.toggle:
                if current_pressed and not prev_pressed:
                    self._event_states[event_name] = not self._event_states[event_name]
                events[event_name] = self._event_states[event_name]
            else:
                events[event_name] = current_pressed

            self._prev_event_key_state[event_name] = current_pressed

        if self.config.include_intervention_event:
            action = self.get_action()
            events[TeleopEvents.IS_INTERVENTION] = any(abs(v) > 0.0 for v in action.values())

        return events

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        return None

    @check_if_not_connected
    def disconnect(self) -> None:
        if self.listener is not None:
            self.listener.stop()
            self.listener = None