
from dataclasses import dataclass
import threading
import time

import numpy as np


@dataclass(slots=True)
class _MockPeriodToken:
    start_time: float
    waited: bool = False


_MOCK_RTDE_STATES: dict[str, dict] = {}
_MOCK_RTDE_STATES_LOCK = threading.Lock()


def _get_or_create_mock_rtde_state(hostname: str, frequency: float | None = None) -> dict:
    with _MOCK_RTDE_STATES_LOCK:
        state = _MOCK_RTDE_STATES.get(hostname)
        if state is None:
            state = {
                "pose": np.zeros(6, dtype=np.float64),
                "vel": np.zeros(6, dtype=np.float64),
                "commanded_wrench": np.zeros(6, dtype=np.float64),
                "measured_wrench": np.zeros(6, dtype=np.float64),
                "ft_bias": np.zeros(6, dtype=np.float64),
                "task_frame": np.zeros(6, dtype=np.float64),
                "selection_vector": np.ones(6, dtype=np.float64),
                "speed_limits": np.full(6, np.inf, dtype=np.float64),
                "mode": "idle",
                "gain_scaling": 1.0,
                "payload_mass": 0.0,
                "payload_cog": np.zeros(3, dtype=np.float64),
                "tcp_offset": np.zeros(6, dtype=np.float64),
                "frequency": float(frequency or 125.0),
                "last_update": time.monotonic(),
                "lock": threading.Lock(),
            }
            _MOCK_RTDE_STATES[hostname] = state
        elif frequency is not None:
            state["frequency"] = float(frequency)
        return state


def _advance_mock_rtde_state(state: dict) -> None:
    now = time.monotonic()
    with state["lock"]:
        dt = max(0.0, min(now - state["last_update"], 0.05))
        if dt <= 0.0:
            return

        if state["mode"] == "force":
            applied_wrench = state["selection_vector"] * state["gain_scaling"] * state["commanded_wrench"]
        else:
            applied_wrench = np.zeros(6, dtype=np.float64)

        # A lightly damped diagonal rigid-body model is enough for controller tests.
        linear_mass = 8.0 + max(float(state["payload_mass"]), 0.0)
        angular_inertia = 1.2 + 0.05 * max(float(state["payload_mass"]), 0.0)
        linear_damping = 5.0
        angular_damping = 2.5

        acc = np.zeros(6, dtype=np.float64)
        acc[:3] = applied_wrench[:3] / linear_mass - linear_damping * state["vel"][:3]
        acc[3:] = applied_wrench[3:] / angular_inertia - angular_damping * state["vel"][3:]

        state["vel"] += acc * dt
        finite_speed_limits = np.where(np.isfinite(state["speed_limits"]), state["speed_limits"], np.inf)
        state["vel"] = np.clip(state["vel"], -finite_speed_limits, finite_speed_limits)
        state["pose"][:3] += state["vel"][:3] * dt
        state["pose"][3:6] = (
            R.from_rotvec(state["vel"][3:6] * dt) * R.from_rotvec(state["pose"][3:6])
        ).as_rotvec()

        alpha = np.clip(12.0 * dt, 0.0, 1.0)
        state["measured_wrench"] += alpha * (applied_wrench - state["measured_wrench"])
        state["last_update"] = now


class MockRTDEControlInterface:
    """Small UR RTDE control mock used by the controller in hardware-free mode."""

    def __init__(self, hostname: str | None = None, frequency: float = 125.0, *args):
        self.hostname = hostname or "mock-ur"
        self.frequency = float(frequency)
        self._state = _get_or_create_mock_rtde_state(self.hostname, self.frequency)

    def initPeriod(self):
        _advance_mock_rtde_state(self._state)
        return _MockPeriodToken(start_time=time.monotonic())

    def waitPeriod(self, t0):
        if isinstance(t0, _MockPeriodToken):
            if t0.waited:
                _advance_mock_rtde_state(self._state)
                return
            deadline = t0.start_time + (1.0 / max(self.frequency, 1e-6))
            t0.waited = True
        else:
            deadline = float(t0) + (1.0 / max(self.frequency, 1e-6))

        remaining = deadline - time.monotonic()
        if remaining > 0.0:
            time.sleep(remaining)
        _advance_mock_rtde_state(self._state)

    def setTcp(self, tcp_pose):
        with self._state["lock"]:
            self._state["tcp_offset"] = np.asarray(tcp_pose, dtype=np.float64)

    def setPayload(self, mass, cog=None):
        with self._state["lock"]:
            self._state["payload_mass"] = float(mass)
            if cog is not None:
                self._state["payload_cog"] = np.asarray(cog, dtype=np.float64)
        return True

    def moveJ(self, joints, speed, acceleration):
        _advance_mock_rtde_state(self._state)
        with self._state["lock"]:
            joints = np.asarray(joints, dtype=np.float64)
            self._state["pose"][3:6] = 0.0
            self._state["vel"][:] = 0.0
            self._state["commanded_wrench"][:] = 0.0
            self._state["measured_wrench"][:] = 0.0
            self._state["mode"] = "idle"
            self._state["pose"][:3] = 0.01 * joints[:3]
        time.sleep(0.02)
        return True

    def servoL(self, pose, vel, acc, dt, lookahead_time, gain):
        _advance_mock_rtde_state(self._state)
        with self._state["lock"]:
            self._state["pose"] = np.asarray(pose, dtype=np.float64)
            self._state["vel"][:] = 0.0
            self._state["commanded_wrench"][:] = 0.0
            self._state["measured_wrench"][:] = 0.0
            self._state["mode"] = "idle"
        return True

    def speedL(self, speed6, acc, dt):
        _advance_mock_rtde_state(self._state)
        with self._state["lock"]:
            self._state["vel"] = np.asarray(speed6, dtype=np.float64)
            self._state["mode"] = "idle"
            self._state["commanded_wrench"][:] = 0.0
            self._state["measured_wrench"][:] = 0.0
        _advance_mock_rtde_state(self._state)
        return True

    def forceModeSetGainScaling(self, scaling):
        with self._state["lock"]:
            self._state["gain_scaling"] = float(scaling)

    def forceMode(self, *args):
        if len(args) < 2:
            raise TypeError("forceMode expects at least a selection vector and wrench")

        if len(args) >= 5:
            candidate_selection = np.asarray(args[1], dtype=np.float64)
            looks_like_v2_signature = candidate_selection.shape == (6,) and np.all(
                np.isin(candidate_selection, [0.0, 1.0])
            )
        else:
            looks_like_v2_signature = False

        if looks_like_v2_signature:
            task_frame, selection_vector, wrench, _force_type, limits = args[:5]
        else:
            task_frame = np.zeros(6, dtype=np.float64)
            selection_vector = args[0]
            wrench = args[1]
            limits = args[3] if len(args) > 3 else np.full(6, np.inf, dtype=np.float64)

        _advance_mock_rtde_state(self._state)
        with self._state["lock"]:
            self._state["task_frame"] = np.asarray(task_frame, dtype=np.float64)
            self._state["selection_vector"] = np.asarray(selection_vector, dtype=np.float64)
            self._state["commanded_wrench"] = np.asarray(wrench, dtype=np.float64)
            self._state["speed_limits"] = np.asarray(limits, dtype=np.float64)
            self._state["mode"] = "force"
        return True

    def zeroFtSensor(self):
        _advance_mock_rtde_state(self._state)
        with self._state["lock"]:
            self._state["ft_bias"] = self._state["measured_wrench"].copy()

    def forceModeStop(self):
        _advance_mock_rtde_state(self._state)
        with self._state["lock"]:
            self._state["mode"] = "idle"
            self._state["commanded_wrench"][:] = 0.0

    def speedStop(self):
        with self._state["lock"]:
            self._state["vel"][:] = 0.0

    def servoStop(self):
        with self._state["lock"]:
            self._state["vel"][:] = 0.0

    def stopScript(self):
        self.forceModeStop()

    def disconnect(self):
        pass


class MockRTDEReceiveInterface:
    """Receive side of the in-process UR RTDE mock."""

    def __init__(self, hostname: str | None = None, *args, **kwargs):
        self.hostname = hostname or "mock-ur"
        self._state = _get_or_create_mock_rtde_state(self.hostname)

    def getActualTCPPose(self):
        _advance_mock_rtde_state(self._state)
        with self._state["lock"]:
            return self._state["pose"].copy()

    def getActualTCPSpeed(self):
        _advance_mock_rtde_state(self._state)
        with self._state["lock"]:
            return self._state["vel"].copy()

    def getActualTCPForce(self):
        _advance_mock_rtde_state(self._state)
        with self._state["lock"]:
            return (self._state["measured_wrench"] - self._state["ft_bias"]).copy()

    def getActualQ(self):
        return np.zeros(6, dtype=np.float64)

    def getActualQd(self):
        return np.zeros(6, dtype=np.float64)

    def disconnect(self):
        pass