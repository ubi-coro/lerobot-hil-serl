# rtde_tff_controller_mock.py
import os
import time
import enum
import collections
import multiprocessing as mp
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from lerobot.common.robot_devices.motors.configs import URArmConfig
from lerobot.common.utils.shared_memory import SharedMemoryRingBuffer, SharedMemoryQueue, Empty

# --- reuse your enums/classes (copy or import them if colocated) ---
class AxisMode(enum.IntEnum):
    POS           = 0
    IMPEDANCE_VEL = 1
    PURE_VEL      = 2
    FORCE         = 3

class Command(enum.IntEnum):
    SET = 0
    STOP = 1
    OPEN = 2
    CLOSE = 3
    ZERO_FT = 4

@dataclass
class TaskFrameCommand:
    cmd: Command = Command.SET
    T_WF: Optional[list | np.ndarray]         = None
    mode: Optional[list[AxisMode]]            = None
    target: Optional[list | np.ndarray]       = None
    kp: Optional[list | np.ndarray]           = None
    kd: Optional[list | np.ndarray]           = None
    max_pose_rpy: Optional[list | np.ndarray] = None
    min_pose_rpy: Optional[list | np.ndarray] = None
    def to_queue_dict(self):
        d = asdict(self)
        d["cmd"] = self.cmd.value
        for k in ("T_WF","target","kp","kd","max_pose_rpy","min_pose_rpy"):
            v = getattr(self, k)
            if v is not None: d[k] = np.asarray(v, dtype=np.float64)
        if self.mode is not None:
            d["mode"] = np.array([int(m) for m in self.mode], dtype=np.int8)
        return d
    def update(self, other: "TaskFrameCommand"):
        if other.T_WF is not None: self.T_WF = other.T_WF
        if other.mode is not None: self.mode = other.mode
        if other.target is not None: self.target = other.target
        if other.kp is not None: self.kp = other.kp
        if other.kd is not None: self.kd = other.kd
        if other.max_pose_rpy is not None: self.max_pose_rpy = other.max_pose_rpy
        if other.min_pose_rpy is not None: self.min_pose_rpy = other.min_pose_rpy

_EXAMPLE_TF_MSG = TaskFrameCommand(
    T_WF=np.zeros(6), mode=[AxisMode.POS]*6, target=np.zeros(6),
    kp=np.full(6,300.0), kd=np.full(6,20.0),
    max_pose_rpy=np.full(6, np.inf), min_pose_rpy=np.full(6, -np.inf)
).to_queue_dict()


class RTDETFFMockController(mp.Process):
    """
    Mock version of RTDE task‑frame controller.
    - No network/hardware calls.
    - Same public API and shared‑memory I/O.
    - Simple, stable time‑stepping so downstream code behaves identically.
    """

    def __init__(self, config: URArmConfig):
        super().__init__(name="MockRTDETFFController")
        self.config = config
        self.ready_event = mp.Event()

        # 1) command queue
        self.robot_cmd_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=config.shm_manager,
            examples=_EXAMPLE_TF_MSG,
            buffer_size=256
        )

        # 2) ring buffer (pose/vel/force + a few extras to match real keys)
        if self.config.receive_keys is None:
            self.config.receive_keys = [
                'ActualTCPPose', 'ActualTCPSpeed', 'ActualTCPForce', 'ActualQ', 'ActualQd'
            ]
        example = {
            'ActualTCPPose': np.zeros(6, dtype=np.float64),
            'ActualTCPSpeed': np.zeros(6, dtype=np.float64),
            'ActualTCPForce': np.zeros(6, dtype=np.float64),
            'ActualQ': np.zeros(6, dtype=np.float64),
            'ActualQd': np.zeros(6, dtype=np.float64),
            'timestamp': time.time(),
        }
        self.robot_out_rb = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=config.shm_manager,
            examples=example,
            get_max_k=config.get_max_k,
            get_time_budget=0.4,
            put_desired_frequency=config.frequency
        )

        # 3) internal state in TASK frame
        self.T_WF = np.zeros(6)  # keep identity TF
        self.mode = np.array([AxisMode.IMPEDANCE_VEL]*6, dtype=np.int8)
        self.kp = np.array([2500,2500,2500,150,150,150], dtype=np.float64)
        self.kd = np.array([80,80,80,8,8,8], dtype=np.float64)
        self.max_pose_rpy = np.full(6, np.inf)
        self.min_pose_rpy = np.full(6, -np.inf)

        # virtual command target and simulated “measured” state
        self._x_cmd = np.zeros(6, dtype=np.float64)
        self._pose_F = np.zeros(6, dtype=np.float64)
        self._v_F = np.zeros(6, dtype=np.float64)
        self._wrench_F = np.zeros(6, dtype=np.float64)

        self.last_robot_cmd = TaskFrameCommand(
            target=self._x_cmd.copy(), mode=self.mode.copy(),
            kp=self.kp.copy(), kd=self.kd.copy(),
            T_WF=self.T_WF.copy(),
            max_pose_rpy=self.max_pose_rpy.copy(),
            min_pose_rpy=self.min_pose_rpy.copy()
        )

    # ===== lifecycle =====
    def connect(self): self.start()
    def start(self, wait=True):
        super().start()
        if wait: self.ready_event.wait(self.config.launch_timeout)
    def stop(self, wait=True):
        self.robot_cmd_queue.put({'cmd': Command.STOP.value})
        if wait: self.join()
    def __enter__(self): self.start(); return self
    def __exit__(self, *a): self.stop()

    @property
    def is_ready(self): return self.ready_event.is_set()

    # ===== API surface used by UR =====
    def send_cmd(self, cmd: TaskFrameCommand):
        if self.last_robot_cmd is None:
            self.last_robot_cmd = cmd
        else:
            self.last_robot_cmd.update(cmd)
        self.robot_cmd_queue.put(self.last_robot_cmd.to_queue_dict())

    def zero_ft(self):
        self.robot_cmd_queue.put(TaskFrameCommand(cmd=Command.ZERO_FT).to_queue_dict())

    def get_robot_state(self, k=None, out=None):
        return self.robot_out_rb.get(out=out) if k is None else self.robot_out_rb.get_last_k(k=k, out=out)

    def get_all_robot_states(self):
        return self.robot_out_rb.get_all()

    # ===== control loop =====
    def run(self):
        dt = 1.0 / max(1, int(self.config.frequency))
        t_last = time.monotonic()
        iter_idx = 0
        keep = True
        try:
            while keep:
                # process queued commands
                try:
                    msgs = self.robot_cmd_queue.get_all()
                    n = len(msgs['cmd'])
                except Empty:
                    n = 0
                for i in range(n):
                    single = {k: msgs[k][i] for k in msgs}
                    cid = int(single['cmd'])
                    if cid == Command.STOP.value:
                        keep = False
                        break
                    elif cid == Command.ZERO_FT.value:
                        self._wrench_F[:] = 0.0
                        continue
                    elif cid == Command.SET.value:
                        if 'T_WF' in single and single['T_WF'] is not None:
                            self.T_WF = np.array(single['T_WF'], dtype=np.float64)
                        if 'mode' in single and single['mode'] is not None:
                            self.mode = np.array(single['mode'], dtype=np.int8)
                        if 'target' in single and single['target'] is not None:
                            self.last_robot_cmd.target = np.array(single['target'], dtype=np.float64)
                        if 'kp' in single and single['kp'] is not None:
                            self.kp = np.array(single['kp'], dtype=np.float64)
                        if 'kd' in single and single['kd'] is not None:
                            self.kd = np.array(single['kd'], dtype=np.float64)
                        if 'max_pose_rpy' in single and single['max_pose_rpy'] is not None:
                            self.max_pose_rpy = np.array(single['max_pose_rpy'], dtype=np.float64)
                        if 'min_pose_rpy' in single and single['min_pose_rpy'] is not None:
                            self.min_pose_rpy = np.array(single['min_pose_rpy'], dtype=np.float64)

                if not keep:
                    break

                # simple impedance + kinematic integration in task frame
                target = np.array(self.last_robot_cmd.target, dtype=np.float64)

                # update virtual position for POS / IMPEDANCE_VEL
                # (rotation handled via rotvec compose)
                for i in range(3):
                    m = AxisMode(self.mode[i])
                    if m == AxisMode.POS:
                        self._x_cmd[i] = target[i]
                    elif m == AxisMode.IMPEDANCE_VEL:
                        self._x_cmd[i] += target[i] * dt

                # SO(3) integration for angular IMPEDANCE_VEL
                mask_ang = np.array([AxisMode(self.mode[i])==AxisMode.IMPEDANCE_VEL for i in range(3,6)], dtype=bool)
                if mask_ang.any():
                    R_cmd = R.from_rotvec(self._x_cmd[3:6])
                    dR = R.from_rotvec(target[3:6] * mask_ang * dt)
                    self._x_cmd[3:6] = (dR * R_cmd).as_rotvec()

                # clamp pose
                self._x_cmd[:3] = np.clip(self._x_cmd[:3], self.min_pose_rpy[:3], self.max_pose_rpy[:3])
                rpy = R.from_rotvec(self._x_cmd[3:6]).as_euler('xyz', degrees=False)
                rpy = np.clip(rpy, self.min_pose_rpy[3:6], self.max_pose_rpy[3:6])
                self._x_cmd[3:6] = R.from_euler('xyz', rpy).as_rotvec()

                # compute “desired” wrench (same rules of thumb as real controller)
                wrench = np.zeros(6, dtype=np.float64)

                # translational part
                pos_err = self._x_cmd[:3] - self._pose_F[:3]
                for i in range(3):
                    m = AxisMode(self.mode[i])
                    if m == AxisMode.POS:
                        wrench[i] = self.kp[i]*pos_err[i] + self.kd[i]*(-self._v_F[i])
                    elif m == AxisMode.IMPEDANCE_VEL:
                        vel_err = target[i] - self._v_F[i]
                        wrench[i] = self.kp[i]*pos_err[i] + self.kd[i]*vel_err
                    elif m == AxisMode.PURE_VEL:
                        vel_err = target[i] - self._v_F[i]
                        wrench[i] = self.kd[i]*vel_err
                    elif m == AxisMode.FORCE:
                        wrench[i] = float(target[i])

                # rotational part
                R_cmd = R.from_rotvec(self._x_cmd[3:6])
                R_act = R.from_rotvec(self._pose_F[3:6])
                rot_err = (R_cmd * R_act.inv()).as_rotvec()
                for i in range(3,6):
                    m = AxisMode(self.mode[i])
                    j = i-3
                    if m == AxisMode.POS:
                        wrench[i] = self.kp[i]*rot_err[j] + self.kd[i]*(-self._v_F[i])
                    elif m == AxisMode.IMPEDANCE_VEL:
                        vel_err = target[i] - self._v_F[i]
                        wrench[i] = self.kp[i]*rot_err[j] + self.kd[i]*vel_err
                    elif m == AxisMode.PURE_VEL:
                        vel_err = target[i] - self._v_F[i]
                        wrench[i] = self.kd[i]*vel_err
                    elif m == AxisMode.FORCE:
                        wrench[i] = float(target[i])

                # very simple dynamics: v += a*dt, pose += v*dt (with small damping)
                # map wrench→acc via diagonal “mass” and “inertia” guesses
                mass = 8.0
                inertia = 1.5
                acc = np.zeros(6)
                acc[:3] = wrench[:3] / mass
                acc[3:6] = wrench[3:6] / inertia
                self._v_F = 0.98*self._v_F + acc*dt
                self._pose_F[:3] += self._v_F[:3] * dt
                R_next = R.from_rotvec(self._pose_F[3:6]) * R.from_rotvec(self._v_F[3:6]*dt)
                self._pose_F[3:6] = R_next.as_rotvec()

                # publish state
                state = {
                    'ActualTCPPose': self._pose_F.copy(),
                    'ActualTCPSpeed': self._v_F.copy(),
                    'ActualTCPForce': wrench.copy(),  # report command as “measured”
                    'ActualQ': np.zeros(6),           # placeholder
                    'ActualQd': np.zeros(6),          # placeholder
                    'timestamp': time.time(),
                }
                self.robot_out_rb.put(state)

                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                # pace the loop
                t_now = time.monotonic()
                sleep_t = dt - (t_now - t_last)
                if sleep_t > 0: time.sleep(sleep_t)
                t_last = time.monotonic()

        finally:
            self.ready_event.set()
