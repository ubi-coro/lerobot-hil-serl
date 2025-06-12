"""Updated RTDEInterpolationController with per‑axis task‑frame formalism.
Only the worker file changed – the public UR class will get a helper next.

Key additions
-------------
* Mode enum  (POS, VEL, FORCE, FIXED)
* TFCommand  (SET)
* new queue schema (servo + task‑frame)
* internal task‑frame state & mixer inside the 1 kHz loop
* automatic force_mode / endForceMode depending on mode_mask

Assumptions
-----------
* Task frame is static after first SET (no tracking‑mode yet)
* At most one translational force axis – UR limit is 3, easy to relax
* kp / kd are diagonal
* Units:  metres, rad, N, N·m
"""
import collections
import os
import time
import enum
import multiprocessing as mp
from dataclasses import dataclass, asdict
from multiprocessing.managers import SharedMemoryManager
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from lerobot.common.robot_devices.motors.configs import URArmConfig
from lerobot.common.robot_devices.motors.rtde_robotiq_controller import RobotiqGripper
from lerobot.common.utils.shared_memory import SharedMemoryRingBuffer, SharedMemoryQueue, Empty



# ---------------------------------------------------------------------------
# Internal enums
# ---------------------------------------------------------------------------
class AxisMode(enum.IntEnum):
    POS           = 0
    IMPEDANCE_VEL = 1  # integrate‐then‐impedance
    PURE_VEL      = 2  # direct velocity‐error
    FORCE         = 3


class Command(enum.IntEnum):
    SET = 0
    STOP = 1
    OPEN = 2
    CLOSE = 3


@dataclass
class TaskFrameCommand:
    """One message = full spec for all 6 DoF"""
    cmd: Command = Command.SET
    T_WF: Optional[list | np.ndarray]     = None  # 4×4 homogeneous transform (world→task)
    mode: Optional[list[AxisMode]]        = None  # len==6
    target: Optional[list | np.ndarray]   = None  # 6 pos [m/rad], vel [m/s], or force [N]
    kp: Optional[list | np.ndarray]       = None  # 6 proportional gains (position‐error → force)
    kd: Optional[list | np.ndarray]       = None  # 6 derivative gains (velocity‐error → force)

    def to_queue_dict(self):
        d = asdict(self)
        try:
            d["cmd"]    = self.cmd.value
            d["T_WF"]   = np.asarray(self.T_WF).astype(np.float64)
            d["mode"]   = np.array([int(m) for m in self.mode], dtype=np.int8)
            d["target"] = np.asarray(self.target).astype(np.float64)
            d["kp"]     = np.asarray(self.kp).astype(np.float64)
            d["kd"]     = np.asarray(self.kd).astype(np.float64)
        except Exception as e:
            print(f"TaskFrameCommand seems to be missing fields: {e}")
        return d

    def update(self, cmd: 'TaskFrameCommand'):
        """Update only the fields that are not None in the new cmd."""
        self.cmd = cmd.cmd
        if cmd.T_WF is not None:
            self.T_WF = cmd.T_WF
        if cmd.mode is not None:
            self.mode = cmd.mode
        if cmd.target is not None:
            self.target = cmd.target
        if cmd.kp is not None:
            self.kp = cmd.kp
        if cmd.kd is not None:
            self.kd = cmd.kd
   
@dataclass
class GripperCommand:
    cmd: Command = Command.OPEN
    value: float = 0.0
    
    def to_queue_dict(self):
        d = asdict(self)
        d["cmd"] = self.cmd.value
        return d


_EXAMPLE_TF_MSG = TaskFrameCommand(
    cmd=Command.SET,
    T_WF=np.eye(4),
    mode=[AxisMode.POS]*6,
    target=np.zeros(6),
    kp=np.full(6,300.0),
    kd=np.full(6,20.0)
).to_queue_dict()


class RTDETFFController(mp.Process):
    """
    An RTDE‐based “task‐frame force‐feedback” controller.  This replaces the old
    servoL/waypoint loop with a full 6D impedance implemented via forceMode(...).
    """

    def __init__(self, config: URArmConfig):
        config = _validate_config(config)
        super().__init__(name="RTDEPositionalController")
        self.config = config
        self.ready_event = mp.Event()  # “ready” event to signal when the loop has started successfully
        self.force_on = False  # are we currently in forceMode?

        # 1) Build the command queue (TaskFrameCommand messages)
        self.robot_cmd_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=config.shm_manager,
            examples=_EXAMPLE_TF_MSG,
            buffer_size=256
        )
        self.gripper_cmd_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=config.shm_manager,
            examples=GripperCommand().to_queue_dict(),
            buffer_size=256
        )

        # 2) Build the ring buffer for streaming back pose/vel/force
        if self.config.mock:
            from tests.motors.mock_ur_rtde import RTDEReceiveInterface
        else:
            from rtde_receive import RTDEReceiveInterface
        rtde_r = RTDEReceiveInterface(hostname=config.robot_ip)

        if config.receive_keys is None:
            config.receive_keys = [
                'ActualTCPPose',
                'ActualTCPSpeed',
                'ActualTCPForce',
                'ActualQ',
                'ActualQd',
            ]
        example = dict()
        for key in config.receive_keys:
            example[key] = np.array(getattr(rtde_r, 'get' + key)())
        example['timestamp'] = time.time()
        self.robot_out_rb = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=config.shm_manager,
            examples=example,
            get_max_k=config.get_max_k,
            get_time_budget=0.4,
            put_desired_frequency=config.frequency
        )
        self.gripper_out_rb = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=config.shm_manager,
            examples={"width_mm": 0.0, "timestamp": time.time()},
            get_max_k=config.get_max_k,
            get_time_budget=0.4,
            put_desired_frequency=config.gripper_frequency
        )

        # 3) Controller state: last TaskFrameCommand, task‐frame state, gains, etc.
        self.T_WF = np.eye(4)  # world←task
        self.mode = np.array([AxisMode.IMPEDANCE_VEL] * 6, dtype=np.int8)
        self.target = np.zeros(6)  # in task frame
        self.kp = np.array([2500, 2500, 2500, 100, 100, 100])
        self.kd = np.array([80, 80, 80, 8, 8, 8])
        self.last_robot_cmd = TaskFrameCommand(
            target=np.zeros((6,)),
            mode=[AxisMode.IMPEDANCE_VEL] * 6,
            kp=self.kp,
            kd=self.kd,
            T_WF=np.eye(4)
        )

        # 4) transform bounds
        self._min_pose_rpy = R.from_rotvec(np.array(self.config.min_pose[3:6])).as_euler('xyz')
        self._max_pose_rpy = R.from_rotvec(np.array(self.config.max_pose[3:6])).as_euler('xyz')

    # =========== launch & shutdown =============
    def connect(self):
        self.start()

    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        # Send a STOP command
        msg = {'cmd': Command.STOP.value}
        self.robot_cmd_queue.put(msg)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.config.launch_timeout)
        assert self.is_alive()

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # =========== context manager ============
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # =========== sending a new TaskFrameCommand ============
    def send_cmd(self, cmd: TaskFrameCommand | GripperCommand):
        """
        Merges the incoming cmd fields into the last_robot_cmd,
        then pushes the updated last_robot_cmd into the shared queue.
        """

        if isinstance(cmd, TaskFrameCommand):
            if self.last_robot_cmd is None:
                # First time ever: store a full copy
                self.last_robot_cmd = TaskFrameCommand(
                    cmd=cmd.cmd,
                    T_WF=cmd.T_WF.copy(),
                    mode=cmd.mode.copy(),
                    target=cmd.target.copy(),
                    kp=cmd.kp.copy(),
                    kd=cmd.kd.copy()
                )
            else:
                # Only update the fields that are not None
                self.last_robot_cmd.update(cmd)

            # Push the entire updated struct into the queue
            self.robot_cmd_queue.put(self.last_robot_cmd.to_queue_dict())

        elif isinstance(cmd, GripperCommand):
            self.gripper_cmd_queue.put(cmd.to_queue_dict())

    # =========== get robot state from ring buffer ============
    def get_robot_state(self, k=None, out=None):
        if k is None:
            return self.robot_out_rb.get(out=out)
        else:
            return self.robot_out_rb.get_last_k(k=k, out=out)

    def get_all_robot_states(self):
        return self.robot_out_rb.get_all()

    def get_gripper_state(self, k=None, out=None):
        if k is None:
            return self.gripper_out_rb.get(out=out)
        else:
            return self.gripper_out_rb.get_last_k(k=k, out=out)

    def get_all_gripper_states(self):
        return self.gripper_out_rb.get_all()

    # ========= main loop in process ============
    def run(self):
        # 1) Enable soft real‐time (optional)
        if self.config.soft_real_time:
            os.sched_setaffinity(0, {self.config.rt_core})
            os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(20))
            # no need for psutil().nice(-priority) if not root

        # 2) Start RTDEControl & RTDEReceive
        if self.config.mock:
            from tests.motors.mock_ur_rtde import RTDEControlInterface, RTDEReceiveInterface
        else:
            from rtde_control import RTDEControlInterface
            from rtde_receive import RTDEReceiveInterface

        robot_ip = self.config.robot_ip
        frequency = self.config.frequency
        dt = 1.0 / frequency
        rtde_c = RTDEControlInterface(robot_ip, frequency)
        rtde_r = RTDEReceiveInterface(robot_ip)

        # 3) Connect to gripper
        if self.config.use_gripper:
            self.gripper = RobotiqGripper(rtde_c)

        try:
            if self.config.verbose:
                print(f"[RTDETFFController] Connecting to {robot_ip}…")

            # 4) Set TCP offset & payload (if provided)
            if self.config.tcp_offset_pose is not None:
                rtde_c.setTcp(self.config.tcp_offset_pose)
            if self.config.payload_mass is not None:
                if self.config.payload_cog is not None:
                    assert rtde_c.setPayload(self.config.payload_mass, self.config.payload_cog)
                else:
                    assert rtde_c.setPayload(self.config.payload_mass)

            # 5) Enter impedance loop via forceMode

            # 5.1) Initialize target pose = current task pose (so we start from zero error)
            pose_W = np.array(rtde_r.getActualTCPPose())  # [x, y, z, Rx, Ry, Rz] in world
            x_cmd = pose_W.copy()
            self.mode = np.array([AxisMode.POS] * 6, dtype=np.int8)
            self.target = x_cmd.copy()  # in task frame

            # 5.2) Put the robot into 6D forceMode (zero‐wrench to begin)
            rtde_c.forceMode(
                to_sixvec(self.T_WF),
                [1, 1, 1, 1, 1, 1],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                2,
                self.config.wrench_limits
            )
            self.force_on = True

            # 5.3) Mark the loop as “ready” from the first successful iteration
            iter_idx = 0
            keep_running = True
            last_gripper_time = time.monotonic()

            # 5.4) Prepare for jitter logging
            hist = collections.deque(maxlen=1000)
            t_prev = time.monotonic()
            log_interval = 1.0
            next_log_time = t_prev + log_interval

            # 6) Start main control loop
            while keep_running:
                t_loop_start = rtde_c.initPeriod()

                # 6.1) Jitter measurement
                t_now = time.monotonic()
                dt_loop = t_now - t_prev
                hist.append(dt_loop)
                t_prev = t_now

                # 6.2) read any pending commands
                try:
                    msgs = self.robot_cmd_queue.get_all()
                    n_cmd = len(msgs['cmd'])
                except Empty:
                    n_cmd = 0

                for i in range(n_cmd):
                    single = {k: msgs[k][i] for k in msgs}
                    cmd_id = int(single['cmd'])
                    if cmd_id == Command.STOP.value:
                        keep_running = False
                        break

                    # Only SET is supported besides STOP
                    elif cmd_id == Command.SET.value:
                        # Update any fields that arrived in the queue
                        # (T_WF, mode, target, kp, kd)
                        new_T = single.get('T_WF', None)
                        if new_T is not None:
                            self.T_WF = new_T.copy()

                        # mode: 6×int8
                        new_mode = single.get('mode', None)
                        if new_mode is not None:
                            # reset virtual position when switching to force-based velocity control
                            for i in range(6):
                                if new_mode[i] != self.mode[i] and new_mode[i] == AxisMode.IMPEDANCE_VEL:
                                    x_cmd[i] = pose_W[i]
                            self.mode = new_mode.copy()

                        # target: 6×float64
                        new_target = single.get('target', None)
                        if new_target is not None:
                            self.target = new_target.copy()

                        # kp, kd gains
                        new_kp = single.get('kp', None)
                        if new_kp is not None:
                            self.kp = new_kp.copy()
                        new_kd = single.get('kd', None)
                        if new_kd is not None:
                            self.kd = new_kd.copy()

                        if self.config.verbose:
                            print("[RTDETFFController] Received SET, updated task‐frame state.")

                    else:
                        # Unknown command → treat as STOP
                        keep_running = False
                        break

                if not keep_running:
                    break

                # 6.3) read current state (world)
                pose_W = np.array(rtde_r.getActualTCPPose())  # [x, y, z, Rx, Ry, Rz]
                vel_W = np.array(rtde_r.getActualTCPSpeed())  # [vx, vy, vz, ωx, ωy, ωz]

                # 6.4) update virtual position

                # --- translation ---
                for i in range(3):
                    mode_i = AxisMode(self.mode[i])
                    if mode_i == AxisMode.POS:
                        x_cmd[i] = self.target[i]
                    elif mode_i == AxisMode.IMPEDANCE_VEL:
                        x_cmd[i] += float(self.target[i]) * dt
                    elif mode_i == AxisMode.PURE_VEL or mode_i == AxisMode.FORCE:
                        pass  # we do not track a virtual position in these modes

                # --- rotation (SO(3) integration) ---
                for i in range(3, 6):
                    mode_i = AxisMode(self.mode[i])
                    if mode_i == AxisMode.POS:
                        x_cmd[i] = self.target[i]
                    elif mode_i == AxisMode.IMPEDANCE_VEL:
                        pass  # we integrate omega afterwards
                    elif mode_i == AxisMode.PURE_VEL or mode_i == AxisMode.FORCE:
                        pass  # we do not track a virtual position in these modes

                mask_vel = np.array([1 if AxisMode(self.mode[i]) == AxisMode.IMPEDANCE_VEL else 0 for i in range(3, 6)])
                if np.any(mask_vel):
                    R_cmd = R.from_rotvec(x_cmd[3:6])
                    dR = R.from_rotvec(self.target[3:6] * mask_vel * dt)
                    R_cmd = dR * R_cmd
                    x_cmd[3:6] = R_cmd.as_rotvec()

                # --- clamp virtual target pos ---
                x_cmd = self._clip_pose_rpy(x_cmd)

                # 6.5) compute wrench based on mode and target
                wrench_W = np.zeros(6, dtype=np.float64)

                # --- translation ---
                mask_virtual = np.array([1 if AxisMode(self.mode[i]) in (AxisMode.IMPEDANCE_VEL, AxisMode.POS) else 0 for i in range(3, 6)])
                if np.any(mask_virtual):
                    pos_err_vec = x_cmd[:3] - np.array(pose_W[:3])

                for i in range(3):
                    mode_i = AxisMode(self.mode[i])
                    if mode_i == AxisMode.POS:
                        wrench_W[i] = self.kp[i] * pos_err_vec[i] + self.kd[i] * -vel_W[i]
                    elif mode_i == AxisMode.IMPEDANCE_VEL:
                        vel_err = self.target[i] - vel_W[i]
                        wrench_W[i] = self.kp[i] * pos_err_vec[i] + self.kd[i] * vel_err  # we use kd[i] as a “velocity‐gain” here
                    elif mode_i == AxisMode.PURE_VEL:
                        vel_err = self.target[i] - vel_W[i]
                        wrench_W[i] = self.kd[i] * vel_err  # we use kd[i] as a “velocity‐gain” here
                    elif mode_i == AxisMode.FORCE:
                        wrench_W[i] = float(self.target[i])  # directly obey commanded force
                    else:
                        wrench_W[i] = 0.0  # safety fallback

                # --- rotation ---
                if np.any(mask_virtual):
                    R_cmd = R.from_rotvec(x_cmd[3:6])
                    R_act = R.from_rotvec(pose_W[3:6])
                    R_err = R_cmd * R_act.inv()
                    rot_err_vec = R_err.as_rotvec()

                for i in range(3, 6):
                    mode_i = AxisMode(self.mode[i])
                    if mode_i == AxisMode.POS:
                        wrench_W[i] = self.kp[i] * rot_err_vec[i - 3] + self.kd[i] * -vel_W[i]
                    elif mode_i == AxisMode.IMPEDANCE_VEL:
                        vel_err = self.target[i] - vel_W[i]
                        wrench_W[i] = self.kp[i] * rot_err_vec[i - 3] + self.kd[i] * vel_err  # we use kd[i] as a “velocity‐gain” here
                    elif mode_i == AxisMode.PURE_VEL:
                        vel_err = self.target[i] - vel_W[i]
                        wrench_W[i] = self.kd[i] * vel_err  # we use kd[i] as a “velocity‐gain” here
                    elif mode_i == AxisMode.FORCE:
                        wrench_W[i] = float(self.target[i])  # directly obey commanded wrench
                    else:
                        wrench_W[i] = 0.0  # safety fallback

                # --- zero wrench if current pose exceeds bounds
                self._apply_wrench_bounds_rpy(pose_W, wrench_W)

                # 6.6) send the wrench to the UR via forceMode(...)
                if not self.force_on:
                    # If for some reason we dropped out of forceMode, re‐enter it
                    rtde_c.forceMode(
                        to_sixvec(self.T_WF),
                        [1, 1, 1, 1, 1, 1],
                        wrench_W.tolist(),
                        2,
                        self.config.wrench_limits
                    )
                    self.force_on = True
                else:
                    # Simply update the wrench each cycle
                    rtde_c.forceMode(
                        to_sixvec(self.T_WF),
                        [1, 1, 1, 1, 1, 1],
                        wrench_W.tolist(),
                        2,
                        self.config.wrench_limits
                    )

                # 6.7) put current state in the ring buffer
                state = {}
                for key in self.config.receive_keys:
                    state[key] = np.array(getattr(rtde_r, 'get' + key)())
                state['timestamp'] = time.time()
                self.robot_out_rb.put(state)

                # 6.8) Jitter print every log_interval
                if t_now >= next_log_time and len(hist) >= 10:
                    arr = np.array(hist)
                    print(f"[Loop Jitter] μ={arr.mean() * 1000:.2f} ms  σ={arr.std() * 1000:.2f} ms  "
                          f"min={arr.min() * 1000:.2f} ms  max={arr.max() * 1000:.2f} ms")
                    next_log_time = t_now + log_interval
                    
                # 6.9) Handle robotiq gripper

                # only poll every 1 / gripper_frequency seconds
                gripper_dt = 1.0 / self.config.gripper_frequency
                if (t_now - last_gripper_time) >= gripper_dt:
                    last_gripper_time = t_now

                    # 6.9.1) Read all pending gripper commands
                    try:
                        gr_msgs = self.gripper_cmd_queue.get_all()
                        n_gr = len(gr_msgs['cmd'])
                    except Empty:
                        n_gr = 0

                    for j in range(n_gr):
                        single = {k: gr_msgs[k][j] for k in gr_msgs}
                        cmd_id = int(single['cmd'])
                        if cmd_id == Command.OPEN.value:
                            self.gripper.open()
                        elif cmd_id == Command.CLOSE.value:
                            self.gripper.close()
                        elif cmd_id == Command.SET.value:
                            self.gripper.move(single["value"])

                    # 6.9.2) Push back the current gripper state
                    #gr_state = {
                    #    'width_mm': float(self.gripper.get_pos()),
                    #    'timestamp': t_now
                    #}
                    #self.gripper_out_rb.put(gr_state)

                # 6.10) After first iteration signal ready
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                # 6.11) regulate loop frequency
                rtde_c.waitPeriod(t_loop_start)

            # end of while keep_running
        finally:
            # 7) cleanup: exit force‐mode, disconnect RTDE
            if self.force_on:
                rtde_c.forceModeStop()
            rtde_c.disconnect()
            rtde_r.disconnect()
            self.ready_event.set()
            if self.config.verbose:
                print(f"[RTDETFFController] Disconnected from robot {robot_ip}")

    @staticmethod
    def _rotvec_to_rpy(rv: np.ndarray) -> np.ndarray:
        """rotation-vector → roll-pitch-yaw (xyz, radians)."""
        return R.from_rotvec(rv).as_euler('xyz', degrees=False)

    @staticmethod
    def _rpy_to_rotvec(rpy: np.ndarray) -> np.ndarray:
        """roll-pitch-yaw → rotation-vector (axis-angle)."""
        return R.from_euler('xyz', rpy, degrees=False).as_rotvec()

    def _clip_pose_rpy(self, pose: np.ndarray) -> np.ndarray:
        """Clamp translation per-axis; clamp rotation in RPY space, return rot-vec form."""
        out = pose.copy()

        # --- translation ---
        out[:3] = np.clip(
            out[:3],
            np.array(self.config.min_pose[:3]),
            np.array(self.config.max_pose[:3])
        )

        # --- rotation (do clamp in Euler) ---
        rpy = self._rotvec_to_rpy(out[3:6])
        rpy = np.clip(rpy, self._min_pose_rpy, self._max_pose_rpy)
        out[3:6] = self._rpy_to_rotvec(rpy)
        return out

    def _apply_wrench_bounds_rpy(self, pose: np.ndarray, wrench: np.ndarray):
        """
        Zero individual wrench components that would push the TCP farther
        outside its per-axis (xyz + RPY) bounds.
        """
        # ----- translation axes -----
        for i in range(3):
            if pose[i] > self.config.max_pose[i] and wrench[i] > 0:
                wrench[i] = 0.0
            elif pose[i] < self.config.min_pose[i] and wrench[i] < 0:
                wrench[i] = 0.0

        # ----- rotation axes (convert to Euler first) -----
        # we ignore rotation axes for that, bc I do not care. Why would you force control rotation anyway?


def to_sixvec(T):
    """
    Convert a 4x4 homogeneous transformation matrix into a 6-vector:
    [tx, ty, tz, rx, ry, rz], where (rx, ry, rz) is the rotation vector (axis-angle).

    Parameters:
    -----------
    T : numpy.ndarray
        4x4 homogeneous transformation matrix

    Returns:
    --------
    six_vec : numpy.ndarray
        6-element vector: [tx, ty, tz, rx, ry, rz]
    """
    if T.shape != (4, 4):
        raise ValueError("Input must be a 4x4 matrix.")

    # 1) Extract the translation component
    t = T[:3, 3]  # (tx, ty, tz)

    # 2) Extract the 3×3 rotation sub‐matrix
    R_mat = T[:3, :3]

    # 3) Convert rotation matrix → rotation vector (axis * angle)
    rot = R.from_matrix(R_mat)
    rot_vec = rot.as_rotvec()  # (rx, ry, rz)

    # 4) Concatenate translation and rotation vector into a single 6-vector
    six_vec = np.concatenate((t, rot_vec))
    return list(six_vec)


def _validate_config(config: URArmConfig) -> URArmConfig:
    assert 0 < config.frequency <= 500
    assert 0.03 <= config.lookahead_time <= 0.2
    if config.tcp_offset_pose is not None:
        config.tcp_offset_pose = np.array(config.tcp_offset_pose)
        assert config.tcp_offset_pose.shape == (6,)
    if config.payload_mass is not None:
        assert 0 <= config.payload_mass <= 5
    if config.payload_cog is not None:
        config.payload_cog = np.array(config.payload_cog)
        assert config.payload_cog.shape == (3,)
        assert config.payload_mass is not None
    if config.shm_manager is None:
        config.shm_manager = SharedMemoryManager()
        config.shm_manager.start()
    assert isinstance(config.shm_manager, SharedMemoryManager)
    return config
