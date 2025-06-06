"""Updated RTDEInterpolationController with per‑axis task‑frame formalism.
Only the worker file changed – the public UR class will get a helper next.

Key additions
-------------
* Mode enum  (POS, VEL, FORCE, FIXED)
* TFCommand  (TF_SET)
* new queue schema (servo + task‑frame)
* internal task‑frame state & mixer inside the 1 kHz loop
* automatic force_mode / endForceMode depending on mode_mask

Assumptions
-----------
* Task frame is static after first TF_SET (no tracking‑mode yet)
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
    TF_SET = 0
    STOP   = 1


@dataclass
class TaskFrameCommand:
    """One message = full spec for all 6 DoF"""
    cmd: Command = Command.TF_SET
    T_WF: np.ndarray | None     = None  # 4×4 homogeneous transform (world→task)
    mode: list[AxisMode] | None = None  # len==6
    target: np.ndarray | None   = None  # 6 pos [m/rad], vel [m/s], or force [N]
    kp: np.ndarray | None       = None  # 6 proportional gains (position‐error → force)
    kd: np.ndarray | None       = None  # 6 derivative gains (velocity‐error → force)

    def to_queue_dict(self):
        d = asdict(self)
        try:
            d["cmd"] = self.cmd.value
            d["T_WF"]   = self.T_WF.astype(np.float64)
            d["mode"]  = np.array([int(m) for m in self.mode], dtype=np.int8)
            d["target"] = self.target.astype(np.float64)
            d["kp"]     = self.kp.astype(np.float64)
            d["kd"]     = self.kd.astype(np.float64)
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


_EXAMPLE_TF_MSG = TaskFrameCommand(
    cmd=Command.TF_SET,
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

        # 1) Build the command queue (TaskFrameCommand messages)
        self.cmd_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=config.shm_manager,
            examples=_EXAMPLE_TF_MSG,
            buffer_size=256
        )

        # 2) Build the ring buffer for streaming back pose/vel/force
        if self.config.mock:
            from tests.motors.mock_ur_rtde import RTDEReceiveInterface
        else:
            from rtde_receive import RTDEReceiveInterface
        print("Trying receive")
        rtde_r = RTDEReceiveInterface(hostname=config.robot_ip)
        print("finished receive")

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
        self.out_rb = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=config.shm_manager,
            examples=example,
            get_max_k=config.get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=config.frequency
        )

        # 3) Controller state: last TaskFrameCommand, task‐frame state, gains, etc.
        self.last_cmd: Optional[TaskFrameCommand] = None
        self.T_WF = np.eye(4)  # world←task
        self.mode = np.array([AxisMode.POS] * 6, dtype=np.int8)
        self.target = np.zeros(6)  # in task frame
        self.kp = np.full(6, 100.0)  # stiffness for POS/FIXED
        self.kd = np.full(6, 0.5)  # damping or vel‐gain
        self.force_on = False  # are we currently in forceMode?

        # “ready” event to signal when the loop has started successfully
        self.ready_event = mp.Event()

    # =========== launch & shutdown =============
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        # Send a STOP command
        msg = {'cmd': Command.STOP.value}
        self.cmd_queue.put(msg)
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
    def send_cmd(self, cmd: TaskFrameCommand):
        """
        Merges the incoming cmd fields into the last_cmd,
        then pushes the updated last_cmd into the shared queue.
        """
        if self.last_cmd is None:
            # First time ever: store a full copy
            self.last_cmd = TaskFrameCommand(
                cmd=cmd.cmd,
                T_WF=cmd.T_WF.copy(),
                mode=cmd.mode.copy(),
                target=cmd.target.copy(),
                kp=cmd.kp.copy(),
                kd=cmd.kd.copy()
            )
        else:
            # Only update the fields that are not None
            self.last_cmd.update(cmd)

        # Push the entire updated struct into the queue
        self.cmd_queue.put(self.last_cmd.to_queue_dict())

    # =========== get robot state from ring buffer ============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.out_rb.get(out=out)
        else:
            return self.out_rb.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.out_rb.get_all()

    # ========= main loop in process ============
    def run(self):
        # 1) Enable soft real‐time (optional)
        if self.config.soft_real_time:
            os.sched_setaffinity(0, {self.config.rt_core})
            os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(20))
            # no need for psutil().nice(-20) if not root

        # 2) Start RTDEControl & RTDEReceive
        if self.config.mock:
            from tests.motors.mock_ur_rtde import RTDEControlInterface, RTDEReceiveInterface
        else:
            from rtde_control import RTDEControlInterface
            from rtde_receive import RTDEReceiveInterface

        robot_ip = self.config.robot_ip
        rtde_c = RTDEControlInterface(robot_ip, self.config.frequency)
        rtde_r = RTDEReceiveInterface(robot_ip)

        # R_WF: world ← task rotation (3×3)
        R_WF = lambda: R.from_matrix(self.T_WF[:3, :3])
        # Task‐frame → world wrench/velocity
        #   If v_task = [v_x, v_y, v_z, ωx, ωy, ωz] in task frame, then
        #     v_world_trans = R_WF.apply(v_task_trans)
        #     v_world_rot   = R_WF.apply(v_task_rot)
        #to_world = lambda v6: np.hstack((
        #    R_WF().apply(v6[:3]),
        #    R_WF().apply(v6[3:])
        #))
        # World → task:  invert the rotation
        #to_task = lambda v6: np.hstack((
        #    R_WF().inv().apply(v6[:3]),
        #    R_WF().inv().apply(v6[3:])
        #))

        try:
            if self.config.verbose:
                print(f"[RTDETFFController] Connecting to {robot_ip}…")

            # 3) Set TCP offset & payload (if provided)
            if self.config.tcp_offset_pose is not None:
                rtde_c.setTcp(self.config.tcp_offset_pose)
            if self.config.payload_mass is not None:
                if self.config.payload_cog is not None:
                    assert rtde_c.setPayload(self.config.payload_mass, self.config.payload_cog)
                else:
                    assert rtde_c.setPayload(self.config.payload_mass)

            # 4) Optionally move to an initial joint pose
            if self.config.joints_init is not None:
                assert rtde_c.moveJ(self.config.joints_init, self.config.joints_init_speed, 1.4)

            # 5) Enter impedance loop via forceMode
            frequency = self.config.frequency
            dt = 1.0 / frequency

            # (a) Read the robot’s current pose & twist in world, and convert into task frame:
            pose_W = np.array(rtde_r.getActualTCPPose())  # [x, y, z, Rx, Ry, Rz] in world
            vel_W = np.array(rtde_r.getActualTCPSpeed())  # [vx, vy, vz, ωx, ωy, ωz] in world

            # (b) Initialize TARGET‐POSE = current task pose (so we start from zero error)
            x_cmd = pose_W.copy()
            self.mode = np.array([AxisMode.POS] * 6, dtype=np.int8)
            self.target = x_cmd.copy()  # in task frame

            # (c) Put the robot into 6D forceMode (zero‐wrench to begin)
            rtde_c.forceMode(
                to_sixvec(self.T_WF),
                [1, 1, 1, 0, 0, 0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                2,
                [5.0, 5.0, 5.0, 0.5, 0.5, 0.5]  # change these force/torque limits as needed
            )
            self.force_on = True

            # (d) Mark the loop as “ready” from the first successful iteration
            iter_idx = 0
            keep_running = True

            # Prepare for jitter logging
            hist = collections.deque(maxlen=1000)
            t_prev = time.monotonic()
            log_interval = 1.0
            next_log_time = t_prev + log_interval

            while keep_running:
                t_loop_start = rtde_c.initPeriod()

                # Jitter measurement
                t_now = time.monotonic()
                dt_loop = t_now - t_prev
                hist.append(dt_loop)
                t_prev = t_now

                # 6.1) READ ANY PENDING TF_SET / STOP COMMANDS
                try:
                    msgs = self.cmd_queue.get_all()
                    n_cmd = len(msgs['cmd'])
                except Empty:
                    n_cmd = 0

                for i in range(n_cmd):
                    single = {k: msgs[k][i] for k in msgs}
                    cmd_id = int(single['cmd'])
                    if cmd_id == Command.STOP.value:
                        keep_running = False
                        break

                    # Only TF_SET is supported besides STOP
                    elif cmd_id == Command.TF_SET.value:
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
                                if new_mode[i] == AxisMode.IMPEDANCE_VEL:
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
                            print("[RTDETFFController] Received TF_SET, updated task‐frame state.")

                    else:
                        # Unknown command → treat as STOP
                        keep_running = False
                        break

                if not keep_running:
                    break

                # 6.2) READ CURRENT ROBOT STATE (WORLD → TASK)
                pose_W = np.array(rtde_r.getActualTCPPose())  # [x, y, z, Rx, Ry, Rz]
                vel_W = np.array(rtde_r.getActualTCPSpeed())  # [vx, vy, vz, ωx, ωy, ωz]

                # 6.3) UPDATE “virtual desired pose” x_cmd based on mode & target
                for i in range(6):
                    mode_i = AxisMode(self.mode[i])
                    if mode_i == AxisMode.POS:
                        x_cmd[i] = self.target[i]
                    elif mode_i == AxisMode.IMPEDANCE_VEL:
                        x_cmd[i] += float(self.target[i]) * dt
                    elif mode_i == AxisMode.FORCE or mode_i == AxisMode.PURE_VEL:
                        # we do NOT move the virtual target: we only use target[i] as a desired wrench
                        pass

                # 6.4) COMPUTE THE 6×1 IMPEDANCE WRENCH IN TASK FRAME
                F_task = np.zeros(6, dtype=np.float64)
                errors = np.zeros(6, dtype=np.float64)
                for i in range(6):
                    mode_i = AxisMode(self.mode[i])
                    if mode_i == AxisMode.POS:
                        pos_err = x_cmd[i] - pose_W[i]
                        errors[i] = pos_err
                        F_task[i] = self.kp[i] * pos_err + self.kd[i] * -vel_W[i]
                    elif mode_i == AxisMode.IMPEDANCE_VEL:
                        pos_err = x_cmd[i] - pose_W[i]
                        vel_err = float(self.target[i]) - vel_W[i]

                        # We use kd[i] as a “velocity‐gain” here
                        F_task[i] = self.kp[i] * pos_err + self.kd[i] * vel_err
                    elif mode_i == AxisMode.PURE_VEL:
                        vel_err = float(self.target[i]) - vel_W[i]
                        # We use kd[i] as a “velocity‐gain” here
                        F_task[i] = self.kd[i] * vel_err
                    elif mode_i == AxisMode.FORCE:
                        # directly obey the commanded wrench in task frame
                        F_task[i] = float(self.target[i])
                    else:
                        # safety fallback
                        F_task[i] = 0.0

                # 6.5) TRANSLATE F_task (in TASK) → F_world → F_tool
                #   (a) TASK→WORLD: rotate each 3-vector by R_WF
                #   (b) WORLD→TOOL: rotate by R_base→tool (extract from pose_W[3:])
                #F_world = to_world(F_task)

                # Build world→tool rotation:
                #R_world2tool = R.from_rotvec(pose_W[3:]).inv()
                #F_tool_trans = R_world2tool.apply(F_world[:3])
                #F_tool_rot = R_world2tool.apply(F_world[3:])
                #wrench_tool = np.hstack((F_tool_trans, F_tool_rot))

                wrench_tool = F_task
                #print("Error", errors[:3], "\nForce_W", F_task[:3], "\nForce_T", wrench_tool[:3], "\n", "=" * 10)

                # 6.6) SEND THE WRENCH TO UR VIA forceMode(...)
                if not self.force_on:
                    # If for some reason we dropped out of forceMode, re‐enter it
                    rtde_c.forceMode(
                        to_sixvec(self.T_WF),
                        [1, 1, 1, 0, 0, 0],
                        wrench_tool.tolist(),
                        2,
                        [5.0, 5.0, 5.0, 0.5, 0.5, 0.5]  # change these force/torque limits as needed
                    )
                    self.force_on = True
                else:
                    # Simply update the wrench each cycle
                    rtde_c.forceMode(
                        to_sixvec(self.T_WF),
                        [1, 1, 1, 0, 0, 0],
                        wrench_tool.tolist(),
                        2,
                        [5.0, 5.0, 5.0, 0.5, 0.5, 0.5]  # change these force/torque limits as needed
                    )

                # 6.7) PUT CURRENT STATE INTO THE RING BUFFER FOR EXTERNAL READERS
                state = {}
                for key in self.config.receive_keys:
                    state[key] = np.array(getattr(rtde_r, 'get' + key)())
                state['timestamp'] = time.time()
                self.out_rb.put(state)

                # 6.8) Jitter print every log_interval
                if t_now >= next_log_time and len(hist) >= 10:
                    arr = np.array(hist)
                    print(f"[Loop Jitter] μ={arr.mean() * 1000:.2f} ms  σ={arr.std() * 1000:.2f} ms  "
                          f"min={arr.min() * 1000:.2f} ms  max={arr.max() * 1000:.2f} ms")
                    next_log_time = t_now + log_interval

                # 6.9) REGULATE LOOP FREQUENCY
                rtde_c.waitPeriod(t_loop_start)

                # 6.10) AFTER FIRST ITERATION, SIGNAL “READY”
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                if self.config.verbose:
                    # measure actual frequency
                    freq_actual = 1.0 / (time.perf_counter() - t_loop_start + 1e-9)
                    print(f"[RTDETFFController] Loop {iter_idx}, freq = {freq_actual:.1f} Hz")

            # end of while keep_running

        finally:
            # 7) CLEANUP: exit force‐mode, disconnect RTDE
            if self.force_on:
                rtde_c.forceModeStop()
            rtde_c.disconnect()
            rtde_r.disconnect()
            self.ready_event.set()
            if self.config.verbose:
                print(f"[RTDETFFController] Disconnected from robot {robot_ip}")


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
    assert 0 < config.max_pos_speed
    assert 0 < config.max_rot_speed
    if config.tcp_offset_pose is not None:
        config.tcp_offset_pose = np.array(config.tcp_offset_pose)
        assert config.tcp_offset_pose.shape == (6,)
    if config.payload_mass is not None:
        assert 0 <= config.payload_mass <= 5
    if config.payload_cog is not None:
        config.payload_cog = np.array(config.payload_cog)
        assert config.payload_cog.shape == (3,)
        assert config.payload_mass is not None
    if config.joints_init is not None:
        config.joints_init = np.array(config.joints_init)
        assert config.joints_init.shape == (6,)
    if config.shm_manager is None:
        config.shm_manager = SharedMemoryManager()
        config.shm_manager.start()
    assert isinstance(config.shm_manager, SharedMemoryManager)
    return config
