"""Updated RTDEInterpolationController with per‑axis task‑frame formalism.
Only the worker file changed – the public TF_UR class will get a helper next.

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
* At most one translational force axis – TF_UR limit is 3, easy to relax
* kp / kd are diagonal
* Units:  metres, rad, N, N·m
"""
import collections
import logging
import os
import time
import enum
import multiprocessing as mp
from dataclasses import dataclass, asdict
from multiprocessing.managers import SharedMemoryManager
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from lerobot.utils.shared_memory import SharedMemoryRingBuffer, SharedMemoryQueue, Empty



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
    ZERO_FT = 4


@dataclass
class TaskFrameCommand:
    """One message = full spec for all 6 DoF"""
    cmd: Command = Command.SET
    T_WF: Optional[list | np.ndarray]         = None  # world→task transform as a 6 vec
    mode: Optional[list[AxisMode]]            = None  # len==6
    target: Optional[list | np.ndarray]       = None  # 6 pos [m/rad], vel [m/s], or force [N]
    kp: Optional[list | np.ndarray]           = None  # 6 proportional gains (position‐error → force)
    kd: Optional[list | np.ndarray]           = None  # 6 derivative gains (velocity‐error → force)
    max_pose_rpy: Optional[list | np.ndarray] = None  # 6 pos [m], rot [rad] in rpy
    min_pose_rpy: Optional[list | np.ndarray] = None  # 6 pos [m], rot [rad] in rpy

    def to_queue_dict(self):
        d = asdict(self)
        try:
            d["cmd"]    = self.cmd.value
            d["T_WF"]   = np.asarray(self.T_WF).astype(np.float64)
            d["mode"]   = np.array([int(m) for m in self.mode], dtype=np.int8)
            d["target"] = np.asarray(self.target).astype(np.float64)
            d["kp"]     = np.asarray(self.kp).astype(np.float64)
            d["kd"]     = np.asarray(self.kd).astype(np.float64)
            d["max_pose_rpy"] = np.asarray(self.max_pose_rpy).astype(np.float64)
            d["min_pose_rpy"] = np.asarray(self.min_pose_rpy).astype(np.float64)
        except Exception as e:
            print(f"TaskFrameCommand seems to be missing fields: {e}")
        return d

    def to_robot_action(self):
        action_dict = {}
        for i, ax in enumerate(["x", "y", "z", "wx", "wy", "wz"]):
            if self.mode[i] == AxisMode.POS:
                action_dict[f"{ax}.pos"] = self.target[i]
            if self.mode[i] == AxisMode.IMPEDANCE_VEL or self.mode[i] == AxisMode.PURE_VEL:
                action_dict[f"{ax}.vel"] = self.target[i]
            if self.mode[i] == AxisMode.FORCE:
                action_dict[f"{ax}.wrench"] = self.target[i]
        return action_dict

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
        if cmd.max_pose_rpy is not None:
            self.max_pose_rpy = cmd.max_pose_rpy
        if cmd.min_pose_rpy is not None:
            self.min_pose_rpy = cmd.min_pose_rpy

    @classmethod
    def make_default_cmd(cls):
        return cls(
            cmd=Command.SET,
            T_WF=np.zeros(6),
            mode=[AxisMode.POS]*6,
            target=np.zeros(6),
            kp=np.full(6, 300.0),
            kd=np.full(6, 20.0),
            max_pose_rpy=np.full(6, np.inf),
            min_pose_rpy=np.full(6, -np.inf)
        )


class RTDETFFController(mp.Process):
    """
    An RTDE‐based “task‐frame force‐feedback” controller.  This replaces the old
    servoL/waypoint loop with a full 6D impedance implemented via forceMode(...).
    """

    def __init__(self, config: 'URConfig'):
        config = _validate_config(config)
        super().__init__(name="RTDEPositionalController")
        self.config = config
        self.ready_event = mp.Event()  # “ready” event to signal when the loop has started successfully
        self.force_on = False  # are we currently in forceMode?

        # 1) Build the command queue (TaskFrameCommand messages)
        self.robot_cmd_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=config.shm_manager,
            examples=TaskFrameCommand.make_default_cmd().to_queue_dict(),
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
        example["SetTCPForce"] = np.array([0.0] * 6)
        example['timestamp'] = time.time()
        self.robot_out_rb = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=config.shm_manager,
            examples=example,
            get_max_k=config.get_max_k,
            get_time_budget=0.4,
            put_desired_frequency=config.frequency
        )

        # 3) Controller state: last TaskFrameCommand, task‐frame state, gains, etc.
        self.T_WF = np.zeros((6,))  # world←task
        self.mode = [AxisMode.IMPEDANCE_VEL] * 6
        self.target = np.zeros(6)  # in task frame
        self.kp = np.array([2500, 2500, 2500, 150, 150, 150])
        self.kd = np.array([80, 80, 80, 8, 8, 8])
        self.max_pose_rpy = np.full(6, np.inf)
        self.min_pose_rpy = np.full(6, -np.inf)
        self.last_robot_cmd = TaskFrameCommand(
            target=self.target,
            mode=self.mode,
            kp=self.kp,
            kd=self.kd,
            T_WF=self.T_WF,
            max_pose_rpy=self.max_pose_rpy,
            min_pose_rpy=self.min_pose_rpy
        )


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
    def send_cmd(self, cmd: TaskFrameCommand):
        """
        Merges the incoming cmd fields into the last_robot_cmd,
        then pushes the updated last_robot_cmd into the shared queue.
        """
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

    def zero_ft(self):
        """Tell the controller thread to re‐zero the force‐torque sensor."""
        # We only need the cmd field for ZERO_FT, everything else can be None
        zero_cmd = TaskFrameCommand(cmd=Command.ZERO_FT)
        self.send_cmd(zero_cmd)

    # =========== get robot state from ring buffer ============
    def get_robot_state(self, k=None, out=None):
        if k is None:
            return self.robot_out_rb.get(out=out)
        else:
            return self.robot_out_rb.get_last_k(k=k, out=out)

    def get_all_robot_states(self):
        return self.robot_out_rb.get_all()

    # ========= main loop in process ============
    # noinspection PyUnreachableCode
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
        wrench_W = [0.0] * 6

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

            # 4) Initialize ur force mode

            # 4.1) Initialize target pose = current task pose (so we start from zero error)
            pose_F = self.read_current_state(rtde_r)["ActualTCPPose"]
            x_cmd = pose_F.copy()  # [x, y, z, Rx, Ry, Rz] in task
            self.mode = np.array([AxisMode.POS] * 6, dtype=np.int8)
            self.target = x_cmd.copy()  # in task frame

            # 4.2) Put the robot into 6D forceMode (zero‐wrench to begin)
            rtde_c.forceMode(
                self.T_WF.tolist(),
                [1, 1, 1, 1, 1, 1],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                2,
                self.config.speed_limits
            )
            self.force_on = True

            # 4.3) Mark the loop as “ready” from the first successful iteration
            iter_idx = 0
            keep_running = True

            # 4.4) Prepare for jitter logging
            hist = collections.deque(maxlen=1000)
            t_prev = time.monotonic()
            log_interval = 5.0
            next_log_time = t_prev + log_interval

            # 5) Start main control loop
            while keep_running:
                t_loop_start = rtde_c.initPeriod()

                # 5.1) Jitter measurement
                t_now = time.monotonic()
                dt_loop = t_now - t_prev
                hist.append(dt_loop)
                t_prev = t_now

                # 5.2) read any pending commands
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

                    elif cmd_id == Command.ZERO_FT.value:
                        rtde_c.zeroFtSensor()
                        continue

                    # Only SET is supported besides STOP
                    elif cmd_id == Command.SET.value:
                        # Update any fields that arrived in the queue
                        # (T_WF, mode, target, gains, bounds)
                        new_T = single.get('T_WF', None)
                        if new_T is not None:
                            self.T_WF = new_T.copy()
                            pose_F = self.read_current_state(rtde_r)["ActualTCPPose"]

                        # mode: 6×int8
                        new_mode = single.get('mode', None)
                        if new_mode is not None:
                            # reset virtual position when switching to force-based velocity control
                            for i in range(6):
                                if new_mode[i] != self.mode[i] and new_mode[i] == AxisMode.IMPEDANCE_VEL:
                                    x_cmd[i] = pose_F[i]
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

                        # bounds
                        new_max_pose_rpy = single.get('max_pose_rpy', None)
                        if new_max_pose_rpy is not None:
                            self.max_pose_rpy = new_max_pose_rpy.copy()
                        new_min_pose_rpy = single.get('min_pose_rpy', None)
                        if new_min_pose_rpy is not None:
                            self.min_pose_rpy = new_min_pose_rpy.copy()

                    else:
                        # Unknown command → treat as STOP
                        keep_running = False
                        break

                # 5.3) exit loop on stop command
                if not keep_running:
                    break

                # 5.4) read current state (task frame)
                current_state = self.read_current_state(rtde_r)
                pose_F = current_state["ActualTCPPose"]
                v_F = current_state["ActualTCPSpeed"]
                measured_wrench_F = current_state["ActualTCPForce"]

                # read remaining keys
                for key in self.config.receive_keys:
                    if key not in current_state:
                        current_state[key] = np.array(getattr(rtde_r, 'get' + key)())
                current_state["SetTCPForce"] = np.array(wrench_W)
                current_state['timestamp'] = time.time()

                # push new state into the ring buffer
                self.robot_out_rb.put(current_state)

                # 5.5) update virtual position

                # --- translation ---
                for i in range(3):
                    mode_i = AxisMode(self.mode[i])
                    if mode_i == AxisMode.POS:
                        x_cmd[i] = self.target[i]
                    elif mode_i == AxisMode.IMPEDANCE_VEL:
                        x_cmd[i] += float(self.target[i]) * dt
                    elif mode_i == AxisMode.PURE_VEL or mode_i == AxisMode.FORCE:
                        pass  # we do not track a virtual position in these modes

                # --- rotation ---
                for i in range(3, 6):
                    mode_i = AxisMode(self.mode[i])
                    if mode_i == AxisMode.POS:
                        x_cmd[i] = self.target[i]
                    elif mode_i == AxisMode.IMPEDANCE_VEL:
                        pass  # we integrate omega afterwards
                    elif mode_i == AxisMode.PURE_VEL or mode_i == AxisMode.FORCE:
                        pass  # we do not track a virtual position in these modes

                # SO(3) integration for velocity
                mask_vel = np.array([1 if AxisMode(self.mode[i]) == AxisMode.IMPEDANCE_VEL else 0 for i in range(3, 6)])
                if np.any(mask_vel):
                    R_cmd = R.from_rotvec(x_cmd[3:6])
                    dR = R.from_rotvec(self.target[3:6] * mask_vel * dt)
                    R_cmd = dR * R_cmd
                    x_cmd[3:6] = R_cmd.as_rotvec()

                # --- clamp virtual target pos ---
                x_cmd = self.clip_pose(x_cmd)

                # 5.6) compute wrench based on mode and target
                wrench_W = np.zeros(6, dtype=np.float64)

                # --- translation ---
                mask_virtual = np.array([1 if AxisMode(self.mode[i]) in (AxisMode.IMPEDANCE_VEL, AxisMode.POS) else 0 for i in range(3, 6)])
                if np.any(mask_virtual):
                    pos_err_vec = x_cmd[:3] - np.array(pose_F[:3])

                for i in range(3):
                    mode_i = AxisMode(self.mode[i])
                    if mode_i == AxisMode.POS:
                        wrench_W[i] = self.kp[i] * pos_err_vec[i] + self.kd[i] * -v_F[i]
                    elif mode_i == AxisMode.IMPEDANCE_VEL:
                        vel_err = self.target[i] - v_F[i]
                        wrench_W[i] = self.kp[i] * pos_err_vec[i] + self.kd[i] * vel_err  # we use kd[i] as a “velocity‐gain” here
                    elif mode_i == AxisMode.PURE_VEL:
                        vel_err = self.target[i] - v_F[i]
                        wrench_W[i] = self.kd[i] * vel_err  # we use kd[i] as a “velocity‐gain” here
                    elif mode_i == AxisMode.FORCE:
                        wrench_W[i] = float(self.target[i])  # directly obey commanded force
                    else:
                        wrench_W[i] = 0.0  # safety fallback

                # --- rotation ---
                if np.any(mask_virtual):
                    R_cmd = R.from_rotvec(x_cmd[3:6])
                    R_act = R.from_rotvec(pose_F[3:6])
                    R_err = R_cmd * R_act.inv()
                    rot_err_vec = R_err.as_rotvec()

                for i in range(3, 6):
                    mode_i = AxisMode(self.mode[i])
                    if mode_i == AxisMode.POS:
                        wrench_W[i] = self.kp[i] * rot_err_vec[i - 3] + self.kd[i] * -v_F[i]
                    elif mode_i == AxisMode.IMPEDANCE_VEL:
                        vel_err = self.target[i] - v_F[i]
                        wrench_W[i] = self.kp[i] * rot_err_vec[i - 3] + self.kd[i] * vel_err  # we use kd[i] as a “velocity‐gain” here
                    elif mode_i == AxisMode.PURE_VEL:
                        vel_err = self.target[i] - v_F[i]
                        wrench_W[i] = self.kd[i] * vel_err  # we use kd[i] as a “velocity‐gain” here
                    elif mode_i == AxisMode.FORCE:
                        wrench_W[i] = float(self.target[i])  # directly obey commanded wrench
                    else:
                        wrench_W[i] = 0.0  # safety fallback

                # 5.7) bound the wrench based on pose constraints and contact forcesyy
                self.apply_wrench_bounds(pose_F, desired_wrench=wrench_W, measured_wrench=measured_wrench_F)

                # 5.8) command the task space wrench via forceMode(...)
                if not self.force_on:
                    # If for some reason we dropped out of forceMode, re‐enter it
                    rtde_c.forceMode(
                        self.T_WF.tolist(),
                        [1, 1, 1, 1, 1, 1],
                        wrench_W.tolist(),
                        2,
                        self.config.speed_limits
                    )
                    self.force_on = True
                else:
                    # Simply update the wrench each cycle
                    rtde_c.forceMode(
                        self.T_WF.tolist(),
                        [1, 1, 1, 1, 1, 1],
                        wrench_W.tolist(),
                        2,
                        self.config.speed_limits
                    )

                # 5.9) Jitter print every log_interval
                if self.config.verbose and t_now >= next_log_time and len(hist) >= 10:
                    arr = np.array(hist)
                    print(f"[RTDETFFController] Loop Jitter: μ={arr.mean() * 1000:.2f} ms  σ={arr.std() * 1000:.2f} ms  "
                          f"min={arr.min() * 1000:.2f} ms  max={arr.max() * 1000:.2f} ms")
                    next_log_time = t_now + log_interval

                # 5.10) After first iteration signal ready
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                # 5.11) regulate loop frequency
                rtde_c.waitPeriod(t_loop_start)

            # end of while keep_running
        finally:
            # 6) cleanup: exit force‐mode, disconnect RTDE
            try:
                if self.force_on:
                    rtde_c.forceModeStop()
            except Exception:
                pass
            try:
                rtde_c.stopScript()
            except Exception:
                pass
            try:
                rtde_c.disconnect()
            except Exception:
                pass
            try:
                rtde_r.disconnect()
            except Exception:
                pass

            self.ready_event.set()
            if self.config.verbose:
                print(f"[RTDETFFController] Disconnected from robot {robot_ip}")

    def read_current_state(self, rtde_r):
        # 1) get the world→frame 4×4
        T = np.linalg.inv(self.sixvec_to_homogeneous(self.T_WF))
        R_fw = T[:3, :3]        # rotation: world → frame
        t_fw = T[:3,  3]        # translation: world origin in frame coords

        # 2) pose in world and speed
        pose_W = np.array(rtde_r.getActualTCPPose())   # [x,y,z, Rx,Ry,Rz]
        v_W    = np.array(rtde_r.getActualTCPSpeed())  # [vx,vy,vz, ωx,ωy,ωz]

        # 3) pose in frame
        p_W_h = np.hstack((pose_W[:3], 1.0))
        p_F   = T.dot(p_W_h)[:3]
        R_W   = R.from_rotvec(pose_W[3:6]).as_matrix()
        R_F   = R_fw.dot(R_W)
        rotvec_F = R.from_matrix(R_F).as_rotvec()
        pose_F   = np.concatenate((p_F, rotvec_F))

        # 4) twist in frame
        v_F = np.empty(6)
        v_F[:3]  = R_fw.dot(v_W[:3])
        v_F[3:6] = R_fw.dot(v_W[3:6])

        # 5) wrench in world
        wrench_W = np.array(rtde_r.getActualTCPForce())  # [Fx,Fy,Fz, Mx,My,Mz]
        f_W = wrench_W[:3]
        m_TCP = wrench_W[3:]

        # compute frame origin in world (base) coords
        p_frame = -R_fw.T.dot(t_fw)  # your p_task

        # TCP position in world coords
        p_TCP = pose_W[:3]

        # vector from TCP to frame origin
        r = p_frame - p_TCP

        # shift the moment from the TCP to your frame origin
        m_frame = m_TCP + np.cross(r, f_W)

        # now express in your frame axes
        f_F = R_fw.dot(f_W)
        m_F = R_fw.dot(m_frame)

        wrench_F = np.concatenate((f_F, m_F))

        return {
            "ActualTCPPose": pose_F,
            "ActualTCPSpeed": v_F,
            "ActualTCPForce": wrench_F
        }

    def clip_pose(self, pose: np.ndarray) -> np.ndarray:
        """Clamp translation per-axis; clamp rotation in RPY space, return rot-vec form."""
        out = pose.copy()

        # --- translation ---
        out[:3] = np.clip(
            out[:3],
            np.array(self.min_pose_rpy[:3]),
            np.array(self.max_pose_rpy[:3])
        )

        # --- rotation (do clamp in Euler) ---
        rpy = self._rotvec_to_rpy(out[3:6])
        rpy = np.clip(
            rpy,
            np.array(self.min_pose_rpy[3:6]),
            np.array(self.max_pose_rpy[3:6])
        )
        out[3:6] = self._rpy_to_rotvec(rpy)

        return out

    def apply_wrench_bounds(self, pose: np.ndarray, desired_wrench: np.ndarray, measured_wrench: np.ndarray):
        """
        Zero individual wrench components that would push the TCP farther
        outside its per-axis (xyz + RPY) bounds.
        """
        scale_vec = np.array([1.0] * 6)
        for i in range(6):
            if not self.config.enable_contact_aware_force_scaling[i]:
                continue

            f_measured = measured_wrench[i]

            if np.sign(desired_wrench[i]) == np.sign(f_measured):
                f_measured = 0.0

            scale_vec[i] = self.exp_scale(
                abs(f_measured),
                self.config.wrench_limits[i],
                self.config.contact_limit_scale_min[i],
                self.config.contact_limit_scale_theta[i],
            )

        scaled_wrench_limits = scale_vec * np.array(self.config.wrench_limits)

        # ----- translation axes -----
        for i in range(3):
            # hard clip wrench
            desired_wrench[i] = np.clip(desired_wrench[i], -scaled_wrench_limits[i], scaled_wrench_limits[i])

            if pose[i] > self.max_pose_rpy[i] and desired_wrench[i] > 0:
                desired_wrench[i] = 0.0

            elif pose[i] < self.min_pose_rpy[i] and desired_wrench[i] < 0:
                desired_wrench[i] = 0.0

        # ----- rotation axes (convert to Euler first) -----
        for i in range(3, 6):
            desired_wrench[i] = np.clip(desired_wrench[i], -scaled_wrench_limits[i], scaled_wrench_limits[i])
            # we ignore rotation pose limits, bc I do not care. Why would you force control rotation anyway?

        if self.config.debug:
            axis = self.config.debug_axis
            print(
                f"[{['X', 'Y', 'Z', 'A', 'B', 'C'][axis]}-Axis]  "
                f"{'Crtl':<6}: {desired_wrench[axis]:10.3f}   "
                f"{'Meas':<6}: {measured_wrench[axis]:10.3f}   "
                f"{'a':<6}: {scale_vec[axis]:10.3f}   "
                f"{'a * F_max':<10}: {scaled_wrench_limits[axis]:10.3f}"
            )

    @staticmethod
    def _rotvec_to_rpy(rv: np.ndarray) -> np.ndarray:
        """rotation-vector → roll-pitch-yaw (xyz, radians)."""
        return R.from_rotvec(rv).as_euler('xyz', degrees=False)

    @staticmethod
    def _rpy_to_rotvec(rpy: np.ndarray) -> np.ndarray:
        """roll-pitch-yaw → rotation-vector (axis-angle)."""
        return R.from_euler('xyz', rpy, degrees=False).as_rotvec()

    @staticmethod
    def homogenous_to_sixvec(T):
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

    @staticmethod
    def sixvec_to_homogeneous(six_vec):
        """
        Convert a 6-element vector [tx, ty, tz, rx, ry, rz]
        into a 4x4 homogeneous transformation matrix.

        Parameters
        ----------
        six_vec : array-like, shape (6,)
            First three elements are translation [tx,ty,tz];
            last three are rotation vector (axis * angle) [rx,ry,rz].

        Returns
        -------
        T : ndarray, shape (4,4)
            Homogeneous transform:
                [ R  t ]
                [ 0  1 ]
            where R = expmap(rot_vec) and t = [tx,ty,tz].
        """
        six = np.asarray(six_vec, dtype=float)
        if six.shape != (6,):
            raise ValueError(f"Expected 6-vector, got shape {six.shape}")

        # translation
        t = six[:3]

        # rotation matrix from axis-angle
        rot_vec = six[3:]
        R_mat = R.from_rotvec(rot_vec).as_matrix()

        # build homogeneous matrix
        T = np.eye(4, dtype=float)
        T[:3, :3] = R_mat
        T[:3, 3] = t
        return T

    @staticmethod
    def exp_scale(f_meas, f_thresh, s_min=0.2, theta=0.1):
        return s_min + (1 - s_min) * np.exp(-f_meas / theta)


def _validate_config(config: 'URConfig') -> 'URConfig':
    assert 0 < config.frequency <= 500
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
