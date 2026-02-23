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
import gc
import math

import numpy as np
from scipy.spatial.transform import Rotation as R

from lerobot.common.robot_devices.motors.configs import URArmConfig
from lerobot.common.utils.shared_memory import SharedMemoryRingBuffer, SharedMemoryQueue, Empty



# --- timing helpers ---
def _ms(x): return 1000.0 * float(x)

class _PerfWin:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.buf = collections.deque(maxlen=maxlen)

    def add(self, d):
        self.buf.append(d)

    def stats(self):
        if not self.buf:
            return None
        a = np.fromiter(self.buf, dtype=np.float64)
        return {
            "n": int(a.size),
            "mean": float(a.mean()),
            "std": float(a.std()),
            "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)),
            "p99": float(np.percentile(a, 99)),
            "max": float(a.max()),
            "min": float(a.min()),
        }



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


_EXAMPLE_TF_MSG = TaskFrameCommand(
    cmd=Command.SET,
    T_WF=np.zeros(6),
    mode=[AxisMode.POS]*6,
    target=np.zeros(6),
    kp=np.full(6,300.0),
    kd=np.full(6,20.0),
    max_pose_rpy=np.full(6, np.inf),
    min_pose_rpy=np.full(6, -np.inf)
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
        example["ActualTCPForceFiltered"] = np.array([0.0] * 6)
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
        measured_wrench_F = np.zeros(6, dtype=np.float64)

        if self.config.ft_filter_cutoff_hz is None:
            ft_alpha = None
        else:
            fc = float(self.config.ft_filter_cutoff_hz)
            tau = 1.0 / (2.0 * np.pi * max(fc, 1e-6))
            ft_alpha = dt / (tau + dt)


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
            rtde_c.forceModeSetGainScaling(self.config.force_mode_gain_scaling)
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

            # --- config-ish knobs ---
            log_interval = 0.5
            win_secs = 2.0
            win_len = int(win_secs * self.config.frequency)

            # spike thresholds (tune)
            dt_nom = dt
            spike_abs_s = max(0.002, 3.0 * dt_nom)  # absolute dt_loop spike
            spike_rel = 3.0  # dt_loop > spike_rel * dt
            spike_compute_s = max(0.0015, 2.0 * dt_nom)  # compute-time spike (pre-wait)

            # windows for metrics
            dt_win = _PerfWin(win_len)
            compute_win = _PerfWin(win_len)

            # per-section windows
            sec_names = [
                "queue_get", "cmd_apply", "read_state", "recv_extra",
                "rb_put", "virt_update", "wrench", "forcemode", "waitPeriod"
            ]
            sec_wins = {k: _PerfWin(win_len) for k in sec_names}

            t_prev = time.monotonic()
            next_log_time = t_prev + log_interval

            # 5) Start main control loop
            while keep_running:
                t_loop_start = rtde_c.initPeriod()
                t_iter0 = time.monotonic()

                # start-to-start loop dt (jitter)
                dt_loop = t_iter0 - t_prev
                t_prev = t_iter0
                dt_win.add(dt_loop)

                # ---------------- section: queue_get ----------------
                t0 = time.monotonic()
                try:
                    msgs = self.robot_cmd_queue.get_all()
                    n_cmd = len(msgs['cmd'])
                except Empty:
                    msgs = None
                    n_cmd = 0
                sec_wins["queue_get"].add(time.monotonic() - t0)

                # ---------------- section: cmd_apply ----------------
                t0 = time.monotonic()
                if n_cmd:
                    for i in range(n_cmd):
                        single = {k: msgs[k][i] for k in msgs}
                        cmd_id = int(single['cmd'])
                        if cmd_id == Command.STOP.value:
                            keep_running = False
                            break
                        elif cmd_id == Command.ZERO_FT.value:
                            rtde_c.zeroFtSensor()
                            continue
                        elif cmd_id == Command.SET.value:
                            new_T = single.get('T_WF', None)
                            if new_T is not None:
                                self.T_WF = new_T.copy()
                                pose_F = self.read_current_state(rtde_r)["ActualTCPPose"]

                            new_mode = single.get('mode', None)
                            if new_mode is not None:
                                for j in range(6):
                                    if new_mode[j] != self.mode[j] and new_mode[j] == AxisMode.IMPEDANCE_VEL:
                                        x_cmd[j] = pose_F[j]
                                self.mode = new_mode.copy()

                            new_target = single.get('target', None)
                            if new_target is not None:
                                self.target = new_target.copy()

                            new_kp = single.get('kp', None)
                            if new_kp is not None:
                                self.kp = new_kp.copy()
                            new_kd = single.get('kd', None)
                            if new_kd is not None:
                                self.kd = new_kd.copy()

                            new_max_pose_rpy = single.get('max_pose_rpy', None)
                            if new_max_pose_rpy is not None:
                                self.max_pose_rpy = new_max_pose_rpy.copy()
                            new_min_pose_rpy = single.get('min_pose_rpy', None)
                            if new_min_pose_rpy is not None:
                                self.min_pose_rpy = new_min_pose_rpy.copy()
                        else:
                            keep_running = False
                            break
                sec_wins["cmd_apply"].add(time.monotonic() - t0)

                if not keep_running:
                    break

                # ---------------- section: read_state ----------------
                t0 = time.monotonic()
                current_state = self.read_current_state(rtde_r)
                pose_F = current_state["ActualTCPPose"]
                v_F = current_state["ActualTCPSpeed"]
                # filtered wrench
                if ft_alpha is None:
                    measured_wrench_F = current_state["ActualTCPForce"]
                else:
                    measured_wrench_F += ft_alpha * (current_state["ActualTCPForce"] - measured_wrench_F)
                sec_wins["read_state"].add(time.monotonic() - t0)

                # ---------------- section: recv_extra ----------------
                t0 = time.monotonic()
                for key in self.config.receive_keys:
                    if key not in current_state:
                        current_state[key] = np.array(getattr(rtde_r, 'get' + key)())
                current_state["ActualTCPForceFiltered"] = np.array(measured_wrench_F)
                current_state["SetTCPForce"] = np.array(wrench_W)
                current_state['timestamp'] = time.time()
                sec_wins["recv_extra"].add(time.monotonic() - t0)

                # ---------------- section: rb_put ----------------
                t0 = time.monotonic()
                self.robot_out_rb.put(current_state)
                sec_wins["rb_put"].add(time.monotonic() - t0)

                # ---------------- section: virt_update ----------------
                t0 = time.monotonic()

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
                sec_wins["virt_update"].add(time.monotonic() - t0)

                # ---------------- section: wrench ----------------
                t0 = time.monotonic()
                wrench_W = np.zeros(6, dtype=np.float64)

                # compute errors
                err_vec = np.zeros(6, dtype=np.float64)
                err_vec[:3] = x_cmd[:3] - np.array(pose_F[:3])

                R_cmd = R.from_rotvec(x_cmd[3:6])
                R_act = R.from_rotvec(pose_F[3:6])
                R_err = R_cmd * R_act.inv()
                err_vec[3:6] = R_err.as_rotvec()

                for i in range(6):
                    mode_i = AxisMode(self.mode[i])

                    if mode_i == AxisMode.FORCE:
                        wrench_W[i] = float(self.target[i])  # directly obey commanded force
                        continue

                    if mode_i == AxisMode.POS:
                        e = float(err_vec[i])
                        edot = float(-v_F[i])  # desired vel = 0
                    elif mode_i == AxisMode.IMPEDANCE_VEL:
                        e = float(err_vec[i])
                        edot = float(self.target[i] - v_F[i])  # desired vel = target vel
                    elif mode_i == AxisMode.PURE_VEL:
                        e = 0.0
                        edot = float(self.target[i] - v_F[i])
                    else:
                        e = 0.0
                        edot = 0.0

                    if (self.config.compliance_safety_mode == "reference_limits" and
                        self.config.compliance_safety_enable[i]):
                        e, edot = self.clip_reference_errors(e, edot, i)

                    wrench_W[i] = self.kp[i] * e + self.kd[i] * edot

                self.apply_wrench_bounds(pose_F, desired_wrench=wrench_W, measured_wrench=measured_wrench_F)
                sec_wins["wrench"].add(time.monotonic() - t0)

                # ---------------- section: forcemode ----------------
                t0 = time.monotonic()
                rtde_c.forceMode(
                    self.T_WF.tolist(),
                    [1, 1, 1, 1, 1, 1],
                    wrench_W.tolist(),
                    2,
                    self.config.speed_limits
                )
                self.force_on = True
                sec_wins["forcemode"].add(time.monotonic() - t0)

                # compute time (everything before wait)
                t_pre_wait = time.monotonic()
                compute_time = t_pre_wait - t_iter0
                compute_win.add(compute_time)

                # ---------------- section: waitPeriod ----------------
                t0 = time.monotonic()
                rtde_c.waitPeriod(t_loop_start)
                sec_wins["waitPeriod"].add(time.monotonic() - t0)

                if self.config.verbose and t_iter0 >= next_log_time and dt_win.buf:
                    dt_s = dt_win.stats()
                    ct_s = compute_win.stats()

                    # rank sections by p99 or max
                    sec_lines = []
                    for k in sec_names:
                        s = sec_wins[k].stats()
                        if s is None:
                            continue
                        sec_lines.append((k, s["p99"], s["max"], s["mean"]))
                    sec_lines.sort(key=lambda x: x[1], reverse=True)

                    top = sec_lines[:5]
                    top_str = "  ".join([f"{k}:p99={_ms(p99):.2f} max={_ms(mx):.2f}" for k, p99, mx, _ in top])

                    print(
                        f"[RTDETFFController] dt_loop(ms) p50={_ms(dt_s['p50']):.2f} p90={_ms(dt_s['p90']):.2f} "
                        f"p99={_ms(dt_s['p99']):.2f} max={_ms(dt_s['max']):.2f} | "
                        f"compute(ms) p50={_ms(ct_s['p50']):.2f} p99={_ms(ct_s['p99']):.2f} max={_ms(ct_s['max']):.2f} | "
                        f"top: {top_str}"
                    )
                    next_log_time = t_iter0 + log_interval

                # regulate loop frequency
                rtde_c.waitPeriod(t_loop_start)
                iter_idx += 1

                is_dt_spike = (dt_loop > spike_abs_s) or (dt_loop > spike_rel * dt_nom)
                is_compute_spike = (compute_time > spike_compute_s)

                if self.config.verbose and (is_dt_spike or is_compute_spike):
                    # snapshot last section durations (use the most recent appended values)
                    last_secs = {k: (sec_wins[k].buf[-1] if sec_wins[k].buf else float("nan")) for k in sec_names}
                    # find culprit
                    culprit = max(last_secs.items(), key=lambda kv: (0.0 if math.isnan(kv[1]) else kv[1]))

                    gc_counts = gc.get_count()
                    # if you want more: gc.get_stats() is heavier; only do it on spike.
                    # gc_stats = gc.get_stats()

                    print(
                        f"[RTDETFFController][SPIKE] iter={iter_idx} "
                        f"dt_loop={_ms(dt_loop):.2f}ms (dt={_ms(dt_nom):.2f}ms) "
                        f"compute={_ms(compute_time):.2f}ms n_cmd={n_cmd} "
                        f"culprit={culprit[0]}:{_ms(culprit[1]):.2f}ms "
                        f"secs(ms)="
                        + " ".join([f"{k}={_ms(last_secs[k]):.2f}" for k in sec_names])
                        + f" gc_count={gc_counts}"
                    )

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
        if self.config.compliance_safety_mode == "adaptive_limits":
            for i in range(6):
                if not self.config.compliance_safety_enable[i]:
                    continue

                f_measured = measured_wrench[i]

                if np.sign(desired_wrench[i]) == np.sign(f_measured):
                    f_measured = 0.0

                scale_vec[i] = self.exp_scale(
                    abs(f_measured),
                    self.config.wrench_limits[i],
                    self.config.compliance_adaptive_limit_min[i],
                    self.config.compliance_adaptive_limit_theta[i],
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

        if self.config.debug:
            axis = self.config.debug_axis
            print(
                f"[{['X', 'Y', 'Z', 'A', 'B', 'C'][axis]}-Axis]  "
                f"{'Crtl':<6}: {desired_wrench[axis]:10.3f}   "
                f"{'Meas':<6}: {measured_wrench[axis]:10.3f}   "
                f"{'a':<6}: {scale_vec[axis]:10.3f}   "
                f"{'a * F_max':<10}: {scaled_wrench_limits[axis]:10.3f}"
            )

    def clip_reference_errors(self, e: float, edot: float, i: int) -> tuple[float, float]:
        """
        Limit position/orientation error e and velocity error edot so that
        kp*e and kd*edot cannot exceed +/- fmax (HIL-SERL style reference limiting).
        """
        _kp = self.kp[i]
        _kd = self.kd[i]
        _fmax = self.config.compliance_desired_wrench[i]

        if _fmax <= 0:
            return 0.0, 0.0

        if _kp > 0:
            e = float(np.clip(e, -_fmax / _kp, _fmax / _kp))
        if _kd > 0:
            edot = float(np.clip(edot, -_fmax / _kd, _fmax / _kd))
        return e, edot

    @staticmethod
    def exp_scale(f_meas, f_thresh, s_min=0.2, theta=0.1):
        return s_min + (1 - s_min) * np.exp(-f_meas / theta)

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
    def _rotvec_to_rpy(rv: np.ndarray) -> np.ndarray:
        """rotation-vector → roll-pitch-yaw (xyz, radians)."""
        return R.from_rotvec(rv).as_euler('xyz', degrees=False)

    @staticmethod
    def _rpy_to_rotvec(rpy: np.ndarray) -> np.ndarray:
        """roll-pitch-yaw → rotation-vector (axis-angle)."""
        return R.from_euler('xyz', rpy, degrees=False).as_rotvec()


def _validate_config(config: URArmConfig) -> URArmConfig:
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
