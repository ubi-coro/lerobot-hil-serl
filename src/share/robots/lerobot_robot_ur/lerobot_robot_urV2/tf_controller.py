import collections
import math
import gc
import os
import time
import enum
import multiprocessing as mp
from dataclasses import dataclass, asdict, replace
from multiprocessing.managers import SharedMemoryManager

import numpy as np
from scipy.spatial.transform import Rotation as R

from lerobot.utils.shared_memory import SharedMemoryRingBuffer, SharedMemoryQueue, Empty
from share.envs.manipulation_primitive.task_frame import ControlMode, PolicyMode, TaskFrame



# ---------------------------------------------------------------------------
# Internal enums
# ---------------------------------------------------------------------------
DeltaMode = PolicyMode

class Command(enum.IntEnum):
    SET = 0
    STOP = 1
    OPEN = 2
    CLOSE = 3
    ZERO_FT = 4


@dataclass
class TaskFrameCommand(TaskFrame):
    """Controller command with user-facing rotational pose inputs in RPY.

    The rotational parts of ``origin`` and absolute rotational ``target`` entries
    are specified as XYZ roll-pitch-yaw angles [rad] at the interface. Internally,
    the controller converts them to rotation vectors before use.
    """
    cmd: Command = Command.SET

    @property
    def delta_mode(self) -> list[DeltaMode]:
        """Per-axis absolute/relative interpretation derived from ``policy_mode``."""
        return [DeltaMode.RELATIVE if m  == PolicyMode.RELATIVE else DeltaMode.ABSOLUTE for m in self.policy_mode]

    def to_queue_dict(self):
        """Convert the command to a queue-friendly dict of NumPy arrays and ints.

        The rotational part of ``origin`` is interpreted as user-facing XYZ
        roll-pitch-yaw [rad] and converted to an internal rotation vector.
        """
        d = asdict(self)
        try:
            d["cmd"] = self.cmd.value
            d.pop("policy_mode", None)
            d["space"] = np.asarray(self.space).astype(np.int8)
            d["control_mode"] = np.array([int(m) if m is not None else -1 for m in self.control_mode])
            d["policy_mode"] = np.array([int(m) if m is not None else -1 for m in self.policy_mode])
            d["delta_mode"] = np.array([int(m) if m is not None else -1 for m in self.delta_mode])
            d["target"] = np.asarray(self.target).astype(np.float64)
            d["origin"] = np.asarray(self.origin).astype(np.float64)
            d["origin"][3:6] = R.from_euler("xyz", d["origin"][3:6], degrees=False).as_rotvec()
            d["max_pose"] = np.asarray(self.max_pose).astype(np.float64)
            d["min_pose"] = np.asarray(self.min_pose).astype(np.float64)
            d["kp"] = np.asarray(self.kp).astype(np.float64)
            d["kd"] = np.asarray(self.kd).astype(np.float64)
        except Exception as e:
            raise ValueError(f"TaskFrameCommand seems to be missing fields: {e}")
        return d

    def to_robot_action(self):
        action_dict = {}
        for i, ax in enumerate(["x", "y", "z", "rx", "ry", "rz"]):
            if self.control_mode[i] == ControlMode.POS:
                action_dict[f"{ax}.ee_pos"] = self.target[i]
            elif self.control_mode[i] == ControlMode.VEL:
                action_dict[f"{ax}.ee_vel"] = self.target[i]
            elif self.control_mode[i] == ControlMode.WRENCH:
                action_dict[f"{ax}.ee_wrench"] = self.target[i]
        return action_dict
    

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


class RTDETaskFrameController(mp.Process):
    """RTDE task-frame controller with per-axis modes and 6D impedance.

    Runs a 1 kHz loop that:
      • Reads commands from shared memory (pose/vel/force modes per axis)
      • Estimates current state in the task frame
      • Integrates virtual targets (for IMPEDANCE_VEL)
      • Computes and bounds a wrench, then applies it via `forceMode(...)`

    Notes:
        - Translation bounds are enforced directly; rotation bounds are applied
          in RPY space but the controller operates internally on rot-vectors.
        - Automatically (re)enters `forceMode` as needed.

    Attributes:
        config (URConfig): Runtime configuration (RTDE IP, gains, limits, etc.).
        ready_event (mp.Event): Set once the control loop is alive.
        robot_cmd_queue (SharedMemoryQueue): Incoming `TaskFrameCommand`s.
        robot_out_rb (SharedMemoryRingBuffer): Outgoing robot state samples.
    """

    def __init__(self, config: 'URConfig'):
        """Initialize controller processes, queues, and default internal state.

        Args:
            config (URConfig): Configuration (frequency, limits, payload/TCP, etc.).

        Raises:
            AssertionError: If `config` fields are inconsistent (validated/normalized).
        """

        config = _validate_config(config)
        super().__init__(name="RTDETaskFrameController")
        self.config = config
        self.ready_event = mp.Event()  # “ready” event to signal when the loop has started successfully
        self.force_on = False  # are we currently in forceMode?
        self._receive_keys = [
            'ActualTCPPose',
            'ActualTCPSpeed',
            'ActualTCPForce',
            'ActualQ',
            'ActualQd',
        ]

        # 1) Build the command queue (TaskFrameCommand messages)
        self.robot_cmd_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=config.shm_manager,
            examples=TaskFrameCommand().to_queue_dict(),
            buffer_size=256
        )

        # 2) Build the ring buffer for streaming back pose/vel/force
        if self.config.mock:
            from tests.mocks.mock_ur_rtde import RTDEReceiveInterface
        else:
            from rtde_receive import RTDEReceiveInterface
        rtde_r = RTDEReceiveInterface(hostname=config.robot_ip)

        example = dict()
        for key in self._receive_keys:
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
        self._last_cmd = TaskFrameCommand(
            origin=np.zeros((6,)),
            target=np.zeros((6,)),
            control_mode=[ControlMode.POS] * 6,
            policy_mode=[PolicyMode.RELATIVE] * 6,  # delta mode is derived from that
            kp=np.array([2500, 2500, 2500, 150, 150, 150]),
            kd=np.array([80, 80, 80, 8, 8, 8]),
            max_pose=np.full(6, np.inf),
            min_pose=np.full(6, -np.inf)
        )
        self.origin = self._last_cmd.origin
        self.control_mode = self._last_cmd.control_mode
        self.delta_mode = self._last_cmd.delta_mode
        self.target = self._last_cmd.target
        self.kp = self._last_cmd.kp
        self.kd = self._last_cmd.kd
        self.max_pose = self._last_cmd.max_pose
        self.min_pose = self._last_cmd.min_pose

    # =========== launch & shutdown =============
    def connect(self):
        """Spawn the control process and block until the first iteration completes."""
        self.start()

    def start(self, wait=True):
        """Start the control process.

        Args:
            wait (bool, optional): If True, block until the loop signals readiness.
        """
        super().start()
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        """Request a graceful shutdown of the control loop.

        Args:
            wait (bool, optional): If True, join the process before returning.
        """
        # Send a STOP command
        stop_cmd = replace(self._last_cmd)
        stop_cmd.cmd = Command.STOP
        self.robot_cmd_queue.put(stop_cmd.to_queue_dict())
        if wait:
            self.stop_wait()

    def start_wait(self):
        """Block until the controller signals ready or the launch timeout elapses.

        Raises:
            AssertionError: If the process is not alive after waiting.
        """
        self.ready_event.wait(self.config.launch_timeout)
        assert self.is_alive()

    def stop_wait(self):
        """Join the control process (blocks until termination)."""
        self.join()

    @property
    def is_ready(self):
        """bool: True once the control loop completed its first successful cycle."""
        return self.ready_event.is_set()

    # =========== context manager ============
    def __enter__(self):
        """Context: start controller and return self (blocks until ready)."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context: stop controller on exit, regardless of exceptions."""
        self.stop()

    # =========== sending a new TaskFrameCommand ============
    def send_cmd(self, cmd: TaskFrameCommand):
        """Merge cmd into the last command and push the result to the queue.
        The first call stores a full copy; subsequent calls update only fields
        that are provided (non-None).

        Args:
            cmd (TaskFrameCommand): Partial or full command to apply.
        """
        self._last_cmd = cmd
        self.robot_cmd_queue.put(cmd.to_queue_dict())

    def zero_ft(self):
        """Re-zero the force-torque sensor in the control loop."""
        # We only need the cmd field for ZERO_FT, everything else can be None
        zero_cmd = replace(self._last_cmd)
        zero_cmd.cmd = Command.ZERO_FT
        self.robot_cmd_queue.put(zero_cmd.to_queue_dict())

    # =========== get robot state from ring buffer ============
    def get_robot_state(self, k=None, out=None):
        """Get the latest (or last k) robot state sample(s).

        Args:
            k (int, optional): If `None`, return the latest sample. If an integer,
                return the last `k` samples.
            out (dict, optional): Optional preallocated output buffer.

        Returns:
            dict or tuple[dict,...]: State dict(s) including:
                - ``'ActualTCPPose'`` (6, ) task-frame pose (x,y,z, rx,ry,rz)
                - ``'ActualTCPSpeed'`` (6, ) task-frame twist
                - ``'ActualTCPForce'`` (6, ) task-frame wrench
                - any additional keys requested via `config.receive_keys`
                - ``'SetTCPForce'`` (6, ) last commanded wrench in the task frame
                - ``'timestamp'`` (float)
        """
        if k is None:
            return self.robot_out_rb.get(out=out)
        else:
            return self.robot_out_rb.get_last_k(k=k, out=out)

    def get_all_robot_states(self):
        """Return all buffered robot states currently stored in the ring buffer.
        Returns:
            list[dict]: Chronologically ordered state samples.
        """
        return self.robot_out_rb.get_all()

    # ========= main loop in process ============
    def run(self):
        """Control-loop entry point (child process).

        Steps:
            1) Configure RT scheduling (optional) and connect RTDE.
            2) Initialize `forceMode` and virtual targets.
            3) Loop at `config.frequency`:
               - Drain and apply queued `TaskFrameCommand`s
               - Read current state and write it to the ring buffer
               - Update the virtual task-frame pose
               - Compute per-axis wrench from mode/targets/gains
               - Clamp wrench using pose bounds and contact-aware scaling
               - Apply wrench via `forceMode`
            4) On shutdown, stop force mode and disconnect cleanly.

        Absolute rotational pose targets are interpreted as XYZ roll-pitch-yaw
        angles [rad] at the interface and converted internally to rotation vectors.
        """
        # 1) Enable soft real‐time (optional)
        if self.config.soft_real_time:
            os.sched_setaffinity(0, {self.config.rt_core})
            os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(20))
            # no need for psutil().nice(-priority) if not root

        # 2) Start RTDEControl & RTDEReceive
        if self.config.mock:
            from tests.mocks.mock_ur_rtde import RTDEControlInterface, RTDEReceiveInterface
        else:
            from rtde_control import RTDEControlInterface
            from rtde_receive import RTDEReceiveInterface

        robot_ip = self.config.robot_ip
        frequency = self.config.frequency
        dt = 1.0 / frequency
        rtde_c = RTDEControlInterface(robot_ip, frequency)
        rtde_r = RTDEReceiveInterface(robot_ip)
        wrench_F = [0.0] * 6
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

            # 4.2) Put the robot into 6D forceMode (zero‐wrench to begin)
            rtde_c.forceModeSetGainScaling(self.config.force_mode_gain_scaling)
            rtde_c.forceMode(
                self.origin.tolist(),
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
            log_interval = 5.0
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
                        # (origin, mode, target, gains, bounds)
                        self.origin = single["origin"].copy()
                        self.target = single["target"].copy()
                        self.kp = single["kp"].copy()
                        self.kd = single["kd"].copy()
                        self.max_pose = single["max_pose"].copy()
                        self.min_pose = single["min_pose"].copy()

                        # if switching to position control with relative/velocity mode, reset virtual target to current pose to avoid jumps
                        pose_F = self.read_current_state(rtde_r)["ActualTCPPose"]

                        # modes: 6×int8 each
                        new_control_mode = single["control_mode"]
                        new_delta_mode = single["delta_mode"]

                        # reset virtual position when switching to delta position control
                        for i in range(6):
                            if new_control_mode[i] != self.control_mode[i] and new_control_mode[i] == ControlMode.POS and new_delta_mode[i] == DeltaMode.RELATIVE: 
                                x_cmd[i] = pose_F[i]

                        self.control_mode = new_control_mode.copy()
                        self.delta_mode = new_delta_mode.copy()
                    else:
                        # Unknown command → treat as STOP
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
                for key in self._receive_keys:
                    if key not in current_state:
                        current_state[key] = np.array(getattr(rtde_r, 'get' + key)())
                current_state["ActualTCPForceFiltered"] = np.array(measured_wrench_F)
                current_state["SetTCPForce"] = np.array(wrench_F)
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
                    control_mode_i = ControlMode(self.control_mode[i])
                    delta_mode_i = DeltaMode(self.delta_mode[i])

                    if control_mode_i == ControlMode.POS and delta_mode_i == DeltaMode.ABSOLUTE:
                        x_cmd[i] = self.target[i]
                    elif control_mode_i == ControlMode.POS and delta_mode_i == DeltaMode.RELATIVE:
                        # integrate velocity, or leak back when input velocity is in deadband
                        v_cmd = float(self.target[i])
                        if abs(v_cmd) > self.config.deadband_pos:
                            x_cmd[i] += v_cmd * dt
                        else:
                            # leak virtual target back to actual pose
                            x_cmd[i] += -self.config.leak_rate_pos * (x_cmd[i] - pose_F[i]) * dt
                    elif control_mode_i == ControlMode.VEL or control_mode_i == ControlMode.WRENCH:
                        pass

                # --- rotation ---
                mask_abs_pos = np.array(
                    [
                        ControlMode(self.control_mode[i]) == ControlMode.POS
                        and DeltaMode(self.delta_mode[i]) == DeltaMode.ABSOLUTE
                        for i in range(3, 6)
                    ],
                    dtype=bool,
                )
                if np.any(mask_abs_pos):
                    rpy_cmd = self._rotvec_to_rpy(x_cmd[3:6])
                    target_rpy = np.asarray(self.target[3:6], dtype=float)
                    rpy_cmd[mask_abs_pos] = target_rpy[mask_abs_pos]
                    x_cmd[3:6] = self._rpy_to_rotvec(rpy_cmd)

                # SO(3) integration for angular velocity with deadband + leak
                mask_delta_pos = np.array(
                    [
                        ControlMode(self.control_mode[i]) == ControlMode.POS
                        and DeltaMode(self.delta_mode[i]) == DeltaMode.RELATIVE
                        for i in range(3, 6)
                    ],
                    dtype=bool,
                )
                if np.any(mask_delta_pos):
                    # 1. Extract the commanded angular velocity for relative POS axes only
                    omega = np.zeros(3, dtype=float)
                    omega[mask_delta_pos] = np.array(self.target[3:6], dtype=float)[mask_delta_pos]

                    # 2. Compute the spherical magnitude (L2 norm) of the commanded vector
                    omega_norm = np.linalg.norm(omega)

                    # Start from current virtual orientation
                    R_cmd = R.from_rotvec(x_cmd[3:6])

                    # 3. If the combined velocity exceeds the deadband, we are "driving" the orientation
                    if omega_norm > self.config.deadband_rot:
                        dR_move = R.from_rotvec(omega * dt)
                        R_cmd = dR_move * R_cmd
                    else:
                        # 4. If below the deadband, let the orientation relax (leak) toward the actual pose
                        R_act = R.from_rotvec(pose_F[3:6])
                        R_err = R_act * R_cmd.inv()
                        rot_err_vec = R_err.as_rotvec()
                        
                        # Prevent leaking on axes governed by absolute commands
                        rot_err_vec[~mask_delta_pos] = 0.0
                        
                        # Leak factor in [0, 1]; take a small step along the error vector
                        alpha = np.clip(self.config.leak_rate_rot * dt, 0.0, 1.0)
                        dR_leak = R.from_rotvec(alpha * rot_err_vec)
                        R_cmd = dR_leak * R_cmd

                    # Write back virtual orientation as rotvec
                    x_cmd[3:6] = R_cmd.as_rotvec()

                # --- clamp virtual target pos ---
                x_cmd = self.clip_pose(x_cmd)
                sec_wins["virt_update"].add(time.monotonic() - t0)

                # ---------------- section: wrench ----------------
                t0 = time.monotonic()
                wrench_F = np.zeros(6, dtype=np.float64)

                # compute errors
                err_vec = np.zeros(6, dtype=np.float64)
                err_vec[:3] = x_cmd[:3] - np.array(pose_F[:3])

                R_cmd = R.from_rotvec(x_cmd[3:6])
                R_act = R.from_rotvec(pose_F[3:6])
                R_err = R_cmd * R_act.inv()
                err_vec[3:6] = R_err.as_rotvec()

                for i in range(6):
                    control_mode_i = ControlMode(self.control_mode[i])

                    if control_mode_i == ControlMode.WRENCH:
                        wrench_F[i] = float(self.target[i])  # directly obey commanded force
                        continue

                    if control_mode_i == ControlMode.POS:
                        e = float(err_vec[i])
                        edot = float(-v_F[i])  # desired vel = 0
                    elif control_mode_i == ControlMode.VEL:
                        e = 0.0
                        edot = float(self.target[i] - v_F[i])
                    else:
                        e = 0.0
                        edot = 0.0

                    if (self.config.compliance_safety_mode == "reference_limits" and
                        self.config.compliance_safety_enable[i]):
                        e, edot = self.clip_reference_errors(e, edot, i)

                    wrench_F[i] = self.kp[i] * e + self.kd[i] * edot

                self.apply_wrench_bounds(pose_F, desired_wrench=wrench_F, measured_wrench=measured_wrench_F)
                sec_wins["wrench"].add(time.monotonic() - t0)

                # ---------------- section: forcemode ----------------
                t0 = time.monotonic()
                rtde_c.forceMode(
                    self.origin.tolist(),
                    [1, 1, 1, 1, 1, 1],
                    wrench_F.tolist(),
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
                        f"[RTDETaskFrameController] dt_loop(ms) p50={_ms(dt_s['p50']):.2f} p90={_ms(dt_s['p90']):.2f} "
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
                        f"[RTDETaskFrameController][SPIKE] iter={iter_idx} "
                        f"dt_loop={_ms(dt_loop):.2f}ms (dt={_ms(dt_nom):.2f}ms) "
                        f"compute={_ms(compute_time):.2f}ms n_cmd={n_cmd} "
                        f"culprit={culprit[0]}:{_ms(culprit[1]):.2f}ms "
                        f"secs(ms)="
                        + " ".join([f"{k}={_ms(last_secs[k]):.2f}" for k in sec_names])
                        + f" gc_count={gc_counts}"
                    )

                if not self.ready_event.is_set():
                    self.ready_event.set()
                # end of while keep_running
        finally:
            # cleanup: exit force‐mode, disconnect RTDE
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
                print(f"[RTDETaskFrameController] Disconnected from robot {robot_ip}")

    def read_current_state(self, rtde_r):
        """Read world state from RTDE and express pose/twist/wrench in the task frame.

        Args:
            rtde_r: `RTDEReceiveInterface` (or mock) used to query current state.

        Returns:
            dict: ``{'ActualTCPPose','ActualTCPSpeed','ActualTCPForce'}`` in task frame.
        """
        # 1) get the world→frame 4×4
        T = np.linalg.inv(self.sixvec_to_homogeneous(self.origin))
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
        p_frame = -R_fw.T.dot(t_fw) 

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
        """Clamp translation per-axis and rotation in RPY space; return rot-vector.

        Args:
            pose (np.ndarray): 6-vector (x,y,z, rx,ry,rz) in task frame.

        Returns:
            np.ndarray: Bounded pose as rotation-vector representation.
        """
        out = pose.copy()

        # --- translation ---
        out[:3] = np.clip(
            out[:3],
            np.array(self.min_pose[:3]),
            np.array(self.max_pose[:3])
        )

        # --- rotation (do clamp in Euler) ---
        rpy = self._rotvec_to_rpy(out[3:6])
        rpy = np.clip(
            rpy,
            np.array(self.min_pose[3:6]),
            np.array(self.max_pose[3:6])
        )
        out[3:6] = self._rpy_to_rotvec(rpy)

        return out

    def apply_wrench_bounds(self, pose: np.ndarray, desired_wrench: np.ndarray, measured_wrench: np.ndarray):
        """Contact-aware wrench limiting and boundary protection (in-place).

        Zeroes or scales components that would push the TCP further outside
        position/orientation limits and applies exponential scaling near contact.

        Args:
            pose (np.ndarray): Current task-frame pose (6,).
            desired_wrench (np.ndarray): Computed wrench to be bounded (modified).
            measured_wrench (np.ndarray): Measured task-frame wrench from RTDE.
        """

        scale_vec = np.array([1.0] * 6)
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

            # 2) if outside bounds, project away outward component and add spring back toward bound
            if pose[i] > self.max_pose[i]:
                # remove outward push (positive wrench on + side)
                if desired_wrench[i] > 0.0:
                    desired_wrench[i] = 0.0
                penetration = pose[i] - self.max_pose[i]  # > 0
                desired_wrench[i] += -self.kp[i] * penetration

            elif pose[i] < self.min_pose[i]:
                # remove outward push (negative wrench on - side)
                if desired_wrench[i] < 0.0:
                    desired_wrench[i] = 0.0
                penetration = self.min_pose[i] - pose[i]  # > 0
                desired_wrench[i] += +self.kp[i] * penetration

        # ----- rotation axes (convert to Euler first) -----
        # Operate in Euler to measure penetration; torques are Nm.
        rpy = self._rotvec_to_rpy(pose[3:6]).astype(np.float64)

        # Optional: wrap angles & bounds to [-pi, pi] if you use bounded RPY ranges
        rpy = self.wrap_to_pi(rpy)
        min_rpy = np.array(self.min_pose[3:6], dtype=np.float64)
        max_rpy = np.array(self.max_pose[3:6], dtype=np.float64)

        for j, i in enumerate(range(3, 6)):
            desired_wrench[i] = np.clip(desired_wrench[i], -scaled_wrench_limits[i], scaled_wrench_limits[i])

            # upper bound violation
            if rpy[j] > max_rpy[j]:
                if desired_wrench[i] > 0.0:  # outward (increasing angle)
                    desired_wrench[i] = 0.0
                penetration = rpy[j] - max_rpy[j]  # > 0 (rad)
                desired_wrench[i] += -self.kp[i] * penetration  # Nm

            # lower bound violation
            elif rpy[j] < min_rpy[j]:
                if desired_wrench[i] < 0.0:  # outward (decreasing angle)
                    desired_wrench[i] = 0.0
                penetration = min_rpy[j] - rpy[j]  # > 0 (rad)
                desired_wrench[i] += +self.kp[i] * penetration  # Nm

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
    def homogenous_to_sixvec(T):
        """4×4 homogeneous transform → 6-vector [tx,ty,tz, rx,ry,rz].

        Args:
            T (np.ndarray): Homogeneous matrix (4,4).

        Returns:
            list[float]: Translation + rotation-vector.

        Raises:
            ValueError: If input is not (4,4).
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
        """6-vector [tx,ty,tz, rx,ry,rz] → 4×4 homogeneous transform.

        Args:
            six_vec (array-like): First 3 translation, last 3 rotation-vector.

        Returns:
            np.ndarray: Homogeneous transform (4,4).

        Raises:
            ValueError: If input shape is not (6,).
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
        """Exponential scaling from contact force to [s_min, 1].

        Args:
            f_meas (float): Measured absolute force/moment (≥0).
            f_thresh (float): Nominal limit (unused here; for symmetry with caller).
            s_min (float, optional): Lower bound of scaling ∈ (0,1].
            theta (float, optional): Decay constant; larger → slower decay.

        Returns:
            float: Scale factor in [s_min, 1].
        """
        return s_min + (1 - s_min) * np.exp(-f_meas / theta)

    @staticmethod
    def wrap_to_pi(angles: np.ndarray) -> np.ndarray:
        """Wrap angles [rad] elementwise to (-pi, pi]."""
        out = (angles + np.pi) % (2 * np.pi) - np.pi
        # map -pi to +pi for consistency if desired:
        out[np.isclose(out, -np.pi)] = np.pi
        return out

    @staticmethod
    def _rotvec_to_rpy(rv: np.ndarray) -> np.ndarray:
        """Rotation-vector → roll-pitch-yaw (xyz order, radians)."""
        return R.from_rotvec(rv).as_euler('xyz', degrees=False)

    @staticmethod
    def _rpy_to_rotvec(rpy: np.ndarray) -> np.ndarray:
        """Roll-pitch-yaw (xyz, radians) → rotation-vector (axis-angle)."""
        return R.from_euler('xyz', rpy, degrees=False).as_rotvec()


def _validate_config(config: 'URConfig') -> 'URConfig':
    """Normalize and validate controller configuration.

    Checks frequency range, TCP/payload shapes, instantiates a shared memory
    manager if missing, and enforces simple physical bounds.

    Args:
        config (URConfig): User-provided configuration.

    Returns:
        URConfig: Possibly modified/normalized config.

    Raises:
        AssertionError: On invalid frequency, payload/TCP shapes, or types.
    """
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
