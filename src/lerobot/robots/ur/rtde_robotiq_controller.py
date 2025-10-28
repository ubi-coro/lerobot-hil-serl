import collections
import os
import time
import enum
import multiprocessing as mp
from dataclasses import dataclass, asdict
from multiprocessing.managers import SharedMemoryManager

import numpy as np

from .interpolator import PoseTrajectoryInterpolator
from .robotiq_gripper import RobotiqGripper
from lerobot.utils.shared_memory import SharedMemoryQueue, SharedMemoryRingBuffer, Empty


class Command(enum.IntEnum):
    """Simple commands for gripper process."""
    OPEN = 0
    CLOSE = 1
    MOVE = 2   # payload in 'value'
    STOP = 3


@dataclass
class GripperCommand:
    cmd: Command = Command.OPEN
    pos: float = 0.0  # for MOVE: target position (0-255)
    vel: float = 100.0  # [%]
    force: float = 100.0  # [%]
    timestamp: float = 100.0

    def to_queue_dict(self):
        d = asdict(self)
        d['cmd'] = int(self.cmd.value)
        d['pos'] = float(self.pos)
        d['vel'] = float(self.vel)
        d['force'] = float(self.force)
        d['timestamp'] = float(self.timestamp)
        return d


class RTDERobotiqController(mp.Process):
    """
    Separate process to drive the Robotiq 2F-85 gripper via shared-memory queues.

    - gripper_cmd_queue: receive GripperCommand messages
    - gripper_out_rb: push back current width & status periodically
    """

    def __init__(self,
                 hostname: str,
                 port: int = 63352,
                 shm_manager: SharedMemoryManager = None,
                 frequency: float = 20.0,
                 soft_real_time: bool = False,
                 rt_core: int = 4,
                 verbose: bool = False):
        super().__init__(name='GripperProcess')
        # network settings
        self.hostname = hostname
        self.port = port
        self.frequency = frequency
        self.soft_real_time = soft_real_time
        self.rt_core = rt_core
        self.verbose = verbose

        # shared-memory setup
        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()
        self.shm_manager = shm_manager

        # command queue example
        example_cmd = GripperCommand().to_queue_dict()
        self.gripper_cmd_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=self.shm_manager,
            examples=example_cmd,
            buffer_size=64
        )

        # state ring-buffer example
        example_state = {
            'width': 0.0,
            'object_status': 0,
            'fault': 0,
            'timestamp': time.time()
        }
        self.gripper_out_rb = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=self.shm_manager,
            examples=example_state,
            get_max_k=256,
            get_time_budget=0.1,
            put_desired_frequency=self.frequency
        )

        # internal control
        self.ready_event = mp.Event()
        self._last_action_time = None
        self._last_action = None

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
        self.gripper_cmd_queue.put(msg)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(5)
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

    # ========= command methods ============
    def move(self, pos: float, vel: float = 1.0, force: float = 1.0):
        msg = {
            'cmd': Command.MOVE.value,
            'pos': pos,
            'vel': vel,
            'force': force,
        }
        self.gripper_cmd_queue.put(msg)

    def move_smooth(self, pos: float, force: float = 1.0):
        t_now = time.perf_counter()
        if self._last_action_time is None:
            vel = 1.0
        else:
            dt = max(1e-4, t_now - self._last_action_time)
            vel = (pos - self._last_action) / dt

        self._last_action_time = t_now
        self._last_action = pos

        msg = {
            'cmd': Command.MOVE.value,
            'pos': pos,
            'vel': abs(vel),
            'force': force,
        }
        self.gripper_cmd_queue.put(msg)

    def open_gripper(self, vel: float = 1.0, force: float = 1.0):
        msg = {
            'cmd': Command.OPEN.value,
            'vel': vel,
            'force': force,
        }
        self.gripper_cmd_queue.put(msg)

    def close_gripper(self, vel: float = 1.0, force: float = 1.0):
        msg = {
            'cmd': Command.CLOSE.value,
            'vel': vel,
            'force': force,
        }
        self.gripper_cmd_queue.put(msg)

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.gripper_out_rb.get(out=out)
        else:
            return self.gripper_out_rb.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.gripper_out_rb.get_all()

    def run(self):
        try:
            if self.soft_real_time:
                os.sched_setaffinity(0, {self.rt_core})
                os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(20))
                # no need for psutil().nice(-priority) if not root

            # 1) Connect to gripper
            gr = RobotiqGripper()
            gr.connect(self.hostname, self.port)
            gr.activate()

            # Prepare for jitter logging
            hist = collections.deque(maxlen=1000)
            t_prev = time.monotonic()
            log_interval = 5.0
            next_log_time = t_prev + log_interval

            keep_running = True
            iter_idx = 0
            while keep_running:
                # Jitter measurement
                t_start = time.monotonic()
                dt_loop = t_start - t_prev
                hist.append(dt_loop)
                t_prev = t_start

                # 2) Get state from robot
                state = {
                    'width': float(gr.get_current_position() / 255.0),
                    'object_status': int(gr._get_var(gr.OBJ)),
                    'fault': int(gr._get_var(gr.FLT)),
                    'timestamp': t_start
                }
                self.gripper_out_rb.put(state)

                # ---------- NEW: scheduler state (init once after connect) ----------
                try:
                    last_sent_pos
                except NameError:
                    last_sent_pos = gr.get_current_position()  # 0..255 device scale
                    last_sent_time = time.monotonic()
                    pending_target = last_sent_pos
                    # knobs
                    min_period = 1 / 30.0  # ~14 Hz max send rate
                    hysteresis = 3  # min counts change to send
                    max_counts_per_s_base = 200.0  # velocity cap (counts/s)
                    SPE_CONST = 255  # smooth speed (out of 255)
                    FOR_CONST = 255  # moderate force
                    vel_scale = 1.0  # scale from last command
                    force_scale = 1.0

                # 3) Fetch command from queue
                try:
                    msgs = self.gripper_cmd_queue.get_all()
                    n_cmd = len(msgs['cmd'])
                except Empty:
                    n_cmd = 0

                # ---------- NEW: coalesce (last command wins) and only update targets ----------
                if n_cmd > 0:
                    i = n_cmd - 1
                    cmd_id = int(msgs['cmd'][i])
                    vel_scale = float(msgs.get('vel', [1.0])[i])  # 0..1
                    force_scale = float(msgs.get('force', [1.0])[i])  # 0..1
                    vel_scale = max(0.05, min(1.0, vel_scale))  # avoid zero
                    force_scale = max(0.05, min(1.0, force_scale))

                    if cmd_id == Command.OPEN.value:
                        pending_target = gr.get_open_position()
                    elif cmd_id == Command.CLOSE.value:
                        pending_target = gr.get_closed_position()
                    elif cmd_id == Command.MOVE.value:
                        # msgs['pos'] is 0..1; convert to 0..255
                        pending_target = int(max(0, min(255, round(255.0 * float(msgs['pos'][i])))))
                    elif cmd_id == Command.STOP.value:
                        keep_running = False

                # ---------- NEW: decide whether to send a new step now ----------
                t_now = time.monotonic()
                dt_since = t_now - last_sent_time
                delta = int(pending_target) - int(last_sent_pos)

                # freeze retargeting on contact/object events
                obj = state['object_status']  # 0=MOVING, 1/2 grip, 3 at rest
                if keep_running and obj != 0 and dt_since >= min_period and abs(delta) >= hysteresis:
                    # clip step by velocity cap
                    v_cap = max_counts_per_s_base * vel_scale
                    max_step = max(1, int(v_cap * dt_since))
                    step = max(-max_step, min(max_step, delta))
                    next_pos = int(last_sent_pos + step)

                    spe = max(10, min(255, int(SPE_CONST * vel_scale)))
                    fr = max(10, min(255, int(FOR_CONST * force_scale)))

                    # single atomic SET: POS,SPE,FOR,GTO
                    gr._set_vars(collections.OrderedDict([
                        (gr.POS, next_pos),
                        (gr.SPE, spe),
                        (gr.FOR, fr),
                        (gr.GTO, 1),
                    ]))
                    last_sent_pos = next_pos
                    last_sent_time = t_now

                # 4) First loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                # 6) Jitter print every log_interval
                if self.verbose and t_start >= next_log_time and len(hist) >= 10:
                    arr = np.array(hist)
                    print(
                        f"[RTDETFFController] Loop Jitter: μ={arr.mean() * 1000:.2f} ms  σ={arr.std() * 1000:.2f} ms  "
                        f"min={arr.min() * 1000:.2f} ms  max={arr.max() * 1000:.2f} ms")
                    next_log_time = t_start + log_interval

                # 5) Regulate frequency
                t4 = time.perf_counter()

                #print(f"dt_loop: {(time.monotonic() - t_start)*1000.0:.2f}ms, "
                #      f"dt_read: {(t1 - t0)*1000.0:.2f}ms, "
                #      f"dt_put: {(t2 - t1) * 1000.0:.2f}ms, "
                #      f"dt_write: {(t3 - t2) * 1000.0:.2f}ms, ")

                dt_target = 1 / self.frequency
                t_wait = dt_target - (time.monotonic() - t_start)
                time.sleep(t_wait)

        finally:
            self.ready_event.set()
            gr.disconnect()
