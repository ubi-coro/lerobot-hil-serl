import time
import enum
import multiprocessing as mp
from dataclasses import dataclass, asdict
from multiprocessing.managers import SharedMemoryManager

from lerobot.common.robot_devices.motors.robotiq_gripper import RobotiqGripper
from lerobot.common.utils.shared_memory import SharedMemoryQueue, SharedMemoryRingBuffer, Empty


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

    def to_queue_dict(self):
        d = asdict(self)
        d['cmd'] = int(self.cmd.value)
        d['value'] = float(self.value)
        d['vel'] = float(self.vel)
        d['force'] = float(self.force)
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
                 frequency: float = 20.0):
        super().__init__(name='GripperProcess')
        # network settings
        self.hostname = hostname
        self.port = port
        self.frequency = frequency

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
    def move(self, pos: float, vel: float = 100.0, force: float = 100.0):
        msg = {
            'cmd': Command.MOVE.value,
            'pos': pos,
            'vel': vel,
            'force': force,
        }
        self.gripper_cmd_queue.put(msg)

    def open_gripper(self, vel: float = 100.0, force: float = 100.0):
        msg = {
            'cmd': Command.OPEN.value,
            'vel': vel,
            'force': force,
        }
        self.gripper_cmd_queue.put(msg)

    def close_gripper(self, vel: float = 100.0, force: float = 100.0):
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
            # 1) Connect to gripper
            gr = RobotiqGripper()
            gr.connect(self.hostname, self.port)
            gr.activate()

            keep_running = True
            iter_idx = 0
            t_start = time.monotonic()
            while keep_running:#
                t_now = time.monotonic()

                # 2) Get state from robot
                state = {
                    'width': float(gr.get_current_position()),
                    'object_status': int(gr._get_var(gr.OBJ)),
                    'fault': int(gr._get_var(gr.FLT)),
                    'timestamp': t_now
                }
                self.gripper_out_rb.put(state)

                # 3) Fetch command from queue
                try:
                    msgs = self.gripper_cmd_queue.get_all()
                    n_cmd = len(msgs['cmd'])
                except Empty:
                    n_cmd = 0

                for i in range(n_cmd):
                    cmd_id = int(msgs['cmd'][i])
                    vel = int(255.0 * msgs['vel'][i])
                    force = int(255.0 * msgs['force'][i])

                    if cmd_id == Command.OPEN.value:
                        gr.move(gr.get_open_position(), vel, force)
                    elif cmd_id == Command.CLOSE.value:
                        gr.move(gr.get_closed_position(), vel, force)
                    elif cmd_id == Command.MOVE.value:
                        pos = int(msgs['pos'][i])
                        gr.move(pos, vel, force)
                    elif cmd_id == Command.STOP.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break

                # 4) First loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                # 5) Regulate frequency
                dt = 1 / self.frequency
                t_end = t_start + dt * iter_idx
                time.sleep(t_end - t_start)
        finally:
            self.ready_event.set()
            gr.disconnect()