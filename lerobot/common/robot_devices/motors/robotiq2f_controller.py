import socket
import struct
import time
import multiprocessing as mp
import numpy as np
from multiprocessing.managers import SharedMemoryManager

from lerobot.common.utils.shared_memory import (
    SharedMemoryRingBuffer,
    SharedMemoryQueue,
    Empty,
)

# ---------------------------------------------------------------------------
#  Very‑lightweight Modbus‑TCP client for a single Robotiq 2F‑85 / 2F‑140
# ---------------------------------------------------------------------------
#  The gripper control box exposes the standard Robotiq Modbus register map.
#  We only need a handful of registers:
#    • 0x07D0   status & feedback (gSTA, gOBJ, gPOS, gCUR, gFLT)
#    • 0x03E8   action request (rACT, rGTO, rATR, rPOS, rFOR, rSPE)
#  This worker process polls status at (default) 30 Hz and applies the
#  *latest* width set‑point it receives via a shared‑memory queue.
# ---------------------------------------------------------------------------

def _mb_pack(unit_id: int, function: int, start_addr: int, words: bytes | int):
    """Build a Modbus‑TCP ADU (Application Data Unit)."""
    # Transaction & protocol ID = 0 (we send one request at a time)
    if isinstance(words, int):
        pdu = struct.pack(">BHH", function, start_addr, words)  # read N words
    else:
        # write N words, words = raw payload (2×n bytes)
        count = len(words) // 2
        pdu = struct.pack(">BHHB", function, start_addr, count, len(words)) + words
    adu = struct.pack(">HHHB", 0, 0, len(pdu) + 1, unit_id) + pdu
    return adu


def _reg_write_payload(pos_steps: int) -> bytes:
    """Return a 6‑word action command payload for the Robotiq register map."""
    # rACT=0x09 (activate), rGTO=0x00 (go to), rATR=0
    # rPOS = desired position (0‑255), rFOR=rSPE=0 (leave default)
    return struct.pack(">HHHH", 0x0900, 0x0000, pos_steps << 8, 0x0000)


class Robotiq2FController(mp.Process):
    """Simple 30Hz controller for a single 2F gripper.

    Parameters
    ----------
    shm_manager : SharedMemoryManager
        Manager instance shared with the parent process.
    ip : str, optional
        IP address of the Modbus‑TCP adapter (default ``192.168.1.11``).
    port : int, optional
        TCP port (default ``502``).
    frequency : float, optional
        Poll/update rate in Hz (default ``30``).
    """

    _UNIT_ID = 0x09          # fixed by Robotiq firmware
    _REG_STATUS = 0x07D0     # 2000 dec
    _REG_CMD = 0x03E8        # 1000 dec

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        ip: str = "192.168.1.11",
        port: int = 502,
        frequency: float = 30.0,
    ) -> None:
        super().__init__(name="Robotiq2FController")
        self.ip, self.port = ip, port
        self.freq = frequency

        # ---------------- shared‑memory channels ---------------------
        # inbound width command (metres) – buffer 64 commands
        self._in_q = SharedMemoryQueue.create_from_examples(
            shm_manager,
            examples={"width_m": np.zeros(1, dtype=np.float32)},
            buffer_size=64,
        )
        # outbound state (width_m, current, status) every poll cycle
        self._out_rb = SharedMemoryRingBuffer.create_from_examples(
            shm_manager,
            examples={
                "width_m": np.zeros(1, dtype=np.float32),
                "current": np.zeros(1, dtype=np.float32),
                "status": np.zeros(1, dtype=np.float32),
                "timestamp": np.zeros(1, dtype=np.float64),
            },
            get_max_k=128,
            put_desired_frequency=frequency,
        )
        self._ready = mp.Event()

    # ========= launch method ===========
    def start(self, *, wait: bool = True):
        super().start()
        if wait:
            self.start_wait(3.0)

    def stop(self, wait=True):
        message = {
            'cmd': Command.SHUTDOWN.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self._ready.wait(self.launch_timeout)
        assert self.is_alive()

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.terminate()
        self.stop()

    # ========= command methods ============

    def schedule_width(self, width_m: float):
        self._in_q.put({"width_m": np.array([width_m], dtype=np.float32)})

    def get_state(self, k: int | None = None):
        if k is None:
            return self._out_rb.get()
        return self._out_rb.get_last_k(k)

    # --------------- process main loop ------------------------------

    def run(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.connect((self.ip, self.port))
        s.settimeout(0.1)

        dt = 1.0 / self.freq
        target_pos_steps: int | None = None  # last commanded width (0‑255)
        self._ready.set()

        while True:
            t_cycle = time.monotonic()

            # 1) check for new target widths --------------------------------
            try:
                cmds = self._in_q.get_all()
                if len(cmds["width_m"]):
                    width_m = float(cmds["width_m"][-1][0])
                    width_m = np.clip(width_m, 0.0, 0.085)
                    target_pos_steps = int((0.085 - width_m) / 0.085 * 255)
                    payload = _reg_write_payload(target_pos_steps)
                    pkt = _mb_pack(self._UNIT_ID, 16, self._REG_CMD, payload)
                    s.sendall(pkt)
                    s.recv(1024)  # swallow echo
            except Empty:
                pass

            # 2) poll status registers --------------------------------------
            pkt = _mb_pack(self._UNIT_ID, 3, self._REG_STATUS, 0x0003)
            s.sendall(pkt)
            raw = s.recv(1024)
            if len(raw) >= 17 and raw[7] == 3:     # func code echo
                bytecnt = raw[8]
                data = raw[9 : 9 + bytecnt]
                gOBJ, gSTA, gPOS, gCUR, gFLT = data[1], data[0], data[3], data[4], data[5]
                width_m = 0.085 * (255 - gPOS) / 255.0
                self._out_rb.put(
                    {
                        "width_m": [width_m],
                        "current": [gCUR],
                        "status": [gSTA << 8 | gFLT],
                        "timestamp": [time.time()],
                    }
                )

            # 3) keep loop time constant ------------------------------------
            sleep = dt - (time.monotonic() - t_cycle)
            if sleep > 0:
                time.sleep(sleep)
