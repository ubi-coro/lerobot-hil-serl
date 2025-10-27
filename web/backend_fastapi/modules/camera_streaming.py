"""Camera Streaming Module
=========================

Provides a lightweight camera streaming manager that emits JPEG frames over
Socket.IO using the already-established global socket (see shared.py).

Design goals:
- Reuse existing connected robot (no second connection)
- Avoid removal of robot.cameras; treat show_cameras flag as streaming on/off
- Provide abstraction layer to allow future simulation camera provider
- Keep CPU usage bounded (default target FPS 12, JPEG quality 70, optional resize)

Public API:
    start_streams(robot, *, camera_ids=None, fps=12, resize=None)
    stop_streams(camera_ids=None)
    stop_all_streams()
    get_active_streams()

Future extension: add SimulationCameraProvider implementing same interface.
"""

from __future__ import annotations

import threading
import time
import logging
import base64
from typing import Dict, List, Optional, Tuple
import asyncio

import cv2  # type: ignore
import numpy as np  # type: ignore

import shared  # shared.py is at backend_fastapi root; added to sys.path by main.py

logger = logging.getLogger(__name__)

_streams_lock = threading.Lock()
_active_streams: Dict[str, dict] = {}

# Default JPEG encode params (balance quality/performance)
_JPEG_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), 70]


def _encode_frame(frame: np.ndarray, resize: Optional[Tuple[int, int]]) -> Optional[str]:
    try:
        if resize:
            w, h = resize
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode('.jpg', frame, _JPEG_PARAMS)
        if not ok:
            return None
        return 'data:image/jpeg;base64,' + base64.b64encode(buf).decode('utf-8')
    except Exception as e:
        logger.debug(f"Frame encode error: {e}")
        return None


def _camera_thread(camera_id: str, robot, fps: int, stop_evt: threading.Event, resize: Optional[Tuple[int, int]]):
    logger.info(f"Camera stream thread started: {camera_id} (target {fps} fps)")
    sio = shared.get_socketio()
    if sio is None:
        logger.warning("Socket.IO instance unavailable; aborting camera stream")
        return

    interval = 1.0 / max(fps, 1)
    last_log = time.time()
    frames_sent = 0
    failures = 0
    while not stop_evt.is_set():
        start_t = time.time()
        frame = None
        try:
            # Access robot camera only if present
            if robot and hasattr(robot, 'cameras') and camera_id in robot.cameras:
                camera_obj = robot.cameras[camera_id]
                # Prefer non-blocking camera access if available
                if hasattr(camera_obj, 'async_read'):
                    frame = camera_obj.async_read()
                else:
                    frame = camera_obj.read()
                # If depth is enabled, async/read may return (color, depth)
                if isinstance(frame, tuple) and len(frame) > 0:
                    frame = frame[0]
            if frame is None:
                # generate fallback test pattern
                frame = _test_pattern(camera_id)
        except Exception as e:
            failures += 1
            if failures <= 5:
                logger.warning(f"Camera {camera_id} read failure ({failures}): {e}")
            frame = _test_pattern(camera_id)

        payload = _encode_frame(frame, resize)
        if payload:
            # Emit via thread-safe helper; no awaiting in this thread
            shared.emit_threadsafe('camera_frame', {
                'camera_id': camera_id,
                'frame': payload,
                'ts': time.time()
            })
            frames_sent += 1
            if frames_sent <= 5:
                logger.debug(f"Emitted initial frame {frames_sent} for {camera_id}")

        # Periodic log
        now = time.time()
        if now - last_log >= 10:
            fps_actual = frames_sent / (now - last_log)
            logger.info(f"Camera {camera_id}: {fps_actual:.1f} fps actual")
            frames_sent = 0
            last_log = now

        # Sleep to maintain fps
        elapsed = time.time() - start_t
        sleep_t = interval - elapsed
        if sleep_t > 0:
            stop_evt.wait(timeout=sleep_t)

    logger.info(f"Camera stream thread exiting: {camera_id}")


def _test_pattern(camera_id: str) -> np.ndarray:
    h, w = 360, 640
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    t = time.strftime('%H:%M:%S')
    cv2.putText(frame, camera_id, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(frame, t, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    # Simple moving bar for visual liveness
    pos = int((time.time() * 40) % w)
    cv2.rectangle(frame, (pos, 200), (min(pos + 80, w - 1), 260), (0, 140, 255), -1)
    return frame


def start_streams(robot, *, camera_ids: Optional[List[str]] = None, fps: int = 12, resize: Optional[Tuple[int, int]] = (640, 360), allow_fallback: bool = True):
    """Start streaming for given cameras (or all if None).

    If no robot cameras match and allow_fallback=True, will start synthetic test pattern
    streams for a default set of camera IDs so the frontend displays something.
    """
    available = []
    if robot is None:
        logger.warning("No robot available for camera streaming (will use fallback test cameras if enabled)")
    elif not hasattr(robot, 'cameras'):
        logger.warning("Robot has no cameras attribute (will use fallback test cameras if enabled)")
    else:
        try:
            available = list(getattr(robot, 'cameras').keys())
        except Exception as e:
            logger.warning(f"Failed to list robot cameras: {e}")

    requested = camera_ids or available
    targets = [c for c in requested if c in available]

    if not targets and allow_fallback:
        # Provide standard ALOHA fallback IDs
        targets = ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
        logger.info(f"Using fallback synthetic cameras: {targets}")
        robot = None  # force test pattern path
    elif not targets:
        logger.warning("No matching cameras to stream and fallback disabled")
        return

    with _streams_lock:
        new_started = []
        for cam_id in targets:
            if cam_id in _active_streams:
                continue
            stop_evt = threading.Event()
            thread = threading.Thread(target=_camera_thread, args=(cam_id, robot, fps, stop_evt, resize), daemon=True)
            _active_streams[cam_id] = {
                'stop_event': stop_evt,
                'thread': thread,
                'fps': fps,
                'resize': resize,
                'synthetic': robot is None or cam_id not in available
            }
            thread.start()
            new_started.append(cam_id)
        if new_started:
            logger.info(f"Started camera streams: {new_started}")

    # Emit updated camera list to clients (fire-and-forget)
    active = get_active_streams()
    shared.emit_threadsafe('camera_list', { 'cameras': active })


def stop_streams(camera_ids: Optional[List[str]] = None):
    with _streams_lock:
        targets = camera_ids or list(_active_streams.keys())
        for cam_id in targets:
            entry = _active_streams.get(cam_id)
            if not entry:
                continue
            entry['stop_event'].set()
        # Join after releasing lock to prevent deadlock
    for cam_id in targets:
        entry = _active_streams.pop(cam_id, None)
        if entry:
            thread = entry['thread']
            thread.join(timeout=2.0)


def stop_all_streams():
    stop_streams()
    # Notify clients that list is now possibly empty
    active = get_active_streams()
    shared.emit_threadsafe('camera_list', { 'cameras': active })


def get_active_streams() -> List[str]:
    with _streams_lock:
        return list(_active_streams.keys())
