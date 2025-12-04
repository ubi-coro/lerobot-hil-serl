"""
ALOHA Teleoperation Module
==========================

Thin FastAPI wrapper around LeRobot's ALOHA teleoperation primitives.

Removed legacy custom preset layer (SAFE / NORMAL / PERFORMANCE) to rely on
one canonical configuration object (AlohaConfig) with optional overrides
from the frontend. This keeps close alignment with upstream LeRobot while
retaining:
    * Operation mode mapping (bimanual / left_only / right_only)
    * Camera enable / disable logic
    * Threaded control loop with stop event
    * Basic safety guard (auto-enable safety_limits for extreme values)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging
import time
import threading
import os
from enum import Enum
from types import SimpleNamespace

import torch
# Optional visualization dependency (rerun). Provide no-op fallbacks if unavailable.
try:  # pragma: no cover - optional dependency guard
    import rerun as rr  # type: ignore
    from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

    def _init_rerun(*args, **kwargs):
        return init_rerun(*args, **kwargs)

    _RERUN_AVAILABLE = True
except Exception:
    _RERUN_AVAILABLE = False

    def _init_rerun(*args, **kwargs):
        return None

    def log_rerun_data(*args, **kwargs):
        return None
import shared
from experiment_config_mapper import ExperimentConfigMapper

import base64
import cv2
import numpy as np

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor.converters import create_transition
from lerobot.processor.core import TransitionKey
from lerobot.processor.hil_processor import TELEOP_ACTION_KEY
from lerobot.teleoperators import TeleopEvents
from lerobot.utils.control_utils import predict_action
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import get_safe_torch_device

# LeRobot imports are deferred to runtime inside functions to tolerate environments
# where optional dependencies (e.g., draccus) are not installed. This keeps the
# backend bootable so the GUI can load and the user can still access non-teleop APIs.

logger = logging.getLogger(__name__)

# Helper function to get absolute calibration directory
def get_calibration_dir():
    """Get absolute path to ALOHA calibration directory"""
    # Get the project root (4 levels up from this file)
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
    calibration_dir = os.path.join(project_root, ".cache", "calibration", "aloha_lemgo_tabea")
    return calibration_dir


# Create router
router = APIRouter(prefix="/api/aloha-teleoperation", tags=["aloha-teleoperation"])

class OperationMode(str, Enum):
    BIMANUAL = "bimanual"
    LEFT_ONLY = "left_only"
    RIGHT_ONLY = "right_only"

    @classmethod
    def from_legacy(cls, value: str) -> "OperationMode":
        mapping = {
            "bimanual": cls.BIMANUAL,
            "left": cls.LEFT_ONLY,
            "left_arm": cls.LEFT_ONLY,
            "left_only": cls.LEFT_ONLY,
            "right": cls.RIGHT_ONLY,
            "right_arm": cls.RIGHT_ONLY,
            "right_only": cls.RIGHT_ONLY,
        }
        try:
            return mapping[value]
        except KeyError:
            raise ValueError(f"Unsupported operation_mode '{value}'")

# Pydantic models
class ApiResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None

class AlohaConfig(BaseModel):
    """ALOHA-specific teleoperation configuration"""
    robot_type: str = Field(default="aloha", description="Robot type (currently only 'aloha' supported)")
    fps: int = Field(default=90, ge=1, le=120, description="Target control loop frames per second")
    # IMPORTANT: Default aligned with ViperXConfig/WidowXConfig (5.0 degrees for safety)
    # Higher values can be set via GUI if needed, but default should be conservative
    max_relative_target: Optional[float] = Field(default=5.0, ge=0, le=100, description="Maximum relative target (degrees)")
    moving_time: float = Field(default=0.1, ge=0.01, le=1.0, description="Moving time for velocity profiles")
    operation_mode: OperationMode = Field(default=OperationMode.BIMANUAL, description="Operation mode")
    show_cameras: bool = Field(default=True, description="Show camera feeds on web interface")
    display_data: bool = Field(default=False, description="Open LeRobot's external display window")
    safety_limits: bool = Field(default=True, description="Enable safety limits")
    performance_monitoring: bool = Field(default=True, description="Enable performance monitoring")
    calibration_dir: str = Field(default_factory=get_calibration_dir, description="Calibration directory")
    demo_mode: bool = Field(default=False, description="Run in policy demo/evaluation mode")
    policy_path: Optional[str] = Field(default=None, description="Optional path to pretrained policy for evaluation")

class AlohaStartRequest(BaseModel):
    """Start ALOHA teleoperation request (single configuration path)."""
    # Accept loose dict so we can map legacy operation_mode strings (e.g. 'right_arm')
    config: Optional[Dict[str, Any]] = None

# Global state for ALOHA teleoperation
aloha_state = {
    "active": False,
    "robot": None,          # Robot instance in use for teleoperation
    "teleop": None,
    "owned_robot": False,    # Whether this module created (and must disconnect) the robot
    "config": None,
    "start_time": None,
    "control_thread": None,
    "stop_event": threading.Event(),
    "events": None,          # ControlEvents instance (for immediate exit signaling)
    "stage": "idle",        # idle|initializing|running|stopping
    "performance_metrics": {
        "frames_processed": 0,
        "average_fps": 0.0,
        "latency_ms": 0.0,
        "last_joint_update": 0.0
    }
}


def _is_dataset_recording_active() -> bool:
    """Best-effort check whether the dataset recording worker is currently active."""
    try:  # Lazy import to avoid circular dependency at module load
        from . import recording_worker  # type: ignore

        worker = getattr(recording_worker, "recording_worker", None)
        return bool(getattr(worker, "active", False))
    except Exception:
        return False


def get_teleoperation_status_snapshot() -> Dict[str, Any]:
    """Return a stable snapshot of current teleoperation state."""
    try:
        session_duration = time.time() - aloha_state["start_time"] if aloha_state.get("start_time") else 0
        cfg = aloha_state.get("config") or {}
        return {
            "active": bool(aloha_state.get("active")),
            "stage": aloha_state.get("stage"),
            "robot_type": "ALOHA",
            "configuration": cfg,
            "performance_metrics": aloha_state.get("performance_metrics", {}),
            "session_duration": session_duration,
            "robot_connected": aloha_state.get("robot") is not None,
            "teleop_connected": aloha_state.get("teleop") is not None,
            "owned_robot": bool(aloha_state.get("owned_robot")),
            "display_data_active": bool(cfg.get("display_data", False)) if isinstance(cfg, dict) else False,
        }
    except Exception as e:
        logger.debug(f"teleop status snapshot error: {e}")
        return {"active": False, "stage": "idle", "robot_type": "ALOHA", "session_duration": 0}


async def emit_teleoperation_status(room: str | None = None):
    """Emit teleoperation_status event via Socket.IO if available."""
    try:
        sio = shared.get_socketio()
        if not sio:
            return
        payload = get_teleoperation_status_snapshot()
        await sio.emit("teleoperation_status", payload, room=room)
    except Exception:
        logger.debug("teleoperation_status emit failed", exc_info=True)

def aloha_teleoperation_worker(config: AlohaConfig):
    """Worker thread for ALOHA teleoperation using experiment-based configs."""

    env = None
    env_processor = None
    action_processor = None
    teleop_dict: dict[str, Any] = {}
    stream_cameras = config.show_cameras
    borrowed_cameras = None

    try:
        op_mode_mapping = {
            OperationMode.BIMANUAL: "bimanual",
            OperationMode.LEFT_ONLY: "left",
            OperationMode.RIGHT_ONLY: "right",
        }

        operation_key = op_mode_mapping[config.operation_mode]
        demo_enabled = bool(config.demo_mode or config.policy_path)

        # Borrow cameras from robot module if streaming is requested
        # The robot module already has cameras connected, so we reuse them
        if stream_cameras:
            try:
                from . import robot as robot_module
                borrowed_cameras = robot_module.borrow_cameras("aloha_teleoperation")
                if borrowed_cameras:
                    logger.info(f"Successfully borrowed {len(borrowed_cameras)} cameras from robot module")
                else:
                    logger.warning("Cameras requested but not available from robot module")
                    stream_cameras = False  # Disable streaming if no cameras available
            except Exception as e:
                logger.warning(f"Could not borrow cameras: {e}")
                stream_cameras = False

        # Create environment WITHOUT cameras (we use borrowed ones for streaming)
        env, env_processor, action_processor, env_cfg, mapping = ExperimentConfigMapper.create_env_from_gui_selection(
            operation_mode=operation_key,
            demo_mode=demo_enabled,
            policy_path_override=config.policy_path,
            use_cameras=False,  # Never create new camera connections; use borrowed ones
        )

        action_dim = env_cfg.action_dim

        policy = None
        policy_cfg: PreTrainedConfig | None = None
        preprocessor = None
        postprocessor = None
        policy_device = torch.device("cpu")
        use_amp = False

        if demo_enabled:
            if not mapping.supports_policy:
                logger.warning(
                    "Demo mode requested for %s but no policy is registered; falling back to teleoperation",
                    mapping.experiment_type,
                )
            else:
                policy_path_resolved = config.policy_path or mapping.default_policy_path
                if not policy_path_resolved:
                    raise ValueError(
                        "Demo mode enabled but no policy path provided. Configure policy_path via GUI or default mapping."
                    )

                try:
                    policy_cfg = PreTrainedConfig.from_pretrained(policy_path_resolved)
                    policy_cfg.pretrained_path = policy_path_resolved
                    policy_device = get_safe_torch_device(getattr(policy_cfg, "device", "cpu") or "cpu")
                    policy_cfg.device = str(policy_device)
                    use_amp = bool(getattr(policy_cfg, "use_amp", False))

                    policy = make_policy(policy_cfg, env_cfg=env_cfg)
                    preprocessor, postprocessor = make_pre_post_processors(
                        policy_cfg=policy_cfg,
                        pretrained_path=policy_cfg.pretrained_path,
                        preprocessor_overrides={"device_processor": {"device": str(policy_device)}},
                        postprocessor_overrides={"device_processor": {"device": str(policy_device)}},
                    )
                    policy.reset()
                    logger.info("Loaded demo policy from %s on device %s", policy_path_resolved, policy_device)
                except Exception:
                    logger.error("Failed to initialize demo policy from %s", policy_path_resolved, exc_info=True)
                    policy = None
                    policy_cfg = None
                    preprocessor = None
                    postprocessor = None

        # Extract teleoperator instances from processor pipeline for later cleanup
        for step in getattr(action_processor, "steps", []):
            potential = getattr(step, "teleoperators", None)
            if isinstance(potential, dict) and potential:
                teleop_dict = potential
                break

        aloha_state["robot"] = env.robot_dict
        aloha_state["teleop"] = teleop_dict
        aloha_state["owned_robot"] = True

        if config.display_data and _RERUN_AVAILABLE:
            logger.info("🖥️ display_data=true: Initializing LeRobot's rerun session...")
            _init_rerun(session_name="lerobot_control_loop_teleop")
            logger.info("✅ LeRobot rerun session initialized.")

        # Camera streaming: read from borrowed cameras in the control loop
        # This avoids separate threads that compete for hardware
        camera_frame_interval = max(1, int(config.fps / 12))  # ~12 fps for streaming
        camera_frame_counter = 0
        
        if stream_cameras and borrowed_cameras:
            # Notify frontend which cameras are available
            camera_ids = list(borrowed_cameras.keys())
            shared.emit_threadsafe('camera_list', {'cameras': camera_ids})
            logger.info(f"Camera streaming enabled for: {camera_ids} (every {camera_frame_interval} frames)")

        obs, info = env.reset()
        env_processor.reset()
        action_processor.reset()
        if policy is not None:
            try:
                policy.reset()
            except Exception:
                logger.debug("Policy reset failed", exc_info=True)

        transition = create_transition(observation=obs, info=info)
        transition = env_processor(data=transition)

        demo_active = bool(policy)

        if isinstance(aloha_state.get("config"), dict):
            aloha_state["config"]["demo_mode_active"] = demo_active

        aloha_state["stage"] = "running"
        logger.info(
            "Starting teleoperation loop (fps=%s, display_data=%s, demo_mode=%s)",
            config.fps,
            config.display_data,
            demo_active,
        )

        while not aloha_state["stop_event"].is_set():
            loop_start = time.perf_counter()

            prev_transition = transition
            # Always mark as intervention for passive teleoperation (leaders stay torque-off)
            info = {TeleopEvents.IS_INTERVENTION: True}

            if policy is not None and policy_cfg is not None:
                policy_observation = {
                    k: v
                    for k, v in prev_transition[TransitionKey.OBSERVATION].items()
                    if k in policy.config.input_features
                }
                try:
                    action = predict_action(
                        observation=policy_observation,
                        policy=policy,
                        device=policy_device,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        use_amp=use_amp,
                        task=getattr(env_cfg, "task", None),
                        robot_type=getattr(env_cfg, "type", None),
                    )
                except Exception:
                    logger.error("Policy inference failed; switching to teleoperation fallback", exc_info=True)
                    policy = None
                    policy_cfg = None
                    preprocessor = None
                    postprocessor = None
                    action = torch.zeros(action_dim, dtype=torch.float32)
            else:
                action = torch.zeros(action_dim, dtype=torch.float32)
            action_transition = create_transition(action=action, info=info)
            processed_action_transition = action_processor(action_transition)

            if processed_action_transition.get(TransitionKey.DONE, False):
                logger.info("Intervention processor requested termination")
                break

            next_action = processed_action_transition[TransitionKey.ACTION]
            obs, reward, terminated, truncated, info = env.step(next_action)

            complementary_data = processed_action_transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).copy()
            info.update(processed_action_transition.get(TransitionKey.INFO, {}).copy())

            if info.get(TeleopEvents.IS_INTERVENTION, False) and TELEOP_ACTION_KEY in complementary_data:
                action_to_apply = complementary_data[TELEOP_ACTION_KEY]
            else:
                action_to_apply = next_action

            transition = create_transition(
                observation=obs,
                action=action_to_apply,
                reward=reward + processed_action_transition.get(TransitionKey.REWARD, 0.0),
                done=terminated or processed_action_transition.get(TransitionKey.DONE, False),
                truncated=truncated or processed_action_transition.get(TransitionKey.TRUNCATED, False),
                info=info,
                complementary_data=complementary_data,
            )
            transition = env_processor(data=transition)

            if config.display_data and _RERUN_AVAILABLE:
                try:
                    def _to_numpy(payload):
                        if isinstance(payload, dict):
                            return {key: _to_numpy(value) for key, value in payload.items()}
                        if hasattr(payload, "detach"):
                            return payload.detach().cpu().numpy()
                        if isinstance(payload, torch.Tensor):
                            return payload.cpu().numpy()
                        return payload

                    rerun_obs = _to_numpy(prev_transition[TransitionKey.OBSERVATION])
                    rerun_action_payload = _to_numpy(action_to_apply)
                    if isinstance(rerun_action_payload, dict):
                        log_rerun_data(observation=rerun_obs, action=rerun_action_payload)
                    else:
                        log_rerun_data(observation=rerun_obs, action={"action": rerun_action_payload})
                except Exception:
                    logger.debug("Failed to log rerun data", exc_info=True)

            # Stream camera frames from borrowed cameras (read in control loop = no hardware contention)
            if stream_cameras and borrowed_cameras:
                camera_frame_counter += 1
                if camera_frame_counter >= camera_frame_interval:
                    camera_frame_counter = 0
                    try:
                        for cam_id, camera in borrowed_cameras.items():
                            # Read frame from borrowed camera using async_read (non-blocking)
                            frame_data = camera.async_read() if hasattr(camera, 'async_read') else camera.read()
                            if frame_data is None:
                                continue
                            # Handle (color, depth) tuple from RealSense
                            if isinstance(frame_data, tuple):
                                frame_data = frame_data[0]
                            if not isinstance(frame_data, np.ndarray):
                                continue
                            # Resize for streaming efficiency
                            h, w = frame_data.shape[:2]
                            if w > 640:
                                scale = 640 / w
                                frame_data = cv2.resize(frame_data, (640, int(h * scale)))
                            # Convert RGB to BGR for JPEG encoding (cameras use RGB, OpenCV expects BGR)
                            frame_bgr = cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR)
                            # Encode as JPEG
                            ok, buf = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                            if ok:
                                b64 = 'data:image/jpeg;base64,' + base64.b64encode(buf).decode('utf-8')
                                shared.emit_threadsafe('camera_frame', {
                                    'camera_id': cam_id,
                                    'frame': b64,
                                    'ts': time.time()
                                })
                    except Exception:
                        logger.debug("Failed to emit camera frames", exc_info=True)

            dt_s = time.perf_counter() - loop_start
            busy_wait(max(0.0, (1 / config.fps) - dt_s))

            loop_s = time.perf_counter() - loop_start
            try:
                pm = aloha_state.get("performance_metrics", {})
                prev_avg = float(pm.get("average_fps", 0.0) or 0.0)
                fps_inst = 1.0 / loop_s if loop_s > 0 else 0.0
                pm["average_fps"] = (0.8 * prev_avg) + (0.2 * fps_inst)
                pm["frames_processed"] = float(pm.get("frames_processed", 0.0) or 0.0) + 1
                pm["latency_ms"] = max(0.0, (loop_s - (1 / config.fps)) * 1000.0)
                pm["last_joint_update"] = time.time()
                aloha_state["performance_metrics"] = pm
            except Exception:
                logger.debug("Failed to update performance metrics", exc_info=True)

    except Exception as exc:
        logger.error("Error in ALOHA teleoperation worker: %s", exc, exc_info=True)
    finally:
        aloha_state["stage"] = "stopping"

        # Clear camera list notification
        if stream_cameras:
            shared.emit_threadsafe('camera_list', {'cameras': []})

        # Return borrowed cameras to robot module
        if borrowed_cameras:
            try:
                from . import robot as robot_module
                robot_module.return_cameras("aloha_teleoperation")
                logger.info("Cameras returned to robot module")
            except Exception as e:
                logger.debug(f"Could not return cameras: {e}")

        if config.display_data and _RERUN_AVAILABLE:
            try:
                rr.rerun_shutdown()
            except Exception as exc:
                logger.debug("Error shutting down rerun: %s", exc)

        if env is not None:
            try:
                env.close()
            except Exception as exc:
                logger.warning("Error closing environment: %s", exc)

        for teleop in teleop_dict.values():
            try:
                teleop.disconnect()
            except Exception as exc:
                logger.debug("Error disconnecting teleoperator %s: %s", getattr(teleop, "id", "unknown"), exc)

        aloha_state["robot"] = None
        aloha_state["teleop"] = None
        aloha_state["owned_robot"] = False
        aloha_state["active"] = False
        aloha_state["stop_event"].clear()
        logger.info("ALOHA teleoperation worker stopped and cleaned up.")

@router.post("/start", response_model=ApiResponse)
async def start_aloha_teleoperation(request: AlohaStartRequest):
    """
    Start ALOHA teleoperation with a single unified configuration.
    Legacy preset layer removed: frontend passes an optional config dict.
    """
    try:
        if aloha_state["active"]:
            return ApiResponse(
                status="error",
                message="ALOHA teleoperation is already active"
            )
        
        logger.info("Starting ALOHA teleoperation (single-config mode)")

        # Build config object (apply mapping for legacy operation_mode values)
        if request.config:
            cfg_dict = {**request.config}
            if "operation_mode" in cfg_dict:
                original = cfg_dict["operation_mode"]
                if original == "right_arm":
                    cfg_dict["operation_mode"] = "right_only"
                elif original == "left_arm":
                    cfg_dict["operation_mode"] = "left_only"
                elif original == "bimanual":
                    cfg_dict["operation_mode"] = "bimanual"
                if original != cfg_dict["operation_mode"]:
                    logger.info(f"Mapped operation_mode: {original} -> {cfg_dict['operation_mode']}")
            config = AlohaConfig(**cfg_dict)
            logger.info("Using provided ALOHA configuration")
        else:
            config = AlohaConfig()
            logger.info("Using default ALOHA configuration")
        
        # Validate configuration for ALOHA
        # Note: Default is now 5.0 (aligned with ViperX/WidowX configs)
        # Warn if user sets it higher than recommended safe value
        if config.max_relative_target and config.max_relative_target > 10:
            logger.warning(f"High relative target detected ({config.max_relative_target}°). "
                         f"Default safe value is 5°. Ensure safety limits are enabled.")
            config.safety_limits = True
        
        # Optimize FPS for single-arm operation to reduce lag
        if config.operation_mode in [OperationMode.LEFT_ONLY, OperationMode.RIGHT_ONLY]:
            if config.fps > 30:
                logger.info(f"Reducing FPS from {config.fps} to 30 for single-arm operation to improve performance")
                config.fps = 30
        
        # Start teleoperation state
        aloha_state["active"] = True
        aloha_state["robot"] = None
        aloha_state["teleop"] = None
        aloha_state["owned_robot"] = False
        config_payload = config.dict()
        if isinstance(config_payload.get("operation_mode"), OperationMode):
            config_payload["operation_mode"] = config.operation_mode.value
        aloha_state["config"] = config_payload
        aloha_state["start_time"] = time.time()
        aloha_state["stop_event"].clear()
        aloha_state["stage"] = "initializing"

        # Reset performance metrics
        aloha_state["performance_metrics"] = {
            "frames_processed": 0,
            "average_fps": 0.0,
            "latency_ms": 0.0,
            "last_joint_update": 0.0
        }

        # Start worker thread (LeLab-style threading)
        aloha_state["control_thread"] = threading.Thread(
            target=aloha_teleoperation_worker,
            args=(config,),
            daemon=True
        )
        aloha_state["control_thread"].start()

        # Emit status immediately after start
        try:
            await emit_teleoperation_status()
        except Exception:
            logger.debug("emit teleop status after start failed", exc_info=True)

        logger.info(f"ALOHA teleoperation started with config: {config_payload}")

        return ApiResponse(
            status="success",
            message="ALOHA teleoperation started successfully",
            data={
                "active": True,
                "configuration": config_payload,
                "start_time": aloha_state["start_time"],
                "operation_mode": config.operation_mode.value
            }
        )
        
    except Exception as e:
        aloha_state["active"] = False
        logger.error(f"Failed to start ALOHA teleoperation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start ALOHA teleoperation: {str(e)}"
        )

@router.post("/stop", response_model=ApiResponse)
async def stop_aloha_teleoperation():
    """
    Stop ALOHA teleoperation and return session summary.
    
    Graceful shutdown with performance metrics (your approach).
    """
    try:
        logger.info("Stopping ALOHA teleoperation")
        
        if not aloha_state["active"]:
            return ApiResponse(
                status="info",
                message="ALOHA teleoperation was not active",
                data={"active": False}
            )
        
        # Signal stop (LeLab-style)
        aloha_state["stop_event"].set()
        
        # Wait for thread to finish
        if aloha_state["control_thread"] and aloha_state["control_thread"].is_alive():
            aloha_state["control_thread"].join(timeout=5.0)
        
        # Calculate session duration
        session_duration = time.time() - aloha_state["start_time"] if aloha_state["start_time"] else 0
        
        # Prepare session summary (your approach)
        session_summary = {
            "duration_seconds": round(session_duration, 2),
            "performance_metrics": aloha_state["performance_metrics"].copy(),
            "configuration_used": aloha_state["config"],
            "robot_type": "ALOHA"
        }
        
        # Reset state
        aloha_state["config"] = None
        aloha_state["start_time"] = None
        aloha_state["control_thread"] = None
        aloha_state["active"] = False
        aloha_state["stage"] = "idle"
        aloha_state["teleop"] = None
        aloha_state["robot"] = None
        aloha_state["owned_robot"] = False
        aloha_state["performance_metrics"] = {
            "frames_processed": 0,
            "average_fps": 0.0,
            "latency_ms": 0.0,
            "last_joint_update": 0.0,
        }
        aloha_state["stop_event"].clear()
        
        logger.info(f"ALOHA teleoperation stopped. Session duration: {session_duration:.2f}s")

        # Emit status after stop
        try:
            await emit_teleoperation_status()
        except Exception:
            logger.debug("emit teleop status after stop failed", exc_info=True)
        
        return ApiResponse(
            status="success",
            message="ALOHA teleoperation stopped successfully",
            data={
                "active": False,
                "session_summary": session_summary
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to stop ALOHA teleoperation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to stop ALOHA teleoperation: {str(e)}"
        )

@router.get("/status", response_model=ApiResponse)
async def get_aloha_status():
    """
    Get current ALOHA teleoperation status and real-time metrics.
    """
    try:
        # Update performance metrics if active
        if aloha_state["active"]:
            current_time = time.time()
            session_duration = current_time - aloha_state["start_time"]
            
        status_data = {
            "active": aloha_state["active"],
            "stage": aloha_state.get("stage"),
            "robot_type": "ALOHA",
            "configuration": aloha_state["config"],
            "performance_metrics": aloha_state["performance_metrics"],
            "session_duration": (
                time.time() - aloha_state["start_time"] if aloha_state["start_time"] else 0
            ),
            "robot_connected": aloha_state["robot"] is not None,
            "teleop_connected": aloha_state.get("teleop") is not None,
            "owned_robot": aloha_state.get("owned_robot"),
            "display_data_active": aloha_state["config"].get("display_data", False) if aloha_state["config"] else False
        }
        
        return ApiResponse(
            status="success",
            message="ALOHA teleoperation status retrieved",
            data=status_data
        )
        
    except Exception as e:
        logger.error(f"Failed to get ALOHA status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get ALOHA status: {str(e)}"
        )

# Preset endpoint removed: legacy custom presets deprecated.
