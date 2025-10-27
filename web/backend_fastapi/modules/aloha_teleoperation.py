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
import json
import os
from pathlib import Path
from enum import Enum

# ALOHA-specific imports from LeRobot
from lerobot.processor import make_default_processors
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging
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
from . import camera_streaming
from . import robot as robot_module
from . import camera_streaming

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

def load_hardware_config():
    """Load hardware configuration from ~/.config/lerobot/hardware_config.json"""
    config_path = Path.home() / ".config" / "lerobot" / "hardware_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Hardware config not found at {config_path}. Please create it with your workstation's hardware settings.")
    with open(config_path, 'r') as f:
        return json.load(f)

def _ensure_calibration_applied(device, device_name: str):
    """Check if device (robot or teleoperator) has calibration applied.
    
    Args:
        device: Robot or teleoperator instance
        device_name: Human-readable name for logging
    """
    if hasattr(device, 'is_calibrated'):
        if not device.is_calibrated:
            logger.warning(f"{device_name} is not calibrated. Teleoperation may not work correctly.")
        else:
            logger.info(f"{device_name} calibration verified.")
    else:
        logger.debug(f"{device_name} does not have calibration checking.")

# Create router
router = APIRouter(prefix="/api/aloha-teleoperation", tags=["aloha-teleoperation"])

class OperationMode(str, Enum):
    BIMANUAL = "bimanual"
    LEFT_ONLY = "left_only"
    RIGHT_ONLY = "right_only"

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

class AlohaStartRequest(BaseModel):
    """Start ALOHA teleoperation request (single configuration path)."""
    # Accept loose dict so we can map legacy operation_mode strings (e.g. 'right_arm')
    config: Optional[Dict[str, Any]] = None

# Global state for ALOHA teleoperation
aloha_state = {
    "active": False,
    "robot": None,          # Robot instance in use for teleoperation
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


def create_aloha_configs(config: AlohaConfig):
    """
    Create robot and teleoperator configs for ALOHA using the new unified configuration system.
    """
    try:
        logger.info(f"Creating ALOHA configs for operation mode: {config.operation_mode}")

        # Import new configuration system
        import sys
        import os
        # Add the backend directory to the path so we can import config modules
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)
        
        from config_resolver import resolve
        from lerobot_adapter import to_lerobot_configs
        from config_models import TeleopRequest

        # Map AlohaConfig to TeleopRequest
        # Convert operation_mode from enum to string and map to TeleopRequest format
        op_mode_str = str(config.operation_mode.value) if hasattr(config.operation_mode, 'value') else str(config.operation_mode)
        if op_mode_str == "left_only":
            teleop_op_mode = "left"
        elif op_mode_str == "right_only":
            teleop_op_mode = "right"
        else:  # "bimanual"
            teleop_op_mode = "bimanual"
            
        req = TeleopRequest(
            operation_mode=teleop_op_mode,
            display_data=config.display_data,
            fps=config.fps,
            robot_type="bi_viperx" if teleop_op_mode == "bimanual" else "viperx",
            teleop_type="bi_widowx" if teleop_op_mode == "bimanual" else "widowx",
            cameras_enabled=config.show_cameras,
            profile_name="bi_viperx" if teleop_op_mode == "bimanual" else ("viperx_left" if teleop_op_mode == "left" else "viperx_right")
        )

        # Resolve configuration through layers
        robot, teleop, runtime = resolve(req)
        
        # Apply AlohaConfig parameters to robot and teleop configs
        # This ensures GUI settings override profile defaults
        robot.max_relative_target = config.max_relative_target
        robot.moving_time = config.moving_time
        teleop.max_relative_target = config.max_relative_target
        teleop.moving_time = config.moving_time
        # Note: use_aloha2_gripper_servo is typically set in hardware profiles
        # but can be overridden here if needed in the future

        # Convert to LeRobot configs
        robot_config, teleop_config = to_lerobot_configs(robot, teleop)

        logger.info(f"Resolved configuration: robot={robot.type} ({robot.id}), teleop={teleop.type} ({teleop.id})")
        logger.info(f"Motion parameters: max_relative_target={config.max_relative_target}°, moving_time={config.moving_time}s")

        return robot_config, teleop_config

    except Exception as e:
        logger.error(f"Error creating ALOHA configs: {e}")
        raise


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

def aloha_teleoperation_worker(config: AlohaConfig, reuse_existing: bool):
    """
    Worker thread for ALOHA teleoperation.
    This function now uses new LeRobot factories for robot and teleoperator.
    """
    # Initialize variables at function scope to avoid UnboundLocalError in finally block
    robot = None
    teleop = None
    
    try:
        # Lazy imports here too
        from lerobot.robots.utils import make_robot_from_config
        from lerobot.teleoperators.utils import make_teleoperator_from_config
        if reuse_existing and aloha_state["robot"] is not None:
            robot = aloha_state["robot"]
            logger.info("Reusing already connected robot instance for teleoperation (no reconnection)")
            # Still create the teleoperator and connect it
            _, teleop_config = create_aloha_configs(config)
            teleop = make_teleoperator_from_config(teleop_config)
            # Connect with calibration if file exists (no interactive prompt needed)
            # This writes calibration to motors, which is critical for correct operation
            teleop.connect(calibrate=True)
            _ensure_calibration_applied(robot, "robot (reused)")
            _ensure_calibration_applied(teleop, "teleoperator")
            aloha_state["teleop"] = teleop
            aloha_state["owned_robot"] = False
            logger.info("Teleoperator created and connected; robot reused from RobotService")
        else:
            # Create robot and teleoperator using new factories
            robot_config, teleop_config = create_aloha_configs(config)
            robot = make_robot_from_config(robot_config)
            teleop = make_teleoperator_from_config(teleop_config)
            
            # Connect with calibration if files exist (no interactive prompt needed)
            # This writes calibration to motors, which is critical for correct operation
            # The calibration files must exist, otherwise this will prompt for calibration
            robot.connect(calibrate=True)
            teleop.connect(calibrate=True)
            _ensure_calibration_applied(robot, "robot")
            _ensure_calibration_applied(teleop, "teleoperator")
            
            aloha_state["robot"] = robot
            aloha_state["teleop"] = teleop
            aloha_state["owned_robot"] = True
            logger.info("Robot and teleoperator created and connected using new factories")

        # Prepare processing pipelines similar to CLI teleoperate script
        try:
            (
                teleop_action_processor,
                robot_action_processor,
                robot_observation_processor,
            ) = make_default_processors()
        except Exception:  # pragma: no cover - fallback when processors unavailable
            logger.warning("Default processor pipelines unavailable; using identity fallbacks")

            def teleop_action_processor(payload):  # type: ignore
                return payload[0]

            def robot_action_processor(payload):  # type: ignore
                return payload[0]

            def robot_observation_processor(observation):  # type: ignore
                return observation

        # 2. Prepare for teleoperation loop
        # Initialize rerun if display_data is enabled
        if config.display_data and _RERUN_AVAILABLE:
            logger.info("🖥️ display_data=true: Initializing LeRobot's rerun session...")
            _init_rerun(session_name="lerobot_control_loop_teleop")
            logger.info("✅ LeRobot rerun session initialized.")

        # Start camera streams if requested
        try:
            if config.show_cameras:
                cam_fps = min(config.fps, 12)
                camera_streaming.start_streams(robot, fps=cam_fps)
                logger.info(f"Started camera streaming (fps={cam_fps})")
        except Exception as e:
            logger.warning(f"Failed to start camera streaming: {e}")

        # 3. Simple teleoperation loop (similar to teleoperate.py)
        aloha_state["stage"] = "running"
        logger.info(f"Starting teleoperation loop (fps={config.fps}, display_data={config.display_data})")
        
        # NOTE: We do NOT use async observation worker to avoid bus conflicts!
        # The CLI script works synchronously and so should we for reliability.
        # Multiple threads accessing the Dynamixel bus simultaneously causes
        # "Port is in use" and "Incorrect status packet" errors.

        while not aloha_state["stop_event"].is_set():
            loop_start = time.perf_counter()
            
            # Always fetch observation synchronously to avoid bus conflicts
            try:
                observation = robot.get_observation()
            except Exception as exc:
                logger.debug("Observation fetch failed: %s", exc)
                time.sleep(0.01)
                continue

            if observation is None:
                busy_wait(0.005)
                continue

            # Get raw action from teleoperator
            teleop_raw_action = teleop.get_action()

            # Run through default processing pipelines to mirror CLI behaviour
            processed_teleop_action = teleop_action_processor((teleop_raw_action, observation))
            robot_action_to_send = robot_action_processor((processed_teleop_action, observation))

            # Send processed action to robot
            robot.send_action(robot_action_to_send)

            # Log data to rerun viewer when enabled
            if config.display_data and _RERUN_AVAILABLE:
                try:
                    obs_for_display = robot_observation_processor(observation)
                except Exception:  # pragma: no cover - visualization fallback
                    obs_for_display = observation
                log_rerun_data(obs_for_display, processed_teleop_action)
            
            # Control timing
            dt_s = time.perf_counter() - loop_start
            busy_wait(1 / config.fps - dt_s)
            
            loop_s = time.perf_counter() - loop_start
            # Update performance metrics
            try:
                pm = aloha_state.get("performance_metrics", {})
                prev_avg = float(pm.get("average_fps", 0.0) or 0.0)
                fps_inst = 1.0 / loop_s if loop_s > 0 else 0.0
                avg = (0.8 * prev_avg) + (0.2 * fps_inst)
                pm["average_fps"] = avg
                pm["frames_processed"] = float(pm.get("frames_processed", 0.0) or 0.0) + 1
                over_ms = max(0.0, (loop_s - (1 / config.fps)) * 1000.0)
                pm["latency_ms"] = over_ms
                pm["last_joint_update"] = time.time()
                aloha_state["performance_metrics"] = pm
            except Exception:
                pass

    except Exception as e:
        logger.error(f"Error in ALOHA teleoperation worker: {e}", exc_info=True)
    finally:
        aloha_state["stage"] = "stopping"
        # Stop camera streams before potentially disconnecting
        try:
            camera_streaming.stop_all_streams()
        except Exception as e:
            logger.debug(f"Error stopping camera streams: {e}")

        # Shutdown rerun if it was initialized
        if config.display_data and _RERUN_AVAILABLE:
            try:
                rr.rerun_shutdown()
            except Exception as e:
                logger.debug(f"Error shutting down rerun: {e}")

        if 'robot' in locals() and aloha_state.get("owned_robot") and robot.is_connected:
            try:
                robot.disconnect()
                logger.info("Disconnected owned robot instance after teleoperation")
            except Exception as e:
                logger.warning(f"Error disconnecting owned robot: {e}")
        if 'teleop' in locals() and aloha_state.get("owned_robot"):
            try:
                teleop.disconnect()
                logger.info("Disconnected owned teleoperator instance after teleoperation")
            except Exception as e:
                logger.warning(f"Error disconnecting owned teleoperator: {e}")
            logger.info("Leaving shared robot connected (owned by RobotService)")
        # Note: observation_thread removed - we now use synchronous observation fetching
        if aloha_state.get("owned_robot"):
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
        
        # Determine reuse of existing robot_service robot, otherwise create new
        reuse_existing = False
        try:
            # Import robot module to access global robot instance
            from . import robot as robot_module
            shared_robot = getattr(robot_module, "robot", None)
            if shared_robot and getattr(shared_robot, "is_connected", False):
                if aloha_state.get("robot") is not shared_robot:
                    aloha_state["robot"] = shared_robot
                reuse_existing = True
                logger.info("Shared robot instance detected; will reuse for teleoperation")
            else:
                aloha_state["robot"] = None
        except Exception:
            # Fallback to create new robot instance
            aloha_state["robot"] = None
            reuse_existing = False

        # Start teleoperation state
        aloha_state["active"] = True
        aloha_state["config"] = config.dict()
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
            args=(config, reuse_existing),
            daemon=True
        )
        aloha_state["control_thread"].start()

        # Emit status immediately after start
        try:
            await emit_teleoperation_status()
        except Exception:
            logger.debug("emit teleop status after start failed", exc_info=True)

        logger.info(f"ALOHA teleoperation started with config: {config.dict()}")

        return ApiResponse(
            status="success",
            message="ALOHA teleoperation started successfully",
            data={
                "active": True,
                "configuration": config.dict(),
                "start_time": aloha_state["start_time"],
                "operation_mode": config.operation_mode
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
