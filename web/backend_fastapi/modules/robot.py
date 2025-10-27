"""
Unified robot connection API for the LeRobot GUI.

This module now routes all hardware connections through the layered configuration
resolver so that the GUI can request bimanual or single-arm setups explicitly.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

import logging
import re
import os
import sys

# Ensure the backend root is on sys.path so we can import config helpers
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import shared
from config_resolver import (
    resolve,
    normalize_operation_mode,
    profile_for_mode,
)
from config_models import TeleopRequest
from lerobot_adapter import to_lerobot_configs

try:  # pragma: no cover - optional hardware dependency
    from lerobot.robots.utils import make_robot_from_config
    _LEROBOT_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    make_robot_from_config = None  # type: ignore
    _LEROBOT_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "LeRobot hardware stack unavailable: %s", exc
    )

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/robot", tags=["robot"])


class ApiResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None


class ConnectRequest(BaseModel):
    """Incoming payload for the connect endpoint."""

    robot_type: str = Field(
        default="aloha", description="Logical robot family requested by the GUI"
    )
    operation_mode: str = Field(
        default="bimanual", description="Desired operation mode (bimanual/left/right)"
    )
    profile_name: Optional[str] = Field(
        default=None, description="Optional hardware profile override"
    )
    show_cameras: bool = Field(
        default=True, description="Enable cameras when resolving the hardware profile"
    )
    display_data: bool = Field(
        default=False, description="Forward display preference to teleoperation runtime"
    )
    fps: int = Field(default=30, ge=1, le=120, description="Target control loop FPS")
    calibrate: bool = Field(
        default=False, description="Allow hardware calibration prompts on connect"
    )
    force_reconnect: bool = Field(
        default=False, description="Disconnect and recreate the robot even if matching"
    )
    overrides: List[str] = Field(
        default_factory=list,
        description="Reserved for legacy override strings (kept for compatibility)",
    )


class DisconnectRequest(BaseModel):
    shutdown: bool = False


# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------
robot = None  # Exposed so other modules (teleoperation) can reuse the instance
_robot_state: Dict[str, Any] = {
    "connected": False,
    "mock_mode": False,
    "mode": None,
    "profile": None,
    "robot_type": None,
    "runtime": {},
    "available_arms": [],
    "cameras": [],
    "robot_cfg": None,
    "teleop_cfg": None,
}


def _arms_for_mode(mode: Optional[str]) -> List[str]:
    if mode == "left":
        return ["left"]
    if mode == "right":
        return ["right"]
    return ["left", "right"]


def _format_connection_error(exc: Exception) -> str:
    message = str(exc).strip()
    if not message:
        return "Failed to connect to the robot hardware."  # pragma: no cover - fallback

    match = re.search(r"RealSenseCamera\(([^)]+)\)", message)
    if match:
        camera_id = match.group(1)
        return (
            "Unable to start Intel RealSense camera "
            f"{camera_id}. Please unplug and reconnect the camera's USB cable, "
            "power-cycle the USB hub if necessary, and try Connect again."
        )

    return f"Failed to connect to the robot hardware: {message}"


def _build_status(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    data = {
        "connected": _robot_state["connected"],
        "mock_mode": _robot_state["mock_mode"],
        "mode": _robot_state["mode"],
        "profile": _robot_state["profile"],
        "robot_type": _robot_state["robot_type"],
        "available_arms": _robot_state["available_arms"],
        "cameras": _robot_state["cameras"],
        "runtime": _robot_state["runtime"],
    }
    if extra:
        data.update(extra)
    return data


def _reset_state() -> None:
    _robot_state.update(
        {
            "connected": False,
            "mock_mode": False,
            "mode": None,
            "profile": None,
            "robot_type": None,
            "runtime": {},
            "available_arms": [],
            "cameras": [],
            "robot_cfg": None,
            "teleop_cfg": None,
        }
    )


def _resolve_hardware(request: ConnectRequest, normalized_mode: str):
    profile = request.profile_name or profile_for_mode(normalized_mode)
    teleop_request = TeleopRequest(
        operation_mode=normalized_mode,
        display_data=request.display_data,
        fps=request.fps,
        robot_type="bi_viperx" if normalized_mode == "bimanual" else "viperx",
        teleop_type="bi_widowx" if normalized_mode == "bimanual" else "widowx",
        cameras_enabled=request.show_cameras,
        profile_name=profile,
    )
    robot_cfg, teleop_cfg, runtime_cfg = resolve(teleop_request)
    robot_config, teleop_config = to_lerobot_configs(robot_cfg, teleop_cfg)
    return profile, robot_cfg, teleop_cfg, runtime_cfg, robot_config, teleop_config


def current_operation_mode() -> Optional[str]:
    """Expose the active robot mode for other modules (e.g. teleoperation)."""
    return _robot_state.get("mode")


def resolved_configs() -> Dict[str, Any]:
    """Return the last resolved Pydantic configs for debugging or reuse."""
    return {
        "robot_cfg": _robot_state.get("robot_cfg"),
        "teleop_cfg": _robot_state.get("teleop_cfg"),
        "runtime": _robot_state.get("runtime"),
    }


@router.post("/connect", response_model=ApiResponse)
async def connect_robot(request: ConnectRequest):
    """Connect to the robot using the layered configuration system."""

    normalized_mode = normalize_operation_mode(request.operation_mode)
    logger.info(
        "Connecting robot (type=%s, mode=%s, profile=%s)",
        request.robot_type,
        normalized_mode,
        request.profile_name,
    )

    global robot
    already_connected = robot and getattr(robot, "is_connected", False)
    if already_connected:
        same_mode = _robot_state.get("mode") == normalized_mode
        if same_mode and not request.force_reconnect:
            logger.info("Robot already connected; reusing existing session")
            return ApiResponse(
                status="success",
                message="Robot already connected",
                data=_build_status(),
            )
        # Disconnect the existing session before reconnecting
        try:
            robot.disconnect()
        except Exception as exc:  # pragma: no cover
            logger.warning("Error while disconnecting previous robot: %s", exc)
        finally:
            robot = None
            _reset_state()

    try:
        profile, robot_cfg, teleop_cfg, runtime_cfg, robot_config, _ = _resolve_hardware(
            request, normalized_mode
        )
        if not _LEROBOT_AVAILABLE or make_robot_from_config is None:
            raise RuntimeError(
                "LeRobot hardware dependencies are not installed for this environment"
            )

        robot_instance = make_robot_from_config(robot_config)
        robot_instance.connect(calibrate=request.calibrate)

        robot = robot_instance
        _robot_state.update(
            {
                "connected": True,
                "mock_mode": False,
                "mode": normalized_mode,
                "profile": profile,
                "robot_type": request.robot_type,
                "runtime": runtime_cfg,
                "available_arms": _arms_for_mode(normalized_mode),
                "cameras": list(robot_cfg.cameras.keys()),
                "robot_cfg": robot_cfg,
                "teleop_cfg": teleop_cfg,
            }
        )
        try:
            shared.emit_threadsafe("camera_list", {"cameras": _robot_state["cameras"]})
        except Exception:
            logger.debug("Failed to emit camera list on connect", exc_info=True)
        logger.info(
            "Robot connected successfully (mode=%s, profile=%s)",
            normalized_mode,
            profile,
        )
        return ApiResponse(
            status="success",
            message=f"Robot connected ({normalized_mode})",
            data=_build_status(),
        )
    except Exception as exc:
        logger.error("Robot connection failed: %s", exc, exc_info=True)
        _reset_state()
        _robot_state.update(
            {
                "connected": False,
                "mock_mode": True,
                "mode": normalized_mode,
                "profile": request.profile_name or profile_for_mode(normalized_mode),
                "robot_type": request.robot_type,
                "available_arms": _arms_for_mode(normalized_mode),
            }
        )
        user_message = _format_connection_error(exc)
        raise HTTPException(status_code=500, detail=user_message)


@router.post("/disconnect", response_model=ApiResponse)
async def disconnect_robot(_: DisconnectRequest = DisconnectRequest()):
    """Disconnect from the robot and reset runtime state."""

    global robot
    try:
        if robot and getattr(robot, "is_connected", False):
            robot.disconnect()
            logger.info("Robot disconnected successfully")
        robot = None
        _reset_state()
        return ApiResponse(
            status="success",
            message="Robot disconnected",
            data=_build_status(),
        )
    except Exception as exc:
        logger.error("Robot disconnection failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to disconnect robot: {exc}",
        )


@router.get("/status", response_model=ApiResponse)
async def get_robot_status():
    """Return the current robot connection status."""

    return ApiResponse(
        status="success",
        message="Robot status retrieved",
        data=_build_status(),
    )


@router.get("/info", response_model=ApiResponse)
async def get_robot_info():
    """Expose high-level robot capabilities to the GUI."""

    info = {
        "robot_type": _robot_state.get("robot_type") or "aloha",
        "mock_mode": _robot_state.get("mock_mode", True),
        "supported_modes": ["bimanual", "left", "right"],
        "active_mode": _robot_state.get("mode"),
        "profile": _robot_state.get("profile"),
        "available_arms": _robot_state.get("available_arms"),
        "features": [
            "teleoperation",
            "recording",
            "emergency_stop",
            "configuration_presets",
            "performance_monitoring",
        ],
    }
    return ApiResponse(
        status="success",
        message="Robot information retrieved",
        data=info,
    )


@router.get("/configs", response_model=ApiResponse)
async def get_robot_configs():
    """Return supported presets and the currently connected hardware summary."""

    data = {
        "presets": [
            {"name": "bimanual", "description": "Dual-arm ALOHA configuration"},
            {"name": "left", "description": "Single left arm (leader/follower)"},
            {"name": "right", "description": "Single right arm (leader/follower)"},
        ],
        "active_profile": _robot_state.get("profile"),
        "available_arms": _robot_state.get("available_arms"),
        "mode": _robot_state.get("mode"),
        "mock_mode": _robot_state.get("mock_mode"),
    }
    return ApiResponse(
        status="success",
        message="Robot configs retrieved",
        data=data,
    )
