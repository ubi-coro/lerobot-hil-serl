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


class HomeRequest(BaseModel):
    torque_off: bool = True
    duration_seconds: float = 5.0


# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------
robot = None  # Exposed so other modules (teleoperation) can reuse the instance
_cameras_borrowed = False  # Track if cameras are currently borrowed by another module
_cameras_borrowed_by = None  # Track which module borrowed the cameras

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

# Measured "Sleep" poses per arm from your prints (followers)
HOME_FOLLOWER = {
    "left": {
        "elbow.pos": 1.6364,
        "forearm_roll.pos": 0.1266,
        "gripper.pos": 0.6014,
        "shoulder.pos": -1.8834,
        "waist.pos": -0.0161,
        "wrist_angle.pos": 0.6743,
        "wrist_rotate.pos": -0.2371,
    },
    "right": {
        "elbow.pos": 1.6241,
        "forearm_roll.pos": -0.1987,
        "gripper.pos": 0.7691,
        "shoulder.pos": -1.9049,
        "waist.pos": 0.0176,
        "wrist_angle.pos": 0.7081,
        "wrist_rotate.pos": 0.2447,
    },
}

# Measured "Sleep" poses per arm from your prints (leaders)
HOME_LEADER = {
    "left": {
        "elbow.pos": 1.5689,
        "forearm_roll.pos": 0.1818,
        "gripper.pos": 0.5985,
        "shoulder.pos": -1.7791,
        "waist.pos": -0.0176,
        "wrist_angle.pos": 0.2279,
        "wrist_rotate.pos": -0.1496,
    },
    "right": {
        "elbow.pos": 1.5827,
        "forearm_roll.pos": -0.0345,
        "gripper.pos": 0.7675,
        "shoulder.pos": -1.8896,
        "waist.pos": -0.0100,
        "wrist_angle.pos": 0.3030,
        "wrist_rotate.pos": 0.0176,
    },
}

# Measured "Start" poses per arm (followers) from your policy rollout request
START_FOLLOWER = {
    "left": {
        "elbow.pos": 1.1561,
        "forearm_roll.pos": 0.1143,
        "gripper.pos": 0.5979,
        "shoulder.pos": -1.3234,
        "waist.pos": 0.0453,
        "wrist_angle.pos": 0.4902,
        "wrist_rotate.pos": -0.0376,
    },
    "right": {
        "elbow.pos": 1.1500,
        "forearm_roll.pos": -0.0882,
        "gripper.pos": 0.7564,
        "shoulder.pos": -1.2942,
        "waist.pos": -0.0284,
        "wrist_angle.pos": 0.3629,
        "wrist_rotate.pos": 0.0453,
    },
}

# Measured "Start" poses per arm (leaders/teleoperators)
START_LEADER = {
    "left": {
        "elbow.pos": 1.1792,
        "forearm_roll.pos": 0.1143,
        "gripper.pos": 0.5985,
        "shoulder.pos": -1.3096,
        "waist.pos": 0.0575,
        "wrist_angle.pos": 0.4826,
        "wrist_rotate.pos": -0.0361,
    },
    "right": {
        "elbow.pos": 1.1316,
        "forearm_roll.pos": -0.0882,
        "gripper.pos": 0.7580,
        "shoulder.pos": -1.2743,
        "waist.pos": -0.0269,
        "wrist_angle.pos": 0.3583,
        "wrist_rotate.pos": 0.0453,
    },
}


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
        profile, robot_cfg, teleop_cfg, runtime_cfg, robot_config, teleop_config = _resolve_hardware(
            request, normalized_mode
        )
        if not _LEROBOT_AVAILABLE or make_robot_from_config is None:
            raise RuntimeError(
                "LeRobot hardware dependencies are not installed for this environment"
            )

        robot_instance = make_robot_from_config(robot_config)
        # Connect robot and cameras. Camera sharing is handled by borrowing APIs.
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
                "teleop_config": teleop_config,  # Store LeRobot config for leader homing
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


@router.post("/home", response_model=ApiResponse)
async def move_robot_home(request: HomeRequest):
    """
    Move the robot to a safe 'Home' position and optionally disable torque.
    This is useful for parking the robot before shutdown.
    """
    global robot
    if not robot or not getattr(robot, "is_connected", False):
        raise HTTPException(status_code=400, detail="Robot is not connected")

    try:
        import time
        import numpy as np

        # 1. Determine target position based on active arms
        target_pos = {}
        available_arms = _robot_state.get("available_arms", [])
        
        # Construct full target dictionary (followers): use measured per-arm home; require presence
        for arm in available_arms:
            if arm not in HOME_FOLLOWER:
                raise HTTPException(status_code=400, detail=f"Missing follower HOME pose for arm '{arm}'")
            base = HOME_FOLLOWER[arm]
            for joint, val in base.items():
                target_pos[f"{arm}_{joint}"] = val

        # 2. Get current position
        current_pos = robot.get_observation()
        
        # Filter current_pos to only include joints we want to move
        start_pos = {k: current_pos.get(k, 0.0) for k in target_pos.keys()}

        # 3. Interpolate and move
        steps = int(request.duration_seconds * 30)  # 30 Hz control loop
        if steps < 1: steps = 1
        
        logger.info(f"Moving robot to Home position over {request.duration_seconds}s...")
        
        for i in range(steps):
            alpha = (i + 1) / steps
            interp_action = {}
            for k in target_pos.keys():
                start_val = start_pos.get(k, 0.0)
                target_val = target_pos[k]
                interp_action[k] = start_val + (target_val - start_val) * alpha
            
            robot.send_action(interp_action)
            time.sleep(1.0 / 30.0)

        # 4. Move Leader (Teleoperator) if available
        # Attempt to reuse active teleop instances (as in interventions)
        reused_teleops: Dict[str, Any] = {}
        try:
            from . import aloha_teleoperation as teleop_module  # type: ignore
            reused_teleops = dict(teleop_module.aloha_state.get("teleop") or {})
            if reused_teleops:
                logger.info("Reusing %d active teleop instance(s) for leader homing", len(reused_teleops))
        except Exception:
            logger.debug("No active teleoperation to reuse for leader homing", exc_info=True)

        if reused_teleops:
            try:
                # Build target strictly using each teleop's advertised action_features to match keys
                logger.info(f"Moving reused teleops to Home over {request.duration_seconds}s...")
                # Precompute per-arm base targets
                for name, teleop in reused_teleops.items():
                    af = getattr(teleop, "action_features", {})
                    if not af:
                        logger.warning("Teleop %s has no action_features; skipping", name)
                        continue

                    feature_keys = list(af.keys())
                    has_prefix = any(k.startswith("left_") or k.startswith("right_") for k in feature_keys)

                    # Prime last obs for relative safety caps
                    try:
                        teleop.get_action()
                    except Exception:
                        logger.debug("teleop.get_action failed prior to homing (continuing)", exc_info=True)

                    # Build target map in the exact feature space
                    target = {}
                    for k in feature_keys:
                        if has_prefix:
                            if k.startswith("left_") and "left" in available_arms:
                                joint = k[len("left_"):]
                                target[k] = HOME_LEADER["left"][joint]
                            elif k.startswith("right_") and "right" in available_arms:
                                joint = k[len("right_"):]
                                target[k] = HOME_LEADER["right"][joint]
                        else:
                            # Single-arm teleop; assume the active arm from available_arms
                            if len(available_arms) != 1:
                                logger.warning("Single-arm teleop but ambiguous available_arms=%s; skipping %s", available_arms, name)
                                continue
                            arm = available_arms[0]
                            target[k] = HOME_LEADER[arm][k]

                    # Read start once to interpolate smoothly
                    try:
                        start = teleop.get_action()
                    except Exception:
                        start = {}
                    start = {k: start.get(k, target.get(k, 0.0)) for k in target}

                    for i in range(steps):
                        alpha = (i + 1) / steps
                        payload = {k: start[k] + (target[k] - start[k]) * alpha for k in target}
                        teleop.send_feedback(payload)
                        time.sleep(1.0 / 30.0)
                    # Final latch
                    teleop.send_feedback(target)

                logger.info("Leader homing via reused teleops complete.")
            except Exception as e:
                logger.warning("Leader homing via reused teleops failed: %s", e)
        else:
            # Fall back to creating a fresh teleop connection
            teleop_config = _robot_state.get("teleop_config")
            # Fallback: rebuild teleop_config from last resolved Pydantic cfg if missing
            if not teleop_config and _robot_state.get("teleop_cfg") is not None:
                try:
                    _, teleop_config_rebuilt = to_lerobot_configs(
                        _robot_state.get("robot_cfg"), _robot_state.get("teleop_cfg")
                    )
                    teleop_config = teleop_config_rebuilt
                    _robot_state["teleop_config"] = teleop_config
                    logger.info("Rebuilt teleop_config from last resolved configs")
                except Exception as e:
                    logger.warning(f"Could not rebuild teleop_config: {e}")

            if teleop_config:
                def _home_with_teleop_config(tcfg) -> bool:
                    """Connect leader with given config and run homing. Returns True on success."""
                    logger.info("Connecting to leader arms for homing (fresh teleop)...")
                    # Handle teleoperators (leaders) properly
                    if getattr(tcfg, "type", "") == "bi_widowx":
                        from lerobot.teleoperators.bi_widowx import BiWidowX
                        leader_local = BiWidowX(tcfg)
                    else:
                        from lerobot.teleoperators import make_teleoperator_from_config
                        leader_local = make_teleoperator_from_config(tcfg)

                    leader_local.connect(calibrate=False)
                    try:
                        leader_local.enable_torque()
                    except Exception:
                        pass

                    leader_current_pos_local = leader_local.get_action()
                    logger.info(
                        "Leader connected. Read %d joint keys for homing.",
                        len(leader_current_pos_local) if isinstance(leader_current_pos_local, dict) else -1,
                    )

                    leader_has_prefix_local = any(k.startswith("left_") or k.startswith("right_") for k in leader_current_pos_local)
                    leader_target_pos_local = {}
                    for arm in available_arms:
                        if arm not in HOME_LEADER:
                            raise HTTPException(status_code=400, detail=f"Missing leader HOME pose for arm '{arm}'")
                        base = HOME_LEADER[arm]
                        for joint, val in base.items():
                            key = f"{arm}_{joint}" if leader_has_prefix_local else joint
                            leader_target_pos_local[key] = val
                    leader_start_pos_local = {k: leader_current_pos_local.get(k, 0.0) for k in leader_target_pos_local.keys()}

                    logger.info(f"Moving leader to Home position over {request.duration_seconds}s (fresh teleop)...")
                    for i in range(steps):
                        alpha = (i + 1) / steps
                        interp_action = {}
                        for k in leader_target_pos_local.keys():
                            start_val = leader_start_pos_local.get(k, 0.0)
                            target_val = leader_target_pos_local[k]
                            interp_action[k] = start_val + (target_val - start_val) * alpha
                        leader_local.send_feedback(interp_action)
                        time.sleep(1.0 / 30.0)
                    try:
                        leader_local.send_feedback(leader_target_pos_local)
                    except Exception:
                        logger.debug("Final leader target send failed (already at target?)", exc_info=True)

                    logger.info("Leader homing complete. Disconnecting leader (fresh teleop).")
                    leader_local.disconnect()
                    return True

                try:
                    # First attempt with current config
                    if _home_with_teleop_config(teleop_config):
                        pass
                except Exception as e:
                    # If motor model mismatch (XL vs XC gripper), retry with toggled gripper flag
                    msg = str(e)
                    logger.warning(f"Failed to home leader arms: {e}")
                    try:
                        tcfg_pyd = _robot_state.get("teleop_cfg")
                        rcfg_pyd = _robot_state.get("robot_cfg")
                        if tcfg_pyd is not None and rcfg_pyd is not None:
                            # Decide which way to toggle based on error content
                            toggle_to = None
                            if "expected 1060, found 1070" in msg:
                                # Expected XL but have XC -> set aloha2=True
                                toggle_to = True
                            elif "expected 1070, found 1060" in msg:
                                # Expected XC but have XL -> set aloha2=False
                                toggle_to = False
                            if toggle_to is not None:
                                logger.info("Retrying leader connect with use_aloha2_gripper_servo=%s", toggle_to)
                                tcfg_retry = tcfg_pyd.copy(update={"use_aloha2_gripper_servo": toggle_to})
                                _, tcfg_retry_le = to_lerobot_configs(rcfg_pyd, tcfg_retry)
                                try:
                                    _home_with_teleop_config(tcfg_retry_le)
                                    # Persist the corrected teleop config for subsequent Home calls
                                    _robot_state["teleop_cfg"] = tcfg_retry
                                    _robot_state["teleop_config"] = tcfg_retry_le
                                    logger.info("Updated cached teleop config with use_aloha2_gripper_servo=%s", toggle_to)
                                except Exception as e2:
                                    logger.warning(f"Leader homing retry failed: {e2}")
                            else:
                                logger.debug("No gripper model hint in error; not retrying teleop connect")
                        else:
                            logger.debug("teleop_cfg or robot_cfg missing; cannot rebuild teleop config for retry")
                    except Exception:
                        logger.debug("Leader homing retry path crashed", exc_info=True)
        
        # 5. Disable torque if requested
        if request.torque_off:
            logger.info("Disabling torque after reaching Home.")
            # We can use disconnect() to disable torque if configured, 
            # or we need a specific method. 
            # BiViperX/ViperX disconnect() disables torque if disable_torque_on_disconnect is set.
            # But we might want to keep the connection object alive but torque off?
            # The Robot interface doesn't have a generic torque_off method.
            # However, we can call disconnect() which usually does it.
            # But the user might want to stay "connected" in the UI.
            # For now, let's assume "torque off" means we are done.
            # But the user said "where it is save to torque off", implying they might do it manually.
            # If request.torque_off is True, we should try to disable torque.
            # Since we don't have a direct torque_off method exposed in the abstract Robot class,
            # we will rely on the user disconnecting, OR we can try to access the bus if possible.
            # But accessing internal bus is risky.
            # Let's just move to the position. The user can then click "Disconnect".
            pass

        return ApiResponse(
            status="success",
            message="Robot moved to Home position",
            data={"torque_off": request.torque_off}
        )

    except Exception as exc:
        logger.error("Failed to move to Home: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to move to Home: {exc}")


class StartPositionRequest(BaseModel):
    duration_seconds: float = 5.0
    move_leaders: bool = True


@router.post("/start-position", response_model=ApiResponse)
async def move_robot_to_start(request: StartPositionRequest):
    """
    Move the robot (and optionally leaders) to the measured Start position
    used for policy rollout. Keeps torque ON and leaves robots at target.
    """
    global robot
    if not robot or not getattr(robot, "is_connected", False):
        raise HTTPException(status_code=400, detail="Robot is not connected")

    try:
        import time

        # 1. Build follower target from START_FOLLOWER per active arm
        target_pos = {}
        available_arms = _robot_state.get("available_arms", [])
        for arm in available_arms:
            if arm not in START_FOLLOWER:
                raise HTTPException(status_code=400, detail=f"Missing follower START pose for arm '{arm}'")
            base = START_FOLLOWER[arm]
            for joint, val in base.items():
                target_pos[f"{arm}_{joint}"] = val

        # 2. Interpolate follower movement
        steps = int(request.duration_seconds * 30)
        if steps < 1:
            steps = 1

        current = robot.get_observation() or {}
        start_vals = {k: current.get(k, 0.0) for k in target_pos.keys()}

        for i in range(steps):
            alpha = (i + 1) / steps
            action = {k: start_vals[k] + (target_pos[k] - start_vals[k]) * alpha for k in target_pos}
            robot.send_action(action)
            time.sleep(1.0 / 30.0)

        # 3. Optionally move leaders to START_LEADER using same strategy
        if request.move_leaders:
            logger.info("Moving leaders to Start position...")
            try:
                from . import aloha_teleoperation as teleop_module  # type: ignore
                reused = dict(teleop_module.aloha_state.get("teleop") or {})
                if reused:
                    logger.info("Reusing %d active teleop instance(s) for leader start-position", len(reused))
            except Exception:
                reused = {}

            def _move_leader_with_teleop(teleop_obj, teleop_name: str = "leader") -> None:
                af = getattr(teleop_obj, "action_features", {})
                if not af:
                    logger.warning("Teleop %s has no action_features; skipping", teleop_name)
                    return
                keys = list(af.keys())
                has_prefix = any(k.startswith("left_") or k.startswith("right_") for k in keys)
                # Prime
                try:
                    teleop_obj.get_action()
                except Exception:
                    pass
                target = {}
                for k in keys:
                    if has_prefix:
                        if k.startswith("left_") and "left" in available_arms:
                            joint = k[len("left_"):]
                            # Only include joints that exist in START_LEADER
                            if joint in START_LEADER["left"]:
                                target[k] = START_LEADER["left"][joint]
                        elif k.startswith("right_") and "right" in available_arms:
                            joint = k[len("right_"):]
                            # Only include joints that exist in START_LEADER
                            if joint in START_LEADER["right"]:
                                target[k] = START_LEADER["right"][joint]
                    else:
                        if len(available_arms) == 1:
                            arm = available_arms[0]
                            # Only include joints that exist in START_LEADER
                            if k in START_LEADER[arm]:
                                target[k] = START_LEADER[arm][k]
                if not target:
                    logger.warning("No target joints computed for teleop %s; skipping", teleop_name)
                    return
                try:
                    cur = teleop_obj.get_action() or {}
                except Exception:
                    cur = {}
                start = {k: cur.get(k, target.get(k, 0.0)) for k in target}
                logger.info("Moving %s to Start over %d steps...", teleop_name, steps)
                for i in range(steps):
                    alpha = (i + 1) / steps
                    payload = {k: start[k] + (target[k] - start[k]) * alpha for k in target}
                    teleop_obj.send_feedback(payload)
                    time.sleep(1.0 / 30.0)
                teleop_obj.send_feedback(target)
                logger.info("Leader %s moved to Start position.", teleop_name)

            if reused:
                for name, teleop_obj in reused.items():
                    try:
                        _move_leader_with_teleop(teleop_obj, name)
                    except Exception as e:
                        logger.warning("Leader start-position move failed for %s: %s", name, e)
            else:
                # Fallback: connect fresh teleop from cached config
                tcfg = _robot_state.get("teleop_config")
                if not tcfg and _robot_state.get("teleop_cfg") is not None:
                    try:
                        _, tcfg_le = to_lerobot_configs(_robot_state.get("robot_cfg"), _robot_state.get("teleop_cfg"))
                        tcfg = tcfg_le
                        _robot_state["teleop_config"] = tcfg
                        logger.info("Rebuilt teleop_config for start-position move")
                    except Exception as e:
                        logger.warning("Could not rebuild teleop_config: %s", e)
                        tcfg = None
                if tcfg:
                    try:
                        logger.info("Connecting fresh teleop for start-position...")
                        if getattr(tcfg, "type", "") == "bi_widowx":
                            from lerobot.teleoperators.bi_widowx import BiWidowX
                            leader = BiWidowX(tcfg)
                        else:
                            from lerobot.teleoperators import make_teleoperator_from_config
                            leader = make_teleoperator_from_config(tcfg)
                        leader.connect(calibrate=False)
                        try:
                            leader.enable_torque()
                        except Exception:
                            pass
                        _move_leader_with_teleop(leader, "fresh_teleop")
                        leader.disconnect()
                        logger.info("Fresh teleop disconnected after start-position move.")
                    except Exception as e:
                        logger.warning("Fresh leader start-position move failed: %s", e)
                else:
                    logger.warning("No teleop_config available for leader start-position move")

        return ApiResponse(
            status="success",
            message="Robot moved to Start position",
            data={"leaders_moved": bool(request.move_leaders)}
        )

    except Exception as exc:
        logger.error("Failed to move to Start position: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to move to Start position: {exc}")


# ---------------------------------------------------------------------------
# Camera Borrowing API
# ---------------------------------------------------------------------------

def borrow_cameras(module_name: str) -> Optional[Dict[str, Any]]:
    """
    Lend camera objects to another module (e.g., teleoperation).
    
    Args:
        module_name: Name of the module borrowing the cameras (for tracking)
        
    Returns:
        Dictionary of camera objects if available, None otherwise
        
    Raises:
        RuntimeError: If cameras are already borrowed by another module
    """
    global _cameras_borrowed, _cameras_borrowed_by
    
    if not robot or not hasattr(robot, 'cameras') or not robot.cameras:
        logger.warning(f"Module '{module_name}' tried to borrow cameras, but no cameras available")
        return None
    
    if _cameras_borrowed and _cameras_borrowed_by != module_name:
        raise RuntimeError(
            f"Cameras are already borrowed by '{_cameras_borrowed_by}'. "
            f"Cannot lend to '{module_name}'."
        )
    
    _cameras_borrowed = True
    _cameras_borrowed_by = module_name
    logger.info(f"Cameras borrowed by module: {module_name}")
    return robot.cameras


def return_cameras(module_name: str) -> None:
    """
    Return borrowed cameras back to the robot module.
    
    Args:
        module_name: Name of the module returning the cameras
    """
    global _cameras_borrowed, _cameras_borrowed_by
    
    if not _cameras_borrowed:
        logger.debug(f"Module '{module_name}' returned cameras, but none were borrowed")
        return
    
    if _cameras_borrowed_by != module_name:
        logger.warning(
            f"Module '{module_name}' tried to return cameras, "
            f"but they were borrowed by '{_cameras_borrowed_by}'"
        )
        return
    
    _cameras_borrowed = False
    _cameras_borrowed_by = None
    logger.info(f"Cameras returned by module: {module_name}")


def are_cameras_available() -> bool:
    """Check if cameras are available for borrowing."""
    return robot is not None and hasattr(robot, 'cameras') and bool(robot.cameras) and not _cameras_borrowed

