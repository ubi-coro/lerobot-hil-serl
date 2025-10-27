"""
Safety and Emergency Systems Module
===================================

Handles all safety-related operations:
- Enhanced emergency stop with API fallback
- Movement limits and safety boundaries
- Real-time safety monitoring
- Safety status reporting

This module implements your advanced safety features that go beyond
basic stop functionality.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import logging
import time
from enum import Enum

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/safety", tags=["safety"])

# Enums
class SafetyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

class EmergencyStopType(str, Enum):
    USER_BUTTON = "user_button"
    KEYBOARD_SHORTCUT = "keyboard_shortcut"
    API_FALLBACK = "api_fallback"
    AUTOMATIC = "automatic"

# Pydantic models
class ApiResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None

class EmergencyStopRequest(BaseModel):
    """Emergency stop request with context"""
    trigger_type: EmergencyStopType = Field(description="What triggered the emergency stop")
    reason: Optional[str] = Field(default="Manual emergency stop", description="Reason for emergency stop")
    force: bool = Field(default=False, description="Force stop even if systems are unresponsive")

class SafetyLimits(BaseModel):
    """Safety limit configuration"""
    max_velocity: float = Field(default=50.0, ge=0, le=100, description="Maximum velocity percentage")
    max_acceleration: float = Field(default=25.0, ge=0, le=100, description="Maximum acceleration percentage")
    workspace_bounds: Dict[str, float] = Field(default={
        "x_min": -0.5, "x_max": 0.5,
        "y_min": -0.5, "y_max": 0.5, 
        "z_min": 0.0, "z_max": 1.0
    }, description="Workspace boundary limits")
    enable_collision_detection: bool = Field(default=True, description="Enable collision detection")

# Global safety state
safety_state = {
    "emergency_stopped": False,
    "safety_level": SafetyLevel.MEDIUM,
    "last_emergency_stop": None,
    "safety_violations": [],
    "limits": SafetyLimits(),
    "monitoring_active": True
}

@router.post("/emergency-stop", response_model=ApiResponse)
async def emergency_stop(request: EmergencyStopRequest):
    """
    Enhanced emergency stop with multiple trigger support
    
    Features:
    - Multiple trigger types (button, keyboard, API, automatic)
    - Force stop option for unresponsive systems
    - Detailed logging and audit trail
    - API fallback mechanism
    """
    try:
        logger.critical(f"EMERGENCY STOP triggered: {request.trigger_type} - {request.reason}")
        
        # Record emergency stop event
        emergency_event = {
            "timestamp": time.time(),
            "trigger_type": request.trigger_type,
            "reason": request.reason,
            "force": request.force,
            "system_state": "active" if not safety_state["emergency_stopped"] else "already_stopped"
        }
        
        # Execute emergency stop sequence
        if request.force:
            logger.critical("FORCE STOP: Bypassing normal shutdown sequence")
            # Immediate hardware stop (mock implementation)
            await _force_emergency_stop()
        else:
            # Graceful emergency stop
            await _graceful_emergency_stop()
        
        # Update safety state
        safety_state["emergency_stopped"] = True
        safety_state["last_emergency_stop"] = emergency_event
        safety_state["safety_level"] = SafetyLevel.CRITICAL
        
        # Stop any active teleoperation
        # This would interface with teleoperation module
        
        logger.critical("Emergency stop completed successfully")
        
        return ApiResponse(
            status="success",
            message="Emergency stop executed successfully",
            data={
                "emergency_stopped": True,
                "trigger_type": request.trigger_type,
                "timestamp": emergency_event["timestamp"],
                "force_stop": request.force,
                "system_secured": True
            }
        )
        
    except Exception as e:
        logger.critical(f"CRITICAL: Emergency stop failed: {e}")
        # Even if emergency stop fails, mark as stopped for safety
        safety_state["emergency_stopped"] = True
        safety_state["safety_level"] = SafetyLevel.CRITICAL
        
        raise HTTPException(
            status_code=500,
            detail=f"Emergency stop failed but system marked as stopped: {str(e)}"
        )

@router.post("/reset", response_model=ApiResponse)
async def reset_safety_system():
    """
    Reset safety system after emergency stop
    
    Features:
    - System health checks before reset
    - Safety validation
    - Gradual system recovery
    - Audit logging
    """
    try:
        logger.info("Resetting safety system")
        
        if not safety_state["emergency_stopped"]:
            return ApiResponse(
                status="info",
                message="Safety system is not in emergency stop state",
                data={"emergency_stopped": False}
            )
        
        # Perform safety checks before reset
        safety_checks = await _perform_safety_checks()
        
        if not safety_checks["all_passed"]:
            logger.warning("Safety checks failed, cannot reset system")
            return ApiResponse(
                status="error",
                message="Safety checks failed, system cannot be reset",
                data={"safety_checks": safety_checks}
            )
        
        # Reset safety state
        safety_state["emergency_stopped"] = False
        safety_state["safety_level"] = SafetyLevel.MEDIUM
        safety_state["safety_violations"] = []
        
        logger.info("Safety system reset successfully")
        
        return ApiResponse(
            status="success",
            message="Safety system reset successfully",
            data={
                "emergency_stopped": False,
                "safety_level": safety_state["safety_level"],
                "safety_checks": safety_checks
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to reset safety system: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset safety system: {str(e)}"
        )

@router.get("/status", response_model=ApiResponse)
async def get_safety_status():
    """
    Get comprehensive safety system status
    
    Returns:
    - Emergency stop status
    - Safety level
    - Active safety limits
    - Recent safety events
    - System health
    """
    try:
        status_data = {
            "emergency_stopped": safety_state["emergency_stopped"],
            "safety_level": safety_state["safety_level"],
            "monitoring_active": safety_state["monitoring_active"],
            "last_emergency_stop": safety_state["last_emergency_stop"],
            "safety_violations_count": len(safety_state["safety_violations"]),
            "recent_violations": safety_state["safety_violations"][-5:],  # Last 5 violations
            "current_limits": safety_state["limits"].dict(),
            "system_health": await _get_system_health()
        }
        
        return ApiResponse(
            status="success",
            message="Safety status retrieved",
            data=status_data
        )
        
    except Exception as e:
        logger.error(f"Failed to get safety status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get safety status: {str(e)}"
        )

@router.post("/limits", response_model=ApiResponse)
async def update_safety_limits(limits: SafetyLimits):
    """
    Update safety limits and boundaries
    
    Features:
    - Velocity and acceleration limits
    - Workspace boundary definitions
    - Collision detection settings
    - Real-time validation
    """
    try:
        logger.info("Updating safety limits")
        
        # Validate limits
        if limits.max_velocity > 80:
            logger.warning("High velocity limit detected, increasing safety monitoring")
            safety_state["safety_level"] = SafetyLevel.HIGH
        
        # Update limits
        safety_state["limits"] = limits
        
        logger.info(f"Safety limits updated: {limits.dict()}")
        
        return ApiResponse(
            status="success",
            message="Safety limits updated successfully",
            data={
                "limits": limits.dict(),
                "safety_level": safety_state["safety_level"]
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to update safety limits: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update safety limits: {str(e)}"
        )

@router.get("/violations", response_model=ApiResponse)
async def get_safety_violations():
    """
    Get recent safety violations and events
    
    Returns audit trail of safety events for analysis
    """
    try:
        violations_data = {
            "total_violations": len(safety_state["safety_violations"]),
            "recent_violations": safety_state["safety_violations"][-10:],  # Last 10
            "violation_types": _analyze_violation_types(),
            "last_emergency_stop": safety_state["last_emergency_stop"]
        }
        
        return ApiResponse(
            status="success",
            message="Safety violations retrieved",
            data=violations_data
        )
        
    except Exception as e:
        logger.error(f"Failed to get safety violations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get safety violations: {str(e)}"
        )

# Helper functions
async def _force_emergency_stop():
    """Force emergency stop (immediate hardware shutdown)"""
    logger.critical("Executing force emergency stop")
    # Mock implementation - would interface with hardware
    pass

async def _graceful_emergency_stop():
    """Graceful emergency stop (proper shutdown sequence)"""
    logger.info("Executing graceful emergency stop")
    # Mock implementation - would:
    # 1. Stop teleoperation
    # 2. Return robot to safe position
    # 3. Disable actuators
    # 4. Stop camera streams
    pass

async def _perform_safety_checks():
    """Perform comprehensive safety checks"""
    # Mock implementation
    return {
        "all_passed": True,
        "hardware_check": True,
        "communication_check": True,
        "sensor_check": True,
        "workspace_clear": True
    }

async def _get_system_health():
    """Get current system health status"""
    # Mock implementation
    return {
        "overall_status": "healthy",
        "hardware_status": "operational",
        "communication_status": "good",
        "sensor_status": "active"
    }

def _analyze_violation_types():
    """Analyze types of safety violations"""
    # Mock implementation
    return {
        "velocity_violations": 0,
        "workspace_violations": 0,
        "collision_events": 0,
        "communication_timeouts": 0
    }
