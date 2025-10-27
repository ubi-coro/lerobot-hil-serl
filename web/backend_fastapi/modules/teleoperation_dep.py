"""
Advanced Teleoperation Module
============================

Handles teleoperation with advanced features:
- Preset configurations (Safe/Normal/Performance)
- Real-time performance monitoring
- Enhanced safety controls
- Bimanual and leader-only modes

This module contains your sophisticated teleoperation system
that goes beyond LeLab's basic implementation.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import logging
import asyncio
import time
from enum import Enum

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/teleoperation", tags=["teleoperation"])

# Enums for better type safety
class OperationMode(str, Enum):
    BIMANUAL = "bimanual"
    LEADER_ONLY = "leader_only"

class PresetType(str, Enum):
    SAFE = "safe"
    NORMAL = "normal" 
    PERFORMANCE = "performance"

# Pydantic models
class ApiResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None

class TeleoperationConfig(BaseModel):
    """Advanced teleoperation configuration"""
    fps: int = Field(default=30, ge=1, le=120, description="Frames per second")
    max_relative_target: Optional[float] = Field(default=25, ge=0, le=100, description="Maximum relative target (degrees)")
    operation_mode: OperationMode = Field(default=OperationMode.BIMANUAL, description="Operation mode")
    show_cameras: bool = Field(default=True, description="Show camera feeds")
    safety_limits: bool = Field(default=True, description="Enable safety limits")
    performance_monitoring: bool = Field(default=True, description="Enable performance monitoring")

class PresetConfig(BaseModel):
    """Preset configuration selection"""
    preset: PresetType = Field(description="Configuration preset")
    show_cameras: Optional[bool] = Field(default=True, description="Show camera feeds")
    custom_overrides: Optional[Dict[str, Any]] = Field(default={}, description="Custom parameter overrides")

class TeleoperationStart(BaseModel):
    """Start teleoperation request"""
    config: Optional[TeleoperationConfig] = None
    preset: Optional[PresetType] = None

# Preset configurations (your advanced feature)
PRESET_CONFIGURATIONS = {
    PresetType.SAFE: TeleoperationConfig(
        fps=30,
        max_relative_target=5.0,
        operation_mode=OperationMode.BIMANUAL,
        show_cameras=True,
        safety_limits=True,
        performance_monitoring=True
    ),
    PresetType.NORMAL: TeleoperationConfig(
        fps=30,
        max_relative_target=25.0,
        operation_mode=OperationMode.BIMANUAL,
        show_cameras=True,
        safety_limits=True,
        performance_monitoring=True
    ),
    PresetType.PERFORMANCE: TeleoperationConfig(
        fps=60,
        max_relative_target=None,
        operation_mode=OperationMode.BIMANUAL,
        show_cameras=False,  # Disabled for performance
        safety_limits=False,  # Advanced users only
        performance_monitoring=True
    )
}

# Global state
teleoperation_state = {
    "active": False,
    "config": None,
    "start_time": None,
    "performance_metrics": {
        "frames_processed": 0,
        "average_fps": 0.0,
        "latency_ms": 0.0
    }
}

@router.post("/start", response_model=ApiResponse)
async def start_teleoperation(request: TeleoperationStart):
    """
    Start teleoperation with advanced configuration options
    
    Features:
    - Preset configurations (Safe/Normal/Performance)
    - Custom configuration override
    - Real-time performance monitoring
    - Enhanced safety validation
    """
    try:
        logger.info("Starting teleoperation with advanced configuration")
        
        # Determine configuration
        if request.preset:
            config = PRESET_CONFIGURATIONS[request.preset].copy()
            logger.info(f"Using preset configuration: {request.preset}")
        elif request.config:
            config = request.config
            logger.info("Using custom configuration")
        else:
            # Default to normal preset
            config = PRESET_CONFIGURATIONS[PresetType.NORMAL].copy()
            logger.info("Using default (normal) configuration")
        
        # Validate configuration
        if config.max_relative_target and config.max_relative_target > 50:
            logger.warning("High relative target detected, enabling safety limits")
            config.safety_limits = True
        
        # Check if robot is connected (mock check for now)
        # In real implementation, check robot_service.is_connected()
        
        # Start teleoperation (mock implementation)
        teleoperation_state["active"] = True
        teleoperation_state["config"] = config.dict()
        teleoperation_state["start_time"] = time.time()
        
        # Reset performance metrics
        teleoperation_state["performance_metrics"] = {
            "frames_processed": 0,
            "average_fps": 0.0,
            "latency_ms": 0.0
        }
        
        logger.info(f"Teleoperation started with config: {config.dict()}")
        
        return ApiResponse(
            status="success",
            message="Teleoperation started successfully",
            data={
                "active": True,
                "configuration": config.dict(),
                "preset_used": request.preset,
                "start_time": teleoperation_state["start_time"]
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to start teleoperation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start teleoperation: {str(e)}"
        )

@router.post("/stop", response_model=ApiResponse)
async def stop_teleoperation():
    """
    Stop teleoperation and return session summary
    
    Features:
    - Graceful shutdown
    - Performance metrics summary
    - Session statistics
    """
    try:
        logger.info("Stopping teleoperation")
        
        if not teleoperation_state["active"]:
            return ApiResponse(
                status="info",
                message="Teleoperation was not active",
                data={"active": False}
            )
        
        # Calculate session duration
        session_duration = time.time() - teleoperation_state["start_time"] if teleoperation_state["start_time"] else 0
        
        # Stop teleoperation (mock implementation)
        teleoperation_state["active"] = False
        
        # Prepare session summary
        session_summary = {
            "duration_seconds": round(session_duration, 2),
            "performance_metrics": teleoperation_state["performance_metrics"].copy(),
            "configuration_used": teleoperation_state["config"]
        }
        
        # Reset state
        teleoperation_state["config"] = None
        teleoperation_state["start_time"] = None
        
        logger.info(f"Teleoperation stopped. Session duration: {session_duration:.2f}s")
        
        return ApiResponse(
            status="success",
            message="Teleoperation stopped successfully",
            data={
                "active": False,
                "session_summary": session_summary
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to stop teleoperation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to stop teleoperation: {str(e)}"
        )

@router.get("/status", response_model=ApiResponse)
async def get_teleoperation_status():
    """
    Get current teleoperation status and real-time metrics
    
    Returns:
    - Active status
    - Current configuration
    - Real-time performance metrics
    - Session information
    """
    try:
        # Update performance metrics (mock data)
        if teleoperation_state["active"]:
            teleoperation_state["performance_metrics"]["frames_processed"] += 1
            session_duration = time.time() - teleoperation_state["start_time"]
            teleoperation_state["performance_metrics"]["average_fps"] = (
                teleoperation_state["performance_metrics"]["frames_processed"] / session_duration
                if session_duration > 0 else 0
            )
            teleoperation_state["performance_metrics"]["latency_ms"] = 15.5  # Mock latency
        
        status_data = {
            "active": teleoperation_state["active"],
            "configuration": teleoperation_state["config"],
            "performance_metrics": teleoperation_state["performance_metrics"],
            "session_duration": (
                time.time() - teleoperation_state["start_time"] 
                if teleoperation_state["start_time"] else 0
            )
        }
        
        return ApiResponse(
            status="success",
            message="Teleoperation status retrieved",
            data=status_data
        )
        
    except Exception as e:
        logger.error(f"Failed to get teleoperation status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get teleoperation status: {str(e)}"
        )

@router.get("/presets", response_model=ApiResponse)
async def get_available_presets():
    """
    Get available configuration presets
    
    Returns all available preset configurations with their parameters
    """
    try:
        presets_info = {}
        for preset_name, config in PRESET_CONFIGURATIONS.items():
            presets_info[preset_name] = {
                "name": preset_name.title(),
                "description": _get_preset_description(preset_name),
                "configuration": config.dict()
            }
        
        return ApiResponse(
            status="success",
            message="Available presets retrieved",
            data={"presets": presets_info}
        )
        
    except Exception as e:
        logger.error(f"Failed to get presets: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get presets: {str(e)}"
        )

def _get_preset_description(preset: PresetType) -> str:
    """Get description for a preset configuration"""
    descriptions = {
        PresetType.SAFE: "Safe mode with low speed limits and full safety features",
        PresetType.NORMAL: "Balanced performance with safety features enabled", 
        PresetType.PERFORMANCE: "High-performance mode for experienced users"
    }
    return descriptions.get(preset, "Custom configuration")
