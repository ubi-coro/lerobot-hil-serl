"""
ALOHA Hardware Configuration API Endpoints
==========================================

API endpoints for ALOHA hardware configuration management.
Bridges LeLab-style hardware configuration with your high-level preset system.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging
import time

# Import our ALOHA hardware config functions
from .aloha_hardware_config import (
    validate_aloha_hardware_config,
    setup_aloha_hardware,
    validate_aloha_configuration,
    load_aloha_hardware_config,
    save_aloha_hardware_config,
    create_aloha_device_mapping_from_config,
    load_aloha_device_mapping,
    save_aloha_device_mapping,
    get_aloha_default_config
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/aloha-hardware", tags=["aloha-hardware"])

# Pydantic models
class ApiResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None

class DeviceMapping(BaseModel):
    """Device mapping configuration"""
    leader_left_port: Optional[str] = None
    leader_right_port: Optional[str] = None
    follower_left_port: Optional[str] = None
    follower_right_port: Optional[str] = None
    
    cam_high_serial: Optional[str] = None
    cam_low_serial: Optional[str] = None
    cam_left_wrist_serial: Optional[str] = None
    cam_right_wrist_serial: Optional[str] = None

@router.get("/detect-devices", response_model=ApiResponse)
async def detect_devices():
    """
    Validate ALOHA hardware configuration against connected devices.
    Since ALOHA systems are pre-configured, this validates rather than discovers.
    """
    try:
        logger.info("Validating ALOHA hardware configuration")
        
        validation_result = validate_aloha_hardware_config()
        
        if validation_result.get("status") == "error":
            raise HTTPException(
                status_code=500,
                detail=f"Hardware validation failed: {validation_result.get('error')}"
            )
        
        return ApiResponse(
            status="success" if validation_result.get("valid", False) else "warning",
            message="ALOHA hardware validation complete",
            data=validation_result
        )
        
    except Exception as e:
        logger.error(f"Error validating ALOHA devices: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to validate ALOHA devices: {str(e)}"
        )

@router.post("/setup", response_model=ApiResponse)
async def setup_hardware():
    """
    Complete ALOHA hardware setup workflow.
    Combines device detection, mapping, and configuration.
    """
    try:
        logger.info("Starting complete ALOHA hardware setup")
        
        setup_result = setup_aloha_hardware()
        
        if not setup_result["success"]:
            return ApiResponse(
                status="warning",
                message="ALOHA hardware setup completed with issues",
                data=setup_result
            )
        
        return ApiResponse(
            status="success",
            message="ALOHA hardware setup completed successfully",
            data=setup_result
        )
        
    except Exception as e:
        logger.error(f"Error in ALOHA hardware setup: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to setup ALOHA hardware: {str(e)}"
        )

@router.get("/config", response_model=ApiResponse)
async def get_hardware_config():
    """Get current ALOHA hardware configuration"""
    try:
        config = load_aloha_hardware_config()
        
        if config is None:
            # Return default config if none exists
            config = get_aloha_default_config()
            return ApiResponse(
                status="info",
                message="No hardware configuration found, returning defaults",
                data={"config": config, "is_default": True}
            )
        
        return ApiResponse(
            status="success",
            message="ALOHA hardware configuration retrieved",
            data={"config": config, "is_default": False}
        )
        
    except Exception as e:
        logger.error(f"Error getting hardware config: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get hardware configuration: {str(e)}"
        )

@router.post("/config", response_model=ApiResponse)
async def save_hardware_config(config: Dict[str, Any]):
    """Save ALOHA hardware configuration"""
    try:
        logger.info("Saving ALOHA hardware configuration")
        
        success = save_aloha_hardware_config(config)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to save hardware configuration"
            )
        
        return ApiResponse(
            status="success",
            message="ALOHA hardware configuration saved",
            data={"config": config}
        )
        
    except Exception as e:
        logger.error(f"Error saving hardware config: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save hardware configuration: {str(e)}"
        )

@router.get("/device-mapping", response_model=ApiResponse)
async def get_device_mapping():
    """Get current ALOHA device mapping"""
    try:
        mapping = load_aloha_device_mapping()
        
        if mapping is None:
            return ApiResponse(
                status="info",
                message="No device mapping found",
                data={"mapping": None}
            )
        
        return ApiResponse(
            status="success",
            message="ALOHA device mapping retrieved",
            data={"mapping": mapping}
        )
        
    except Exception as e:
        logger.error(f"Error getting device mapping: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get device mapping: {str(e)}"
        )

@router.post("/device-mapping", response_model=ApiResponse)
async def update_device_mapping(mapping: DeviceMapping):
    """Update ALOHA device mapping"""
    try:
        logger.info("Updating ALOHA device mapping")
        
        # Convert Pydantic model to dict format expected by hardware config
        mapping_dict = {
            "arms": {
                "leader_left": {"port": mapping.leader_left_port, "status": "manual"},
                "leader_right": {"port": mapping.leader_right_port, "status": "manual"},
                "follower_left": {"port": mapping.follower_left_port, "status": "manual"},
                "follower_right": {"port": mapping.follower_right_port, "status": "manual"}
            },
            "cameras": {
                "cam_high": {"serial_number": mapping.cam_high_serial, "status": "manual"},
                "cam_low": {"serial_number": mapping.cam_low_serial, "status": "manual"},
                "cam_left_wrist": {"serial_number": mapping.cam_left_wrist_serial, "status": "manual"},
                "cam_right_wrist": {"serial_number": mapping.cam_right_wrist_serial, "status": "manual"}
            },
            "updated_at": time.time(),
            "mapping_type": "manual"
        }
        
        success = save_aloha_device_mapping(mapping_dict)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to save device mapping"
            )
        
        return ApiResponse(
            status="success",
            message="ALOHA device mapping updated",
            data={"mapping": mapping_dict}
        )
        
    except Exception as e:
        logger.error(f"Error updating device mapping: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update device mapping: {str(e)}"
        )

@router.post("/device-mapping/from-config", response_model=ApiResponse)
async def create_device_mapping_from_config():
    """
    Create device mapping from predefined ALOHA configuration.
    Uses the standard ALOHA hardware configuration.
    """
    try:
        logger.info("Creating device mapping from standard ALOHA configuration")
        
        mapping = create_aloha_device_mapping_from_config()
        
        if "error" in mapping:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create device mapping: {mapping['error']}"
            )
        
        # Save the mapping
        save_aloha_device_mapping(mapping)
        
        return ApiResponse(
            status="success",
            message="Device mapping created from standard configuration",
            data={"mapping": mapping}
        )
        
    except Exception as e:
        logger.error(f"Error creating device mapping from config: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create device mapping from config: {str(e)}"
        )

@router.get("/validate", response_model=ApiResponse)
async def validate_hardware_config():
    """Validate current ALOHA hardware configuration"""
    try:
        logger.info("Validating ALOHA hardware configuration")
        
        # Get current config to find calibration directory
        config = load_aloha_hardware_config()
        if not config:
            raise HTTPException(
                status_code=404,
                detail="No hardware configuration found to validate"
            )
        
        calibration_dir = config.get("calibration_dir", "~/.cache/lerobot/calibration/aloha_lemgo_tabea")
        validation_result = validate_aloha_configuration(calibration_dir)
        
        status = "success" if validation_result["valid"] else "warning"
        message = "Configuration is valid" if validation_result["valid"] else "Configuration has issues"
        
        return ApiResponse(
            status=status,
            message=message,
            data=validation_result
        )
        
    except Exception as e:
        logger.error(f"Error validating hardware config: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to validate hardware configuration: {str(e)}"
        )

@router.get("/status", response_model=ApiResponse)
async def get_hardware_status():
    """Get overall ALOHA hardware status"""
    try:
        status_data = {
            "hardware_config_exists": load_aloha_hardware_config() is not None,
            "device_mapping_exists": load_aloha_device_mapping() is not None,
            "last_detection": None,
            "setup_complete": False
        }
        
        # Check if setup is complete
        config = load_aloha_hardware_config()
        if config:
            status_data["setup_complete"] = config.get("setup_completed", False)
            status_data["last_detection"] = config.get("saved_at")
        
        # Get validation status
        if config:
            calibration_dir = config.get("calibration_dir", "~/.cache/lerobot/calibration/aloha_lemgo_tabea")
            validation = validate_aloha_configuration(calibration_dir)
            status_data["validation"] = validation
        
        return ApiResponse(
            status="success",
            message="ALOHA hardware status retrieved",
            data=status_data
        )
        
    except Exception as e:
        logger.error(f"Error getting hardware status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get hardware status: {str(e)}"
        )

# Add missing import
import time
