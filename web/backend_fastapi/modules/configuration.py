"""
Configuration Management Module
==============================

Handles all configuration operations:
- Robot preset management (Safe/Normal/Performance)
- System settings and parameters
- User preferences and profiles
- Configuration validation and backup
- Import/export capabilities

This module centralizes all configuration management
with your advanced preset system.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List, Union
import logging
import json

# Import ExperimentConfigMapper for demo config
try:
    from ..experiment_config_mapper import ExperimentConfigMapper
except ImportError:
    # Fallback for when running as main module
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from experiment_config_mapper import ExperimentConfigMapper
import time
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/configuration", tags=["configuration"])

# Enums
class PresetType(str, Enum):
    SAFE = "safe"
    NORMAL = "normal"
    PERFORMANCE = "performance"
    CUSTOM = "custom"

class ConfigScope(str, Enum):
    SYSTEM = "system"
    USER = "user"
    SESSION = "session"
    ROBOT = "robot"

# Pydantic models
class ApiResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None

class RobotPreset(BaseModel):
    """Robot operation preset configuration"""
    name: str = Field(description="Preset name")
    type: PresetType = Field(description="Preset type")
    description: str = Field(description="Preset description")
    
    # Movement parameters
    max_velocity: float = Field(ge=0.1, le=100.0, description="Maximum velocity percentage")
    max_acceleration: float = Field(ge=0.1, le=100.0, description="Maximum acceleration percentage")
    movement_smoothing: float = Field(ge=0.0, le=1.0, description="Movement smoothing factor")
    
    # Safety parameters
    collision_sensitivity: float = Field(ge=0.1, le=1.0, description="Collision detection sensitivity")
    workspace_limits: Dict[str, float] = Field(description="Workspace boundary limits")
    emergency_stop_delay: float = Field(ge=0.0, le=2.0, description="Emergency stop reaction delay")
    
    # Camera parameters
    camera_fps: int = Field(ge=5, le=60, description="Camera frame rate")
    camera_quality: str = Field(description="Camera quality setting")
    enable_recording: bool = Field(default=True, description="Enable automatic recording")
    
    # Advanced features
    predictive_assistance: bool = Field(default=False, description="Enable predictive movement assistance")
    auto_grip_adjustment: bool = Field(default=False, description="Enable automatic grip adjustment")
    motion_learning: bool = Field(default=False, description="Enable motion learning")
    
    # Performance monitoring
    enable_performance_monitoring: bool = Field(default=True, description="Enable performance monitoring")
    telemetry_level: str = Field(default="standard", description="Telemetry collection level")

class SystemConfiguration(BaseModel):
    """System-level configuration"""
    # Network settings
    websocket_timeout: int = Field(default=30, ge=5, le=300, description="WebSocket timeout in seconds")
    api_rate_limit: int = Field(default=100, ge=10, le=1000, description="API rate limit per minute")
    max_concurrent_connections: int = Field(default=10, ge=1, le=50, description="Maximum concurrent connections")
    
    # Logging settings
    log_level: str = Field(default="INFO", description="System log level")
    log_retention_days: int = Field(default=7, ge=1, le=30, description="Log retention in days")
    enable_debug_mode: bool = Field(default=False, description="Enable debug mode")
    
    # Storage settings
    max_storage_gb: float = Field(default=10.0, ge=1.0, le=100.0, description="Maximum storage usage in GB")
    auto_cleanup: bool = Field(default=True, description="Enable automatic cleanup")
    backup_enabled: bool = Field(default=True, description="Enable configuration backup")

class UserPreferences(BaseModel):
    """User-specific preferences"""
    # UI preferences
    theme: str = Field(default="light", description="UI theme")
    language: str = Field(default="en", description="Interface language")
    keyboard_shortcuts: Dict[str, str] = Field(default={}, description="Custom keyboard shortcuts")
    
    # Default settings
    default_preset: PresetType = Field(default=PresetType.NORMAL, description="Default robot preset")
    auto_connect: bool = Field(default=False, description="Auto-connect to robot on startup")
    show_advanced_controls: bool = Field(default=False, description="Show advanced control options")
    
    # Notification preferences
    enable_notifications: bool = Field(default=True, description="Enable system notifications")
    notification_level: str = Field(default="important", description="Notification verbosity level")

class ConfigurationProfile(BaseModel):
    """Complete configuration profile"""
    profile_name: str = Field(description="Profile name")
    description: str = Field(description="Profile description")
    created_at: float = Field(description="Creation timestamp")
    updated_at: float = Field(description="Last update timestamp")
    
    robot_presets: Dict[str, RobotPreset] = Field(description="Robot presets")
    system_config: SystemConfiguration = Field(description="System configuration")
    user_preferences: UserPreferences = Field(description="User preferences")
    
    is_active: bool = Field(default=False, description="Whether this profile is currently active")
    is_default: bool = Field(default=False, description="Whether this is the default profile")

# Default presets
DEFAULT_PRESETS = {
    "safe": RobotPreset(
        name="Safe Mode",
        type=PresetType.SAFE,
        description="Conservative settings prioritizing safety and stability",
        max_velocity=25.0,
        max_acceleration=15.0,
        movement_smoothing=0.8,
        collision_sensitivity=0.9,
        workspace_limits={
            "x_min": -0.3, "x_max": 0.3,
            "y_min": -0.3, "y_max": 0.3,
            "z_min": 0.1, "z_max": 0.8
        },
        emergency_stop_delay=0.1,
        camera_fps=15,
        camera_quality="medium",
        enable_recording=True,
        predictive_assistance=False,
        auto_grip_adjustment=True,
        motion_learning=False,
        enable_performance_monitoring=True,
        telemetry_level="standard"
    ),
    "normal": RobotPreset(
        name="Normal Operation",
        type=PresetType.NORMAL,
        description="Balanced settings for general teleoperation tasks",
        max_velocity=50.0,
        max_acceleration=25.0,
        movement_smoothing=0.5,
        collision_sensitivity=0.7,
        workspace_limits={
            "x_min": -0.5, "x_max": 0.5,
            "y_min": -0.5, "y_max": 0.5,
            "z_min": 0.0, "z_max": 1.0
        },
        emergency_stop_delay=0.2,
        camera_fps=30,
        camera_quality="high",
        enable_recording=True,
        predictive_assistance=True,
        auto_grip_adjustment=True,
        motion_learning=True,
        enable_performance_monitoring=True,
        telemetry_level="detailed"
    ),
    "performance": RobotPreset(
        name="Performance Mode",
        type=PresetType.PERFORMANCE,
        description="High-performance settings for experienced operators",
        max_velocity=80.0,
        max_acceleration=50.0,
        movement_smoothing=0.2,
        collision_sensitivity=0.5,
        workspace_limits={
            "x_min": -0.7, "x_max": 0.7,
            "y_min": -0.7, "y_max": 0.7,
            "z_min": -0.1, "z_max": 1.2
        },
        emergency_stop_delay=0.05,
        camera_fps=60,
        camera_quality="ultra",
        enable_recording=True,
        predictive_assistance=True,
        auto_grip_adjustment=False,
        motion_learning=True,
        enable_performance_monitoring=True,
        telemetry_level="full"
    )
}

# Global configuration state
config_state = {
    "active_profile": None,
    "profiles": {},
    "current_preset": "normal",
    "config_directory": "./config",
    "backup_directory": "./config/backups",
    "last_backup": None
}

@router.post("/profiles", response_model=ApiResponse)
async def create_profile(profile: ConfigurationProfile):
    """
    Create a new configuration profile
    
    Features:
    - Complete profile creation
    - Validation and conflict checking
    - Automatic backup
    """
    try:
        logger.info(f"Creating configuration profile: {profile.profile_name}")
        
        # Check if profile already exists
        if profile.profile_name in config_state["profiles"]:
            raise HTTPException(
                status_code=400,
                detail=f"Profile '{profile.profile_name}' already exists"
            )
        
        # Set timestamps
        current_time = time.time()
        profile.created_at = current_time
        profile.updated_at = current_time
        
        # Validate configuration
        validation_result = _validate_profile(profile)
        if not validation_result["valid"]:
            raise HTTPException(
                status_code=400,
                detail=f"Profile validation failed: {validation_result['errors']}"
            )
        
        # Store profile
        config_state["profiles"][profile.profile_name] = profile
        
        # Save to disk
        await _save_profile(profile)
        
        logger.info(f"Configuration profile created: {profile.profile_name}")
        
        return ApiResponse(
            status="success",
            message="Configuration profile created",
            data={
                "profile_name": profile.profile_name,
                "created_at": profile.created_at,
                "validation": validation_result
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to create profile: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create profile: {str(e)}"
        )

@router.get("/presets", response_model=ApiResponse)
async def get_robot_presets():
    """
    Get available robot presets
    
    Returns all available presets including default and custom ones
    """
    try:
        # Get active profile presets or defaults
        if config_state["active_profile"]:
            presets = config_state["active_profile"].robot_presets
        else:
            presets = {name: preset.dict() for name, preset in DEFAULT_PRESETS.items()}
        
        # Add metadata
        presets_with_meta = {}
        for name, preset in presets.items():
            if isinstance(preset, dict):
                presets_with_meta[name] = preset
            else:
                presets_with_meta[name] = preset.dict()
            
            presets_with_meta[name]["is_current"] = (name == config_state["current_preset"])
        
        return ApiResponse(
            status="success",
            message="Robot presets retrieved",
            data={
                "presets": presets_with_meta,
                "current_preset": config_state["current_preset"],
                "total_presets": len(presets_with_meta)
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get robot presets: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get robot presets: {str(e)}"
        )

@router.post("/presets/{preset_name}/activate", response_model=ApiResponse)
async def activate_preset(preset_name: str):
    """
    Activate a robot preset
    
    Features:
    - Preset validation
    - Safety checks
    - Configuration application
    - Status updates
    """
    try:
        logger.info(f"Activating robot preset: {preset_name}")
        
        # Get available presets
        if config_state["active_profile"]:
            available_presets = config_state["active_profile"].robot_presets
        else:
            available_presets = DEFAULT_PRESETS
        
        if preset_name not in available_presets:
            raise HTTPException(
                status_code=404,
                detail=f"Preset '{preset_name}' not found"
            )
        
        preset = available_presets[preset_name]
        
        # Validate preset before activation
        validation_result = _validate_preset(preset)
        if not validation_result["valid"]:
            raise HTTPException(
                status_code=400,
                detail=f"Preset validation failed: {validation_result['errors']}"
            )
        
        # Apply preset configuration
        await _apply_preset_configuration(preset)
        
        # Update current preset
        config_state["current_preset"] = preset_name
        
        logger.info(f"Robot preset activated: {preset_name}")
        
        return ApiResponse(
            status="success",
            message=f"Robot preset '{preset_name}' activated",
            data={
                "preset_name": preset_name,
                "preset_config": preset.dict() if hasattr(preset, 'dict') else preset,
                "validation": validation_result,
                "applied_at": time.time()
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to activate preset: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to activate preset: {str(e)}"
        )

@router.post("/presets", response_model=ApiResponse)
async def create_custom_preset(preset: RobotPreset):
    """
    Create a custom robot preset
    
    Features:
    - Custom preset creation
    - Parameter validation
    - Safety boundary checking
    - Automatic saving
    """
    try:
        logger.info(f"Creating custom preset: {preset.name}")
        
        # Set preset type to custom
        preset.type = PresetType.CUSTOM
        
        # Validate preset
        validation_result = _validate_preset(preset)
        if not validation_result["valid"]:
            raise HTTPException(
                status_code=400,
                detail=f"Preset validation failed: {validation_result['errors']}"
            )
        
        # Add to active profile or create temporary storage
        if config_state["active_profile"]:
            config_state["active_profile"].robot_presets[preset.name] = preset
            await _save_profile(config_state["active_profile"])
        else:
            # Create temporary custom presets storage
            if "custom_presets" not in config_state:
                config_state["custom_presets"] = {}
            config_state["custom_presets"][preset.name] = preset
        
        logger.info(f"Custom preset created: {preset.name}")
        
        return ApiResponse(
            status="success",
            message="Custom preset created",
            data={
                "preset_name": preset.name,
                "preset_config": preset.dict(),
                "validation": validation_result
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to create custom preset: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create custom preset: {str(e)}"
        )

@router.get("/profiles", response_model=ApiResponse)
async def get_configuration_profiles():
    """Get list of all configuration profiles"""
    try:
        profiles_summary = []
        
        for profile_name, profile in config_state["profiles"].items():
            summary = {
                "profile_name": profile_name,
                "description": profile.description,
                "created_at": profile.created_at,
                "updated_at": profile.updated_at,
                "is_active": profile.is_active,
                "is_default": profile.is_default,
                "preset_count": len(profile.robot_presets),
                "last_used": getattr(profile, 'last_used', None)
            }
            profiles_summary.append(summary)
        
        # Sort by last updated
        profiles_summary.sort(key=lambda x: x["updated_at"], reverse=True)
        
        return ApiResponse(
            status="success",
            message="Configuration profiles retrieved",
            data={
                "profiles": profiles_summary,
                "total_profiles": len(profiles_summary),
                "active_profile": config_state["active_profile"].profile_name if config_state["active_profile"] else None
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get configuration profiles: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get configuration profiles: {str(e)}"
        )

@router.post("/backup", response_model=ApiResponse)
async def create_configuration_backup():
    """Create backup of current configuration"""
    try:
        logger.info("Creating configuration backup")
        
        backup_data = {
            "timestamp": time.time(),
            "active_profile": config_state["active_profile"].dict() if config_state["active_profile"] else None,
            "profiles": {name: profile.dict() for name, profile in config_state["profiles"].items()},
            "current_preset": config_state["current_preset"],
            "custom_presets": getattr(config_state, "custom_presets", {})
        }
        
        backup_path = await _create_backup(backup_data)
        config_state["last_backup"] = time.time()
        
        logger.info(f"Configuration backup created: {backup_path}")
        
        return ApiResponse(
            status="success",
            message="Configuration backup created",
            data={
                "backup_path": backup_path,
                "backup_timestamp": backup_data["timestamp"],
                "backup_size_kb": _get_file_size_kb(backup_path) if backup_path else 0
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to create configuration backup: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create configuration backup: {str(e)}"
        )

@router.get("/status", response_model=ApiResponse)
async def get_configuration_status():
    """Get current configuration status"""
    try:
        status_data = {
            "active_profile": config_state["active_profile"].profile_name if config_state["active_profile"] else None,
            "current_preset": config_state["current_preset"],
            "total_profiles": len(config_state["profiles"]),
            "custom_presets_count": len(getattr(config_state, "custom_presets", {})),
            "last_backup": config_state["last_backup"],
            "config_directory": config_state["config_directory"],
            "default_presets_available": list(DEFAULT_PRESETS.keys())
        }
        
        return ApiResponse(
            status="success",
            message="Configuration status retrieved",
            data=status_data
        )
        
    except Exception as e:
        logger.error(f"Failed to get configuration status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get configuration status: {str(e)}"
        )

# Helper functions
def _validate_profile(profile: ConfigurationProfile) -> Dict[str, Any]:
    """Validate configuration profile"""
    errors = []
    warnings = []
    
    # Validate robot presets
    for preset_name, preset in profile.robot_presets.items():
        preset_validation = _validate_preset(preset)
        if not preset_validation["valid"]:
            errors.extend([f"Preset {preset_name}: {error}" for error in preset_validation["errors"]])
    
    # Validate system configuration
    if profile.system_config.max_storage_gb < 1.0:
        warnings.append("Storage limit is very low, may cause issues")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }

def _validate_preset(preset: Union[RobotPreset, Dict]) -> Dict[str, Any]:
    """Validate robot preset configuration"""
    errors = []
    warnings = []
    
    if isinstance(preset, dict):
        # Convert dict to object for validation
        try:
            preset_obj = RobotPreset(**preset)
        except Exception as e:
            return {"valid": False, "errors": [f"Invalid preset structure: {str(e)}"]}
        preset = preset_obj
    
    # Validate velocity and acceleration
    if preset.max_velocity > 70 and preset.max_acceleration > 40:
        warnings.append("High velocity and acceleration combination may be unsafe")
    
    # Validate workspace limits
    workspace = preset.workspace_limits
    if (workspace["x_max"] - workspace["x_min"]) < 0.1:
        errors.append("X workspace range too small")
    if (workspace["y_max"] - workspace["y_min"]) < 0.1:
        errors.append("Y workspace range too small")
    if (workspace["z_max"] - workspace["z_min"]) < 0.1:
        errors.append("Z workspace range too small")
    
    # Validate camera settings
    if preset.camera_fps > 30 and preset.camera_quality == "ultra":
        warnings.append("High FPS with ultra quality may impact performance")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }

async def _apply_preset_configuration(preset: Union[RobotPreset, Dict]):
    """Apply preset configuration to system"""
    logger.info("Applying preset configuration")
    # Mock implementation - would configure actual robot systems
    pass

async def _save_profile(profile: ConfigurationProfile) -> str:
    """Save configuration profile to disk"""
    config_dir = Path(config_state["config_directory"])
    config_dir.mkdir(parents=True, exist_ok=True)
    
    profile_path = config_dir / f"{profile.profile_name}.json"
    
    with open(profile_path, 'w') as f:
        json.dump(profile.dict(), f, indent=2)
    
    return str(profile_path)

async def _create_backup(backup_data: Dict) -> str:
    """Create configuration backup file"""
    backup_dir = Path(config_state["backup_directory"])
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = int(time.time())
    backup_path = backup_dir / f"config_backup_{timestamp}.json"
    
    with open(backup_path, 'w') as f:
        json.dump(backup_data, f, indent=2)
    
    return str(backup_path)

def _get_file_size_kb(file_path: str) -> float:
    """Get file size in KB"""
    try:
        size_bytes = Path(file_path).stat().st_size
        return size_bytes / 1024
    except:
        return 0.0


# ============================================================================
# Demo Mode Configuration
# ============================================================================

@router.get("/demo-config/{operation_mode}", response_model=ApiResponse)
async def get_demo_config(operation_mode: str):
    """
    Get pre-configured demo settings for a given operation mode.
    
    When connecting with a demo-enabled experiment (e.g., aloha_bimanual_lemgo_v2_demo),
    this endpoint returns all the pre-configured settings needed for policy evaluation
    without requiring user input.
    
    Returns:
    - policy_path: Path to the pre-trained policy
    - task_description: Description of the demo task
    - fps, episode_time_s, reset_time_s, num_episodes: Timing settings
    - interactive: Whether interventions are enabled
    """
    try:
        demo_config = ExperimentConfigMapper.get_demo_config_for_mode(operation_mode)
        
        if demo_config is None:
            return ApiResponse(
                status="error",
                message=f"No demo configuration found for operation mode: {operation_mode}",
                data={"available_modes": ["bimanual", "left", "right"]}
            )
        
        return ApiResponse(
            status="success",
            message="Demo configuration retrieved",
            data=demo_config
        )
        
    except Exception as e:
        logger.error(f"Failed to get demo config: {e}")
        return ApiResponse(
            status="error",
            message=f"Failed to get demo configuration: {str(e)}",
            data=None
        )


@router.get("/demo-available", response_model=ApiResponse)
async def get_available_demos():
    """
    Get list of all available demo configurations.
    
    Returns a list of operation modes that have demo configurations available.
    """
    try:
        available_demos = {}
        for mode in ["bimanual", "left", "right"]:
            config = ExperimentConfigMapper.get_demo_config_for_mode(mode)
            if config:
                available_demos[mode] = config
        
        return ApiResponse(
            status="success",
            message=f"Found {len(available_demos)} demo configuration(s)",
            data={
                "demos": available_demos,
                "count": len(available_demos)
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get available demos: {e}")
        return ApiResponse(
            status="error",
            message=f"Failed to get available demos: {str(e)}",
            data=None
        )


# ============================================================================
# Initialization
# ============================================================================

# Initialize default configuration on module load
def initialize_default_config():
    """Initialize default configuration"""
    try:
        # Create default profile
        default_profile = ConfigurationProfile(
            profile_name="Default",
            description="Default system configuration",
            created_at=time.time(),
            updated_at=time.time(),
            robot_presets={name: preset for name, preset in DEFAULT_PRESETS.items()},
            system_config=SystemConfiguration(),
            user_preferences=UserPreferences(),
            is_active=True,
            is_default=True
        )
        
        config_state["profiles"]["Default"] = default_profile
        config_state["active_profile"] = default_profile
        
        logger.info("Default configuration initialized")
        
    except Exception as e:
        logger.error(f"Failed to initialize default configuration: {e}")

# Initialize on import
initialize_default_config()
