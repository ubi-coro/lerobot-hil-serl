"""
ALOHA Hardware Configuration Module
==================================

Hardware-level configuration management for ALOHA robots, inspired by LeLab's approach
but adapted for ALOHA-specific needs. This complements the high-level configuration.py
module by handling hardware calibration, device detection, and low-level settings.

This module bridges the gap between:
- LeLab's hardware-focused config management
- Your high-level preset and user configuration system
- ALOHA-specific hardware requirements
"""

import os
import shutil
import logging
import platform
import time
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# ALOHA-specific calibration paths (following LeRobot conventions)
ALOHA_CALIBRATION_BASE = os.path.expanduser("~/.cache/lerobot/calibration")
ALOHA_CONFIG_BASE = os.path.expanduser("~/.cache/lerobot/aloha_config")

# ALOHA calibration directories
ALOHA_LEADER_LEFT_PATH = os.path.join(ALOHA_CALIBRATION_BASE, "aloha_leader_left")
ALOHA_LEADER_RIGHT_PATH = os.path.join(ALOHA_CALIBRATION_BASE, "aloha_leader_right")
ALOHA_FOLLOWER_LEFT_PATH = os.path.join(ALOHA_CALIBRATION_BASE, "aloha_follower_left")
ALOHA_FOLLOWER_RIGHT_PATH = os.path.join(ALOHA_CALIBRATION_BASE, "aloha_follower_right")

# ALOHA hardware configuration files
ALOHA_HARDWARE_CONFIG = os.path.join(ALOHA_CONFIG_BASE, "hardware_config.json")
ALOHA_DEVICE_MAPPING = os.path.join(ALOHA_CONFIG_BASE, "device_mapping.json")
ALOHA_CAMERA_CONFIG = os.path.join(ALOHA_CONFIG_BASE, "camera_config.json")

def setup_aloha_calibration_directory(calibration_dir: str) -> bool:
    """
    Setup ALOHA calibration directory structure.
    Similar to LeLab's approach but for ALOHA hardware.
    
    Args:
        calibration_dir: Path to the ALOHA calibration directory
        
    Returns:
        bool: True if setup successful
    """
    try:
        logger.info(f"Setting up ALOHA calibration directory: {calibration_dir}")
        
        # Create base calibration directory
        Path(calibration_dir).mkdir(parents=True, exist_ok=True)
        
        # Create arm-specific subdirectories
        arm_dirs = [
            "leader_left", "leader_right",
            "follower_left", "follower_right"
        ]
        
        for arm_dir in arm_dirs:
            arm_path = Path(calibration_dir) / arm_dir
            arm_path.mkdir(exist_ok=True)
            logger.info(f"Created calibration directory: {arm_path}")
        
        # Create camera calibration directory
        camera_path = Path(calibration_dir) / "cameras"
        camera_path.mkdir(exist_ok=True)
        
        logger.info("ALOHA calibration directory setup complete")
        return True
        
    except Exception as e:
        logger.error(f"Failed to setup ALOHA calibration directory: {e}")
        return False

def validate_aloha_hardware_config() -> Dict[str, Any]:
    """
    Simple validation - just return the config as valid.
    Real validation happens when hardware is actually connected.
    ALOHA systems use predefined configuration, so no complex detection needed.
    """
    try:
        logger.info("Validating ALOHA hardware configuration...")
        
        config = get_aloha_config()
        
        validation_result = {
            "valid": True,
            "configured_devices": config,
            "status": "configured",
            "message": "Using predefined ALOHA configuration - no hardware detection needed",
            "validation_time": time.time(),
            "warnings": ["Hardware validation skipped - using predefined ALOHA config"]
        }
        
        logger.info("✅ ALOHA hardware validation complete - using predefined configuration")
        return validation_result
        
    except Exception as e:
        logger.error(f"Error validating ALOHA hardware: {e}")
        return {"status": "error", "error": str(e), "valid": False}

def save_aloha_hardware_config(config: Dict[str, Any]) -> bool:
    """
    Save ALOHA hardware configuration.
    Similar to LeLab's config saving but structured for ALOHA.
    
    Args:
        config: Hardware configuration dictionary
        
    Returns:
        bool: True if saved successfully
    """
    try:
        # Create config directory
        os.makedirs(ALOHA_CONFIG_BASE, exist_ok=True)
        
        # Add metadata
        config["saved_at"] = time.time()
        config["version"] = "1.0"
        
        # Save configuration
        with open(ALOHA_HARDWARE_CONFIG, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"ALOHA hardware configuration saved to: {ALOHA_HARDWARE_CONFIG}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving ALOHA hardware config: {e}")
        return False

def load_aloha_hardware_config() -> Optional[Dict[str, Any]]:
    """
    Load ALOHA hardware configuration.
    
    Returns:
        Dict with hardware config or None if not found
    """
    try:
        if os.path.exists(ALOHA_HARDWARE_CONFIG):
            with open(ALOHA_HARDWARE_CONFIG, 'r') as f:
                config = json.load(f)
            logger.info("ALOHA hardware configuration loaded")
            return config
        else:
            logger.info("No ALOHA hardware configuration found")
            return None
            
    except Exception as e:
        logger.error(f"Error loading ALOHA hardware config: {e}")
        return None

def create_aloha_device_mapping_from_config() -> Dict[str, Any]:
    """
    Create simple device mapping from predefined ALOHA configuration.
    No complex detection - just use the standard config.
    """
    try:
        logger.info("Creating ALOHA device mapping from simple configuration...")
        
        config = get_aloha_config()
        
        mapping = {
            "arms": {
                "leader_left": {"port": config["leader_left"], "status": "configured"},
                "leader_right": {"port": config["leader_right"], "status": "configured"},
                "follower_left": {"port": config["follower_left"], "status": "configured"},
                "follower_right": {"port": config["follower_right"], "status": "configured"}
            },
            "cameras": {
                "cam_high": {"serial_number": config["cameras"][0], "status": "configured"},
                "cam_low": {"serial_number": config["cameras"][1], "status": "configured"},
                "cam_left_wrist": {"serial_number": config["cameras"][2], "status": "configured"},
                "cam_right_wrist": {"serial_number": config["cameras"][3], "status": "configured"}
            },
            "mapping_source": "simple_aloha_config",
            "created_at": time.time()
        }
        
        logger.info("ALOHA device mapping created from simple configuration")
        return mapping
        
    except Exception as e:
        logger.error(f"Error creating ALOHA device mapping: {e}")
        return {"error": str(e)}

def save_aloha_device_mapping(mapping: Dict[str, Any]) -> bool:
    """Save ALOHA device mapping to file"""
    try:
        os.makedirs(ALOHA_CONFIG_BASE, exist_ok=True)
        
        with open(ALOHA_DEVICE_MAPPING, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        logger.info(f"ALOHA device mapping saved to: {ALOHA_DEVICE_MAPPING}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving ALOHA device mapping: {e}")
        return False

def load_aloha_device_mapping() -> Optional[Dict[str, Any]]:
    """Load ALOHA device mapping from file"""
    try:
        if os.path.exists(ALOHA_DEVICE_MAPPING):
            with open(ALOHA_DEVICE_MAPPING, 'r') as f:
                mapping = json.load(f)
            logger.info("ALOHA device mapping loaded")
            return mapping
        else:
            logger.info("No ALOHA device mapping found")
            return None
            
    except Exception as e:
        logger.error(f"Error loading ALOHA device mapping: {e}")
        return None

def validate_aloha_configuration(config_dir: str) -> Dict[str, Any]:
    """
    Validate ALOHA configuration directory and files.
    Similar to LeLab's validation but for ALOHA structure.
    
    Args:
        config_dir: ALOHA configuration directory path
        
    Returns:
        Validation results
    """
    try:
        logger.info(f"Validating ALOHA configuration: {config_dir}")
        
        validation = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "checked_at": time.time()
        }
        
        # Check if calibration directory exists
        if not os.path.exists(config_dir):
            validation["errors"].append(f"Calibration directory not found: {config_dir}")
            validation["valid"] = False
            return validation
        
        # Check for required arm calibration files
        required_arms = ["leader_left", "leader_right", "follower_left", "follower_right"]
        for arm in required_arms:
            arm_dir = os.path.join(config_dir, arm)
            if not os.path.exists(arm_dir):
                validation["warnings"].append(f"Missing calibration directory for {arm}")
            else:
                # Look for calibration files in the arm directory
                arm_files = list(Path(arm_dir).glob("*.json"))
                if not arm_files:
                    validation["warnings"].append(f"No calibration files found for {arm}")
        
        # Check hardware configuration
        hardware_config = load_aloha_hardware_config()
        if not hardware_config:
            validation["warnings"].append("No hardware configuration found")
        
        # Check device mapping
        device_mapping = load_aloha_device_mapping()
        if not device_mapping:
            validation["warnings"].append("No device mapping found")
        elif device_mapping:
            # Validate device mapping
            unmapped_arms = [arm for arm, config in device_mapping.get("arms", {}).items() 
                           if config.get("status") == "unmapped"]
            if unmapped_arms:
                validation["warnings"].append(f"Unmapped arms: {', '.join(unmapped_arms)}")
            
            unmapped_cameras = [cam for cam, config in device_mapping.get("cameras", {}).items() 
                               if config.get("status") == "unmapped"]
            if unmapped_cameras:
                validation["warnings"].append(f"Unmapped cameras: {', '.join(unmapped_cameras)}")
        
        logger.info(f"ALOHA configuration validation complete. Valid: {validation['valid']}")
        return validation
        
    except Exception as e:
        logger.error(f"Error validating ALOHA configuration: {e}")
        return {
            "valid": False,
            "errors": [str(e)],
            "warnings": [],
            "checked_at": time.time()
        }

def get_aloha_config() -> Dict[str, Any]:
    """Simple hardware config - no complex detection needed"""
    return {
        "leader_left": "/dev/ttyDXL_master_left",
        "leader_right": "/dev/ttyDXL_master_right", 
        "follower_left": "/dev/ttyDXL_puppet_left",
        "follower_right": "/dev/ttyDXL_puppet_right",
        "cameras": [218722270994, 130322272007, 218622276088, 130322274116],
        "robot_type": "aloha",
        "mock": False
    }

def get_aloha_default_config() -> Dict[str, Any]:
    """
    Get default ALOHA hardware configuration - simplified approach.
    Based on standard ALOHA setup with minimal complexity.
    """
    base_config = get_aloha_config()
    
    # Expand the simple config for backward compatibility
    return {
        "robot_type": base_config["robot_type"],
        "arm_count": 4,
        "camera_count": 4,
        "default_ports": {
            "leader_left": base_config["leader_left"],
            "leader_right": base_config["leader_right"],
            "follower_left": base_config["follower_left"],
            "follower_right": base_config["follower_right"]
        },
        "default_camera_serials": {
            "cam_high": str(base_config["cameras"][0]),
            "cam_low": str(base_config["cameras"][1]), 
            "cam_left_wrist": str(base_config["cameras"][2]),
            "cam_right_wrist": str(base_config["cameras"][3])
        },
        "motor_config": {
            "baud_rate": 1000000,
            "protocol_version": 2.0,
            "timeout_ms": 1000
        },
        "camera_config": {
            "fps": 30,
            "width": 640,
            "height": 480,
            "depth_enabled": True
        },
        "mock": base_config["mock"]
    }

# Convenience function to setup complete ALOHA hardware configuration
def setup_aloha_hardware() -> Dict[str, Any]:
    """
    Simple hardware setup - just return the predefined config.
    No complex detection workflow needed for ALOHA.
    """
    try:
        logger.info("Starting simple ALOHA hardware setup...")
        
        config = get_aloha_config()
        device_mapping = create_aloha_device_mapping_from_config()
        
        setup_result = {
            "success": True,
            "configuration": config,
            "device_mapping": device_mapping,
            "message": "ALOHA hardware configured with standard setup",
            "steps_completed": ["simple_config", "device_mapping"],
            "setup_time": time.time()
        }
        
        # Optional: Save the configuration
        try:
            save_aloha_device_mapping(device_mapping)
            setup_result["steps_completed"].append("config_saved")
        except Exception as e:
            logger.warning(f"Could not save device mapping: {e}")
            setup_result["warnings"] = [f"Config save failed: {e}"]
        
        logger.info("✅ Simple ALOHA hardware setup complete")
        return setup_result
        
    except Exception as e:
        logger.error(f"Error in simple ALOHA hardware setup: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Simple ALOHA setup failed"
        }
