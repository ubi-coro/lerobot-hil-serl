"""
LeRobot FastAPI Backend Modules
===============================

This package contains modular FastAPI routers organized by functionality,
inspired by LeLab's clean architecture but enhanced with LeRobot's advanced features.

Modules:
- aloha_teleoperation.py - ALOHA teleoperation (current)
- robot.py - Robot connection and hardware management
- safety.py - Emergency stop and safety systems
- monitoring.py - Performance monitoring and health checks
- recording.py - Dataset recording (inspired by LeLab)
- configuration.py - Configuration management

Deprecated:
- teleoperation_deprecated.py (was: preset-based advanced teleoperation)
"""

__all__ = [
    "aloha_teleoperation",
    "robot",
    "safety",
    "monitoring",
    "recording",
    "configuration"
]
