"""FastAPI services package.

Exports hardware-capable RobotService implementation.
"""

from .robot_service_fastapi import RobotService  # noqa: F401

__all__ = ["RobotService"]
