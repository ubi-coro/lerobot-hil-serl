"""
Deprecated Teleoperation Module
===============================

This module previously implemented an advanced teleoperation layer with
custom preset configurations (Safe / Normal / Performance). It has been
deprecated in favor of the streamlined ALOHA teleoperation module
(`aloha_teleoperation.py`) which directly leverages LeRobot's core
teleoperation primitives.

Reason for deprecation:
- Removed duplication of control logic already provided by LeRobot
- Simplified configuration surface (single config dict)
- Eliminated preset abstraction layer not present upstream
- Reduced maintenance overhead

If legacy behavior is ever needed again, recover it from git history.
This stub remains only as documentation; it intentionally does not
register any FastAPI router.
"""

# Intentionally left without active code.
