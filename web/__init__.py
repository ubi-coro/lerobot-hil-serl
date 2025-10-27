"""
LeRobot Web GUI Package
=======================

Modern web interface for LeRobot Framework providing:
- Vue.js 3 frontend with Bootstrap UI
- FastAPI backend with async support  
- Real-time WebSocket communication
- Simplified robot teleoperation
- Enhanced safety features
- Development tools and CLI commands

Usage:
    # Install with web dependencies
    pip install -e .[web]
    
    # Start GUI
    lerobot-gui
    
    # Development mode
    lerobot-gui-dev
    
    # Create desktop shortcut
    lerobot-gui-shortcut

For more information, see README.md in this directory.
"""

__version__ = "1.0.0"
__author__ = "LeRobot Web GUI Team"
__description__ = "Modern web interface for LeRobot Framework"

# Make CLI commands available for direct import
try:
    from .cli import cli
except ImportError:
    # CLI dependencies not available
    cli = None

__all__ = ["cli", "__version__", "__author__", "__description__"]
