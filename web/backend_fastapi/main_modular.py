"""
LeRobot FastAPI Backend - Modular Architecture
==============================================

Modern modular FastAPI backend for LeRobot Web Interface with organized
modules for different functionalities.

Features:
- Modular architecture with separated concerns
- Async FastAPI application with uvicorn server
- Socket.IO integration for real-time communication
- (Legacy service bridge removed)
- Interactive API documentation
- Comprehensive robot management modules

Modules:
- robot.py: Robot connection and hardware management
- aloha_teleoperation.py: ALOHA teleoperation (LeRobot native)
- safety.py: Enhanced emergency stop and safety systems
- monitoring.py: Performance tracking and analytics
- recording.py: Dataset management and episode recording
- configuration.py: Advanced preset and settings management
"""

import asyncio
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import socketio
from pydantic import BaseModel
from typing import Optional, Dict, Any
import sys
import os
from pathlib import Path

# Add backend_fastapi to Python path for imports
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

service_bridge = None  # legacy bridge removed

# Import shared module for Socket.IO access
import shared

# Create modular FastAPI application
app = FastAPI(
    title="LeRobot Web Interface API - Modular",
    description="""
    Modern modular FastAPI backend for LeRobot teleoperation and control.
    
    ## Modules
    
    - **Robot**: Connection and hardware management
    - **Teleoperation**: Advanced control with presets (Safe/Normal/Performance)
    - **Safety**: Enhanced emergency stop and safety systems
    - **Monitoring**: Performance tracking and analytics
    - **Recording**: Dataset management and episode recording
    - **Configuration**: Advanced preset and settings management
    - **Dataset**: Directory browsing and dataset visualization
    
    ## Features
    
    - Real-time Socket.IO communication
    - Interactive API documentation
    - Modular architecture for maintainability
    - (Legacy service bridge removed)
    """,
    version="3.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5000",
        "http://127.0.0.1:5000",
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Socket.IO server with CORS support
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5000",
        "http://127.0.0.1:5000",
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ],
    logger=True,
    engineio_logger=False  # Reduce log noise
)

# Set shared Socket.IO instance for modules that need it
shared.set_socketio(sio)

# Import module routers AFTER Socket.IO is set up
try:
    from modules.robot import router as robot_router
    from modules.aloha_teleoperation import router as aloha_teleoperation_router
    from modules.aloha_teleoperation import get_teleoperation_status_snapshot
    from modules.safety import router as safety_router
    from modules.monitoring import router as monitoring_router
    from modules.recording import router as recording_router
    from modules.configuration import router as configuration_router
    from modules.dataset import router as dataset_router
    # Import recording worker for Socket.IO handlers
    from modules.recording_worker import register_socketio_handlers, init_recording_worker
    logger.info("✅ All modules imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import modules: {e}")
    raise

# Include module routers with their prefixes
app.include_router(robot_router)
app.include_router(aloha_teleoperation_router)
app.include_router(safety_router)
app.include_router(monitoring_router)
app.include_router(recording_router)
app.include_router(configuration_router)
app.include_router(dataset_router, prefix="/api/dataset")

# Register recording worker Socket.IO handlers
register_socketio_handlers(sio)

# Create Socket.IO ASGI app
socket_app = socketio.ASGIApp(sio, app)

# Pydantic models for main app
class ApiResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None

# Global state for Socket.IO clients
connected_clients = set()

# Background task: broadcast teleoperation status periodically
async def _teleop_status_broadcaster():
    while True:
        try:
            sio_instance = shared.get_socketio()
            if sio_instance:
                payload = get_teleoperation_status_snapshot()
                await sio_instance.emit('teleoperation_status', payload)
        except Exception:
            logger.debug('teleoperation_status periodic emit failed', exc_info=True)
        await asyncio.sleep(2.0)

# Socket.IO event handlers
@sio.event
async def connect(sid, environ):
    """Handle Socket.IO client connection"""
    logger.info(f"🔌 Socket.IO client connected: {sid}")
    connected_clients.add(sid)
    
    # Send welcome message and initial status
    await sio.emit('connected', {
        'status': 'success',
        'message': 'Connected to LeRobot modular backend',
        'modules': ['robot', 'teleoperation', 'safety', 'monitoring', 'recording', 'configuration', 'dataset'],
        'api_docs': '/api/docs'
    }, room=sid)
    # Also push an immediate teleoperation status to new client
    try:
        await sio.emit('teleoperation_status', get_teleoperation_status_snapshot(), room=sid)
    except Exception:
        logger.debug('initial teleop status emit failed', exc_info=True)
    
    # Initial robot status intentionally not sent (legacy bridge removed)

@sio.event
async def disconnect(sid):
    """Handle Socket.IO client disconnection"""
    logger.info(f"🔌 Socket.IO client disconnected: {sid}")
    connected_clients.discard(sid)

@sio.event
async def ping(sid, data):
    """Handle ping from client"""
    await sio.emit('pong', {'timestamp': asyncio.get_event_loop().time()}, room=sid)

@sio.event
async def robot_command(sid, data):
    """Handle robot commands via Socket.IO (legacy compatibility)"""
    try:
        command = data.get('command')
        logger.info(f"📡 Received legacy robot command: {command}")
        
        if command == 'get_status':
            await sio.emit('robot_status', {
                'status': 'disconnected',
                'message': 'No hardware status available yet'
            }, room=sid)
        
        elif command == 'emergency_stop':
            # Forward to safety module
            logger.critical("Emergency stop triggered via Socket.IO")
            await sio.emit('emergency_stop_acknowledged', {
                'status': 'success',
                'message': 'Emergency stop executed'
            }, room=sid)
        
        else:
            logger.warning(f"Unknown legacy command: {command}")
            await sio.emit('error', {
                'message': f'Unknown command: {command}. Use the modular API endpoints instead.'
            }, room=sid)
        
    except Exception as e:
        logger.error(f"Error handling robot command: {e}")
        await sio.emit('error', {'message': str(e)}, room=sid)

@sio.event
async def teleoperation_command(sid, data):
    """Handle teleoperation commands via Socket.IO"""
    try:
        command = data.get('command')
        preset = data.get('preset', 'normal')
        
        logger.info(f"🎮 Received teleoperation command: {command} (preset: {preset})")
        
        # Emit acknowledgment
        await sio.emit('teleoperation_response', {
            'command': command,
            'preset': preset,
            'status': 'acknowledged',
            'timestamp': asyncio.get_event_loop().time()
        }, room=sid)
        
    except Exception as e:
        logger.error(f"Error handling teleoperation command: {e}")
        await sio.emit('error', {'message': str(e)}, room=sid)

# Main API endpoints
@app.get("/", response_model=ApiResponse)
async def root():
    """Root endpoint with API information"""
    return ApiResponse(
        status="success",
        message="LeRobot Modular FastAPI Backend",
        data={
            "version": "3.0.0",
            "modules": [
                {"name": "robot", "description": "Robot connection and hardware management"},
                {"name": "teleoperation", "description": "Advanced control with presets"},
                {"name": "safety", "description": "Emergency stop and safety systems"},
                {"name": "monitoring", "description": "Performance tracking and analytics"},
                {"name": "recording", "description": "Dataset management and recording"},
                {"name": "configuration", "description": "Preset and settings management"}
            ],
            "api_docs": "/api/docs",
            "websocket_endpoint": "/socket.io/",
            "connected_clients": len(connected_clients)
        }
    )

@app.get("/api/health", response_model=ApiResponse)
async def health_check():
    """Health check endpoint"""
    return ApiResponse(
        status="success",
        message="All systems operational",
        data={
            "modules_loaded": 7,
            "socket_clients": len(connected_clients),
            "service_bridge": False
        }
    )

@app.on_event("startup")
async def _start_background_tasks():
    # Start teleop status broadcaster
    asyncio.create_task(_teleop_status_broadcaster())

@app.on_event("startup")
async def _store_event_loop_for_threads():
    """Capture the running loop so background threads can schedule emits safely."""
    try:
        loop = asyncio.get_running_loop()
        shared.set_event_loop(loop)
    except Exception:
        pass

@app.get("/api/modules", response_model=ApiResponse)
async def get_modules():
    """Get information about loaded modules"""
    modules_info = [
        {
            "name": "robot",
            "prefix": "/api/robot",
            "description": "Robot connection and hardware management",
            "endpoints": ["connect", "disconnect", "status", "reboot"]
        },
        {
            "name": "teleoperation", 
            "prefix": "/api/teleoperation",
            "description": "Advanced teleoperation with presets",
            "endpoints": ["start", "stop", "presets", "status"]
        },
        {
            "name": "safety",
            "prefix": "/api/safety", 
            "description": "Enhanced emergency stop and safety systems",
            "endpoints": ["emergency-stop", "reset", "status", "limits"]
        },
        {
            "name": "monitoring",
            "prefix": "/api/monitoring",
            "description": "Performance tracking and analytics", 
            "endpoints": ["start", "stop", "metrics", "history", "alerts"]
        },
        {
            "name": "recording",
            "prefix": "/api/recording",
            "description": "Dataset management and episode recording",
            "endpoints": ["session/start", "episode/start", "episode/stop", "datasets"]
        },
        {
            "name": "configuration",
            "prefix": "/api/configuration",
            "description": "Advanced preset and settings management",
            "endpoints": ["presets", "profiles", "config", "backup"]
        },
        {
            "name": "dataset",
            "prefix": "/api/dataset",
            "description": "Directory browsing and dataset visualization",
            "endpoints": ["browse", "browse-directory", "visualize", "count"]
        }
    ]
    
    return ApiResponse(
        status="success",
        message="Modules information retrieved",
        data={
            "modules": modules_info,
            "total_modules": len(modules_info)
        }
    )

# Broadcast utility function for modules
async def broadcast_to_clients(event: str, data: Any):
    """Utility function for modules to broadcast to all connected clients"""
    if connected_clients:
        await sio.emit(event, data)
        logger.info(f"📡 Broadcasted {event} to {len(connected_clients)} clients")

# Make broadcast function available to modules
app.state.broadcast = broadcast_to_clients
app.state.sio = sio

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("🚀 Starting LeRobot Modular FastAPI Backend")
    logger.info("📋 Loaded modules: robot, teleoperation, safety, monitoring, recording, configuration")
    
    # legacy bridge removed
    
    logger.info("✅ Backend initialization complete")

@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup"""
    logger.info("🚀 Starting LeRobot Modular FastAPI Backend")
    
    # Initialize recording worker with event loop
    loop = asyncio.get_event_loop()
    init_recording_worker(loop)
    
    logger.info("✅ All services initialized")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    logger.info("🛑 Shutting down LeRobot Modular FastAPI Backend")
    
    # Notify connected clients
    if connected_clients:
        await sio.emit('server_shutdown', {
            'message': 'Server is shutting down'
        })
    
    logger.info("✅ Shutdown complete")

# Export the ASGI app for uvicorn
if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting development server...")
    uvicorn.run(
        "main:socket_app",  # Use socket_app for Socket.IO support
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
