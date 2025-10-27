"""
LeRobot FastAPI Backend
======================

Modern async backend for LeRobot Web GUI with:
- FastAPI for high-performance async operations
- Automatic API documentation
- Better WebSocket handling for camera streaming
- Improved error handling and validation

This replaces the Flask backend with a more modern and performant solution.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging
import asyncio
import json
import os
import sys

# Socket.IO support for compatibility with frontend
import socketio

# Add the parent directory to import LeRobot modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import existing services via bridge
try:
    from services import RobotService, StreamService
except ImportError as e:
    logging.warning(f"Could not import services: {e}")
    RobotService = None
    StreamService = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LeRobot Web API",
    description="Modern async backend for LeRobot teleoperation and control",
    version="2.0.0",
    docs_url="/api/docs",  # API documentation at /api/docs
    redoc_url="/api/redoc"  # Alternative docs at /api/redoc
)

# Initialize Socket.IO for compatibility with frontend
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5000",
        "http://127.0.0.1:5000"
    ]
)

# Combine FastAPI and Socket.IO
socket_app = socketio.ASGIApp(sio, app)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5000",
        "http://127.0.0.1:5000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services (will be properly typed once imported)
robot_service = None
stream_service = None
connected_websockets: List[WebSocket] = []

# Socket.IO event handlers
@sio.event
async def connect(sid, environ):
    """Handle Socket.IO client connection"""
    logger.info(f"🔌 Socket.IO client connected: {sid}")
    
    # Send initial status
    if robot_service:
        try:
            status = robot_service.get_status()
            await sio.emit('status_update', status, room=sid)
        except Exception as e:
            logger.error(f"Error sending initial status: {e}")

@sio.event
async def disconnect(sid):
    """Handle Socket.IO client disconnection"""
    logger.info(f"🔌 Socket.IO client disconnected: {sid}")

@sio.event
async def robot_command(sid, data):
    """Handle robot commands via Socket.IO"""
    try:
        command = data.get('command')
        logger.info(f"📡 Received robot command: {command}")
        
        if command == 'get_status' and robot_service:
            status = robot_service.get_status()
            await sio.emit('status_update', status, room=sid)
        
        elif command == 'ping':
            await sio.emit('pong', {'timestamp': asyncio.get_event_loop().time()}, room=sid)
        
    except Exception as e:
        logger.error(f"Error handling robot command: {e}")
        await sio.emit('error', {'message': str(e)}, room=sid)

# Pydantic models for request/response validation
class RobotConnectionRequest(BaseModel):
    overrides: Optional[List[str]] = []
    enable_cameras: bool = True

class TeleoperationStartRequest(BaseModel):
    fps: Optional[int] = 30
    show_cameras: bool = True
    max_relative_target: Optional[int] = 25
    operation_mode: str = "bimanual"

class RobotStatusResponse(BaseModel):
    status: str
    mode: Optional[str] = None
    is_connected: bool = False
    message: Optional[str] = None
    cameras_enabled: bool = False

class ApiResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[Any, Any]] = None

# Startup event to initialize services
@app.on_event("startup")
async def startup_event():
    """Initialize services when the app starts"""
    global robot_service, stream_service
    
    logger.info("🚀 Starting FastAPI LeRobot backend...")
    
    try:
        if RobotService and StreamService:
            # Initialize services with async support
            robot_service = RobotService(use_mock=False)
            stream_service = StreamService()
            logger.info("✅ Services initialized successfully")
        else:
            logger.warning("⚠️ Services not available, running in mock mode")
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup when the app shuts down"""
    logger.info("🛑 Shutting down FastAPI backend...")
    
    # Close all WebSocket connections
    for websocket in connected_websockets:
        try:
            await websocket.close()
        except:
            pass
    
    # Stop robot services
    if robot_service:
        try:
            robot_service.emergency_stop()
        except:
            pass

# Robot API endpoints
@app.get("/api/robot/status", response_model=RobotStatusResponse)
async def get_robot_status():
    """Get current robot status"""
    try:
        if robot_service:
            status = robot_service.get_status()
            return RobotStatusResponse(
                status="success",
                mode=status.get("mode"),
                is_connected=status.get("is_connected", False),
                cameras_enabled=status.get("cameras_enabled", False)
            )
        else:
            return RobotStatusResponse(
                status="mock",
                mode=None,
                is_connected=False,
                message="Running in mock mode"
            )
    except Exception as e:
        logger.error(f"Error getting robot status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/robot/connect", response_model=ApiResponse)
async def connect_robot(request: RobotConnectionRequest):
    """Connect to ALOHA robot"""
    try:
        logger.info(f"🔌 Connecting to robot with overrides: {request.overrides}")
        
        if robot_service:
            result = robot_service.connect_aloha(
                overrides=request.overrides,
                enable_cameras=request.enable_cameras
            )
            return ApiResponse(
                status="success",
                message="Robot connected successfully",
                data=result
            )
        else:
            # Mock mode
            await asyncio.sleep(1)  # Simulate connection time
            return ApiResponse(
                status="success",
                message="Robot connected (mock mode)",
                data={"mock": True, "cameras": request.enable_cameras}
            )
            
    except Exception as e:
        logger.error(f"❌ Error connecting to robot: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/robot/disconnect", response_model=ApiResponse)
async def disconnect_robot():
    """Disconnect from robot"""
    try:
        logger.info("🔌 Disconnecting from robot")
        
        if robot_service:
            robot_service.disconnect()
        
        return ApiResponse(
            status="success",
            message="Robot disconnected successfully"
        )
        
    except Exception as e:
        logger.error(f"❌ Error disconnecting robot: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/robot/teleoperate/start", response_model=ApiResponse)
async def start_teleoperation(request: TeleoperationStartRequest):
    """Start teleoperation with specified configuration"""
    try:
        logger.info(f"▶️ Starting teleoperation: {request.dict()}")
        
        if robot_service:
            robot_service.start_teleoperation(
                fps=request.fps,
                show_cameras=request.show_cameras
            )
        else:
            # Mock mode - just simulate
            await asyncio.sleep(0.5)
        
        # Immediately notify all Socket.IO clients
        await sio.emit('teleoperation_started', {
            "type": "teleoperation_started",
            "config": request.dict()
        })
        
        return ApiResponse(
            status="success",
            message="Teleoperation started successfully"
        )
        
    except Exception as e:
        logger.error(f"❌ Error starting teleoperation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/robot/teleoperate/stop", response_model=ApiResponse)
async def stop_teleoperation():
    """Stop teleoperation"""
    try:
        logger.info("⏹️ Stopping teleoperation")
        
        if robot_service:
            robot_service.stop_teleoperation()
        else:
            # Mock mode
            await asyncio.sleep(0.3)
        
        # Immediately notify all Socket.IO clients
        await sio.emit('teleoperation_stopped', {
            "type": "teleoperation_stopped"
        })
        
        return ApiResponse(
            status="success",
            message="Teleoperation stopped successfully"
        )
        
    except Exception as e:
        logger.error(f"❌ Error stopping teleoperation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/robot/teleoperate/emergency-stop", response_model=ApiResponse)
async def emergency_stop():
    """Emergency stop - immediate halt of all operations"""
    try:
        logger.warning("🚨 EMERGENCY STOP ACTIVATED")
        
        if robot_service:
            robot_service.emergency_stop()
        else:
            # Mock mode
            await asyncio.sleep(0.1)
        
        # Immediately notify all Socket.IO clients
        await sio.emit('emergency_stop', {
            "type": "emergency_stop",
            "timestamp": asyncio.get_event_loop().time()
        })
        
        return ApiResponse(
            status="success",
            message="Emergency stop activated"
        )
        
    except Exception as e:
        logger.error(f"❌ Error during emergency stop: {e}")
        # For emergency stop, we return success even if there's an error
        # to ensure the frontend gets a response
        return ApiResponse(
            status="partial_success",
            message=f"Emergency stop completed with warnings: {str(e)}"
        )

@app.get("/api/robot/teleoperate/performance")
async def get_performance_metrics():
    """Get current performance metrics"""
    try:
        if robot_service:
            metrics = robot_service.get_performance_metrics()
            return {"status": "success", "data": metrics}
        else:
            # Mock metrics
            import time
            return {
                "status": "success", 
                "data": {
                    "fps": 30,
                    "latency_ms": 15,
                    "cpu_usage": 45.2,
                    "memory_usage": 1024,
                    "timestamp": time.time()
                }
            }
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time communication
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()
    connected_websockets.append(websocket)
    
    logger.info(f"🔌 New WebSocket connection. Total: {len(connected_websockets)}")
    
    try:
        # Send initial status
        if robot_service:
            status = robot_service.get_status()
            await websocket.send_text(json.dumps({
                "type": "status_update",
                "data": status
            }))
        
        while True:
            # Receive data from client
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                
                # Handle different message types
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": asyncio.get_event_loop().time()
                    }))
                
                elif message.get("type") == "request_status":
                    if robot_service:
                        status = robot_service.get_status()
                        await websocket.send_text(json.dumps({
                            "type": "status_update",
                            "data": status
                        }))
                
                # Handle camera streaming requests
                elif message.get("type") == "start_camera_stream":
                    logger.info("📷 Camera stream requested via WebSocket")
                    # TODO: Implement camera streaming
                    await websocket.send_text(json.dumps({
                        "type": "camera_stream_started",
                        "message": "Camera streaming will be implemented here"
                    }))
                
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received: {data}")
                
    except WebSocketDisconnect:
        logger.info("🔌 WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if websocket in connected_websockets:
            connected_websockets.remove(websocket)
        logger.info(f"🔌 WebSocket removed. Total: {len(connected_websockets)}")

async def broadcast_to_websockets(message: dict):
    """Broadcast a message to all connected WebSocket clients"""
    if not connected_websockets:
        return
    
    message_str = json.dumps(message)
    disconnected = []
    
    for websocket in connected_websockets:
        try:
            await websocket.send_text(message_str)
        except:
            disconnected.append(websocket)
    
    # Remove disconnected WebSockets
    for ws in disconnected:
        connected_websockets.remove(ws)

# Serve frontend static files
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main frontend page"""
    try:
        frontend_path = os.path.join(os.path.dirname(__file__), "../frontend/dist/index.html")
        if os.path.exists(frontend_path):
            with open(frontend_path, 'r') as f:
                return HTMLResponse(content=f.read())
        else:
            return HTMLResponse(content="""
            <html>
                <head><title>LeRobot Web GUI</title></head>
                <body>
                    <h1>LeRobot FastAPI Backend</h1>
                    <p>Frontend not built yet. Run: <code>cd frontend && npm run build</code></p>
                    <p><a href="/api/docs">📚 API Documentation</a></p>
                </body>
            </html>
            """)
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error loading frontend: {e}</h1>")

# Mount static files (if frontend is built)
frontend_dist = os.path.join(os.path.dirname(__file__), "../frontend/dist")
if os.path.exists(frontend_dist):
    app.mount("/static", StaticFiles(directory=frontend_dist), name="static")

# Development mode
if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting FastAPI development server...")
    logger.info("📚 API Documentation will be available at: http://localhost:5000/api/docs")
    
    uvicorn.run(
        socket_app,  # Use the combined app with Socket.IO
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info"
    )
