"""
Recording and Dataset Management Module
======================================

Handles all recording and dataset operations:
- Teleoperation session recording
- Dataset creation and management
- Episode recording with metadata
- Data export and import
- Recording quality validation

This module manages the data collection pipeline for
training and analysis.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import logging
import time
import json
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/recording", tags=["recording"])

# Enums
class RecordingStatus(str, Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"

class RecordingQuality(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"

class DatasetFormat(str, Enum):
    LEROBOT = "lerobot"
    HDF5 = "hdf5"
    ZARR = "zarr"
    PICKLE = "pickle"

# Pydantic models
class ApiResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None

class RecordingSession(BaseModel):
    """Recording session information"""
    session_id: str = Field(description="Unique session identifier")
    name: str = Field(description="Human-readable session name")
    description: Optional[str] = Field(default="", description="Session description")
    quality: RecordingQuality = Field(default=RecordingQuality.MEDIUM, description="Recording quality level")
    auto_save: bool = Field(default=True, description="Auto-save episodes")
    max_episode_length: int = Field(default=300, ge=10, le=3600, description="Maximum episode length in seconds")

class RecordingConfig(BaseModel):
    """Recording configuration settings"""
    save_directory: str = Field(description="Directory to save recordings")
    filename_template: str = Field(default="episode_{timestamp}_{session_id}", description="Filename template")
    video_fps: int = Field(default=30, ge=10, le=60, description="Video recording frame rate")
    compress_data: bool = Field(default=True, description="Compress recorded data")
    include_metadata: bool = Field(default=True, description="Include episode metadata")
    format: DatasetFormat = Field(default=DatasetFormat.LEROBOT, description="Dataset format")

class EpisodeMetadata(BaseModel):
    """Episode metadata information"""
    episode_id: str
    session_id: str
    timestamp: float
    duration_seconds: float
    frame_count: int
    robot_preset: str
    operator_notes: Optional[str] = None
    success_rating: Optional[int] = Field(default=None, ge=1, le=5, description="Success rating 1-5")
    tags: List[str] = Field(default=[], description="Episode tags for categorization")

class DatasetInfo(BaseModel):
    """Dataset information"""
    name: str
    path: str
    episode_count: int
    total_duration: float
    created_date: str
    last_modified: str
    format: DatasetFormat
    size_mb: float

# Global recording state
recording_state = {
    "status": RecordingStatus.IDLE,
    "current_session": None,
    "current_episode": None,
    "episode_start_time": None,
    "episode_data": [],
    "sessions": {},
    "datasets": {},
    "config": RecordingConfig(save_directory="./recordings"),
    "total_episodes_recorded": 0
}

@router.post("/session/start", response_model=ApiResponse)
async def start_recording_session(session: RecordingSession):
    """
    Start a new recording session
    
    Features:
    - Session management
    - Quality settings
    - Auto-save configuration
    - Episode length limits
    """
    try:
        if recording_state["status"] == RecordingStatus.RECORDING:
            return ApiResponse(
                status="error",
                message="Cannot start new session while recording is active",
                data={"current_status": recording_state["status"]}
            )
        
        logger.info(f"Starting recording session: {session.name}")
        
        # Generate session ID if not provided
        if not session.session_id:
            session.session_id = f"session_{int(time.time())}"
        
        # Create session directory
        session_dir = Path(recording_state["config"].save_directory) / session.session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Store session
        recording_state["current_session"] = session
        recording_state["sessions"][session.session_id] = {
            **session.dict(),
            "created_at": time.time(),
            "episodes": [],
            "directory": str(session_dir)
        }
        
        recording_state["status"] = RecordingStatus.IDLE
        
        logger.info(f"Recording session started: {session.session_id}")
        
        return ApiResponse(
            status="success",
            message="Recording session started",
            data={
                "session_id": session.session_id,
                "session_directory": str(session_dir),
                "status": recording_state["status"]
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to start recording session: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start recording session: {str(e)}"
        )

@router.post("/episode/start", response_model=ApiResponse)
async def start_episode_recording():
    """
    Start recording a new episode
    
    Features:
    - Episode timing
    - Data collection initiation
    - Metadata capture
    - Quality validation
    """
    try:
        if not recording_state["current_session"]:
            raise HTTPException(
                status_code=400,
                detail="No active recording session. Start a session first."
            )
        
        if recording_state["status"] == RecordingStatus.RECORDING:
            return ApiResponse(
                status="error",
                message="Episode recording already in progress",
                data={"current_status": recording_state["status"]}
            )
        
        logger.info("Starting episode recording")
        
        # Generate episode ID
        episode_id = f"episode_{int(time.time())}_{len(recording_state['sessions'][recording_state['current_session'].session_id]['episodes'])}"
        
        # Initialize episode
        recording_state["current_episode"] = episode_id
        recording_state["episode_start_time"] = time.time()
        recording_state["episode_data"] = []
        recording_state["status"] = RecordingStatus.RECORDING
        
        # Start data collection (mock implementation)
        await _start_data_collection()
        
        logger.info(f"Episode recording started: {episode_id}")
        
        return ApiResponse(
            status="success",
            message="Episode recording started",
            data={
                "episode_id": episode_id,
                "session_id": recording_state["current_session"].session_id,
                "start_time": recording_state["episode_start_time"],
                "status": recording_state["status"]
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to start episode recording: {e}")
        recording_state["status"] = RecordingStatus.ERROR
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start episode recording: {str(e)}"
        )

@router.post("/episode/stop", response_model=ApiResponse)
async def stop_episode_recording(metadata: Optional[EpisodeMetadata] = None):
    """
    Stop current episode recording
    
    Features:
    - Data finalization
    - Metadata attachment
    - Quality validation
    - Auto-save (if enabled)
    """
    try:
        if recording_state["status"] != RecordingStatus.RECORDING:
            return ApiResponse(
                status="error",
                message="No episode recording in progress",
                data={"current_status": recording_state["status"]}
            )
        
        logger.info("Stopping episode recording")
        
        # Calculate episode duration
        end_time = time.time()
        duration = end_time - recording_state["episode_start_time"]
        
        # Stop data collection
        await _stop_data_collection()
        
        # Create episode metadata
        if not metadata:
            metadata = EpisodeMetadata(
                episode_id=recording_state["current_episode"],
                session_id=recording_state["current_session"].session_id,
                timestamp=recording_state["episode_start_time"],
                duration_seconds=duration,
                frame_count=len(recording_state["episode_data"]),
                robot_preset="unknown"
            )
        
        # Validate episode quality
        quality_check = _validate_episode_quality()
        
        # Save episode if auto-save is enabled
        saved_path = None
        if recording_state["current_session"].auto_save:
            saved_path = await _save_episode(metadata, quality_check)
        
        # Add to session episodes list
        episode_info = {
            **metadata.dict(),
            "end_time": end_time,
            "quality_check": quality_check,
            "saved_path": saved_path
        }
        
        recording_state["sessions"][recording_state["current_session"].session_id]["episodes"].append(episode_info)
        recording_state["total_episodes_recorded"] += 1
        
        # Reset recording state
        recording_state["status"] = RecordingStatus.COMPLETED
        recording_state["current_episode"] = None
        recording_state["episode_start_time"] = None
        
        logger.info(f"Episode recording completed: {metadata.episode_id}")
        
        return ApiResponse(
            status="success",
            message="Episode recording completed",
            data={
                "episode_metadata": metadata.dict(),
                "duration_seconds": duration,
                "quality_check": quality_check,
                "saved_path": saved_path,
                "total_episodes": recording_state["total_episodes_recorded"]
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to stop episode recording: {e}")
        recording_state["status"] = RecordingStatus.ERROR
        raise HTTPException(
            status_code=500,
            detail=f"Failed to stop episode recording: {str(e)}"
        )

@router.get("/status", response_model=ApiResponse)
async def get_recording_status():
    """Get current recording status and information"""
    try:
        status_data = {
            "recording_status": recording_state["status"],
            "current_session": recording_state["current_session"].dict() if recording_state["current_session"] else None,
            "current_episode": recording_state["current_episode"],
            "episode_duration": time.time() - recording_state["episode_start_time"] if recording_state["episode_start_time"] else None,
            "total_sessions": len(recording_state["sessions"]),
            "total_episodes": recording_state["total_episodes_recorded"],
            "config": recording_state["config"].dict()
        }
        
        return ApiResponse(
            status="success",
            message="Recording status retrieved",
            data=status_data
        )
        
    except Exception as e:
        logger.error(f"Failed to get recording status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get recording status: {str(e)}"
        )

@router.get("/sessions", response_model=ApiResponse)
async def get_recording_sessions():
    """Get list of all recording sessions"""
    try:
        sessions_data = []
        
        for session_id, session_info in recording_state["sessions"].items():
            session_summary = {
                "session_id": session_id,
                "name": session_info["name"],
                "description": session_info.get("description", ""),
                "created_at": session_info["created_at"],
                "episode_count": len(session_info["episodes"]),
                "total_duration": sum(ep["duration_seconds"] for ep in session_info["episodes"]),
                "directory": session_info["directory"]
            }
            sessions_data.append(session_summary)
        
        # Sort by creation time (newest first)
        sessions_data.sort(key=lambda x: x["created_at"], reverse=True)
        
        return ApiResponse(
            status="success",
            message="Recording sessions retrieved",
            data={
                "sessions": sessions_data,
                "total_sessions": len(sessions_data)
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get recording sessions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get recording sessions: {str(e)}"
        )

@router.get("/datasets", response_model=ApiResponse)
async def get_datasets():
    """Get list of available datasets"""
    try:
        datasets_data = []
        
        # Scan recording directory for datasets
        recording_dir = Path(recording_state["config"].save_directory)
        if recording_dir.exists():
            for dataset_path in recording_dir.iterdir():
                if dataset_path.is_dir():
                    dataset_info = _analyze_dataset(dataset_path)
                    if dataset_info:
                        datasets_data.append(dataset_info)
        
        # Sort by last modified (newest first)
        datasets_data.sort(key=lambda x: x["last_modified"], reverse=True)
        
        return ApiResponse(
            status="success",
            message="Datasets retrieved",
            data={
                "datasets": datasets_data,
                "total_datasets": len(datasets_data)
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get datasets: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get datasets: {str(e)}"
        )

@router.post("/config", response_model=ApiResponse)
async def update_recording_config(config: RecordingConfig):
    """Update recording configuration"""
    try:
        logger.info("Updating recording configuration")
        
        # Validate save directory
        save_dir = Path(config.save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Update configuration
        recording_state["config"] = config
        
        logger.info(f"Recording config updated: {config.dict()}")
        
        return ApiResponse(
            status="success",
            message="Recording configuration updated",
            data=config.dict()
        )
        
    except Exception as e:
        logger.error(f"Failed to update recording config: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update recording config: {str(e)}"
        )

@router.post("/export/{session_id}", response_model=ApiResponse)
async def export_session(session_id: str, export_format: DatasetFormat = DatasetFormat.LEROBOT):
    """Export recording session to specified format"""
    try:
        if session_id not in recording_state["sessions"]:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {session_id}"
            )
        
        logger.info(f"Exporting session {session_id} to {export_format}")
        
        session_info = recording_state["sessions"][session_id]
        export_path = await _export_session_data(session_info, export_format)
        
        return ApiResponse(
            status="success",
            message=f"Session exported to {export_format}",
            data={
                "session_id": session_id,
                "export_format": export_format,
                "export_path": export_path,
                "episode_count": len(session_info["episodes"])
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to export session: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to export session: {str(e)}"
        )

# Helper functions
async def _start_data_collection():
    """Start collecting teleoperation data"""
    logger.info("Starting data collection")
    # Mock implementation - would interface with robot and camera systems
    pass

async def _stop_data_collection():
    """Stop data collection and finalize episode data"""
    logger.info("Stopping data collection")
    # Mock implementation - would stop all data streams
    pass

def _validate_episode_quality() -> Dict[str, Any]:
    """Validate episode recording quality"""
    # Mock implementation
    return {
        "overall_quality": "good",
        "frame_rate_consistent": True,
        "data_completeness": 98.5,
        "timestamp_accuracy": True,
        "motion_smoothness": "acceptable",
        "warnings": [],
        "recommendations": []
    }

async def _save_episode(metadata: EpisodeMetadata, quality_check: Dict) -> str:
    """Save episode data to disk"""
    try:
        session_dir = Path(recording_state["sessions"][metadata.session_id]["directory"])
        filename = recording_state["config"].filename_template.format(
            timestamp=int(metadata.timestamp),
            session_id=metadata.session_id,
            episode_id=metadata.episode_id
        )
        
        episode_path = session_dir / f"{filename}.json"
        
        # Create episode data package
        episode_package = {
            "metadata": metadata.dict(),
            "quality_check": quality_check,
            "data": recording_state["episode_data"],  # Mock data
            "config": recording_state["config"].dict()
        }
        
        # Save to file
        with open(episode_path, 'w') as f:
            json.dump(episode_package, f, indent=2)
        
        logger.info(f"Episode saved: {episode_path}")
        return str(episode_path)
        
    except Exception as e:
        logger.error(f"Failed to save episode: {e}")
        raise

def _analyze_dataset(dataset_path: Path) -> Optional[Dict]:
    """Analyze dataset directory and return info"""
    try:
        # Mock implementation
        episode_files = list(dataset_path.glob("*.json"))
        
        if not episode_files:
            return None
        
        # Calculate dataset statistics
        total_duration = len(episode_files) * 60  # Mock: 60 seconds per episode
        size_mb = sum(f.stat().st_size for f in episode_files) / (1024 * 1024)
        
        return {
            "name": dataset_path.name,
            "path": str(dataset_path),
            "episode_count": len(episode_files),
            "total_duration": total_duration,
            "created_date": datetime.fromtimestamp(dataset_path.stat().st_ctime).isoformat(),
            "last_modified": datetime.fromtimestamp(dataset_path.stat().st_mtime).isoformat(),
            "format": "lerobot",
            "size_mb": size_mb
        }
        
    except Exception as e:
        logger.error(f"Failed to analyze dataset {dataset_path}: {e}")
        return None

async def _export_session_data(session_info: Dict, export_format: DatasetFormat) -> str:
    """Export session data to specified format"""
    # Mock implementation
    export_filename = f"{session_info['name']}_{export_format.value}_export.zip"
    export_path = Path(session_info["directory"]) / export_filename
    
    logger.info(f"Mock export to {export_path}")
    
    # In real implementation, would convert data to specified format
    return str(export_path)
