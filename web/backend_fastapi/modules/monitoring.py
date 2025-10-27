"""
Performance Monitoring and Metrics Module
=========================================

Handles all monitoring and performance tracking:
- Real-time system performance metrics
- Teleoperation performance monitoring
- Resource usage tracking
- Health monitoring and alerts
- Performance analytics

This module provides comprehensive monitoring capabilities
for system optimization and debugging.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import logging
import time
import psutil
import threading
from datetime import datetime, timedelta
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])

# Enums
class PerformanceLevel(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

class AlertLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

# Pydantic models
class ApiResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None

class PerformanceMetrics(BaseModel):
    """Real-time performance metrics"""
    timestamp: float = Field(description="Timestamp of measurement")
    cpu_usage: float = Field(description="CPU usage percentage")
    memory_usage: float = Field(description="Memory usage percentage")
    fps: float = Field(description="Camera frames per second")
    latency_ms: float = Field(description="Communication latency in milliseconds")
    network_usage: Dict[str, float] = Field(description="Network usage statistics")
    teleoperation_responsive: bool = Field(description="Teleoperation responsiveness")

class PerformanceAlert(BaseModel):
    """Performance alert/warning"""
    level: AlertLevel
    component: str
    message: str
    timestamp: float
    threshold_exceeded: Optional[str] = None
    suggested_action: Optional[str] = None

class MonitoringConfig(BaseModel):
    """Monitoring configuration settings"""
    collection_interval: float = Field(default=1.0, ge=0.1, le=60.0, description="Data collection interval in seconds")
    history_duration: int = Field(default=300, ge=60, le=3600, description="History retention in seconds")
    enable_alerts: bool = Field(default=True, description="Enable performance alerts")
    cpu_threshold: float = Field(default=80.0, ge=50.0, le=95.0, description="CPU usage alert threshold")
    memory_threshold: float = Field(default=85.0, ge=50.0, le=95.0, description="Memory usage alert threshold")
    latency_threshold: float = Field(default=100.0, ge=10.0, le=1000.0, description="Latency alert threshold in ms")
    fps_threshold: float = Field(default=15.0, ge=5.0, le=60.0, description="Minimum FPS threshold")

# Global monitoring state
monitoring_state = {
    "active": False,
    "config": MonitoringConfig(),
    "metrics_history": deque(maxlen=300),  # 5 minutes at 1Hz
    "alerts": deque(maxlen=100),
    "performance_summary": None,
    "last_update": None,
    "monitoring_thread": None
}

@router.post("/start", response_model=ApiResponse)
async def start_monitoring():
    """
    Start performance monitoring
    
    Begins real-time collection of:
    - System resource usage
    - Network performance
    - Camera frame rates
    - Teleoperation latency
    """
    try:
        if monitoring_state["active"]:
            return ApiResponse(
                status="info",
                message="Monitoring is already active",
                data={"active": True}
            )
        
        logger.info("Starting performance monitoring")
        
        # Start monitoring thread
        monitoring_thread = threading.Thread(target=_monitoring_loop, daemon=True)
        monitoring_thread.start()
        
        monitoring_state["active"] = True
        monitoring_state["monitoring_thread"] = monitoring_thread
        monitoring_state["last_update"] = time.time()
        
        logger.info("Performance monitoring started successfully")
        
        return ApiResponse(
            status="success",
            message="Performance monitoring started",
            data={
                "active": True,
                "collection_interval": monitoring_state["config"].collection_interval,
                "history_duration": monitoring_state["config"].history_duration
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to start monitoring: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start monitoring: {str(e)}"
        )

@router.post("/stop", response_model=ApiResponse)
async def stop_monitoring():
    """Stop performance monitoring"""
    try:
        if not monitoring_state["active"]:
            return ApiResponse(
                status="info",
                message="Monitoring is not active",
                data={"active": False}
            )
        
        logger.info("Stopping performance monitoring")
        
        monitoring_state["active"] = False
        # Thread will stop on next iteration
        
        logger.info("Performance monitoring stopped")
        
        return ApiResponse(
            status="success",
            message="Performance monitoring stopped",
            data={"active": False}
        )
        
    except Exception as e:
        logger.error(f"Failed to stop monitoring: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to stop monitoring: {str(e)}"
        )

@router.get("/metrics", response_model=ApiResponse)
async def get_current_metrics():
    """
    Get current performance metrics
    
    Returns real-time system performance data
    """
    try:
        current_metrics = _collect_metrics()
        
        # Calculate performance level
        performance_level = _calculate_performance_level(current_metrics)
        
        response_data = {
            "current_metrics": current_metrics.dict(),
            "performance_level": performance_level,
            "monitoring_active": monitoring_state["active"],
            "last_update": monitoring_state["last_update"]
        }
        
        return ApiResponse(
            status="success",
            message="Current metrics retrieved",
            data=response_data
        )
        
    except Exception as e:
        logger.error(f"Failed to get current metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get current metrics: {str(e)}"
        )

@router.get("/history", response_model=ApiResponse)
async def get_metrics_history(duration_minutes: int = 5):
    """
    Get historical performance metrics
    
    Args:
        duration_minutes: How many minutes of history to return (1-60)
    """
    try:
        duration_minutes = max(1, min(60, duration_minutes))
        cutoff_time = time.time() - (duration_minutes * 60)
        
        # Filter history by time
        recent_metrics = [
            metric for metric in monitoring_state["metrics_history"]
            if metric["timestamp"] >= cutoff_time
        ]
        
        # Calculate statistics
        stats = _calculate_history_stats(recent_metrics)
        
        return ApiResponse(
            status="success",
            message=f"Metrics history for last {duration_minutes} minutes",
            data={
                "metrics": recent_metrics[-100:],  # Limit to last 100 points
                "duration_minutes": duration_minutes,
                "data_points": len(recent_metrics),
                "statistics": stats
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get metrics history: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get metrics history: {str(e)}"
        )

@router.get("/alerts", response_model=ApiResponse)
async def get_alerts(level: Optional[AlertLevel] = None):
    """
    Get performance alerts
    
    Args:
        level: Filter by alert level (info, warning, error, critical)
    """
    try:
        alerts = list(monitoring_state["alerts"])
        
        if level:
            alerts = [alert for alert in alerts if alert["level"] == level]
        
        # Sort by timestamp (newest first)
        alerts.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return ApiResponse(
            status="success",
            message="Performance alerts retrieved",
            data={
                "alerts": alerts,
                "total_count": len(monitoring_state["alerts"]),
                "filtered_count": len(alerts),
                "filter_level": level
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get alerts: {str(e)}"
        )

@router.post("/config", response_model=ApiResponse)
async def update_monitoring_config(config: MonitoringConfig):
    """
    Update monitoring configuration
    
    Allows adjustment of:
    - Collection intervals
    - Alert thresholds
    - History retention
    """
    try:
        logger.info("Updating monitoring configuration")
        
        # Update configuration
        monitoring_state["config"] = config
        
        # Adjust history buffer size based on new settings
        new_max_size = int(config.history_duration / config.collection_interval)
        monitoring_state["metrics_history"] = deque(
            monitoring_state["metrics_history"], 
            maxlen=new_max_size
        )
        
        logger.info(f"Monitoring config updated: {config.dict()}")
        
        return ApiResponse(
            status="success",
            message="Monitoring configuration updated",
            data=config.dict()
        )
        
    except Exception as e:
        logger.error(f"Failed to update monitoring config: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update monitoring config: {str(e)}"
        )

@router.get("/status", response_model=ApiResponse)
async def get_monitoring_status():
    """Get monitoring system status and health"""
    try:
        status_data = {
            "monitoring_active": monitoring_state["active"],
            "config": monitoring_state["config"].dict(),
            "metrics_history_count": len(monitoring_state["metrics_history"]),
            "alerts_count": len(monitoring_state["alerts"]),
            "last_update": monitoring_state["last_update"],
            "performance_summary": monitoring_state["performance_summary"],
            "recent_alerts": list(monitoring_state["alerts"])[-5:]  # Last 5 alerts
        }
        
        return ApiResponse(
            status="success",
            message="Monitoring status retrieved",
            data=status_data
        )
        
    except Exception as e:
        logger.error(f"Failed to get monitoring status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get monitoring status: {str(e)}"
        )

# Helper functions
def _monitoring_loop():
    """Main monitoring loop (runs in separate thread)"""
    logger.info("Monitoring loop started")
    
    while monitoring_state["active"]:
        try:
            # Collect metrics
            metrics = _collect_metrics()
            
            # Store in history
            monitoring_state["metrics_history"].append(metrics.dict())
            
            # Check for alerts
            _check_performance_alerts(metrics)
            
            # Update timestamp
            monitoring_state["last_update"] = time.time()
            
            # Sleep until next collection
            time.sleep(monitoring_state["config"].collection_interval)
            
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
            time.sleep(1.0)  # Wait before retrying
    
    logger.info("Monitoring loop stopped")

def _collect_metrics() -> PerformanceMetrics:
    """Collect current performance metrics"""
    try:
        # System metrics
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # Network metrics (mock for now)
        network_stats = psutil.net_io_counters()
        network_usage = {
            "bytes_sent": network_stats.bytes_sent,
            "bytes_recv": network_stats.bytes_recv,
            "packets_sent": network_stats.packets_sent,
            "packets_recv": network_stats.packets_recv
        }
        
        # Mock camera and latency metrics
        # In real implementation, these would come from actual systems
        fps = 30.0  # Mock camera FPS
        latency_ms = 25.0  # Mock communication latency
        teleoperation_responsive = latency_ms < 100.0
        
        return PerformanceMetrics(
            timestamp=time.time(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            fps=fps,
            latency_ms=latency_ms,
            network_usage=network_usage,
            teleoperation_responsive=teleoperation_responsive
        )
        
    except Exception as e:
        logger.error(f"Failed to collect metrics: {e}")
        # Return default metrics on error
        return PerformanceMetrics(
            timestamp=time.time(),
            cpu_usage=0.0,
            memory_usage=0.0,
            fps=0.0,
            latency_ms=999.0,
            network_usage={},
            teleoperation_responsive=False
        )

def _check_performance_alerts(metrics: PerformanceMetrics):
    """Check metrics against thresholds and generate alerts"""
    config = monitoring_state["config"]
    
    if not config.enable_alerts:
        return
    
    alerts = []
    
    # CPU usage alert
    if metrics.cpu_usage > config.cpu_threshold:
        alerts.append(PerformanceAlert(
            level=AlertLevel.WARNING if metrics.cpu_usage < 90 else AlertLevel.ERROR,
            component="CPU",
            message=f"High CPU usage: {metrics.cpu_usage:.1f}%",
            timestamp=metrics.timestamp,
            threshold_exceeded=f"CPU > {config.cpu_threshold}%",
            suggested_action="Check for resource-intensive processes"
        ))
    
    # Memory usage alert
    if metrics.memory_usage > config.memory_threshold:
        alerts.append(PerformanceAlert(
            level=AlertLevel.WARNING if metrics.memory_usage < 90 else AlertLevel.ERROR,
            component="Memory",
            message=f"High memory usage: {metrics.memory_usage:.1f}%",
            timestamp=metrics.timestamp,
            threshold_exceeded=f"Memory > {config.memory_threshold}%",
            suggested_action="Check for memory leaks or close unused applications"
        ))
    
    # Latency alert
    if metrics.latency_ms > config.latency_threshold:
        alerts.append(PerformanceAlert(
            level=AlertLevel.WARNING,
            component="Network",
            message=f"High latency: {metrics.latency_ms:.1f}ms",
            timestamp=metrics.timestamp,
            threshold_exceeded=f"Latency > {config.latency_threshold}ms",
            suggested_action="Check network connection and reduce network load"
        ))
    
    # FPS alert
    if metrics.fps < config.fps_threshold:
        alerts.append(PerformanceAlert(
            level=AlertLevel.WARNING,
            component="Camera",
            message=f"Low frame rate: {metrics.fps:.1f} FPS",
            timestamp=metrics.timestamp,
            threshold_exceeded=f"FPS < {config.fps_threshold}",
            suggested_action="Check camera connection and system performance"
        ))
    
    # Add alerts to queue
    for alert in alerts:
        monitoring_state["alerts"].append(alert.dict())

def _calculate_performance_level(metrics: PerformanceMetrics) -> PerformanceLevel:
    """Calculate overall performance level based on metrics"""
    config = monitoring_state["config"]
    
    # Score each metric (0-100, higher is better)
    cpu_score = max(0, 100 - metrics.cpu_usage)
    memory_score = max(0, 100 - metrics.memory_usage)
    latency_score = max(0, 100 - (metrics.latency_ms / config.latency_threshold * 100))
    fps_score = min(100, (metrics.fps / config.fps_threshold) * 100)
    
    # Calculate weighted average
    overall_score = (cpu_score * 0.3 + memory_score * 0.3 + 
                    latency_score * 0.2 + fps_score * 0.2)
    
    # Map to performance level
    if overall_score >= 80:
        return PerformanceLevel.EXCELLENT
    elif overall_score >= 60:
        return PerformanceLevel.GOOD
    elif overall_score >= 40:
        return PerformanceLevel.FAIR
    else:
        return PerformanceLevel.POOR

def _calculate_history_stats(metrics_list: List[Dict]) -> Dict[str, Any]:
    """Calculate statistics from historical metrics"""
    if not metrics_list:
        return {}
    
    cpu_values = [m["cpu_usage"] for m in metrics_list]
    memory_values = [m["memory_usage"] for m in metrics_list]
    latency_values = [m["latency_ms"] for m in metrics_list]
    fps_values = [m["fps"] for m in metrics_list]
    
    return {
        "cpu": {
            "average": sum(cpu_values) / len(cpu_values),
            "max": max(cpu_values),
            "min": min(cpu_values)
        },
        "memory": {
            "average": sum(memory_values) / len(memory_values),
            "max": max(memory_values),
            "min": min(memory_values)
        },
        "latency": {
            "average": sum(latency_values) / len(latency_values),
            "max": max(latency_values),
            "min": min(latency_values)
        },
        "fps": {
            "average": sum(fps_values) / len(fps_values),
            "max": max(fps_values),
            "min": min(fps_values)
        }
    }
