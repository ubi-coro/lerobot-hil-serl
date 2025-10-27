"""
Dataset management module for LeRobot web interface.
Provides endpoints for browsing and visualizing local datasets.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel


router = APIRouter()


class DatasetInfo(BaseModel):
    """Dataset information model"""
    id: str
    name: str
    path: str
    episodes: int
    size: str
    created: str


class VisualizationRequest(BaseModel):
    """Request model for dataset visualization"""
    repo_id: str
    root_path: str


class DirectoryBrowseRequest(BaseModel):
    """Request model for directory browsing"""
    path: str


@router.get("/browse", response_model=List[DatasetInfo])
async def browse_local_datasets():
    """
    Browse local datasets in common data directories.
    Returns a list of discovered dataset directories.
    """
    datasets = []
    
    # Common dataset locations to check
    search_paths = [
        Path.home() / "data",
        Path.home() / "datasets", 
        Path("/data"),
        Path("/datasets"),
        Path(os.getcwd()) / "data",
        Path(os.getcwd()) / "datasets"
    ]
    
    for search_path in search_paths:
        if search_path.exists() and search_path.is_dir():
            try:
                for item in search_path.iterdir():
                    if item.is_dir():
                        # Check if it looks like a dataset directory
                        # (contains episode files or has dataset-like structure)
                        episode_files = list(item.glob("episode_*.parquet"))
                        if episode_files or (item / "data").exists():
                            datasets.append(DatasetInfo(
                                id=f"local_{item.name}",
                                name=item.name,
                                path=str(item),
                                episodes=len(episode_files),
                                size=_get_directory_size(item),
                                created=_get_creation_time(item)
                            ))
            except PermissionError:
                # Skip directories we can't access
                continue
    
    return datasets


@router.get("/info/{dataset_id}")
async def get_dataset_info(dataset_id: str):
    """
    Get detailed information about a specific dataset.
    """
    # This would typically query a database or filesystem
    # For now, return basic info
    return {
        "id": dataset_id,
        "message": f"Dataset info for {dataset_id}",
        "available": True
    }


@router.post("/visualize")
async def visualize_dataset(request: VisualizationRequest, background_tasks: BackgroundTasks):
    """
    Launch dataset HTML visualization.
    """
    try:
        # Validate paths
        root_path = Path(request.root_path)
        if not root_path.exists():
            raise HTTPException(status_code=400, detail=f"Root path does not exist: {request.root_path}")
        
        # Find the HTML visualize_dataset script
        script_path = Path(__file__).parent.parent.parent.parent / "lerobot" / "scripts" / "visualize_dataset_html.py"
        
        if not script_path.exists():
            raise HTTPException(status_code=500, detail="HTML visualization script not found")
        
        # Prepare command for HTML visualization (no episodes parameter - shows all episodes)
        cmd = [
            sys.executable,
            str(script_path),
            "--root", request.root_path,
            "--repo-id", request.repo_id,
            "--serve", "1",
            "--host", "127.0.0.1",
            "--port", "9090"
        ]
        
        # Launch visualization in background
        background_tasks.add_task(_run_visualization_command, cmd)
        
        return {
            "status": "launched",
            "message": f"HTML visualization started for {request.repo_id}",
            "command": " ".join(cmd),
            "web_url": "http://localhost:9090"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to launch visualization: {str(e)}")




@router.post("/browse-directory")
async def browse_directory(request: DirectoryBrowseRequest):
    """
    Browse directory contents for folder selection.
    Returns a list of subdirectories in the specified path.
    """
    try:
        raw_path = request.path.strip() if request.path else ''
        # Support home expansion using ~ similar to shell behavior
        if raw_path in ('', '~', '~/'):
            path = Path.home()
        elif raw_path.startswith('~/'):
            path = Path.home() / raw_path[2:]
        else:
            path = Path(raw_path)
        
        # Validate path exists and is accessible
        if not path.exists():
            raise HTTPException(status_code=400, detail=f"Path does not exist: {request.path}")
        
        if not path.is_dir():
            raise HTTPException(status_code=400, detail=f"Path is not a directory: {request.path}")
        
        folders = []
        
        try:
            # List only directories
            for item in path.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    try:
                        # Get basic folder info
                        stat_info = item.stat()
                        folders.append({
                            "name": item.name,
                            "path": str(item),
                            "permissions": oct(stat_info.st_mode)[-3:] if hasattr(stat_info, 'st_mode') else None
                        })
                    except (OSError, PermissionError):
                        # Skip folders we can't access
                        continue
                        
        except PermissionError:
            raise HTTPException(status_code=403, detail=f"Permission denied accessing: {request.path}")
        
        # Sort folders alphabetically
        folders.sort(key=lambda x: x['name'].lower())
        
        # Get mount/drive information for better navigation
        mount_info = _get_mount_info()
        
        return {
            "path": str(path),
            "folders": folders,
            "total": len(folders),
            "mounts": mount_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to browse directory: {str(e)}")


@router.get("/count")
async def get_dataset_count():
    """
    Get count of local datasets for dashboard.
    """
    try:
        datasets = await browse_local_datasets()
        return {
            "count": len(datasets),
            "total": len(datasets)
        }
    except Exception as e:
        return {
            "count": 0,
            "total": 0,
            "error": str(e)
        }


def _get_directory_size(path: Path) -> str:
    """Get human-readable directory size"""
    try:
        total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        # Convert to human readable format
        for unit in ['B', 'KB', 'MB', 'GB']:
            if total_size < 1024:
                return f"{total_size:.1f} {unit}"
            total_size /= 1024
        return f"{total_size:.1f} TB"
    except (OSError, PermissionError):
        return "Unknown"


def _get_creation_time(path: Path) -> str:
    """Get directory creation time"""
    try:
        import datetime
        timestamp = path.stat().st_ctime
        return datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
    except (OSError, PermissionError):
        return "Unknown"


async def _run_visualization_command(cmd: List[str]):
    """
    Run visualization command in background.
    """
    try:
        # Run the command
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Don't wait for completion to allow background execution
        print(f"Started visualization process with PID: {process.pid}")
        print(f"Command: {' '.join(cmd)}")
        
        # Track the process
        import datetime
        _visualization_processes[process.pid] = {
            "process": process,
            "command": " ".join(cmd),
            "started_at": datetime.datetime.now().isoformat(),
            "web_url": "http://localhost:9090",
            "ws_url": "ws://localhost:9087"
        }
        
    except Exception as e:
        print(f"Error running visualization command: {e}")


# Global variable to track visualization processes
_visualization_processes = {}


@router.get("/visualization/status")
async def get_visualization_status():
    """
    Get status of running visualization processes.
    """
    active_processes = []
    
    # Clean up finished processes
    finished_pids = []
    for pid, process_info in _visualization_processes.items():
        if process_info["process"].poll() is not None:  # Process has finished
            finished_pids.append(pid)
    
    for pid in finished_pids:
        del _visualization_processes[pid]
    
    # Return active processes
    for pid, process_info in _visualization_processes.items():
        active_processes.append({
            "pid": pid,
            "command": process_info["command"],
            "started_at": process_info["started_at"],
            "web_url": process_info.get("web_url"),
            "ws_url": process_info.get("ws_url")
        })
    
    return {
        "active_processes": active_processes,
        "total": len(active_processes)
    }


@router.post("/visualization/stop")
async def stop_visualization():
    """
    Stop all running visualization processes.
    """
    stopped_count = 0
    
    for pid, process_info in _visualization_processes.items():
        try:
            process = process_info["process"]
            if process.poll() is None:  # Still running
                process.terminate()
                stopped_count += 1
        except Exception as e:
            print(f"Error stopping process {pid}: {e}")
    
    # Clear the tracking dictionary
    _visualization_processes.clear()
    
    return {
        "status": "stopped",
        "message": f"Stopped {stopped_count} visualization processes"
    }


def _get_mount_info() -> Dict[str, Any]:
    """Get information about mounted drives and filesystems (Cross-platform)"""
    try:
        mounts = []
        import platform
        
        system = platform.system().lower()
        
        if system == "linux":
            mounts = _get_mounts_linux()
        elif system == "windows":
            mounts = _get_mounts_windows()
        elif system == "darwin":  # macOS
            mounts = _get_mounts_macos()
        else:
            # Fallback for other systems
            mounts = _get_mounts_generic()
        
        # Sort by mount point
        mounts.sort(key=lambda x: x['mount_point'])
        
        return {
            "mounts": mounts,
            "total_mounts": len(mounts),
            "system": system,
            "common_paths": _get_common_paths(system)
        }
        
    except Exception as e:
        # Return basic info if mount detection fails
        return {
            "mounts": [],
            "total_mounts": 0,
            "error": str(e),
            "common_paths": _get_common_paths(platform.system().lower())
        }


def _get_mounts_linux():
    """Get mount information for Linux systems"""
    mounts = []
    
    try:
        # Read /proc/mounts for mount information
        with open('/proc/mounts', 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    device, mount_point, fs_type = parts[0], parts[1], parts[2]
                    
                    # Skip virtual filesystems and common system mounts
                    skip_fs_types = ['proc', 'sysfs', 'devtmpfs', 'devpts', 'tmpfs', 'cgroup', 'pstore', 'debugfs', 'hugetlbfs', 'securityfs', 'cgroup2', 'bpf', 'fusectl', 'configfs', 'tracefs']
                    if fs_type in skip_fs_types:
                        continue
                    
                    # Skip /proc, /sys, /dev mounts but allow others
                    skip_mounts = ['/proc', '/sys', '/dev', '/run']
                    if any(mount_point.startswith(skip) for skip in skip_mounts):
                        continue
                    
                    try:
                        mount_path = Path(mount_point)
                        if mount_path.exists() and mount_path.is_dir():
                            # Get disk usage information
                            statvfs = os.statvfs(mount_point)
                            total_bytes = statvfs.f_blocks * statvfs.f_frsize
                            available_bytes = statvfs.f_bavail * statvfs.f_frsize
                            used_bytes = total_bytes - available_bytes
                            
                            mounts.append({
                                "device": device,
                                "mount_point": mount_point,
                                "filesystem": fs_type,
                                "total_gb": round(total_bytes / (1024**3), 1),
                                "used_gb": round(used_bytes / (1024**3), 1),
                                "available_gb": round(available_bytes / (1024**3), 1),
                                "usage_percent": round((used_bytes / total_bytes) * 100, 1) if total_bytes > 0 else 0
                            })
                    except (OSError, PermissionError):
                        # Skip mounts we can't access
                        continue
    except Exception:
        pass
    
    return mounts


def _get_mounts_windows():
    """Get mount information for Windows systems"""
    mounts = []
    
    try:
        import subprocess
        # Use wmic to get drive information
        result = subprocess.run(['wmic', 'logicaldisk', 'get', 'name,filesystem,size,freespace', '/format:csv'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines[1:]:  # Skip header
                parts = line.split(',')
                if len(parts) >= 4:
                    drive_letter = parts[1].strip()
                    fs_type = parts[2].strip()
                    size_str = parts[3].strip()
                    free_str = parts[4].strip()
                    
                    try:
                        total_bytes = int(size_str) if size_str else 0
                        free_bytes = int(free_str) if free_str else 0
                        used_bytes = total_bytes - free_bytes
                        
                        if total_bytes > 0:
                            mounts.append({
                                "device": drive_letter,
                                "mount_point": f"{drive_letter}:\\",
                                "filesystem": fs_type,
                                "total_gb": round(total_bytes / (1024**3), 1),
                                "used_gb": round(used_bytes / (1024**3), 1),
                                "available_gb": round(free_bytes / (1024**3), 1),
                                "usage_percent": round((used_bytes / total_bytes) * 100, 1)
                            })
                    except (ValueError, ZeroDivisionError):
                        continue
    except Exception:
        pass
    
    return mounts


def _get_mounts_macos():
    """Get mount information for macOS systems"""
    mounts = []
    
    try:
        import subprocess
        # Use diskutil for macOS
        result = subprocess.run(['diskutil', 'list'], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            # Parse diskutil output (simplified)
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.startswith('/dev/'):
                    parts = line.split()
                    if len(parts) >= 3:
                        device = parts[0]
                        mount_point = parts[-1] if len(parts) > 1 else ""
                        
                        if mount_point and mount_point.startswith('/'):
                            try:
                                mount_path = Path(mount_point)
                                if mount_path.exists() and mount_path.is_dir():
                                    statvfs = os.statvfs(mount_point)
                                    total_bytes = statvfs.f_blocks * statvfs.f_frsize
                                    available_bytes = statvfs.f_bavail * statvfs.f_frsize
                                    used_bytes = total_bytes - available_bytes
                                    
                                    mounts.append({
                                        "device": device,
                                        "mount_point": mount_point,
                                        "filesystem": "apfs",  # Assume APFS for macOS
                                        "total_gb": round(total_bytes / (1024**3), 1),
                                        "used_gb": round(used_bytes / (1024**3), 1),
                                        "available_gb": round(available_bytes / (1024**3), 1),
                                        "usage_percent": round((used_bytes / total_bytes) * 100, 1) if total_bytes > 0 else 0
                                    })
                            except (OSError, PermissionError):
                                continue
    except Exception:
        pass
    
    return mounts


def _get_mounts_generic():
    """Generic mount detection fallback"""
    mounts = []
    
    # Try to detect common mount points
    common_mounts = ['/', '/home', '/usr', '/var', '/tmp']
    
    for mount_point in common_mounts:
        try:
            mount_path = Path(mount_point)
            if mount_path.exists() and mount_path.is_dir():
                statvfs = os.statvfs(mount_point)
                total_bytes = statvfs.f_blocks * statvfs.f_frsize
                available_bytes = statvfs.f_bavail * statvfs.f_frsize
                used_bytes = total_bytes - available_bytes
                
                mounts.append({
                    "device": "unknown",
                    "mount_point": mount_point,
                    "filesystem": "unknown",
                    "total_gb": round(total_bytes / (1024**3), 1),
                    "used_gb": round(used_bytes / (1024**3), 1),
                    "available_gb": round(available_bytes / (1024**3), 1),
                    "usage_percent": round((used_bytes / total_bytes) * 100, 1) if total_bytes > 0 else 0
                })
        except (OSError, PermissionError, AttributeError):
            continue
    
    return mounts


def _get_common_paths(system: str):
    """Get common paths for different operating systems"""
    if system == "windows":
        return [
            {"name": "Home", "path": str(Path.home()), "description": "User home directory"},
            {"name": "Desktop", "path": str(Path.home() / "Desktop"), "description": "Desktop folder"},
            {"name": "Documents", "path": str(Path.home() / "Documents"), "description": "Documents folder"},
            {"name": "C:", "path": "C:\\", "description": "System drive"},
            {"name": "Data", "path": str(Path.home() / "data"), "description": "Data directory"}
        ]
    elif system == "darwin":  # macOS
        return [
            {"name": "Home", "path": str(Path.home()), "description": "User home directory"},
            {"name": "Desktop", "path": str(Path.home() / "Desktop"), "description": "Desktop folder"},
            {"name": "Documents", "path": str(Path.home() / "Documents"), "description": "Documents folder"},
            {"name": "Root", "path": "/", "description": "System root"},
            {"name": "Volumes", "path": "/Volumes", "description": "Mounted volumes"}
        ]
    else:  # Linux and others
        return [
            {"name": "Home", "path": str(Path.home()), "description": "User home directory"},
            {"name": "Root", "path": "/", "description": "System root"},
            {"name": "Media", "path": "/media", "description": "Removable media"},
            {"name": "Mount", "path": "/mnt", "description": "Additional mounts"},
            {"name": "Data", "path": "/data", "description": "Data directory"},
            {"name": "Datasets", "path": str(Path.home() / "datasets"), "description": "User datasets"}
        ]


def _get_mounts_linux():
    """Get mount information for Linux systems"""
    mounts = []
    
    try:
        # Read /proc/mounts for mount information
        with open('/proc/mounts', 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    device, mount_point, fs_type = parts[0], parts[1], parts[2]
                    
                    # Skip virtual filesystems and common system mounts
                    skip_fs_types = ['proc', 'sysfs', 'devtmpfs', 'devpts', 'tmpfs', 'cgroup', 'pstore', 'debugfs', 'hugetlbfs', 'securityfs', 'cgroup2', 'bpf', 'fusectl', 'configfs', 'tracefs']
                    if fs_type in skip_fs_types:
                        continue
                    
                    # Skip /proc, /sys, /dev mounts but allow others
                    skip_mounts = ['/proc', '/sys', '/dev', '/run']
                    if any(mount_point.startswith(skip) for skip in skip_mounts):
                        continue
                    
                    try:
                        mount_path = Path(mount_point)
                        if mount_path.exists() and mount_path.is_dir():
                            # Get disk usage information
                            statvfs = os.statvfs(mount_point)
                            total_bytes = statvfs.f_blocks * statvfs.f_frsize
                            available_bytes = statvfs.f_bavail * statvfs.f_frsize
                            used_bytes = total_bytes - available_bytes
                            
                            mounts.append({
                                "device": device,
                                "mount_point": mount_point,
                                "filesystem": fs_type,
                                "total_gb": round(total_bytes / (1024**3), 1),
                                "used_gb": round(used_bytes / (1024**3), 1),
                                "available_gb": round(available_bytes / (1024**3), 1),
                                "usage_percent": round((used_bytes / total_bytes) * 100, 1) if total_bytes > 0 else 0
                            })
                    except (OSError, PermissionError):
                        # Skip mounts we can't access
                        continue
    except Exception:
        pass
    
    return mounts


def _get_mounts_windows():
    """Get mount information for Windows systems"""
    mounts = []
    
    try:
        import subprocess
        # Use wmic to get drive information
        result = subprocess.run(['wmic', 'logicaldisk', 'get', 'name,filesystem,size,freespace', '/format:csv'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines[1:]:  # Skip header
                parts = line.split(',')
                if len(parts) >= 4:
                    drive_letter = parts[1].strip()
                    fs_type = parts[2].strip()
                    size_str = parts[3].strip()
                    free_str = parts[4].strip()
                    
                    try:
                        total_bytes = int(size_str) if size_str else 0
                        free_bytes = int(free_str) if free_str else 0
                        used_bytes = total_bytes - free_bytes
                        
                        if total_bytes > 0:
                            mounts.append({
                                "device": drive_letter,
                                "mount_point": f"{drive_letter}:\\",
                                "filesystem": fs_type,
                                "total_gb": round(total_bytes / (1024**3), 1),
                                "used_gb": round(used_bytes / (1024**3), 1),
                                "available_gb": round(free_bytes / (1024**3), 1),
                                "usage_percent": round((used_bytes / total_bytes) * 100, 1)
                            })
                    except (ValueError, ZeroDivisionError):
                        continue
    except Exception:
        pass
    
    return mounts


def _get_mounts_macos():
    """Get mount information for macOS systems"""
    mounts = []
    
    try:
        import subprocess
        # Use diskutil for macOS
        result = subprocess.run(['diskutil', 'list'], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            # Parse diskutil output (simplified)
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.startswith('/dev/'):
                    parts = line.split()
                    if len(parts) >= 3:
                        device = parts[0]
                        mount_point = parts[-1] if len(parts) > 1 else ""
                        
                        if mount_point and mount_point.startswith('/'):
                            try:
                                mount_path = Path(mount_point)
                                if mount_path.exists() and mount_path.is_dir():
                                    statvfs = os.statvfs(mount_point)
                                    total_bytes = statvfs.f_blocks * statvfs.f_frsize
                                    available_bytes = statvfs.f_bavail * statvfs.f_frsize
                                    used_bytes = total_bytes - available_bytes
                                    
                                    mounts.append({
                                        "device": device,
                                        "mount_point": mount_point,
                                        "filesystem": "apfs",  # Assume APFS for macOS
                                        "total_gb": round(total_bytes / (1024**3), 1),
                                        "used_gb": round(used_bytes / (1024**3), 1),
                                        "available_gb": round(available_bytes / (1024**3), 1),
                                        "usage_percent": round((used_bytes / total_bytes) * 100, 1) if total_bytes > 0 else 0
                                    })
                            except (OSError, PermissionError):
                                continue
    except Exception:
        pass
    
    return mounts


def _get_mounts_generic():
    """Generic mount detection fallback"""
    mounts = []
    
    # Try to detect common mount points
    common_mounts = ['/', '/home', '/usr', '/var', '/tmp']
    
    for mount_point in common_mounts:
        try:
            mount_path = Path(mount_point)
            if mount_path.exists() and mount_path.is_dir():
                statvfs = os.statvfs(mount_point)
                total_bytes = statvfs.f_blocks * statvfs.f_frsize
                available_bytes = statvfs.f_bavail * statvfs.f_frsize
                used_bytes = total_bytes - available_bytes
                
                mounts.append({
                    "device": "unknown",
                    "mount_point": mount_point,
                    "filesystem": "unknown",
                    "total_gb": round(total_bytes / (1024**3), 1),
                    "used_gb": round(used_bytes / (1024**3), 1),
                    "available_gb": round(available_bytes / (1024**3), 1),
                    "usage_percent": round((used_bytes / total_bytes) * 100, 1) if total_bytes > 0 else 0
                })
        except (OSError, PermissionError, AttributeError):
            continue
    
    return mounts


def _get_common_paths(system: str):
    """Get common paths for different operating systems"""
    if system == "windows":
        return [
            {"name": "Home", "path": str(Path.home()), "description": "User home directory"},
            {"name": "Desktop", "path": str(Path.home() / "Desktop"), "description": "Desktop folder"},
            {"name": "Documents", "path": str(Path.home() / "Documents"), "description": "Documents folder"},
            {"name": "C:", "path": "C:\\", "description": "System drive"},
            {"name": "Data", "path": str(Path.home() / "data"), "description": "Data directory"}
        ]
    elif system == "darwin":  # macOS
        return [
            {"name": "Home", "path": str(Path.home()), "description": "User home directory"},
            {"name": "Desktop", "path": str(Path.home() / "Desktop"), "description": "Desktop folder"},
            {"name": "Documents", "path": str(Path.home() / "Documents"), "description": "Documents folder"},
            {"name": "Root", "path": "/", "description": "System root"},
            {"name": "Volumes", "path": "/Volumes", "description": "Mounted volumes"}
        ]
    else:  # Linux and others
        return [
            {"name": "Home", "path": str(Path.home()), "description": "User home directory"},
            {"name": "Root", "path": "/", "description": "System root"},
            {"name": "Media", "path": "/media", "description": "Removable media"},
            {"name": "Mount", "path": "/mnt", "description": "Additional mounts"},
            {"name": "Data", "path": "/data", "description": "Data directory"},
            {"name": "Datasets", "path": str(Path.home() / "datasets"), "description": "User datasets"}
        ]
