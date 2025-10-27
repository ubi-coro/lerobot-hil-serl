#!/usr/bin/env python3
"""
LeRobot Web GUI Advanced Development Launcher
=============================================

Enhanced version supporting both Flask and FastAPI backends.
Perfect for testing the FastAPI migration while keeping Flask as backup.

Usage:
    cd web/scripts
    python start_dev_advanced.py --backend flask     # Use Flask (current)
    python start_dev_advanced.py --backend fastapi   # Use FastAPI (new)
    python start_dev_advanced.py                     # Use FastAPI by default

Features:
- Backend selection (Flask/FastAPI)
- Automatic dependency installation
- Cross-platform support
- Health checks and monitoring
- Development tools integration
"""

import subprocess
import sys
import os
import time
import webbrowser
import signal
import platform
import argparse
from pathlib import Path
from typing import Optional, List


class Colors:
    """ANSI color codes for terminal output"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'


class AdvancedProcessManager:
    """Enhanced process manager supporting multiple backend types"""
    
    def __init__(self, backend_type: str = "fastapi"):
        self.backend_type = backend_type
        self.backend_process: Optional[subprocess.Popen] = None
        self.frontend_process: Optional[subprocess.Popen] = None
        self.processes: List[subprocess.Popen] = []
        
    def install_fastapi_dependencies(self, backend_fastapi_dir: Path) -> bool:
        """Install FastAPI dependencies if needed"""
        requirements_file = backend_fastapi_dir / "requirements.txt"
        
        if not requirements_file.exists():
            print(f"{Colors.YELLOW}⚠️  FastAPI requirements.txt not found{Colors.RESET}")
            return True  # Continue anyway
        
        try:
            print(f"{Colors.BLUE}📦 Checking FastAPI dependencies...{Colors.RESET}")
            
            # Check if FastAPI is installed
            check_cmd = [sys.executable, "-c", "import fastapi; print(fastapi.__version__)"]
            result = subprocess.run(check_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"{Colors.GREEN}✅ FastAPI already installed (v{result.stdout.strip()}){Colors.RESET}")
                return True
            
            print(f"{Colors.YELLOW}📦 Installing FastAPI dependencies...{Colors.RESET}")
            install_cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
            install_result = subprocess.run(install_cmd)
            
            if install_result.returncode == 0:
                print(f"{Colors.GREEN}✅ FastAPI dependencies installed{Colors.RESET}")
                return True
            else:
                print(f"{Colors.RED}❌ Failed to install FastAPI dependencies{Colors.RESET}")
                return False
                
        except Exception as e:
            print(f"{Colors.RED}❌ Error installing dependencies: {e}{Colors.RESET}")
            return False
    
    def start_backend(self, web_dir: Path) -> bool:
        """Start the selected backend (Flask or FastAPI)"""
        try:
            if self.backend_type == "fastapi":
                return self._start_fastapi_backend(web_dir)
            else:
                return self._start_flask_backend(web_dir)
        except Exception as e:
            print(f"{Colors.RED}❌ Failed to start {self.backend_type} backend: {e}{Colors.RESET}")
            return False
    
    def _start_fastapi_backend(self, web_dir: Path) -> bool:
        """Start FastAPI backend"""
        backend_dir = web_dir / "backend_fastapi"
        
        print(f"{Colors.MAGENTA}🚀 Starting FastAPI backend...{Colors.RESET}")
        
        # Check if main.py exists
        main_file = backend_dir / "main.py"
        if not main_file.exists():
            print(f"{Colors.RED}❌ FastAPI main.py not found at: {main_file}{Colors.RESET}")
            return False
        
        # Install dependencies
        if not self.install_fastapi_dependencies(backend_dir):
            return False
        
        # Start FastAPI with uvicorn
        self.backend_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "main:socket_app", 
            "--host", "0.0.0.0", "--port", "5000", "--reload"
        ], cwd=backend_dir)  # Remove stdout/stderr capture to see output
        
        self.processes.append(self.backend_process)
        print(f"{Colors.GREEN}✅ FastAPI backend started (PID: {self.backend_process.pid}){Colors.RESET}")
        return True
    
    def _start_flask_backend(self, web_dir: Path) -> bool:
        """Start Flask backend"""
        backend_dir = web_dir / "backend"
        
        print(f"{Colors.BLUE}📡 Starting Flask backend...{Colors.RESET}")
        
        # Check if app.py exists
        app_file = backend_dir / "app.py"
        if not app_file.exists():
            print(f"{Colors.RED}❌ Flask app.py not found at: {app_file}{Colors.RESET}")
            return False
        
        # Start Flask
        self.backend_process = subprocess.Popen([
            sys.executable, "app.py", "--host", "0.0.0.0", "--port", "5000"
        ], cwd=backend_dir)  # Remove stdout/stderr capture to see output
        
        self.processes.append(self.backend_process)
        print(f"{Colors.GREEN}✅ Flask backend started (PID: {self.backend_process.pid}){Colors.RESET}")
        return True
    
    def start_frontend(self, frontend_dir: Path) -> bool:
        """Start the Vue.js frontend"""
        try:
            print(f"{Colors.CYAN}🎨 Starting frontend...{Colors.RESET}")
            
            # Check if package.json exists
            package_file = frontend_dir / "package.json"
            if not package_file.exists():
                print(f"{Colors.RED}❌ Frontend package.json not found at: {package_file}{Colors.RESET}")
                return False
            
            # Check if node_modules exists
            node_modules = frontend_dir / "node_modules"
            if not node_modules.exists():
                print(f"{Colors.YELLOW}⚠️  Node modules not found. Installing dependencies...{Colors.RESET}")
                npm_install = subprocess.run(["npm", "install"], cwd=frontend_dir)
                if npm_install.returncode != 0:
                    print(f"{Colors.RED}❌ Failed to install frontend dependencies{Colors.RESET}")
                    return False
            
            # Start frontend
            npm_cmd = "npm.cmd" if platform.system() == "Windows" else "npm"
            self.frontend_process = subprocess.Popen([
                npm_cmd, "run", "dev"
            ], cwd=frontend_dir)  # Remove stdout/stderr capture to see output
            
            self.processes.append(self.frontend_process)
            print(f"{Colors.GREEN}✅ Frontend started (PID: {self.frontend_process.pid}){Colors.RESET}")
            return True
            
        except Exception as e:
            print(f"{Colors.RED}❌ Failed to start frontend: {e}{Colors.RESET}")
            return False
    
    def wait_for_services(self, backend_url: str = "http://localhost:5000", 
                         frontend_url: str = "http://localhost:5173") -> bool:
        """Wait for services to be ready with enhanced checks"""
        try:
            import requests
        except ImportError:
            print(f"{Colors.YELLOW}⚠️  'requests' not available. Skipping health checks{Colors.RESET}")
            time.sleep(5)
            return True
        
        print(f"{Colors.YELLOW}⏳ Waiting for services to start...{Colors.RESET}")
        
        # Wait for backend
        backend_ready = False
        for i in range(30):  # 30 seconds timeout
            try:
                if self.backend_type == "fastapi":
                    # FastAPI health check
                    response = requests.get(f"{backend_url}/api/robot/status", timeout=2)
                else:
                    # Flask health check  
                    response = requests.get(f"{backend_url}/api/robot/status", timeout=2)
                
                if response.status_code == 200:
                    print(f"{Colors.GREEN}✅ {self.backend_type.title()} backend is ready{Colors.RESET}")
                    backend_ready = True
                    break
            except:
                time.sleep(1)
        
        if not backend_ready:
            print(f"{Colors.YELLOW}⚠️  Backend may not be fully ready yet{Colors.RESET}")
        
        # Frontend typically takes a bit longer
        time.sleep(3)
        print(f"{Colors.GREEN}✅ Frontend should be ready{Colors.RESET}")
        return True
    
    def stop_all(self):
        """Stop all managed processes"""
        print(f"{Colors.YELLOW}🛑 Stopping services...{Colors.RESET}")
        
        for process in self.processes:
            if process and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                    print(f"{Colors.GREEN}✅ Process {process.pid} stopped{Colors.RESET}")
                except subprocess.TimeoutExpired:
                    print(f"{Colors.RED}⚠️  Force killing process {process.pid}{Colors.RESET}")
                    process.kill()
                except Exception as e:
                    print(f"{Colors.RED}❌ Error stopping process: {e}{Colors.RESET}")
        
        print(f"{Colors.GREEN}✅ All services stopped{Colors.RESET}")


def print_advanced_banner(backend_type: str):
    """Print enhanced startup banner"""
    backend_emoji = "🚀" if backend_type == "fastapi" else "📡"
    banner = f"""
{Colors.BOLD}{Colors.MAGENTA}
╔════════════════════════════════════════════════════════════════════════╗
║                   🤖 LeRobot Web GUI Development                       ║
║                    Advanced FastTrack Launcher                        ║
║                                                                        ║
║  Backend: {backend_emoji} {backend_type.upper():^8} {'(Modern Async)' if backend_type == 'fastapi' else '(Current Flask)':^15}                     ║
╚════════════════════════════════════════════════════════════════════════╝
{Colors.RESET}
"""
    print(banner)


def print_advanced_status(backend_type: str, backend_url: str, frontend_url: str):
    """Print enhanced service status"""
    status = f"""
{Colors.BOLD}🌐 Advanced Development Environment Ready!{Colors.RESET}

{Colors.MAGENTA}🔧 Backend ({backend_type.title()}):{Colors.RESET}  {backend_url}
{Colors.CYAN}🎨 Frontend (Vue.js):{Colors.RESET}        {frontend_url}

{Colors.YELLOW}📚 FastAPI Features:{Colors.RESET}
- Interactive API docs: {backend_url}/api/docs
- Alternative docs:     {backend_url}/api/redoc
- Async performance for camera streaming
- Automatic request/response validation

{Colors.YELLOW}📋 Available Features:{Colors.RESET}
- Robot connection (mock mode when no hardware)
- Simplified teleoperation controls
- Enhanced emergency stop (Space key)
- Real-time WebSocket communication
- Performance monitoring

{Colors.BOLD}💡 Development Tips:{Colors.RESET}
- Switch backends with --backend flag
- Open browser dev tools (F12) for debug info
- Backend logs appear in this terminal
- Frontend auto-reloads on file changes
- Use {Colors.BOLD}Ctrl+C{Colors.RESET} to stop both services

{Colors.GREEN}🚀 Ready for Phase 2 development!{Colors.RESET}
"""
    print(status)


def main():
    """Enhanced main entry point with backend selection"""
    parser = argparse.ArgumentParser(description="LeRobot Advanced Development Launcher")
    parser.add_argument("--backend", choices=["flask", "fastapi"], default="fastapi",
                       help="Backend type to use (default: fastapi)")
    args = parser.parse_args()
    
    print_advanced_banner(args.backend)
    
    # Get paths
    script_dir = Path(__file__).parent
    web_dir = script_dir.parent
    backend_dir = web_dir / ("backend_fastapi" if args.backend == "fastapi" else "backend")
    frontend_dir = web_dir / "frontend"
    
    print(f"{Colors.BOLD}📁 Project Structure:{Colors.RESET}")
    print(f"   Web Directory: {web_dir}")
    print(f"   Backend:       {backend_dir} ({args.backend})")
    print(f"   Frontend:      {frontend_dir}")
    print()
    
    # Validate directories
    if not backend_dir.exists():
        print(f"{Colors.RED}❌ Backend directory not found: {backend_dir}{Colors.RESET}")
        if args.backend == "fastapi":
            print(f"{Colors.YELLOW}💡 Tip: Run with --backend flask to use existing Flask backend{Colors.RESET}")
        sys.exit(1)
        
    if not frontend_dir.exists():
        print(f"{Colors.RED}❌ Frontend directory not found: {frontend_dir}{Colors.RESET}")
        sys.exit(1)
    
    # Initialize process manager
    pm = AdvancedProcessManager(args.backend)
    
    # Set up signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print(f"\n{Colors.YELLOW}🛑 Received shutdown signal{Colors.RESET}")
        pm.stop_all()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    if platform.system() != "Windows":
        signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start backend
        if not pm.start_backend(web_dir):
            print(f"{Colors.RED}❌ Failed to start {args.backend} backend. Exiting.{Colors.RESET}")
            sys.exit(1)
        
        # Wait a moment for backend to initialize
        time.sleep(3)
        
        # Start frontend
        if not pm.start_frontend(frontend_dir):
            print(f"{Colors.RED}❌ Failed to start frontend. Stopping backend.{Colors.RESET}")
            pm.stop_all()
            sys.exit(1)
        
        # Wait for services to be ready
        backend_url = "http://localhost:5000"
        frontend_url = "http://localhost:5173"
        
        pm.wait_for_services(backend_url, frontend_url)
        
        # Open browser
        print(f"{Colors.CYAN}🌐 Opening browser...{Colors.RESET}")
        try:
            webbrowser.open(frontend_url)
            if args.backend == "fastapi":
                # Also open API docs
                time.sleep(2)
                webbrowser.open(f"{backend_url}/api/docs")
        except Exception as e:
            print(f"{Colors.YELLOW}⚠️  Could not auto-open browser: {e}{Colors.RESET}")
            print(f"   Please manually open: {frontend_url}")
        
        # Print status
        print_advanced_status(args.backend, backend_url, frontend_url)
        
        # Wait for processes to complete (or Ctrl+C)
        try:
            if pm.backend_process:
                pm.backend_process.wait()
        except KeyboardInterrupt:
            pass  # Handled by signal handler
            
    except Exception as e:
        print(f"{Colors.RED}❌ Unexpected error: {e}{Colors.RESET}")
        pm.stop_all()
        sys.exit(1)


if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 7):
        print(f"{Colors.RED}❌ Python 3.7+ required. Current: {sys.version}{Colors.RESET}")
        sys.exit(1)
    
    main()
