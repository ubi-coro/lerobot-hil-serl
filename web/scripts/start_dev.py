#!/usr/bin/env python3
"""
LeRobot Web GUI Development Launcher
====================================

This script simplifies the development workflow by automatically:
1. Starting the FastAPI backend
2. Starting the Vue.js frontend
3. Opening the browser with API docs
4. Handling graceful shutdown

Usage:
    cd web/scripts
    python start_dev.py

Features:
- FastAPI backend by default (modern async)
- Cross-platform support (Windows/Linux/macOS)
- Automatic dependency installation
- Graceful shutdown (Ctrl+C)
- Browser auto-opening with API docs
- Process management

Note: Legacy Flask backend removed; this launcher now always uses FastAPI.
"""

import subprocess
import sys
import os
import time
import webbrowser
import signal
import platform
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


class ProcessManager:
    """Manages backend and frontend processes"""
    
    def __init__(self):
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
        """Start the FastAPI backend"""
        try:
            backend_dir = web_dir / "backend_fastapi"
            
            print(f"{Colors.BLUE}� Starting FastAPI backend...{Colors.RESET}")
            
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
                "--host", "0.0.0.0", "--port", "8000", "--reload"
            ], cwd=backend_dir)
            
            self.processes.append(self.backend_process)
            print(f"{Colors.GREEN}✅ FastAPI backend started (PID: {self.backend_process.pid}){Colors.RESET}")
            return True
            
        except Exception as e:
            print(f"{Colors.RED}❌ Failed to start FastAPI backend: {e}{Colors.RESET}")
            return False
    
    def start_frontend(self, frontend_dir: Path) -> bool:
        """Start the Vue.js frontend"""
        try:
            print(f"{Colors.CYAN}🎨 Starting frontend...{Colors.RESET}")
            
            # Check if package.json exists
            package_file = frontend_dir / "package.json"
            if not package_file.exists():
                print(f"{Colors.RED}❌ Frontend package.json not found at: {package_file}{Colors.RESET}")
                return False
            
            # Check if node_modules exists (dependencies installed)
            node_modules = frontend_dir / "node_modules"
            if not node_modules.exists():
                print(f"{Colors.YELLOW}⚠️  Node modules not found. Installing dependencies...{Colors.RESET}")
                npm_install = subprocess.run(["npm", "install"], cwd=frontend_dir)
                if npm_install.returncode != 0:
                    print(f"{Colors.RED}❌ Failed to install frontend dependencies{Colors.RESET}")
                    return False
            
            # Start frontend process
            npm_cmd = "npm.cmd" if platform.system() == "Windows" else "npm"
            self.frontend_process = subprocess.Popen([
                npm_cmd, "run", "dev"
            ], cwd=frontend_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes.append(self.frontend_process)
            print(f"{Colors.GREEN}✅ Frontend started (PID: {self.frontend_process.pid}){Colors.RESET}")
            return True
            
        except Exception as e:
            print(f"{Colors.RED}❌ Failed to start frontend: {e}{Colors.RESET}")
            return False
    
    def wait_for_services(self, backend_url: str = "http://localhost:8000", 
                         frontend_url: str = "http://localhost:5173") -> bool:
        """Wait for services to be ready"""
        try:
            import requests
        except ImportError:
            print(f"{Colors.YELLOW}⚠️  'requests' not available. Skipping health checks{Colors.RESET}")
            time.sleep(5)
            return True
        
        print(f"{Colors.YELLOW}⏳ Waiting for services to start...{Colors.RESET}")
        
        # Wait for FastAPI backend
        backend_ready = False
        for i in range(30):  # 30 seconds timeout
            try:
                response = requests.get(f"{backend_url}/api/robot/status", timeout=2)
                if response.status_code == 200:
                    print(f"{Colors.GREEN}✅ FastAPI backend is ready{Colors.RESET}")
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


def print_banner():
    """Print startup banner"""
    banner = f"""
{Colors.BOLD}{Colors.CYAN}
╔══════════════════════════════════════════════════════════════╗
║                 🤖 LeRobot Web GUI Development                ║
║                     FastAPI Launcher                         ║
╚══════════════════════════════════════════════════════════════╝
{Colors.RESET}
"""
    print(banner)


def print_status(backend_url: str, frontend_url: str):
    """Print service status"""
    status = f"""
{Colors.BOLD}🌐 FastAPI Development Environment Ready!{Colors.RESET}

{Colors.BLUE}� Backend (FastAPI):{Colors.RESET}  {backend_url}
{Colors.CYAN}🎨 Frontend (Vue.js):{Colors.RESET}   {frontend_url}

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

{Colors.BOLD}💡 Tips:{Colors.RESET}
- Open browser dev tools (F12) to see debug info
- Backend logs show in this terminal
- Frontend auto-reloads on file changes
- Use {Colors.BOLD}Ctrl+C{Colors.RESET} to stop both services
- For Flask backend, use start_dev_advanced.py --backend flask

{Colors.GREEN}🚀 Ready for FastAPI development!{Colors.RESET}
"""
    print(status)


def main():
    """Main entry point"""
    print_banner()
    
    # Get paths
    script_dir = Path(__file__).parent
    web_dir = script_dir.parent
    backend_dir = web_dir / "backend_fastapi"
    frontend_dir = web_dir / "frontend"
    
    print(f"{Colors.BOLD}📁 Project Structure:{Colors.RESET}")
    print(f"   Web Directory: {web_dir}")
    print(f"   Backend:       {backend_dir} (FastAPI)")
    print(f"   Frontend:      {frontend_dir}")
    print()
    
    # Validate directories
    if not backend_dir.exists():
        print(f"{Colors.RED}❌ FastAPI backend directory not found: {backend_dir}{Colors.RESET}")
        # Legacy Flask backend removed
        sys.exit(1)
        
    if not frontend_dir.exists():
        print(f"{Colors.RED}❌ Frontend directory not found: {frontend_dir}{Colors.RESET}")
        sys.exit(1)
    
    # Initialize process manager
    pm = ProcessManager()
    
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
            print(f"{Colors.RED}❌ Failed to start FastAPI backend. Exiting.{Colors.RESET}")
            sys.exit(1)
        
        # Wait a moment for backend to initialize
        time.sleep(3)
        
        # Start frontend
        if not pm.start_frontend(frontend_dir):
            print(f"{Colors.RED}❌ Failed to start frontend. Stopping backend.{Colors.RESET}")
            pm.stop_all()
            sys.exit(1)
        
        # Wait for services to be ready
        backend_url = "http://localhost:8000"
        frontend_url = "http://localhost:5173"
        
        pm.wait_for_services(backend_url, frontend_url)
        
        # Open browser
        print(f"{Colors.CYAN}🌐 Opening browser...{Colors.RESET}")
        try:
            webbrowser.open(frontend_url)
            # Also open API docs for FastAPI
            time.sleep(2)
            webbrowser.open(f"{backend_url}/api/docs")
        except Exception as e:
            print(f"{Colors.YELLOW}⚠️  Could not auto-open browser: {e}{Colors.RESET}")
            print(f"   Please manually open: {frontend_url}")
            print(f"   API docs available at: {backend_url}/api/docs")
        
        # Print status
        print_status(backend_url, frontend_url)
        
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
    
    # Check if requests is available (for service health checks)
    try:
        import requests
    except ImportError:
        print(f"{Colors.YELLOW}⚠️  'requests' not installed. Service health checks disabled.{Colors.RESET}")
        print(f"   Install with: pip install requests")
        print()
    
    main()
