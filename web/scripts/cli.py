#!/usr/bin/env python3
"""
LeRobot Web GUI Command Line Interface
======================================

Provides simple CLI commands for starting the LeRobot Web GUI,
inspired by LeLab's user experience but maintaining integrated architecture.

Commands:
- lerobot-gui           : Start production GUI (FastAPI + Frontend)
- lerobot-gui-dev       : Start development environment with advanced options
- lerobot-gui-backend   : Start only FastAPI backend
- lerobot-gui-frontend  : Start only Vue.js frontend

Features:
- Auto dependency management
- Desktop shortcut creation
- Cross-platform support
- Graceful error handling
"""

import click
import sys
import subprocess
import webbrowser
import time
from pathlib import Path
from typing import Optional

# Import the existing launchers (now in same directory)
from .start_dev import main as start_dev_main, ProcessManager


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


def print_lerobot_banner():
    """Print LeRobot GUI banner"""
    banner = f"""
{Colors.BOLD}{Colors.CYAN}
╔════════════════════════════════════════════════════════════════════════╗
║                     🤖 LeRobot Web GUI                                 ║
║                   Command Line Interface                               ║
║                                                                        ║
║   Modern FastAPI + Vue.js interface for LeRobot Framework             ║
╚════════════════════════════════════════════════════════════════════════╝
{Colors.RESET}
"""
    print(banner)


def get_web_dir() -> Path:
    """Get the web directory path"""
    return Path(__file__).parent.parent


def check_dependencies(require_node: bool | None = None) -> bool:
    """Check required dependencies.

    If require_node is None, we require Node/npm only when the built frontend
    is missing (no web/frontend/dist/index.html). This lets production start
    without Node when assets are already built.
    """
    web_dir = get_web_dir()
    
    # Check backend
    backend_dir = web_dir / "backend_fastapi"
    if not backend_dir.exists():
        print(f"{Colors.RED}❌ FastAPI backend not found at: {backend_dir}{Colors.RESET}")
        return False
    
    # Check frontend
    frontend_dir = web_dir / "frontend"
    if not frontend_dir.exists():
        print(f"{Colors.RED}❌ Vue.js frontend not found at: {frontend_dir}{Colors.RESET}")
        return False
    
    # Decide whether Node is required
    dist_index = frontend_dir / 'dist' / 'index.html'
    if require_node is None:
        require_node = not dist_index.exists()
    
    # Check if npm is available only when needed
    if require_node:
        try:
            subprocess.run(["npm", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"{Colors.RED}❌ npm not found. Please install Node.js or build the frontend once (npm install && npm run build){Colors.RESET}")
            return False
    
    print(f"{Colors.GREEN}✅ Dependencies check passed{Colors.RESET}")
    return True


@click.group()
def cli():
    """LeRobot Web GUI - Modern interface for robot control and monitoring"""
    pass


@cli.command()
@click.option('--port', default=5000, help='Backend port (default: 5000)')
@click.option('--host', default='0.0.0.0', help='Backend host (default: 0.0.0.0)')
@click.option('--no-browser', is_flag=True, help='Don\'t auto-open browser')
def gui(port: int, host: str, no_browser: bool):
    """Start the LeRobot Web GUI (production mode)"""
    print_lerobot_banner()
    print(f"{Colors.BOLD}🚀 Starting LeRobot Web GUI{Colors.RESET}")
    
    if not check_dependencies():
        sys.exit(1)
    
    # Use the existing start_dev.py launcher
    try:
        start_dev_main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}🛑 GUI stopped by user{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}❌ Error starting GUI: {e}{Colors.RESET}")
        sys.exit(1)


@cli.command()
@click.option('--no-browser', is_flag=True, help='Don\'t auto-open browser')
def dev(no_browser: bool):
    """Start development environment (FastAPI + frontend)"""
    print_lerobot_banner()
    print(f"{Colors.BOLD}🔧 Starting Development Environment (fastapi){Colors.RESET}")

    if not check_dependencies(require_node=True):
        sys.exit(1)

    try:
        start_dev_main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}🛑 Development environment stopped by user{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}❌ Error starting development environment: {e}{Colors.RESET}")
        sys.exit(1)


@cli.command()
@click.option('--port', default=5000, help='Backend port (default: 5000)')
@click.option('--host', default='0.0.0.0', help='Backend host (default: 0.0.0.0)')
def backend(port: int, host: str):
    """Start only the FastAPI backend server"""
    print_lerobot_banner()
    print(f"{Colors.BOLD}📡 Starting FastAPI Backend Only{Colors.RESET}")
    
    web_dir = get_web_dir()
    backend_dir = web_dir / "backend_fastapi"
    
    if not backend_dir.exists():
        print(f"{Colors.RED}❌ FastAPI backend not found at: {backend_dir}{Colors.RESET}")
        sys.exit(1)
    
    try:
        print(f"{Colors.BLUE}🚀 Starting FastAPI backend on {host}:{port}{Colors.RESET}")
        subprocess.run([
            sys.executable, "-m", "uvicorn", "main:socket_app",
            "--host", host, "--port", str(port), "--reload"
        ], cwd=backend_dir)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}🛑 Backend stopped by user{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}❌ Error starting backend: {e}{Colors.RESET}")
        sys.exit(1)


@cli.command()
@click.option('--port', default=5173, help='Frontend port (default: 5173)')
@click.option('--no-browser', is_flag=True, help='Don\'t auto-open browser')
def frontend(port: int, no_browser: bool):
    """Start only the Vue.js frontend development server"""
    print_lerobot_banner()
    print(f"{Colors.BOLD}🎨 Starting Vue.js Frontend Only{Colors.RESET}")
    
    web_dir = get_web_dir()
    frontend_dir = web_dir / "frontend"
    
    if not frontend_dir.exists():
        print(f"{Colors.RED}❌ Frontend not found at: {frontend_dir}{Colors.RESET}")
        sys.exit(1)
    
    # Check and install frontend dependencies
    node_modules = frontend_dir / "node_modules"
    if not node_modules.exists():
        print(f"{Colors.YELLOW}📦 Installing frontend dependencies...{Colors.RESET}")
        try:
            subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)
            print(f"{Colors.GREEN}✅ Frontend dependencies installed{Colors.RESET}")
        except subprocess.CalledProcessError:
            print(f"{Colors.RED}❌ Failed to install frontend dependencies{Colors.RESET}")
            sys.exit(1)
    
    try:
        print(f"{Colors.CYAN}🎨 Starting frontend on port {port}{Colors.RESET}")
        
        # Auto-open browser unless disabled
        if not no_browser:
            time.sleep(2)  # Give frontend time to start
            frontend_url = f"http://localhost:{port}"
            print(f"{Colors.CYAN}🌐 Opening browser: {frontend_url}{Colors.RESET}")
            webbrowser.open(frontend_url)
        
        subprocess.run(["npm", "run", "dev"], cwd=frontend_dir)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}🛑 Frontend stopped by user{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}❌ Error starting frontend: {e}{Colors.RESET}")
        sys.exit(1)


@cli.command()
@click.option('--force', is_flag=True, help='Force recreate existing shortcut')
def create_shortcut(force: bool):
    """Create a desktop shortcut for LeRobot GUI"""
    print_lerobot_banner()
    print(f"{Colors.BOLD}🖥️ Creating Desktop Shortcut{Colors.RESET}")
    
    try:
        from .create_shortcut import create_desktop_shortcut
        success = create_desktop_shortcut(force=force)
        
        if success:
            print(f"{Colors.GREEN}✅ Desktop shortcut created successfully{Colors.RESET}")
            print(f"{Colors.CYAN}💡 You can now start LeRobot GUI from your desktop{Colors.RESET}")
        else:
            print(f"{Colors.YELLOW}⚠️ Desktop shortcut creation failed or skipped{Colors.RESET}")
    except ImportError:
        print(f"{Colors.YELLOW}⚠️ Shortcut creation not yet implemented{Colors.RESET}")
        print(f"{Colors.CYAN}💡 You can manually create a shortcut to: python -m lerobot.web.cli gui{Colors.RESET}")


@cli.command()
def status():
    """Show LeRobot Web GUI status and information"""
    print_lerobot_banner()
    
    web_dir = get_web_dir()
    
    print(f"{Colors.BOLD}📊 LeRobot Web GUI Status{Colors.RESET}")
    print(f"   Web Directory: {web_dir}")
    print(f"   FastAPI Backend: {'✅' if (web_dir / 'backend_fastapi').exists() else '❌'}")
    print(f"   Vue.js Frontend: {'✅' if (web_dir / 'frontend').exists() else '❌'}")
    
    # Check if services are running
    try:
        import requests
        try:
            response = requests.get("http://localhost:5000/api/robot/status", timeout=2)
            backend_status = "🟢 Running" if response.status_code == 200 else "🟡 Issues"
        except:
            backend_status = "🔴 Not running"
            
        try:
            response = requests.get("http://localhost:5173", timeout=2)
            frontend_status = "🟢 Running" if response.status_code == 200 else "🟡 Issues"
        except:
            frontend_status = "🔴 Not running"
            
        print(f"   Backend Status: {backend_status}")
        print(f"   Frontend Status: {frontend_status}")
        
    except ImportError:
        print(f"   Status Check: {Colors.YELLOW}⚠️ 'requests' not available{Colors.RESET}")
    
    print(f"\n{Colors.BOLD}🚀 Quick Commands:{Colors.RESET}")
    print(f"   lerobot-gui              # Start GUI")
    print(f"   lerobot-gui-dev          # Development mode")
    print(f"   lerobot-gui-backend      # Backend only")
    print(f"   lerobot-gui-frontend     # Frontend only")


if __name__ == '__main__':
    cli()


# Individual command wrappers for entry points
def gui_main():
    """Entry point for lerobot-gui command"""
    gui.main(standalone_mode=False)

def dev_main():
    """Entry point for lerobot-gui-dev command"""
    dev.main(standalone_mode=False)

def backend_main():
    """Entry point for lerobot-gui-backend command"""
    backend.main(standalone_mode=False)

def frontend_main():
    """Entry point for lerobot-gui-frontend command"""
    frontend.main(standalone_mode=False)

def create_shortcut_main():
    """Entry point for lerobot-gui-shortcut command"""
    create_shortcut.main(standalone_mode=False)

def status_main():
    """Entry point for lerobot-gui-status command"""
    status.main(standalone_mode=False)
