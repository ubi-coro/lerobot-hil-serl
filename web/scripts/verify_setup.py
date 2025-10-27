#!/usr/bin/env python3
"""
LeRobot Web GUI Setup Verification
==================================

Verifies that the LeRobot Web GUI installation and CLI commands are working correctly.
This script checks dependencies, tests CLI imports, and validates the setup.
"""

import sys
import subprocess
import importlib
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'


def print_header():
    """Print verification header"""
    print(f"""
{Colors.BOLD}{Colors.CYAN}
╔════════════════════════════════════════════════════════════════════════╗
║                 🤖 LeRobot Web GUI Setup Verification                  ║
╚════════════════════════════════════════════════════════════════════════╝
{Colors.RESET}
""")


def check_python_version():
    """Check Python version"""
    print(f"{Colors.BOLD}🐍 Checking Python Version{Colors.RESET}")
    
    version = sys.version_info
    if version >= (3, 10):
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (Requires 3.10+)")
        return False


def check_core_dependencies():
    """Check core LeRobot dependencies"""
    print(f"\n{Colors.BOLD}📦 Checking Core Dependencies{Colors.RESET}")
    
    core_deps = [
        "torch",
        "numpy", 
        "opencv-python",
        "datasets"
    ]
    
    all_good = True
    for dep in core_deps:
        try:
            importlib.import_module(dep.replace('-', '_'))
            print(f"   ✅ {dep}")
        except ImportError:
            print(f"   ❌ {dep} (Missing)")
            all_good = False
    
    return all_good


def check_web_dependencies():
    """Check web GUI dependencies"""
    print(f"\n{Colors.BOLD}🌐 Checking Web Dependencies{Colors.RESET}")
    
    web_deps = {
        "fastapi": "FastAPI",
        "uvicorn": "Uvicorn", 
        "socketio": "python-socketio",
        "click": "Click",
        "requests": "Requests"
    }
    
    all_good = True
    for module, name in web_deps.items():
        try:
            importlib.import_module(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} (Install with: pip install -e .[web])")
            all_good = False
    
    return all_good


def check_cli_commands():
    """Check CLI command installation"""
    print(f"\n{Colors.BOLD}⚙️ Checking CLI Commands{Colors.RESET}")
    
    commands = [
        "lerobot-gui",
        "lerobot-gui-dev", 
        "lerobot-gui-backend",
        "lerobot-gui-frontend",
        "lerobot-gui-status"
    ]
    
    all_good = True
    for cmd in commands:
        try:
            result = subprocess.run([cmd, "--help"], capture_output=True, timeout=5)
            if result.returncode == 0:
                print(f"   ✅ {cmd}")
            else:
                print(f"   ❌ {cmd} (Command failed)")
                all_good = False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print(f"   ❌ {cmd} (Not found - run: pip install -e .[web])")
            all_good = False
    
    return all_good


def check_web_structure():
    """Check web directory structure"""
    print(f"\n{Colors.BOLD}📁 Checking Web Structure{Colors.RESET}")
    
    web_dir = Path(__file__).parent
    required_paths = [
        ("backend_fastapi", "FastAPI Backend"),
        ("frontend", "Vue.js Frontend"),
        ("scripts", "Development Scripts"),
        ("cli.py", "CLI Module")
    ]
    
    all_good = True
    for path, description in required_paths:
        full_path = web_dir / path
        if full_path.exists():
            print(f"   ✅ {description}")
        else:
            print(f"   ❌ {description} (Missing: {full_path})")
            all_good = False
    
    return all_good


def check_frontend_dependencies():
    """Check frontend dependencies"""
    print(f"\n{Colors.BOLD}🎨 Checking Frontend Dependencies{Colors.RESET}")
    
    frontend_dir = Path(__file__).parent / "frontend"
    
    if not frontend_dir.exists():
        print(f"   ❌ Frontend directory not found")
        return False
    
    # Check package.json
    package_json = frontend_dir / "package.json"
    if package_json.exists():
        print(f"   ✅ package.json")
    else:
        print(f"   ❌ package.json (Missing)")
        return False
    
    # Check node_modules
    node_modules = frontend_dir / "node_modules"
    if node_modules.exists():
        print(f"   ✅ node_modules (Dependencies installed)")
    else:
        print(f"   ⚠️ node_modules (Run: npm install in frontend/)")
        return False
    
    # Check npm availability
    try:
        subprocess.run(["npm", "--version"], capture_output=True, check=True)
        print(f"   ✅ npm")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"   ❌ npm (Install Node.js)")
        return False
    
    return True


def run_verification():
    """Run complete verification"""
    print_header()
    
    checks = [
        ("Python Version", check_python_version),
        ("Core Dependencies", check_core_dependencies),
        ("Web Dependencies", check_web_dependencies),
        ("CLI Commands", check_cli_commands),
        ("Web Structure", check_web_structure),
        ("Frontend Dependencies", check_frontend_dependencies)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"   ❌ Error in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print(f"\n{Colors.BOLD}📊 Verification Summary{Colors.RESET}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {name}")
    
    print(f"\n{Colors.BOLD}Result: {passed}/{total} checks passed{Colors.RESET}")
    
    if passed == total:
        print(f"""
{Colors.GREEN}{Colors.BOLD}🎉 All checks passed! LeRobot Web GUI is ready to use.{Colors.RESET}

{Colors.CYAN}Quick Start:{Colors.RESET}
   lerobot-gui                    # Start GUI
   lerobot-gui-shortcut          # Create desktop shortcut
   lerobot-gui-status            # Check system status
""")
        return True
    else:
        print(f"""
{Colors.YELLOW}{Colors.BOLD}⚠️ Some checks failed. Please fix the issues above.{Colors.RESET}

{Colors.CYAN}Common fixes:{Colors.RESET}
   pip install -e .[web]         # Install web dependencies
   cd frontend && npm install    # Install frontend dependencies
""")
        return False


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
