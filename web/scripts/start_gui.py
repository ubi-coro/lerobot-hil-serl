#!/usr/bin/env python3
"""
LeRobot Web GUI Production Launcher
===================================

Starts the FastAPI backend (no auto-reload) and serves the built Vue frontend
from web/frontend/dist. Builds the frontend automatically the first time (or
when --force-build is provided) and opens the browser.

Usage:
  python start_gui.py            # normal launch
  python start_gui.py --force-build  # rebuild frontend

Differences vs start_dev.py:
  - No Vite dev server / HMR
  - Uvicorn without --reload
  - Static optimized assets served (must run build once)
  - Quieter output
"""

from __future__ import annotations
import subprocess, sys, os, time, webbrowser, signal, platform
from pathlib import Path
from typing import Optional


class Colors:
    RESET='\033[0m'; BOLD='\033[1m'; GREEN='\033[92m'; BLUE='\033[94m'; YELLOW='\033[93m'; RED='\033[91m'; CYAN='\033[96m'


def banner():
    print(f"{Colors.BOLD}{Colors.CYAN}\n╔══════════════════════════════════════════════╗\n║           🤖 LeRobot GUI (Production)         ║\n╚══════════════════════════════════════════════╝{Colors.RESET}")


def run(cmd, cwd: Path|None=None, check=True):
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if check and proc.returncode!=0:
        print(proc.stdout)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return proc


def ensure_frontend_build(frontend_dir: Path, force: bool=False):
    """Ensure production frontend assets exist.

    Behavior when npm is missing:
      - If dist already exists: skip rebuild (warn once) and continue.
      - If dist missing: create minimal placeholder index.html so backend still launches.
    """
    dist = frontend_dir / 'dist'
    npm_available = False
    try:
        subprocess.run(["npm","--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        npm_available = True
    except Exception:
        npm_available = False

    # Decide if we need a build
    need_build = force or not dist.exists()
    if need_build and npm_available:
        print(f"{Colors.BLUE}🔧 Building frontend (vite build)...{Colors.RESET}")
        if not (frontend_dir/'node_modules').exists():
            print(f"{Colors.YELLOW}📦 Installing frontend dependencies...{Colors.RESET}")
            run(["npm","install"], cwd=frontend_dir)
        run(["npm","run","build"], cwd=frontend_dir)
        print(f"{Colors.GREEN}✅ Frontend build complete{Colors.RESET}")
        return

    if need_build and not npm_available:
        # Create placeholder
        print(f"{Colors.YELLOW}⚠️ npm not found; creating placeholder frontend (limited UI).{Colors.RESET}")
        dist.mkdir(parents=True, exist_ok=True)
        placeholder = dist / 'index.html'
        if not placeholder.exists() or force:
            placeholder.write_text(f"""<!DOCTYPE html><html><head><meta charset='utf-8'/><title>LeRobot GUI - Placeholder</title>
<style>body{{font-family:Arial,sans-serif;padding:2rem;background:#111;color:#eee;max-width:780px;margin:auto;line-height:1.5}}code{{background:#222;padding:2px 4px;border-radius:4px}}h1{{color:#6cf}}a{{color:#6cf}}</style></head>
<body><h1>LeRobot GUI Backend Running</h1><p>The production frontend assets are not built.</p>
<p>To enable the full interface:</p>
<ol><li>Install Node.js & npm</li><li>Run: <code>cd {frontend_dir}</code></li><li><code>npm install</code></li><li><code>npm run build</code></li><li>Relaunch the GUI</li></ol>
<p>FastAPI docs: <a href='http://localhost:8000/api/docs'>http://localhost:8000/api/docs</a></p>
<p>This placeholder file was generated automatically because npm was not detected.</p></body></html>""")
        print(f"{Colors.YELLOW}⚠️ Placeholder frontend created at {placeholder}{Colors.RESET}")
        return

    if not need_build and not npm_available:
        print(f"{Colors.YELLOW}⚠️ npm not found; using existing built assets (no rebuild).{Colors.RESET}")
        return

    # No build needed and npm available
    print(f"{Colors.GREEN}✅ Frontend build up-to-date{Colors.RESET}")


def start_backend(backend_dir: Path):
    print(f"{Colors.BLUE}🚀 Starting FastAPI backend (production mode)...{Colors.RESET}")
    env = os.environ.copy()
    env['LEROBOT_GUI_MODE'] = 'production'
    env['LEROBOT_GUI_SERVE_FRONTEND'] = '1'  # Serve built frontend from dist/
    # No --reload here
    return subprocess.Popen([
        sys.executable, "-m", "uvicorn", "main:socket_app", "--host","0.0.0.0","--port","8000"
    ], cwd=backend_dir, env=env)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Launch LeRobot GUI (production mode)")
    parser.add_argument('--force-build', action='store_true', help='Force rebuild of frontend assets')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser automatically')
    parser.add_argument('--open-docs', action='store_true', help='Also open API docs')
    args = parser.parse_args()

    banner()
    web_dir = Path(__file__).parent.parent
    backend_dir = web_dir / 'backend_fastapi'
    frontend_dir = web_dir / 'frontend'
    if not backend_dir.exists() or not (backend_dir/'main.py').exists():
        print(f"{Colors.RED}❌ Backend not found at {backend_dir}{Colors.RESET}"); sys.exit(1)
    if not frontend_dir.exists():
        print(f"{Colors.RED}❌ Frontend not found at {frontend_dir}{Colors.RESET}"); sys.exit(1)

    try:
        ensure_frontend_build(frontend_dir, force=args.force_build)
    except Exception as e:
        print(f"{Colors.RED}❌ Frontend build failed: {e}{Colors.RESET}"); sys.exit(1)

    # Start backend
    proc = start_backend(backend_dir)

    def shutdown(signame):
        print(f"\n{Colors.YELLOW}🛑 Received {signame}, shutting down...{Colors.RESET}")
        if proc.poll() is None:
            proc.terminate()
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired: proc.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, lambda *_: shutdown('SIGINT'))
    if platform.system() != 'Windows':
        signal.signal(signal.SIGTERM, lambda *_: shutdown('SIGTERM'))

    # Light wait for backend
    time.sleep(2)
    url_app = 'http://localhost:8000'
    if not args.no_browser:
        try:
            webbrowser.open(url_app)
            if args.open_docs:
                time.sleep(1)
                webbrowser.open('http://localhost:8000/api/docs')
        except Exception as e:
            print(f"{Colors.YELLOW}⚠️ Could not open browser automatically: {e}{Colors.RESET}")

    print(f"{Colors.GREEN}✅ LeRobot GUI running at {url_app}{Colors.RESET}")

    # Wait on backend
    try:
        proc.wait()
    except KeyboardInterrupt:
        shutdown('KeyboardInterrupt')


if __name__ == '__main__':
    if sys.version_info < (3,8):
        print(f"{Colors.RED}Python 3.8+ required{Colors.RESET}"); sys.exit(1)
    main()
