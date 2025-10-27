#!/usr/bin/env python3
"""
Desktop Shortcut Creation for LeRobot Web GUI
==============================================

Creates cross-platform desktop shortcuts for easy GUI access.
Supports Windows (.lnk), macOS (.app), and Linux (.desktop).
"""

import os
import sys
import platform
from pathlib import Path

try:
    import click  # optional
except Exception:  # pragma: no cover
    click = None


class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"


def get_desktop_path() -> Path:
    system = platform.system()
    if system == "Windows":
        d = Path.home() / "Desktop"
        return d if d.exists() else Path.home() / "OneDrive" / "Desktop"
    elif system == "Darwin":
        return Path.home() / "Desktop"
    else:
        d = Path.home() / "Desktop"
        return d if d.exists() else Path.home()


def create_windows_shortcut(force: bool = False) -> bool:
    try:
        desktop = get_desktop_path()
        web_dir = Path(__file__).resolve().parent.parent
        launcher_content = f"""@echo off
title LeRobot Web GUI
cd /d "{web_dir}"
where conda >nul 2>nul
if %ERRORLEVEL% == 0 (
  conda info --envs | find "lerobot" >nul 2>nul && call conda activate lerobot
)
python -m web.scripts.cli gui
if %ERRORLEVEL% NEQ 0 (
  echo.
  echo Error occurred. Press any key to close...
  pause >nul
)
"""
        bat_path = desktop / "LeRobot_GUI_Launcher.bat"
        if bat_path.exists() and not force:
            print(f"{Colors.YELLOW}⚠️ Launcher already exists: {bat_path}{Colors.RESET}")
            return False
        bat_path.write_text(launcher_content)
        print(f"{Colors.GREEN}✅ Windows launcher created: {bat_path}{Colors.RESET}")
        return True
    except Exception as e:
        print(f"{Colors.RED}❌ Error creating Windows shortcut: {e}{Colors.RESET}")
        return False


def create_macos_app(force: bool = False) -> bool:
    try:
        desktop = get_desktop_path()
        web_dir = Path(__file__).resolve().parent.parent
        app_path = desktop / "LeRobot GUI.app"
        if app_path.exists() and not force:
            print(f"{Colors.YELLOW}⚠️ App already exists: {app_path}{Colors.RESET}")
            return False
        contents = app_path / "Contents"
        macos = contents / "MacOS"
        resources = contents / "Resources"
        macos.mkdir(parents=True, exist_ok=True)
        resources.mkdir(exist_ok=True)
        (contents / "Info.plist").write_text(
            """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">
<plist version=\"1.0\"><dict>
<key>CFBundleExecutable</key><string>lerobot_gui</string>
<key>CFBundleIdentifier</key><string>com.lerobot.gui</string>
<key>CFBundleName</key><string>LeRobot GUI</string>
<key>CFBundleVersion</key><string>1.0</string>
<key>CFBundleShortVersionString</key><string>1.0</string>
<key>CFBundlePackageType</key><string>APPL</string>
</dict></plist>"""
        )
        script = macos / "lerobot_gui"
        script.write_text(
            f"""#!/bin/bash
cd "{web_dir}"
if command -v conda >/dev/null 2>&1; then
  conda info --envs | grep -q lerobot && source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate lerobot
fi
python -m web.scripts.cli gui
"""
        )
        os.chmod(script, 0o755)
        print(f"{Colors.GREEN}✅ macOS app created: {app_path}{Colors.RESET}")
        return True
    except Exception as e:
        print(f"{Colors.RED}❌ Error creating macOS app: {e}{Colors.RESET}")
        return False


def create_linux_desktop(force: bool = False) -> bool:
    try:
        desktop = get_desktop_path()
        web_dir = Path(__file__).resolve().parent.parent
        scripts_dir = web_dir / "scripts"
        launcher_script = scripts_dir / "lerobot_gui_launcher.sh"
        desktop_file = desktop / "lerobot-gui.desktop"

        # Ensure launcher script exists and is executable
        scripts_dir.mkdir(parents=True, exist_ok=True)
        if not launcher_script.exists():
            # Minimal fallback launcher if missing
            launcher_script.write_text(
                f"""#!/usr/bin/env bash
cd "{scripts_dir}"
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda env list | grep -q lerobot && conda activate lerobot || true
fi
python -m web.scripts.cli gui
"""
            )
        os.chmod(launcher_script, 0o755)

        if desktop_file.exists() and not force:
            print(f"{Colors.YELLOW}⚠️ Desktop file already exists: {desktop_file}{Colors.RESET}")
            print(f"{Colors.CYAN}💡 Use --force to recreate{Colors.RESET}")
            return False

        icon_path = web_dir / "media" / "lerobot-logo-light.png"
        icon_entry = str(icon_path) if icon_path.exists() else "applications-science"

        desktop_content = (
            "[Desktop Entry]\n"
            "Version=1.0\n"
            "Type=Application\n"
            "Name=LeRobot GUI\n"
            "Comment=FastAPI + Vue.js interface for robot control and dataset operations\n"
            f"Exec={launcher_script}\n"
            f"Path={scripts_dir}\n"
            f"Icon={icon_entry}\n"
            "Terminal=false\n"
            "Categories=Science;Development;Robotics;Education;\n"
            "StartupNotify=true\n"
        )

        desktop_file.write_text(desktop_content)
        os.chmod(desktop_file, 0o755)
        print(f"{Colors.GREEN}✅ Linux desktop file created: {desktop_file}{Colors.RESET}")

        try:
            apps_dir = Path.home() / ".local" / "share" / "applications"
            apps_dir.mkdir(parents=True, exist_ok=True)
            app_desktop = apps_dir / "lerobot-gui.desktop"
            app_desktop.write_text(desktop_content)
            print(f"{Colors.GREEN}✅ Added to applications menu: {app_desktop}{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.YELLOW}⚠️ Could not add to applications menu: {e}{Colors.RESET}")

        return True
    except Exception as e:
        print(f"{Colors.RED}❌ Error creating Linux desktop file: {e}{Colors.RESET}")
        return False


def create_desktop_shortcut(force: bool = False) -> bool:
    system = platform.system()
    print(f"{Colors.BOLD}🖥️ Creating desktop shortcut for {system}{Colors.RESET}")
    if system == "Windows":
        return create_windows_shortcut(force)
    elif system == "Darwin":
        return create_macos_app(force)
    else:
        return create_linux_desktop(force)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Create LeRobot GUI desktop shortcut")
    p.add_argument("--force", action="store_true", help="Force recreate existing shortcut")
    a = p.parse_args()
    ok = create_desktop_shortcut(force=a.force)
    sys.exit(0 if ok else 1)


if click:
    @click.command()
    @click.option("--force", is_flag=True, help="Force recreate existing shortcut")
    def main(force):
        create_desktop_shortcut(force=force)


def create_desktop_shortcut(force: bool = False) -> bool:
    """Create platform-appropriate desktop shortcut"""
    system = platform.system()
    
    print(f"{Colors.BOLD}🖥️ Creating desktop shortcut for {system}{Colors.RESET}")
    
    if system == "Windows":
        return create_windows_shortcut(force)
    elif system == "Darwin":
        return create_macos_app(force)
    else:  # Linux and others
        return create_linux_desktop(force)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create LeRobot GUI desktop shortcut")
    parser.add_argument("--force", action="store_true", help="Force recreate existing shortcut")
    args = parser.parse_args()
    
    success = create_desktop_shortcut(force=args.force)
    sys.exit(0 if success else 1)


@click.command()
@click.option("--force", is_flag=True, help="Force recreate existing shortcut")
def main(force):
    """Main entry point for CLI"""
    create_desktop_shortcut(force=force)
