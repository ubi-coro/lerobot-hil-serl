# backend/config_resolver.py
try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Fallback für ältere Python-Versionen
from pathlib import Path
from typing import Tuple
from config_models import TeleopRequest, RobotCfg, TeleopCfg, RobotArmCfg, TeleopArmCfg


MODE_PROFILE_MAP = {
    "bimanual": "bi_viperx",
    "left": "viperx_left",
    "right": "viperx_right",
}


def normalize_operation_mode(mode: str | None) -> str:
    if not mode:
        return "bimanual"
    lowered = mode.lower()
    if lowered in {"left", "left_only", "single_left", "aloha_left", "widowx_left"}:
        return "left"
    if lowered in {"right", "right_only", "single_right", "aloha_right", "widowx_right"}:
        return "right"
    return "bimanual"


def profile_for_mode(mode: str | None) -> str:
    normalized = normalize_operation_mode(mode)
    return MODE_PROFILE_MAP.get(normalized, MODE_PROFILE_MAP["bimanual"])

def load_profile(profile_name: str) -> Tuple[RobotCfg, TeleopCfg]:
    from config_models import CameraCfg
    profile_path = Path(__file__).parent / "hardware_profiles.toml"
    with open(profile_path, "rb") as f:
        data = tomllib.load(f)
    prof_data = data["profiles"][profile_name]
    
    # Konvertiere Kameras zu CameraCfg-Objekten
    if "cameras" in prof_data["robot"]:
        cameras = {}
        for name, cam_data in prof_data["robot"]["cameras"].items():
            cameras[name] = CameraCfg(**cam_data)
        prof_data["robot"]["cameras"] = cameras
    
    robot = RobotCfg(**prof_data["robot"])
    teleop = TeleopCfg(**prof_data["teleop"])
    return robot, teleop

def resolve(req: TeleopRequest) -> Tuple[RobotCfg, TeleopCfg, dict]:
    mode = normalize_operation_mode(req.operation_mode)

    # 1) Defaults
    robot = RobotCfg(type=req.robot_type, id="follower_right")
    teleop = TeleopCfg(type=req.teleop_type, id="leader_right")

    # 2) Hardware-Profil laden
    profile_name = req.profile_name or profile_for_mode(mode)
    if profile_name:
        prof_robot, prof_teleop = load_profile(profile_name)
        # Merge profile data manually to preserve nested objects
        if prof_robot.type:
            robot.type = prof_robot.type
        if prof_robot.id:
            robot.id = prof_robot.id
        if prof_robot.left_arm:
            robot.left_arm = prof_robot.left_arm
        if prof_robot.right_arm:
            robot.right_arm = prof_robot.right_arm
        if prof_robot.port:
            robot.port = prof_robot.port
        if prof_robot.cameras:
            robot.cameras = prof_robot.cameras
        if prof_robot.calibration_dir:
            robot.calibration_dir = prof_robot.calibration_dir
            
        if prof_teleop.type:
            teleop.type = prof_teleop.type
        if prof_teleop.id:
            teleop.id = prof_teleop.id
        if prof_teleop.left_arm:
            teleop.left_arm = prof_teleop.left_arm
        if prof_teleop.right_arm:
            teleop.right_arm = prof_teleop.right_arm
        if prof_teleop.port:
            teleop.port = prof_teleop.port
        if prof_teleop.calibration_dir:
            teleop.calibration_dir = prof_teleop.calibration_dir

    # 3) Umgebung (z. B. ENV-Mapping für Ports)
    # robot = apply_env_overrides(robot); teleop = apply_env_overrides(teleop)

    # 4) GUI-Overrides
    if mode == "bimanual":
        robot.type = "bi_viperx"
        teleop.type = "bi_widowx"
    elif mode == "left":
        robot.type = "viperx"
        teleop.type = "widowx"
        # Links-Arm aktivieren, Rechts deaktivieren
        robot.left_arm = robot.left_arm or RobotArmCfg(port="/dev/ttyDXL_follower_left", id="follower_left")
        teleop.left_arm = teleop.left_arm or TeleopArmCfg(port="/dev/ttyDXL_leader_left", id="leader_left")
        robot.right_arm = None
        teleop.right_arm = None
    else:  # "right"
        robot.type = "viperx"
        teleop.type = "widowx"
        robot.right_arm = robot.right_arm or RobotArmCfg(port="/dev/ttyDXL_follower_right", id="follower_right")
        teleop.right_arm = teleop.right_arm or TeleopArmCfg(port="/dev/ttyDXL_leader_right", id="leader_right")
        robot.left_arm = None
        teleop.left_arm = None

    if not req.cameras_enabled:
        robot.cameras = {}

    runtime = {"display_data": req.display_data, "fps": req.fps}
    return robot, teleop, runtime