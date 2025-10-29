# backend/lerobot_adapter.py
from lerobot.robots.bi_viperx.config_bi_viperx import BiViperXConfig
from lerobot.robots.viperx.config_viperx import ViperXConfig
from lerobot.teleoperators.bi_widowx.config_bi_widowx import BiWidowXConfig
from lerobot.teleoperators.widowx.config_widowx import WidowXConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from config_models import RobotCfg, TeleopCfg

def to_lerobot_configs(robot: RobotCfg, teleop: TeleopCfg) -> tuple:
    """Konvertiere Pydantic-Modelle in LeRobot-Config-Objekte."""

    # Kameras konvertieren
    cameras = {}
    for name, cam in robot.cameras.items():
        cameras[name] = RealSenseCameraConfig(
            serial_number_or_name=cam.serial_number_or_name,
            width=cam.width,
            height=cam.height,
            fps=cam.fps
        )

    # Robot-Config
    if robot.type == "bi_viperx":
        left_arm_cfg = robot.left_arm
        right_arm_cfg = robot.right_arm

        left_calibration_dir = None
        if left_arm_cfg and left_arm_cfg.calibration_dir:
            left_calibration_dir = left_arm_cfg.calibration_dir
        elif robot.calibration_dir:
            left_calibration_dir = robot.calibration_dir

        right_calibration_dir = None
        if right_arm_cfg and right_arm_cfg.calibration_dir:
            right_calibration_dir = right_arm_cfg.calibration_dir
        elif robot.calibration_dir:
            right_calibration_dir = robot.calibration_dir

        # Fallback shared calibration directory if none provided
        shared_calibration_dir = (
            robot.calibration_dir
            or left_calibration_dir
            or right_calibration_dir
        )

        robot_config = BiViperXConfig(
            id=robot.id,
            left_arm_port=left_arm_cfg.port if left_arm_cfg else None,
            right_arm_port=right_arm_cfg.port if right_arm_cfg else None,
            left_arm_id=left_arm_cfg.id if left_arm_cfg else None,
            right_arm_id=right_arm_cfg.id if right_arm_cfg else None,
            left_arm_calibration_dir=left_calibration_dir,
            right_arm_calibration_dir=right_calibration_dir,
            left_arm_max_relative_target=robot.max_relative_target,
            left_arm_moving_time=robot.moving_time,
            right_arm_max_relative_target=robot.max_relative_target,
            right_arm_moving_time=robot.moving_time,
            cameras=cameras,
            calibration_dir=shared_calibration_dir,
            show_debugging_graphs=False,
        )
    else:  # viperx
        robot_config = ViperXConfig(
            id=robot.id,
            port=robot.port,
            max_relative_target=robot.max_relative_target,
            moving_time=robot.moving_time,
            calibration_dir=robot.calibration_dir,
            cameras=cameras,
        )

    # Teleop-Config
    if teleop.type == "bi_widowx":
        left_arm_cfg = teleop.left_arm
        right_arm_cfg = teleop.right_arm

        left_calibration_dir = None
        if left_arm_cfg and left_arm_cfg.calibration_dir:
            left_calibration_dir = left_arm_cfg.calibration_dir
        elif teleop.calibration_dir:
            left_calibration_dir = teleop.calibration_dir

        right_calibration_dir = None
        if right_arm_cfg and right_arm_cfg.calibration_dir:
            right_calibration_dir = right_arm_cfg.calibration_dir
        elif teleop.calibration_dir:
            right_calibration_dir = teleop.calibration_dir

        shared_calibration_dir = (
            teleop.calibration_dir
            or left_calibration_dir
            or right_calibration_dir
        )

        teleop_config = BiWidowXConfig(
            id=teleop.id,
            left_arm_port=left_arm_cfg.port if left_arm_cfg else None,
            right_arm_port=right_arm_cfg.port if right_arm_cfg else None,
            left_arm_id=left_arm_cfg.id if left_arm_cfg else None,
            right_arm_id=right_arm_cfg.id if right_arm_cfg else None,
            left_arm_calibration_dir=left_calibration_dir,
            right_arm_calibration_dir=right_calibration_dir,
            left_arm_max_relative_target=teleop.max_relative_target,
            left_arm_moving_time=teleop.moving_time,
            left_arm_use_aloha2_gripper_servo=teleop.use_aloha2_gripper_servo,
            right_arm_max_relative_target=teleop.max_relative_target,
            right_arm_moving_time=teleop.moving_time,
            right_arm_use_aloha2_gripper_servo=teleop.use_aloha2_gripper_servo,
            calibration_dir=shared_calibration_dir,
        )
    else:  # widowx
        teleop_config = WidowXConfig(
            id=teleop.id,
            port=teleop.port,
            max_relative_target=teleop.max_relative_target,
            moving_time=teleop.moving_time,
            use_aloha2_gripper_servo=teleop.use_aloha2_gripper_servo,
            calibration_dir=teleop.calibration_dir
        )

    return robot_config, teleop_config