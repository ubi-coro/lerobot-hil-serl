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
        # BiViperX uses a single calibration_dir for both arms
        # The arms will append "_left" and "_right" to the id when loading calibration files
        calibration_dir = None
        if robot.left_arm and robot.left_arm.calibration_dir:
            calibration_dir = robot.left_arm.calibration_dir
        elif robot.right_arm and robot.right_arm.calibration_dir:
            calibration_dir = robot.right_arm.calibration_dir
            
        robot_config = BiViperXConfig(
            id=robot.id,
            left_arm_port=robot.left_arm.port if robot.left_arm else None,
            right_arm_port=robot.right_arm.port if robot.right_arm else None,
            left_arm_max_relative_target=robot.max_relative_target,
            left_arm_moving_time=robot.moving_time,
            right_arm_max_relative_target=robot.max_relative_target,
            right_arm_moving_time=robot.moving_time,
            cameras=cameras,
            calibration_dir=calibration_dir,
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
        # BiWidowX uses a single calibration_dir for both arms
        # The arms will append "_left" and "_right" to the id when loading calibration files
        calibration_dir = None
        if teleop.left_arm and teleop.left_arm.calibration_dir:
            calibration_dir = teleop.left_arm.calibration_dir
        elif teleop.right_arm and teleop.right_arm.calibration_dir:
            calibration_dir = teleop.right_arm.calibration_dir
            
        teleop_config = BiWidowXConfig(
            id=teleop.id,
            left_arm_port=teleop.left_arm.port if teleop.left_arm else None,
            right_arm_port=teleop.right_arm.port if teleop.right_arm else None,
            left_arm_max_relative_target=teleop.max_relative_target,
            left_arm_moving_time=teleop.moving_time,
            left_arm_use_aloha2_gripper_servo=teleop.use_aloha2_gripper_servo,
            right_arm_max_relative_target=teleop.max_relative_target,
            right_arm_moving_time=teleop.moving_time,
            right_arm_use_aloha2_gripper_servo=teleop.use_aloha2_gripper_servo,
            calibration_dir=calibration_dir,
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