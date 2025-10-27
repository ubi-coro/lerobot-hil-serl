# web/backend_fastapi/modules/gui_config_manager.py
"""
GUI Configuration Manager

Manages the generic GUI environment configurations and injects
hardware parameters from hardware profiles at runtime.
"""

import logging
from pathlib import Path
from typing import Any

from lerobot.envs.configs import EnvConfig
from lerobot.share.configs import RecordConfig, DatasetRecordConfig

logger = logging.getLogger(__name__)


class GuiConfigManager:
    """
    Manages configuration for GUI-based robot control.
    Bridges hardware_profiles.toml and the experiments-based EnvConfig system.
    """
    
    @staticmethod
    def create_env_config_from_profile(
        operation_mode: str,
        hardware_profile: dict,
        fps: int = 30,
        control_time_s: float = 60.0
    ) -> EnvConfig:
        """
        Create an EnvConfig from a hardware profile and operation mode.
        
        Args:
            operation_mode: "bimanual", "left", or "right"
            hardware_profile: Dictionary with robot, teleop, and camera configs
            fps: Frames per second for control loop
            control_time_s: Maximum time per episode
            
        Returns:
            Configured EnvConfig ready for env.make()
        """
        # Import here to ensure configs are registered
        from experiments.gui_generic.config import (
            GuiAlohaBimanualEnvConfig,
            GuiAlohaSingleLeftEnvConfig,
            GuiAlohaSingleRightEnvConfig
        )
        
        if operation_mode == "bimanual":
            env_cfg = GuiAlohaBimanualEnvConfig()
            
            # Extract ports from profile
            robot_cfg = hardware_profile["robot"]
            teleop_cfg = hardware_profile["teleop"]
            
            env_cfg.set_hardware_config(
                robot_left_port=robot_cfg.get("left_arm", {}).get("port"),
                robot_right_port=robot_cfg.get("right_arm", {}).get("port"),
                teleop_left_port=teleop_cfg.get("left_arm", {}).get("port"),
                teleop_right_port=teleop_cfg.get("right_arm", {}).get("port"),
                cameras=hardware_profile.get("cameras", {}),
                robot_id=robot_cfg.get("id", "gui_bimanual"),
                teleop_id=teleop_cfg.get("id", "gui_bimanual"),
                calibration_dir=robot_cfg.get("calibration_dir")
            )
            
        elif operation_mode == "left":
            env_cfg = GuiAlohaSingleLeftEnvConfig()
            
            robot_cfg = hardware_profile["robot"]
            teleop_cfg = hardware_profile["teleop"]
            
            # For single arm, check if we have arm-specific config or direct port
            robot_port = robot_cfg.get("port") or robot_cfg.get("left_arm", {}).get("port")
            teleop_port = teleop_cfg.get("port") or teleop_cfg.get("left_arm", {}).get("port")
            
            env_cfg.set_hardware_config(
                robot_port=robot_port,
                teleop_port=teleop_port,
                cameras=hardware_profile.get("cameras", {}),
                robot_id=robot_cfg.get("id", "gui_left"),
                teleop_id=teleop_cfg.get("id", "gui_left"),
                calibration_dir=robot_cfg.get("calibration_dir")
            )
            
        elif operation_mode == "right":
            env_cfg = GuiAlohaSingleRightEnvConfig()
            
            robot_cfg = hardware_profile["robot"]
            teleop_cfg = hardware_profile["teleop"]
            
            robot_port = robot_cfg.get("port") or robot_cfg.get("right_arm", {}).get("port")
            teleop_port = teleop_cfg.get("port") or teleop_cfg.get("right_arm", {}).get("port")
            
            env_cfg.set_hardware_config(
                robot_port=robot_port,
                teleop_port=teleop_port,
                cameras=hardware_profile.get("cameras", {}),
                robot_id=robot_cfg.get("id", "gui_right"),
                teleop_id=teleop_cfg.get("id", "gui_right"),
                calibration_dir=robot_cfg.get("calibration_dir")
            )
        else:
            raise ValueError(f"Unknown operation mode: {operation_mode}")
        
        # Set runtime parameters
        env_cfg.fps = fps
        env_cfg.processor.control_time_s = control_time_s
        
        return env_cfg
    
    @staticmethod
    def create_record_config_from_gui(
        env_cfg: EnvConfig,
        repo_id: str,
        task_description: str,
        num_episodes: int,
        output_dir: str | None = None,
        policy_path: str | None = None
    ) -> RecordConfig:
        """
        Create a RecordConfig for recording or policy rollout.
        
        Args:
            env_cfg: Environment configuration (from create_env_config_from_profile)
            repo_id: Dataset repository ID
            task_description: Task description for the dataset
            num_episodes: Number of episodes to record
            output_dir: Output directory for dataset (optional)
            policy_path: Path to policy for rollout (optional)
            
        Returns:
            RecordConfig ready for lerobot.share.record.record()
        """
        dataset_cfg = DatasetRecordConfig(
            repo_id=repo_id,
            single_task=task_description,
            root=output_dir,
            num_episodes=num_episodes,
            fps=env_cfg.fps,
            video=True,
            push_to_hub=False  # GUI sessions typically stay local
        )
        
        record_cfg = RecordConfig(
            env=env_cfg,
            dataset=dataset_cfg,
            policy=None if not policy_path else {"path": policy_path},
            display_data=False,  # Can be overridden
            play_sounds=True
        )
        
        return record_cfg
    
    @staticmethod
    def list_available_experiments() -> list[dict[str, Any]]:
        """
        List all registered experiment configurations.
        
        Returns:
            List of experiment metadata dicts with name, description, etc.
        """
        experiments = []
        
        # Get all registered EnvConfig subclasses
        for name in EnvConfig.get_all_choice_names():
            try:
                env_class = EnvConfig.build_class_by_name(name)
                
                # Skip GUI generic configs (they're not real experiments)
                if name.startswith("gui_"):
                    continue
                
                # Try to get docstring or other metadata
                description = env_class.__doc__ or "No description available"
                
                experiments.append({
                    "name": name,
                    "class_name": env_class.__name__,
                    "description": description.strip().split('\n')[0],  # First line only
                    "is_sim": name in ["aloha", "pusht", "xarm", "libero"],
                    "is_real": name not in ["aloha", "pusht", "xarm", "libero"]
                })
            except Exception as e:
                logger.warning(f"Could not load experiment '{name}': {e}")
                continue
        
        return experiments
    
    @staticmethod
    def get_experiment_config(experiment_name: str) -> EnvConfig:
        """
        Load a specific experiment configuration by name.
        
        Args:
            experiment_name: Name of the registered experiment
            
        Returns:
            Instantiated EnvConfig
            
        Raises:
            ValueError: If experiment not found
        """
        try:
            env_class = EnvConfig.build_class_by_name(experiment_name)
            return env_class()
        except Exception as e:
            raise ValueError(f"Could not load experiment '{experiment_name}': {e}") from e
    
    @staticmethod
    def save_session_as_experiment(
        session_config: dict,
        experiment_name: str,
        output_dir: str | None = None
    ) -> Path:
        """
        Save a GUI session configuration as a persistent experiment.
        Generates a Python config file in the experiments directory.
        
        Args:
            session_config: Dictionary with robot, teleop, camera configs
            experiment_name: Name for the new experiment
            output_dir: Optional custom output directory
            
        Returns:
            Path to the generated config file
        """
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent.parent / "src" / "experiments"
        else:
            output_dir = Path(output_dir)
        
        exp_dir = output_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate Python config file
        config_content = f'''# Generated from GUI session
from dataclasses import dataclass
from lerobot.envs.configs import HilSerlRobotEnvConfig, EnvConfig
from lerobot.robots.viperx import ViperXConfig
from lerobot.teleoperators.widowx import WidowXConfig
from lerobot.cameras.realsense import RealSenseCameraConfig

@dataclass
@EnvConfig.register_subclass("{experiment_name}")
class {_to_class_name(experiment_name)}(HilSerlRobotEnvConfig):
    """Generated from GUI session."""
    
    def __post_init__(self):
        # TODO: Fill in from session_config
        pass
'''
        
        config_path = exp_dir / "config.py"
        config_path.write_text(config_content)
        
        logger.info(f"Created experiment config at {config_path}")
        return config_path


def _to_class_name(snake_case: str) -> str:
    """Convert snake_case to PascalCase for class names."""
    return ''.join(word.capitalize() for word in snake_case.split('_'))
