"""
Experiment Configuration Mapper for GUI

Maps GUI operation modes to predefined experiment configurations.
This allows the GUI to use fully-configured experiments with calibration
migration, processor pipelines, and policy paths.
"""

import logging
from typing import Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExperimentMapping:
    """Mapping between GUI mode and experiment config type"""
    experiment_type: str  # Registered EnvConfig subclass name
    description: str
    supports_policy: bool = False
    default_policy_path: Optional[str] = None


class ExperimentConfigMapper:
    """
    Maps GUI operation modes to predefined experiment configurations.
    
    This provides a clean interface for the GUI to select between:
    - Demo mode (with pre-trained policy for evaluation)
    - Teleoperation mode (bimanual, left-only, right-only)
    """
    
    # Mapping: (operation_mode, demo_mode) -> experiment_type
    EXPERIMENT_MAP = {
        # Demo configurations (with policy for evaluation)
        ("bimanual", True): ExperimentMapping(
            experiment_type="aloha_bimanual_lemgo_v2_demo",
            description="Bimanual Demo with Pre-trained Policy",
            supports_policy=True,
            default_policy_path="/media/jannick/DATA/aloha_model_lerobot/jannick-st/act-241025/pretrained_model"
        ),
        
        # Teleoperation configurations (recording/manual control)
        ("bimanual", False): ExperimentMapping(
            experiment_type="aloha_bimanual_lemgo_v2",
            description="Bimanual Teleoperation",
            supports_policy=False
        ),
        ("left", False): ExperimentMapping(
            experiment_type="aloha_bimanual_lemgo_v2_left",
            description="Left Arm Only Teleoperation",
            supports_policy=False
        ),
        ("right", False): ExperimentMapping(
            experiment_type="aloha_bimanual_lemgo_v2_right",
            description="Right Arm Only Teleoperation",
            supports_policy=False
        ),
    }
    
    @classmethod
    def get_experiment_type(
        cls,
        operation_mode: str,
        demo_mode: bool = False
    ) -> str:
        """
        Get the experiment type for a given operation mode and demo setting.
        
        Args:
            operation_mode: "bimanual", "left", or "right"
            demo_mode: True for demo/evaluation, False for teleoperation
            
        Returns:
            Registered experiment type name (e.g., "aloha_bimanual_lemgo_v2_demo")
            
        Raises:
            ValueError: If the combination is not supported
        """
        key = (operation_mode, demo_mode)
        
        if key not in cls.EXPERIMENT_MAP:
            available = list(cls.EXPERIMENT_MAP.keys())
            raise ValueError(
                f"No experiment mapping for mode={operation_mode}, demo={demo_mode}. "
                f"Available: {available}"
            )
        
        mapping = cls.EXPERIMENT_MAP[key]
        logger.info(f"Mapped GUI mode to experiment: {mapping.experiment_type} - {mapping.description}")
        
        return mapping.experiment_type
    
    @classmethod
    def get_experiment_info(
        cls,
        operation_mode: str,
        demo_mode: bool = False
    ) -> ExperimentMapping:
        """Get full experiment mapping info including policy support"""
        key = (operation_mode, demo_mode)
        return cls.EXPERIMENT_MAP[key]
    
    @classmethod
    def list_available_experiments(cls) -> dict:
        """List all available experiment configurations"""
        result = {}
        for (mode, is_demo), mapping in cls.EXPERIMENT_MAP.items():
            key = f"{mode}_{'demo' if is_demo else 'teleop'}"
            result[key] = {
                "operation_mode": mode,
                "demo_mode": is_demo,
                "experiment_type": mapping.experiment_type,
                "description": mapping.description,
                "supports_policy": mapping.supports_policy,
                "default_policy_path": mapping.default_policy_path
            }
        return result
    
    @classmethod
    def supports_demo_mode(cls, operation_mode: str) -> bool:
        """Check if an operation mode supports demo/evaluation"""
        return (operation_mode, True) in cls.EXPERIMENT_MAP
    
    @classmethod
    def create_env_from_gui_selection(
        cls,
        operation_mode: str,
        demo_mode: bool = False,
        policy_path_override: Optional[str] = None,
    ) -> tuple[Any, Any, Any, "EnvConfig", ExperimentMapping]:
        """
        Create environment from GUI selection.
        
        Args:
            operation_mode: "bimanual", "left", or "right"
            demo_mode: Enable demo/evaluation mode
            policy_path_override: Override default policy path
            
        Returns:
            Tuple of (env, env_processor, action_processor, env_cfg, mapping)
        """
        from lerobot.envs.configs import EnvConfig
        
        # Get experiment type
        experiment_type = cls.get_experiment_type(operation_mode, demo_mode)
        mapping = cls.get_experiment_info(operation_mode, demo_mode)
        
        # Import experiments to register configs
        import experiments  # noqa: F401
        
        # Create config instance
        env_cfg = EnvConfig.get_subclass(experiment_type)()
        
        # Override policy path if provided and supported
        if policy_path_override and mapping.supports_policy:
            logger.info(f"Overriding policy path: {policy_path_override}")
            # Note: Policy loading happens separately in teleoperation module
        
        # Create environment with processors
        logger.info(f"Creating environment from experiment: {experiment_type}")
        env, env_processor, action_processor = env_cfg.make(device="cpu")

        return env, env_processor, action_processor, env_cfg, mapping


# Example usage for GUI integration:
"""
# In aloha_teleoperation.py:

from modules.experiment_config_mapper import ExperimentConfigMapper

async def start_teleoperation(config: AlohaConfig):
    # Check if demo mode
    demo_mode = config.demo_mode or config.policy_path is not None
    
    # Get experiment and create env
    env, env_proc, action_proc, env_cfg, mapping = ExperimentConfigMapper.create_env_from_gui_selection(
        operation_mode=config.operation_mode,
        demo_mode=demo_mode,
        policy_path_override=config.policy_path
    )
    
    # Continue with teleoperation...
"""
