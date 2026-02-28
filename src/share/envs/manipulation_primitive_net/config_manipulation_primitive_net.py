from dataclasses import dataclass, field

import draccus

from lerobot.envs import EnvConfig
from lerobot.cameras import CameraConfig
from lerobot.teleoperators import TeleoperatorConfig
from lerobot.robots import RobotConfig
from lerobot.utils.constants import DEFAULT_ROBOT_NAME

from .transitions import MP_Transition
from ..manipulation_primitive.config_manipulation_primitive import ManipulationPrimitiveConfig


@dataclass
class ManipulationPrimitiveNetConfig(draccus.ChoiceRegistry):
    start_primitive: str
    primitives: dict[str, ManipulationPrimitiveConfig]
    transitions: list[tuple[str, str, MP_Transition]]
    reset_primitives: list[str] = field(default_factory=list)

    fps: int = 10
    robot: RobotConfig | dict[str, RobotConfig] | None = None
    teleop: TeleoperatorConfig | dict[str, TeleoperatorConfig] | None = None
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self):
        # Handle multi robot configuration
        self.robot = self.robot if isinstance(self.robot, dict) else {DEFAULT_ROBOT_NAME: self.robot}
        self.teleop = self.teleop if isinstance(self.teleop, dict) else {DEFAULT_ROBOT_NAME: self.teleop}
        for name, robot_cfg in self.robot.items():
            if robot_cfg is not None:
                robot_cfg.cameras = {}

        primitive_names = set(self.primitives)
        if self.start_primitive not in primitive_names:
            raise ValueError(f"start_primitive '{self.start_primitive}' is not present in primitives.")

        reset_primitive_set = set(self.reset_primitives)
        unknown_reset_primitives = reset_primitive_set - primitive_names
        if unknown_reset_primitives:
            unknown = ", ".join(sorted(unknown_reset_primitives))
            raise ValueError(f"reset_primitives contain unknown primitive(s): {unknown}")

        for source, target, _ in self.transitions:
            if source not in primitive_names:
                raise ValueError(f"Transition source '{source}' is not present in primitives.")
            if target not in primitive_names:
                raise ValueError(f"Transition target '{target}' is not present in primitives.")

            source_config = self.primitives[source]
            if getattr(source_config, "is_terminal_primitive", False) and target not in reset_primitive_set:
                raise ValueError(
                    "Terminal primitive transitions must target a reset primitive. "
                    f"Invalid edge: {source} -> {target}."
                )
