from dataclasses import dataclass, field

import draccus

from lerobot.envs import EnvConfig
from lerobot.cameras import CameraConfig
from lerobot.teleoperators import TeleoperatorConfig
from lerobot.robots import RobotConfig

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
        for name in self.robot:
            self.robot[name].cameras = {}

