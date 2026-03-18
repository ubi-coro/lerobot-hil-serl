from dataclasses import dataclass, field

from lerobot.envs import EnvConfig
from lerobot.cameras import CameraConfig
from lerobot.teleoperators import TeleoperatorConfig
from lerobot.robots import RobotConfig
from lerobot.utils.constants import DEFAULT_ROBOT_NAME

from .transitions import Transition
from ..manipulation_primitive.config_manipulation_primitive import ManipulationPrimitiveConfig


@dataclass
class ManipulationPrimitiveNetConfig(EnvConfig):
    """Serializable config for MP-Net transition routing and reset/start semantics."""

    start_primitive: str | None = None
    reset_primitive: str | None = None
    primitives: dict[str, ManipulationPrimitiveConfig] = field(default_factory=dict)
    transitions: list[Transition] = field(default_factory=list)

    fps: int = 10
    robot: RobotConfig | dict[str, RobotConfig] | None = None
    teleop: TeleoperatorConfig | dict[str, TeleoperatorConfig] | None = None
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    @property
    def gym_kwargs(self) -> dict:
        return {}

    def make(self):
        from .env_manipulation_primitive_net import ManipulationPrimitiveNet
        return ManipulationPrimitiveNet(self)

    def __post_init__(self):
        """Validate primitive roles and transition graph semantics for MP-Net."""
        # Handle multi robot configuration
        self.robot = self.robot if isinstance(self.robot, dict) else {DEFAULT_ROBOT_NAME: self.robot}
        self.teleop = self.teleop if isinstance(self.teleop, dict) else {DEFAULT_ROBOT_NAME: self.teleop}
        for name, robot_cfg in self.robot.items():
            if robot_cfg is not None:
                robot_cfg.cameras = {}

        assert self.primitives
        primitive_names = list(self.primitives.keys())
        if self.start_primitive is None:
            self.start_primitive = primitive_names[0]
        if self.reset_primitive is None:
            self.reset_primitive = primitive_names[0]
        if self.start_primitive not in primitive_names:
            raise ValueError(f"start_primitive '{self.start_primitive}' is not present in primitives.")

        outgoing_edges: dict[str, set[str]] = {name: set() for name in primitive_names}

        for transition in self.transitions:
            if transition.source not in primitive_names:
                raise ValueError(f"Transition source '{transition.source}' is not present in primitives.")
            if transition.target not in primitive_names:
                raise ValueError(f"Transition target '{transition.target}' is not present in primitives.")

            outgoing_edges[transition.source].add(transition.target)

        def _reachable_from(start: str) -> set[str]:
            visited = {start}
            frontier = [start]

            while frontier:
                node = frontier.pop()
                for nxt in outgoing_edges[node]:
                    if nxt in visited:
                        continue
                    visited.add(nxt)
                    frontier.append(nxt)

            return visited

        for primitive_name, primitive_cfg in self.primitives.items():
            is_terminal = bool(getattr(primitive_cfg, "is_terminal", False))
            if not is_terminal and not outgoing_edges[primitive_name]:
                raise ValueError(
                    "Detected non-terminal dead-end primitive without outgoing transitions: "
                    f"'{primitive_name}'. Mark it terminal or add an outgoing transition."
                )

        reachable_from_start = _reachable_from(self.start_primitive)
        unreachable_terminals = sorted(
            name
            for name, primitive_cfg in self.primitives.items()
            if bool(getattr(primitive_cfg, "is_terminal", False)) and name not in reachable_from_start
        )
        if unreachable_terminals:
            raise ValueError(
                "Terminal primitive(s) are unreachable from start_primitive "
                f"'{self.start_primitive}': {', '.join(unreachable_terminals)}"
            )

    @property
    def terminals(self):
        return [k for k, v in self.primitives.items() if v.is_terminal]
