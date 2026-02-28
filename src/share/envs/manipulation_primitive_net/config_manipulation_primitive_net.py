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

        outgoing_edges: dict[str, set[str]] = {name: set() for name in primitive_names}

        for source, target, transition in self.transitions:
            if source not in primitive_names:
                raise ValueError(f"Transition source '{source}' is not present in primitives.")
            if target not in primitive_names:
                raise ValueError(f"Transition target '{target}' is not present in primitives.")

            resolved_target = target
            transition_target = getattr(transition, "next_primitive", None)
            if transition_target is not None:
                if transition_target not in primitive_names:
                    raise ValueError(
                        "Transition resolver points to unknown primitive "
                        f"'{transition_target}' from source '{source}'."
                    )
                resolved_target = transition_target

            outgoing_edges[source].add(resolved_target)

            source_config = self.primitives[source]
            if getattr(source_config, "is_terminal_primitive", False) and resolved_target not in reset_primitive_set:
                raise ValueError(
                    "Terminal primitive transitions must target a reset primitive. "
                    f"Invalid edge: {source} -> {resolved_target}."
                )

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
            is_terminal = bool(getattr(primitive_cfg, "is_terminal_primitive", False))
            if not is_terminal and not outgoing_edges[primitive_name]:
                raise ValueError(
                    "Detected non-terminal dead-end primitive without outgoing transitions: "
                    f"'{primitive_name}'. Mark it terminal or add an outgoing transition."
                )

        reachable_from_start = _reachable_from(self.start_primitive)
        unreachable_terminals = sorted(
            name
            for name, primitive_cfg in self.primitives.items()
            if bool(getattr(primitive_cfg, "is_terminal_primitive", False)) and name not in reachable_from_start
        )
        if unreachable_terminals:
            raise ValueError(
                "Terminal primitive(s) are unreachable from start_primitive "
                f"'{self.start_primitive}': {', '.join(unreachable_terminals)}"
            )

        if reset_primitive_set:
            for reset_name in sorted(reset_primitive_set):
                if self.start_primitive not in _reachable_from(reset_name):
                    raise ValueError(
                        "Reset primitive has no transition path to start_primitive: "
                        f"'{reset_name}' -> '{self.start_primitive}'."
                    )
