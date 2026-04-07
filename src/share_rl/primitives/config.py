from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class ControlSpace(IntEnum):
    JOINT = 0
    TASK = 1


class Origin(IntEnum):
    ABSOLUTE = 0
    RELATIVE = 1


class AxisMode(IntEnum):
    STIFF_POS = 0
    STIFF_VEL = 1
    COMPLIANT_POS = 2
    COMPLIANT_VEL = 3
    FORCE = 4

    @property
    def is_position_mode(self) -> bool:
        return self.name.endswith("_POS")


@dataclass(slots=True)
class TaskFrameCommand:
    """Serializable task-frame command shared by policy, processors, and robots."""

    space: ControlSpace
    origin: Origin
    target: list[float]
    policy_indices: list[bool]
    mode: list[AxisMode]
    domain_context: list[list[float]] | None = None

    def __post_init__(self) -> None:
        width = len(self.target)
        if width == 0:
            raise ValueError("target must contain at least one axis")
        if len(self.policy_indices) != width:
            raise ValueError("policy_indices must have the same length as target")
        if len(self.mode) != width:
            raise ValueError("mode must have the same length as target")

        if self.space == ControlSpace.TASK:
            if self.domain_context is None:
                raise ValueError("domain_context is required when space == TASK")
            if len(self.domain_context) != 4 or any(len(row) != 4 for row in self.domain_context):
                raise ValueError("domain_context must be a 4x4 transform matrix")
        elif self.domain_context is not None:
            raise ValueError("domain_context must be None when space == JOINT")

        if self.origin == Origin.RELATIVE and any(not axis_mode.is_position_mode for axis_mode in self.mode):
            raise ValueError("origin == RELATIVE only supports *_POS axis modes")

        if self.space == ControlSpace.JOINT and any(not axis_mode.is_position_mode for axis_mode in self.mode):
            raise ValueError("space == JOINT only supports *_POS axis modes")

    @property
    def learnable_axis_indices(self) -> list[int]:
        return [idx for idx, learnable in enumerate(self.policy_indices) if learnable]

    @property
    def is_adaptive(self) -> bool:
        return any(self.policy_indices)

    def to_dict(self) -> dict:
        return {
            "space": int(self.space),
            "origin": int(self.origin),
            "domain_context": self.domain_context,
            "target": self.target,
            "policy_indices": self.policy_indices,
            "mode": [int(axis_mode) for axis_mode in self.mode],
        }

    @classmethod
    def from_dict(cls, raw: dict) -> TaskFrameCommand:
        return cls(
            space=ControlSpace(raw["space"]),
            origin=Origin(raw["origin"]),
            domain_context=raw.get("domain_context"),
            target=list(raw["target"]),
            policy_indices=list(raw["policy_indices"]),
            mode=[AxisMode(item) for item in raw["mode"]],
        )


@dataclass(slots=True)
class RobotPrimitiveConfig:
    """Per-robot primitive configuration treated as a first-class env config object."""

    robot_id: str
    task_frame: TaskFrameCommand
    gripper_enabled: bool = True


@dataclass(slots=True)
class PrimitiveGraphNodeConfig:
    primitive_id: str
    primitive_type: str
    robot_primitives: dict[str, RobotPrimitiveConfig]
    transitions: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class PrimitiveGraphConfig:
    """Serializable primitive graph config that can be loaded from JSON/YAML."""

    start_primitive_id: str
    nodes: list[PrimitiveGraphNodeConfig]

    def __post_init__(self) -> None:
        node_ids = {node.primitive_id for node in self.nodes}
        if self.start_primitive_id not in node_ids:
            raise ValueError("start_primitive_id must reference an existing primitive_id")

        for node in self.nodes:
            for _, target in node.transitions.items():
                if target not in node_ids:
                    raise ValueError(
                        f"primitive '{node.primitive_id}' has transition to unknown primitive '{target}'"
                    )

    def node_by_id(self, primitive_id: str) -> PrimitiveGraphNodeConfig:
        for node in self.nodes:
            if node.primitive_id == primitive_id:
                return node
        raise KeyError(f"unknown primitive_id '{primitive_id}'")
