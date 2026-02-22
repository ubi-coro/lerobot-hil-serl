from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class ControlSpace(IntEnum):
    JOINT = 0
    TASK = 1


class PolicyMode(IntEnum):
    ABSOLUTE = 0
    RELATIVE = 1


class ControlMode(IntEnum):
    POS = 0
    VEL = 1
    FORCE = 2

@dataclass(slots=True)
class TaskFrame:
    """Serializable task-frame command shared by policy, processors, and robots."""
    target: list[float] = field(default_factory=lambda: 6 * [0.0])
    space: ControlSpace = ControlSpace.TASK
    policy_mode: list[PolicyMode | None] = field(default_factory=lambda: 6 * [None])
    control_mode: list[ControlMode] = field(default_factory=lambda: 6 * [ControlMode.VEL])
    origin: list[float] | None = None
    min_pose: list[float] | None = None  # 6-vector: min xyz (m), min extrinsic euler (rad)
    max_pose: list[float] | None = None  # 6-vector: max xyz (m), max extrinsic euler (rad)

    def __post_init__(self) -> None:
        width = len(self.target)
        if width == 0:
            raise ValueError("target must contain at least one axis")
        if len(self.policy_mode) != width:
            raise ValueError("policy_mode must have the same length as target")
        if len(self.control_mode) != width:
            raise ValueError("control_mode must have the same length as target")

        if self.space == ControlSpace.TASK:
            if self.origin is None:
                self.origin = 6 * [0.0]
            if len(self.origin) != 6:
                raise ValueError("origin must be a 6 vector (xyz + rotation vector in rad)")
        elif self.origin is not None:
            raise ValueError("origin must be None when space == JOINT")
        
        for i in range(width):
            if self.policy_mode[i] is None:
                continue
                
            if self.policy_mode[i] == PolicyMode.RELATIVE and not self.control_mode[i] == ControlMode.POS:
                raise ValueError("policy_mode == RELATIVE only supports POS control modes")
            
            if self.space == ControlSpace.JOINT and not self.control_mode[i] == ControlMode.POS:
                raise ValueError("space == JOINT only supports POS axis modes")

    @property
    def learnable_axis_indices(self) -> list[int]:
        return [i for i, _policy_mode in enumerate(self.policy_mode) if _policy_mode is not None]

    @property
    def is_adaptive(self) -> bool:
        return len(self.learnable_axis_indices) > 0

    def to_dict(self) -> dict:
        return {
            "space": int(self.space),
            "origin": self.origin,
            "target": self.target,
            "policy_mode": [int(policy_mode) for policy_mode in self.policy_mode],
            "control_mode": [int(control_mode) for control_mode in self.control_mode],
            "min_pose": self.min_pose,
            "max_pose": self.max_pose,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> TaskFrame:
        return cls(
            space=ControlSpace(raw["space"]),
            origin=raw.get("origin"),
            target=list(raw["target"]),
            policy_mode=[PolicyMode(item) for item in raw["policy_mode"]],
            control_mode=[ControlMode(item) for item in raw["control_mode"]],
            min_pose=list(raw["min_pose"]),
            max_pose=list(raw["max_pose"]),
        )

@dataclass(slots=True)
class PrimitiveGraphConfig:
    """Serializable primitive graph config that can be loaded from JSON/YAML."""

    start_primitive_id: str
    nodes: list

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

    def node_by_id(self, primitive_id: str):
        for node in self.nodes:
            if node.primitive_id == primitive_id:
                return node
        raise KeyError(f"unknown primitive_id '{primitive_id}'")
