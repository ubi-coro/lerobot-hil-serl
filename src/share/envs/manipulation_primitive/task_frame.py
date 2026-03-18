from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class ControlSpace(IntEnum):
    """Target command space used by a task frame."""

    JOINT = 0
    TASK = 1


class PolicyMode(IntEnum):
    """How policy outputs are interpreted for a learnable axis."""

    ABSOLUTE = 0
    RELATIVE = 1


class ControlMode(IntEnum):
    """Low-level command type applied on each axis."""

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
        """Validate task-frame axis layout and mode compatibility."""
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
        """Return axis indices controlled by policy outputs."""
        return [i for i, _policy_mode in enumerate(self.policy_mode) if _policy_mode is not None]

    @property
    def is_adaptive(self) -> bool:
        """Whether at least one axis is policy-controlled."""
        return len(self.learnable_axis_indices) > 0

    @property
    def policy_action_dim(self) -> int:
        """Infer learning-space action dimension from the task-frame contract.

        Rules:
        - Any learnable VEL/FORCE axis contributes +1.
        - Any learnable POS translational axis (x/y/z) contributes +1.
        - Any learnable POS axis in RELATIVE mode contributes +1.
        - Learnable rotational POS axes (rx/ry/rz) in ABSOLUTE mode are represented on manifolds:
            * 1 axis -> +2 (S1)
            * 2 axes -> +3 (S2)
            * 3 axes -> +6 (SO(3), 6D representation)
        """
        dim = 0
        absolute_rotation_axes = 0

        for axis in self.learnable_axis_indices:
            if self.is_absolute_rotation_axis(axis):
                absolute_rotation_axes += 1
            else:
                dim += 1

        if absolute_rotation_axes == 0:
            return dim
        if absolute_rotation_axes == 1:
            return dim + 2
        if absolute_rotation_axes == 2:
            return dim + 3
        if absolute_rotation_axes == 3:
            return dim + 6

        raise ValueError(
            "Invalid absolute rotation axis count while inferring policy_action_dim. "
            f"Expected 0..3, got {absolute_rotation_axes}."
        )

    def is_absolute_rotation_axis(self, axis: int):
        return (
                axis >= 3 and
                self.control_mode[axis] == ControlMode.POS and
                self.policy_mode[axis] == PolicyMode.ABSOLUTE and
                self.space == ControlSpace.TASK
        )

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "space": int(self.space),
            "origin": self.origin,
            "target": self.target,
            "policy_mode": [int(policy_mode) if policy_mode is not None else None for policy_mode in self.policy_mode],
            "control_mode": [int(control_mode) for control_mode in self.control_mode],
            "min_target": self.min_pose,
            "max_target": self.max_pose,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> TaskFrame:
        """Build a task frame from a serialized dictionary."""
        min_target = raw.get("min_target", raw.get("min_pose"))
        max_target = raw.get("max_target", raw.get("max_pose"))
        return cls(
            space=ControlSpace(raw["space"]),
            origin=raw.get("origin"),
            target=list(raw["target"]),
            policy_mode=[PolicyMode(item) if item is not None else None for item in raw["policy_mode"]],
            control_mode=[ControlMode(item) for item in raw["control_mode"]],
            min_pose=list(min_target) if min_target is not None else None,
            max_pose=list(max_target) if max_target is not None else None,
        )

    @property
    def min_target(self) -> list[float] | None:
        """Canonical alias used by processor/docs for lower task-frame bounds."""
        return self.min_pose

    @min_target.setter
    def min_target(self, value: list[float] | None) -> None:
        """Set lower task-frame bounds using canonical alias."""
        self.min_pose = value

    @property
    def max_target(self) -> list[float] | None:
        """Canonical alias used by processor/docs for upper task-frame bounds."""
        return self.max_pose

    @max_target.setter
    def max_target(self, value: list[float] | None) -> None:
        """Set upper task-frame bounds using canonical alias."""
        self.max_pose = value


@dataclass(slots=True)
class PrimitiveGraphConfig:
    """Serializable primitive graph config that can be loaded from JSON/YAML."""

    start_primitive_id: str
    nodes: list

    def __post_init__(self) -> None:
        """Ensure primitive references and transitions are valid."""
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
        """Return the primitive node matching ``primitive_id``."""
        for node in self.nodes:
            if node.primitive_id == primitive_id:
                return node
        raise KeyError(f"unknown primitive_id '{primitive_id}'")
