"""Utilities for loading, validating, summarizing, and editing MP-Net configs."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

import draccus

from share.envs.manipulation_primitive_net.transitions import (
    Always,
    OnObservationThreshold,
    OnSuccess,
    OnTimeLimit,
    RewardClassifierTransition,
    Transition,
)

try:
    from lerobot.configs.policies import PreTrainedConfig
    from share.envs.manipulation_primitive.config_manipulation_primitive import ManipulationPrimitiveConfig
    from share.envs.manipulation_primitive.task_frame import ControlMode, PolicyMode, TASK_FRAME_AXIS_NAMES, TaskFrame
    from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import ManipulationPrimitiveNetConfig

    REAL_MPNET_BACKEND = True
except Exception:  # noqa: BLE001
    REAL_MPNET_BACKEND = False

    class ControlMode(IntEnum):
        POS = 0
        VEL = 1
        FORCE = 2

    class PolicyMode(IntEnum):
        ABSOLUTE = 0
        RELATIVE = 1

    TASK_FRAME_AXIS_NAMES = ["x", "y", "z", "wx", "wy", "wz"]

    @dataclass(slots=True)
    class TaskFrame:
        target: list[float] = field(default_factory=lambda: 6 * [0.0])
        space: int = 1
        policy_mode: list[PolicyMode | None] = field(default_factory=lambda: 6 * [None])
        control_mode: list[ControlMode] = field(default_factory=lambda: 6 * [ControlMode.VEL])
        origin: list[float] | None = None
        kp: list[float] | None = None
        kd: list[float] | None = None
        min_pose: list[float] | None = None
        max_pose: list[float] | None = None

        def __post_init__(self) -> None:
            width = len(self.target)
            if len(self.policy_mode) != width:
                raise ValueError("policy_mode must have the same length as target")
            if len(self.control_mode) != width:
                raise ValueError("control_mode must have the same length as target")
            if self.space == 1 and width != len(TASK_FRAME_AXIS_NAMES):
                raise ValueError("task-space frames must be 6D")
            if self.space == 1 and self.origin is None:
                self.origin = 6 * [0.0]

        @property
        def learnable_axis_indices(self) -> list[int]:
            return [idx for idx, item in enumerate(self.policy_mode) if item is not None]

        @property
        def is_adaptive(self) -> bool:
            return bool(self.learnable_axis_indices)

        def is_absolute_rotation_axis(self, axis: int) -> bool:
            return axis >= 3 and self.control_mode[axis] == ControlMode.POS and self.policy_mode[axis] == PolicyMode.ABSOLUTE

        @property
        def policy_action_dim(self) -> int:
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
            raise ValueError("Invalid absolute rotation axis count")

    @dataclass(slots=True)
    class PreTrainedConfig:
        pretrained_path: Path | None = None
        device: str = "cpu"

        @classmethod
        def from_pretrained(cls, pretrained_name_or_path: str, **_: Any) -> "PreTrainedConfig":
            return cls(pretrained_path=Path(pretrained_name_or_path))

    @dataclass(slots=True)
    class ManipulationPrimitiveConfig:
        task_frame: TaskFrame | dict[str, TaskFrame] = field(default_factory=TaskFrame)
        processor: dict[str, Any] = field(default_factory=dict)
        policy: PreTrainedConfig | None = None
        policy_overwrites: dict[str, Any] = field(default_factory=dict)
        notes: str | None = None
        is_terminal: bool = False

        def __post_init__(self) -> None:
            task_frames = self.task_frame.values() if isinstance(self.task_frame, dict) else [self.task_frame]
            for frame in task_frames:
                frame.__post_init__()

    @dataclass(slots=True)
    class ManipulationPrimitiveNetConfig:
        start_primitive: str | None = None
        reset_primitive: str | None = None
        primitives: dict[str, ManipulationPrimitiveConfig] = field(default_factory=dict)
        transitions: list[Transition] = field(default_factory=list)
        fps: int = 10
        robot: Any = None
        teleop: Any = None
        cameras: dict[str, Any] = field(default_factory=dict)

        def __post_init__(self) -> None:
            if not self.primitives:
                raise ValueError("At least one primitive is required")
            primitive_names = list(self.primitives.keys())
            if self.start_primitive is None:
                self.start_primitive = primitive_names[0]
            if self.reset_primitive is None:
                self.reset_primitive = primitive_names[0]
            if self.start_primitive not in self.primitives:
                raise ValueError("start_primitive must exist in primitives")

            outgoing_edges: dict[str, set[str]] = {name: set() for name in primitive_names}
            for transition in self.transitions:
                if transition.source not in self.primitives:
                    raise ValueError(f"Transition source '{transition.source}' is not present in primitives")
                if transition.target not in self.primitives:
                    raise ValueError(f"Transition target '{transition.target}' is not present in primitives")
                outgoing_edges[transition.source].add(transition.target)

            def reachable_from(start: str) -> set[str]:
                visited = {start}
                frontier = [start]
                while frontier:
                    current = frontier.pop()
                    for nxt in outgoing_edges[current]:
                        if nxt not in visited:
                            visited.add(nxt)
                            frontier.append(nxt)
                return visited

            for primitive_name, primitive_cfg in self.primitives.items():
                primitive_cfg.__post_init__()
                if not primitive_cfg.is_terminal and not outgoing_edges[primitive_name]:
                    raise ValueError(f"Detected non-terminal dead-end primitive '{primitive_name}'")

            reachable = reachable_from(self.start_primitive)
            unreachable_terminals = [
                name for name, primitive_cfg in self.primitives.items() if primitive_cfg.is_terminal and name not in reachable
            ]
            if unreachable_terminals:
                raise ValueError(
                    "Terminal primitive(s) are unreachable from start_primitive "
                    f"'{self.start_primitive}': {', '.join(sorted(unreachable_terminals))}"
                )

        @property
        def terminals(self) -> list[str]:
            return [name for name, primitive in self.primitives.items() if primitive.is_terminal]


TRANSITION_TYPES: dict[str, type[Transition]] = {
    "always": Always,
    "on_success": OnSuccess,
    "on_observation_threshold": OnObservationThreshold,
    "on_time_limit": OnTimeLimit,
    "reward_classifier": RewardClassifierTransition,
}
TRANSITION_TYPE_NAMES = {cls: name for name, cls in TRANSITION_TYPES.items()}

AXIS_TO_INDEX = {name: idx for idx, name in enumerate(TASK_FRAME_AXIS_NAMES)}


def create_template_mpnet(primitive_name: str = "main", notes: str | None = None) -> ManipulationPrimitiveNetConfig:
    """Create a minimal valid MP-Net with one terminal primitive."""
    primitive = ManipulationPrimitiveConfig(
        task_frame=TaskFrame(
            target=[0.0] * 6,
            control_mode=[ControlMode.POS] * 6,
            policy_mode=[None] * 6,
        ),
        notes=notes,
        is_terminal=True,
    )
    return ManipulationPrimitiveNetConfig(
        start_primitive=primitive_name,
        reset_primitive=primitive_name,
        primitives={primitive_name: primitive},
        transitions=[],
    )


def load_mpnet_config(path: str | Path) -> ManipulationPrimitiveNetConfig:
    """Load an MP-Net config from a JSON file."""
    path = Path(path)
    if not REAL_MPNET_BACKEND:
        return _decode_mpnet(json.loads(path.read_text(encoding="utf-8")))
    with draccus.config_type("json"):
        return draccus.parse(ManipulationPrimitiveNetConfig, config_path=path, args=[])


def save_mpnet_config(config: ManipulationPrimitiveNetConfig, path: str | Path) -> None:
    """Persist an MP-Net config as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not REAL_MPNET_BACKEND:
        path.write_text(json.dumps(_encode_mpnet(config), indent=2), encoding="utf-8")
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        return
    with open(path, "w", encoding="utf-8") as f, draccus.config_type("json"):
        draccus.dump(config, f, indent=2)
        f.write("\n")


def _encode_task_frame(frame: TaskFrame) -> dict[str, Any]:
    return {
        "target": list(frame.target),
        "space": int(frame.space),
        "policy_mode": [None if item is None else int(item) for item in frame.policy_mode],
        "control_mode": [int(item) for item in frame.control_mode],
        "origin": list(frame.origin) if frame.origin is not None else None,
        "kp": frame.kp,
        "kd": frame.kd,
        "min_pose": frame.min_pose,
        "max_pose": frame.max_pose,
    }


def _decode_task_frame(payload: dict[str, Any]) -> TaskFrame:
    return TaskFrame(
        target=list(payload.get("target", [0.0] * 6)),
        space=payload.get("space", 1),
        policy_mode=[PolicyMode(item) if item is not None else None for item in payload.get("policy_mode", [None] * 6)],
        control_mode=[ControlMode(item) for item in payload.get("control_mode", [0] * 6)],
        origin=payload.get("origin"),
        kp=payload.get("kp"),
        kd=payload.get("kd"),
        min_pose=payload.get("min_pose"),
        max_pose=payload.get("max_pose"),
    )


def _encode_transition(transition: Transition) -> dict[str, Any]:
    payload = {"type": TRANSITION_TYPE_NAMES.get(type(transition), type(transition).__name__)}
    payload.update(vars(transition))
    return payload


def _decode_transition(payload: dict[str, Any]) -> Transition:
    payload = dict(payload)
    transition_type = payload.pop("type", "always")
    transition_cls = TRANSITION_TYPES[transition_type]
    return transition_cls(**payload)


def _encode_primitive(primitive: ManipulationPrimitiveConfig) -> dict[str, Any]:
    if isinstance(primitive.task_frame, dict):
        task_frame = {name: _encode_task_frame(frame) for name, frame in primitive.task_frame.items()}
    else:
        task_frame = _encode_task_frame(primitive.task_frame)
    payload = {
        "task_frame": task_frame,
        "policy_overwrites": primitive.policy_overwrites,
        "notes": primitive.notes,
        "is_terminal": primitive.is_terminal,
    }
    if primitive.policy is not None and primitive.policy.pretrained_path is not None:
        payload["policy"] = {"pretrained_path": str(primitive.policy.pretrained_path), "device": getattr(primitive.policy, "device", "cpu")}
    if getattr(primitive, "processor", None):
        payload["processor"] = primitive.processor
    return payload


def _decode_primitive(payload: dict[str, Any]) -> ManipulationPrimitiveConfig:
    raw_task_frame = payload.get("task_frame", {})
    if isinstance(raw_task_frame, dict) and "target" not in raw_task_frame:
        task_frame = {name: _decode_task_frame(frame_payload) for name, frame_payload in raw_task_frame.items()}
    else:
        task_frame = _decode_task_frame(raw_task_frame)
    policy_payload = payload.get("policy")
    policy = None
    if isinstance(policy_payload, dict) and policy_payload.get("pretrained_path"):
        policy = PreTrainedConfig(pretrained_path=Path(policy_payload["pretrained_path"]), device=policy_payload.get("device", "cpu"))
    return ManipulationPrimitiveConfig(
        task_frame=task_frame,
        processor=payload.get("processor", {}),
        policy=policy,
        policy_overwrites=payload.get("policy_overwrites", {}),
        notes=payload.get("notes"),
        is_terminal=payload.get("is_terminal", False),
    )


def _encode_mpnet(config: ManipulationPrimitiveNetConfig) -> dict[str, Any]:
    return {
        "start_primitive": config.start_primitive,
        "reset_primitive": config.reset_primitive,
        "primitives": {name: _encode_primitive(primitive) for name, primitive in config.primitives.items()},
        "transitions": [_encode_transition(transition) for transition in config.transitions],
        "fps": config.fps,
    }


def _decode_mpnet(payload: dict[str, Any]) -> ManipulationPrimitiveNetConfig:
    config = ManipulationPrimitiveNetConfig(
        start_primitive=payload.get("start_primitive"),
        reset_primitive=payload.get("reset_primitive"),
        primitives={name: _decode_primitive(primitive) for name, primitive in payload.get("primitives", {}).items()},
        transitions=[_decode_transition(transition) for transition in payload.get("transitions", [])],
        fps=payload.get("fps", 10),
    )
    config.__post_init__()
    return config


def _validate_task_frame(frame: TaskFrame) -> None:
    frame.__post_init__()


def _validate_primitive(primitive: ManipulationPrimitiveConfig) -> None:
    primitive.__post_init__()
    task_frames = primitive.task_frame.values() if isinstance(primitive.task_frame, dict) else [primitive.task_frame]
    for frame in task_frames:
        _validate_task_frame(frame)


def validate_mpnet_config(config: ManipulationPrimitiveNetConfig) -> ManipulationPrimitiveNetConfig:
    """Validate an MP-Net config after structured edits."""
    candidate = copy.deepcopy(config)
    for primitive in candidate.primitives.values():
        _validate_primitive(primitive)
    candidate.__post_init__()
    return candidate


def summarize_mpnet(config: ManipulationPrimitiveNetConfig) -> dict[str, Any]:
    """Return a JSON-friendly MP-Net summary for prompts and CLI output."""
    primitives = []
    for name, primitive in config.primitives.items():
        task_frames = primitive.task_frame if isinstance(primitive.task_frame, dict) else {"default": primitive.task_frame}
        frame_summary = {}
        for robot_name, frame in task_frames.items():
            frame_summary[robot_name] = {
                "target": list(frame.target),
                "policy_mode": [None if item is None else int(item) for item in frame.policy_mode],
                "control_mode": [int(item) for item in frame.control_mode],
                "learnable_axes": [TASK_FRAME_AXIS_NAMES[idx] for idx in frame.learnable_axis_indices if idx < len(TASK_FRAME_AXIS_NAMES)],
                "policy_action_dim": frame.policy_action_dim,
            }
        primitives.append(
            {
                "name": name,
                "is_terminal": primitive.is_terminal,
                "has_policy": primitive.policy is not None,
                "policy_path": str(primitive.policy.pretrained_path) if primitive.policy is not None and primitive.policy.pretrained_path is not None else None,
                "policy_overwrites": primitive.policy_overwrites,
                "notes": primitive.notes,
                "task_frames": frame_summary,
            }
        )
    transitions = [
        {
            "index": idx,
            "type": TRANSITION_TYPE_NAMES.get(type(transition), type(transition).__name__),
            "source": transition.source,
            "target": transition.target,
            "details": {
                key: value
                for key, value in vars(transition).items()
                if key not in {"source", "target"}
            },
        }
        for idx, transition in enumerate(config.transitions)
    ]
    return {
        "start_primitive": config.start_primitive,
        "reset_primitive": config.reset_primitive,
        "fps": config.fps,
        "primitive_count": len(config.primitives),
        "transition_count": len(config.transitions),
        "terminals": config.terminals,
        "primitives": primitives,
        "transitions": transitions,
    }


def list_primitives(config: ManipulationPrimitiveNetConfig) -> list[dict[str, Any]]:
    """List primitive summaries only."""
    return summarize_mpnet(config)["primitives"]


def describe_transitions(config: ManipulationPrimitiveNetConfig) -> list[dict[str, Any]]:
    """List transition summaries only."""
    return summarize_mpnet(config)["transitions"]


def _primitive_task_frames(primitive: ManipulationPrimitiveConfig) -> dict[str, TaskFrame]:
    if isinstance(primitive.task_frame, dict):
        return primitive.task_frame
    return {"default": primitive.task_frame}


def _resolve_frame(primitive: ManipulationPrimitiveConfig, robot_name: str | None = None) -> tuple[str, TaskFrame]:
    task_frames = _primitive_task_frames(primitive)
    if robot_name is not None:
        if robot_name not in task_frames:
            raise KeyError(f"Robot/task-frame key '{robot_name}' not found in primitive")
        return robot_name, task_frames[robot_name]
    first_name = next(iter(task_frames))
    return first_name, task_frames[first_name]


def add_primitive(
    config: ManipulationPrimitiveNetConfig,
    name: str,
    *,
    template_from: str | None = None,
    is_terminal: bool = False,
    notes: str | None = None,
    connect_from: str | None = None,
    connect_transition_type: str | None = None,
    connect_parameters: dict[str, Any] | None = None,
) -> ManipulationPrimitiveNetConfig:
    """Add one primitive, optionally cloning an existing one."""
    if name in config.primitives:
        raise ValueError(f"Primitive '{name}' already exists")
    if template_from is None:
        primitive = ManipulationPrimitiveConfig(
            task_frame=TaskFrame(
                target=[0.0] * 6,
                control_mode=[ControlMode.POS] * 6,
                policy_mode=[None] * 6,
            ),
            notes=notes,
            is_terminal=is_terminal,
        )
    else:
        if template_from not in config.primitives:
            raise KeyError(f"Template primitive '{template_from}' does not exist")
        primitive = copy.deepcopy(config.primitives[template_from])
        primitive.is_terminal = is_terminal
        primitive.notes = notes if notes is not None else primitive.notes
    config.primitives[name] = primitive
    if config.start_primitive is None:
        config.start_primitive = name
    if config.reset_primitive is None:
        config.reset_primitive = name
    if connect_from is not None:
        transition_type = connect_transition_type or "always"
        add_transition(
            config,
            source=connect_from,
            target=name,
            transition_type=transition_type,
            parameters=connect_parameters,
        )
    return validate_mpnet_config(config)


def remove_primitive(config: ManipulationPrimitiveNetConfig, name: str) -> ManipulationPrimitiveNetConfig:
    """Remove a primitive and any transitions referencing it."""
    if name not in config.primitives:
        raise KeyError(f"Primitive '{name}' does not exist")
    if len(config.primitives) == 1:
        raise ValueError("Cannot remove the only primitive in the MP-Net")
    del config.primitives[name]
    config.transitions = [
        transition for transition in config.transitions if transition.source != name and transition.target != name
    ]
    remaining_names = list(config.primitives)
    if config.start_primitive == name:
        config.start_primitive = remaining_names[0]
    if config.reset_primitive == name:
        config.reset_primitive = remaining_names[0]
    return validate_mpnet_config(config)


def set_start_primitive(config: ManipulationPrimitiveNetConfig, name: str) -> ManipulationPrimitiveNetConfig:
    """Set the start primitive."""
    if name not in config.primitives:
        raise KeyError(f"Primitive '{name}' does not exist")
    config.start_primitive = name
    return validate_mpnet_config(config)


def set_reset_primitive(config: ManipulationPrimitiveNetConfig, name: str) -> ManipulationPrimitiveNetConfig:
    """Set the reset primitive."""
    if name not in config.primitives:
        raise KeyError(f"Primitive '{name}' does not exist")
    config.reset_primitive = name
    return validate_mpnet_config(config)


def set_terminal(config: ManipulationPrimitiveNetConfig, primitive_name: str, is_terminal: bool) -> ManipulationPrimitiveNetConfig:
    """Update the terminal flag on one primitive."""
    config.primitives[primitive_name].is_terminal = is_terminal
    return validate_mpnet_config(config)


def attach_policy(
    config: ManipulationPrimitiveNetConfig,
    primitive_name: str,
    policy_path: str,
    policy_overwrites: dict[str, Any] | None = None,
) -> ManipulationPrimitiveNetConfig:
    """Attach an existing pretrained policy config to one primitive."""
    primitive = config.primitives[primitive_name]
    primitive.policy = PreTrainedConfig.from_pretrained(policy_path, local_files_only=True)
    primitive.policy.pretrained_path = Path(policy_path)
    if policy_overwrites:
        primitive.policy_overwrites = dict(policy_overwrites)
    return validate_mpnet_config(config)


def set_policy_overwrites(
    config: ManipulationPrimitiveNetConfig,
    primitive_name: str,
    policy_overwrites: dict[str, Any],
) -> ManipulationPrimitiveNetConfig:
    """Replace policy overwrite metadata for one primitive."""
    config.primitives[primitive_name].policy_overwrites = dict(policy_overwrites)
    return validate_mpnet_config(config)


def set_primitive_notes(
    config: ManipulationPrimitiveNetConfig,
    primitive_name: str,
    notes: str | None,
) -> ManipulationPrimitiveNetConfig:
    """Replace human notes on one primitive."""
    config.primitives[primitive_name].notes = notes
    return validate_mpnet_config(config)


def set_learnable_axes(
    config: ManipulationPrimitiveNetConfig,
    primitive_name: str,
    axes: list[str] | dict[str, str | None],
    *,
    robot_name: str | None = None,
    mode: str = "relative",
) -> ManipulationPrimitiveNetConfig:
    """Mark task-frame axes as learnable or fixed."""
    primitive = config.primitives[primitive_name]
    _, frame = _resolve_frame(primitive, robot_name)
    if isinstance(axes, dict):
        axis_modes = axes
    else:
        axis_modes = {axis: mode for axis in axes}
    next_policy_mode = list(frame.policy_mode)
    for axis_name, axis_mode in axis_modes.items():
        if axis_name not in AXIS_TO_INDEX:
            raise KeyError(f"Unsupported task-frame axis '{axis_name}'")
        idx = AXIS_TO_INDEX[axis_name]
        if axis_mode is None or str(axis_mode).lower() == "fixed":
            next_policy_mode[idx] = None
        elif str(axis_mode).lower() == "relative":
            next_policy_mode[idx] = PolicyMode.RELATIVE
        elif str(axis_mode).lower() == "absolute":
            next_policy_mode[idx] = PolicyMode.ABSOLUTE
        else:
            raise ValueError(f"Unsupported policy mode '{axis_mode}'")
    frame.policy_mode = next_policy_mode
    _validate_task_frame(frame)
    return validate_mpnet_config(config)


def set_axis_targets(
    config: ManipulationPrimitiveNetConfig,
    primitive_name: str,
    targets: dict[str, float],
    *,
    robot_name: str | None = None,
) -> ManipulationPrimitiveNetConfig:
    """Update one or more task-frame target coordinates."""
    primitive = config.primitives[primitive_name]
    _, frame = _resolve_frame(primitive, robot_name)
    next_target = list(frame.target)
    for axis_name, value in targets.items():
        if axis_name not in AXIS_TO_INDEX:
            raise KeyError(f"Unsupported task-frame axis '{axis_name}'")
        next_target[AXIS_TO_INDEX[axis_name]] = float(value)
    frame.target = next_target
    _validate_task_frame(frame)
    return validate_mpnet_config(config)


def add_transition(
    config: ManipulationPrimitiveNetConfig,
    source: str,
    target: str,
    transition_type: str,
    parameters: dict[str, Any] | None = None,
) -> ManipulationPrimitiveNetConfig:
    """Append one typed transition."""
    if source not in config.primitives:
        raise KeyError(f"Primitive '{source}' does not exist")
    if target not in config.primitives:
        raise KeyError(f"Primitive '{target}' does not exist")
    parameters = parameters or {}
    transition_cls = TRANSITION_TYPES.get(transition_type)
    if transition_cls is None:
        raise KeyError(f"Unknown transition type '{transition_type}'")
    transition = transition_cls(source=source, target=target, **parameters)
    config.transitions.append(transition)
    return validate_mpnet_config(config)


def remove_transition(
    config: ManipulationPrimitiveNetConfig,
    *,
    index: int | None = None,
    source: str | None = None,
    target: str | None = None,
) -> ManipulationPrimitiveNetConfig:
    """Remove one transition by index or by source/target match."""
    if index is not None:
        if index < 0 or index >= len(config.transitions):
            raise IndexError(f"Transition index {index} is out of range")
        del config.transitions[index]
        return validate_mpnet_config(config)
    remaining = [
        transition
        for transition in config.transitions
        if not (
            (source is None or transition.source == source)
            and (target is None or transition.target == target)
        )
    ]
    if len(remaining) == len(config.transitions):
        raise ValueError("No transition matched the requested removal criteria")
    config.transitions = remaining
    return validate_mpnet_config(config)


def apply_edit(
    config: ManipulationPrimitiveNetConfig,
    operation: str,
    arguments: dict[str, Any],
) -> ManipulationPrimitiveNetConfig:
    """Dispatch one structured MP-Net edit operation."""
    handlers = {
        "add_primitive": lambda: add_primitive(config, **arguments),
        "remove_primitive": lambda: remove_primitive(config, **arguments),
        "set_start": lambda: set_start_primitive(config, **arguments),
        "set_reset": lambda: set_reset_primitive(config, **arguments),
        "set_terminal": lambda: set_terminal(config, **arguments),
        "attach_policy": lambda: attach_policy(config, **arguments),
        "set_policy_overwrites": lambda: set_policy_overwrites(config, **arguments),
        "set_learnable_axes": lambda: set_learnable_axes(config, **arguments),
        "set_axis_targets": lambda: set_axis_targets(config, **arguments),
        "add_transition": lambda: add_transition(config, **arguments),
        "remove_transition": lambda: remove_transition(config, **arguments),
        "set_primitive_notes": lambda: set_primitive_notes(config, **arguments),
        "validate": lambda: validate_mpnet_config(config),
    }
    if operation not in handlers:
        raise KeyError(f"Unsupported MP-Net edit operation '{operation}'")
    return handlers[operation]()
