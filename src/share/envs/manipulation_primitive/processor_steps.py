from __future__ import annotations

from collections import deque
import math
from dataclasses import dataclass, field
from typing import Any, Literal

import einops
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.processor.hil_processor import TELEOP_ACTION_KEY, GRIPPER_KEY
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from lerobot.teleoperators import TeleopEvents
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
from share.envs.manipulation_primitive.task_frame import ControlMode, ControlSpace, PolicyMode, TaskFrame, \
    TASK_FRAME_AXIS_NAMES
from share.envs.utils import check_delta_teleoperator


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for nested in value.values():
            tensor = _first_tensor(nested)
            if tensor is not None:
                return tensor
    return None


def _flatten_nested_policy_action(
    action: dict[str, dict[str, Any]],
    task_frame: dict[str, TaskFrame],
    gripper_enable: dict[str, bool],
    like: Any | None = None,
) -> torch.Tensor:
    tensor = _first_tensor(action)
    if tensor is None and like is not None:
        tensor = _first_tensor(like) if isinstance(like, dict) else like if isinstance(like, torch.Tensor) else None
    dtype = tensor.dtype if isinstance(tensor, torch.Tensor) else torch.float32
    device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")

    values: list[torch.Tensor] = []
    for name, frame in task_frame.items():
        robot_action = action.get(name, {})
        for key in policy_action_keys_for_robot(frame, gripper_enable[name]):
            if key not in robot_action:
                raise ValueError(f"Missing policy action key '{name}.{key}' while flattening action dict")
            values.append(torch.as_tensor(robot_action[key], dtype=dtype, device=device).reshape(1))

    if not values:
        return torch.empty(0, dtype=dtype, device=device)
    return torch.cat(values)


def _rotation_from_extrinsic_xyz(rx: float, ry: float, rz: float) -> Rotation:
    """Build a rotation from extrinsic XYZ angles using explicit axis composition."""

    # Extrinsic XYZ composition applies X then Y then Z in the world frame.
    # Rotation multiplication order in scipy is right-to-left application.
    rot_x = Rotation.from_rotvec([rx, 0.0, 0.0])
    rot_y = Rotation.from_rotvec([0.0, ry, 0.0])
    rot_z = Rotation.from_rotvec([0.0, 0.0, rz])
    return rot_z * rot_y * rot_x


def _euler_xyz_from_rotation(rotation: Rotation) -> list[float]:
    """Convert a ``Rotation`` back to XYZ Euler angles in radians."""

    return rotation.as_euler("xyz", degrees=False).tolist()


def policy_action_keys_for_robot(frame: TaskFrame, gripper_enable: bool) -> list[str]:
    keys = list(frame.policy_action_keys())
    if gripper_enable:
        keys.append(f"{GRIPPER_KEY}.pos")
    return keys


def rotation_component_keys(frame: TaskFrame, absolute_rot_axes: list[int]) -> list[str]:
    if len(absolute_rot_axes) == 1:
        axis_name = frame.action_key_for_axis(absolute_rot_axes[0]).removesuffix(".pos")
        return [f"{axis_name}.pos.cos", f"{axis_name}.pos.sin"]
    if len(absolute_rot_axes) == 2:
        return ["rotation.s2.x", "rotation.s2.y", "rotation.s2.z"]
    if len(absolute_rot_axes) == 3:
        return [
            "rotation.so3.a1.x",
            "rotation.so3.a1.y",
            "rotation.so3.a1.z",
            "rotation.so3.a2.x",
            "rotation.so3.a2.y",
            "rotation.so3.a2.z",
        ]
    return []


@dataclass
@ProcessorStepRegistry.register("to_nested_action")
class ToNestedActionProcessorStep(ProcessorStep):
    """Convert the flat policy action tensor into a per-robot keyed dict."""

    task_frame: dict[str, TaskFrame] = field(default_factory=dict)
    gripper_enable: dict[str, bool] = field(default_factory=dict)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if isinstance(action, dict):
            return transition

        action_tensor = torch.as_tensor(action)
        nested_action: dict[str, dict[str, torch.Tensor]] = {}
        idx = 0
        for name, frame in self.task_frame.items():
            robot_action: dict[str, torch.Tensor] = {}
            for key in policy_action_keys_for_robot(frame, self.gripper_enable[name]):
                if idx >= action_tensor.numel():
                    raise ValueError("Policy action tensor is shorter than expected for the configured action schema")
                robot_action[key] = action_tensor[idx]
                idx += 1
            nested_action[name] = robot_action

        if idx != action_tensor.numel():
            raise ValueError("Policy action tensor has trailing values beyond the configured action schema")

        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = nested_action
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("match_teleop_to_policy_action")
class MatchTeleopToPolicyActionProcessorStep(ProcessorStep):
    """Map raw teleop commands into the keyed policy learning-space action format."""

    teleoperators: dict[str, Any] = field(default_factory=dict)
    task_frame: dict[str, TaskFrame] = field(default_factory=dict)
    kinematics: dict[str, Any] = field(default_factory=dict)
    joint_names: dict[str, list[str]] = field(default_factory=dict)
    use_virtual_reference: dict[str, bool] = field(default_factory=dict)
    gripper_enable: dict[str, bool] = field(default_factory=dict)

    _is_delta_teleoperator: dict[str, bool] = field(default_factory=dict, init=False)
    _virtual_task_pose: dict[str, list[float]] = field(default_factory=dict, init=False)
    _virtual_joint_target: dict[str, dict[str, float]] = field(default_factory=dict, init=False)
    _prev_fk_pose: dict[str, list[float]] = field(default_factory=dict, init=False)
    _prev_joint_state: dict[str, dict[str, float]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._is_delta_teleoperator = check_delta_teleoperator(self.teleoperators)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        complementary_data = dict(new_transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        teleop_action_dict = complementary_data.get(TELEOP_ACTION_KEY)
        if not isinstance(teleop_action_dict, dict):
            return new_transition

        converted_actions: dict[str, dict[str, float]] = {}
        for name, teleop_action in teleop_action_dict.items():
            frame = self.task_frame.get(name)
            if frame is None:
                continue

            if self._is_delta_teleoperator.get(name, False):
                converted = self._map_delta_teleop(name, frame, teleop_action, transition)
            else:
                converted = self._map_absolute_joint_teleop(name, frame, teleop_action)

            if self.gripper_enable[name] and f"{GRIPPER_KEY}.pos" in teleop_action:
                converted[f"{GRIPPER_KEY}.pos"] = teleop_action[f"{GRIPPER_KEY}.pos"]

            converted_actions[name] = converted

        complementary_data[TELEOP_ACTION_KEY] = converted_actions
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        return new_transition

    def _map_delta_teleop(self, name: str, frame: TaskFrame, teleop_action: Any, transition: EnvTransition) -> dict[str, float]:
        deltas = self._extract_delta_action(teleop_action)

        # ik to get from integrated deltas to joints
        if frame.space == ControlSpace.JOINT:
            solver = self._require_solver(name)
            base_pose = self._integration_base_pose(name, frame, transition)
            pose_target = [base_pose[i] + deltas[i] for i in range(6)]
            joint_target = solver.inverse_kinematics(pose_target)
            base_joint_state = self._integration_base_joint_state(name, transition)
            encoded: dict[str, float] = {}
            for axis in frame.learnable_axis_indices:
                joint_name = self.joint_names[name][axis]
                joint_value = joint_target[joint_name]
                if frame.policy_mode[axis] == PolicyMode.RELATIVE:
                    joint_value -= float(base_joint_state.get(joint_name, joint_value))
                encoded[frame.action_key_for_axis(axis)] = joint_value

            # update virtual
            if any(frame.policy_mode[axis] == PolicyMode.ABSOLUTE for axis in frame.learnable_axis_indices):
                learnable_joints = [self.joint_names[name][axis] for axis in frame.learnable_axis_indices]
                self._virtual_joint_target[name] = {name: joint_target[name] for name in learnable_joints}

            return encoded

        source_pose = deltas
        if any(
            frame.control_mode[axis] == ControlMode.POS and frame.policy_mode[axis] == PolicyMode.ABSOLUTE
            for axis in frame.learnable_axis_indices
        ):
            base_pose = self._integration_base_pose(name, frame, transition)
            source_pose = [base_pose[i] + deltas[i] for i in range(6)]
            self._virtual_task_pose[name] = source_pose

        return self._encode_learning_space(frame, source_pose)

    def _map_absolute_joint_teleop(self, name: str, frame: TaskFrame, teleop_action: Any) -> dict[str, float]:
        joint_state = self._extract_joint_action(teleop_action)
        if frame.space == ControlSpace.JOINT:
            prev_joint_state = self._prev_joint_state.get(name, joint_state)
            self._prev_joint_state[name] = dict(joint_state)
            encoded: dict[str, float] = {}
            for axis in frame.learnable_axis_indices:
                joint_name = self._joint_name_for_axis(name, axis)
                joint_value = float(joint_state.get(joint_name, joint_state.get(f"joint_{axis + 1}", 0.0)))
                if frame.policy_mode[axis] == PolicyMode.RELATIVE:
                    joint_value -= float(prev_joint_state.get(joint_name, joint_value))
                encoded[frame.action_key_for_axis(axis)] = joint_value
            return encoded

        solver = self._require_solver(name)
        pose = solver.forward_kinematics(joint_state)
        prev_pose = self._prev_fk_pose.get(name, pose)
        self._prev_fk_pose[name] = pose

        source = []
        for axis in range(6):
            if frame.policy_mode[axis] == PolicyMode.RELATIVE:
                source.append(pose[axis] - prev_pose[axis])
            else:
                source.append(pose[axis])

        return self._encode_learning_space(frame, source)

    def _encode_learning_space(self, frame: TaskFrame, source_pose: list[float]) -> dict[str, float]:
        values: list[float] = []
        absolute_rot_axes = [
            axis for axis in frame.learnable_axis_indices if frame.is_absolute_rotation_axis(axis)
        ]

        for axis in frame.learnable_axis_indices:
            control_mode = frame.control_mode[axis]
            policy_mode = frame.policy_mode[axis]
            if axis in absolute_rot_axes:
                continue
            if control_mode in {ControlMode.VEL, ControlMode.WRENCH}:
                values.append(source_pose[axis])
            elif axis < 3 or policy_mode == PolicyMode.RELATIVE:
                values.append(source_pose[axis])

        if absolute_rot_axes:
            rot = [source_pose[3], source_pose[4], source_pose[5]]
            if len(absolute_rot_axes) == 1:
                angle = rot[absolute_rot_axes[0] - 3]
                values.extend([math.cos(angle), math.sin(angle)])
            elif len(absolute_rot_axes) == 2:
                matrix = _rotation_from_extrinsic_xyz(*rot).as_matrix()
                values.extend(matrix[:, 0].tolist())
            else:
                matrix = _rotation_from_extrinsic_xyz(*rot).as_matrix()
                values.extend(np.concatenate([matrix[:, 0], matrix[:, 1]]).tolist())

        keys = frame.policy_action_keys()
        if len(keys) != len(values):
            raise ValueError("Learning-space key/value mismatch while encoding teleop action")
        return {key: float(value) for key, value in zip(keys, values, strict=True)}

    def _integration_base_pose(self, name: str, frame: TaskFrame, transition: EnvTransition) -> list[float]:
        use_virtual = self.use_virtual_reference[name] if isinstance(self.use_virtual_reference, dict) else self.use_virtual_reference
        if use_virtual and name in self._virtual_task_pose:
            return self._virtual_task_pose[name]

        observation = transition.get(TransitionKey.OBSERVATION)
        if isinstance(observation, dict):
            obs_pose = []
            for axis_name in TASK_FRAME_AXIS_NAMES:
                key = f"{name}.{axis_name}.ee_pos"
                if key not in observation:
                    obs_pose = []
                    break
                obs_pose.append(observation[key])
            if len(obs_pose) == 6:
                return obs_pose

        return list(frame.target)

    def _integration_base_joint_state(self, name: str, transition: EnvTransition) -> dict[str, float]:
        use_virtual = self.use_virtual_reference[name] if isinstance(self.use_virtual_reference, dict) else self.use_virtual_reference
        if use_virtual and name in self._virtual_joint_target:
            return dict(self._virtual_joint_target[name])

        observation = transition.get(TransitionKey.OBSERVATION)
        if isinstance(observation, dict):
            joint_state: dict[str, float] = {}
            for joint_name in self.joint_names.get(name, []):
                key = f"{name}.{joint_name}.pos"
                if key in observation:
                    joint_state[joint_name] = observation[key]
            if joint_state:
                return joint_state

        return dict(self._prev_joint_state.get(name, {}))

    def _require_solver(self, name: str) -> Any:
        solver = self.kinematics.get(name)
        if solver is None:
            raise ValueError(f"Missing kinematics solver for '{name}'")
        return solver

    @staticmethod
    def _extract_delta_action(teleop_action: Any) -> list[float]:
        if isinstance(teleop_action, dict):
            return [
                float(teleop_action.get("delta_x", teleop_action.get("x.vel", 0.0))),
                float(teleop_action.get("delta_y", teleop_action.get("y.vel", 0.0))),
                float(teleop_action.get("delta_z", teleop_action.get("z.vel", 0.0))),
                float(teleop_action.get("delta_rx", teleop_action.get("wx.vel", 0.0))),
                float(teleop_action.get("delta_ry", teleop_action.get("wy.vel", 0.0))),
                float(teleop_action.get("delta_rz", teleop_action.get("wz.vel", 0.0))),
            ]
        return [float(v) for v in teleop_action][:6]

    @staticmethod
    def _extract_joint_action(teleop_action: Any) -> dict[str, float]:
        if isinstance(teleop_action, dict):
            joint_state: dict[str, float] = {}
            for key, value in teleop_action.items():
                if key.endswith(".pos"):
                    joint_state[key.removesuffix(".pos")] = float(value)
                elif key.endswith(".q"):
                    joint_state[key.removesuffix(".q")] = float(value)
                elif "." not in key:
                    joint_state[key] = float(value)
            return joint_state
        return {f"joint_{i + 1}": float(v) for i, v in enumerate(teleop_action)}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("task_frame_intervention_action_processor")
class InterventionActionProcessorStep(ProcessorStep):
    """Merge keyed learning-space actions and project them into full robot actions."""

    teleoperators: dict[str, Any] = field(default_factory=dict)
    task_frame: dict[str, TaskFrame] = field(default_factory=dict)
    gripper_enable: bool | dict[str, bool] = False
    gripper_static_pos: float | dict[str, float] = 0.0

    def __post_init__(self):
        self._disable_torque_on_intervention = {name: hasattr(teleop, "bus") for name, teleop in self.teleoperators.items()}
        self._intervention_occurred = False

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        policy_actions = transition.get(TransitionKey.ACTION)
        if not isinstance(policy_actions, dict):
            raise TypeError(f"Action should be a dict, got {type(policy_actions)}")

        new_transition = transition.copy()
        info = dict(new_transition.get(TransitionKey.INFO) or {})
        complementary_data = dict(new_transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        teleop_action_dict = complementary_data.get(TELEOP_ACTION_KEY)
        is_intervention = bool(info.get(TeleopEvents.IS_INTERVENTION, False))

        self._intervention_occurred = self._intervention_occurred | is_intervention
        if self._intervention_occurred and not is_intervention:
            info[TeleopEvents.INTERVENTION_COMPLETED] = True

        if is_intervention and isinstance(teleop_action_dict, dict):
            source_actions = teleop_action_dict
            for name in teleop_action_dict:
                if self._disable_torque_on_intervention.get(name, False):
                    self.teleoperators[name].disable_torque()
        else:
            source_actions = policy_actions
            if self._intervention_occurred:
                for name, teleop in self.teleoperators.items():
                    if self._disable_torque_on_intervention.get(name, False):
                        teleop.enable_torque()
            else:
                for teleop_name, teleop in self.teleoperators.items():
                    teleop.send_feedback(self._map_to_teleop_action(source_actions, teleop_name))

        full_action_dict: dict[str, dict[str, float]] = {}
        for name, frame in self.task_frame.items():
            full_action = self._project_policy_action(frame, source_actions[name])
            if self.gripper_enable[name] and f"{GRIPPER_KEY}.pos" in source_actions[name]:
                full_action[f"{GRIPPER_KEY}.pos"] = source_actions[name][f"{GRIPPER_KEY}.pos"]
            full_action_dict[name] = full_action

        complementary_data[TELEOP_ACTION_KEY] = _flatten_nested_policy_action(
            source_actions,
            task_frame=self.task_frame,
            gripper_enable=self.gripper_enable,
            like=policy_actions,
        )

        new_transition[TransitionKey.ACTION] = full_action_dict
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        new_transition[TransitionKey.INFO] = info
        return new_transition

    def _project_policy_action(self, frame: TaskFrame, encoded_action: dict[str, Any]) -> dict[str, float]:
        full_target = {
            frame.action_key_for_axis(axis): float(frame.target[axis])
            for axis in range(len(frame.target))
        }
        absolute_rot_axes = [axis for axis in frame.learnable_axis_indices if frame.is_absolute_rotation_axis(axis)]

        for axis in frame.learnable_axis_indices:
            if axis in absolute_rot_axes:
                continue
            key = frame.action_key_for_axis(axis)
            if key not in encoded_action:
                raise ValueError(f"Missing learning-space action key '{key}' for task-frame projection")
            value = encoded_action[key]
            if frame.control_mode[axis] in {ControlMode.VEL, ControlMode.WRENCH}:
                value = self._bound_differential_axis(frame, axis, value)
            full_target[key] = float(value)

        if absolute_rot_axes:
            rotation_keys = rotation_component_keys(frame, absolute_rot_axes)
            rotation_raw = []
            for key in rotation_keys:
                if key not in encoded_action:
                    raise ValueError(f"Missing rotation learning-space key '{key}' for task-frame projection")
                rotation_raw.append(encoded_action[key])
            rotation_values, _ = self._decode_absolute_rotation(absolute_rot_axes, rotation_raw)
            for axis in absolute_rot_axes:
                full_target[frame.action_key_for_axis(axis)] = float(rotation_values[axis - 3])

        return full_target

    def _map_to_teleop_action(self, policy_action: dict[str, dict[str, Any]], name: str) -> dict[str, float]:
        if name not in policy_action or name not in self.teleoperators:
            return {}

        action_features = getattr(self.teleoperators[name], "action_features", {})
        if isinstance(action_features, dict) and isinstance(action_features.get("names"), dict):
            feature_names = list(action_features["names"].keys())
        elif isinstance(action_features, dict):
            feature_names = list(action_features.keys())
        else:
            feature_names = []

        aliases = {
            "delta_x": "x.vel",
            "delta_y": "y.vel",
            "delta_z": "z.vel",
            "delta_rx": "rx.vel",
            "delta_ry": "ry.vel",
            "delta_rz": "rz.vel",
            "gripper": f"{GRIPPER_KEY}.pos",
        }
        robot_action = policy_action[name]
        teleop_action: dict[str, float] = {}
        for feature_name in feature_names:
            key = aliases.get(feature_name, feature_name)
            if key in robot_action:
                teleop_action[feature_name] = robot_action[key]
        return teleop_action

    @staticmethod
    def _bound_differential_axis(frame: TaskFrame, axis: int, value: float) -> float:
        if frame.min_target is not None and frame.max_target is not None:
            scale = max(abs(frame.min_target[axis]), abs(frame.max_target[axis]))
            if scale > 0:
                return math.tanh(value) * scale
        return math.tanh(value)

    def _decode_absolute_rotation(self, absolute_rot_axes: list[int], raw: list[float]) -> tuple[list[float], int]:
        rot = [0.0, 0.0, 0.0]
        if len(absolute_rot_axes) == 1:
            if len(raw) < 2:
                raise ValueError("S1 rotation representation requires 2 values")
            rot[absolute_rot_axes[0] - 3] = math.atan2(raw[1], raw[0])
            return rot, 2
        if len(absolute_rot_axes) == 2:
            if len(raw) < 3:
                raise ValueError("S2 rotation representation requires 3 values")
            direction = np.asarray(raw[:3], dtype=float)
            norm = np.linalg.norm(direction)
            direction = np.array([1.0, 0.0, 0.0], dtype=float) if norm < 1e-8 else direction / norm
            reference = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(float(np.dot(reference, direction))) > 0.95:
                reference = np.array([0.0, 1.0, 0.0], dtype=float)
            col1 = direction
            col2 = np.cross(reference, col1)
            col2_norm = np.linalg.norm(col2)
            col2 = np.array([0.0, 1.0, 0.0], dtype=float) if col2_norm < 1e-8 else col2 / col2_norm
            col3 = np.cross(col1, col2)
            matrix = np.column_stack([col1, col2, col3])
            rx, ry, rz = Rotation.from_matrix(matrix).as_euler("xyz", degrees=False)
            return [float(rx), float(ry), float(rz)], 3
        if len(absolute_rot_axes) == 3:
            if len(raw) < 6:
                raise ValueError("SO(3) 6D representation requires 6 values")
            matrix = self._rotation_6d_to_matrix(raw[:6])
            euler = Rotation.from_matrix(matrix).as_euler("xyz", degrees=False)
            return euler.tolist(), 6
        raise ValueError(f"Expected 1..3 absolute rotation axes, got {len(absolute_rot_axes)}")

    @staticmethod
    def _rotation_6d_to_matrix(raw: list[float]) -> list[list[float]]:
        a1 = np.asarray(raw[:3], dtype=float)
        a2 = np.asarray(raw[3:6], dtype=float)

        def normalize(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
            n = float(np.linalg.norm(v))
            if n < 1e-8:
                return fallback
            return v / n

        b1 = normalize(a1, np.array([1.0, 0.0, 0.0], dtype=float))
        u2 = a2 - float(np.dot(a2, b1)) * b1
        fallback = np.array([0.0, 1.0, 0.0], dtype=float) if abs(float(b1[0])) > 0.9 else np.array([1.0, 0.0, 0.0], dtype=float)
        b2 = normalize(u2, normalize(fallback - float(np.dot(fallback, b1)) * b1, np.array([0.0, 1.0, 0.0], dtype=float)))
        b3 = np.cross(b1, b2)
        return np.column_stack([b1, b2, b3]).tolist()

    def _resolved_gripper_static_pos(self, name: str) -> float:
        if isinstance(self.gripper_static_pos, dict):
            return float(self.gripper_static_pos.get(name, 0.0))
        return float(self.gripper_static_pos)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def reset(self) -> None:
        self._intervention_occurred = False


@dataclass
@ProcessorStepRegistry.register("discretize_gripper_processor_v2")
class DiscretizeGripperProcessorStep(ProcessorStep):
    """Discretize gripper actions using a per-robot internal gripper state."""

    discretize: dict[str, bool] = field(default_factory=dict)
    min_pos: dict[str, float] = field(default_factory=dict)
    max_pos: dict[str, float] = field(default_factory=dict)
    threshold: dict[str, float] = field(default_factory=dict)
    mode: dict[str, Literal["state", "pulse"]] = field(default_factory=dict)

    _robot_names: list[str] = field(default_factory=list, init=False)
    _gripper_state: dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        min_keys = set(self.min_pos.keys()) if isinstance(self.min_pos, dict) else set()
        max_keys = set(self.max_pos.keys()) if isinstance(self.max_pos, dict) else set()
        if min_keys and max_keys and min_keys != max_keys:
            raise ValueError("DiscretizeGripperProcessorStep requires min_pos and max_pos to have the same robot keys")

        if min_keys:
            self._robot_names = sorted(min_keys)
        elif max_keys:
            self._robot_names = sorted(max_keys)
        else:
            self._robot_names = []

        self.reset()

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, dict):
            return transition

        new_transition = transition.copy()
        new_action: dict[str, dict[str, Any]] = {}
        for name, robot_action in action.items():
            if not isinstance(robot_action, dict) or not self.discretize.get(name, False):
                new_action[name] = robot_action
                continue

            robot_action_out = dict(robot_action)
            if f"{GRIPPER_KEY}.pos" not in robot_action_out:
                new_action[name] = robot_action_out
                continue

            # initialize gripper state
            if name not in self._gripper_state:
                self._gripper_state[name] = self.min_pos.get(name, 0.0)
                if name not in self._robot_names:
                    self._robot_names.append(name)

            # update gripper state
            input_val = robot_action_out[f"{GRIPPER_KEY}.pos"]
            mode = self.mode.get(name, "state")
            threshold = self.threshold.get(name, 0.5)
            min_pos = self.min_pos.get(name, 0.0)
            max_pos = self.max_pos.get(name, 1.0)
            if mode == "pulse":
                if input_val > threshold:
                    self._gripper_state[name] = max_pos
                elif input_val < -threshold:
                    self._gripper_state[name] = min_pos
            elif mode == "state":
                if input_val > threshold:
                    self._gripper_state[name] = max_pos
                elif input_val < threshold:
                    self._gripper_state[name] = min_pos
            else:
                raise ValueError(f"Unsupported gripper discretization mode '{mode}'")

            robot_action_out[f"{GRIPPER_KEY}.pos"] = float(self._gripper_state[name])
            new_action[name] = robot_action_out

        new_transition[TransitionKey.ACTION] = new_action
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "discretize": self.discretize,
            "min_pos": self.min_pos,
            "max_pos": self.max_pos,
            "threshold": self.threshold,
            "mode": self.mode,
        }

    def reset(self) -> None:
        self._gripper_state = {name: self.min_pos.get(name, 0.0) for name in self._robot_names}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("to_joint_action_processor")
class ToJointActionProcessorStep(ProcessorStep):
    """Convert nested task-frame robot actions into nested joint robot actions when needed."""

    is_task_frame_robot: dict[str, bool] = field(default_factory=dict)
    task_frame: dict[str, TaskFrame] = field(default_factory=dict)
    kinematics: dict[str, Any] = field(default_factory=dict)
    joint_names: dict[str, list[str]] = field(default_factory=dict)
    use_virtual_reference: bool | dict[str, bool] = True

    _virtual_task_pose: dict[str, list[float]] = field(default_factory=dict, init=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Convert task-space robot action dicts to joint-space robot action dicts."""
        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, dict):
            return transition

        new_transition = transition.copy()
        joint_action: dict[str, dict[str, float]] = {}

        for name, robot_action in action.items():
            frame = self.task_frame.get(name)
            if frame is None:
                joint_action[name] = dict(robot_action) if isinstance(robot_action, dict) else robot_action
                continue

            if self.is_task_frame_robot.get(name, False):
                joint_action[name] = dict(robot_action)
                continue

            if not isinstance(robot_action, dict):
                raise TypeError(f"Task-frame action for '{name}' must be a dict, got {type(robot_action)}")

            task_target = self._task_target_from_action(name, frame, robot_action)
            absolute_target = self._integrate_relative_axes(name, frame, task_target, transition)
            bounded_target = self._clamp_target(frame, absolute_target)

            solver = self.kinematics.get(name)
            if solver is None:
                raise ValueError(f"Missing kinematics solver for joint-only robot '{name}'")

            try:
                ik_solution = solver.inverse_kinematics(bounded_target)
            except Exception as exc:  # pragma: no cover - exercised with mock solver in unit tests
                raise ValueError(f"IK failed for '{name}': {exc}") from exc

            robot_joint_action: dict[str, float] = {}
            for joint_name in self.joint_names.get(name, []):
                if joint_name not in ik_solution:
                    raise ValueError(f"IK solution for '{name}' missing joint '{joint_name}'")
                robot_joint_action[f"{joint_name}.pos"] = float(ik_solution[joint_name])

            if f"{GRIPPER_KEY}.pos" in robot_action:
                robot_joint_action[f"{GRIPPER_KEY}.pos"] = robot_action[f"{GRIPPER_KEY}.pos"]

            joint_action[name] = robot_joint_action
            self._virtual_task_pose[name] = bounded_target

        new_transition[TransitionKey.ACTION] = joint_action
        return new_transition

    def _task_target_from_action(self, name: str, frame: TaskFrame, robot_action: dict[str, Any]) -> list[float]:
        task_target: list[float] = []
        for axis in range(len(frame.target)):
            key = frame.action_key_for_axis(axis)
            if key not in robot_action:
                raise ValueError(f"Missing task-frame action key '{name}.{key}' for joint conversion")
            task_target.append(robot_action[key])
        return task_target

    def _integrate_relative_axes(
        self,
        name: str,
        frame: TaskFrame,
        task_target: list[float],
        transition: EnvTransition,
    ) -> list[float]:
        """Integrate relative POS axes on top of the current/base task pose."""
        base_pose = self._base_pose(name, frame, transition)
        out = list(task_target)
        for axis in frame.learnable_axis_indices:
            if frame.control_mode[axis] == ControlMode.POS and frame.policy_mode[axis] == PolicyMode.RELATIVE:
                out[axis] = base_pose[axis] + task_target[axis]
        return out

    def _base_pose(self, name: str, frame: TaskFrame, transition: EnvTransition) -> list[float]:
        """Resolve integration base pose from virtual state, observation, or default target."""
        use_virtual = self.use_virtual_reference[name] if isinstance(self.use_virtual_reference, dict) else self.use_virtual_reference
        if use_virtual and name in self._virtual_task_pose:
            return list(self._virtual_task_pose[name])

        observation = transition.get(TransitionKey.OBSERVATION)
        if isinstance(observation, dict):
            obs_pose = []
            for axis_name in TASK_FRAME_AXIS_NAMES:
                key = f"{name}.{axis_name}.ee_pos"
                if key not in observation:
                    obs_pose = []
                    break
                obs_pose.append(observation[key])
            if len(obs_pose) == 6:
                return obs_pose

        return list(frame.target)

    @staticmethod
    def _clamp_target(frame: TaskFrame, target: list[float]) -> list[float]:
        """Clamp task target to configured min/max bounds when available."""
        if frame.min_target is None or frame.max_target is None:
            return target
        return [max(frame.min_target[i], min(frame.max_target[i], target[i])) for i in range(len(target))]

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Leave feature specs unchanged."""
        return features



@dataclass
@ProcessorStepRegistry.register("mp_vanilla_observation_processor")
class VanillaMPObservationProcessorStep(ProcessorStep):
    """Build ``observation.state`` from configured robot modalities and normalize images."""

    device: str = "cpu"

    gripper_enable: dict[str, bool] = field(default_factory=dict)
    add_joint_position_to_observation: dict[str, bool] = field(default_factory=dict)
    add_joint_velocity_to_observation: dict[str, bool] = field(default_factory=dict)
    add_current_to_observation: dict[str, bool] = field(default_factory=dict)

    add_ee_pos_to_observation: dict[str, bool] = field(default_factory=dict)
    ee_pos_axes: dict[str, list[str]] = field(default_factory=dict)

    add_ee_velocity_to_observation: dict[str, bool] = field(default_factory=dict)
    ee_velocity_axes: dict[str, list[str]] = field(default_factory=dict)

    add_ee_wrench_to_observation: dict[str, bool] = field(default_factory=dict)
    ee_wrench_axes: dict[str, list[str]] = field(default_factory=dict)

    stack_frames: dict[str, int] = field(default_factory=dict)

    _prev_obs: dict[str, dict[str, float]] = field(default_factory=dict, init=False)
    _state_buffer: deque[torch.Tensor] = field(init=False)

    def __post_init__(self):
        self._state_buffer = deque(maxlen=self._resolved_stack_frames())

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)
        if not isinstance(observation, dict):
            return transition

        new_transition = transition.copy()
        new_observation = dict(observation)

        state_values = self._collect_state_values(observation)
        if state_values:
            state_tensor = torch.tensor(state_values, dtype=torch.float32)
            stack_frames = self._resolved_stack_frames()

            if stack_frames > 1:
                if not self._state_buffer:
                    for _ in range(stack_frames):
                        self._state_buffer.append(state_tensor)
                else:
                    self._state_buffer.append(state_tensor)
                state_tensor = torch.cat(list(self._state_buffer), dim=-1)

            new_observation[OBS_STATE] = state_tensor

        for key, value in observation.items():
            if "image" in key:
                new_observation[key] = self._process_image(value)

        new_transition[TransitionKey.OBSERVATION] = new_observation
        return new_transition

    def _collect_state_values(self, observation: dict[str, Any]) -> list[float]:
        values: list[float] = []

        for name in sorted(self._robot_names(observation)):
            if self._is_enabled(self.add_joint_position_to_observation, name):
                values.extend(self._joint_values(observation, name, "pos"))

            if self._is_enabled(self.add_joint_velocity_to_observation, name):
                vals = self._joint_values(observation, name, "vel")
                if not vals:
                    vals = self._differentiate(name, observation, self._joint_keys(observation, name, "pos"))
                values.extend(vals)

            if self._is_enabled(self.add_current_to_observation, name):
                values.extend(self._joint_values(observation, name, "current"))

            if self._is_enabled(self.add_ee_pos_to_observation, name):
                values.extend(
                    self._ee_values(
                        observation,
                        name,
                        self._axes(self.ee_pos_axes, name, ".ee_pos"),
                    )
                )

            if self._is_enabled(self.add_ee_velocity_to_observation, name):
                axes = self._axes(self.ee_velocity_axes, name, ".ee_vel")
                vals = self._ee_values(observation, name, axes)
                if not vals:
                    vals = self._differentiate(
                        name,
                        observation,
                        [f"{name}.{axis}.ee_pos" for axis in axes],
                    )
                values.extend(vals)

            if self._is_enabled(self.add_ee_wrench_to_observation, name):
                values.extend(
                    self._ee_values(
                        observation,
                        name,
                        self._axes(self.ee_wrench_axes, name, ".ee_wrench"),
                    )
                )

            if self._is_enabled(self.gripper_enable, name):
                key = f"{name}.gripper.pos"
                if key in observation:
                    values.append(self._to_float(observation[key]))

        self._update_prev_obs(observation)
        return values

    def _process_image(self, image: Any) -> torch.Tensor:
        img = image if isinstance(image, torch.Tensor) else torch.from_numpy(np.asarray(image))

        if img.ndim == 3:
            h, w, c = img.shape
            if c < h and c < w:
                img = einops.rearrange(img, "h w c -> c h w")
        elif img.ndim == 4:
            _, h, w, c = img.shape
            if c < h and c < w:
                img = einops.rearrange(img, "b h w c -> b c h w")
        else:
            raise ValueError(f"Expected image tensor with 3 or 4 dimensions, got shape {tuple(img.shape)}")

        img = img.to(torch.float32)
        return img / 255.0 if img.max() > 1.0 else img

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        new_features = {ft: dict(bucket) for ft, bucket in features.items()}
        obs_features = new_features.get(PipelineFeatureType.OBSERVATION, {})

        state_dim = 0
        for name in sorted(self._robot_names(obs_features)):
            if self._is_enabled(self.add_joint_position_to_observation, name):
                state_dim += len(self._joint_keys(obs_features, name, "pos"))

            if self._is_enabled(self.add_joint_velocity_to_observation, name):
                vel_keys = self._joint_keys(obs_features, name, "vel")
                state_dim += len(vel_keys) if vel_keys else len(self._joint_keys(obs_features, name, "pos"))

            if self._is_enabled(self.add_current_to_observation, name):
                state_dim += len(self._joint_keys(obs_features, name, "current"))

            if self._is_enabled(self.add_ee_pos_to_observation, name):
                state_dim += len(self._ee_keys(obs_features, name, self._axes(self.ee_pos_axes, name, ".ee_pos")))

            if self._is_enabled(self.add_ee_velocity_to_observation, name):
                filter_vel = self._axes(self.ee_velocity_axes, name, ".ee_vel")
                filter_pos = self._axes(self.ee_pos_axes, name, ".ee_pos")
                vel_keys = self._ee_keys(obs_features, name, filter_vel)
                state_dim += len(vel_keys) if vel_keys else len(self._ee_keys(obs_features, name, filter_pos))

            if self._is_enabled(self.add_ee_wrench_to_observation, name):
                state_dim += len(self._ee_keys(obs_features, name, self._axes(self.ee_wrench_axes, name, ".ee_wrench")))

            if self._is_enabled(self.gripper_enable, name) and f"{name}.gripper.pos" in obs_features:
                state_dim += 1

        if state_dim > 0:
            obs_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(state_dim * self._resolved_stack_frames(),),
            )

        for name, feature in obs_features.items():
            if feature.type == FeatureType.VISUAL:
                h, w, c = feature.shape
                if c < h and c < w:
                    obs_features[name].shape = (c, h, w)

        return new_features

    @staticmethod
    def _robot_names(observation: dict[str, Any]) -> set[str]:
        return {
            key.split(".", 1)[0]
            for key in observation
            if "." in key and not key.startswith(OBS_IMAGES)
        }

    @staticmethod
    def _is_enabled(flag_dict: dict[str, bool], name: str) -> bool:
        return bool(flag_dict.get(name, False))

    @staticmethod
    def _axes(axis_dict: dict[str, list[str]], name: str, suffix: str = ".pos") -> list[str]:
        return list(axis_dict.get(name, [f"{ax}{suffix}" for ax in TASK_FRAME_AXIS_NAMES]))

    @staticmethod
    def _to_float(value: Any) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.item()) if value.ndim == 0 else float(value.flatten()[0].item())
        return float(value)

    def _joint_keys(self, observation: dict[str, Any], robot_name: str, suffix: str) -> list[str]:
        prefix = f"{robot_name}."
        return [
            key
            for key in observation
            if key.startswith(prefix)
            and key.endswith(f".{suffix}")
            and ".ee_" not in key
            and ".gripper." not in key
        ]

    def _joint_values(self, observation: dict[str, Any], robot_name: str, suffix: str) -> list[float]:
        return [self._to_float(observation[key]) for key in self._joint_keys(observation, robot_name, suffix)]

    @staticmethod
    def _ee_keys(
        observation: dict[str, Any],
        robot_name: str,
        axis_names: list[str],
    ) -> list[str]:
        return [f"{robot_name}.{axis}" for axis in axis_names if f"{robot_name}.{axis}" in observation]

    def _ee_values(
        self,
        observation: dict[str, Any],
        robot_name: str,
        axis_names: list[str],
    ) -> list[float]:
        return [self._to_float(observation[key]) for key in self._ee_keys(observation, robot_name, axis_names)]

    def _differentiate(self, robot_name: str, observation: dict[str, Any], keys: list[str]) -> list[float]:
        prev = self._prev_obs.get(robot_name, {})
        return [
            self._to_float(observation[key]) - prev.get(key, self._to_float(observation[key]))
            for key in keys
            if key in observation
        ]

    def _update_prev_obs(self, observation: dict[str, Any]) -> None:
        for name in self._robot_names(observation):
            prefix = f"{name}."
            self._prev_obs[name] = {
                key: self._to_float(value)
                for key, value in observation.items()
                if key.startswith(prefix) and "image" not in key
            }

    def _resolved_stack_frames(self) -> int:
        unique = {int(v) for v in self.stack_frames.values()}
        if not unique:
            return 1
        if len(unique) > 1:
            raise ValueError("VanillaMPObservationProcessorStep requires uniform stack_frames across robots.")
        return max(1, unique.pop())

    def reset(self) -> None:
        self._prev_obs.clear()
        self._state_buffer = deque(maxlen=self._resolved_stack_frames())


@dataclass
@ProcessorStepRegistry.register("joints_to_ee_observation")
class JointsToEEObservation(ProcessorStep):
    """Append deterministic end-effector pose channels from joint observations."""

    kinematics: dict[str, Any] = field(default_factory=dict)
    motor_names: dict[str, list[str]] = field(default_factory=dict)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)
        if not isinstance(observation, dict):
            return transition

        is_batched = any(isinstance(v, torch.Tensor) and v.ndim > 0 for v in observation.values())
        batch_size = None
        if is_batched:
            for value in observation.values():
                if isinstance(value, torch.Tensor) and value.ndim > 0:
                    batch_size = int(value.shape[0])
                    break

        new_transition = transition.copy()
        new_observation = dict(observation)

        for robot_name, solver in self.kinematics.items():
            joints = self.motor_names.get(robot_name, [])
            if not joints:
                continue

            if is_batched:
                if batch_size is None:
                    continue
                axis_values = [[] for _ in range(6)]
                for b in range(batch_size):
                    joint_state = self._extract_joint_state(observation, robot_name, joints, index=b)
                    pose = solver.forward_kinematics(joint_state)
                    for axis in range(6):
                        axis_values[axis].append(float(pose[axis]))

                for axis, axis_name in enumerate(TASK_FRAME_AXIS_NAMES):
                    new_observation[f"{robot_name}.{axis_name}.ee_pos"] = torch.tensor(axis_values[axis], dtype=torch.float32)
            else:
                joint_state = self._extract_joint_state(observation, robot_name, joints, index=None)
                pose = solver.forward_kinematics(joint_state)
                for axis, axis_name in enumerate(TASK_FRAME_AXIS_NAMES):
                    new_observation[f"{robot_name}.{axis_name}.ee_pos"] = float(pose[axis])

        new_transition[TransitionKey.OBSERVATION] = new_observation
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Leave feature specs unchanged."""
        return features

    @staticmethod
    def _extract_joint_state(
        observation: dict[str, Any],
        robot_name: str,
        joints: list[str],
        index: int | None,
    ) -> dict[str, float]:
        state: dict[str, float] = {}
        for joint_name in joints:
            key = f"{robot_name}.{joint_name}.pos"
            if key not in observation:
                raise ValueError(f"Missing joint observation key '{key}' for robot '{robot_name}'")
            value = observation[key]
            if isinstance(value, torch.Tensor):
                if index is None:
                    state[joint_name] = float(value.item()) if value.ndim == 0 else float(value[0].item())
                else:
                    state[joint_name] = float(value[index].item())
            else:
                state[joint_name] = float(value)
        return state


@dataclass
@ProcessorStepRegistry.register("relative_frame_observation")
class RelativeFrameObservationProcessor(ProcessorStep):
    """Re-express absolute EE pose channels relative to a per-episode reference pose."""

    enable: bool | dict[str, bool] = True

    _reference_pose: dict[str, list[float]] = field(default_factory=dict, init=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)
        if not isinstance(observation, dict):
            return transition

        new_transition = transition.copy()
        new_observation = dict(observation)
        axis_names = ["x", "y", "z", "wx", "wy", "wz"]

        robot_names = self._robot_names(observation)
        for name in robot_names:
            if not self._enabled(name):
                continue
            pose = self._extract_pose(observation, name)
            if pose is None:
                continue
            reference = self._reference_pose.setdefault(name, pose)

            # Positions are simple vector offsets; orientations are composed on SO(3).
            relative_position = [pose[i] - reference[i] for i in range(3)]
            pose_rot = _rotation_from_extrinsic_xyz(*pose[3:6])
            ref_rot = _rotation_from_extrinsic_xyz(*reference[3:6])
            relative_orientation = _euler_xyz_from_rotation(pose_rot * ref_rot.inv())

            relative_pose = relative_position + relative_orientation
            for axis, axis_name in enumerate(TASK_FRAME_AXIS_NAMES):
                new_observation[f"{name}.{axis_name}.ee_pos"] = relative_pose[axis]

        new_transition[TransitionKey.OBSERVATION] = new_observation
        return new_transition

    def _enabled(self, name: str) -> bool:
        if isinstance(self.enable, dict):
            return bool(self.enable.get(name, False))
        return bool(self.enable)

    @staticmethod
    def _robot_names(observation: dict[str, Any]) -> set[str]:
        names: set[str] = set()
        for key in observation:
            if "." in key:
                names.add(key.split(".", 1)[0])
        return names

    @staticmethod
    def _extract_pose(observation: dict[str, Any], name: str) -> list[float] | None:
        pose = []
        for axis_name in ["x", "y", "z", "wx", "wy", "wz"]:
            key = f"{name}.{axis_name}.ee_pos"
            if key not in observation:
                return None
            value = observation[key]
            pose.append(float(value.item()) if isinstance(value, torch.Tensor) else float(value))
        return pose

    def reset(self) -> None:
        self._reference_pose.clear()

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("relative_frame_action")
class RelativeFrameActionProcessor(ProcessorStep):
    """Pass-through placeholder for relative action transforms (currently identity)."""

    enable: bool | dict[str, bool] = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, dict):
            return transition
        if not any(self._enabled(name) for name in action):
            return transition

        # Current implementation intentionally no-ops numerically for kinematic axis channels.
        # It preserves gripper/non-kinematic channels exactly and remains invertible.
        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = dict(action)
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Leave feature specs unchanged."""
        return features

    def _enabled(self, name: str) -> bool:
        if isinstance(self.enable, dict):
            return bool(self.enable.get(name, False))
        return bool(self.enable)


@dataclass
@ProcessorStepRegistry.register("to_flat_action")
class ToFlatActionProcessorStep(ProcessorStep):
    """Flatten keyed robot actions to the env-facing robot action tensor."""

    robot_action_keys: dict[str, list[str]] = field(default_factory=dict)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, dict):
            return transition

        base_tensor = _first_tensor(action)
        dtype = base_tensor.dtype if isinstance(base_tensor, torch.Tensor) else torch.float32
        device = base_tensor.device if isinstance(base_tensor, torch.Tensor) else torch.device("cpu")
        out_parts: list[torch.Tensor] = []
        for name, keys in self.robot_action_keys.items():
            robot_action = action.get(name, {})
            for key in keys:
                if key not in robot_action:
                    raise ValueError(f"Missing robot action key '{name}.{key}' while flattening action dict")
                out_parts.append(torch.as_tensor(robot_action[key], dtype=dtype, device=device).reshape(1))

        out = torch.cat(out_parts) if out_parts else torch.empty(0, dtype=dtype, device=device)
        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = out
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


ToFlatAction = ToFlatActionProcessorStep
