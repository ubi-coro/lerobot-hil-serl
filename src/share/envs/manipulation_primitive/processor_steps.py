from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import einops
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.processor.hil_processor import TELEOP_ACTION_KEY
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from lerobot.teleoperators import TeleopEvents
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
from share.envs.manipulation_primitive.task_frame import ControlMode, ControlSpace, PolicyMode, TaskFrame
from share.envs.utils import check_delta_teleoperator


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


@dataclass
@ProcessorStepRegistry.register("match_teleop_to_policy_action")
class MatchTeleopToPolicyActionProcessorStep(ProcessorStep):
    """Map raw teleop commands into the policy learning-space action format."""

    teleoperators: dict[str, Any] = field(default_factory=dict)
    task_frame: dict[str, TaskFrame] = field(default_factory=dict)
    kinematics: dict[str, Any] = field(default_factory=dict)
    joint_names: dict[str, list[str]] = field(default_factory=dict)
    use_virtual_reference: bool | dict[str, bool] = True

    _is_delta_teleoperator: dict[str, bool] = field(default_factory=dict, init=False)
    _virtual_task_pose: dict[str, list[float]] = field(default_factory=dict, init=False)
    _prev_fk_pose: dict[str, list[float]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        """Cache teleoperator modality flags for fast dispatch."""
        self._is_delta_teleoperator = check_delta_teleoperator(self.teleoperators)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Rewrite complementary teleop actions using task-frame encodings."""
        new_transition = transition.copy()
        complementary_data = dict(new_transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        teleop_action_dict = complementary_data.get(TELEOP_ACTION_KEY)
        if not isinstance(teleop_action_dict, dict):
            return new_transition

        converted_actions: dict[str, torch.Tensor] = {}
        for name, teleop_action in teleop_action_dict.items():
            frame = self.task_frame.get(name)
            if frame is None or not frame.is_adaptive:
                continue

            if self._is_delta_teleoperator.get(name, False):
                # FIX: Pass the transition down to extract physical state
                converted_actions[name] = self._map_delta_teleop(name, frame, teleop_action, transition)
            else:
                converted_actions[name] = self._map_absolute_joint_teleop(name, frame, teleop_action)

        complementary_data[TELEOP_ACTION_KEY] = converted_actions
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        return new_transition

    def _map_delta_teleop(self, name: str, frame: TaskFrame, teleop_action: Any, transition: EnvTransition) -> torch.Tensor:
        """Map Cartesian delta teleop input into learning-space values."""
        deltas = self._extract_delta_action(teleop_action)

        if frame.space == ControlSpace.JOINT:
            solver = self._require_solver(name)
            # FIX: Pass transition
            base_pose = self._integration_base_pose(name, frame, transition)
            pose_target = [base_pose[i] + deltas[i] for i in range(6)]
            joint_target = solver.inverse_kinematics(pose_target)
            values = [joint_target.get(f"joint_{axis + 1}", 0.0) for axis in frame.learnable_axis_indices]
            return torch.tensor(values, dtype=torch.float32)

        source_pose = deltas
        if any(
            frame.control_mode[axis] == ControlMode.POS and frame.policy_mode[axis] == PolicyMode.ABSOLUTE
            for axis in frame.learnable_axis_indices
        ):
            # FIX: Pass transition
            base_pose = self._integration_base_pose(name, frame, transition)
            source_pose = [base_pose[i] + deltas[i] for i in range(6)]
            self._virtual_task_pose[name] = source_pose

        return self._encode_learning_space(frame, source_pose)

    def _map_absolute_joint_teleop(self, name: str, frame: TaskFrame, teleop_action: Any) -> torch.Tensor:
        """Map absolute joint teleop input into learning-space values."""
        joint_state = self._extract_joint_action(teleop_action)
        if frame.space == ControlSpace.JOINT:
            #values = [joint_state.get(f"joint_{axis + 1}", joint_state.get(self.joint_names[name][axis])) for axis in frame.learnable_axis_indices]
            return torch.tensor(list(joint_state.values()), dtype=torch.float32)

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

    def _encode_learning_space(self, frame: TaskFrame, source_pose: list[float]) -> torch.Tensor:
        """Encode a 6-DoF source pose into manifold-aware policy action vectors."""
        values: list[float] = []
        absolute_rot_axes = [
            axis
            for axis in frame.learnable_axis_indices
            if axis >= 3 and frame.control_mode[axis] == ControlMode.POS and frame.policy_mode[axis] == PolicyMode.ABSOLUTE
        ]

        for axis in frame.learnable_axis_indices:
            control_mode = frame.control_mode[axis]
            policy_mode = frame.policy_mode[axis]

            if axis in absolute_rot_axes:
                continue

            if control_mode in {ControlMode.VEL, ControlMode.FORCE}:
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

        return torch.tensor(values, dtype=torch.float32)

    def _integration_base_pose(self, name: str, frame: TaskFrame, transition: EnvTransition) -> list[float]:
        """Get pose baseline for integrating relative teleop commands."""
        use_virtual = self.use_virtual_reference[name] if isinstance(self.use_virtual_reference, dict) else self.use_virtual_reference

        # 1. Primary: Use the virtual reference if enabled and populated
        if use_virtual and name in self._virtual_task_pose:
            return self._virtual_task_pose[name]

        # 2. Secondary: Fall back to the actual physical observation
        observation = transition.get(TransitionKey.OBSERVATION)
        if isinstance(observation, dict):
            axis_names = ["x", "y", "z", "wx", "wy", "wz"]
            obs_pose = []
            for axis_name in axis_names:
                key = f"{name}.{axis_name}.ee_pos"
                if key not in observation:
                    obs_pose = []
                    break
                value = observation[key]
                obs_pose.append(float(value.item()) if isinstance(value, torch.Tensor) else float(value))

            if len(obs_pose) == 6:
                return obs_pose

        # 3. Ultimate Fallback: The static target configuration
        return list(frame.target)

    def _require_solver(self, name: str) -> Any:
        """Return configured kinematics solver for ``name`` or raise."""
        solver = self.kinematics.get(name)
        if solver is None:
            raise ValueError(f"Missing kinematics solver for '{name}'")
        return solver

    @staticmethod
    def _extract_delta_action(teleop_action: Any) -> list[float]:
        """Normalize teleop delta input into a 6-value Cartesian delta list."""
        if isinstance(teleop_action, dict):
            return [
                float(teleop_action.get("delta_x", 0.0)),
                float(teleop_action.get("delta_y", 0.0)),
                float(teleop_action.get("delta_z", 0.0)),
                float(teleop_action.get("delta_rx", 0.0)),
                float(teleop_action.get("delta_ry", 0.0)),
                float(teleop_action.get("delta_rz", 0.0)),
            ]
        return [float(v) for v in teleop_action][:6]

    @staticmethod
    def _extract_joint_action(teleop_action: Any) -> dict[str, float]:
        """Normalize teleop joint input into ``joint_name -> position``."""
        if isinstance(teleop_action, dict):
            return teleop_action
        else:
            return {f"joint_{i + 1}": float(v) for i, v in enumerate(teleop_action)}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Leave feature specs unchanged."""
        return features


@dataclass
@ProcessorStepRegistry.register("task_frame_intervention_action_processor")
class InterventionActionProcessorStep(ProcessorStep):
    """Project learning-space actions into full task-frame targets with intervention override."""

    teleoperators: dict[str, Any] = field(default_factory=dict)
    task_frame: dict[str, TaskFrame] = field(default_factory=dict)

    def __post_init__(self):
        self._disable_torque_on_intervention = {name: hasattr(teleop, "bus") for name, teleop in self.teleoperators.items()}
        self._intervention_occurred = False

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Select policy or teleop action source and emit full task-frame targets."""
        # 1) read out transition info
        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, torch.Tensor):
            raise TypeError(f"Action should be a torch.Tensor, got {type(action)}")

        new_transition = transition.copy()
        info = dict(new_transition.get(TransitionKey.INFO) or {})
        complementary_data = dict(new_transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        teleop_action_dict = complementary_data.get(TELEOP_ACTION_KEY)
        is_intervention = bool(info.get(TeleopEvents.IS_INTERVENTION, False))

        # 2) process teleop action, decide leader torque, send feedback, prepare actions
        self._intervention_occurred = self._intervention_occurred | is_intervention
        if self._intervention_occurred and not is_intervention:
            info[TeleopEvents.INTERVENTION_COMPLETED] = True

        if is_intervention and isinstance(teleop_action_dict, dict):
            source_actions = teleop_action_dict

            # torque leaders off during teleop
            for name, teleop_action in teleop_action_dict.items():
                if self._disable_torque_on_intervention[name]:
                    self.teleoperators[name].disable_torque()

        else:
            source_actions = self._split_policy_action(action)

            if self._intervention_occurred:
                # torque leader on intervention end
                # todo: this takes forever (2-3ms -> 25-30ms) when recording normally, ie not interactive
                for name, teleop_action in self.teleoperators.items():
                    if self._disable_torque_on_intervention[name]:
                        self.teleoperators[name].enable_torque()
            else:
                # send feedback to the leaders
                # disabled during the first cycle where the intervention ended -> requires reset
                for teleop_name, teleop in self.teleoperators.items():
                        teleop.send_feedback(self._map_to_teleop_action(source_actions, teleop_name))

        full_action: dict[str, torch.Tensor] = {}
        for name, frame in self.task_frame.items():
            encoded_action = source_actions.get(name)

            # task frame is the backup
            if encoded_action is None:
                full_action[name] = torch.tensor(frame.target, dtype=action.dtype, device=action.device)
                continue

            # project partial action on all task frame targets
            projected = self._project_policy_action(frame, encoded_action)
            full_action[name] = torch.tensor(projected, dtype=action.dtype, device=action.device)

        # build teleop action that looks exactly like the action that came in
        teleop_action_store = torch.tensor([])
        if teleop_action_dict:
            teleop_action_store = torch.concatenate([a for a in teleop_action_dict.values()])
        complementary_data[TELEOP_ACTION_KEY] = teleop_action_store

        new_transition[TransitionKey.ACTION] = full_action
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        new_transition[TransitionKey.INFO] = info
        return new_transition

    def _split_policy_action(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Split flat policy action tensor into per-robot slices."""
        policy_by_robot: dict[str, torch.Tensor] = {}
        idx = 0
        for name, frame in self.task_frame.items():
            dim = frame.policy_action_dim
            policy_by_robot[name] = action[idx : idx + dim]
            idx += dim
        return policy_by_robot

    def _project_policy_action(self, frame: TaskFrame, encoded_action: Any) -> list[float]:
        """Project encoded learning-space vectors into a full 6-DoF task target."""
        raw = torch.as_tensor(encoded_action, dtype=torch.float32).flatten().tolist()

        full_target = list(frame.target)
        cursor = 0
        absolute_rot_axes = [axis for axis in frame.learnable_axis_indices if frame.is_absolute_rotation_axis(axis)]

        for axis in frame.learnable_axis_indices:
            if axis in absolute_rot_axes:
                continue

            if cursor >= len(raw):
                raise ValueError("Encoded action is shorter than expected for task-frame projection")

            value = raw[cursor]
            cursor += 1
            if frame.control_mode[axis] in {ControlMode.VEL, ControlMode.FORCE}:
                value = self._bound_differential_axis(frame, axis, value)

            full_target[axis] = float(value)

        if absolute_rot_axes:
            rotation_values, consumed = self._decode_absolute_rotation(absolute_rot_axes, raw[cursor:])
            cursor += consumed
            for axis in absolute_rot_axes:
                full_target[axis] = float(rotation_values[axis - 3])

        if cursor != len(raw):
            raise ValueError("Encoded action has trailing values that do not match task-frame manifold layout")

        return full_target

    def _map_to_teleop_action(self, policy_action: dict[str, torch.Tensor], name: str) -> dict[str, float]:
        teleop_action = {}
        policy_action_idx = 0

        for teleop_action_idx, ft in enumerate(self.teleoperators[name].action_features):
            if teleop_action_idx in self.task_frame[name].learnable_axis_indices:
                teleop_action[ft] = float(policy_action[name][policy_action_idx])
                policy_action_idx += 1

        return teleop_action

    @staticmethod
    def _bound_differential_axis(frame: TaskFrame, axis: int, value: float) -> float:
        """Bound velocity/force-like scalars with tanh and optional axis scaling."""
        if frame.min_target is not None and frame.max_target is not None:
            scale = max(abs(frame.min_target[axis]), abs(frame.max_target[axis]))
            if scale > 0:
                return math.tanh(value) * scale
        return math.tanh(value)

    def _decode_absolute_rotation(self, absolute_rot_axes: list[int], raw: list[float]) -> tuple[list[float], int]:
        """Decode manifold rotation chunks (S1/S2/SO3) into Euler XYZ angles."""
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
            if norm < 1e-8:
                direction = np.array([1.0, 0.0, 0.0], dtype=float)
            else:
                direction = direction / norm

            # Complete the first-axis direction into a full orthonormal frame, then
            # decode through scipy Rotation so all matrix->Euler handling is consistent.
            reference = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(float(np.dot(reference, direction))) > 0.95:
                reference = np.array([0.0, 1.0, 0.0], dtype=float)

            col1 = direction
            col2 = np.cross(reference, col1)
            col2_norm = np.linalg.norm(col2)
            if col2_norm < 1e-8:
                col2 = np.array([0.0, 1.0, 0.0], dtype=float)
            else:
                col2 = col2 / col2_norm
            col3 = np.cross(col1, col2)
            matrix = np.column_stack([col1, col2, col3])
            rx, ry, rz = Rotation.from_matrix(matrix).as_euler("xyz", degrees=False)
            rot = [float(rx), float(ry), float(rz)]
            return rot, 3

        if len(absolute_rot_axes) == 3:
            if len(raw) < 6:
                raise ValueError("SO(3) 6D representation requires 6 values")
            matrix = self._rotation_6d_to_matrix(raw[:6])
            euler = Rotation.from_matrix(matrix).as_euler("xyz", degrees=False)
            return euler.tolist(), 6

        raise ValueError(f"Expected 1..3 absolute rotation axes, got {len(absolute_rot_axes)}")

    @staticmethod
    def _rotation_6d_to_matrix(raw: list[float]) -> list[list[float]]:
        """Convert 6D continuous rotation representation into a numerically stable matrix."""
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
        matrix = np.column_stack([b1, b2, b3])
        return matrix.tolist()

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Leave feature specs unchanged."""
        return features

    def reset(self) -> None:
        """Clear intervention completion tracking state."""
        self._intervention_occurred = False


@dataclass
@ProcessorStepRegistry.register("to_joint_action_processor")
class ToJointActionProcessorStep(ProcessorStep):
    """Convert task-frame action dictionaries into joint command dictionaries when needed."""

    is_task_frame_robot: dict[str, bool] = field(default_factory=dict)
    task_frame: dict[str, TaskFrame] = field(default_factory=dict)
    kinematics: dict[str, Any] = field(default_factory=dict)
    joint_names: dict[str, list[str]] = field(default_factory=dict)
    use_virtual_reference: bool | dict[str, bool] = True

    _virtual_task_pose: dict[str, list[float]] = field(default_factory=dict, init=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Convert task-space actions to joint commands for joint-only robots."""
        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, dict):
            return transition

        new_transition = transition.copy()
        joint_action: dict[str, float] = {}

        for name, robot_action in action.items():
            frame = self.task_frame.get(name)
            if frame is None:
                continue

            if self.is_task_frame_robot.get(name, False):
                raise ValueError(
                    f"ToJointActionProcessorStep received task-frame robot '{name}', "
                    "but this step only supports joint-only robots."
                )

            task_target = torch.as_tensor(robot_action, dtype=torch.float32).flatten().tolist()
            if len(task_target) != len(frame.target):
                raise ValueError(
                    f"Task-frame action for '{name}' has width {len(task_target)}, "
                    f"expected {len(frame.target)}"
                )

            absolute_target = self._integrate_relative_axes(name, frame, task_target, transition)
            bounded_target = self._clamp_target(frame, absolute_target)

            solver = self.kinematics.get(name)
            if solver is None:
                raise ValueError(f"Missing kinematics solver for joint-only robot '{name}'")

            try:
                ik_solution = solver.inverse_kinematics(bounded_target)
            except Exception as exc:  # pragma: no cover - exercised with mock solver in unit tests
                raise ValueError(f"IK failed for '{name}': {exc}") from exc

            for joint_name in self.joint_names.get(name, []):
                if joint_name not in ik_solution:
                    raise ValueError(f"IK solution for '{name}' missing joint '{joint_name}'")
                joint_action[f"{joint_name}.pos"] = float(ik_solution[joint_name])

            self._virtual_task_pose[name] = bounded_target

        new_transition[TransitionKey.ACTION] = joint_action
        return new_transition

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
            axis_names = ["x", "y", "z", "wx", "wy", "wz"]
            obs_pose = []
            for axis_name in axis_names:
                key = f"{name}.{axis_name}.ee_pos"
                if key not in observation:
                    obs_pose = []
                    break
                value = observation[key]
                obs_pose.append(float(value.item()) if isinstance(value, torch.Tensor) else float(value))
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
    gripper_enable: bool | dict[str, bool] = False
    add_joint_position_to_observation: bool | dict[str, bool] = True
    add_joint_velocity_to_observation: bool | dict[str, bool] = False
    add_current_to_observation: bool | dict[str, bool] = False
    add_ee_pos_to_observation: bool | dict[str, bool] = False
    add_ee_velocity_to_observation: bool | dict[str, bool] = False
    add_ee_wrench_to_observation: bool | dict[str, bool] = False

    _prev_obs: dict[str, dict[str, float]] = field(default_factory=dict, init=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)
        if not isinstance(observation, dict):
            return transition

        new_transition = transition.copy()
        new_observation = dict(observation)

        state_values = self._collect_state_values(observation)
        if state_values:
            new_observation[OBS_STATE] = torch.tensor(state_values, dtype=torch.float32)

        for key, value in observation.items():
            if "image" not in key:
                continue
            new_observation[key] = self._process_image(value)

        new_transition[TransitionKey.OBSERVATION] = new_observation
        return new_transition

    def _collect_state_values(self, observation: dict[str, Any]) -> list[float]:
        values: list[float] = []
        robot_names = self._robot_names(observation)
        axis_names = ["x", "y", "z", "wx", "wy", "wz"]

        for name in sorted(robot_names):
            if self._enabled(self.add_joint_position_to_observation, name):
                values.extend(self._collect_joint_channel(observation, name, "pos"))

            if self._enabled(self.add_joint_velocity_to_observation, name):
                joint_vel = self._collect_joint_channel(observation, name, "vel")
                if not joint_vel:
                    pos_keys = self._joint_keys(observation, name, "pos")
                    joint_vel = self._differentiate(name, observation, pos_keys)
                values.extend(joint_vel)

            if self._enabled(self.add_current_to_observation, name):
                values.extend(self._collect_joint_channel(observation, name, "current"))

            if self._enabled(self.add_ee_pos_to_observation, name):
                values.extend(self._collect_ee_channel(observation, name, axis_names, "ee_pos"))

            if self._enabled(self.add_ee_velocity_to_observation, name):
                ee_vel = self._collect_ee_channel(observation, name, axis_names, "ee_vel")
                if not ee_vel:
                    ee_pos_keys = [f"{name}.{axis}.ee_pos" for axis in axis_names]
                    ee_vel = self._differentiate(name, observation, ee_pos_keys)
                values.extend(ee_vel)

            if self._enabled(self.add_ee_wrench_to_observation, name):
                values.extend(self._collect_ee_channel(observation, name, axis_names, "ee_wrench"))

            if self._enabled(self.gripper_enable, name):
                gripper_key = f"{name}.gripper.pos"
                if gripper_key in observation:
                    values.append(self._to_float(observation[gripper_key]))

        self._update_prev_obs(observation)
        return values

    def _process_image(self, image: Any) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            img = image
        else:
            img = torch.from_numpy(np.asarray(image))

        if img.ndim == 3:
            h, w, c = img.shape
            if c < h and c < w:  # to channel first
                img = einops.rearrange(img, "h w c -> c h w")
        elif img.ndim == 4:
            _, h, w, c = img.shape
            if c < h and c < w:  # to channel first
                img = einops.rearrange(img, "b h w c -> b c h w")
        else:
            raise ValueError(f"Expected image tensor with 3 or 4 dimensions, got shape {tuple(img.shape)}")

        if img.dtype != torch.float32:
            img = img.to(torch.float32)
        return img / 255.0 if img.max() > 1.0 else img

    @staticmethod
    def _robot_names(observation: dict[str, Any]) -> set[str]:
        names: set[str] = set()
        for key in observation:
            if key.startswith(OBS_IMAGES):
                continue
            if "." in key:
                names.add(key.split(".", 1)[0])
        return names

    @staticmethod
    def _enabled(flag: bool | dict[str, bool], name: str) -> bool:
        if isinstance(flag, dict):
            return bool(flag.get(name, False))
        return bool(flag)

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

    def _collect_joint_channel(self, observation: dict[str, Any], robot_name: str, suffix: str) -> list[float]:
        return [self._to_float(observation[key]) for key in self._joint_keys(observation, robot_name, suffix)]

    def _collect_ee_channel(
        self,
        observation: dict[str, Any],
        robot_name: str,
        axis_names: list[str],
        suffix: str,
    ) -> list[float]:
        values: list[float] = []
        for axis in axis_names:
            key = f"{robot_name}.{axis}.{suffix}"
            if key in observation:
                values.append(self._to_float(observation[key]))
        return values

    def _differentiate(self, robot_name: str, observation: dict[str, Any], keys: list[str]) -> list[float]:
        if not keys:
            return []
        prev = self._prev_obs.get(robot_name, {})
        return [self._to_float(observation[key]) - prev.get(key, self._to_float(observation[key])) for key in keys if key in observation]

    def _update_prev_obs(self, observation: dict[str, Any]) -> None:
        robot_names = self._robot_names(observation)
        for name in robot_names:
            self._prev_obs[name] = {
                key: self._to_float(value)
                for key, value in observation.items()
                if key.startswith(f"{name}.") and "image" not in key
            }

    def reset(self) -> None:
        self._prev_obs.clear()

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        new_features = {ft: dict(bucket) for ft, bucket in features.items()}
        obs_features = new_features.get(PipelineFeatureType.OBSERVATION, {})

        state_dim = 0
        robot_names = self._robot_names(obs_features)
        for name in sorted(robot_names):
            if self._enabled(self.add_joint_position_to_observation, name):
                state_dim += len(self._joint_keys(obs_features, name, "pos"))
            if self._enabled(self.add_joint_velocity_to_observation, name):
                joint_vel_keys = self._joint_keys(obs_features, name, "vel")
                state_dim += len(joint_vel_keys) if joint_vel_keys else len(self._joint_keys(obs_features, name, "pos"))
            if self._enabled(self.add_current_to_observation, name):
                state_dim += len(self._joint_keys(obs_features, name, "current"))
            if self._enabled(self.add_ee_pos_to_observation, name):
                state_dim += len(self._collect_ee_feature_keys(obs_features, name, "ee_pos"))
            if self._enabled(self.add_ee_velocity_to_observation, name):
                ee_vel_keys = self._collect_ee_feature_keys(obs_features, name, "ee_vel")
                state_dim += len(ee_vel_keys) if ee_vel_keys else len(self._collect_ee_feature_keys(obs_features, name, "ee_pos"))
            if self._enabled(self.add_ee_wrench_to_observation, name):
                state_dim += len(self._collect_ee_feature_keys(obs_features, name, "ee_wrench"))
            if self._enabled(self.gripper_enable, name) and f"{name}.gripper.pos" in obs_features:
                state_dim += 1

        if state_dim > 0:
            obs_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))

        # transform to channel first images
        for name, feature in obs_features.items():
            if feature.type == FeatureType.VISUAL:
                h, w, c = feature.shape
                if c < h and c < w:
                    obs_features[name].shape = (feature.shape[2], feature.shape[0], feature.shape[1])

        return new_features

    @staticmethod
    def _collect_ee_feature_keys(observation: dict[str, Any], robot_name: str, suffix: str) -> list[str]:
        axis_names = ["x", "y", "z", "wx", "wy", "wz"]
        return [f"{robot_name}.{axis}.{suffix}" for axis in axis_names if f"{robot_name}.{axis}.{suffix}" in observation]


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
        axis_names = ["x", "y", "z", "wx", "wy", "wz"]

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

                for axis, axis_name in enumerate(axis_names):
                    new_observation[f"{robot_name}.{axis_name}.ee_pos"] = torch.tensor(axis_values[axis], dtype=torch.float32)
            else:
                joint_state = self._extract_joint_state(observation, robot_name, joints, index=None)
                pose = solver.forward_kinematics(joint_state)
                for axis, axis_name in enumerate(axis_names):
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
            for axis, axis_name in enumerate(axis_names):
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
@ProcessorStepRegistry.register("robot_action_to_policy_action_dict")
class RobotActionToPolicyActionProcessorStep(ProcessorStep):
    """Flatten robot action dict to policy tensor with stable robot->joint ordering."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, dict):
            return transition

        out = torch.concatenate([robot_action for robot_action in action.values()])

        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = out
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
