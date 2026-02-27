from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.processor.hil_processor import TELEOP_ACTION_KEY
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from share.envs.manipulation_primitive.task_frame import ControlMode, ControlSpace, PolicyMode, TaskFrame
from share.envs.utils import check_delta_teleoperator


def _euler_xyz_to_matrix(rx: float, ry: float, rz: float) -> list[list[float]]:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    # Extrinsic XYZ == Rz @ Ry @ Rx.
    return [
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ]


@dataclass
@ProcessorStepRegistry.register("match_teleop_to_policy_action")
class MatchTeleopToPolicyActionProcessorStep(ProcessorStep):
    teleoperators: dict[str, Any] = field(default_factory=dict)
    task_frame: dict[str, TaskFrame] = field(default_factory=dict)
    kinematics: dict[str, Any] = field(default_factory=dict)
    use_virtual_reference: bool | dict[str, bool] = True

    _is_delta_teleoperator: dict[str, bool] = field(default_factory=dict, init=False)
    _virtual_task_pose: dict[str, list[float]] = field(default_factory=dict, init=False)
    _prev_fk_pose: dict[str, list[float]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._is_delta_teleoperator = check_delta_teleoperator(self.teleoperators)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
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
                converted_actions[name] = self._map_delta_teleop(name, frame, teleop_action)
            else:
                converted_actions[name] = self._map_absolute_joint_teleop(name, frame, teleop_action)

        complementary_data[TELEOP_ACTION_KEY] = converted_actions
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        return new_transition

    def _map_delta_teleop(self, name: str, frame: TaskFrame, teleop_action: Any) -> torch.Tensor:
        deltas = self._extract_delta_action(teleop_action)

        if frame.space == ControlSpace.JOINT:
            solver = self._require_solver(name)
            base_pose = self._integration_base_pose(name, frame)
            pose_target = [base_pose[i] + deltas[i] for i in range(6)]
            joint_target = solver.inverse_kinematics(pose_target)
            values = [joint_target.get(f"joint_{axis + 1}", 0.0) for axis in frame.learnable_axis_indices]
            return torch.tensor(values, dtype=torch.float32)

        source_pose = deltas
        if any(
            frame.control_mode[axis] == ControlMode.POS and frame.policy_mode[axis] == PolicyMode.ABSOLUTE
            for axis in frame.learnable_axis_indices
        ):
            base_pose = self._integration_base_pose(name, frame)
            source_pose = [base_pose[i] + deltas[i] for i in range(6)]
            self._virtual_task_pose[name] = source_pose

        return self._encode_learning_space(frame, source_pose)

    def _map_absolute_joint_teleop(self, name: str, frame: TaskFrame, teleop_action: Any) -> torch.Tensor:
        joint_state = self._extract_joint_action(teleop_action)
        if frame.space == ControlSpace.JOINT:
            values = [joint_state.get(f"joint_{axis + 1}", 0.0) for axis in frame.learnable_axis_indices]
            return torch.tensor(values, dtype=torch.float32)

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
                matrix = _euler_xyz_to_matrix(*rot)
                values.extend([matrix[0][0], matrix[1][0], matrix[2][0]])
            else:
                matrix = _euler_xyz_to_matrix(*rot)
                values.extend(
                    [
                        matrix[0][0],
                        matrix[1][0],
                        matrix[2][0],
                        matrix[0][1],
                        matrix[1][1],
                        matrix[2][1],
                    ]
                )

        return torch.tensor(values, dtype=torch.float32)

    def _integration_base_pose(self, name: str, frame: TaskFrame) -> list[float]:
        use_virtual = self.use_virtual_reference[name] if isinstance(self.use_virtual_reference, dict) else self.use_virtual_reference
        if use_virtual and name in self._virtual_task_pose:
            return self._virtual_task_pose[name]
        return list(frame.target)

    def _require_solver(self, name: str) -> Any:
        solver = self.kinematics.get(name)
        if solver is None:
            raise ValueError(f"Missing kinematics solver for '{name}'")
        return solver

    @staticmethod
    def _extract_delta_action(teleop_action: Any) -> list[float]:
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
        if isinstance(teleop_action, dict):
            joint_state: dict[str, float] = {}
            for k, v in teleop_action.items():
                name = k.replace(".pos", "")
                joint_state[name] = float(v)
            return joint_state
        return {f"joint_{i + 1}": float(v) for i, v in enumerate(teleop_action)}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
