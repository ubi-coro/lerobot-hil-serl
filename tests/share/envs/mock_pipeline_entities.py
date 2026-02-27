from dataclasses import dataclass, field


@dataclass
class MockJointOnlyRobot:
    """Minimal robot stub without task-frame capability."""

    name: str = "mock_joint_robot"
    joint_names: list[str] = field(default_factory=lambda: ["joint_1", "joint_2", "joint_3"])


@dataclass
class MockTaskFrameRobot(MockJointOnlyRobot):
    """Minimal robot stub exposing task-frame capability."""

    last_task_frame_command: list[float] | None = None

    def set_task_frame(self, command: list[float]) -> None:
        self.last_task_frame_command = list(command)


@dataclass
class MockDeltaTeleoperator:
    """Delta teleoperator stub (SpaceMouse/keyboard style)."""

    action_features: dict[str, type] = field(
        default_factory=lambda: {
            "delta_x": float,
            "delta_y": float,
            "delta_z": float,
            "delta_rx": float,
            "delta_ry": float,
            "delta_rz": float,
        }
    )


@dataclass
class MockAbsoluteJointTeleoperator:
    """Absolute-joint teleoperator stub (leader-arm style)."""

    action_features: dict[str, type] = field(
        default_factory=lambda: {
            "joint_1.pos": float,
            "joint_2.pos": float,
            "joint_3.pos": float,
        }
    )


@dataclass
class MockKinematicsSolver:
    """Deterministic FK/IK mock used by processor-pipeline unit tests."""

    joint_names: list[str] = field(default_factory=lambda: ["joint_1", "joint_2", "joint_3"])

    def forward_kinematics(self, joint_positions: dict[str, float]) -> list[float]:
        """Map joints to a deterministic 6D task-frame pose."""
        x = sum(joint_positions[name] for name in self.joint_names)
        y = joint_positions[self.joint_names[0]]
        z = joint_positions[self.joint_names[-1]]
        rx = 0.1 * x
        ry = 0.1 * y
        rz = 0.1 * z
        return [x, y, z, rx, ry, rz]

    def inverse_kinematics(self, pose: list[float]) -> dict[str, float]:
        """Map a task-frame pose back to deterministic joint targets."""
        x, y, z, _, _, _ = pose
        return {
            self.joint_names[0]: y,
            self.joint_names[1]: x - y - z,
            self.joint_names[2]: z,
        }
