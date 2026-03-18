from dataclasses import dataclass, field


@dataclass
class MockComplexObservationRobot:
    """Robot stub emitting a richer observation dictionary for env-pipeline tests."""

    name: str = "mock_complex_robot"
    joint_names: list[str] = field(default_factory=lambda: ["joint_1", "joint_2", "joint_3"])

    def get_observation(self, prefix: str = "arm") -> dict[str, float]:
        joints = {"joint_1": 0.35, "joint_2": -0.25, "joint_3": 0.55}
        ee_x = 0.5 * joints["joint_1"] + 0.2 * joints["joint_2"] - 0.1 * joints["joint_3"]
        ee_y = -0.3 * joints["joint_1"] + 0.4 * joints["joint_2"] + 0.2 * joints["joint_3"]
        ee_z = joints["joint_1"] + joints["joint_2"] + joints["joint_3"]
        ee_wx = 0.1 * joints["joint_1"]
        ee_wy = -0.05 * joints["joint_2"]
        ee_wz = 0.2 * joints["joint_3"]

        return {
            f"{prefix}.joint_1.pos": joints["joint_1"],
            f"{prefix}.joint_2.pos": joints["joint_2"],
            f"{prefix}.joint_3.pos": joints["joint_3"],
            f"{prefix}.joint_1.vel": 0.03,
            f"{prefix}.joint_2.vel": -0.01,
            f"{prefix}.joint_3.vel": 0.02,
            f"{prefix}.joint_1.current": 0.4,
            f"{prefix}.joint_2.current": 0.2,
            f"{prefix}.joint_3.current": 0.1,
            f"{prefix}.x.ee_pos": ee_x,
            f"{prefix}.y.ee_pos": ee_y,
            f"{prefix}.z.ee_pos": ee_z,
            f"{prefix}.wx.ee_pos": ee_wx,
            f"{prefix}.wy.ee_pos": ee_wy,
            f"{prefix}.wz.ee_pos": ee_wz,
        }


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
class MockVelocityDeltaTeleoperator:
    """Delta teleoperator stub exposing Cartesian velocity keys directly."""

    action_features: dict[str, type] = field(
        default_factory=lambda: {
            "x.vel": float,
            "y.vel": float,
            "z.vel": float,
            "wx.vel": float,
            "wy.vel": float,
            "wz.vel": float,
        }
    )


@dataclass
class MockKeyboardStyleDeltaTeleoperator:
    """Delta teleoperator stub exposing metadata-style action names."""

    action_features: dict = field(
        default_factory=lambda: {
            "dtype": "float32",
            "shape": (4,),
            "names": {"x.vel": 0, "y.vel": 1, "z.vel": 2, "gripper": 3},
        }
    )


@dataclass
class MockGamepadStyleDeltaTeleoperator:
    """Delta teleoperator stub exposing legacy metadata-style delta names."""

    action_features: dict = field(
        default_factory=lambda: {
            "dtype": "float32",
            "shape": (4,),
            "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
        }
    )


@dataclass
class MockPhoneLikeTeleoperator:
    """Special-schema teleoperator stub that should not be treated as delta-like."""

    action_features: dict[str, type] = field(
        default_factory=lambda: {
            "phone.pos": object,
            "phone.rot": object,
            "phone.raw_inputs": dict,
            "phone.enabled": bool,
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


@dataclass
class MockComplexKinematicsSolver(MockKinematicsSolver):
    """Kinematics mock with a richer affine mapping for FK/IK tests."""

    def forward_kinematics(self, joint_positions: dict[str, float]) -> list[float]:
        q1 = joint_positions[self.joint_names[0]]
        q2 = joint_positions[self.joint_names[1]]
        q3 = joint_positions[self.joint_names[2]]
        return [
            0.5 * q1 + 0.2 * q2 - 0.1 * q3,
            -0.3 * q1 + 0.4 * q2 + 0.2 * q3,
            q1 + q2 + q3,
            0.1 * q1,
            -0.05 * q2,
            0.2 * q3,
        ]

    def inverse_kinematics(self, pose: list[float]) -> dict[str, float]:
        x, y, z, _, _, _ = pose
        q2 = (5.0 * x + y + 0.1 * z) / 2.7
        q1 = 2.0 * x - 0.4 * q2 + 0.2 * z
        q3 = z - q1 - q2
        return {
            self.joint_names[0]: q1,
            self.joint_names[1]: q2,
            self.joint_names[2]: q3,
        }
