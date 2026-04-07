from dataclasses import dataclass, field

from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    AddTeleopActionAsComplimentaryDataStep,
    AddTeleopEventsAsInfoStep,
    DeviceProcessorStep,
    GripperPenaltyProcessorStep,
)
from lerobot.processor.hil_processor import DiscretizeGripperProcessorStep
from lerobot.processor.tf_processor import (
    SixDofVelocityInterventionActionProcessorStep,
    VanillaTFFProcessorStep,
)
from share_rl.primitives.processor import PrimitiveProcessorBuilder


@dataclass
class _ObsCfg:
    add_ee_velocity_to_observation: bool | dict = False
    add_ee_wrench_to_observation: bool | dict = False
    ee_pos_mask: list[int] | dict = field(default_factory=lambda: [1] * 6)


@dataclass
class _GripperCfg:
    use_gripper: bool | dict = False
    min_pos: float | dict = 0.0
    max_pos: float | dict = 1.0
    penalty: float | dict | None = None


@dataclass
class _TaskFrameCfg:
    control_mask: list[int] | dict = field(default_factory=lambda: [1] * 6)


@dataclass
class _ResetCfg:
    terminate_on_success: bool | dict = True


@dataclass
class _EventCfg:
    key_mapping: dict = field(default_factory=dict)
    foot_switch_mapping: dict = field(default_factory=dict)


@dataclass
class _HookCfg:
    time_action_processor: bool = False
    time_env_processor: bool = False
    log_every: int = 10


@dataclass
class _Cfg:
    observation: _ObsCfg = field(default_factory=_ObsCfg)
    gripper: _GripperCfg = field(default_factory=_GripperCfg)
    task_frame: _TaskFrameCfg = field(default_factory=_TaskFrameCfg)
    reset: _ResetCfg = field(default_factory=_ResetCfg)
    events: _EventCfg = field(default_factory=_EventCfg)
    hooks: _HookCfg = field(default_factory=_HookCfg)
    image_preprocessing: None = None
    reward_classifier: None = None
    control_time_s: float | None = None


def test_gripper_indices_follow_masks_and_grippers() -> None:
    processor = _Cfg()
    processor.task_frame.control_mask = {"a": [1, 0, 1, 0, 1, 0], "b": [1, 1, 1, 1, 1, 1]}
    processor.gripper.use_gripper = {"a": True, "b": False}

    builder = PrimitiveProcessorBuilder(robot_names=["a", "b"], processor=processor)

    assert builder.gripper_idc == {"a": 3, "b": None}


def test_action_pipeline_matches_tf_env_shape() -> None:
    builder = PrimitiveProcessorBuilder(robot_names=["arm"], processor=_Cfg())

    pipeline = builder.make_action_processor(teleoperators={})
    step_types = [type(step) for step in pipeline.steps]

    assert AddTeleopEventsAsInfoStep in step_types
    assert AddTeleopActionAsComplimentaryDataStep in step_types
    assert SixDofVelocityInterventionActionProcessorStep in step_types
    assert DiscretizeGripperProcessorStep in step_types


def test_env_pipeline_matches_tf_env_shape() -> None:
    builder = PrimitiveProcessorBuilder(robot_names=["arm"], processor=_Cfg(), fps=30)

    pipeline = builder.make_env_processor(device="cpu")
    step_types = [type(step) for step in pipeline.steps]

    assert isinstance(pipeline.steps[0], VanillaTFFProcessorStep)
    assert GripperPenaltyProcessorStep in step_types
    assert AddBatchDimensionProcessorStep in step_types
    assert isinstance(pipeline.steps[-1], DeviceProcessorStep)
