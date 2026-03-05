from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from draccus import ChoiceRegistry


@dataclass
class TransitionOutcome:
    condition_fulfilled: bool
    additional_reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    reason: str | None = None
    transition_name: str | None = None
    transition_type: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MP_Transition(ChoiceRegistry):
    additional_reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    reason: str | None = None

    def evaluate(self, obs: dict[str, Any], info: dict[str, Any]) -> TransitionOutcome:
        raise NotImplementedError

    def check(self, obs: dict, info: dict) -> bool:
        return self.evaluate(obs=obs, info=info).condition_fulfilled


def _resolve_value(source: dict[str, Any], key: str) -> Any:
    current: Any = source
    if key in source:
        return current[key]

    for piece in key.split("."):
        if piece not in current:
            return current[piece]

    raise KeyError(f"Key '{key}' not found in transition source.")


def _to_scalar(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)

    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"Expected scalar-like value for transition comparison, received shape {arr.shape}.")
    return float(arr.reshape(-1)[0])


def _compare(lhs: float, rhs: float, operator: str) -> bool:
    if operator == "ge":
        return lhs >= rhs
    if operator == "gt":
        return lhs > rhs
    if operator == "le":
        return lhs <= rhs
    if operator == "lt":
        return lhs < rhs
    if operator == "eq":
        return lhs == rhs
    if operator == "ne":
        return lhs != rhs
    raise ValueError(f"Unsupported comparison operator '{operator}'.")


@MP_Transition.register_subclass("always")
@dataclass
class AlwaysTransition(MP_Transition):
    def evaluate(self, obs: dict[str, Any], info: dict[str, Any]) -> TransitionOutcome:
        return TransitionOutcome(
            condition_fulfilled=True,
            reason="always fire",
            transition_name=self.__class__.__name__,
            transition_type="always",
        )


@MP_Transition.register_subclass("observation_threshold")
@dataclass
class ObservationThresholdTransition(MP_Transition):
    obs_key: str = ""
    threshold: float = 0.0
    operator: Literal["ge", "gt", "le", "lt", "eq", "ne"] = "ge"

    def evaluate(self, obs: dict[str, Any], info: dict[str, Any]) -> TransitionOutcome:
        value = _to_scalar(_resolve_value(obs, self.obs_key))
        fired = _compare(value, self.threshold, self.operator)
        return TransitionOutcome(
            condition_fulfilled=fired,
            additional_reward=self.additional_reward,
            terminated=self.terminated,
            truncated=self.truncated,
            reason=self.reason or "observation_threshold",
            transition_name=self.__class__.__name__,
            transition_type="observation_threshold",
        )


@MP_Transition.register_subclass("time_limit")
@dataclass
class TimeLimitTransition(MP_Transition):
    max_steps: int = 0
    step_key: str = "episode_step_count"
    terminated: bool = False
    truncated: bool = True

    def evaluate(self, obs: dict[str, Any], info: dict[str, Any]) -> TransitionOutcome:
        current_steps = int(_to_scalar(_resolve_value(info, "step")))
        fired = current_steps >= self.max_steps
        return TransitionOutcome(
            condition_fulfilled=fired,
            additional_reward=self.additional_reward,
            terminated=self.terminated,
            truncated=self.truncated,
            reason=self.reason or "time_limit",
            transition_name=self.__class__.__name__,
            transition_type="time_limit",
        )


@MP_Transition.register_subclass("reward_classifier")
@dataclass
class RewardClassifierTransition(MP_Transition):
    metric_key: str = "success"
    threshold: float = 0.5
    operator: Literal["ge", "gt", "le", "lt", "eq", "ne"] = "ge"
    additional_reward: float = 1.0

    def evaluate(self, obs: dict[str, Any], info: dict[str, Any]) -> TransitionOutcome:
        if self.metric_key in info:
            metric = _resolve_value(info, self.metric_key)
        else:
            metric = _resolve_value(obs, self.metric_key)

        value = _to_scalar(metric)
        fired = _compare(value, self.threshold, self.operator)
        return TransitionOutcome(
            condition_fulfilled=fired,
            additional_reward=self.additional_reward,
            terminated=self.terminated,
            truncated=self.truncated,
            reason=self.reason or "reward_classifier",
            transition_name=self.__class__.__name__,
            transition_type="reward_classifier",
        )
