from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from draccus import ChoiceRegistry

from lerobot.teleoperators import TeleopEvents


@dataclass
class TransitionOutcome:
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    reason: str | None = None


@dataclass
class Transition(ChoiceRegistry):
    source: str
    target: str

    def evaluate(self, obs: dict[str, Any], info: dict[str, Any]) -> TransitionOutcome:
        raise NotImplementedError

    def check(self, obs: dict, info: dict) -> bool:
        result = self.evaluate(obs=obs, info=info)
        return result.terminated or result.truncated


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


@Transition.register_subclass("always")
@dataclass
class Always(Transition):
    def evaluate(self, obs: dict[str, Any], info: dict[str, Any]) -> TransitionOutcome:
        return TransitionOutcome(
            terminated=True,
            reason="always"
        )


@Transition.register_subclass("on_success")
@dataclass
class OnSuccess(Transition):
    success_key: str = TeleopEvents.SUCCESS

    def evaluate(self, obs: dict[str, Any], info: dict[str, Any]) -> TransitionOutcome:
        return TransitionOutcome(
            terminated=info.get(self.success_key, False),
            reason="success"
        )


@Transition.register_subclass("on_observation_threshold")
@dataclass
class OnObservationThreshold(Transition):
    obs_key: str = ""
    threshold: float = 0.0
    operator: Literal["ge", "gt", "le", "lt", "eq", "ne"] = "ge"

    def evaluate(self, obs: dict[str, Any], info: dict[str, Any]) -> TransitionOutcome:
        value = _to_scalar(_resolve_value(obs, self.obs_key))
        fired = _compare(value, self.threshold, self.operator)
        return TransitionOutcome(
            terminated=fired,
            reason="observation_threshold"
        )


@Transition.register_subclass("on_time_limit")
@dataclass
class OnTimeLimit(Transition):
    max_steps: int = 0
    step_key: str = "step"

    def evaluate(self, obs: dict[str, Any], info: dict[str, Any]) -> TransitionOutcome:
        current_steps = int(_to_scalar(_resolve_value(info, self.step_key)))
        fired = current_steps >= self.max_steps
        return TransitionOutcome(
            terminated=False,
            truncated=fired,
            reason="time_limit",
        )


@Transition.register_subclass("reward_classifier")
@dataclass
class RewardClassifierTransition(Transition):
    metric_key: str = "success"
    threshold: float = 0.5
    operator: Literal["ge", "gt", "le", "lt", "eq", "ne"] = "ge"
    additional_reward: float = 1.0

    def evaluate(self, obs: dict[str, Any], info: dict[str, Any]) -> TransitionOutcome:

        # todo: run the classifier here

        if self.metric_key in info:
            metric = _resolve_value(info, self.metric_key)
        else:
            metric = _resolve_value(obs, self.metric_key)

        value = _to_scalar(metric)
        fired = _compare(value, self.threshold, self.operator)
        return TransitionOutcome(
            condition_fulfilled=fired,
            additional_reward=self.additional_reward,
            terminated=True,
            truncated=False,
            reason="reward_classifier",
            transition_name=self.__class__.__name__,
            transition_type="reward_classifier",
        )
