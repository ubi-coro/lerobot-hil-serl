"""Unit tests for MP-Net transition primitives.

Each test documents one transition behavior in isolation so regressions in the
routing contract are easy to localize.
"""

from __future__ import annotations

import numpy as np

from share.envs.manipulation_primitive_net.transitions import (
    Always,
    OnObservationThreshold,
    OnSuccess,
    OnTimeLimit,
    RewardClassifierTransition,
)


def test_always_transition_terminates_immediately():
    """Transition primitive: unconditional routing/end-of-segment trigger."""
    outcome = Always(source="pick", target="place").evaluate(obs={}, info={})

    assert outcome.terminated is True
    assert outcome.truncated is False
    assert outcome.reward == 0.0
    assert outcome.reason == "always"


def test_on_success_reads_success_event_from_info():
    """Transition primitive: teleop success events should end the current segment."""
    transition = OnSuccess(source="pick", target="place")

    outcome = transition.evaluate(obs={}, info={transition.success_key: True})

    assert outcome.terminated is True
    assert outcome.truncated is False
    assert outcome.reason == "success"


def test_observation_threshold_transition_uses_nested_scalar_observation():
    """Transition primitive: nested observation keys are compared after scalar normalization."""
    transition = OnObservationThreshold(
        source="pick",
        target="place",
        obs_key="metrics.height",
        threshold=0.4,
        operator="ge",
    )

    outcome = transition.evaluate(obs={"metrics": {"height": np.array([0.6])}}, info={})

    assert outcome.terminated is True
    assert outcome.truncated is False
    assert outcome.reward == 0.0
    assert outcome.reason == "observation_threshold"


def test_time_limit_transition_requests_truncation_when_budget_is_reached():
    """Transition primitive: step budgets should truncate rather than terminate the episode."""
    transition = OnTimeLimit(source="pick", target="reset", max_steps=3, step_key="episode_step")

    outcome = transition.evaluate(obs={}, info={"episode_step": 3})

    assert outcome.terminated is False
    assert outcome.truncated is True
    assert outcome.reward == 0.0
    assert outcome.reason == "time_limit"


def test_reward_classifier_transition_adds_sparse_reward_on_threshold_crossing():
    """Transition primitive: classifier-style success metrics should add shaping reward and terminate."""
    transition = RewardClassifierTransition(
        source="pick",
        target="done",
        metric_key="success",
        threshold=0.5,
        operator="ge",
        additional_reward=2.0,
    )

    outcome = transition.evaluate(obs={}, info={"success": 1.0})

    assert outcome.terminated is True
    assert outcome.truncated is False
    assert outcome.reward == 2.0
    assert outcome.reason == "reward_classifier"


def test_reward_classifier_transition_is_noop_below_threshold():
    """Transition primitive: non-firing classifier transitions should not alter reward or done flags."""
    transition = RewardClassifierTransition(
        source="pick",
        target="done",
        metric_key="success",
        threshold=0.5,
        operator="ge",
        additional_reward=2.0,
    )

    outcome = transition.evaluate(obs={}, info={"success": 0.1})

    assert outcome.terminated is False
    assert outcome.truncated is False
    assert outcome.reward == 0.0
    assert outcome.reason is None
