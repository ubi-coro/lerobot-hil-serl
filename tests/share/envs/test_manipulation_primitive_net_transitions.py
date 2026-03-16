import numpy as np

from share.envs.manipulation_primitive_net.transitions import (
    ObservationThresholdTransition,
    RewardClassifierTransition,
    TimeLimitTransition,
)


def test_transition_threshold_triggers_and_returns_next_primitive():
    transition = ObservationThresholdTransition(
        obs_key="metrics.height",
        threshold=0.4,
        operator="ge",
        next_primitive="place",
        additional_reward=0.25,
        reason="height_reached",
    )

    outcome = transition.evaluate(obs={"metrics": {"height": np.array([0.6])}}, info={})

    assert outcome.condition_fulfilled
    assert outcome.next_primitive == "place"
    assert outcome.additional_reward == 0.25
    assert outcome.reason == "height_reached"
    assert outcome.transition_type == "observation_threshold"


def test_transition_time_limit_sets_truncation_style_flags():
    transition = TimeLimitTransition(max_steps=3)

    outcome = transition.evaluate(obs={}, info={"episode_step_count": 3})

    assert outcome.condition_fulfilled
    assert not outcome.terminated
    assert outcome.truncated
    assert outcome.transition_type == "time_limit"


def test_transition_classifier_adds_sparse_success_reward():
    transition = RewardClassifierTransition(
        metric_key="success",
        threshold=0.5,
        operator="ge",
        additional_reward=2.0,
        next_primitive="done",
        terminated=True,
    )

    outcome = transition.evaluate(obs={}, info={"success": 1.0})

    assert outcome.condition_fulfilled
    assert outcome.additional_reward == 2.0
    assert outcome.next_primitive == "done"
    assert outcome.terminated
    assert outcome.transition_type == "reward_classifier"
