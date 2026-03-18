"""Runtime tests for ``ManipulationPrimitiveNet`` step/reset orchestration.

Each test targets one piece of the MP-Net control loop so routing, reward
propagation, and reset behavior stay aligned with the current implementation.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from lerobot.processor import TransitionKey, create_transition
from lerobot.processor.hil_processor import TELEOP_ACTION_KEY
from lerobot.utils.constants import ACTION
from share.envs.manipulation_primitive_net.env_manipulation_primitive_net import (
    ManipulationPrimitiveNet,
)
from share.envs.manipulation_primitive_net.transitions import TransitionOutcome


class DummyEnv:
    """Minimal env stub that records actions and returns a fixed step/reset payload."""

    def __init__(self, *, obs, reward=0.0, terminated=False, truncated=False, info=None):
        self.obs = obs
        self.reward = reward
        self.terminated = terminated
        self.truncated = truncated
        self.info = {} if info is None else dict(info)
        self.last_action = None
        self.reset_calls: list[dict] = []

    def step(self, action):
        self.last_action = action
        return self.obs, self.reward, self.terminated, self.truncated, dict(self.info)

    def reset(self, *, seed=None, options=None):
        self.reset_calls.append({"seed": seed, "options": dict(options or {})})
        return self.obs, {"reset_seed": seed}


class IdentityProcessor:
    """Processor stub that keeps transitions unchanged while tracking resets."""

    def __init__(self):
        self.reset_count = 0

    def __call__(self, transition):
        return transition

    def reset(self):
        self.reset_count += 1


class StaticActionProcessor(IdentityProcessor):
    """Action processor stub that can inject teleop complementary data or info."""

    def __init__(self, *, info=None, complementary_data=None):
        super().__init__()
        self._info = {} if info is None else dict(info)
        self._complementary_data = {} if complementary_data is None else dict(complementary_data)

    def __call__(self, transition):
        out = create_transition(
            action=transition[TransitionKey.ACTION],
            reward=transition[TransitionKey.REWARD],
            done=transition[TransitionKey.DONE],
            truncated=transition[TransitionKey.TRUNCATED],
            info={**transition[TransitionKey.INFO], **self._info},
            complementary_data={**transition[TransitionKey.COMPLEMENTARY_DATA], **self._complementary_data},
        )
        return out


class StaticTransition:
    """Transition stub returning a fixed outcome."""

    def __init__(self, source: str, target: str, outcome: TransitionOutcome):
        self.source = source
        self.target = target
        self._outcome = outcome

    def evaluate(self, obs, info):
        return self._outcome


def _primitive(*, is_terminal: bool = False, policy=None):
    return SimpleNamespace(
        is_terminal=is_terminal,
        policy=policy,
        features={ACTION: SimpleNamespace(shape=(1,))},
    )


def _make_net(
    *,
    envs: dict[str, DummyEnv],
    transitions: dict[str, list[StaticTransition]] | None = None,
    active: str = "pick",
    start: str = "pick",
    reset: str = "pick",
    primitives: dict[str, SimpleNamespace] | None = None,
):
    net = ManipulationPrimitiveNet.__new__(ManipulationPrimitiveNet)
    net._envs = envs
    net._env_processors = {name: IdentityProcessor() for name in envs}
    net._action_processors = {name: IdentityProcessor() for name in envs}
    net._transitions = {name: [] for name in envs}
    if transitions is not None:
        net._transitions.update(transitions)

    if primitives is None:
        primitives = {name: _primitive() for name in envs}

    net.config = SimpleNamespace(
        primitives=primitives,
        start_primitive=start,
        reset_primitive=reset,
        fps=10_000,
        terminals=[name for name, primitive in primitives.items() if primitive.is_terminal],
    )
    net._active = active
    net._last_reset_info = {}
    net._episode_step_count = 0
    net._primitive_step_count = 0
    net._needs_full_reset = False
    return net


def test_mp_net_step_rejects_calls_before_reset():
    """MP-Net lifecycle: completed episodes must be reset before the next step."""
    net = _make_net(envs={"pick": DummyEnv(obs={"obs": torch.tensor([1.0])})})
    net._needs_full_reset = True

    with pytest.raises(RuntimeError, match="call reset"):
        net.step(torch.tensor([0.0]))


def test_mp_net_step_executes_active_primitive_and_emits_transition_metadata():
    """Step orchestration: the active primitive env should run and emit per-step routing metadata."""
    pick_env = DummyEnv(obs={"obs": torch.tensor([1.0])}, reward=0.5)
    place_env = DummyEnv(obs={"obs": torch.tensor([2.0])}, reward=1.0)
    net = _make_net(
        envs={"pick": pick_env, "place": place_env},
        transitions={"pick": []},
    )

    transition = net.step(torch.tensor([0.2]))

    assert torch.equal(transition[TransitionKey.OBSERVATION]["obs"], torch.tensor([1.0]))
    assert transition[TransitionKey.REWARD] == pytest.approx(0.5)
    assert transition[TransitionKey.DONE] is False
    assert transition[TransitionKey.TRUNCATED] is False
    assert pick_env.last_action is not None
    assert place_env.last_action is None
    assert net._active == "pick"
    assert transition[TransitionKey.INFO]["primitive_step"] == 1
    assert transition[TransitionKey.INFO]["episode_step"] == 1
    assert transition[TransitionKey.INFO]["transition_from"] == "pick"
    assert transition[TransitionKey.INFO]["transition_to"] == "pick"
    assert transition[TransitionKey.INFO]["transition_reason"] is None


def test_mp_net_step_applies_transition_outcome_and_switches_active_primitive():
    """Step orchestration: fired transitions should update active primitive, reward, and done flags."""
    pick_env = DummyEnv(obs={"obs": torch.tensor([1.0])}, reward=0.5)
    place_env = DummyEnv(obs={"obs": torch.tensor([2.0])}, reward=1.0)
    net = _make_net(
        envs={"pick": pick_env, "place": place_env},
        transitions={
            "pick": [
                StaticTransition(
                    source="pick",
                    target="place",
                    outcome=TransitionOutcome(reward=1.75, terminated=True, reason="success"),
                )
            ]
        },
    )

    transition = net.step(torch.tensor([0.0]))

    assert net._active == "place"
    assert transition[TransitionKey.REWARD] == pytest.approx(2.25)
    assert transition[TransitionKey.DONE] is True
    assert transition[TransitionKey.TRUNCATED] is False
    assert transition[TransitionKey.INFO]["transition_from"] == "pick"
    assert transition[TransitionKey.INFO]["transition_to"] == "place"
    assert transition[TransitionKey.INFO]["transition_reason"] == "success"


def test_mp_net_step_records_teleop_action_during_intervention():
    """Action recording: teleop overrides should be stored instead of the policy placeholder action."""
    teleop_action = torch.tensor([0.4, -0.2], dtype=torch.float32)
    pick_env = DummyEnv(obs={"obs": torch.tensor([1.0])})
    net = _make_net(envs={"pick": pick_env}, primitives={"pick": _primitive(policy=None)})
    net._action_processors["pick"] = StaticActionProcessor(
        info={},
        complementary_data={TELEOP_ACTION_KEY: teleop_action},
    )

    transition = net.step(torch.tensor([0.0]))

    assert torch.equal(transition[TransitionKey.ACTION], teleop_action)


def test_mp_net_reset_routes_from_reset_primitive_to_start(monkeypatch):
    """Reset orchestration: reset() should internally walk reset transitions before returning start obs."""
    monkeypatch.setattr(
        "share.envs.manipulation_primitive_net.env_manipulation_primitive_net.precise_sleep",
        lambda _dt: None,
    )

    reset_env = DummyEnv(obs={"obs": torch.tensor([9.0])})
    start_env = DummyEnv(obs={"obs": torch.tensor([3.0])})
    net = _make_net(
        envs={"reset": reset_env, "start": start_env},
        transitions={
            "reset": [
                StaticTransition(
                    source="reset",
                    target="start",
                    outcome=TransitionOutcome(terminated=True, reason="reset_complete"),
                )
            ],
            "start": [],
        },
        active="place",
        start="start",
        reset="reset",
        primitives={
            "reset": _primitive(),
            "start": _primitive(),
        },
    )
    net._envs["place"] = DummyEnv(obs={"obs": torch.tensor([0.0])})
    net._env_processors["place"] = IdentityProcessor()
    net._action_processors["place"] = IdentityProcessor()
    net._transitions["place"] = []
    net.config.primitives["place"] = _primitive()
    net._needs_full_reset = True

    transition = net.reset(seed=7)

    assert net._active == "start"
    assert transition[TransitionKey.OBSERVATION]["obs"].equal(torch.tensor([3.0]))
    assert reset_env.reset_calls and start_env.reset_calls
    assert net._episode_step_count == 0
    assert net._primitive_step_count == 0


def test_mp_net_reset_resets_all_processors(monkeypatch):
    """Reset orchestration: env/action processors for every primitive should receive reset callbacks."""
    monkeypatch.setattr(
        "share.envs.manipulation_primitive_net.env_manipulation_primitive_net.precise_sleep",
        lambda _dt: None,
    )

    net = _make_net(
        envs={"pick": DummyEnv(obs={"obs": torch.tensor([1.0])})},
        active="pick",
        start="pick",
        reset="pick",
    )
    net._needs_full_reset = True

    transition = net.reset(seed=3)

    assert transition[TransitionKey.INFO]["reset_seed"] == 426
    assert net._env_processors["pick"].reset_count == 1
    assert net._action_processors["pick"].reset_count == 1
