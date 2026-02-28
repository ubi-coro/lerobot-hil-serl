from types import SimpleNamespace

import numpy as np

from share.envs.manipulation_primitive_net.env_manipulation_primitive_net import ManipulationPrimitiveNet


class DummyEnv:
    def __init__(self, obs, reward=0.0, terminated=False, truncated=False, info=None):
        self.obs = obs
        self.reward = reward
        self.terminated = terminated
        self.truncated = truncated
        self.info = info or {}
        self.last_action = None
        self.reset_calls = []

    def step(self, action):
        self.last_action = action
        return self.obs, self.reward, self.terminated, self.truncated, dict(self.info)

    def reset(self, *, seed=None, options=None):
        self.reset_calls.append({"seed": seed, "options": dict(options or {})})
        return self.obs, {"env": "dummy", "seed": seed}


class StaticBoolTransition:
    def __init__(self, should_fire: bool):
        self.should_fire = should_fire

    def check(self, obs, info):
        return self.should_fire


class SequencedTransition:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)

    def evaluate(self, obs, info):
        if self._outcomes:
            fired, metadata = self._outcomes.pop(0)
            return fired, metadata
        return False, {}


class RichTransition:
    def evaluate(self, obs, info):
        return True, {
            "next_primitive": "place",
            "additional_reward": 1.75,
            "terminated": True,
            "reason": "success",
            "transition_name": "success_edge",
            "transition_type": "threshold",
        }


def _make_net(envs, transitions, active="pick"):
    net = ManipulationPrimitiveNet.__new__(ManipulationPrimitiveNet)
    net._envs = envs
    net._active_primitive = active
    net.config = SimpleNamespace(transitions=transitions, start_primitive=active, reset_primitives=[])
    net._episode_step_count = 0
    net._last_reset_info = {}
    return net


def test_mp_net_step_executes_active_primitive():
    pick_env = DummyEnv(obs={"obs": np.array([1.0])}, reward=0.5)
    place_env = DummyEnv(obs={"obs": np.array([2.0])}, reward=5.0)
    net = _make_net(
        envs={"pick": pick_env, "place": place_env},
        transitions=[("pick", "place", StaticBoolTransition(False))],
    )

    obs, reward, terminated, truncated, info = net.step(np.array([0.2, -0.1]))

    assert np.allclose(obs["obs"], np.array([1.0]))
    assert reward == 0.5
    assert not terminated
    assert not truncated
    assert pick_env.last_action is not None
    assert place_env.last_action is None
    assert net._active_primitive == "pick"
    assert info["active_primitive"] == "pick"
    assert info["transition"]["from"] == "pick"
    assert info["transition"]["to"] == "pick"


def test_mp_net_step_switches_primitive_when_transition_fires():
    pick_env = DummyEnv(obs={"obs": np.array([1.0])}, reward=0.25)
    net = _make_net(
        envs={"pick": pick_env, "place": DummyEnv(obs={"obs": np.array([2.0])})},
        transitions=[("pick", "place", StaticBoolTransition(True))],
    )

    _, reward, terminated, truncated, info = net.step(np.array([0.0]))

    assert reward == 0.25
    assert not terminated
    assert not truncated
    assert net._active_primitive == "place"
    assert info["active_primitive"] == "place"
    assert info["transition"]["from"] == "pick"
    assert info["transition"]["to"] == "place"
    assert info["transition"]["reason"] == "transition_fired"


def test_mp_net_step_applies_transition_reward_and_done_flags():
    pick_env = DummyEnv(obs={"obs": np.array([1.0])}, reward=0.5)
    net = _make_net(
        envs={"pick": pick_env, "place": DummyEnv(obs={"obs": np.array([2.0])})},
        transitions=[("pick", "place", RichTransition())],
    )

    _, reward, terminated, truncated, info = net.step(np.array([0.0]))

    assert reward == 2.25
    assert terminated
    assert not truncated
    assert net._active_primitive == "place"
    assert info["transition"]["from"] == "pick"
    assert info["transition"]["to"] == "place"
    assert info["transition"]["reason"] == "success"
    assert info["transition"]["transition_name"] == "success_edge"
    assert info["transition"]["transition_type"] == "threshold"


def test_mp_net_reset_starts_from_start_primitive():
    pick_env = DummyEnv(obs={"obs": np.array([1.0])})
    place_env = DummyEnv(obs={"obs": np.array([2.0])})
    net = _make_net(
        envs={"pick": pick_env, "place": place_env},
        transitions=[],
        active="place",
    )
    net.config.start_primitive = "pick"
    net.config.reset_primitives = ["reset"]
    net._episode_step_count = 99

    obs, info = net.reset(seed=7)

    assert np.allclose(obs["obs"], np.array([1.0]))
    assert net._active_primitive == "pick"
    assert net._episode_step_count == 0
    assert info["active_primitive"] == "pick"
    assert info["reset_reason"] == "start_primitive"
    assert len(pick_env.reset_calls) == 1
    assert len(place_env.reset_calls) == 1


def test_mp_net_reset_path_via_reset_primitives():
    reset_env = DummyEnv(obs={"obs": np.array([9.0])})
    pick_env = DummyEnv(obs={"obs": np.array([3.0])})
    place_env = DummyEnv(obs={"obs": np.array([2.0])})
    net = _make_net(
        envs={"reset": reset_env, "pick": pick_env, "place": place_env},
        transitions=[("reset", "pick", StaticBoolTransition(True))],
        active="place",
    )
    net.config.start_primitive = "pick"
    net.config.reset_primitives = ["reset", "alt_reset"]

    obs, info = net.reset()

    assert np.allclose(obs["obs"], np.array([9.0]))
    assert net._active_primitive == "pick"
    assert info["active_primitive"] == "pick"
    assert info["reset_reason"] == "reset_primitive_fallback"
    assert info["reset_transition_steps"] == 1
    assert len(reset_env.reset_calls) == 1
    assert len(pick_env.reset_calls) == 1
    assert len(place_env.reset_calls) == 1


def test_mp_net_reset_path_steps_intermediate_primitives_until_start():
    reset_env = DummyEnv(obs={"obs": np.array([9.0])})
    stage_env = DummyEnv(obs={"obs": np.array([7.0])})
    start_env = DummyEnv(obs={"obs": np.array([5.0])})

    net = _make_net(
        envs={"reset": reset_env, "stage": stage_env, "start": start_env},
        transitions=[
            ("reset", "stage", SequencedTransition([(True, {})])),
            ("stage", "start", SequencedTransition([(True, {})])),
        ],
        active="stage",
    )
    net.config.start_primitive = "start"
    net.config.reset_primitives = ["reset"]

    _, info = net.reset()

    assert net._active_primitive == "start"
    assert info["active_primitive"] == "start"
    assert info["reset_transition_steps"] == 2
    assert stage_env.last_action is not None
