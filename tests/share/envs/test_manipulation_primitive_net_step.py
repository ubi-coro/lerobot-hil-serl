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

    def step(self, action):
        self.last_action = action
        return self.obs, self.reward, self.terminated, self.truncated, dict(self.info)


class StaticBoolTransition:
    def __init__(self, should_fire: bool):
        self.should_fire = should_fire

    def check(self, obs, info):
        return self.should_fire


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
    net.config = SimpleNamespace(transitions=transitions)
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
