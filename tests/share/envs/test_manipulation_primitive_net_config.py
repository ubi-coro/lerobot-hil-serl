from types import SimpleNamespace

import pytest

from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import (
    ManipulationPrimitiveNetConfig,
)
from share.envs.manipulation_primitive_net.transitions import ObservationThresholdTransition


def _transition():
    return ObservationThresholdTransition(obs_key="x", threshold=0.0)


def test_mp_net_config_rejects_unknown_primitive_in_transition():
    with pytest.raises(ValueError, match="Transition target 'unknown'"):
        ManipulationPrimitiveNetConfig(
            start_primitive="pick",
            primitives={"pick": SimpleNamespace(), "reset": SimpleNamespace()},
            transitions=[("pick", "unknown", _transition())],
            reset_primitives=["reset"],
        )


def test_mp_net_config_rejects_missing_start_primitive():
    with pytest.raises(ValueError, match="start_primitive 'pick'"):
        ManipulationPrimitiveNetConfig(
            start_primitive="pick",
            primitives={"reset": SimpleNamespace()},
            transitions=[],
            reset_primitives=["reset"],
        )


def test_mp_net_config_rejects_terminal_transition_to_non_reset_primitive():
    with pytest.raises(ValueError, match="Terminal primitive transitions must target a reset primitive"):
        ManipulationPrimitiveNetConfig(
            start_primitive="pick",
            primitives={
                "pick": SimpleNamespace(),
                "terminal": SimpleNamespace(is_terminal_primitive=True),
                "reset": SimpleNamespace(),
            },
            transitions=[("terminal", "pick", _transition())],
            reset_primitives=["reset"],
        )


def test_mp_net_config_allows_terminal_transition_to_reset_primitive():
    config = ManipulationPrimitiveNetConfig(
        start_primitive="pick",
        primitives={
            "pick": SimpleNamespace(),
            "terminal": SimpleNamespace(is_terminal_primitive=True),
            "reset": SimpleNamespace(),
        },
        transitions=[("terminal", "reset", _transition())],
        reset_primitives=["reset"],
    )

    assert config.start_primitive == "pick"
