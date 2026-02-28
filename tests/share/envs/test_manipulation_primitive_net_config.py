from types import SimpleNamespace

import pytest

from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import (
    ManipulationPrimitiveNetConfig,
)
from share.envs.manipulation_primitive_net.transitions import ObservationThresholdTransition


def _transition(*, next_primitive=None):
    return ObservationThresholdTransition(obs_key="x", threshold=0.0, next_primitive=next_primitive)


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
        transitions=[("pick", "terminal", _transition()), ("terminal", "reset", _transition())],
        reset_primitives=["reset"],
    )

    assert config.start_primitive == "pick"


def test_mp_net_config_rejects_non_terminal_dead_end():
    with pytest.raises(ValueError, match="non-terminal dead-end primitive"):
        ManipulationPrimitiveNetConfig(
            start_primitive="pick",
            primitives={
                "pick": SimpleNamespace(),
                "place": SimpleNamespace(),
                "terminal": SimpleNamespace(is_terminal_primitive=True),
                "reset": SimpleNamespace(),
            },
            transitions=[
                ("pick", "terminal", _transition()),
                ("terminal", "reset", _transition()),
                ("reset", "pick", _transition()),
            ],
            reset_primitives=["reset"],
        )


def test_mp_net_config_rejects_unreachable_terminal():
    with pytest.raises(ValueError, match="unreachable from start_primitive"):
        ManipulationPrimitiveNetConfig(
            start_primitive="pick",
            primitives={
                "pick": SimpleNamespace(),
                "place": SimpleNamespace(),
                "terminal": SimpleNamespace(is_terminal_primitive=True),
                "reset": SimpleNamespace(),
            },
            transitions=[
                ("pick", "place", _transition()),
                ("place", "pick", _transition()),
                ("terminal", "reset", _transition()),
                ("reset", "pick", _transition()),
            ],
            reset_primitives=["reset"],
        )


def test_mp_net_config_rejects_reset_without_path_to_start():
    with pytest.raises(ValueError, match="Reset primitive has no transition path to start_primitive"):
        ManipulationPrimitiveNetConfig(
            start_primitive="pick",
            primitives={
                "pick": SimpleNamespace(),
                "terminal": SimpleNamespace(is_terminal_primitive=True),
                "reset": SimpleNamespace(),
            },
            transitions=[
                ("pick", "terminal", _transition()),
                ("terminal", "reset", _transition()),
                ("reset", "terminal", _transition()),
            ],
            reset_primitives=["reset"],
        )


def test_mp_net_config_rejects_unknown_next_primitive_override():
    with pytest.raises(ValueError, match="Transition resolver points to unknown primitive"):
        ManipulationPrimitiveNetConfig(
            start_primitive="pick",
            primitives={
                "pick": SimpleNamespace(),
                "terminal": SimpleNamespace(is_terminal_primitive=True),
                "reset": SimpleNamespace(),
            },
            transitions=[
                ("pick", "terminal", _transition(next_primitive="terminal")),
                ("terminal", "reset", _transition(next_primitive="bogus")),
                ("reset", "pick", _transition(next_primitive="pick")),
            ],
            reset_primitives=["reset"],
        )


def test_mp_net_config_allows_intentional_terminal_dead_end():
    config = ManipulationPrimitiveNetConfig(
        start_primitive="pick",
        primitives={
            "pick": SimpleNamespace(),
            "terminal": SimpleNamespace(is_terminal_primitive=True),
            "reset": SimpleNamespace(),
        },
        transitions=[
            ("pick", "terminal", _transition()),
            ("terminal", "reset", _transition()),
            ("reset", "pick", _transition()),
        ],
        reset_primitives=["reset"],
    )

    assert config.start_primitive == "pick"


def test_mp_net_config_collects_reset_primitives_from_metadata():
    config = ManipulationPrimitiveNetConfig(
        start_primitive="pick",
        primitives={
            "pick": SimpleNamespace(),
            "terminal": SimpleNamespace(is_terminal_primitive=True),
            "reset": SimpleNamespace(is_reset_primitive=True),
        },
        transitions=[
            ("pick", "terminal", _transition()),
            ("terminal", "reset", _transition()),
            ("reset", "pick", _transition()),
        ],
        reset_primitives=[],
    )

    assert config.reset_primitives == ["reset"]


def test_mp_net_config_rejects_reset_list_entry_without_metadata_flag():
    with pytest.raises(ValueError, match="must set is_reset_primitive=True"):
        ManipulationPrimitiveNetConfig(
            start_primitive="pick",
            primitives={
                "pick": SimpleNamespace(),
                "terminal": SimpleNamespace(is_terminal_primitive=True),
                "reset": SimpleNamespace(),
            },
            transitions=[
                ("pick", "terminal", _transition()),
                ("terminal", "reset", _transition()),
                ("reset", "pick", _transition()),
            ],
            reset_primitives=["reset"],
        )
