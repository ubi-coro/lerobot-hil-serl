"""Graph-validation tests for ``ManipulationPrimitiveNetConfig``.

Each test covers one piece of MP-Net config validation so graph mistakes fail
at load time rather than during hardware rollouts.

Redundancy note: older draccus/YAML roundtrip coverage targeted a registration
shape that is no longer a stable contract for this base config class, so this
file now focuses on constructor-level graph validation only.
"""

from __future__ import annotations

import pytest

from share.envs.manipulation_primitive.config_manipulation_primitive import ManipulationPrimitiveConfig
from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import (
    ManipulationPrimitiveNetConfig,
)
from share.envs.manipulation_primitive_net.transitions import Always


def _primitive(*, is_terminal: bool = False) -> ManipulationPrimitiveConfig:
    return ManipulationPrimitiveConfig(is_terminal=is_terminal)


def test_mp_net_config_defaults_start_and_reset_to_first_primitive():
    """Config validation: default routing should start from the first declared primitive."""
    cfg = ManipulationPrimitiveNetConfig(
        primitives={"pick": _primitive(), "place": _primitive(is_terminal=True)},
        transitions=[Always(source="pick", target="place")],
    )

    assert cfg.start_primitive == "pick"
    assert cfg.reset_primitive == "pick"


def test_mp_net_config_rejects_unknown_transition_source():
    """Config validation: transition sources must reference declared primitive names."""
    with pytest.raises(ValueError, match="Transition source 'unknown'"):
        ManipulationPrimitiveNetConfig(
            primitives={"pick": _primitive(), "place": _primitive(is_terminal=True)},
            transitions=[Always(source="unknown", target="place")],
        )


def test_mp_net_config_rejects_unknown_transition_target():
    """Config validation: transition targets must reference declared primitive names."""
    with pytest.raises(ValueError, match="Transition target 'unknown'"):
        ManipulationPrimitiveNetConfig(
            primitives={"pick": _primitive(), "place": _primitive(is_terminal=True)},
            transitions=[Always(source="pick", target="unknown")],
        )


def test_mp_net_config_rejects_missing_start_primitive():
    """Config validation: explicit start primitives must exist in the primitive registry."""
    with pytest.raises(ValueError, match="start_primitive 'missing'"):
        ManipulationPrimitiveNetConfig(
            start_primitive="missing",
            primitives={"pick": _primitive(), "place": _primitive(is_terminal=True)},
            transitions=[Always(source="pick", target="place")],
        )


def test_mp_net_config_rejects_non_terminal_dead_end():
    """Config validation: non-terminal primitives need at least one outgoing edge."""
    with pytest.raises(ValueError, match="non-terminal dead-end primitive"):
        ManipulationPrimitiveNetConfig(
            primitives={
                "pick": _primitive(),
                "place": _primitive(),
                "done": _primitive(is_terminal=True),
            },
            transitions=[Always(source="pick", target="done")],
        )


def test_mp_net_config_allows_terminal_dead_end_when_marked_terminal():
    """Config validation: terminal primitives may intentionally have no outgoing transitions."""
    cfg = ManipulationPrimitiveNetConfig(
        primitives={"pick": _primitive(), "done": _primitive(is_terminal=True)},
        transitions=[Always(source="pick", target="done")],
    )

    assert cfg.terminals == ["done"]


def test_mp_net_config_rejects_unreachable_terminal_primitive():
    """Config validation: terminal primitives should be reachable from the configured start node."""
    with pytest.raises(ValueError, match="unreachable from start_primitive"):
        ManipulationPrimitiveNetConfig(
            start_primitive="pick",
            primitives={
                "pick": _primitive(),
                "place": _primitive(),
                "done": _primitive(is_terminal=True),
            },
            transitions=[Always(source="pick", target="place"), Always(source="place", target="pick")],
        )
