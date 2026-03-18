"""Tests for the migrated HAN insertion MP-net experiment config.

These tests cover the experiment-local functionality added in
``src/experiments/env/han_insertion_mp_net.py``:

- primitive-graph construction,
- randomized reset target propagation into explicit reset primitives,
- and the custom press-to-insert transition logic.
"""

import importlib.util
from pathlib import Path
import sys


_MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "src" / "experiments" / "env" / "han_insertion_mp_net.py"
)
_SPEC = importlib.util.spec_from_file_location("han_insertion_mp_net", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

HanInsertionMPNetEnvConfig = _MODULE.HanInsertionMPNetEnvConfig
InsertionReadyTransition = _MODULE.InsertionReadyTransition


def test_han_insertion_config_builds_expected_primitive_graph():
    """The migrated config should expose the reset path, press, insert, and terminal primitives."""

    cfg = HanInsertionMPNetEnvConfig(use_vision=False)

    assert cfg.start_primitive == "press"
    assert cfg.reset_primitive == "reset_lift"
    assert set(cfg.primitives) == {"reset_lift", "reset_hover", "reset_settle", "press", "insert", "terminal"}
    assert cfg.primitives["insert"].task_frame["main"].policy_action_dim == 3
    assert cfg.primitives["terminal"].is_terminal is True


def test_han_insertion_reset_pose_updates_explicit_reset_primitives():
    """Sampling a reset pose should update the hover and settle reset primitives consistently."""

    cfg = HanInsertionMPNetEnvConfig(use_vision=False)
    sampled_pose = cfg.sample_reset_pose(__import__("numpy").random.default_rng(0))
    cfg.apply_reset_pose(sampled_pose)

    hover_target = cfg.primitives["reset_hover"].task_frame["main"].target
    settle_target = cfg.primitives["reset_settle"].task_frame["main"].target

    assert settle_target == sampled_pose
    assert hover_target[0] == sampled_pose[0]
    assert hover_target[1] == sampled_pose[1]
    assert hover_target[5] == sampled_pose[5]
    assert hover_target[2] == min(cfg.reset_hover_z_m, cfg._workspace_max_pose[2])


def test_insertion_ready_transition_fires_on_force_or_depth():
    """The custom press transition should fire on either contact force or insertion depth."""

    transition = InsertionReadyTransition(
        source="press",
        target="insert",
        position_threshold=0.032,
        force_threshold=5.0,
    )

    assert transition.evaluate({"main.z.ee_pos": 0.0, "main.z.ee_wrench": 5.1}, {}).terminated is True
    assert transition.evaluate({"main.z.ee_pos": 0.0321, "main.z.ee_wrench": 0.0}, {}).terminated is True
    assert transition.evaluate({"main.z.ee_pos": 0.0, "main.z.ee_wrench": 0.0}, {}).terminated is False
