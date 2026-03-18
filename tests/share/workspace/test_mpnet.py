from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from share.workspace import mpnet


@dataclass
class _FakePolicyConfig:
    pretrained_path: Path | None = None
    device: str = "cpu"


def test_mpnet_editing_round_trip_supports_versions_and_axis_edits(tmp_path):
    config = mpnet.create_template_mpnet(notes="Initial primitive")
    config = mpnet.add_primitive(
        config,
        name="handoff",
        is_terminal=True,
        notes="Transfer to the next primitive.",
        connect_from="main",
        connect_transition_type="always",
    )
    config = mpnet.set_learnable_axes(
        config,
        primitive_name="main",
        axes={"x": "relative", "y": "relative", "z": "absolute"},
    )
    config = mpnet.set_axis_targets(config, primitive_name="main", targets={"x": 0.4, "z": 0.2})
    config = mpnet.set_primitive_notes(config, primitive_name="main", notes="Updated primitive intent.")

    path = tmp_path / "current.json"
    mpnet.save_mpnet_config(config, path)
    loaded = mpnet.load_mpnet_config(path)
    summary = mpnet.summarize_mpnet(loaded)

    assert summary["primitive_count"] == 2
    assert summary["transition_count"] == 1
    main_summary = next(item for item in summary["primitives"] if item["name"] == "main")
    assert set(main_summary["task_frames"]["default"]["learnable_axes"]) == {"x", "y", "z"}
    assert main_summary["notes"] == "Updated primitive intent."


def test_attach_policy_uses_pretrained_loader(monkeypatch):
    config = mpnet.create_template_mpnet()

    def fake_from_pretrained(policy_path: str, local_files_only: bool = True):
        assert policy_path == "/tmp/policy"
        assert local_files_only is True
        return _FakePolicyConfig()

    monkeypatch.setattr(mpnet.PreTrainedConfig, "from_pretrained", staticmethod(fake_from_pretrained))
    updated = mpnet.attach_policy(config, primitive_name="main", policy_path="/tmp/policy")

    assert updated.primitives["main"].policy is not None
    assert str(updated.primitives["main"].policy.pretrained_path) == "/tmp/policy"


def test_remove_transition_rejects_invalid_graph():
    config = mpnet.create_template_mpnet()
    config = mpnet.add_primitive(
        config,
        name="done",
        is_terminal=True,
        connect_from="main",
        connect_transition_type="always",
    )

    with pytest.raises(ValueError):
        mpnet.remove_transition(config, index=0)
