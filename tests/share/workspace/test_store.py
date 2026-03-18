from __future__ import annotations

from share.workspace.store import ArtifactRef, SessionEvent, WorkspaceStore


def test_workspace_store_creates_expected_layout_and_records(tmp_path):
    store = WorkspaceStore(tmp_path / "workspace")
    workspace = store.ensure_workspace(name="demo", project="pick", task="block")
    store.set_active_task("pick", "block")

    assert workspace.name == "demo"
    assert store.workspace_path.exists()
    assert store.project_path("pick").exists()
    assert store.task_path("pick", "block").exists()
    assert store.notes_path("pick", "block").exists()
    assert store.memory_path("pick", "block").exists()

    store.save_notes("pick", "block", "Collect more demos for approach.")
    assert store.read_notes("pick", "block") == "Collect more demos for approach."

    artifact = ArtifactRef(
        artifact_id="dataset-1",
        kind="dataset",
        path="/tmp/dataset",
        project="pick",
        task="block",
    )
    store.save_artifact("pick", "block", artifact)
    assert store.load_artifacts("pick", "block")[0].artifact_id == "dataset-1"

    run = store.create_run("pick", "block", run_type="evaluation", status="succeeded", metrics={"success_rate": 0.7})
    reloaded_run = store.load_run("pick", "block", run.run_id)
    assert reloaded_run.metrics == {"success_rate": 0.7}

    store.append_session_event(
        "pick",
        "block",
        "session-1",
        SessionEvent(timestamp="2026-01-01T00:00:00+00:00", role="user", content="status"),
    )
    events = store.load_session_events("pick", "block", "session-1")
    assert len(events) == 1
    assert events[0].content == "status"
