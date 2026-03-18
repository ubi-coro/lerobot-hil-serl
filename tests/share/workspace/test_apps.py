from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path

from share.scripts.eval_on_dataset import summarize_dataset_eval
from share.scripts.train import build_train_config
from share.workspace.store import WorkspaceStore
from share.workspace.tools import WorkspaceAppRunner


@dataclass
class _FakePolicy:
    pretrained_path: Path | None = None
    device: str = "cpu"


def test_app_runner_captures_logs_and_summary_fallback(monkeypatch, tmp_path):
    store = WorkspaceStore(tmp_path / "workspace")
    store.ensure_workspace(project="proj", task="task")
    runner = WorkspaceAppRunner(store=store, repo_root=tmp_path)

    def fake_run(command, cwd, text, capture_output, check):
        assert command == ["echo", "hello"]
        return subprocess.CompletedProcess(command, 0, stdout="hello\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    record = runner.run_subprocess(project="proj", task="task", run_type="demo", command=["echo", "hello"])

    assert Path(record.stdout_path).read_text(encoding="utf-8") == "hello\n"
    assert record.status == "succeeded"
    assert Path(record.summary_path).exists()


def test_train_adapter_builds_config(monkeypatch, tmp_path):
    def fake_from_pretrained(policy_path: str, local_files_only: bool = False):
        assert policy_path == "/tmp/policy"
        return _FakePolicy()

    monkeypatch.setattr("share.scripts.train.PreTrainedConfig.from_pretrained", fake_from_pretrained)
    args = argparse.Namespace(
        policy_path="/tmp/policy",
        dataset_repo_id="demo/repo",
        dataset_root=str(tmp_path / "dataset"),
        output_dir=str(tmp_path / "output"),
        job_name="demo-job",
        policy_device=None,
        steps=42,
        batch_size=4,
        eval_freq=0,
        log_freq=10,
        save_freq=42,
        summary_path=None,
        local_files_only=True,
        no_save_checkpoint=False,
    )

    cfg = build_train_config(args)

    assert cfg.dataset.repo_id == "demo/repo"
    assert cfg.steps == 42
    assert str(cfg.policy.pretrained_path) == "/tmp/policy"


def test_eval_adapter_summarizes_primitive_datasets(tmp_path):
    dataset_root = tmp_path / "dataset"
    primitive_dir = dataset_root / "approach" / "meta"
    primitive_dir.mkdir(parents=True)
    (primitive_dir / "info.json").write_text(
        '{"repo_id":"demo/repo-approach","total_episodes":3,"total_frames":90,"codebase_version":"v1"}',
        encoding="utf-8",
    )

    summary = summarize_dataset_eval(dataset_root, "/tmp/policy")

    assert summary["overall"]["episode_count"] == 3
    assert summary["primitives"][0]["primitive"] == "approach"
