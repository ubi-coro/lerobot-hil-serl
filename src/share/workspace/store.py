"""Filesystem-backed workspace storage for MP-Net programming sessions."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp with second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def load_json(path: Path, default: Any | None = None) -> Any:
    """Load JSON from disk or return the provided default."""
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload: Any) -> None:
    """Write JSON to disk with a stable, human-readable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")


@dataclass(slots=True)
class MPNetRecord:
    """Metadata for one workspace-managed MP-Net."""

    name: str
    current_version: str | None = None
    origin_path: str | None = None
    description: str | None = None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)


@dataclass(slots=True)
class TaskRecord:
    """Task-level metadata tracked inside the workspace."""

    name: str
    description: str | None = None
    active_mp_net: str | None = None
    mp_nets: dict[str, MPNetRecord] = field(default_factory=dict)
    latest_run_id: str | None = None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)


@dataclass(slots=True)
class ProjectRecord:
    """Project metadata stored in the workspace."""

    name: str
    description: str | None = None
    tasks: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)


@dataclass(slots=True)
class Workspace:
    """Top-level workspace metadata."""

    schema_version: int = 1
    name: str = "mp-net-workspace"
    current_project: str | None = None
    current_task: str | None = None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)


@dataclass(slots=True)
class ArtifactRef:
    """A human-inspectable reference to a dataset, policy, report, or config."""

    artifact_id: str
    kind: str
    path: str
    project: str
    task: str
    label: str | None = None
    mp_net: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)


@dataclass(slots=True)
class RunRecord:
    """Metadata describing one launched or externally registered run."""

    run_id: str
    run_type: str
    status: str
    project: str
    task: str
    command: list[str] = field(default_factory=list)
    cwd: str | None = None
    mp_net: str | None = None
    mp_net_version: str | None = None
    primitive: str | None = None
    robot_type: str | None = None
    policy_id: str | None = None
    dataset_id: str | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None
    summary_path: str | None = None
    output_paths: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None
    started_at: str = field(default_factory=utc_now)
    ended_at: str | None = None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)


@dataclass(slots=True)
class SessionEvent:
    """One persisted chat or tool event inside the agent session log."""

    timestamp: str
    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class WorkspaceStore:
    """CRUD helpers for the file-backed MP-Net workspace layout."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()

    @property
    def workspace_path(self) -> Path:
        return self.root / "workspace.json"

    def project_dir(self, project: str) -> Path:
        return self.root / "projects" / project

    def project_path(self, project: str) -> Path:
        return self.project_dir(project) / "project.json"

    def task_dir(self, project: str, task: str) -> Path:
        return self.project_dir(project) / "tasks" / task

    def task_path(self, project: str, task: str) -> Path:
        return self.task_dir(project, task) / "task.json"

    def notes_path(self, project: str, task: str) -> Path:
        return self.task_dir(project, task) / "notes.md"

    def agent_dir(self, project: str, task: str) -> Path:
        return self.task_dir(project, task) / "agent"

    def session_path(self, project: str, task: str, session_id: str) -> Path:
        return self.agent_dir(project, task) / "sessions" / f"{session_id}.jsonl"

    def memory_path(self, project: str, task: str) -> Path:
        return self.agent_dir(project, task) / "memory.md"

    def mp_nets_dir(self, project: str, task: str) -> Path:
        return self.task_dir(project, task) / "mp_nets"

    def mp_net_dir(self, project: str, task: str, mp_net: str) -> Path:
        return self.mp_nets_dir(project, task) / mp_net

    def mp_net_current_path(self, project: str, task: str, mp_net: str) -> Path:
        return self.mp_net_dir(project, task, mp_net) / "current.json"

    def mp_net_version_path(self, project: str, task: str, mp_net: str, version_id: str) -> Path:
        return self.mp_net_dir(project, task, mp_net) / "versions" / f"{version_id}.json"

    def runs_dir(self, project: str, task: str) -> Path:
        return self.task_dir(project, task) / "runs"

    def run_dir(self, project: str, task: str, run_id: str) -> Path:
        return self.runs_dir(project, task) / run_id

    def run_path(self, project: str, task: str, run_id: str) -> Path:
        return self.run_dir(project, task, run_id) / "run.json"

    def artifacts_dir(self, project: str, task: str) -> Path:
        return self.task_dir(project, task) / "artifacts"

    def artifact_dir(self, project: str, task: str, artifact_id: str) -> Path:
        return self.artifacts_dir(project, task) / artifact_id

    def artifact_path(self, project: str, task: str, artifact_id: str) -> Path:
        return self.artifact_dir(project, task, artifact_id) / "artifact.json"

    def ensure_workspace(
        self,
        name: str = "mp-net-workspace",
        project: str = "default",
        task: str = "default",
    ) -> Workspace:
        """Create the workspace layout if it does not already exist."""
        self.root.mkdir(parents=True, exist_ok=True)
        workspace = self.load_workspace()
        if workspace is None:
            workspace = Workspace(name=name, current_project=project, current_task=task)
        else:
            workspace.name = workspace.name or name
            workspace.current_project = workspace.current_project or project
            workspace.current_task = workspace.current_task or task
            workspace.updated_at = utc_now()
        self.save_workspace(workspace)
        self.ensure_project(project)
        self.ensure_task(project, task)
        return workspace

    def load_workspace(self) -> Workspace | None:
        payload = load_json(self.workspace_path)
        if payload is None:
            return None
        return Workspace(**payload)

    def save_workspace(self, workspace: Workspace) -> None:
        workspace.updated_at = utc_now()
        dump_json(self.workspace_path, workspace)

    def ensure_project(self, project: str, description: str | None = None) -> ProjectRecord:
        """Create or load one project record."""
        payload = load_json(self.project_path(project))
        if payload is None:
            record = ProjectRecord(name=project, description=description)
        else:
            record = ProjectRecord(**payload)
            if description and not record.description:
                record.description = description
            record.updated_at = utc_now()
        dump_json(self.project_path(project), record)
        return record

    def load_project(self, project: str) -> ProjectRecord:
        payload = load_json(self.project_path(project))
        if payload is None:
            raise FileNotFoundError(f"Project '{project}' does not exist in workspace {self.root}")
        return ProjectRecord(**payload)

    def save_project(self, project: ProjectRecord) -> None:
        project.updated_at = utc_now()
        dump_json(self.project_path(project.name), project)

    def ensure_task(self, project: str, task: str, description: str | None = None) -> TaskRecord:
        """Create or load one task and link it from its parent project."""
        project_record = self.ensure_project(project)
        payload = load_json(self.task_path(project, task))
        if payload is None:
            task_record = TaskRecord(name=task, description=description)
        else:
            task_record = self._task_from_payload(payload)
            if description and not task_record.description:
                task_record.description = description
            task_record.updated_at = utc_now()
        if task not in project_record.tasks:
            project_record.tasks.append(task)
            self.save_project(project_record)
        dump_json(self.task_path(project, task), task_record)
        notes_path = self.notes_path(project, task)
        notes_path.parent.mkdir(parents=True, exist_ok=True)
        if not notes_path.exists():
            notes_path.write_text("", encoding="utf-8")
        memory_path = self.memory_path(project, task)
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        if not memory_path.exists():
            memory_path.write_text("", encoding="utf-8")
        return task_record

    def _task_from_payload(self, payload: dict[str, Any]) -> TaskRecord:
        mp_nets = {
            name: MPNetRecord(**record_payload)
            for name, record_payload in payload.get("mp_nets", {}).items()
        }
        payload = dict(payload)
        payload["mp_nets"] = mp_nets
        return TaskRecord(**payload)

    def load_task(self, project: str, task: str) -> TaskRecord:
        payload = load_json(self.task_path(project, task))
        if payload is None:
            raise FileNotFoundError(f"Task '{task}' does not exist in project '{project}'")
        return self._task_from_payload(payload)

    def save_task(self, project: str, task: TaskRecord) -> None:
        task.updated_at = utc_now()
        dump_json(self.task_path(project, task.name), task)

    def set_active_task(self, project: str, task: str) -> None:
        """Persist the currently selected project/task pair."""
        workspace = self.load_workspace()
        if workspace is None:
            workspace = Workspace(current_project=project, current_task=task)
        else:
            workspace.current_project = project
            workspace.current_task = task
        self.save_workspace(workspace)

    def read_notes(self, project: str, task: str) -> str:
        return self.notes_path(project, task).read_text(encoding="utf-8") if self.notes_path(project, task).exists() else ""

    def save_notes(self, project: str, task: str, notes: str) -> None:
        path = self.notes_path(project, task)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(notes, encoding="utf-8")

    def read_memory(self, project: str, task: str) -> str:
        return self.memory_path(project, task).read_text(encoding="utf-8") if self.memory_path(project, task).exists() else ""

    def save_memory(self, project: str, task: str, text: str) -> None:
        path = self.memory_path(project, task)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def register_mp_net(
        self,
        project: str,
        task: str,
        mp_net: str,
        version_id: str,
        origin_path: str | None = None,
        description: str | None = None,
        set_active: bool = True,
    ) -> None:
        """Update task metadata after saving an MP-Net version."""
        task_record = self.ensure_task(project, task)
        existing = task_record.mp_nets.get(mp_net)
        if existing is None:
            existing = MPNetRecord(name=mp_net, origin_path=origin_path, description=description)
        existing.current_version = version_id
        existing.origin_path = origin_path or existing.origin_path
        existing.description = description or existing.description
        existing.updated_at = utc_now()
        task_record.mp_nets[mp_net] = existing
        if set_active:
            task_record.active_mp_net = mp_net
        self.save_task(project, task_record)

    def list_mp_nets(self, project: str, task: str) -> list[MPNetRecord]:
        task_record = self.load_task(project, task)
        return sorted(task_record.mp_nets.values(), key=lambda item: item.name)

    def create_run(
        self,
        project: str,
        task: str,
        run_type: str,
        status: str = "pending",
        **kwargs: Any,
    ) -> RunRecord:
        """Create a run directory and initial run metadata."""
        run_id = kwargs.pop("run_id", f"{run_type}-{utc_now().replace(':', '-').replace('+00:00', 'z')}-{uuid.uuid4().hex[:8]}")
        run_dir = self.run_dir(project, task, run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = kwargs.pop("stdout_path", str(run_dir / "stdout.log"))
        stderr_path = kwargs.pop("stderr_path", str(run_dir / "stderr.log"))
        summary_path = kwargs.pop("summary_path", str(run_dir / "summary.json"))
        record = RunRecord(
            run_id=run_id,
            run_type=run_type,
            status=status,
            project=project,
            task=task,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            summary_path=summary_path,
            **kwargs,
        )
        self.save_run(project, task, record)
        task_record = self.ensure_task(project, task)
        task_record.latest_run_id = run_id
        self.save_task(project, task_record)
        return record

    def save_run(self, project: str, task: str, record: RunRecord) -> None:
        record.updated_at = utc_now()
        dump_json(self.run_path(project, task, record.run_id), record)

    def load_run(self, project: str, task: str, run_id: str) -> RunRecord:
        payload = load_json(self.run_path(project, task, run_id))
        if payload is None:
            raise FileNotFoundError(f"Run '{run_id}' does not exist for {project}/{task}")
        return RunRecord(**payload)

    def list_runs(self, project: str, task: str) -> list[RunRecord]:
        run_root = self.runs_dir(project, task)
        if not run_root.exists():
            return []
        records: list[RunRecord] = []
        for run_path in sorted(run_root.glob("*/run.json")):
            payload = load_json(run_path)
            if payload is not None:
                records.append(RunRecord(**payload))
        return sorted(records, key=lambda item: item.started_at, reverse=True)

    def save_artifact(self, project: str, task: str, artifact: ArtifactRef) -> None:
        dump_json(self.artifact_path(project, task, artifact.artifact_id), artifact)

    def load_artifacts(self, project: str, task: str) -> list[ArtifactRef]:
        artifact_root = self.artifacts_dir(project, task)
        if not artifact_root.exists():
            return []
        artifacts: list[ArtifactRef] = []
        for artifact_path in sorted(artifact_root.glob("*/artifact.json")):
            payload = load_json(artifact_path)
            if payload is not None:
                artifacts.append(ArtifactRef(**payload))
        return sorted(artifacts, key=lambda item: item.created_at, reverse=True)

    def append_session_event(self, project: str, task: str, session_id: str, event: SessionEvent) -> None:
        """Append one JSONL event to the persisted session transcript."""
        path = self.session_path(project, task, session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(event), sort_keys=True))
            f.write("\n")

    def load_session_events(self, project: str, task: str, session_id: str) -> list[SessionEvent]:
        path = self.session_path(project, task, session_id)
        if not path.exists():
            return []
        events: list[SessionEvent] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                events.append(SessionEvent(**json.loads(line)))
        return events

