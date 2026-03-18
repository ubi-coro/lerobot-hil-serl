from __future__ import annotations

from share.workspace.providers import LLMProvider, ProviderConfig
from share.workspace.runtime import AgentRuntime
from share.workspace.store import WorkspaceStore
from share.workspace.tools import WorkspaceToolbox


class _QueuedProvider(LLMProvider):
    def __init__(self, payloads: list[str]):
        super().__init__(ProviderConfig(provider_type="queued", model="queued"))
        self.payloads = payloads

    def complete(self, messages, tools):  # type: ignore[override]
        return self._parse_response_text(self.payloads.pop(0))


def test_runtime_requires_confirmation_for_mutating_tools(tmp_path):
    store = WorkspaceStore(tmp_path / "workspace")
    store.ensure_workspace(project="proj", task="task")
    toolbox = WorkspaceToolbox(store=store, project="proj", task="task")
    provider = _QueuedProvider(
        [
            '{"message":"I should store the note.","tool_calls":[{"name":"save_notes","arguments":{"notes":"hello"}}]}'
        ]
    )
    runtime = AgentRuntime(store=store, toolbox=toolbox, provider=provider, project="proj", task="task")

    result = runtime.handle_user_message("save this note")

    assert result.pending_confirmation is not None
    assert result.pending_confirmation.tool_call.name == "save_notes"


def test_runtime_executes_tool_and_summarizes_follow_up(tmp_path):
    store = WorkspaceStore(tmp_path / "workspace")
    store.ensure_workspace(project="proj", task="task")
    toolbox = WorkspaceToolbox(store=store, project="proj", task="task")
    provider = _QueuedProvider(
        [
            '{"message":"Checking status.","tool_calls":[{"name":"status","arguments":{}}]}',
            '{"message":"Workspace status is available.","tool_calls":[]}',
        ]
    )
    runtime = AgentRuntime(store=store, toolbox=toolbox, provider=provider, project="proj", task="task")

    result = runtime.handle_user_message("what is the current status?")

    assert result.pending_confirmation is None
    assert len(result.tool_results) == 1
    assert result.tool_results[0].name == "status"
    assert result.message == "Workspace status is available."
