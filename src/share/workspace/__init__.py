"""Local-first MP-Net workspace primitives and agent runtime."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AgentRuntime",
    "ArtifactRef",
    "LLMProvider",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "PendingConfirmation",
    "ProviderConfig",
    "ProviderResponse",
    "RunRecord",
    "ToolCall",
    "ToolResult",
    "ToolSpec",
    "TurnResult",
    "Workspace",
    "WorkspaceStore",
]


def __getattr__(name: str) -> Any:
    if name in {"ArtifactRef", "RunRecord", "Workspace", "WorkspaceStore"}:
        module = import_module("share.workspace.store")
        return getattr(module, name)
    if name in {"LLMProvider", "OllamaProvider", "OpenAICompatibleProvider", "ProviderConfig", "ProviderResponse"}:
        module = import_module("share.workspace.providers")
        return getattr(module, name)
    if name in {"ToolCall", "ToolResult", "ToolSpec"}:
        module = import_module("share.workspace.tools")
        return getattr(module, name)
    if name in {"AgentRuntime", "PendingConfirmation", "TurnResult"}:
        module = import_module("share.workspace.runtime")
        return getattr(module, name)
    raise AttributeError(name)
