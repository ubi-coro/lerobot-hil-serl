"""Bounded conversational runtime for the MP-Net workspace agent."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from share.workspace.providers import LLMProvider, ProviderResponse
from share.workspace.store import SessionEvent, WorkspaceStore, utc_now
from share.workspace.tools import ToolCall, ToolResult, WorkspaceToolbox


@dataclass(slots=True)
class PendingConfirmation:
    """One queued tool call awaiting explicit human approval."""

    tool_call: ToolCall
    confirmation: str
    assistant_message: str


@dataclass(slots=True)
class TurnResult:
    """Runtime response to one user turn."""

    message: str
    tool_results: list[ToolResult] = field(default_factory=list)
    pending_confirmation: PendingConfirmation | None = None


class AgentRuntime:
    """A bounded tool-using agent over the local workspace."""

    def __init__(
        self,
        store: WorkspaceStore,
        toolbox: WorkspaceToolbox,
        provider: LLMProvider,
        *,
        project: str,
        task: str,
        max_tool_rounds: int = 4,
        session_id: str | None = None,
    ):
        self.store = store
        self.toolbox = toolbox
        self.provider = provider
        self.project = project
        self.task = task
        self.max_tool_rounds = max_tool_rounds
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.pending_confirmation: PendingConfirmation | None = None

    def _append_event(self, role: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        self.store.append_session_event(
            self.project,
            self.task,
            self.session_id,
            SessionEvent(timestamp=utc_now(), role=role, content=content, metadata=metadata or {}),
        )

    def _workspace_context(self) -> dict[str, Any]:
        status = self.toolbox.status()
        notes = self.toolbox.read_notes()
        memory = self.store.read_memory(self.project, self.task)
        return {
            "status": status.data,
            "notes": notes.data.get("notes", ""),
            "memory": memory,
        }

    def _build_messages(self, user_message: str, tool_results: list[ToolResult]) -> list[dict[str, str]]:
        context = self._workspace_context()
        events = self.store.load_session_events(self.project, self.task, self.session_id)[-12:]
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a bounded robotics workflow assistant operating over a structured robot programming workspace."

                    "Your purpose is to help the user iteratively create, inspect, execute, evaluate, and revise robot tasks "
                    "represented as manipulation primitive nets."
                    "You are not a general autonomous robot controller and not a generic coding assistant."

                    "Core principles:"

                    "- Treat the workspace state and current manipulation primitive net as the source of truth."
                    "- Translate user intent into structured edits, notes, tool calls, and clear next-step suggestions."
                    "- Prefer explicit, inspectable, reversible actions over freeform behavior."
                    "- Prefer existing tools and structured config updates over arbitrary code generation."
                    "- Keep the human in control of task intent, safety-critical actions, and final approval."

                    "Behavior rules:"

                    "- Start by inspecting relevant current state before proposing or taking action."
                    "- When modifying a task, reason in terms of primitives, transitions, policies, resets, learnable subspaces, and evaluation history."
                    "- Use concise, grounded explanations and clearly separate facts, assumptions, and suggestions."
                    "- Before hardware motion, expensive runs, or important writes, request confirmation."
                    "- When uncertain, say what is missing and propose the smallest useful next step."
                    "- Record important task decisions, run outcomes, and user intent in workspace memory."

                    "Preferred workflow:"

                    "1. Understand the user's goal in the context of the current task and workspace."
                    "2. Inspect the current MP-Net, run history, notes, and available artifacts."
                    "3. Propose a small number of structured next actions."
                    "4. Execute approved tool calls."
                    "5. Summarize what changed, what was learned, and what the next likely step is."

                    "You should help the user manage an iterative human-in-the-loop robot programming loop, not replace them."

                    "Interaction guidelines:"

                    "- Be collaborative, not domineering."
                    "- Do not overwhelm the user with every internal detail."
                    "- Surface the most decision-relevant information first."
                    "- Prefer “Here is the current state; here are two sensible next steps” over long monologues."
                    "- When failure occurs, explain it in terms of the task graph and evidence from runs."
                    "- When the user gives high-level intent, map it into the shared task representation."
                    "- When the user speaks imprecisely, preserve their intent and propose a concrete structured interpretation."

                    "Do not:"

                    "- invent tools or workspace state that do not exist"
                    "- claim a run succeeded if metrics are missing"
                    "- edit arbitrary source files unless explicitly requested and no structured alternative exists"
                    "- trigger robot motion without confirmation"
                    "- make broad task changes when a local change is sufficient"
                    "- hide uncertainty behind confident language"
                    "- treat natural-language discussion as equivalent to applied config changes"
                ),
            },
            {"role": "system", "content": json.dumps(context, indent=2, sort_keys=True)},
        ]
        for event in events:
            if event.role in {"user", "assistant", "tool"}:
                messages.append({"role": event.role if event.role != "tool" else "system", "content": event.content})
        for tool_result in tool_results:
            messages.append(
                {
                    "role": "system",
                    "content": json.dumps(
                        {
                            "tool_name": tool_result.name,
                            "tool_result": {
                                "ok": tool_result.ok,
                                "content": tool_result.content,
                                "data": tool_result.data,
                                "error": tool_result.error,
                            },
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                }
            )
        messages.append({"role": "user", "content": user_message})
        return messages

    def _normalize_tool_calls(self, response: ProviderResponse) -> list[ToolCall]:
        calls = []
        for item in response.tool_calls:
            if not isinstance(item, dict) or "name" not in item:
                continue
            arguments = item.get("arguments", {}) or {}
            if not isinstance(arguments, dict):
                continue
            calls.append(ToolCall(name=str(item["name"]), arguments=arguments))
        return calls

    def handle_user_message(self, user_message: str) -> TurnResult:
        """Handle one natural-language user turn."""
        self._append_event("user", user_message)
        tool_results: list[ToolResult] = []
        latest_message = ""

        for _ in range(self.max_tool_rounds):
            response = self.provider.complete(self._build_messages(user_message, tool_results), self.toolbox.tool_specs_payload())
            latest_message = response.message.strip()
            tool_calls = self._normalize_tool_calls(response)
            self._append_event("assistant", latest_message or "", metadata={"tool_calls": [asdict(call) for call in tool_calls]})
            if not tool_calls:
                return TurnResult(message=latest_message, tool_results=tool_results)

            for call in tool_calls:
                spec = self.toolbox.registry.get_spec(call.name)
                if spec.confirmation is not None:
                    pending = PendingConfirmation(
                        tool_call=call,
                        confirmation=spec.confirmation,
                        assistant_message=latest_message or f"Confirmation required before running '{call.name}'.",
                    )
                    self.pending_confirmation = pending
                    return TurnResult(
                        message=pending.assistant_message,
                        tool_results=tool_results,
                        pending_confirmation=pending,
                    )

                result = self.toolbox.registry.execute(call)
                tool_results.append(result)
                self._append_event("tool", result.to_message(), metadata={"tool_name": result.name})

        if not latest_message:
            latest_message = "I reached the tool-round limit for this turn."
        return TurnResult(message=latest_message, tool_results=tool_results)

    def confirm_pending(self, approved: bool) -> TurnResult:
        """Approve or reject the currently pending tool call."""
        if self.pending_confirmation is None:
            return TurnResult(message="There is no pending action to confirm.")
        pending = self.pending_confirmation
        self.pending_confirmation = None
        if not approved:
            self._append_event("assistant", f"User rejected tool call '{pending.tool_call.name}'.")
            return TurnResult(message=f"Skipped '{pending.tool_call.name}'.")
        result = self.toolbox.registry.execute(pending.tool_call)
        self._append_event("tool", result.to_message(), metadata={"tool_name": result.name, "confirmed": True})
        follow_up = self.provider.complete(
            self._build_messages(
                f"Confirmed and executed tool '{pending.tool_call.name}'. Please summarize the result and next step.",
                [result],
            ),
            self.toolbox.tool_specs_payload(),
        )
        self._append_event("assistant", follow_up.message, metadata={"tool_calls": follow_up.tool_calls})
        return TurnResult(message=follow_up.message, tool_results=[result])
