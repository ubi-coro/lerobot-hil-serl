"""Terminal entrypoint for the local MP-Net robot programming workspace."""

from __future__ import annotations

import argparse
import json
from typing import Any

from share.workspace.providers import LLMProvider, OllamaProvider, OpenAICompatibleProvider, ProviderConfig
from share.workspace.runtime import AgentRuntime, PendingConfirmation
from share.workspace.store import WorkspaceStore
from share.workspace.tools import ToolCall, ToolResult, WorkspaceToolbox
from share.workspace.voice import capture_once


class ScriptedProvider(LLMProvider):
    """Deterministic provider used by tests and as a local fallback."""

    def __init__(self) -> None:
        super().__init__(ProviderConfig(provider_type="scripted", model="scripted"))

    def complete(self, messages: list[dict[str, str]], tools: list[dict[str, Any]]):  # type: ignore[override]
        user_text = messages[-1]["content"].strip().lower()
        if "status" in user_text:
            return super()._parse_response_text('{"message":"Checking the current workspace state.","tool_calls":[{"name":"status","arguments":{}}]}')
        if "note" in user_text:
            return super()._parse_response_text('{"message":"I can help store notes with the explicit save_notes tool.","tool_calls":[]}')
        if "run" in user_text:
            return super()._parse_response_text('{"message":"I can launch explicit record, train, or evaluation tools when requested.","tool_calls":[]}')
        return super()._parse_response_text('{"message":"I can inspect the workspace, edit MP-Nets, and launch explicit apps. Try /tools or ask for status.","tool_calls":[]}')


def build_provider(args: argparse.Namespace) -> LLMProvider:
    """Instantiate the selected provider backend."""
    if args.provider == "scripted":
        return ScriptedProvider()
    config = ProviderConfig(
        provider_type=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    if args.provider == "ollama":
        return OllamaProvider(config)
    return OpenAICompatibleProvider(config)


class WorkspaceShell:
    """Small REPL wrapper over the bounded runtime and direct slash commands."""

    def __init__(
        self,
        *,
        store: WorkspaceStore,
        toolbox: WorkspaceToolbox,
        runtime: AgentRuntime,
        voice_enabled: bool = False,
    ):
        self.store = store
        self.toolbox = toolbox
        self.runtime = runtime
        self.voice_enabled = voice_enabled

    def _print_tool_result(self, result: ToolResult) -> None:
        print(result.to_message())

    def _handle_pending_direct(self, tool_call: ToolCall, confirmation: str) -> str:
        self.runtime.pending_confirmation = PendingConfirmation(
            tool_call=tool_call,
            confirmation=confirmation,
            assistant_message=f"Confirmation required for '{tool_call.name}' ({confirmation}). Use /confirm or /reject.",
        )
        return self.runtime.pending_confirmation.assistant_message

    def execute_tool(self, tool_call: ToolCall) -> str:
        spec = self.toolbox.registry.get_spec(tool_call.name)
        if spec.confirmation is not None:
            return self._handle_pending_direct(tool_call, spec.confirmation)
        result = self.toolbox.registry.execute(tool_call)
        self._print_tool_result(result)
        return result.content

    def handle_line(self, line: str) -> str:
        stripped = line.strip()
        if not stripped:
            return ""
        if stripped in {"/exit", "/quit"}:
            raise EOFError
        if stripped == "/help":
            return (
                "Commands: /status, /tools, /runs, /current-net, /tool <name> <json>, "
                "/confirm, /reject, /voice, /exit"
            )
        if stripped == "/status":
            return self.execute_tool(ToolCall(name="status"))
        if stripped == "/runs":
            return self.execute_tool(ToolCall(name="list_runs"))
        if stripped == "/current-net":
            return self.execute_tool(ToolCall(name="summarize_current_mpnet"))
        if stripped == "/tools":
            tool_lines = []
            for spec in self.toolbox.registry.specs():
                suffix = f" [confirm={spec.confirmation}]" if spec.confirmation else ""
                tool_lines.append(f"- {spec.name}: {spec.description}{suffix}")
            return "\n".join(tool_lines)
        if stripped == "/confirm":
            result = self.runtime.confirm_pending(True)
            for tool_result in result.tool_results:
                self._print_tool_result(tool_result)
            return result.message
        if stripped == "/reject":
            result = self.runtime.confirm_pending(False)
            return result.message
        if stripped.startswith("/tool "):
            try:
                _, tool_name, json_args = stripped.split(" ", 2)
                arguments = json.loads(json_args)
            except ValueError:
                return "Usage: /tool <name> <json-arguments>"
            except json.JSONDecodeError as exc:
                return f"Invalid JSON arguments: {exc}"
            return self.execute_tool(ToolCall(name=tool_name, arguments=arguments))
        if stripped == "/voice":
            voice = capture_once()
            if voice.text:
                return self.handle_line(voice.text)
            return voice.message or "No speech was captured."

        result = self.runtime.handle_user_message(stripped)
        for tool_result in result.tool_results:
            self._print_tool_result(tool_result)
        if result.pending_confirmation is not None:
            return (
                f"{result.message}\n"
                f"Pending confirmation: {result.pending_confirmation.tool_call.name} "
                f"({result.pending_confirmation.confirmation}). Use /confirm or /reject."
            )
        return result.message

    def prompt_once(self) -> str:
        if self.voice_enabled:
            raw = input("voice/text> ").strip()
            if raw:
                return raw
            voice = capture_once()
            if voice.text:
                return voice.text
            print(voice.message or "No speech was captured, falling back to text.")
        return input("workspace> ")

    def loop(self) -> None:
        while True:
            try:
                line = self.prompt_once()
                output = self.handle_line(line)
                if output:
                    print(output)
            except EOFError:
                print("Exiting workspace shell.")
                break


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the workspace shell."""
    parser = argparse.ArgumentParser(description="Local MP-Net robotics workspace shell.")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--project", default="default")
    parser.add_argument("--task", default="default")
    parser.add_argument("--provider", choices=["ollama", "openai", "scripted"], default="scripted")
    parser.add_argument("--model", default="qwen2.5:7b-instruct")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--voice", action="store_true")
    parser.add_argument("--command", help="Run one command or one natural-language turn non-interactively.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Start the workspace CLI or execute one command non-interactively."""
    args = parse_args(argv)
    store = WorkspaceStore(args.workspace)
    store.ensure_workspace(project=args.project, task=args.task)
    store.set_active_task(args.project, args.task)
    toolbox = WorkspaceToolbox(store=store, project=args.project, task=args.task)
    runtime = AgentRuntime(
        store=store,
        toolbox=toolbox,
        provider=build_provider(args),
        project=args.project,
        task=args.task,
    )
    shell = WorkspaceShell(store=store, toolbox=toolbox, runtime=runtime, voice_enabled=args.voice)
    if args.command:
        output = shell.handle_line(args.command)
        if output:
            print(output)
        return 0
    shell.loop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
