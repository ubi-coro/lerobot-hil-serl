from __future__ import annotations

from share.scripts.robot_workspace import ScriptedProvider, WorkspaceShell, main
from share.workspace.runtime import AgentRuntime
from share.workspace.store import WorkspaceStore
from share.workspace.tools import WorkspaceToolbox


def test_workspace_shell_smoke_flow(tmp_path, capsys):
    workspace_root = tmp_path / "workspace"
    store = WorkspaceStore(workspace_root)
    store.ensure_workspace(project="proj", task="task")
    toolbox = WorkspaceToolbox(store=store, project="proj", task="task")
    runtime = AgentRuntime(store=store, toolbox=toolbox, provider=ScriptedProvider(), project="proj", task="task")
    shell = WorkspaceShell(store=store, toolbox=toolbox, runtime=runtime)

    create_message = shell.handle_line('/tool create_mpnet {"name":"demo"}')
    assert "Confirmation required" in create_message
    confirm_message = shell.handle_line("/confirm")
    assert "Created MP-Net 'demo'" in capsys.readouterr().out
    assert "I can inspect" in shell.handle_line("hello")

    register_message = shell.handle_line('/tool register_run {"run_type":"evaluation","status":"succeeded","metrics":{"success_rate":0.7}}')
    assert "Confirmation required" in register_message
    shell.handle_line("/confirm")
    assert "Registered external run" in capsys.readouterr().out
    status_message = shell.handle_line("/status")
    assert "Current task status" in status_message


def test_cli_main_command_mode(tmp_path, capsys):
    exit_code = main(
        [
            "--workspace",
            str(tmp_path / "workspace"),
            "--project",
            "proj",
            "--task",
            "task",
            "--provider",
            "scripted",
            "--command",
            "/status",
        ]
    )

    assert exit_code == 0
    assert "Current task status" in capsys.readouterr().out
