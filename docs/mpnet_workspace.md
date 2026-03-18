# MP-Net Workspace

`lerobot-workspace` adds a local-first robotics workflow shell on top of the existing MP-Net stack in `src/share`.

## Architecture

- `share.workspace.store`: file-backed workspace metadata, MP-Net versions, run logs, artifacts, notes, and agent transcripts.
- `share.workspace.mpnet`: JSON load/save helpers, graph summaries, validation, and structured MP-Net edit operations.
- `share.workspace.tools`: explicit read/edit/run tools plus a subprocess runner that captures logs and summaries.
- `share.workspace.providers`: a minimal provider interface with `OllamaProvider` and `OpenAICompatibleProvider`.
- `share.workspace.runtime`: the bounded agent loop with tool rounds, confirmation gating, and persistent session history.

## Workspace Layout

The workspace is rooted in a user-specified directory.

```text
workspace.json
projects/<project>/project.json
projects/<project>/tasks/<task>/task.json
projects/<project>/tasks/<task>/notes.md
projects/<project>/tasks/<task>/mp_nets/<net>/current.json
projects/<project>/tasks/<task>/mp_nets/<net>/versions/<version>.json
projects/<project>/tasks/<task>/runs/<run>/run.json
projects/<project>/tasks/<task>/runs/<run>/stdout.log
projects/<project>/tasks/<task>/runs/<run>/stderr.log
projects/<project>/tasks/<task>/runs/<run>/summary.json
projects/<project>/tasks/<task>/artifacts/<artifact>/artifact.json
projects/<project>/tasks/<task>/agent/sessions/<session>.jsonl
projects/<project>/tasks/<task>/agent/memory.md
```

Workspace-managed MP-Net copies are the source of truth. Imported configs are copied in and their origin paths are tracked in task metadata.

## Tools And Apps

Read-only tools:

- `status`
- `list_mp_nets`
- `summarize_current_mpnet`
- `list_primitives`
- `describe_transitions`
- `read_notes`
- `list_runs`
- `summarize_latest_evals`
- `compare_runs`
- `validate_mpnet`

Editing tools:

- `create_mpnet`
- `import_mpnet`
- `edit_mpnet`
- `save_notes`
- `pin_artifact`
- `register_run`

Execution tools:

- `run_demo_mpnet`
- `run_recording`
- `run_training`
- `run_evaluation`

Safety model:

- `config_write`: confirmation required before writing task notes, artifacts, or MP-Net versions.
- `expensive_run`: confirmation required before training or evaluation jobs.
- `hardware_motion`: confirmation required before robot-moving operations like recording.

## MP-Net Editing Model

The workspace agent edits structured `ManipulationPrimitiveNetConfig` JSON rather than arbitrary Python.

Supported operations include:

- create/import an MP-Net
- list primitives and transitions
- add/remove primitives
- set start/reset primitives
- mark primitives terminal or non-terminal
- attach a pretrained policy to a primitive
- store per-primitive notes
- set learnable axes by editing `task_frame.policy_mode`
- set axis targets
- add/remove transitions

Every successful edit writes a new version under `versions/` and then updates `current.json`.

## Running

```bash
lerobot-workspace \
  --workspace examples/workspace/demo_workspace \
  --project pick_and_place \
  --task block_pick \
  --provider scripted
```

Useful shell commands:

- `/status`
- `/tools`
- `/current-net`
- `/runs`
- `/tool <name> <json>`
- `/confirm`
- `/reject`

Optional voice mode:

```bash
lerobot-workspace --workspace <dir> --voice
```

If `speech_recognition` is unavailable, the shell falls back to text input without breaking the session.

## Example Interaction

```text
/status
/tool create_mpnet {"name":"pick_block","primitive_name":"main"}
/confirm
/tool edit_mpnet {"name":"pick_block","operation":"set_learnable_axes","arguments":{"primitive_name":"main","axes":{"x":"relative","y":"relative","z":"relative"}}}
/confirm
/tool run_recording {"name":"pick_block","dataset_repo_id":"local/pick-block","dataset_root":"data/pick-block","single_task":"Pick the block from the tray","num_episodes":5}
/confirm
/tool run_training {"policy_path":"outputs/bootstrap_policy","dataset_repo_id":"local/pick-block","dataset_root":"data/pick-block","output_dir":"outputs/train/pick-block"}
/confirm
/tool run_evaluation {"policy_path":"outputs/train/pick-block/checkpoints/001000/pretrained_model","dataset_root":"data/pick-block","output_dir":"outputs/eval/pick-block"}
/confirm
what should we improve next?
```
