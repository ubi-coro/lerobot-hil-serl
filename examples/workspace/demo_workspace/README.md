# Demo Workspace

This is a minimal example workspace for the MP-Net robot programming shell.

Try it with:

```bash
lerobot-workspace \
  --workspace examples/workspace/demo_workspace \
  --project pick_and_place \
  --task block_pick \
  --provider scripted
```

Suggested flow:

```text
/status
/current-net
/tool edit_mpnet {"name":"pick_block_v1","operation":"set_axis_targets","arguments":{"primitive_name":"main","targets":{"x":0.45,"z":0.16}}}
/confirm
/tool register_run {"run_type":"evaluation","status":"succeeded","metrics":{"success_rate":0.7,"trials":10}}
/confirm
what should we improve next?
```
