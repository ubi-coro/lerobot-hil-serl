"""Thin share-level training adapter for workspace-managed workflows."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lerobot.configs.policies import PreTrainedConfig
from share.workspace.store import dump_json

try:
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig

    TRAINING_BACKEND_AVAILABLE = True
    TRAINING_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    TRAINING_BACKEND_AVAILABLE = False
    TRAINING_IMPORT_ERROR = exc

    @dataclass
    class DatasetConfig:
        repo_id: str
        root: str | None = None

    @dataclass
    class TrainPipelineConfig:
        dataset: DatasetConfig
        env: Any | None = None
        policy: Any | None = None
        output_dir: Path | None = None
        job_name: str | None = None
        steps: int = 1000
        batch_size: int = 8
        eval_freq: int = 0
        log_freq: int = 200
        save_freq: int = 1000
        save_checkpoint: bool = True


def build_train_config(args: argparse.Namespace) -> TrainPipelineConfig:
    """Build a `TrainPipelineConfig` from simple CLI arguments."""
    policy = PreTrainedConfig.from_pretrained(args.policy_path, local_files_only=args.local_files_only)
    policy.pretrained_path = Path(args.policy_path)
    if args.policy_device:
        policy.device = args.policy_device

    dataset = DatasetConfig(repo_id=args.dataset_repo_id, root=args.dataset_root)
    cfg = TrainPipelineConfig(
        dataset=dataset,
        env=None,
        policy=policy,
        output_dir=Path(args.output_dir),
        job_name=args.job_name,
        steps=args.steps,
        batch_size=args.batch_size,
        eval_freq=args.eval_freq,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        save_checkpoint=not args.no_save_checkpoint,
    )
    return cfg


def summarize_training_output(output_dir: Path) -> dict[str, Any]:
    """Collect a small summary of the produced training artifacts."""
    checkpoint_dirs = sorted((output_dir / "checkpoints").glob("*")) if (output_dir / "checkpoints").exists() else []
    latest_checkpoint = checkpoint_dirs[-1] if checkpoint_dirs else None
    pretrained_model_dir = latest_checkpoint / "pretrained_model" if latest_checkpoint is not None else None
    summary = {
        "output_dir": str(output_dir),
        "checkpoint_count": len(checkpoint_dirs),
        "latest_checkpoint": str(latest_checkpoint) if latest_checkpoint is not None else None,
        "latest_pretrained_model": str(pretrained_model_dir) if pretrained_model_dir is not None and pretrained_model_dir.exists() else None,
        "output_paths": [str(output_dir)],
        "metrics": {
            "checkpoint_count": len(checkpoint_dirs),
        },
    }
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse adapter CLI arguments."""
    parser = argparse.ArgumentParser(description="Thin workspace training adapter for LeRobot policies.")
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--job-name")
    parser.add_argument("--policy-device")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-freq", type=int, default=0)
    parser.add_argument("--log-freq", type=int, default=200)
    parser.add_argument("--save-freq", type=int, default=1000)
    parser.add_argument("--summary-path")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-save-checkpoint", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Run the adapter and optionally write a summary file."""
    args = parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    cfg = build_train_config(args)
    if not TRAINING_BACKEND_AVAILABLE:
        raise RuntimeError(
            "The lightweight workspace environment can build the training request, "
            "but launching training requires the full LeRobot runtime dependencies. "
            f"Original import error: {TRAINING_IMPORT_ERROR}"
        )
    from lerobot.scripts.lerobot_train import train as lerobot_train

    lerobot_train(cfg)
    summary = summarize_training_output(output_dir)
    if args.summary_path:
        dump_json(Path(args.summary_path), summary)
    return summary


if __name__ == "__main__":
    main()
