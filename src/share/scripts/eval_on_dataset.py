"""Lightweight offline evaluation summary adapter for workspace runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from share.workspace.store import dump_json, load_json


def _primitive_dataset_dirs(dataset_root: Path) -> list[tuple[str, Path]]:
    if (dataset_root / "meta" / "info.json").exists():
        return [(dataset_root.name, dataset_root)]
    items = []
    for child in sorted(dataset_root.iterdir()):
        if child.is_dir() and (child / "meta" / "info.json").exists():
            items.append((child.name, child))
    return items


def _read_dataset_info(dataset_dir: Path) -> dict[str, Any]:
    info = load_json(dataset_dir / "meta" / "info.json", default={}) or {}
    stats = load_json(dataset_dir / "meta" / "stats.json", default={}) or {}
    episode_count = info.get("total_episodes", info.get("num_episodes"))
    frame_count = info.get("total_frames", info.get("num_frames"))
    return {
        "dataset_dir": str(dataset_dir),
        "repo_id": info.get("repo_id"),
        "episode_count": episode_count,
        "frame_count": frame_count,
        "codebase_version": info.get("codebase_version"),
        "stats_keys": sorted(stats.keys()) if isinstance(stats, dict) else [],
    }


def summarize_dataset_eval(
    dataset_root: str | Path,
    policy_path: str,
    *,
    metrics_file: str | None = None,
) -> dict[str, Any]:
    """Summarize available dataset-level evaluation metadata."""
    dataset_root = Path(dataset_root).expanduser().resolve()
    primitive_entries = []
    for primitive_name, primitive_dir in _primitive_dataset_dirs(dataset_root):
        summary = _read_dataset_info(primitive_dir)
        summary["primitive"] = primitive_name
        primitive_entries.append(summary)

    extra_metrics = load_json(Path(metrics_file), default={}) if metrics_file else {}
    if not isinstance(extra_metrics, dict):
        extra_metrics = {}

    overall = {
        "primitive_count": len(primitive_entries),
        "episode_count": sum(entry["episode_count"] or 0 for entry in primitive_entries),
        "frame_count": sum(entry["frame_count"] or 0 for entry in primitive_entries),
    }
    if "overall" in extra_metrics and isinstance(extra_metrics["overall"], dict):
        overall.update(extra_metrics["overall"])

    merged_primitives = []
    by_name = extra_metrics.get("primitives", {}) if isinstance(extra_metrics.get("primitives"), dict) else {}
    for entry in primitive_entries:
        merged = dict(entry)
        if entry["primitive"] in by_name and isinstance(by_name[entry["primitive"]], dict):
            merged.update(by_name[entry["primitive"]])
        merged_primitives.append(merged)

    return {
        "policy_path": str(Path(policy_path).expanduser().resolve()),
        "dataset_root": str(dataset_root),
        "overall": overall,
        "primitives": merged_primitives,
        "metrics": overall,
        "output_paths": [str(dataset_root)],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse adapter CLI arguments."""
    parser = argparse.ArgumentParser(description="Summarize dataset-level eval information for workspace runs.")
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metrics-file")
    parser.add_argument("--summary-path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Compute and persist dataset evaluation summaries."""
    args = parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_dataset_eval(args.dataset_root, args.policy_path, metrics_file=args.metrics_file)
    dump_json(output_dir / "eval_info.json", summary)
    dump_json(output_dir / "summary.json", summary)
    if args.summary_path:
        dump_json(Path(args.summary_path), summary)
    return summary


if __name__ == "__main__":
    main()
