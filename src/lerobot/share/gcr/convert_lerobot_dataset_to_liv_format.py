#!/usr/bin/env python

import os
import argparse

import numpy as np
import torch
import imageio.v2 as imageio

from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def tensor_to_uint8_img(t: torch.Tensor) -> np.ndarray:
    """
    Convert a torch tensor (C,H,W) to uint8 numpy image (H,W,C).
    Handles float [0,1] or uint8.
    """
    if t.dtype == torch.uint8:
        img = t
    else:
        # assume float in [0,1] or [0,255]
        img = t.clone()
        if img.max() <= 1.5:
            img = img * 255.0
        img = img.clamp(0, 255).to(torch.uint8)

    img = img.permute(1, 2, 0).cpu().numpy()  # (H,W,C)
    return img


def export_episodes_to_videos(
    root: str,
    repo_id: str,
    out_dir: str,
    camera_key: str | None = None,
    video_backend: str = "pyav",
    max_episodes: int | None = None,
):
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading LeRobotDataset from root='{root}', repo_id='{repo_id}'...")
    dataset = LeRobotDataset(
        root=root,
        repo_id=repo_id,
        video_backend=video_backend,
    )

    # Choose camera key if not provided
    if camera_key is None:
        if len(dataset.meta.camera_keys) == 0:
            raise ValueError("No camera_keys found in dataset.meta.camera_keys.")
        camera_key = dataset.meta.camera_keys[0]
        print(f"No camera_key provided, using first camera: '{camera_key}'")
    else:
        if camera_key not in dataset.meta.camera_keys:
            raise ValueError(
                f"camera_key='{camera_key}' not in dataset.meta.camera_keys={dataset.meta.camera_keys}"
            )
        print(f"Using camera_key='{camera_key}'")

    fps = dataset.fps
    num_episodes = dataset.num_episodes
    if max_episodes is not None:
        num_episodes = min(num_episodes, max_episodes)

    print(f"Found {dataset.num_episodes} episodes, exporting {num_episodes} of them.")
    print(f"Writing videos to '{out_dir}' at {fps} fps.")

    # Use metadata to get frame index ranges per episode
    episodes_meta = dataset.meta.episodes  # structured array / dataframe-like

    for ep_idx in tqdm(range(num_episodes), desc="Episodes"):
        ep = episodes_meta[ep_idx]
        start_idx = int(ep["dataset_from_index"])
        end_idx = int(ep["dataset_to_index"])  # exclusive

        out_path = os.path.join(out_dir, f"episode_{ep_idx:04d}.mp4")
        if os.path.exists(out_path):
            # Skip if already written
            continue

        # Collect frames
        frames = []
        for global_idx in range(start_idx, end_idx):
            frame = dataset[global_idx]
            img_t = frame[camera_key]  # C x H x W
            frames.append(tensor_to_uint8_img(img_t))

        if len(frames) == 0:
            print(f"Episode {ep_idx} has no frames, skipping.")
            continue

        # Write video
        with imageio.get_writer(out_path, fps=fps) as writer:
            for f in frames:
                writer.append_data(f)

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Export each episode from a LeRobotDataset as a single video "
                    "for a specified camera key."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root folder where the LeRobot dataset lives.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Dataset repo_id (as used when creating/loading LeRobotDataset).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory where per-episode videos will be written.",
    )
    parser.add_argument(
        "--camera-key",
        type=str,
        default=None,
        help="Camera key to export (defaults to first in dataset.meta.camera_keys).",
    )
    parser.add_argument(
        "--video-backend",
        type=str,
        default="pyav",
        help="Video backend for LeRobotDataset (default: pyav).",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional: limit the number of episodes to export.",
    )

    args = parser.parse_args()

    export_episodes_to_videos(
        root=args.root,
        repo_id=args.repo_id,
        out_dir=args.out_dir,
        camera_key=args.camera_key,
        video_backend=args.video_backend,
        max_episodes=args.max_episodes,
    )


if __name__ == "__main__":
    main()
