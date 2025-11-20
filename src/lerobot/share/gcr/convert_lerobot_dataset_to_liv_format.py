#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
import torch
import imageio.v2 as imageio

from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def tensor_to_uint8_img(t: torch.Tensor) -> np.ndarray:
    """
    Convert torch Tensor (C, H, W) → uint8 numpy image (H, W, C).
    """
    if t.dtype == torch.uint8:
        img = t
    else:
        img = t.clone()
        if img.max() <= 1.1:
            img = img * 255.0
        img = img.clamp(0, 255).to(torch.uint8)

    return img.permute(1, 2, 0).cpu().numpy()


def export_episodes_and_manifest(
    root: str,
    repo_id: str,
    out_dir: str,
    camera_key: str | None = None,
    video_backend: str = "pyav",
    max_episodes: int | None = None,
):
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading dataset from: root='{root}', repo_id='{repo_id}'")
    dataset = LeRobotDataset(
        root=root,
        repo_id=repo_id,
        video_backend=video_backend,
    )

    if camera_key is None:
        if len(dataset.meta.camera_keys) == 0:
            raise ValueError("No camera keys in dataset!")
        camera_key = dataset.meta.camera_keys[0]
        print(f"camera_key not specified — using '{camera_key}'")
    else:
        print(f"Using camera_key='{camera_key}'")

    episodes_meta = dataset.meta.episodes
    total_episodes = dataset.num_episodes

    if max_episodes is not None:
        total_episodes = min(total_episodes, max_episodes)

    print(f"Exporting {total_episodes} episodes → {out_dir}")

    manifest = {
        "index": [],
        "directory": [],
        "num_frames": [],
        "text": [],
    }

    for ep_idx in tqdm(range(total_episodes), desc="Episodes"):
        ep = episodes_meta[ep_idx]

        start_idx = int(ep["dataset_from_index"])
        end_idx = int(ep["dataset_to_index"])  # exclusive

        ep_task = ep.get("task", "")
        # If metadata does not include task, fallback to dataset[start]["task"]
        if ep_task in ("", None):
            ep_task = dataset[start_idx].get("task", "")

        ep_dir = os.path.join(out_dir, f"episode_{ep_idx:04d}")
        os.makedirs(ep_dir, exist_ok=True)

        num_frames = end_idx - start_idx

        # Write all frames as PNGs
        for local_i, global_i in enumerate(range(start_idx, end_idx)):
            frame = dataset[global_i]
            img_t = frame[camera_key]  # CxHxW
            img = tensor_to_uint8_img(img_t)
            img_path = os.path.join(ep_dir, f"{local_i:06d}.png")
            imageio.imwrite(img_path, img)

        # Add to manifest
        manifest["index"].append(ep_idx)
        manifest["directory"].append(ep_dir)
        manifest["num_frames"].append(num_frames)
        manifest["text"].append(ep_task)

    # Save manifest
    manifest_df = pd.DataFrame(manifest)
    manifest_path = os.path.join(out_dir, "manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)

    print(f"\nDone.")
    print(f"Saved manifest: {manifest_path}")
    print(f"Extracted episodes to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Export each episode in a LeRobotDataset as image frames and generate LIV-style manifest.csv"
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--camera-key", default=None)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--max-episodes", type=int, default=None)

    args = parser.parse_args()

    export_episodes_and_manifest(
        root=args.root,
        repo_id=args.repo_id,
        out_dir=args.out_dir,
        camera_key=args.camera_key,
        video_backend=args.video_backend,
        max_episodes=args.max_episodes,
    )


if __name__ == "__main__":
    main()
