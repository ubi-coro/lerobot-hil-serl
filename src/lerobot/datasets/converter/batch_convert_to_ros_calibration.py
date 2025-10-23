import argparse
import logging
from pathlib import Path

from datasets import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor.migrate_calibration_processor import invert_calibration
from lerobot.utils.constants import ACTION, OBS_STATE, OBS_IMAGE, DONE, REWARD
from lerobot.utils.utils import init_logging


def convert_dataset(repo_id, base, num_robots, task):
    root = base / repo_id

    original_dataset = LeRobotDataset(repo_id, root=root)

    new_repo_id = str(repo_id) + "_ros_calibration"
    new_root = str(root) + "_ros_calibration"

    features = original_dataset.meta.info["features"]
    for key, ft in features.items():
        if key in (ACTION, OBS_STATE):
           ft["shape"] = (ft["shape"][0] - 2 * num_robots, )
        elif len(ft["shape"]) == 3:
            # swap hwc to chw if we need to
            h, w, c = ft["shape"]
            if c < h and c < w:
                ft["shape"] = (c, h, w)
                ft["names"] = (ft["names"][2], ft["names"][0], ft["names"][1])

    new_dataset = LeRobotDataset.create(
        repo_id=new_repo_id,
        fps=int(original_dataset.fps),
        root=new_root,
        robot_type=original_dataset.meta.robot_type,
        features=features,
        use_videos=len(original_dataset.meta.video_keys) > 0,
    )

    prev_episode_index = 0
    for frame_idx in tqdm(range(len(original_dataset))):
        frame = original_dataset[frame_idx]

        if frame["episode_index"].item() != prev_episode_index:
           # Save the episode
           new_dataset.save_episode()
           prev_episode_index = frame["episode_index"].item()

        # Create a copy of the frame to add to the new dataset
        new_frame = {}
        for key, value in frame.items():
            if key in ("task_index", "timestamp", "episode_index", "frame_index", "index", "task"):
               continue
            if key in (DONE, REWARD):
               # if not isinstance(value, str) and len(value.shape) == 0:
               value = value.unsqueeze(0)
            if key in (ACTION, OBS_STATE):
                value = invert_calibration(value, num_robots=num_robots)
            new_frame[key] = value

        new_frame["task"] = task

        new_dataset.add_frame(new_frame)

    # Save the last episode
    new_dataset.save_episode()


if __name__ == "__main__":
    init_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_id",
        type=str,
        default="",
        help="Repository identifier on Hugging Face: a community or a user name `/` the name of the dataset "
        "(e.g. `lerobot/pusht`, `cadene/aloha_sim_insertion_human`).",
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Process all datasets in root",
    )
    parser.add_argument(
        "--num_robots",
        type=int,
        default=2,
        help="Process all datasets in root",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Unfold the hoodie",
    )
    args = parser.parse_args()

    if args.root is not None:
        root = Path(args.root)
        for sub in sorted(root.iterdir()):
            base = sub.parent.parent
            repo_id = "/".join(sub.parts[-2:])
            logging.info(f"Converting dataset in folder: {base} as repo {repo_id}")
            convert_dataset(repo_id=repo_id, base=base, num_robots=args.num_robots, task=args.task)
