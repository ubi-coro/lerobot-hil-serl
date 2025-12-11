#!/usr/bin/env python3
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION

# ----------------------------------------------------------------------
# Config – keep in sync with the teleop rollout script
# ----------------------------------------------------------------------
root = "/mnt/nvme0n1p3/data/polytec/teleop_long/"
repo_id = "test/test"

# None => all episodes, or e.g. [0, 1, 2]
requested_episodes = None
# requested_episodes = [0, 1]

# ACTION is 5D:
# 0: x velocity
# 1: y velocity
# 2: z velocity
# 3: c angular velocity
# 4: gripper state (ignored for histograms)

# ----------------------------------------------------------------------
# Build cache filename (stored next to this script)
# ----------------------------------------------------------------------
if requested_episodes is None:
    ep_spec = "all"
else:
    ep_spec = "ep_" + "_".join(str(e) for e in requested_episodes)

cache_filename = f"teleop_cache_actions_{repo_id.replace('/', '_')}_{ep_spec}.pkl"

# script directory (fallback to CWD if __file__ not defined)
script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
cache_path = os.path.join(script_dir, cache_filename)

print(f"Using action cache file: {cache_path}")

# ----------------------------------------------------------------------
# Load from cache or compute from dataset
# ----------------------------------------------------------------------
if os.path.exists(cache_path):
    print("Loading actions from cache...")
    with open(cache_path, "rb") as f:
        actions_dict = pickle.load(f)
else:
    print("Cache not found, iterating dataset and collecting actions...")

    dataset = LeRobotDataset(
        root=root,
        repo_id=repo_id,
        episodes=requested_episodes,
    )

    # Flat lists over all frames
    ax_list = []  # x vel
    ay_list = []  # y vel
    az_list = []  # z vel
    ac_list = []  # c angular vel

    for frame in tqdm(dataset):
        act = frame[ACTION]

        # Make sure we convert to floats (in case these are tensors)
        ax_list.append(float(act[0]))
        ay_list.append(float(act[1]))
        az_list.append(float(act[2]))
        ac_list.append(float(act[3]))

    actions_dict = {
        "ax": np.array(ax_list),
        "ay": np.array(ay_list),
        "az": np.array(az_list),
        "ac": np.array(ac_list),
    }

    # Save to cache
    print("Saving actions to cache...")
    with open(cache_path, "wb") as f:
        pickle.dump(actions_dict, f)

# ----------------------------------------------------------------------
# Histogram visualization
# ----------------------------------------------------------------------
ax_vals = actions_dict["ax"]
ay_vals = actions_dict["ay"]
az_vals = actions_dict["az"]
ac_vals = actions_dict["ac"]

print(f"Total action samples: {len(ax_vals)}")

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.ravel()

data = [ax_vals, ay_vals, az_vals, ac_vals]
titles = [
    "x velocity",
    "y velocity",
    "z velocity",
    "c angular velocity",
]

for ax, arr, title in zip(axes, data, titles):
    ax.hist(arr, bins=80, density=True, alpha=0.8)
    ax.set_title(title)
    ax.set_ylabel("density")
    ax.grid(True, linestyle="--", alpha=0.4)

# Optional: add x-labels only on bottom row
axes[2].set_xlabel("value")
axes[3].set_xlabel("value")

fig.suptitle("Teleop Action Distributions", fontsize=14)
plt.tight_layout()
plt.show()
