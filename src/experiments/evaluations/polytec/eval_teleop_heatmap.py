#!/usr/bin/env python3
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import OBS_STATE

# ----------------------------------------------------------------------
# Matplotlib / LaTeX setup
# ----------------------------------------------------------------------
plt.rcParams["text.usetex"] = True      # comment out if LaTeX not installed
plt.rcParams["font.family"] = "serif"

root = "/mnt/nvme0n1p3/data/polytec/teleop_long/"
repo_id = "test/test"

# None => all episodes, or e.g. [0, 1, 2]
requested_episodes = None
# requested_episodes = [0, 1]

# Image overlay parameters (tunable)
IMAGE_SCALE    = 1.45          # 1.0 = auto-fit, >1 = bigger, <1 = smaller
IMAGE_X_OFFSET = 0.518         # centre of the image in x (m)
IMAGE_Y_OFFSET = -0.10         # centre of the image in y (m)
IMAGE_ROT_DEG  = 0.0           # rotation in degrees, CCW
IMAGE_ALPHA    = 0.5           # transparency of overlay image

# Heatmap parameters
HEATMAP_BINS   = 150           # number of bins in x and y
HEATMAP_ALPHA  = 0.8           # transparency of state-occupancy heatmap
HEATMAP_CMAP   = "magma"       # colormap for occupancy

# ----------------------------------------------------------------------
# Cache + script paths
# ----------------------------------------------------------------------
if requested_episodes is None:
    ep_spec = "all"
else:
    ep_spec = "ep_" + "_".join(str(e) for e in requested_episodes)

cache_filename = f"teleop_cache_{repo_id.replace('/', '_')}_{ep_spec}.pkl"

script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
cache_path = os.path.join(script_dir, cache_filename)
layover_path = os.path.join(script_dir, "layover.jpeg")

print(f"Using cache file:    {cache_path}")
print(f"Using overlay file:  {layover_path}")

# ----------------------------------------------------------------------
# Load from cache or compute from dataset
# ----------------------------------------------------------------------
if os.path.exists(cache_path):
    print("Loading episodes from cache...")
    with open(cache_path, "rb") as f:
        episodes = pickle.load(f)
else:
    print("Cache not found, iterating dataset and building episodes...")

    dataset = LeRobotDataset(
        root=root,
        repo_id=repo_id,
        episodes=requested_episodes,
    )

    episodes = {}
    prev_gripper_state = {}

    for frame in tqdm(dataset):
        ep_idx = int(frame["episode_index"])

        if ep_idx not in episodes:
            episodes[ep_idx] = {
                "x": [],
                "y": [],
                "gripper_changed": [],
            }

        obs = frame[OBS_STATE]

        gripper_state = float(obs[-1])
        changed = False
        if ep_idx in prev_gripper_state:
            changed = (gripper_state != prev_gripper_state[ep_idx])
        prev_gripper_state[ep_idx] = gripper_state

        episodes[ep_idx]["x"].append(float(obs[0]))
        episodes[ep_idx]["y"].append(float(obs[3]))
        episodes[ep_idx]["gripper_changed"].append(changed)

    print("Saving episodes to cache...")
    with open(cache_path, "wb") as f:
        pickle.dump(episodes, f)

# ----------------------------------------------------------------------
# Global bounds from data
# ----------------------------------------------------------------------
all_x = np.concatenate([np.array(v["x"]) for v in episodes.values()])
all_y = np.concatenate([np.array(v["y"]) for v in episodes.values()])

x_min, x_max = all_x.min(), all_x.max()
y_min, y_max = all_y.min(), all_y.max()

x_range = x_max - x_min
y_range = y_max - y_min
max_range = max(x_range, y_range)

x_mid = 0.5 * (x_max + x_min)
y_mid = 0.5 * (y_max + y_min)

plot_xlim = (x_mid - max_range / 2, x_mid + max_range / 2)
plot_ylim = (y_mid - max_range / 2, y_mid + max_range / 2)

# ----------------------------------------------------------------------
# Load, crop, flip, rotate, make background transparent and compute extent
# ----------------------------------------------------------------------
overlay_img = None
overlay_extent = None

if os.path.exists(layover_path):
    img = Image.open(layover_path).convert("RGBA")
    w, h = img.size

    # symmetric crop (keeps aspect)
    border_frac = 0.03
    left   = int(border_frac * w)
    right  = int((1.0 - border_frac) * w)
    top    = int(border_frac * h)
    bottom = int((1.0 - border_frac) * h)
    img = img.crop((left, top, right, bottom))

    # flip horizontally to match your alignment
    img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # rotate around image centre
    if abs(IMAGE_ROT_DEG) > 1e-6:
        img = img.rotate(IMAGE_ROT_DEG, resample=Image.BICUBIC, expand=True)

    # update size after crop/flip/rotate
    w_c, h_c = img.size

    # make (almost) white background transparent AFTER rotation
    arr = np.array(img)
    r, g, b, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
    white_mask = (r > 250) & (g > 250) & (b > 250)
    a[white_mask] = 0
    arr = np.stack([r, g, b, a], axis=-1)
    overlay_img = arr

    # isotropic pixel->meter mapping, keep aspect ratio
    base_pixel_to_meter = max_range / max(w_c, h_c)
    pixel_to_meter = base_pixel_to_meter * IMAGE_SCALE

    width_phys  = pixel_to_meter * w_c
    height_phys = pixel_to_meter * h_c

    overlay_extent = (
        IMAGE_X_OFFSET - width_phys / 2.0,
        IMAGE_X_OFFSET + width_phys / 2.0,
        IMAGE_Y_OFFSET - height_phys / 2.0,
        IMAGE_Y_OFFSET + height_phys / 2.0,
    )
else:
    print("WARNING: layover.jpeg not found, skipping overlay.")

# ----------------------------------------------------------------------
# Compute state-occupancy heatmap (2D histogram over x,y)
# ----------------------------------------------------------------------
# Use the same overall region as the trajectories
heatmap, x_edges, y_edges = np.histogram2d(
    all_x,
    all_y,
    bins=HEATMAP_BINS,
    range=[plot_xlim, plot_ylim],
)

# Log-transform for visibility (compress high-count regions)
heatmap_log = np.log1p(heatmap)  # log(1 + count)

# ----------------------------------------------------------------------
# Visualization: overlay + heatmap
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 10))

# draw overlay image first (under heatmap)
if overlay_img is not None and overlay_extent is not None:
    ax.imshow(
        overlay_img,
        extent=overlay_extent,
        origin="lower",
        alpha=IMAGE_ALPHA,
        zorder=0,
    )

# draw heatmap (transpose because histogram2d returns [x_bins, y_bins])
im = ax.imshow(
    heatmap_log.T,
    origin="lower",
    extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
    cmap=HEATMAP_CMAP,
    alpha=HEATMAP_ALPHA,
    zorder=1,
    interpolation="bilinear",
    aspect="equal",
)

# colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r"State occupancy (log-scaled)", rotation=90)

ax.set_xlabel(r"$x~[\mathrm{m}]$")
ax.set_ylabel(r"$y~[\mathrm{m}]$")
ax.set_title(r"\textbf{Teleop State Occupancy} ($x$–$y$)")

ax.grid(False)  # heatmap is clearer without grid
ax.set_aspect("equal", adjustable="box")

ax.set_xlim(*plot_xlim)
ax.set_ylim(*plot_ylim)

ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(6))

plt.tight_layout()
plt.show()
