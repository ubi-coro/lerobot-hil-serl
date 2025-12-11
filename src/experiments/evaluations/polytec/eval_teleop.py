import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
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

SHOW_GRIPPER_EVENTS = True
SHOW_COLORED_TRAJ = True

# Image overlay parameters (tunable)
IMAGE_SCALE = 1.45          # 1.0 = auto-fit, >1 = bigger, <1 = smaller
IMAGE_X_OFFSET = 0.518      # centre of the image in x (in meters)
IMAGE_Y_OFFSET = -0.1       # centre of the image in y (in meters)
IMAGE_ROT_DEG  = 0.0        # rotation in degrees, CCW
IMAGE_ALPHA    = 0.5        # transparency of overlay

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
# Load dataset (for fps) and episodes (from cache or fresh)
# ----------------------------------------------------------------------
dataset = LeRobotDataset(
    root=root,
    repo_id=repo_id,
    episodes=requested_episodes,
)

fps = dataset.fps  # frames per second

if os.path.exists(cache_path):
    print("Loading episodes from cache...")
    with open(cache_path, "rb") as f:
        episodes = pickle.load(f)
else:
    print("Cache not found, iterating dataset and building episodes...")

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

plot_xlim = (x_mid - max_range / 2, x_mid + max_range / 2 - 0.03)
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

    # flip to align with your setup
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
# Setup color normalization in *seconds*
# ----------------------------------------------------------------------
# Find longest trajectory in steps and seconds
max_len = max(len(data["x"]) for data in episodes.values() if len(data["x"]) > 0)
if max_len < 2:
    raise ValueError("Not enough timesteps to build trajectories.")

T_max = (max_len - 1) / fps  # longest duration in seconds
cmap = plt.get_cmap("viridis")
norm = Normalize(vmin=0.0, vmax=T_max)

# ----------------------------------------------------------------------
# Visualization (2D, time-colored + overlay + colorbar)
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 10))

if overlay_img is not None and overlay_extent is not None:
    ax.imshow(
        overlay_img,
        extent=overlay_extent,
        origin="lower",
        alpha=IMAGE_ALPHA,
        zorder=0,
    )

for i, (ep_idx, data) in enumerate(sorted(episodes.items())):
    x = np.array(data["x"])
    y = np.array(data["y"])
    gripper_changed = np.array(data.get("gripper_changed", [False] * len(x)), dtype=bool)

    if len(x) < 2:
        continue

    # absolute time for each frame in this episode
    indices = np.arange(len(x))
    t_sec = indices / fps  # time in seconds from episode start

    points = np.stack([x, y], axis=1)
    segments = np.stack([points[:-1], points[1:]], axis=1)

    if SHOW_COLORED_TRAJ:
        # colors according to seconds, normalized to [0, T_max]
        colors = cmap(norm(t_sec[:-1]))
        start_color = colors[0]
        end_color = "red"
        ls = "-"
        lw = 3.0
    else:
        colors = "k"
        start_color = end_color = "k"
        ls = "--"
        lw = 1.2

    lc = LineCollection(
        segments,
        colors=colors,
        linewidths=lw,
        alpha=0.9,
        zorder=5,
        linestyles=ls,
    )
    ax.add_collection(lc)

    ax.scatter(
        x[0], y[0],
        marker="o",
        s=40,
        label=r"Episode start" if i == 0 else None,
        color=start_color,
        zorder=10,
    )

    ax.scatter(
        x[-1], y[-1],
        marker="x",
        s=60,
        color=end_color,
        zorder=11,
        label=r"Episode end" if i == 0 else None,
    )

    if gripper_changed.any() and SHOW_GRIPPER_EVENTS:
        ax.scatter(
            x[gripper_changed],
            y[gripper_changed],
            marker="*",
            s=50,
            color="orange",
            zorder=12,
            label=r"Gripper change" if i == 0 else None,
        )

ax.set_xlabel(r"$x~[\mathrm{m}]$")
ax.set_ylabel(r"$y~[\mathrm{m}]$")

ax.legend(loc="best")
ax.grid(True)
ax.set_aspect("equal", adjustable="box")

ax.set_xlim(*plot_xlim)
ax.set_ylim(*plot_ylim)

ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(6))

# ----------------------------------------------------------------------
# Colorbar for trajectory time (seconds)
# ----------------------------------------------------------------------
if SHOW_COLORED_TRAJ:
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # required for some matplotlib versions

    cbar = fig.colorbar(
        sm,
        ax=ax,
        orientation="horizontal",
        location="top",
        pad=0.01
    )

    # move colorbar to the top
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.tick_top()

    # append " s" to tick labels
    ticks = cbar.get_ticks()
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.1f} s" for t in ticks])

# ----------------------------------------------------------------------
# Episode length statistics
# ----------------------------------------------------------------------
lengths_steps = np.array([len(data["x"]) for data in episodes.values() if len(data["x"]) > 0])
lengths_sec = lengths_steps / fps

for i, data in enumerate(episodes.values()):
    print(f"Episode {i}: {len(data['x']) / fps :.2f} s")

print(f"Episode length (steps): mean = {lengths_steps.mean():.1f}, std = {lengths_steps.std(ddof=1):.1f}")
print(f"Episode length (seconds): mean = {lengths_sec.mean():.2f} s, std = {lengths_sec.std(ddof=1):.2f} s")

plt.tight_layout()
plt.show()
