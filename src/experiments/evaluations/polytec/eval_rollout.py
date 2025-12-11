import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import OBS_STATE

# ----------------------------------------------------------------------
# Matplotlib / LaTeX setup
# ----------------------------------------------------------------------
plt.rcParams["text.usetex"] = True      # comment out if LaTeX not installed
plt.rcParams["font.family"] = "serif"

root = "/mnt/nvme0n1p3/data/polytec/eval_act_031225"
repo_id = "test/test"

# None => all episodes, or e.g. [0, 1, 2]
requested_episodes = None
# requested_episodes = [0, 1]

# obs space is 17 dimensional
#  0.  x/pos
#  1.  x/vel
#  2.  x/wrench
#  3.  y/pos
#  4.  y/vel
#  5.  y/wrench
#  6.  z/pos
#  7.  z/vel
#  8.  z/wrench
#  9.  a/vel
# 10.  a/wrench
# 11.  b/vel
# 12.  b/wrench
# 13.  c/pos
# 14.  c/vel
# 15.  c/wrench
# 16.  gripper/pos

# ----------------------------------------------------------------------
# Build cache filename
# ----------------------------------------------------------------------
if requested_episodes is None:
    ep_spec = "all"
else:
    ep_spec = "ep_" + "_".join(str(e) for e in requested_episodes)

cache_filename = f"rollout_cache_{repo_id.replace('/', '_')}_{ep_spec}.pkl"

# script directory (fallback to CWD if __file__ not defined)
script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
cache_path = os.path.join(script_dir, cache_filename)

print(f"Using cache file: {cache_path}")

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

    # Collect per-episode trajectories
    # ep_idx -> dict(x=[...], y=[...], z=[...], rewards=[...], gripper_changed=[...])
    episodes = {}
    prev_gripper_state = {}  # track gripper state per episode

    for frame in tqdm(dataset):
        ep_idx = int(frame["episode_index"])

        if ep_idx not in episodes:
            episodes[ep_idx] = {
                "x": [],
                "y": [],
                "z": [],
                "rewards": [],
                "gripper_changed": [],
            }

        obs = frame[OBS_STATE]

        # detect gripper changes
        gripper_state = float(obs[-1])
        changed = False
        if ep_idx in prev_gripper_state:
            changed = (gripper_state != prev_gripper_state[ep_idx])
        prev_gripper_state[ep_idx] = gripper_state

        # Make sure we convert to plain floats (in case these are tensors)
        episodes[ep_idx]["x"].append(float(obs[0]))
        episodes[ep_idx]["y"].append(float(obs[3]))
        episodes[ep_idx]["z"].append(float(obs[6]))
        episodes[ep_idx]["gripper_changed"].append(changed)
        episodes[ep_idx]["rewards"].append(float(frame["next.reward"]))

    # Save to cache
    print("Saving episodes to cache...")
    with open(cache_path, "wb") as f:
        pickle.dump(episodes, f)

# ----------------------------------------------------------------------
# 3D rollout figure (unchanged, just LaTeX + nicer ticks)
# ----------------------------------------------------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

cmap = plt.get_cmap("viridis")

for i, (ep_idx, data) in enumerate(sorted(episodes.items())):
    x = np.array(data["x"])
    y = np.array(data["y"])
    z = np.array(data["z"])
    rewards = np.array(data["rewards"])
    gripper_changed = np.array(data["gripper_changed"], dtype=bool)

    if len(x) < 2:
        continue  # not enough points to draw a line

    # Time parameter from 0 to 1 over the episode
    t = np.linspace(0.0, 1.0, len(x))

    # Build line segments for a 3D line with gradient color
    points = np.stack([x, y, z], axis=1)
    segments = np.stack([points[:-1], points[1:]], axis=1)

    # Colors along the trajectory (one color per segment)
    colors = cmap(t[:-1])

    lc = Line3DCollection(segments, colors=colors, linewidth=2, alpha=0.9)
    ax.add_collection3d(lc)

    # Mark the start state
    ax.scatter(
        x[0], y[0], z[0],
        marker="o",
        s=40,
        label=r"Episode start" if i == 0 else None,
        color=colors[0],
    )

    # Determine success from final reward
    success = rewards[-1] > 0.0
    end_marker = "^" if success else "x"

    ax.scatter(
        x[-1], y[-1], z[-1],
        marker=end_marker,
        s=60,
        color="green" if success else "red",
        zorder=10,
        label=r"Successful end" if (i == 0 and success) else
              r"Failed end" if (i == 0 and not success) else None,
    )

    # Plot all gripper-change events (consistent symbol & color)
    if gripper_changed.any():
        ax.scatter(
            x[gripper_changed],
            y[gripper_changed],
            z[gripper_changed],
            marker="*",
            s=50,
            color="orange",
            zorder=15,
            label=r"Gripper change" if i == 0 else None,
        )

ax.set_xlabel(r"$x~[\mathrm{m}]$")
ax.set_ylabel(r"$y~[\mathrm{m}]$")
ax.set_zlabel(r"$z~[\mathrm{m}]$")
ax.set_title(r"\textbf{Policy Rollouts}")

ax.legend(loc="best")
ax.grid(True)

# Equal-ish aspect
all_x = np.concatenate([np.array(v["x"]) for v in episodes.values()])
all_y = np.concatenate([np.array(v["y"]) for v in episodes.values()])
all_z = np.concatenate([np.array(v["z"]) for v in episodes.values()])

x_range = all_x.max() - all_x.min()
y_range = all_y.max() - all_y.min()
z_range = all_z.max() - all_z.min()
max_range = max(x_range, y_range, z_range)

x_mid = 0.5 * (all_x.max() + all_x.min())
y_mid = 0.5 * (all_y.max() + all_y.min())
z_mid = 0.5 * (all_z.max() + all_z.min())

ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
ax.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)

# Clean ticks: few, nicely spaced
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.set_major_locator(MaxNLocator(4))

plt.tight_layout()

# ----------------------------------------------------------------------
# Trajectory consistency: time-normalized mean + std
# ----------------------------------------------------------------------
# Normalize all episodes to a fixed length N in [0,1] time
N = 100
tau = np.linspace(0.0, 1.0, N)

norm_x_list = []
norm_y_list = []
norm_z_list = []

for ep_idx, data in sorted(episodes.items()):
    x = np.array(data["x"])
    y = np.array(data["y"])
    z = np.array(data["z"])

    if len(x) < 2:
        continue

    t_orig = np.linspace(0.0, 1.0, len(x))
    # 1D interpolation per coordinate
    x_interp = np.interp(tau, t_orig, x)
    y_interp = np.interp(tau, t_orig, y)
    z_interp = np.interp(tau, t_orig, z)

    norm_x_list.append(x_interp)
    norm_y_list.append(y_interp)
    norm_z_list.append(z_interp)

norm_x = np.vstack(norm_x_list)  # shape [E, N]
norm_y = np.vstack(norm_y_list)
norm_z = np.vstack(norm_z_list)

mean_x = norm_x.mean(axis=0)
mean_y = norm_y.mean(axis=0)
mean_z = norm_z.mean(axis=0)

std_x = norm_x.std(axis=0)
std_y = norm_y.std(axis=0)
std_z = norm_z.std(axis=0)

# Plot per-axis trajectories with mean and ±1σ band
fig2, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)

coords = [
    (norm_x, mean_x, std_x, r"x"),
    (norm_y, mean_y, std_y, r"y"),
    (norm_z, mean_z, std_z, r"z"),
]

for ax_i, (all_traj, mean_traj, std_traj, label) in enumerate(coords):
    # individual episodes (light grey)
    for ep_traj in all_traj:
        axes[ax_i].plot(tau, ep_traj, color="0.85", linewidth=0.8)

    # mean trajectory
    axes[ax_i].plot(tau, mean_traj, color="C0", linewidth=2, label=r"Mean")

    # ±1σ band
    axes[ax_i].fill_between(
        tau,
        mean_traj - std_traj,
        mean_traj + std_traj,
        color="C0",
        alpha=0.25,
        label=r"$\pm 1\sigma$" if ax_i == 0 else None,
    )

    axes[ax_i].set_ylabel(fr"${label}~[\mathrm{{m}}]$")
    axes[ax_i].yaxis.set_major_locator(MaxNLocator(4))
    axes[ax_i].grid(True, linestyle="--", alpha=0.4)

axes[-1].set_xlabel(r"Normalized time $\tau$")
axes[0].set_title(r"\textbf{Time-normalized end-effector trajectory}")

axes[0].legend(loc="best")

plt.tight_layout()
plt.show()
