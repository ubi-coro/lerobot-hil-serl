import math
import pickle
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.path import Path as PltPath
from matplotlib.patches import PathPatch, Circle
from matplotlib.lines import Line2D
from tqdm import tqdm

try:
    # optional but makes the densities prettier
    from scipy.ndimage import gaussian_filter as _gaussian_filter
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "text.usetex": True,
    "axes.linewidth": 0.7,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linewidth": 0.33,
    "text.latex.preamble": r"\usepackage{bm}"
})

SPARSE_ROOT = "/mnt/nvme0n1p3/Paper & Projects (Disk)/SHaRe/paper/data//dataset_dense/"
DENSE_ROOT  = "/mnt/nvme0n1p3/Paper & Projects (Disk)/SHaRe/paper/data//dataset_sparse"

SPARSE_REPO_ID = "hil_amp_main/rlpd_reward_sparse_cam_toWindow_terminate_early_init_large_demos_itv_2"
DENSE_REPO_ID  = "hil_amp_main/rlpd_reward_dense_cam_toWindow_terminate_early_init_large_no_priors_1"

SUB_SAMPLING = 1
MAX_LEN = 1000000000
ROT_DEG = 1.0
POSE_KEY = 'complementary_info.observation.main_eef_pos'

connector_img = plt.imread("connector.png")
cross_section_img = plt.imread("cross_section.png")

# --- helpers -----------------------------------------------------------------
def load_dense(dataset, max_len, sub=1):
    xs, ys = [], []
    for sample in tqdm(dataset, desc="Load pose information"):
        p = sample[POSE_KEY]
        xs.append(-p[0] * 1000.0)        # x
        ys.append(-p[2] * 1000.0)        # -z (goal is low)
        if len(xs) >= max_len:
            break
    xs = np.asarray(xs)[::sub]
    ys = np.asarray(ys)[::sub]
    return xs, ys

def load_sparse(dataset: LeRobotDataset, max_len, sub=1):
    xs, ys = [], []
    for e, (start, end) in tqdm(
        enumerate(zip(dataset.episode_data_index["from"], dataset.episode_data_index["to"])),
        desc="Load pose information"
    ):
        xs_episode, ys_episode = [], []
        is_intervention = False
        for i in range(start, end):
            is_intervention = is_intervention or bool(dataset[i]["complementary_info.is_intervention"])
            if is_intervention:
                break

            xs_episode.append(-dataset[i][POSE_KEY][0] * 1000.0)
            ys_episode.append(-dataset[i][POSE_KEY][2] * 1000.0)

        if is_intervention:
            print(f" [Ignored episode {e}]")
            continue

        xs.extend(xs_episode)
        ys.extend(ys_episode)

        if len(xs) >= max_len:
            break

    xs = np.asarray(xs)[::sub]
    ys = np.asarray(ys)[::sub]
    return xs, ys

def rotate_points(x: np.ndarray, y: np.ndarray, deg: float):
    if abs(deg) < 1e-9:
        return x, y
    rad = math.radians(deg)
    cos_t, sin_t = math.cos(rad), math.sin(rad)
    x_r = cos_t * x - sin_t * y
    y_r = sin_t * x + cos_t * y
    return x_r, y_r

def density_grid(x, y, bins=200, xy_range=None, smooth_sigma=1.2):
    if xy_range is None:
        pad_x = 0.05 * (np.max(x) - np.min(x) + 1e-12)
        pad_y = 0.05 * (np.max(y) - np.min(y) + 1e-12)
        xy_range = [[np.min(x)-pad_x, np.max(x)+pad_x],
                    [np.min(y)-pad_y, np.max(y)+pad_y]]

    H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=xy_range, density=True)
    if _HAS_SCIPY and smooth_sigma and smooth_sigma > 0:
        H = _gaussian_filter(H, smooth_sigma, mode='reflect')
    H = H / (np.max(H) + 1e-12)
    H = np.power(H, 0.7)
    return H.T, xedges, yedges

# --- load datasets ------------------------------------------------------------
dataset_path = Path("preprocessed") / "occupancies.pkl"
if dataset_path.exists():
    with open(dataset_path, "rb") as fn:
        data = pickle.load(fn)
    sx, sy = data["sx"], data["sy"]
    dx, dy = data["dx"], data["dy"]
else:
    sparse_dataset = LeRobotDataset(repo_id=SPARSE_REPO_ID, root=SPARSE_ROOT)
    dense_dataset  = LeRobotDataset(repo_id=DENSE_REPO_ID,  root=DENSE_ROOT)

    sx, sy = load_sparse(sparse_dataset, MAX_LEN, SUB_SAMPLING)
    dx, dy = load_dense(dense_dataset,  MAX_LEN, SUB_SAMPLING)

    data = {"sx": sx, "sy": sy, "dx": dx, "dy": dy}
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dataset_path, "wb") as fn:
        pickle.dump(data, fn, protocol=pickle.HIGHEST_PROTOCOL)

sy += 30.0
dy += 30.0

sx, sy = rotate_points(sx, sy, ROT_DEG)
dx, dy = rotate_points(dx, dy, ROT_DEG)

# swap since I f'ed up the paths
dx, sx = sx, dx
dy, sy = sy, dy

x_min = min(sx.min(), dx.min()); x_max = max(sx.max(), dx.max())
y_min = min(sy.min(), dy.min()); y_max = max(sy.max(), dy.max())
xy_range = [[x_min, x_max], [y_min, y_max]]
xy_diffs = [x_max - x_min, y_max - y_min]

bins = 500
H_sparse, xedges, yedges = density_grid(sx, sy, bins=bins, xy_range=xy_range, smooth_sigma=1.1)
H_dense,  _,      _      = density_grid(dx, dy, bins=bins, xy_range=xy_range, smooth_sigma=1.1)

eps = 1e-6
levels_sparse = np.linspace(0.03, 1.0 + eps, 8)
levels_dense  = np.linspace(0.03, 1.0 + eps, 8)

# --- plot ---------------------------------------------------------------------
fig = plt.figure(figsize=(4.6, 3.15), dpi=200)
ax = fig.add_axes([0.075, 0.14, 0.59, 0.79])  # main plot (smaller, tighter)

ax.grid(True, linewidth=0.5, alpha=0.8)
for spine in ax.spines.values():
    spine.set_linewidth(1.1)

ax.set_xlim(left=xy_range[0][0] - 0.05 * xy_diffs[0], right=xy_range[0][1] + 0.2 * xy_diffs[0])
ax.set_ylim(bottom=xy_range[1][0] - 0.1 * xy_diffs[1], top=xy_range[1][1] + 0.05 * xy_diffs[1])

# --- background connector image on main axis ---------------------------------
x0, x1 = ax.get_xlim()
y0, y1 = ax.get_ylim()
xr, yr = (x1 - x0), (y1 - y0)

conn_width_frac = 0.27
conn_right_margin = 0.195
conn_y_center_frac = 0.56
conn_alpha = 0.8

conn_w = conn_width_frac * xr
conn_aspect = connector_img.shape[0] / connector_img.shape[1]
conn_h = conn_w * conn_aspect
conn_x2 = x1 - conn_right_margin * xr
conn_x1 = conn_x2 - conn_w
conn_yc = y0 + conn_y_center_frac * yr
conn_y1 = conn_yc - conn_h / 2
conn_y2 = conn_yc + conn_h / 2

ax.imshow(
    connector_img,
    extent=[conn_x1, conn_x2, conn_y1, conn_y2],
    alpha=conn_alpha,
    zorder=0,
    interpolation="bilinear",
    clip_on=True,
)

# --- geometry path ------------------------------------------------------------
center = [
    (-25, 24),
    (-6.3, 24),
    (-6.3, -5),
    (5.9, -5),
    (5.9, 16),
    (11, 16),
    (11, 24),
]
codes = [PltPath.MOVETO] + [PltPath.LINETO] * (len(center) - 1)
geom_patch = PathPatch(
    PltPath(center, codes),
    transform=ax.transData,
    facecolor="none",
    edgecolor=(0, 0, 0, 1.0),
    linewidth=3.0,
    capstyle="round",
    joinstyle="round",
    label="Geometry",
    zorder=4
)
ax.add_patch(geom_patch)

# TCP and Goal markers (as in the paper legend)
tcp_xy = (0.6, 6.5)
goal_xy = (0.0, 0.0)

ax.add_patch(Circle(tcp_xy, 1.0, transform=ax.transData, facecolor="black",
                    edgecolor="black", linewidth=0.8, alpha=0.95, zorder=6))
ax.plot(goal_xy[0], goal_xy[1], marker='x', markersize=8, markeredgewidth=2.0,
        color='black', linestyle='None', zorder=6)

# --- densities ----------------------------------------------------------------
Xc = 0.5 * (xedges[1:] + xedges[:-1])
Yc = 0.5 * (yedges[1:] + yedges[:-1])

from matplotlib import cm, colors
greens  = colors.ListedColormap(cm.Greens(np.linspace(0.45, 1.00, 256)))
oranges = colors.ListedColormap(cm.Oranges(np.linspace(0.45, 1.00, 256)))

# Dense - SHaRe (orange)
ax.contourf(
    Xc, Yc, H_dense * 1.3,
    levels=levels_dense,
    cmap=oranges,
    vmin=0.0, vmax=0.6,
    alpha=0.70,
    antialiased=True,
    zorder=1
)

# Sparse + SHaRe (green)
ax.contourf(
    Xc, Yc, H_sparse * 1.3,
    levels=levels_sparse,
    cmap=greens,
    vmin=0.0, vmax=0.6,
    alpha=0.72,
    antialiased=True,
    zorder=2
)

# --- labels / style -----------------------------------------------------------
ax.set_xlabel(r"$\mathbf{x\ [mm]}$")
ax.set_ylabel(r"$\mathbf{z\ [mm]}$")
ax.set_aspect("equal")

# --- upper-right inset image (cross-section) ----------------------------------
ax_inset = fig.add_axes([0.58, 0.50, 0.45, 0.43])
ax_inset.imshow(cross_section_img)
ax_inset.axis("off")

# --- legend (paper-style) -----------------------------------------------------
from matplotlib.patches import Patch

legend_handles = [
    Patch(facecolor="white", edgecolor="black", linewidth=1.2, label="Geometry"),
    Line2D([0], [0], marker='o', color='black', markerfacecolor='black',
           markersize=8, linestyle='None', label="TCP"),
    Line2D([0], [0], marker='x', color='black', markersize=7,
           markeredgewidth=1.8, linestyle='None', label="Goal"),
    Patch(facecolor=cm.Oranges(0.65), edgecolor=cm.Oranges(0.85), linewidth=1.5, label=r"SAC (dense)"),
    Patch(facecolor=cm.Greens(0.60), edgecolor=cm.Greens(0.85), linewidth=1.5, label=r"SHaRe-RL (sparse)"),
]

ax_leg = fig.add_axes([0.61, 0.185, 0.4, 0.305])
ax_leg.axis("off")
leg = ax_leg.legend(handles=legend_handles, loc="upper left", frameon=True, fontsize=10)
leg.get_frame().set_alpha(0.95)

Path("results").mkdir(parents=True, exist_ok=True)
plt.savefig("results/occupancy.pdf", dpi=600, bbox_inches='tight', pad_inches=0.0, transparent=False)
plt.savefig("results/occupancy.png", dpi=400, bbox_inches='tight', pad_inches=0.01)
plt.show()
