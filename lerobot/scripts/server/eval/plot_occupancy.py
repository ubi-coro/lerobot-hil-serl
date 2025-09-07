import os
import numpy as np

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import pyplot as plt
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
    "grid.alpha": 0.12,
    "grid.linewidth": 0.33,
    "text.latex.preamble": r"\usepackage{bm}"
})

SPARSE_ROOT = "/home/jannick/data/paper/hil-amp/rlpd_reward_sparse_cam_toWindow_terminate_early_init_large_demos_itv_2/run/learner-2025-09-05-09-45-48/insert/dataset"
DENSE_ROOT  = "/home/jannick/data/paper/hil-amp/rlpd_reward_dense_cam_toWindow_terminate_early_init_large_no_priors_1/run/learner-2025-09-04-11-01-19/insert/dataset/"

SPARSE_REPO_ID = "hil_amp_main/rlpd_reward_sparse_cam_toWindow_terminate_early_init_large_demos_itv_2"
DENSE_REPO_ID  = "hil_amp_main/rlpd_reward_dense_cam_toWindow_terminate_early_init_large_no_priors_1"

SUB_SAMPLING = 1
MAX_LEN = 1000
POSE_KEY = 'complementary_info.observation.main_eef_pos'

connector_img   = plt.imread("connector.png")
cross_section_img = plt.imread("cross_section.png")

# --- helpers -----------------------------------------------------------------
def load_dense(dataset, max_len, sub=1):
    xs, ys = [], []
    for i in tqdm(range(len(dataset)), desc="Load pose information"):
        p = dataset[i][POSE_KEY]
        xs.append(-p[0] * 1000.0)        # x
        ys.append(-p[2] * 1000.0)       # -z (goal is low)
        if i >= max_len:
            break
    xs = np.asarray(xs)[::sub]
    ys = np.asarray(ys)[::sub]
    return xs, ys

def load_sparse(dataset: LeRobotDataset, max_len, sub=1):
    xs, ys = [], []
    for e, (start, end) in tqdm(enumerate(zip(dataset.episode_data_index["from"], dataset.episode_data_index["to"])), desc="Load pose information"):
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

def density_grid(x, y, bins=200, xy_range=None, smooth_sigma=1.2):
    """Return normalized 2D density H, xedges, yedges."""
    if xy_range is None:
        pad_x = 0.05 * (np.max(x) - np.min(x) + 1e-12)
        pad_y = 0.05 * (np.max(y) - np.min(y) + 1e-12)
        xy_range = [[np.min(x)-pad_x, np.max(x)+pad_x],
                    [np.min(y)-pad_y, np.max(y)+pad_y]]
    #H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=xy_range, density=True)
    #if _HAS_SCIPY and smooth_sigma and smooth_sigma > 0:
    #    H = _gaussian_filter(H, smooth_sigma, mode='nearest')
    ## Avoid zero-only arrays
    #if np.max(H) > 0:
    #    H /= np.max(H)
    #return H.T, xedges, yedges  # transpose so H[y, x] matches imshow/contour

    H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=xy_range, density=True)
    if _HAS_SCIPY and smooth_sigma and smooth_sigma > 0:
        # slightly stronger smoothing + reflect to improve connectivity near edges
        H = _gaussian_filter(H, smooth_sigma, mode='reflect')
    # normalize and boost contrast so edges look crisper
    H = H / (np.max(H) + 1e-12)
    H = np.power(H, 0.7)  # gamma < 1 sharpens transitions
    return H.T, xedges, yedges  # transpose so H[y, x] matches imshow/contour


def quantile_levels(H, qs=(0.70, 0.85, 0.95, 0.99)):
    """Compute contour levels so that each level encloses approximately q-mass."""
    flat = H.ravel()
    # sort descending (highest density center → outward)
    idx = np.argsort(flat)[::-1]
    cumsum = np.cumsum(flat[idx])
    cumsum /= cumsum[-1] if cumsum[-1] > 0 else 1.0
    levels = []
    for q in qs:
        # find threshold t where mass inside >= q
        pos = np.searchsorted(cumsum, q, side='left')
        t = flat[idx[pos]] if pos < len(idx) else flat[idx[-1]]
        levels.append(t)
    # ensure strictly increasing for contour
    levels = sorted(set(levels))
    # if degenerate (e.g., tiny dataset), fall back to linear levels
    if len(levels) < 3:
        levels = np.linspace(0.3, 0.95, 4)
    return levels

# --- load datasets ------------------------------------------------------------
sparse_dataset = LeRobotDataset(repo_id=SPARSE_REPO_ID, root=SPARSE_ROOT)
dense_dataset  = LeRobotDataset(repo_id=DENSE_REPO_ID,  root=DENSE_ROOT)

sx, sy = load_sparse(sparse_dataset, MAX_LEN, SUB_SAMPLING)
dx, dy = load_dense(dense_dataset,  MAX_LEN, SUB_SAMPLING)

# Use a shared range so contours are comparable
x_min = min(sx.min(), dx.min()); x_max = max(sx.max(), dx.max())
y_min = min(sy.min(), dy.min()); y_max = max(sy.max(), dy.max())
xy_range = [[x_min, x_max], [y_min, y_max]]
xy_diffs = [x_max - x_min, y_max - y_min]

# Build densities
bins = 500  # higher resolution → crisper edges
H_sparse, xedges, yedges = density_grid(sx, sy, bins=bins, xy_range=xy_range, smooth_sigma=1.1)
H_dense,  _,      _      = density_grid(dx, dy, bins=bins, xy_range=xy_range, smooth_sigma=1.1)


# Compute quantile-based contour levels (shared for fairness)
eps = 1e-6
levels_sparse = np.linspace(0.03, 1.0 + eps, 8)
levels_dense  = np.linspace(0.03, 1.0 + eps, 8)

# --- plot ---------------------------------------------------------------------
plt.figure(figsize=(6.2, 4.4), dpi=200)
ax = plt.gca()

ax.set_xlim(left=xy_range[0][0] - 0.1 * xy_diffs[0], right=xy_range[0][1] + 0.2 * xy_diffs[0])
ax.set_ylim(bottom=xy_range[1][0] - 0.1 * xy_diffs[1], top=xy_range[1][1] + 0.1 * xy_diffs[1])

# --- background connector on right -------------------------------------------
# compute placement in data units so it aligns with your axes (mm)
x0, x1 = ax.get_xlim()
y0, y1 = ax.get_ylim()
xr, yr = (x1 - x0), (y1 - y0)

# tweakable parameters
conn_width_frac   = 0.26   # fraction of x-range the connector should span
conn_right_margin = 0.03   # fraction of x-range as right margin
conn_y_center_frac = 0.55  # vertical center of image as fraction of y-range
conn_alpha = 0.20

conn_w = conn_width_frac * xr
conn_aspect = connector_img.shape[0] / connector_img.shape[1]  # (px height / px width)
conn_h = conn_w * conn_aspect  # equal aspect => mm in x == mm in y

conn_x2 = x1 - conn_right_margin * xr
conn_x1 = conn_x2 - conn_w
conn_yc = y0 + conn_y_center_frac * yr
conn_y1 = conn_yc - conn_h / 2
conn_y2 = conn_yc + conn_h / 2

ax.imshow(
    connector_img,
    extent=[conn_x1, conn_x2, conn_y1, conn_y2],
    alpha=conn_alpha,
    zorder=0,                  # behind contourf (which defaults > 0)
    interpolation="bilinear",
    clip_on=True,
)

# Filled contours (semi transparent). Choose two readable colormaps.
Xc = 0.5 * (xedges[1:] + xedges[:-1])
Yc = 0.5 * (yedges[1:] + yedges[:-1])

# trim the colormaps so the lightest shades are not near-white
from matplotlib import cm, colors
blues = colors.ListedColormap(cm.Blues(np.linspace(0.18, 1.00, 256)))
oranges = colors.ListedColormap(cm.Oranges(np.linspace(0.20, 1.00, 256)))

cs2 = ax.contourf(
    Xc, Yc, H_dense,
    levels = levels_dense,
    cmap = oranges,
    vmin=0.0, vmax=0.6,
    alpha = 0.68,
    antialiased = True
)

cs1 = ax.contourf(
    Xc, Yc, H_sparse,
    levels=levels_sparse,  # include top bin so center isn't white
    cmap = blues,
    vmin=0.0, vmax=0.6,
    alpha = 0.72,
    antialiased = True
)

# Cosmetics
ax.set_xlabel("$\mathbf{X-Position [mm]}$")
ax.set_ylabel("$\mathbf{Z-Position [mm]}$")
ax.set_aspect("equal")  # distances comparable
ax.grid(True, linewidth=0.3, alpha=0.3)

# Legend patches
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor=plt.cm.Blues(0.6),  edgecolor="navy",       label="Sparse + Priors"),
    Patch(facecolor=plt.cm.Oranges(0.6), edgecolor="darkorange", label="Dense, No Priors"),
]
ax.legend(handles=legend_handles, loc="lower left", frameon=True)

plt.tight_layout()
# Save a vector and a bitmap version for the paper
plt.savefig("results/state_occupancy_density.pdf")
plt.savefig("results/state_occupancy_density.png", dpi=400)
plt.show()
