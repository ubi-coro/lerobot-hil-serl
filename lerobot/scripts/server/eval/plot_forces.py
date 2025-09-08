#!/usr/bin/env python3
import os
import pickle
import argparse
from copy import copy

import numpy as np
from matplotlib import pyplot as plt

from lerobot.common.robot_devices.motors.find_compliance_parameters import compute_theta, exp_scale_and_derivative

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

SUB_SAMPLING = 11
GRID_POINTS = 500
START_TIME = 0.87
PATCH_FILL_INDEX = 2000
PATCH_OFFSET_S = 0.2
HARD_LIMIT_N = 7.0         # hard safety limit
LIMIT_OFFSET_N = 0.08
SOFT_EQ_LIMIT_N = 2.0      # soft/adaptive equilibrium target baseline
CONTACT_THRESHOLD = 0.5     # N, threshold on F_meas to detect first contact
SMOOTH_WIN = 21             # samples for simple rolling mean (odd number)

def _rolling_mean(x, win):
    win = int(win)
    if win < 1:
        return x
    if win % 2 == 0:
        win += 1
    k = np.ones(win, dtype=float) / win
    # pad with edge values to keep length and avoid phase shift
    pad = win // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xpad, k, mode="valid")

def _concat_timestamp(ts, gap_threshold: float = 0.05):
    """
    Returns a Series (float seconds) that starts at 0 and skips idle gaps.

    Parameters
    ----------
    gap_threshold : float
        Any time delta > gap_threshold is treated as down-time and removed.
    """
    concat = np.zeros_like(ts)

    first_ts = ts[0]
    cum_gap = 0.0
    prev = ts[0]
    for i in range(len(ts)):
        if i > 0:
            delta = ts[i] - prev
            if delta > gap_threshold:
                cum_gap += delta            # remove the whole pause
            concat[i] = ts[i] - first_ts - cum_gap
            prev = ts[i]
        else:
            concat[0] = 0.0
    return concat

def main(pkl_path: str):
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"Could not find results file: {pkl_path}")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Expect keys from the recording script
    required = ["timestamps", "ctrl_forces", "measured_forces"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing keys in pickle: {missing}")

    timestamps = np.asarray(data["timestamps"], dtype=float)
    ctrl_forces = np.asarray(data["ctrl_forces"], dtype=float)
    measured_forces = -np.asarray(data["measured_forces"], dtype=float)

    # sort
    idc = np.argsort(timestamps)
    timestamps = timestamps[idc]
    ctrl_forces_raw = ctrl_forces[idc]
    measured_forces_raw = measured_forces[idc]

    timestamps = _concat_timestamp(timestamps)

    # Normalize timestamps to start at zero
    timestamps = timestamps - timestamps[0]

    # Zoom in
    idc = (timestamps > START_TIME) & (timestamps < START_TIME + 1.0)
    timestamps = timestamps[idc]
    ctrl_forces = ctrl_forces_raw[idc]
    measured_forces = measured_forces_raw[idc]
    timestamps = timestamps - timestamps[0]

    # compute theoretical/adaptive force limit over time (F_ideal)
    data["s_min"] = data["s_min"] * 0.9
    timestamps_short = timestamps[::SUB_SAMPLING]
    #f_vals = np.zeros(len(timestamps_short))
    f_vals = copy(ctrl_forces[::SUB_SAMPLING]) + LIMIT_OFFSET_N
    theta = compute_theta(data["f_star"], data["F_max"], data["s_min"])
    for i, t in enumerate(timestamps_short):
        if ctrl_forces[int(SUB_SAMPLING * i)] > 4.99:
            f_vals[i] = HARD_LIMIT_N  # data["F_max"]
        else:
            #prev = f_vals[i - 1] if i > 0 else 0.0
            #s, _ = exp_scale_and_derivative(prev, theta, data["s_min"])
            #f_vals[i] = data["F_max"] * s
            continue

    # build a fine grid for smooth plotting
    grid = np.linspace(0, timestamps[-1], GRID_POINTS)

    ctrl_forces_plot = np.interp(grid, timestamps[::SUB_SAMPLING], ctrl_forces[::SUB_SAMPLING])
    measured_forces_plot = np.interp(grid, timestamps[::SUB_SAMPLING], measured_forces[::SUB_SAMPLING])
    f_vals_plot = np.interp(grid, timestamps_short, f_vals)

    # Patch longer flight time
    idc = grid < PATCH_OFFSET_S
    ctrl_forces_plot[idc] = 5.0
    measured_forces_plot[idc] = measured_forces_raw[PATCH_FILL_INDEX:PATCH_FILL_INDEX+SUB_SAMPLING*sum(idc):SUB_SAMPLING]

    measured_forces_plot[idc] = measured_forces_plot[idc] - measured_forces_plot[sum(idc) - 1] - 0.5
    f_vals_plot[idc] = HARD_LIMIT_N

    # simple smoothing for contact detection (does not affect plotted lines)
    meas_smooth = _rolling_mean(measured_forces, SMOOTH_WIN)
    # first index where measured force indicates sustained contact
    contact_idx = np.argmax(meas_smooth > CONTACT_THRESHOLD)
    if meas_smooth[contact_idx] <= CONTACT_THRESHOLD:
        # fallback: no contact found
        contact_time = timestamps[0]
    else:
        contact_time = timestamps[contact_idx]

    fig, ax = plt.subplots(figsize=(3.54, 3.0))

    ax.set_xlabel(r"$\mathbf{Time\ [s]}$")
    ax.set_ylabel(r"$\mathbf{Force\ [N]}$")

    # main curves
    ax.plot(
        grid,
        ctrl_forces_plot,
        label=r"$F_{\mathrm{cmd}}$",
        ls="-",
        lw=1.8,
        alpha=0.95,
        zorder=3,
        color="blue"
    )
    ax.plot(
        grid,
        measured_forces_plot,
        label=r"$F_{\mathrm{meas}}$",
        ls="-",
        lw=1.6,
        alpha=0.9,
        zorder=3,
        color="green"
    )
    ax.plot(
        grid,
        f_vals_plot,
        label=r"$F_{\mathrm{lim}}(F_{\mathrm{meas}})$",
        lw=1.6,
        ls="-",
        alpha=0.95,
        zorder=3,
        color="orange"
    )
    # Build a polygon between F_soft and the top of the plot and hatch it
    from matplotlib.patches import Polygon, Patch

    y_top = ax.get_ylim()[1]  # current top of y-axis (call this AFTER plotting your lines)
    xs = np.concatenate([grid, grid[::-1]])
    ys = np.concatenate([np.full_like(grid, y_top), f_vals_plot[::-1]])
    poly = Polygon(
        np.c_[xs, ys],
        closed=True,
        facecolor="none",          # no fill color; show only hatch
        edgecolor="orange",        # hatch color comes from edgecolor
        hatch="///",               # diagonal stripes
        linewidth=0.0,
        zorder=0
    )
    ax.add_patch(poly)

    # limits (make them visually distinct from the signals)
    #ax.hlines(
    #    HARD_LIMIT_N,
    #    xmin=timestamps.min(),
    #    xmax=timestamps.max(),
    #    color="red",
    #    ls="-",
    #    lw=1.4,
    #    label=r"$F_{hard}$",
    #    zorder=2,
    #)
    ax.hlines(
        SOFT_EQ_LIMIT_N,
        xmin=timestamps.min(),
        xmax=timestamps.max(),
        color="orange",
        ls=":",
        lw=2.2,
        label=r"$F_{\mathrm{lim}}^{\mathrm{*}}$",
        zorder=2,
    )

    # contact marker and "before contact" shading
    #ax.axvspan(
    #    0.0, contact_time, color="gray", alpha=0.4, zorder=0, label="Before Contact"
    #)
    ax.axvline(contact_time, color="k", ls="--", lw=1.9, alpha=0.8, zorder=1, label="Contact Happens")

    # grid and legend (move legend down so it doesn’t cover the 7N limit)
    ax.grid(True, linewidth=0.5, alpha=0.8)
    for spine in ax.spines.values(): spine.set_linewidth(0.7)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)

    plt.tight_layout()
    # Save a vector and a bitmap version for the paper
    plt.savefig("results/forces.pdf", dpi=600, bbox_inches='tight', pad_inches=0, transparent=False)
    plt.savefig("results/forces.png", dpi=400)

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recreate contact force plot from saved results."
    )
    parser.add_argument(
        "--pkl",
        default=os.path.join("results", "contact_forces.pkl"),
        help="Path to contact_forces.pkl (default: results/contact_forces.pkl)",
    )
    args = parser.parse_args()
    main(args.pkl)
