#!/usr/bin/env python3
"""
Generalization visualization for SHaRe-RL rebuttal:
- Top row: 3 connector images (center-cropped to square)
- Bottom row: grouped bar chart with dual y-axis
    * Left y-axis: Success [%]
    * Right y-axis: Cycle Time [s]

Visual tweaks:
- Type labels moved to x-axis (stronger image-bar association)
- Figure-level separators connecting top image gaps to bottom group gaps
- "Train" / "Eval" semantic annotation above the bar chart
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

BEND_Y1 = 3.0
BEND_Y2 = 8.0


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "text.usetex": True,
    "axes.linewidth": 0.7,
    "grid.alpha": 0.4,
    "grid.linewidth": 0.33,
    "text.latex.preamble": r"\usepackage{bm}"
})


def center_crop_square(img: np.ndarray, x_offset: int = 0) -> np.ndarray:
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2 + x_offset
    x0 = max(0, min(x0, w - side))  # safe clamp
    return img[y0:y0 + side, x0:x0 + side]


def data_x_to_fig_x(fig, ax, x_data: float) -> float:
    """Convert x in data coordinates (of ax) to figure x-coordinate."""
    x_disp = ax.transData.transform((x_data, 0))[0]
    x_fig = fig.transFigure.inverted().transform((x_disp, 0))[0]
    return x_fig


def draw_cut_separator(fig, x_top, x_bottom, y_top, y_img_bottom, y_bar_top, y_bar_bottom,
                       color="0.89", lw=1.5, zorder=20, alpha=1.0):
    """
    Draw a separator that is vertical in the image row, then diagonally transitions,
    then continues vertically through the bar chart.
    All coordinates are in figure coords.
    """
    # Make the "bend" happen in the whitespace between top images and bottom chart.
    gap = y_img_bottom - y_bar_top
    y_kink_1 = y_img_bottom - BEND_Y1 * gap
    y_kink_2 = y_bar_top - BEND_Y2 * gap

    xs = [x_top, x_top, x_bottom, x_bottom]
    ys = [y_top, y_kink_1, y_kink_2, y_bar_bottom]

    fig.add_artist(Line2D(
        xs, ys,
        transform=fig.transFigure,
        color=color,
        linewidth=lw,
        solid_capstyle="round",
        zorder=zorder,
        alpha=alpha
    ))

def add_split_bend_labels(fig, left_text, right_text, x1, y1, x2, y2, offset_px=12, gap_px=14, fontsize=8):
    """
    Place two labels around the same angled bend segment:
    - left_text to the screen-left of the separator
    - right_text to the screen-right of the separator
    Both labels are rotated with the bend and offset away from the line along its normal.
    Coordinates are in figure coords.
    """
    p1 = np.array(fig.transFigure.transform((x1, y1)))
    p2 = np.array(fig.transFigure.transform((x2, y2)))
    v = p2 - p1
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-9:
        return

    vu = v / v_norm
    n = np.array([-vu[1], vu[0]])  # normal (display coords)

    angle = np.degrees(np.arctan2(vu[1], vu[0]))
    if angle < -90 or angle > 90:
        angle += 180

    pm = 0.5 * (p1 + p2)

    # Base point: move off the line along the normal
    p_base = pm + offset_px * n

    # Enforce semantic left/right placement in screen coordinates
    p_left = p_base + np.array([-gap_px, 0.0])
    p_right = p_base + np.array([gap_px, 0.0])

    for text, p in [(left_text, p_left), (right_text, p_right)]:
        x_text, y_text = fig.transFigure.inverted().transform(p)
        fig.text(
            x_text, y_text, text,
            transform=fig.transFigure,
            rotation=angle,
            rotation_mode="anchor",
            ha="center", va="center",
            fontsize=fontsize,
            bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.95),
            zorder=30,
        )


def main():
    # Data
    success_rates = [100, 95, 90]      # [%]
    cycle_times = [3.5, 4.8, 5.9]   # [s]

    img_paths = ["train.jpeg", "eval 1.jpeg", "eval 2.jpeg"]
    x_offsets = [0, 0, 20]

    # Type labels become x-axis labels
    x_type_labels = [r"HanDD (f)", r"HanDDD (f)", r"HanDD (m)"]

    fig = plt.figure(figsize=(5.2, 4.0))
    gs = GridSpec(
        nrows=2,
        ncols=3,
        figure=fig,
        height_ratios=[1.45, 1.0],
        hspace=0.00,
        wspace=0.04,
    )

    # ---- Top row images ----
    top_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for ax, img_path, o in zip(top_axes, img_paths, x_offsets):
        img_file = Path(img_path)
        if img_file.exists():
            img = mpimg.imread(img_file)
            ax.imshow(center_crop_square(img, x_offset=o))
        else:
            ax.text(0.5, 0.5, f"Image not found:\n{img_path}",
                    ha="center", va="center", fontsize=8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Reference geometry from top row
    pos_left = top_axes[0].get_position()
    pos_mid = top_axes[1].get_position()
    pos_right = top_axes[2].get_position()
    top_left = pos_left.x0
    top_right = pos_right.x1
    top_width = top_right - top_left

    # ---- Bottom row manual placement ----
    bottom_y0 = 0.13
    bottom_h = 0.34
    pad_left = 0.095
    pad_right = 0.080

    ax_success = fig.add_axes([
        top_left + pad_left,
        bottom_y0,
        top_width - pad_left - pad_right,
        bottom_h,
    ])
    ax_cycle = ax_success.twinx()

    # Bar positions
    x = np.arange(3, dtype=float)
    bar_w = 0.28

    # Colors / styles (pi0-like clean bars: solid + hatched)
    success_fc = "#4C9BCB"  # muted blue
    cycle_fc = "white"

    # Success bars (left axis)
    bars_success = ax_success.bar(
        x - bar_w/2,
        success_rates,
        width=bar_w,
        label="Success rate",
        color=success_fc,
        edgecolor=success_fc,
        linewidth=0.8,
        zorder=2,
    )

    # Cycle-time bars (right axis)
    bars_cycle = ax_cycle.bar(
        x + bar_w/2,
        cycle_times,
        width=bar_w,
        label="Cycle time",
        color=cycle_fc,
        edgecolor=success_fc,
        hatch="///",
        linewidth=0.8,
        zorder=2,
    )

    # Axis formatting
    ax_success.set_xlim(-0.6, 2.6)
    ax_success.set_xticks(x)
    ax_success.set_xticklabels(x_type_labels)

    ax_success.set_ylabel(r"$\mathbf{Success\ [\%]}$")
    ax_success.set_yticks([0, 20, 40, 60, 80, 100])
    ax_success.set_ylim(0, 120)

    ax_cycle.set_ylabel(r"$\mathbf{Cycle\ Time\ [s]}$")
    ax_cycle.set_yticks([0, 2, 4, 6, 8, 10])
    ax_cycle.set_ylim(0, 12)

    # Grid only from left axis
    ax_success.grid(True, axis="y", linewidth=0.45, alpha=0.75, zorder=0)
    ax_cycle.grid(False)

    # Value labels
    for rect, s in zip(bars_success, success_rates):
        cx = rect.get_x() + rect.get_width() / 2
        cy = rect.get_height()
        ax_success.annotate(
            rf"{s}\%",
            (cx, cy),
            textcoords="offset points",
            xytext=(0, 2),
            ha="center",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85)
        )

    for rect, ct in zip(bars_cycle, cycle_times):
        cx = rect.get_x() + rect.get_width() / 2
        cy = rect.get_height()
        ax_cycle.annotate(
            rf"{ct:.1f}\,s",
            (cx + 0.02, cy),
            textcoords="offset points",
            xytext=(0, 2),
            ha="center",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85)
        )

    # Legend
    ax_success.set_zorder(1)
    ax_cycle.set_zorder(2)
    ax_success.patch.set_alpha(0)

    handles = [bars_success[0], bars_cycle[0]]
    labels = ["Success Rate", "Cycle Time"]
    leg = ax_cycle.legend(
        handles, labels,
        loc="lower right",
        frameon=True,
        borderpad=0.3,
        handlelength=1.6,
    )
    leg.set_zorder(100)
    leg.get_frame().set_alpha(1.0)

    # Styling
    for spine in ax_success.spines.values():
        spine.set_linewidth(0.7)
    for spine in ax_cycle.spines.values():
        spine.set_linewidth(0.7)

    ax_success.tick_params(axis="x", which="both", length=0)
    ax_cycle.tick_params(axis="x", which="both", length=0)
    ax_success.minorticks_off()
    ax_cycle.minorticks_off()

    # ----------------------------
    # Figure-level cut separators
    # ----------------------------
    pos_img0 = top_axes[0].get_position()
    pos_img1 = top_axes[1].get_position()
    pos_img2 = top_axes[2].get_position()
    pos_bar = ax_success.get_position()

    # Separators between x groups in bar plot (data x = 0.5 and 1.5)
    x_sep_01_bar = data_x_to_fig_x(fig, ax_success, 0.5)

    # Centers of the gaps between image panels (top row)
    x_sep_01_top = 0.5 * (pos_img0.x1 + pos_img1.x0)

    y_top = pos_img0.y1
    y_img_bottom = pos_img0.y0
    y_bar_top = pos_bar.y1
    y_bar_bottom = pos_bar.y0

    draw_cut_separator(
        fig, x_sep_01_top, x_sep_01_bar,
        y_top, y_img_bottom, y_bar_top, y_bar_bottom,
        color="#4C9BCB", lw=1.5, zorder=20, alpha=0.8
    )

    # Single "Train | Eval" annotation at the bend of the separator
    gap = y_img_bottom - y_bar_top
    y_kink_1 = y_img_bottom - BEND_Y1 * gap
    y_kink_2 = y_bar_top - BEND_Y2 * gap

    add_split_bend_labels(
        fig,
        left_text="Train",
        right_text="Eval",
        x1=x_sep_01_top, y1=y_kink_1,
        x2=x_sep_01_bar, y2=y_kink_2,
        offset_px=0,
        gap_px=12,
        fontsize=8,
    )

    # ----------------------------
    # "Train" / "Eval" annotations
    # ----------------------------
    # Segment centers in bar chart data coordinates:
    # Train segment = left side to first separator  -> center approx x=0.0
    # Eval segment  = between first and last chart bounds over groups 1 and 2 -> center x=1.5
    # We place them in figure coords just above the bar chart frame.

    # Export
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "generalization_bar.pdf", dpi=600, bbox_inches="tight", pad_inches=0)
    plt.savefig(out_dir / "generalization_bar.png", dpi=400, bbox_inches="tight", pad_inches=0)
    plt.show()


if __name__ == "__main__":
    main()
