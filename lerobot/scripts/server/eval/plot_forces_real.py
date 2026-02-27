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
    "legend.fontsize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "text.usetex": True,
    "axes.linewidth": 0.7,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linewidth": 0.33,
    "text.latex.preamble": r"\usepackage{bm}"
})

SUB_SAMPLING = 5
GRID_POINTS = 5000
START_TIME = 0.2
PATCH_FILL_INDEX = 2000
PATCH_OFFSET_S = 0.2
HARD_LIMIT_N = 7.0
LIMIT_OFFSET_N = 0.3
SOFT_EQ_LIMIT_N = 5.0
CONTACT_THRESHOLD = 6.0      # N, threshold on F_meas to detect first contact
SMOOTH_WIN = 21              # samples for simple rolling mean (odd number)
DURATION = 1.0               # [s]

# ---- metric settings ----
CMD_DROP_THRESHOLD_N = 6.0   # "contact regime" threshold for F_cmd
SETTLE_BAND_N = 1.0          # settle band around SOFT_EQ_LIMIT_N, e.g. ±1N
SETTLE_MIN_DWELL_S = 0.20    # must stay in band for at least this long
SETTLE_USE_SMOOTHED = True   # use smoothed F_meas for settle detection
PRINT_METRICS = True
HIST_Y_OFFSET = -2.0
VLINE_LABEL_OFFSET = -12.5


def _rolling_mean(x, win):
    win = int(win)
    if win < 1:
        return x
    if win % 2 == 0:
        win += 1
    k = np.ones(win, dtype=float) / win
    pad = win // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xpad, k, mode="valid")


def _concat_timestamp(ts, gap_threshold: float = 0.2):
    concat = np.zeros_like(ts)
    first_ts = ts[0]
    cum_gap = 0.0
    prev = ts[0]
    for i in range(len(ts)):
        if i > 0:
            delta = ts[i] - prev
            if delta > gap_threshold:
                cum_gap += delta
            print(cum_gap)
            concat[i] = ts[i] - first_ts - cum_gap
            prev = ts[i]
        else:
            concat[0] = 0.0
    return concat


def _first_true_run(mask: np.ndarray, min_run_len: int):
    """
    Return the first index i such that mask[i:i+min_run_len] are all True.
    Returns None if not found.
    """
    if min_run_len <= 1:
        idx = np.argmax(mask)
        return int(idx) if mask[idx] else None

    count = 0
    start_idx = None
    for i, m in enumerate(mask):
        if m:
            if count == 0:
                start_idx = i
            count += 1
            if count >= min_run_len:
                return int(start_idx)
        else:
            count = 0
            start_idx = None
    return None


def _estimate_sample_dt(timestamps):
    if len(timestamps) < 2:
        return 0.0
    dts = np.diff(timestamps)
    # robust estimate
    return float(np.median(dts))


def _safe_text(v, fmt="{:.3f}"):
    if v is None:
        return "n/a"
    try:
        return fmt.format(float(v))
    except Exception:
        return str(v)


def _compute_single_run_metrics(
    timestamps,
    ctrl_forces,
    measured_forces,
    contact_time,
    contact_idx,
    target_force=SOFT_EQ_LIMIT_N,
    cmd_drop_threshold=CMD_DROP_THRESHOLD_N,
    settle_band=SETTLE_BAND_N,
    settle_min_dwell_s=SETTLE_MIN_DWELL_S,
    smooth_win=SMOOTH_WIN,
    use_smoothed_for_settle=True,
):
    """
    Compute metrics for a single run from contact onward.
    Steady-state metrics are computed from detected settling index to end.
    """
    metrics = {}

    # Basic guards
    if contact_idx is None or contact_idx >= len(timestamps):
        metrics["valid"] = False
        metrics["reason"] = "No valid contact index."
        return metrics

    dt = _estimate_sample_dt(timestamps)
    if dt <= 0:
        metrics["valid"] = False
        metrics["reason"] = "Invalid timestamps / dt."
        return metrics

    metrics["valid"] = True
    metrics["dt_s"] = dt
    metrics["target_force_N"] = float(target_force)
    metrics["contact_time_s"] = float(contact_time)

    # Slice from contact onward
    t_post = timestamps[contact_idx:]
    f_cmd_post = ctrl_forces[contact_idx:]
    f_meas_post = measured_forces[contact_idx:]
    f_meas_smooth_post = _rolling_mean(f_meas_post, smooth_win)

    # 1) Peak force + overshoot
    peak_idx_rel = int(np.argmax(f_meas_post))
    peak_force = float(f_meas_post[peak_idx_rel])
    peak_time = float(t_post[peak_idx_rel])
    overshoot_N = float(peak_force - target_force)
    overshoot_pct = float(100.0 * overshoot_N / target_force) if target_force > 1e-9 else np.nan

    metrics["peak_force_N"] = peak_force
    metrics["peak_force_time_s"] = peak_time
    metrics["overshoot_N"] = overshoot_N
    metrics["overshoot_pct"] = overshoot_pct

    # 2) Contact -> commanded force drop below threshold
    cmd_drop_rel = np.argmax(f_cmd_post <= cmd_drop_threshold)
    if f_cmd_post[cmd_drop_rel] <= cmd_drop_threshold:
        cmd_drop_time = float(t_post[cmd_drop_rel])
        metrics["cmd_drop_threshold_N"] = float(cmd_drop_threshold)
        metrics["cmd_drop_time_s"] = cmd_drop_time
        metrics["contact_to_cmd_drop_s"] = float(cmd_drop_time - contact_time)
    else:
        metrics["cmd_drop_threshold_N"] = float(cmd_drop_threshold)
        metrics["cmd_drop_time_s"] = None
        metrics["contact_to_cmd_drop_s"] = None

    # 3) Settling detection
    # Use smoothed or raw for the *decision*, but steady-state stats are always computed on raw
    f_settle_signal = f_meas_smooth_post if use_smoothed_for_settle else f_meas_post
    in_band = np.abs(f_settle_signal - target_force) <= settle_band

    min_dwell_samples = max(1, int(np.ceil(settle_min_dwell_s / dt)))
    settle_idx_rel = _first_true_run(in_band, min_dwell_samples)

    if settle_idx_rel is None:
        metrics["settled"] = False
        metrics["settle_time_s"] = None
        metrics["contact_to_settle_s"] = None
        metrics["settle_band_N"] = float(settle_band)
        metrics["settle_min_dwell_s"] = float(settle_min_dwell_s)
        metrics["steady_state_start_idx"] = None
        # Still compute excess-force integral over whole post-contact segment as fallback
        excess = np.maximum(f_meas_post - target_force, 0.0)
        metrics["excess_force_integral_Ns"] = float(np.trapz(excess, t_post))
        return metrics

    settle_idx = contact_idx + settle_idx_rel
    settle_time = float(timestamps[settle_idx])

    metrics["settled"] = True
    metrics["settle_time_s"] = settle_time
    metrics["contact_to_settle_s"] = float(settle_time - contact_time)
    metrics["settle_band_N"] = float(settle_band)
    metrics["settle_min_dwell_s"] = float(settle_min_dwell_s)
    metrics["steady_state_start_idx"] = int(settle_idx)

    # 4) Steady-state metrics (from settle -> end), on RAW measured force
    t_ss = timestamps[settle_idx:]
    f_ss = measured_forces[settle_idx:]

    metrics["steady_state_duration_s"] = float(t_ss[-1] - t_ss[0]) if len(t_ss) > 1 else 0.0
    metrics["steady_state_mean_force_N"] = float(np.mean(f_ss))
    metrics["steady_state_mean_error_N"] = float(np.mean(f_ss) - target_force)
    metrics["steady_state_std_force_N"] = float(np.std(f_ss))
    metrics["steady_state_mae_N"] = float(np.mean(np.abs(f_ss - target_force)))
    metrics["steady_state_p95_abs_error_N"] = float(np.percentile(np.abs(f_ss - target_force), 95))

    # 5) Excess-force integral (contact -> settle and contact -> end)
    excess_post = np.maximum(f_meas_post - target_force, 0.0)
    metrics["excess_force_integral_total_Ns"] = float(np.trapz(excess_post, t_post))

    t_imp = timestamps[contact_idx:settle_idx + 1]
    f_imp = measured_forces[contact_idx:settle_idx + 1]
    excess_imp = np.maximum(f_imp - target_force, 0.0)
    metrics["excess_force_integral_to_settle_Ns"] = float(np.trapz(excess_imp, t_imp))

    return metrics


def _print_metrics(metrics):
    print("\n=== Single-run compliance metrics ===")
    if not metrics.get("valid", False):
        print(f"Invalid metrics: {metrics.get('reason', 'unknown')}")
        return

    def _fmt(v, nd=4):
        if v is None:
            return "n/a"
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.{nd}f}"
        return str(v)

    print(f"Target force [N]:                 {_fmt(metrics.get('target_force_N'), 3)}")
    print(f"Contact time [s]:                 {_fmt(metrics.get('contact_time_s'), 4)}")

    print(f"Peak force [N]:                   {_fmt(metrics.get('peak_force_N'), 3)}")
    print(f"Peak force time [s]:              {_fmt(metrics.get('peak_force_time_s'), 4)}")
    print(f"Overshoot [N]:                    {_fmt(metrics.get('overshoot_N'), 3)}")
    print(f"Overshoot [% of target]:          {_fmt(metrics.get('overshoot_pct'), 1)}")

    print(f"Cmd drop threshold [N]:           {_fmt(metrics.get('cmd_drop_threshold_N'), 3)}")
    print(f"Cmd drop time [s]:                {_fmt(metrics.get('cmd_drop_time_s'), 4)}")
    print(f"Contact -> cmd drop [s]:          {_fmt(metrics.get('contact_to_cmd_drop_s'), 4)}")

    print(f"Settled:                          {metrics.get('settled')}")
    print(f"Settle band [N]:                  ±{_fmt(metrics.get('settle_band_N'), 3)}")
    print(f"Settle min dwell [s]:             {_fmt(metrics.get('settle_min_dwell_s'), 3)}")
    print(f"Settle time [s]:                  {_fmt(metrics.get('settle_time_s'), 4)}")
    print(f"Contact -> settle [s]:            {_fmt(metrics.get('contact_to_settle_s'), 4)}")

    if metrics.get("settled", False):
        print(f"Steady-state duration [s]:        {_fmt(metrics.get('steady_state_duration_s'), 3)}")
        print(f"Steady-state mean force [N]:      {_fmt(metrics.get('steady_state_mean_force_N'), 3)}")
        print(f"Steady-state mean error [N]:      {_fmt(metrics.get('steady_state_mean_error_N'), 3)}")
        print(f"Steady-state std [N]:             {_fmt(metrics.get('steady_state_std_force_N'), 3)}")
        print(f"Steady-state MAE [N]:             {_fmt(metrics.get('steady_state_mae_N'), 3)}")
        print(f"Steady-state p95 abs err [N]:     {_fmt(metrics.get('steady_state_p95_abs_error_N'), 3)}")
        print(f"Excess-force integral to settle:  {_fmt(metrics.get('excess_force_integral_to_settle_Ns'), 4)} N·s")

    # Always print total excess force from contact onward
    if "excess_force_integral_total_Ns" in metrics:
        print(f"Excess-force integral total:      {_fmt(metrics.get('excess_force_integral_total_Ns'), 4)} N·s")
    elif "excess_force_integral_Ns" in metrics:
        print(f"Excess-force integral total:      {_fmt(metrics.get('excess_force_integral_Ns'), 4)} N·s")


def main(pkl_path: str):
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"Could not find results file: {pkl_path}")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

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
    timestamps = timestamps - timestamps[0]

    # Zoom in
    idc = (timestamps > START_TIME) & (timestamps < START_TIME + DURATION)
    timestamps = timestamps[idc]
    ctrl_forces = ctrl_forces_raw[idc]
    measured_forces = measured_forces_raw[idc]
    timestamps = timestamps - timestamps[0]

    # simple smoothing for contact detection (does not affect plotted lines)
    meas_smooth = _rolling_mean(measured_forces, SMOOTH_WIN)
    contact_idx = np.argmax(meas_smooth > CONTACT_THRESHOLD)
    if meas_smooth[contact_idx] <= CONTACT_THRESHOLD:
        contact_time = timestamps[0]
        contact_idx = 0
    else:
        contact_time = timestamps[contact_idx]

    # --- compute actual adaptive force limit F_lim(F_meas) from saved params ---
    # We evaluate the limit on the subsampled timeline for plotting.
    # Assumes results file contains: f_star, F_max, s_min
    for k in ["f_star", "F_max", "s_min"]:
        if k not in data:
            raise KeyError(f"Missing '{k}' in results file. Available keys: {list(data.keys())}")

    f_star = float(data["f_star"])
    F_max = float(data["F_max"])
    s_min = float(data["s_min"])

    s_min_plot = 1.0 * s_min

    theta = compute_theta(f_star, F_max, s_min_plot)

    timestamps_short = timestamps[::SUB_SAMPLING]
    meas_short = measured_forces[::SUB_SAMPLING]

    f_vals = np.zeros_like(meas_short, dtype=float)

    for i, f_meas in enumerate(meas_short):
        # Use non-negative contact force magnitude for the limiter law
        f_in = max(0.0, float(f_meas))

        # exp_scale_and_derivative returns scale s in [s_min, 1] (expected)
        s, _ = exp_scale_and_derivative(f_in, theta, s_min_plot)

        # Numerical safety clamp
        s = float(np.clip(s, s_min_plot, 1.0))

        # Adaptive limit in Newtons
        f_vals[i] = F_max * s

    f_vals = copy(ctrl_forces)

    max_lim_idx = np.argmax(measured_forces > 2.0)
    f_vals[:max_lim_idx] = 30.0

    # Optional visual offset so the hatch doesn't sit exactly on the line
    f_vals = f_vals + LIMIT_OFFSET_N

    #ctrl_forces[timestamps < contact_time] = 20.0

    # build a fine grid for smooth plotting
    grid = np.linspace(0, timestamps[-1], GRID_POINTS)
    ctrl_forces_plot = np.interp(grid, timestamps[::SUB_SAMPLING], ctrl_forces[::SUB_SAMPLING])
    measured_forces_plot = np.interp(grid, timestamps[::SUB_SAMPLING], measured_forces[::SUB_SAMPLING])
    f_vals_plot = np.interp(grid, timestamps_short, f_vals[::SUB_SAMPLING])


    # ---- compute metrics for this single run ----
    metrics = _compute_single_run_metrics(
        timestamps=timestamps,
        ctrl_forces=ctrl_forces,
        measured_forces=measured_forces,
        contact_time=contact_time,
        contact_idx=contact_idx,
        target_force=SOFT_EQ_LIMIT_N,
        cmd_drop_threshold=CMD_DROP_THRESHOLD_N,
        settle_band=SETTLE_BAND_N,
        settle_min_dwell_s=SETTLE_MIN_DWELL_S,
        smooth_win=SMOOTH_WIN,
        use_smoothed_for_settle=SETTLE_USE_SMOOTHED,
    )

    if PRINT_METRICS:
        _print_metrics(metrics)

    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.set_xlabel(r"$\mathbf{Time\ [s]}$")
    ax.set_ylabel(r"$\mathbf{Force\ [N]}$")
    ax.set_xlim([timestamps.min(), timestamps.max()])
    ax.set_ylim([-1.0, 1.05 * measured_forces_raw.max()])

    # Main curves
    ax.plot(
        grid, ctrl_forces_plot,
        label=r"$F_{\mathrm{cmd}}$",
        ls="-", lw=1.6, alpha=0.95, zorder=3, color="blue"
    )
    ax.plot(
        grid, measured_forces_plot,
        label=r"$F_{\mathrm{meas}}$",
        ls="-", lw=1.6, alpha=0.95, zorder=4, color="green"
    )

    # hatch "forbidden"/clipped region above adaptive limit (as before)
    from matplotlib.patches import Polygon
    y_top = ax.get_ylim()[1]
    xs = np.concatenate([grid, grid[::-1]])
    ys = np.concatenate([np.full_like(grid, y_top), f_vals_plot[::-1]])
    poly = Polygon(
        np.c_[xs, ys],
        closed=True,
        facecolor="none",
        edgecolor="red",
        hatch="///",
        linewidth=0.0,
        zorder=0,
        alpha=0.3
    )
    ax.add_patch(poly)

    # (Optional) adaptive limit line, if you still want it in the plot
    ax.plot(
        grid, f_vals_plot,
        label=r"$F_{\mathrm{lim}}(F_{\mathrm{meas}})$",
        lw=1.3, ls="-", alpha=0.85, zorder=5, color="red"
    )

    # --- Settling band (highlight "measured width of settle band") ---
    band_lo = SOFT_EQ_LIMIT_N - SETTLE_BAND_N
    band_hi = SOFT_EQ_LIMIT_N + SETTLE_BAND_N
    ax.axhline(
        SOFT_EQ_LIMIT_N,
        color="orange", ls=":", lw=1.6, alpha=0.95, zorder=1, label=r"$F_{\mathrm{lim}}^*$",
    )
    ax.axhspan(
        band_lo, band_hi,
        color="orange", alpha=0.13, zorder=0,
        label=r"Settle Band ($F_{\mathrm{lim}}^*\pm"+f"{SETTLE_BAND_N:.1f}$ N)"
    )


    # Add a compact bracket on the left to show band width
    x0 = timestamps.min() + 0.015 * (timestamps.max() - timestamps.min())
    tick_w = 0.015 * (timestamps.max() - timestamps.min())
    #ax.plot([x0, x0], [band_lo, band_hi], color="orange", lw=1.1, zorder=5)
    #ax.plot([x0, x0 + tick_w], [band_lo, band_lo], color="orange", lw=1.1, zorder=5)
    #ax.plot([x0, x0 + tick_w], [band_hi, band_hi], color="orange", lw=1.1, zorder=5)
    #ax.text(
    #    x0 + 1.2 * tick_w,
    #    SOFT_EQ_LIMIT_N,
    #    rf"$\pm {SETTLE_BAND_N:.1f}\,\mathrm{{N}}$",
    #    color="orange",
    #    fontsize=8,
    #    va="center",
    #    ha="left",
    #    zorder=6,
    #    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8)
    #)

    # --- Contact marker ---
    ax.axvline(
        contact_time, color="k", ls="--", lw=1.6, alpha=0.9, zorder=1
    )
    ax.text(
        contact_time - 0.02, ax.get_ylim()[1] + VLINE_LABEL_OFFSET,
        "contact",
        rotation=90, va="bottom", ha="right",
        fontsize=10, color="k",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8)
    )


    # --- Settling marker + steady-state region shading ---
    settle_t = metrics.get("settle_time_s", None)
    if metrics.get("settled", False) and settle_t is not None:
        ax.axvline(
            settle_t, color="k", ls="--", lw=1.3, alpha=0.9, zorder=1
        )
        #ax.axvspan(
        #    settle_t, timestamps[-1],
        #    color="k", alpha=0.08, zorder=0
        #)
        ax.text(
            settle_t + 0.02, ax.get_ylim()[1] + VLINE_LABEL_OFFSET,
            "settled",
            rotation=90, va="bottom", ha="left",
            fontsize=10, color="k",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8)
        )

        # Single duration arrow: contact -> settled
        y_arrow = ax.get_ylim()[1] - 5.0
        ax.annotate(
            "",
            xy=(settle_t, y_arrow), xytext=(contact_time, y_arrow),
            arrowprops=dict(arrowstyle="<->", lw=0.9, color="0.2")
        )
        dt_cs = metrics.get("contact_to_settle_s", None)
        if dt_cs is not None:
            ax.text(
                0.5 * (contact_time + settle_t),
                y_arrow + 1.5,
                rf"$t_{{\mathrm{{settle}}}}={dt_cs:.3f}\,\mathrm{{s}}$",
                ha="center", va="bottom", fontsize=10, color="0.2",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.8)
            )

    # --- Side distribution attached to right side (steady-state histogram) ---
    if metrics.get("settled", False) and metrics.get("steady_state_start_idx") is not None and False:
        ss_idx = int(metrics["steady_state_start_idx"])
        f_ss = measured_forces[ss_idx:] + HIST_Y_OFFSET

        # Inset axis attached to right side; shares y visually (force axis)
        # [x0, y0, width, height] in axis coordinates
        ax_hist = ax.inset_axes([1.01, 0.06, 0.18, 0.88], transform=ax.transAxes)

        # Histogram along force-axis (horizontal bars => "rotated" look)
        n_bins = 300
        counts, edges = np.histogram(f_ss, bins=n_bins, range=(ax.get_ylim()[0], ax.get_ylim()[1]))
        centers = 0.5 * (edges[:-1] + edges[1:])
        heights = np.diff(edges)

        # Normalize for compact width
        if counts.max() > 0:
            widths = counts / counts.max()
        else:
            widths = counts.astype(float)

        ax_hist.barh(
            centers, widths*0.1, height=heights * 1.0,
            color="darkgreen", linewidth=0.0, alpha=0.5
        )

        # Overlay mean and ±1σ of steady-state
        mu = metrics.get("steady_state_mean_force_N", None)
        sigma = metrics.get("steady_state_std_force_N", None)
        if mu is not None:
            ax_hist.axhline(mu + HIST_Y_OFFSET, color="darkgreen", lw=1.0, ls="-")
        #if (mu is not None) and (sigma is not None):
        #    ax_hist.axhspan(mu - sigma + HIST_Y_OFFSET, mu + sigma + HIST_Y_OFFSET, color="darkgreen", alpha=0.10)

        # Match y-range to main axis exactly
        ax_hist.set_ylim(ax.get_ylim())
        ax_hist.set_xlim(0, 0.12)

        # Clean look: no ticks except maybe a tiny label
        ax_hist.set_xticks([])
        ax_hist.set_yticks([])
        for spine in ["top", "bottom", "left", "right"]:
            ax_hist.spines[spine].set_visible(False)

        #ax_hist.text(
        #    0.98, 0.98, "ss dist.",
        #    transform=ax_hist.transAxes,
        #    ha="right", va="top", fontsize=7, color="0.25"
        #)

    # Cosmetics
    ax.grid(True, linewidth=0.45, alpha=0.75)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)

    # Keep legend minimal
    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        fontsize=8,
        borderpad=0.35,
        handlelength=1.8
    )

    # Pad y-limits a touch for the settle arrow
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + 0.6)

    # Make room on the right for the attached histogram
    plt.subplots_adjust(right=0.82)
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/forces.pdf", dpi=600, bbox_inches='tight', pad_inches=0.01, transparent=False)
    plt.savefig("results/forces.png", dpi=400)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recreate contact force plot from saved results and compute single-run metrics.")
    parser.add_argument(
        "--pkl",
        default=os.path.join("results", "contact_forces.pkl"),
        help="Path to contact_forces.pkl (default: results/contact_forces.pkl)",
    )
    args = parser.parse_args()
    main(args.pkl)
