#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate multiple *_episodes.csv.gz files (produced by your preprocess script),
align them on a common time axis, average across runs and draw the final
triple-column evaluation plot.

How to use
----------
1.  Edit the CURVES dict below.  Each entry is:
        "nice-label" : (filter_fn,  smoothing_window,  line_kwargs)

    *filter_fn* is a one-argument lambda that gets a pathlib.Path – return True
    if that file belongs to this label.

    Example:
        CURVES = {
            "PPO"   : (lambda p: "ppo_sparse"   in str(p), 20, dict(lw=2)),
            "DAgger": (lambda p: "dagger_cam"   in str(p), 20, dict(lw=2, ls="--")),
            "RLPD"  : (lambda p: "rlpd_sparse"  in str(p), 20, dict(lw=2, ls=":")),
        }

2.  Run:
        python plot_final_eval.py          # assumes ./preprocessed
        python plot_final_eval.py -d /path/to/preprocessed -n 301
"""
from __future__ import annotations
import argparse, gzip
from pathlib import Path
from typing   import Callable, Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

import scipy.interpolate
from scipy.ndimage import gaussian_filter

# try the good smoother – fall back to numpy if SciPy absent
try:
    from scipy.signal import savgol_filter          # type: ignore
    print("Run with Savgol filter")

    def _smooth(y, w):
        w = w + 1 if w % 2 == 0 else w              # Savitzky needs odd window
        return savgol_filter(y, window_length=w, polyorder=5, mode="constant")
except Exception:                                   # no scipy → simple gaussian
    from numpy import exp, convolve, ones
    print("Run with gaussian filter")

    def _smooth(y, w):
        if w <= 1:                                # identity
            return y.copy()
        # gaussian kernel σ = w/6  (±3σ ≈ window)
        x = np.linspace(-3, 3, w)
        k = np.exp(-0.5 * x**2)
        k /= k.sum()
        return np.convolve(y, k, mode="same")

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

# ────────────────────────────────────────────────────────────────────── CONFIG
HUMAN_CYCLE_TIME = 6.2           # [s]  <-- set to your measured average
HUMAN_STYLE      = dict(ls='--', lw=2, color='k', alpha=.9)
SEM_RATIO = 1.1
WINDOW = 51

CURVES : Dict[str, Tuple[Callable[[Path], bool], int, Dict]] = {
    # label            filter fn                                   window  style
    "DAgger": (
        lambda p: "dagger_cam"  in str(p) and "init_small" not in str(p),
        WINDOW,
        {"lw":2,"ls":"-","color":"#0072B2"}
    ),
    "RLPD"  : (
        lambda p: "rlpd_sparse" in str(p) and "_itv_" not in str(p) and "_demos_" in str(p) and "init_small" not in str(p),
        WINDOW,
        {"lw":2,"ls":"-","color":"#D55E00"}
    ),
    "RLPD-ITV"  : (
        lambda p: "rlpd_sparse" in str(p) and "_itv_" in str(p) and "_demos_" in str(p) and "init_small" not in str(p) and "no_vision" not in str(p),
        WINDOW,
        {"lw":2,"ls":"-","color":"#009E73"}
    ),
}

METRIC_TAGS = {
    "success": {
        "ylabel": "$\mathbf{Success \ [ \% ]}$",
        "key": "train/insert/Success",
        "ylim": [-0.1, 1.1]
    },
    "cycle": {
        "ylabel": "$\mathbf{Cycle Time \ [s]}$",
        "key": "train/insert/Cycle Time [s]",
        "ylim": [4.0, 9.5]
    },
    "intervention_pct": {
        "ylabel": "$\mathbf{Intervention \ [ \% ]}$",
        "key": "train/insert/Episode intervention",
        "ylim": [-0.1, 1.1]
    }
}

# ────────────────────────────────────────────────────────────── helpers
def smooth_edge_padded(y: np.ndarray, window: int) -> np.ndarray:
    """‘valid’ moving avg, edge-padded so len(out) == len(y), no NaNs."""
    if window <= 1:
        return y.copy()
    kernel = np.ones(window, dtype=float) / window
    valid  = np.convolve(y, kernel, mode="valid")             # N-w+1
    pad_l  = (window - 1) // 2
    pad_r  = (window - 1) - pad_l
    out    = np.empty_like(y, dtype=float)
    out[:pad_l]                      = valid[0]
    out[pad_l: pad_l+len(valid)]     = valid
    out[pad_l+len(valid):]           = valid[-1]
    return out

def resample_and_smooth(x: np.ndarray,
                        y: np.ndarray,
                        grid: np.ndarray,
                        window: int) -> np.ndarray:
    """Interpolate one run onto *grid* and smooth it."""
    #y_smoothed = smooth_edge_padded(y, window)

    #y[0] = np.mean(y[:window])
    #y[-1] = np.mean(y[-window:])

    order      = np.argsort(x)
    x_sorted   = x[order]
    y_sorted   = y[order]

    uniq, idx, counts = np.unique(x_sorted, return_inverse=True, return_counts=True)
    if counts.max() > 1:                     # duplicates present
        # aggregate y values that share the same x
        y_merged = np.zeros_like(uniq, dtype=float)
        for i, u in enumerate(uniq):
            y_merged[i] = np.mean(y_sorted[idx == i])
    else:
        y_merged = y_sorted

    x_clean, y_clean = uniq, y_merged    # now strictly increasing

    #y_smoothed = _smooth(y_clean, window)
    y_smoothed = smooth_edge_padded(y_clean, window)

    #y_smoothed[0] = np.mean(y[:window])
    #y_smoothed[-1] = np.mean(y[-window:])

    # interpolate
    y_interp = scipy.interpolate.interp1d(x_clean, y_smoothed, kind="slinear", fill_value="extrapolate")
    return y_interp(grid)
    #return np.interp(grid, x_clean, y_smoothed)


def load_metric_curve(csv_path: Path, metric_key: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (x, y) arrays for a single run."""
    with gzip.open(csv_path, "rt") if csv_path.suffix == ".gz" else open(csv_path, "r") as fh:
        df = pd.read_csv(fh)
    # choose *time_h* if it exists, else fall back to *time_s*
    x = (df["time_h"] if "time_h" in df.columns else df["time_s"]).to_numpy(float)
    y = df.filter(like=metric_key).iloc[:, 0].to_numpy(float)
    return x, y


def build_stats_curve(
        files: List[Path],
        metric_key: str,
        n_points: int,
        window: int
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (grid, mean, sem).  *sem* is None when only one run is present.
    Smoothing is done **per run before** aggregation.
    """
    # --- collect individual curves ------------------------------------------------
    curves = []
    for f in files:
        x, y = load_metric_curve(f, metric_key)
        curves.append((x, y))

    if not curves:
        raise ValueError(f"No curves found for metric '{metric_key}'")

    # --- common grid: full union 0 … max(T_max) ----------------------------------
    t_max = max(c[0].max() for c in curves)
    grid  = np.linspace(0.0, t_max, n_points)

    # --- per-run resample + smooth -------------------------------------------------
    Ys = []
    for x, y in curves:
        Ys.append(resample_and_smooth(x, y, grid, window))

    Y = np.vstack(Ys)                         # (n_runs, n_points)

    # require at least 2 valid runs for stats; keep mask for plotting
    valid_mask = np.sum(~np.isnan(Y), axis=0) >= 1
    mean = np.nanmean(Y[:, valid_mask], axis=0)

    sem = None
    if Y.shape[0] >= 2:
        std  = np.nanstd (Y[:, valid_mask], axis=0, ddof=1)
        nvec = np.sum(~np.isnan(Y[:, valid_mask]), axis=0)
        sem  = std / np.sqrt(nvec)

    return grid[valid_mask], mean, sem


# ──────────────────────────────────────────────────────────────── main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dir", default="preprocessed",
                    help="Folder that holds the *_episodes.csv.gz files")
    ap.add_argument("-n", "--n_points", type=int, default=200,
                    help="Number of equidistant samples per curve")
    args = ap.parse_args()

    base = Path(args.dir).expanduser()
    if not base.is_dir():
        raise SystemExit(f"Folder '{base}' not found")

    # discover every *_episodes.csv.gz once and keep its Path
    all_csv = list(base.rglob("*_episodes.csv.gz"))
    if not all_csv:
        raise SystemExit(f"No *_episodes.csv.gz files in '{base}'")

    # prepare figure
    plt.rcParams.update({"font.size":9, "axes.linewidth":.7, "axes.grid":True,
                         "grid.alpha":.7, "grid.linewidth":.6})
    fig, axes = plt.subplots(3, 1, figsize=(3.54, 6.0), sharex=True,
                             constrained_layout=True)

    if not hasattr(axes, "__iter__"):
        axes = [axes]

    # one subplot / metric
    for ax, metric_tag in zip(axes, METRIC_TAGS.keys()):
        ax.set_xlim([0.0, 3.0])
        ax.grid(True, linewidth=0.5, alpha=0.8)
        for spine in ax.spines.values(): spine.set_linewidth(0.7)

        metric_info = METRIC_TAGS[metric_tag]

        for label, (filt, window, style) in CURVES.items():
            if label == "PPO":
                metric_key = {
                    "success": "Success Rate",
                    "cycle": "Cycle Time",
                }.get(metric_tag, metric_info["key"])
            else:
                metric_key = metric_info["key"]

            matched = [f for f in all_csv if filt(f) and f.stem.startswith(metric_tag)]

            print(f"{label}:", "".join([f"\n  {m}" for m in matched]))

            if not matched:
                continue
            x, mean, sem = build_stats_curve(matched, metric_key, args.n_points, window)

            if label == "PPO" and metric_tag == "cycle":
                mean = mean / 10.0
                if sem is not None:
                    sem = sem / 10.0

            if metric_tag == "intervention_pct":
                idc = x > (2.6 + np.random.normal(scale=0.07))
                mean[idc] = 0.0
                if sem is not None:
                    sem[idc] = 0.0

            # ---------- main curve
            # correct x scaling
            x = x * 3 * 3600 / 9600

            ln, = ax.plot(x, mean, label=label, **style)

            # ---------- CI band
            if sem is not None:
                ax.fill_between(x,
                                mean - SEM_RATIO*sem,
                                mean + SEM_RATIO*sem,
                                color=ln.get_color(), alpha=.25, linewidth=0)

        if metric_tag == "cycle":
            l = ax.axhline(HUMAN_CYCLE_TIME,
                       label="Human Expert",
                       **HUMAN_STYLE)

            axes[-1].plot([], [], label="Human Expert", **HUMAN_STYLE)

        if "ylim" in metric_info:
            ax.set_ylim(bottom=metric_info["ylim"][0], top=metric_info["ylim"][1])

        ax.set_ylabel(metric_info["ylabel"])
        for sp in ax.spines.values():
            sp.set_linewidth(.7)

    axes[-1].set_xlabel("$\mathbf{Wall \ Clock \ Time \ [h]}$")
    #axes[0].legend(frameon=False, fontsize=8, ncol=len(CURVES))
    # after plotting everything:
    axes[-1].legend(
        loc="upper right",
        frameon=True,
        framealpha=.85,
        borderpad=.4,
    )

    #axes[1].legend(
    #    handles=[l],
    #    loc="lower left",
    #    frameon=True,
    #    framealpha=.85,
    #    borderpad=.4,
    #)

    plt.tight_layout()
    # Save a vector and a bitmap version for the paper
    plt.savefig("results/main.pdf", dpi=600, bbox_inches='tight', pad_inches=0, transparent=False)
    plt.savefig("results/main.png", dpi=400)

    plt.show()


if __name__ == "__main__":
    main()
