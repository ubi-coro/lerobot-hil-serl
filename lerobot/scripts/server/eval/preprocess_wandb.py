#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess a resumed W&B run by building a *gap-free* wall-clock axis
from `_timestamp`.

A “gap” is any interval > --gap_threshold seconds (default 5 min).
These gaps are **removed**, so the new axis (`concat_time_h`) rises
continuously across resumes without flat pauses.

Output files (per metric):
    preprocessed/<run_id>_{success|cycle|intervention}.csv.gz
    preprocessed/<run_id>_{...}.pkl

Columns:
    concat_time_h , _step , <metric>

Usage
-----
python preprocess_run_concat_ts.py --run_id 2b77nhhc
python preprocess_run_concat_ts.py --run_id kgpn2oi6 --gap_threshold 600
"""
from __future__ import annotations
import argparse, os, pickle
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import wandb  # type: ignore
except Exception as e:
    raise SystemExit("wandb is required; install with `pip install wandb`.") from e


# Keys
SUCCESS_KEY = "train/insert/Success" #"Success Rate"
CYCLE_KEY   = "train/insert/Cycle Time [s]" #"Cycle Time"
ITV_KEY_PCT     = "train/insert/Episode intervention"
ITV_KEY     = "train/insert/Intervention rate [%]"
STEP_KEY    = "_step"
TS_KEY      = "_timestamp"        # absolute epoch seconds

METRIC_INFO = [
    ("success",      SUCCESS_KEY, "Success [%]", 40),
    ("cycle",        CYCLE_KEY,   "Cycle-time [s]", 40),
    ("intervention", ITV_KEY,     "Intervention [%]", 40),
    ("intervention_pct",  ITV_KEY_PCT,     "Intervention [%]", 30),
]

#CUSTOM_PREPROCESS = "scale"
#CUSTOM_KWARGS = {"gap_threshold": 300, "duration_s": 9600}

CUSTOM_PREPROCESS = "scale_and_cut"
CUSTOM_KWARGS = {"gap_threshold": 300, "duration_s": 9600, "max_time_s": 19680}

def ScaleToDuration(df, duration_s: float = 9600):
    max_time_s = df["time_s"].max()
    df["time_s"] = df["time_s"] * duration_s / max_time_s
    return df

def ScaleToDurationAndCut(df, duration_s: float = 9600, max_time_s: float = 9600):
    df = df.copy()
    max_time_s = min([df["time_s"].max(), max_time_s])
    df = df[df["time_s"] <= float(max_time_s)].copy()
    df["time_s"] = df["time_s"] * duration_s / max_time_s
    return df

CUMSTOM_PREPROCESS_DICT = {
    "scale": ScaleToDuration,
    "scale_and_cut": ScaleToDurationAndCut
}

# ──────────────────────────────────────────────────────────────────────────────
def fetch_history(entity: str, project: str, run_id: str,
                  max_points: int | None = None) -> List[Dict[str, Any]]:
    api = wandb.Api(timeout=60)
    run = api.run(f"{entity}/{project}/{run_id}")
    #rows = run.history(samples=max_points, pandas=False)
    rows: List[Dict[str, Any]] = []
    for row in run.scan_history(page_size=1000):
        rows.append(row)
        if max_points and len(rows) >= max_points:
            break
    return rows, run.name


def concat_timestamp(df: pd.DataFrame, gap_threshold: float = 300.0) -> pd.Series:
    """
    Returns a Series (float seconds) that starts at 0 and skips idle gaps.

    Parameters
    ----------
    gap_threshold : float
        Any time delta > gap_threshold is treated as down-time and removed.
    """
    if TS_KEY not in df.columns:
        raise ValueError(f"History has no '{TS_KEY}'. Cannot build concat time.")

    ts = df[TS_KEY].to_numpy(float)
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
    return pd.Series(concat, index=df.index, name="concat_time_s")


def build_metric_df(df, key: str, window: int) -> pd.DataFrame:
    from copy import copy, deepcopy
    kwargs = copy(CUSTOM_KWARGS)
    df = deepcopy(df)
    df = df[df[key].notna()]
    df["time_s"] = concat_timestamp(df, kwargs["gap_threshold"])
    del kwargs["gap_threshold"]

    if CUSTOM_PREPROCESS in CUMSTOM_PREPROCESS_DICT:
        df = CUMSTOM_PREPROCESS_DICT[CUSTOM_PREPROCESS](df, **kwargs)

    df["time_h"] = df["time_s"] / 3600.0
    df = df[[STEP_KEY, "time_h", key]].copy()

    df.reset_index(drop=True, inplace=True)

    # we'll smooth on aggregation
    """
    kernel = np.ones(window, dtype=float) / window                 # length = window
    valid  = np.convolve(df[key], kernel, mode="valid")            # length = N - window + 1

    # how much we have to pad so that len(padded) == len(original)
    pad_left  = (window - 1) // 2          # floor for even windows
    pad_right = (window - 1) - pad_left    # the remainder; keeps total = window-1

    # build full-length array – replicate the first / last valid value
    padded = np.empty_like(df[key], dtype=float)
    padded[:pad_left]                      = valid[0]      # left edge
    padded[pad_left : pad_left+len(valid)] = valid         # middle (true valid conv.)
    padded[pad_left+len(valid):]           = valid[-1]     # right edge

    df[key] = padded
    """

    return df


def save_df(df: pd.DataFrame, stem: str):
    csv = str(stem) + ".csv.gz"
    pkl = str(stem) + ".pkl"
    df.to_csv(csv, index=False, compression="gzip")
    with open(pkl, "wb") as fh:
        pickle.dump(df, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  ↳ {os.path.basename(csv)} ({len(df):,d} rows)")


def plot_stacked(dfs: Dict[str, pd.DataFrame], title: str = None):
    plt.rcParams.update({
        "font.size": 9,
        "axes.linewidth": .7,
        "axes.grid": True,
        "grid.alpha": .15,
        "grid.linewidth": .35,
    })
    fig, axes = plt.subplots(3, 1, figsize=(4.2, 3.6), sharex=True,
                             constrained_layout=True)

    if title:
        plt.title(title)

    for ax, (tag, key, ylabel, _) in zip(axes, METRIC_INFO):
        if tag not in dfs:
            continue

        d = dfs[tag]

        kernel = np.ones(21, dtype=float) / 21                 # length = window
        valid  = np.convolve(d[key], kernel, mode="valid")            # length = N - window + 1

        # how much we have to pad so that len(padded) == len(original)
        pad_left  = (21 - 1) // 2          # floor for even windows
        pad_right = (21 - 1) - pad_left    # the remainder; keeps total = window-1

        # build full-length array – replicate the first / last valid value
        padded = np.empty_like(d[key], dtype=float)
        padded[:pad_left]                      = valid[0]      # left edge
        padded[pad_left : pad_left+len(valid)] = valid         # middle (true valid conv.)
        padded[pad_left+len(valid):]           = valid[-1]     # right edge

        ax.plot(d["time_h"], padded, lw=1.0)
        ax.set_ylabel(ylabel)
        for sp in ax.spines.values():
            sp.set_linewidth(.7)

    axes[-1].set_xlabel("Wall Clock Time [h]")
    plt.show(block=False)


# ──────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description=(
                "Pre-process one W&B run **or all runs** of a project.\n"
                "If --run_id is omitted the script loops over every run in "
                "the given project."))
    p.add_argument("--run_id",
                   help="Single run-ID to process.  "
                        "Leave empty to process ALL runs in the project.")
    p.add_argument("--entity", default="bielefeld-coro")
    p.add_argument("--project", default="hil_amp_main")
    p.add_argument("--max_points", type=int, default=None)
    p.add_argument("--no_plot", action="store_true")
    args = p.parse_args()

    api = wandb.Api(timeout=60)

    # ---------- choose which runs ----------
    if args.run_id:
        runs = [api.run(f"{args.entity}/{args.project}/{args.run_id}")]
    else:
        runs = api.runs(f"{args.entity}/{args.project}")

    if not runs:
        raise SystemExit("No runs found in the project.")

    print(f"Found {len(runs)} run(s) to process.")
    os.makedirs("preprocessed", exist_ok=True)

    for run in runs:
        out_dir = Path("preprocessed") / f"{run.name}_{run.id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        if (out_dir / f"base.pkl").exists():
            print(f"Loading local history for {run.name} ({run.id}) …")

            with open(out_dir / f"base.pkl", "rb") as fh:
                df_base = pickle.load(fh)
        else:
            print(f"Downloading history for {run.name} ({run.id}) …")
            hist, _ = fetch_history(args.entity, args.project, run.id,
                                max_points=args.max_points)

            if not hist:
                print("   (no history rows – skipped)")
                continue

            df_base = pd.DataFrame(hist).sort_values(TS_KEY).reset_index(drop=True)

            save_df(df_base, out_dir / f"base")

        df_metrics = {}
        is_ppo = "ppo" in run.name

        for tag, key, _, window in METRIC_INFO:
            if is_ppo:
                old_key = key
                if tag == "success":
                    key = "Success Rate"
                elif tag == "cycle":
                    key = "Cycle Time"
                else:
                    continue

            df_metric = build_metric_df(df_base, key, window)

            if is_ppo:
                df_metric[old_key] = df_metric[key]

            df_metrics[tag] = df_metric
            save_df(df_metric, out_dir / f"{tag}_{len(df_metric[key])}_episodes")

        if not args.no_plot:
            plot_stacked(df_metrics, title=f"{run.name}_{run.id}")

    plt.show()


if __name__ == "__main__":
    main()
