#!/usr/bin/env python3
import argparse
from typing import Tuple, Optional, List, Dict, Any

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def _to_numpy(x):
    if hasattr(x, "detach"):  # torch tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _extract_wrench_from_frame(frame, state_key="observation.state") -> Optional[np.ndarray]:
    """
    Returns wrench np.ndarray shape (6,) = [Fx, Fy, Fz, Tx, Ty, Tz].
    """
    if "observation.main_eef_wrench" in frame:
        w = _to_numpy(frame["observation.main_eef_wrench"]).reshape(-1)
        if w.shape[0] >= 6:
            return w[:6].astype(np.float64)

    if state_key in frame:
        s = _to_numpy(frame[state_key]).reshape(-1)
        # AMPObsWrapper layout with torque:
        # [v0..v5, Fx,Fy,Fz,Tx,Ty,Tz, ...]
        if s.shape[0] >= 12:
            return s[6:12].astype(np.float64)

    return None


def _extract_speed_from_frame(frame, state_key="observation.state") -> Optional[np.ndarray]:
    """
    Returns speed np.ndarray shape (6,) = [vx, vy, vz, wx, wy, wz].
    """
    if "observation.main_eef_speed" in frame:
        v = _to_numpy(frame["observation.main_eef_speed"]).reshape(-1)
        if v.shape[0] >= 6:
            return v[:6].astype(np.float64)

    if state_key in frame:
        s = _to_numpy(frame[state_key]).reshape(-1)
        # AMPObsWrapper layout: first 6 are speed
        if s.shape[0] >= 6:
            return s[:6].astype(np.float64)

    return None


def _find_image_key(frame: Dict[str, Any]) -> Optional[str]:
    # Prefer a deterministic order
    preferred = [
        "observation.image",
        "observation.images.front",
        "observation.images.cam_high",
        "observation.images.cam_wrist",
        "observation.image.front",
    ]
    for k in preferred:
        if k in frame:
            return k
    for k in frame.keys():
        if "image" in k.lower():
            return k
    return None


def _extract_image_from_frame(frame: Dict[str, Any], image_key: Optional[str] = None) -> Optional[np.ndarray]:
    key = image_key if image_key is not None else _find_image_key(frame)
    if key is None or key not in frame:
        return None
    img = _to_numpy(frame[key])

    # Try to normalize shape to HxWxC for plotting
    img = np.squeeze(img)
    if img.ndim == 3:
        # CHW -> HWC
        if img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
            img = np.transpose(img, (1, 2, 0))
    elif img.ndim == 2:
        pass
    else:
        return None

    # If float image, clip
    if np.issubdtype(img.dtype, np.floating):
        img = np.clip(img, 0.0, 1.0)
    return img


def _collect_dataset_signals(
    dataset: LeRobotDataset,
    state_key: str = "observation.state",
    max_frames: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[Optional[np.ndarray]], dict]:
    """
    Collect wrench, speed, frame index, and image for all frames that have wrench+speed.
    Returns:
      w_all: (N,6)
      v_all: (N,6)
      frame_ids: list[int] (global dataset frame ids)
      images: list[np.ndarray or None]
      stats: dict
    """
    w_list, v_list, ids_list, img_list = [], [], [], []
    total = 0
    missing_wrench = 0
    missing_speed = 0
    missing_both = 0

    iterator = dataset if max_frames is None else (dataset[i] for i in range(min(len(dataset), max_frames)))
    n_iter = len(dataset) if max_frames is None else min(len(dataset), max_frames)

    # detect image key from first frame if possible
    first_frame = dataset[0]
    img_key = _find_image_key(first_frame)

    for local_i, frame in enumerate(tqdm(iterator, total=n_iter, desc="Collecting wrench/speed/image")):
        total += 1
        global_i = local_i  # if max_frames is None, same as dataset idx; if capped, still fine for diagnostics

        w = _extract_wrench_from_frame(frame, state_key=state_key)
        v = _extract_speed_from_frame(frame, state_key=state_key)

        if w is None and v is None:
            missing_both += 1
            continue
        if w is None:
            missing_wrench += 1
            continue
        if v is None:
            missing_speed += 1
            continue

        w_list.append(w)
        v_list.append(v)
        ids_list.append(global_i)
        img_list.append(_extract_image_from_frame(frame, img_key))

    if len(w_list) == 0:
        w_all = np.zeros((0, 6), dtype=np.float64)
        v_all = np.zeros((0, 6), dtype=np.float64)
    else:
        w_all = np.stack(w_list, axis=0)
        v_all = np.stack(v_list, axis=0)

    stats = {
        "total_frames": total,
        "usable_frames": int(w_all.shape[0]),
        "missing_wrench_frames": missing_wrench,
        "missing_speed_frames": missing_speed,
        "missing_both_frames": missing_both,
        "image_key": img_key,
    }
    return w_all, v_all, ids_list, img_list, stats


def plot_contact_histograms(
    w_all: np.ndarray,
    bins: int = 50,
    title: str = "Contact wrench distribution",
    force_x_idx: int = 0,
    force_y_idx: int = 1,
    torque_c_idx: int = 5,
    save_path: Optional[str] = None,
):
    if w_all.shape[0] == 0:
        raise RuntimeError("No usable samples found.")

    fx = w_all[:, force_x_idx]
    fy = w_all[:, force_y_idx]
    tc = w_all[:, torque_c_idx]

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), sharey=False)

    channels = [
        (fx, r"Force $F_x$ [N]", "Fx"),
        (fy, r"Force $F_y$ [N]", "Fy"),
        (tc, r"Torque $T_c$ [Nm]", "Tc"),
    ]

    soft_bounds = [5.0, 5.0, 0.5]
    print("\n=== Probability mass inside adaptive bounds ===")
    masses = {}

    for ax, (vals, xlabel, short_name), b in zip(axes, channels, soft_bounds):
        vals = np.asarray(vals, dtype=np.float64)
        n = len(vals)

        inside_mask = np.abs(vals) <= b
        inside_count = int(np.sum(inside_mask))
        p_inside = float(inside_count / max(n, 1))
        masses[short_name] = p_inside

        print(f"{short_name:>2} in [-{b:.3f}, {b:.3f}]: {inside_count:6d}/{n:<6d} = {100.0 * p_inside:6.2f} %")

        ax.hist(vals, bins=bins, density=False, alpha=0.85, edgecolor="grey", linewidth=0.4, color="orange")
        ax.axvspan(-b, b, alpha=0.12, color="orange")
        ax.axvline(-b, color="orange", linestyle="--", linewidth=1.0, alpha=0.9)
        ax.axvline(+b, color="orange", linestyle="--", linewidth=1.0, alpha=0.9)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.25)

        mu = float(np.mean(vals))
        sigma = float(np.std(vals))
        txt = f"μ = {mu:.3f}\nσ = {sigma:.3f}\nP(|x|≤{b:g}) = {100.0*p_inside:.1f}%"
        ax.text(
            0.97, 0.97, txt,
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.9)
        )

    fig.suptitle(title, y=1.03)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")
    return masses


def _top_outliers(vals: np.ndarray, top_n: int) -> np.ndarray:
    """Return indices of top-N by absolute value (descending)."""
    if len(vals) == 0:
        return np.array([], dtype=int)
    idx = np.argsort(np.abs(vals))[::-1]
    return idx[:top_n]


def print_top_outliers_and_show_images(
    w_all: np.ndarray,
    v_all: np.ndarray,
    frame_ids: List[int],
    images: List[Optional[np.ndarray]],
    torque_c_idx: int = 5,
    top_n: int = 8,
):
    channels = [
        ("Fx", 0, 5.0, "N"),
        ("Fy", 1, 5.0, "N"),
        ("Tc", torque_c_idx, 0.5, "Nm"),
    ]

    for name, axis_idx, bound, unit in channels:
        vals = w_all[:, axis_idx]
        idxs = _top_outliers(vals, top_n)

        print(f"\n=== Top {len(idxs)} outliers for {name} (sorted by |value|) ===")
        print(f"{'rank':>4}  {'frame':>8}  {'value':>10}  {'|x|/bound':>10}  {'vel':>10}")
        for rank, i in enumerate(idxs, start=1):
            # velocity axis mapping:
            # Fx -> vx (0), Fy -> vy (1), Tc(Tz default=5) -> wz (5)
            vel_axis = 0 if name == "Fx" else (1 if name == "Fy" else (axis_idx if axis_idx >= 3 else 5))
            print(
                f"{rank:4d}  {frame_ids[i]:8d}  {vals[i]:10.4f}  "
                f"{abs(vals[i]) / bound:10.3f}  {v_all[i, vel_axis]:10.4f}"
            )

        # Display associated images (if present)
        any_img = any(images[i] is not None for i in idxs)
        if not any_img:
            print(f"No images available for {name} outlier display (image key not found in dataset frames).")
            continue

        n = len(idxs)
        cols = min(4, n)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows))
        axes = np.atleast_1d(axes).reshape(rows, cols)

        for ax in axes.flat:
            ax.axis("off")

        for ax, i, rank in zip(axes.flat, idxs, range(1, n + 1)):
            img = images[i]
            if img is not None:
                if img.ndim == 2:
                    ax.imshow(img, cmap="gray")
                else:
                    ax.imshow(img)
            ax.set_title(
                f"#{rank} frame {frame_ids[i]}\n{name}={w_all[i, axis_idx]:.3f}",
                fontsize=8
            )
            ax.axis("off")

        fig.suptitle(f"Top {n} |{name}| outliers with images", y=0.98)
        plt.tight_layout()


def plot_axis_velocity_correlation_heatmaps(
    w_all: np.ndarray,
    v_all: np.ndarray,
    lag: int = 0,
    torque_c_idx: int = 5,
    bins: int = 80,
    save_path: Optional[str] = None,
):
    """
    Plot 2D hist heatmaps:
      Fx(t) vs vx(t-lag)
      Fy(t) vs vy(t-lag)
      Tc(t) vs wc(t-lag)
    backward shift by lag means velocity is taken from earlier samples.
    """
    if w_all.shape[0] == 0:
        raise RuntimeError("No usable samples found.")
    if lag < 0:
        raise ValueError("lag must be >= 0")

    N = w_all.shape[0]
    if lag >= N:
        raise ValueError(f"lag={lag} too large for N={N}")

    # align: force at t (later), velocity at t-lag (earlier)
    if lag == 0:
        w = w_all
        v = v_all
    else:
        w = w_all[lag:]
        v = v_all[:-lag]

    fx, fy, tc = w[:, 0], w[:, 1], w[:, torque_c_idx]
    vx, vy = v[:, 0], v[:, 1]
    wc = v[:, torque_c_idx] if torque_c_idx in (3, 4, 5) else v[:, 5]

    pairs = [
        (vx, fx, r"$v_x(t-\Delta)$ [m/s]", r"$F_x(t)$ [N]", "Fx vs lagged vx"),
        (vy, fy, r"$v_y(t-\Delta)$ [m/s]", r"$F_y(t)$ [N]", "Fy vs lagged vy"),
        (wc, tc, r"$\omega_c(t-\Delta)$ [rad/s]", r"$T_c(t)$ [Nm]", "Tc vs lagged ωc"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
    for ax, (x, y, xlabel, ylabel, title) in zip(axes, pairs):
        h = ax.hist2d(x, y, bins=bins, cmap="magma")
        plt.colorbar(h[3], ax=ax, fraction=0.046, pad=0.04, label="Count")

        # Pearson correlation
        if len(x) > 1 and np.std(x) > 1e-12 and np.std(y) > 1e-12:
            rho = float(np.corrcoef(x, y)[0, 1])
        else:
            rho = np.nan

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.15)

        ax.text(
            0.03, 0.97,
            f"lag = {lag} frames\nρ = {rho:.3f}",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.9)
        )

    fig.suptitle("Axis force/torque vs lagged axis velocity (2D histograms)", y=1.03)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved correlation heatmaps to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Histogram + outlier inspection + lagged force/velocity correlation for LeRobot dataset.")
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--torque-c-idx", type=int, default=5, choices=[3, 4, 5])
    parser.add_argument("--state-key", type=str, default="observation.state")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--save", type=str, default=None, help="Histogram figure path")
    parser.add_argument("--lag", type=int, default=0, help="Backward shift for velocity in correlation plots (frames)")
    parser.add_argument("--corr-bins", type=int, default=80, help="Bins for 2D hist correlation plots")
    parser.add_argument("--top-n", type=int, default=8, help="Top N outliers per axis to inspect")
    parser.add_argument("--save-corr", type=str, default=None, help="Correlation heatmap figure path")
    args = parser.parse_args()

    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root)

    w_all, v_all, frame_ids, images, stats = _collect_dataset_signals(
        dataset=dataset,
        state_key=args.state_key,
        max_frames=args.max_frames,
    )

    print("\n=== Dataset signal stats ===")
    print(f"Total frames:            {stats['total_frames']}")
    print(f"Usable frames:           {stats['usable_frames']}")
    print(f"Missing wrench frames:   {stats['missing_wrench_frames']}")
    print(f"Missing speed frames:    {stats['missing_speed_frames']}")
    print(f"Missing both frames:     {stats['missing_both_frames']}")
    print(f"Detected image key:      {stats['image_key']}")

    if w_all.shape[0] == 0:
        print("No usable frames with wrench+speed found.")
        return

    fx = w_all[:, 0]
    fy = w_all[:, 1]
    tc = w_all[:, args.torque_c_idx]

    joint_mask = (np.abs(fx) <= 5.0) & (np.abs(fy) <= 5.0) & (np.abs(tc) <= 0.5)
    p_joint = float(np.mean(joint_mask))
    print(f"\nJoint mass (Fx,Fy,Tc all inside bounds): {100.0 * p_joint:.2f} %")

    names = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
    print("\n=== Wrench summary (mean ± std) ===")
    for i, n in enumerate(names):
        print(f"{n:>2}: {np.mean(w_all[:, i]): .4f} ± {np.std(w_all[:, i]):.4f}")

    _ = plot_contact_histograms(
        w_all=w_all,
        bins=args.bins,
        torque_c_idx=args.torque_c_idx,
        save_path=args.save,
    )

    print_top_outliers_and_show_images(
        w_all=w_all,
        v_all=v_all,
        frame_ids=frame_ids,
        images=images,
        torque_c_idx=args.torque_c_idx,
        top_n=args.top_n,
    )

    plot_axis_velocity_correlation_heatmaps(
        w_all=w_all,
        v_all=v_all,
        lag=args.lag,
        torque_c_idx=args.torque_c_idx,
        bins=args.corr_bins,
        save_path=args.save_corr,
    )

    plt.show()


if __name__ == "__main__":
    main()
