#!/usr/bin/env python
"""
Script: eval_one_episode.py
Description:
    Loads a dataset and trained policy, runs inference on one episode,
    and plots ground-truth vs predicted actions for each action dimension.
"""

import logging
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from pprint import pformat
import numpy as np
from tqdm import tqdm

from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import MultiLeRobotDataset, LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor.rename_processor import rename_stats
from lerobot.utils.constants import ACTION
from lerobot.utils.utils import init_logging, get_safe_torch_device
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig


@torch.no_grad()
def run_inference_on_episode(dataset: LeRobotDataset, policy: PreTrainedPolicy, preprocessor, postprocessor, device="cuda"):
    """
    Run policy inference on one episode and return ground truth vs predicted actions.
    """
    gt_actions = []
    pred_actions = []

    for frame in tqdm(dataset):
        obs = {k: torch.as_tensor(v).unsqueeze(0).to(device) for k, v in frame.items() if k in policy.config.input_features}
        obs = preprocessor(obs)
        out = policy.select_action(obs)
        out = postprocessor(out)
        pred_actions.append(out.cpu().squeeze(0))
        gt_actions.append(frame[ACTION].cpu().squeeze(0))

    return torch.stack(gt_actions), torch.stack(pred_actions)


def plot_predictions(gt_actions: torch.Tensor, pred_actions: torch.Tensor, save_dir: Path):
    """
    Generates one plot per action dimension showing ground truth vs predicted values over time.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    T, dim = gt_actions.shape
    t = np.arange(T)

    for d in range(dim):
        plt.figure(figsize=(8, 3))
        plt.plot(t, gt_actions[:, d], label="Ground Truth", linewidth=2)
        plt.plot(t, pred_actions[:, d], label="Prediction", linewidth=2, linestyle="--")
        plt.xlabel("Timestep")
        plt.ylabel(f"Action[{d}]")
        plt.title(f"Action dimension {d}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / f"action_dim_{d}.png")
        plt.close()
    logging.info(f"Saved {dim} plots to {save_dir}")


@parser.wrap()
def eval_one_episode(cfg: TrainPipelineConfig):
    """
    Evaluation entrypoint — uses same config format as train.py for convenience.
    """
    episode_idx = 0

    init_logging()
    logging.info(pformat(cfg.to_dict()))
    cfg.validate()

    device = get_safe_torch_device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    #dataset = make_dataset(cfg)
    dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=[episode_idx]
    )
    if isinstance(dataset, MultiLeRobotDataset):
        dataset = dataset._datasets[0]
    ds_meta = dataset.meta
    ds_stats = ds_meta.stats

    # Load trained policy
    logging.info("Loading trained policy")
    policy = make_policy(cfg.policy, ds_meta=ds_meta, rename_map=cfg.rename_map)
    policy.to(device)
    policy.eval()

    # Pre/Post processors
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=rename_stats(ds_stats, cfg.rename_map),
        preprocessor_overrides={
            "device_processor": {"device": device.type},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )

    # Run inference on one episode (hardcoded index)
    gt_actions, pred_actions = run_inference_on_episode(
        dataset, policy, preprocessor, postprocessor, device
    )

    # Plot results
    save_dir = Path(cfg.output_dir) / f"eval_episode_{episode_idx}"
    plot_predictions(gt_actions, pred_actions, save_dir)


if __name__ == "__main__":
    import experiments  # ensures config store is loaded
    eval_one_episode()
