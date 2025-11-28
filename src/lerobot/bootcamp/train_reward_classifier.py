#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Offline Reward Classifier Training Script (VLM-Embedding Mode).

This script trains a reward classifier model on a static LeRobotDataset,
using the VLM embedding (observation.state) as the input feature.
It uses a training structure similar to the offline RL script.
"""

import logging
import os
import time
from pathlib import Path
from pprint import pformat
from collections.abc import Iterator
from typing import TypedDict

import torch
from termcolor import colored
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.bootcamp.vlm_classifier import VLMClassifier # <-- Assuming VLMClassifier is the final implementation
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.sac.reward_model.configuration_classifier import RewardClassifierConfig
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.share.configs import TrainRLServerPipelineConfig
from lerobot.utils.constants import (
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    TRAINING_STATE_DIR,
    REWARD,
    OBS_STATE, # Use OBS_STATE for VLM embedding input
)
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state as utils_load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    init_logging,
)
from lerobot.bootcamp.train_rlpd import cfg
from lerobot.bootcamp.xvla_client import INSTRUCTION, EMBEDDING_DIM

# --- Type Definitions ---

class BatchClassification(TypedDict):
    state: dict[str, torch.Tensor]
    reward: torch.Tensor # Labels (e.g., [0.0, 1.0, 0.0, ...])


# --- Core Data Loading Functions ---

def create_dataloader_iterator(dataset: LeRobotDataset, batch_size: int, num_workers: int, shuffle: bool = True) -> Iterator:
    """Creates a DataLoader and returns its infinite iterator for classification."""
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return iter(dataloader)

def infinite_data_iterator(dataset: LeRobotDataset, batch_size: int, num_workers: int) -> Iterator:
    """
    Creates an infinite iterator over a LeRobotDataset by recreating the DataLoader
    iterator whenever it is exhausted.
    """
    iterator = create_dataloader_iterator(dataset, batch_size, num_workers)
    while True:
        try:
            # NOTE: We only need the current state and reward for classification
            batch = next(iterator)
            # Filter the batch to only include necessary keys (VLM embedding and reward label)

            # Identify the VLM embedding input key
            state_key = OBS_STATE

            # Create the input dictionary: { 'state': { OBS_STATE: VLM_TENSOR }, 'reward': REWARD_TENSOR }
            input_batch = BatchClassification(
                # Ensure the input dictionary key matches what the VLMEmbeddingEncoder expects
                state={state_key: batch[state_key]},
                reward=batch.get(REWARD, torch.tensor([0.0] * batch[state_key].shape[0])).float() # Labels must be float for BCE loss
            )
            yield input_batch
        except StopIteration:
            logging.info(f"Restarting infinite iterator for dataset {dataset.repo_id}")
            iterator = create_dataloader_iterator(dataset, batch_size, num_workers)

def load_static_datasets(
    cfg: TrainRLServerPipelineConfig
) -> tuple[Iterator, LeRobotDataset]:
    """
    Loads a single LeRobotDataset and prepares an infinite iterator for training the classifier.
    """
    repo_id = cfg.dataset.repo_id

    # 1. Load Dataset (Expert demonstrations provide the ground truth for reward)
    logging.info(f"Loading Classifier Training Dataset: {repo_id}")
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=Path(cfg.dataset.root) / repo_id,
    )

    batch_size = cfg.batch_size
    num_workers = cfg.num_workers

    # 2. Create infinite iterator
    data_iterator = infinite_data_iterator(
        dataset, batch_size, num_workers
    )

    return data_iterator, dataset


# --- Core Algorithm Functions ---

def make_optimizer(cfg: TrainRLServerPipelineConfig, model: nn.Module) -> Optimizer:
    """Creates the Adam optimizer for the classifier model."""
    lr = getattr(cfg.classifier, 'learning_rate', cfg.policy.actor_lr) if hasattr(cfg, 'classifier') else cfg.policy.actor_lr
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    return optimizer

def initialize_classifier(cfg: TrainRLServerPipelineConfig) -> VLMClassifier:
    """Initializes the reward classifier model."""
    # Assuming classifier config is passed in cfg.classifier
    
    # Re-use the feature definition from the policy for the classifier
    classifier_config = RewardClassifierConfig(
        # Pass the feature dictionary so the classifier knows the shape of OBS_STATE
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(EMBEDDING_DIM, ))}, 
        latent_dim=256, 
        hidden_dim=256, 
        num_classes=2,
    )

    classifier: VLMClassifier = VLMClassifier(config=classifier_config)

    return classifier

def run_classifier_training(
    cfg: TrainRLServerPipelineConfig,
    wandb_logger: WandBLogger | None,
):
    """
    Offline training loop for the Reward Classifier using a static LeRobotDataset.
    """
    # Extract all configuration variables
    device = get_safe_torch_device(try_device=cfg.policy.device, log=True)
    log_freq = cfg.log_freq
    save_freq = cfg.save_freq
    max_steps = cfg.policy.online_steps # Reusing this for total optimization steps

    # Initialize classifier and optimizer
    logging.info("Initializing Reward Classifier")
    classifier: VLMClassifier = initialize_classifier(cfg).to(device)
    classifier.train()
    optimizer = make_optimizer(cfg=cfg, model=classifier)

    # Load training state if resuming
    resume_optimization_step, _ = load_training_state(cfg=cfg, optimizers={"classifier": optimizer})
    optimization_step = resume_optimization_step if resume_optimization_step is not None else 0

    log_training_info(cfg=cfg, model=classifier)  # Reusing log_training_info

    # --- OFFLINE DATA SETUP ---
    data_iterator, dataset = load_static_datasets(cfg)

    total_dataset_size = len(dataset)
    min_samples_for_training = cfg.batch_size

    if total_dataset_size < min_samples_for_training:
        logging.error(f"[CLASSIFIER] Not enough data. Need {min_samples_for_training} samples, found {total_dataset_size}.")
        return

    logging.info(f"[CLASSIFIER] Starting offline training on {total_dataset_size} total samples.")

    # NOTE: THIS IS THE MAIN CLASSIFIER TRAINING LOOP
    while optimization_step < max_steps:
        time_for_one_optimization_step = time.time()

        # --- 1. Sample Batch (s_t, r_t) ---
        batch = next(data_iterator)

        # Move batch to device
        observations = batch["state"]
        labels = batch["reward"]

        # NOTE: Observations contains {OBS_STATE: VLM_TENSOR}. Move manually.
        observations = {k: v.to(device) for k, v in observations.items()}
        labels = labels.to(device)

        # --- 2. Forward Pass and Loss Calculation ---
        
        # The classifier forward method expects a dictionary with the state and reward label
        # We pass the input state dict and the reward labels
        # NOTE: We assume VLMClassifier's forward method is compatible with the general API:
        # loss, metrics = model.forward(batch={'state': ..., REWARD: labels, ACTION: optional_action})
        observations[REWARD] = labels
        loss, metrics = classifier.forward(observations)

        # --- 3. Optimization Step ---
        optimizer.zero_grad()
        loss.backward()
        # NOTE: No grad clipping needed for a simple classifier head, but kept for consistency
        # torch.nn.utils.clip_grad_norm_(parameters=classifier.parameters(), max_norm=cfg.policy.grad_clip_norm)
        optimizer.step()

        # --- 4. Logging and Checkpointing ---

        if optimization_step % log_freq == 0:
            training_infos = {
                "loss_classifier": loss.item(),
                "accuracy": metrics["accuracy"],
                "total_samples": total_dataset_size,
                "Optimization step": optimization_step,
            }

            if wandb_logger:
                wandb_logger.log_dict(d=training_infos, mode="train", custom_step_key="Optimization step")

            # Log frequency
            time_for_one_optimization_step = time.time() - time_for_one_optimization_step
            frequency_for_one_optimization_step = 1 / (time_for_one_optimization_step + 1e-9)

            if wandb_logger:
                wandb_logger.log_dict(
                    {"Optimization frequency loop [Hz]": frequency_for_one_optimization_step, "Optimization step": optimization_step},
                    mode="train", custom_step_key="Optimization step",
                )
            logging.info(f"[CLASSIFIER] Step {optimization_step}, Loss: {loss.item():.4f}, Accuracy: {metrics['accuracy']:.2f}%, Freq: {frequency_for_one_optimization_step:.2f} Hz")


        # Save checkpoint at specified intervals
        if optimization_step % save_freq == 0 or optimization_step == max_steps - 1:
            save_training_checkpoint_classifier(
                cfg=cfg,
                optimization_step=optimization_step,
                max_steps=max_steps,
                classifier=classifier,
                optimizer=optimizer,
            )

        optimization_step += 1

    logging.info("[CLASSIFIER] Offline training finished.")


# --- Simplified Setup and Utility Functions ---

def save_training_checkpoint_classifier(
    cfg: TrainRLServerPipelineConfig,
    optimization_step: int,
    max_steps: int,
    classifier: nn.Module,
    optimizer: Optimizer,
) -> None:
    """Saves the classifier training checkpoint."""
    logging.info(f"Checkpoint classifier after step {optimization_step}")

    # Reuse RL step checkpoint logic structure
    checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, max_steps, optimization_step)

    # Note: save_checkpoint expects 'policy' and an optimizer dictionary
    save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        step=optimization_step,
        cfg=cfg,
        policy=classifier,
        optimizer={"classifier": optimizer},
        scheduler=None,
    )

    # Save optimization step manually
    training_state_dir = os.path.join(checkpoint_dir, TRAINING_STATE_DIR)
    os.makedirs(training_state_dir, exist_ok=True)
    training_state = {"step": optimization_step}
    torch.save(training_state, os.path.join(training_state_dir, "training_state.pt"))

    update_last_checkpoint(checkpoint_dir)

    logging.info(f"Checkpoint saved to {checkpoint_dir}")


# --- Reusing / Simplified Utilities from RL Script ---

def handle_resume_logic(cfg: TrainRLServerPipelineConfig) -> TrainRLServerPipelineConfig:
    """Handle resume logic (simplified)."""
    out_dir = cfg.output_dir

    if not cfg.resume:
        checkpoint_dir = os.path.join(out_dir, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK)
        if os.path.exists(checkpoint_dir):
            raise RuntimeError(
                f"Output directory {checkpoint_dir} already exists. Use `resume=true` to resume training."
            )
        return cfg

    checkpoint_dir = os.path.join(out_dir, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK)
    if not os.path.exists(checkpoint_dir):
        raise RuntimeError(f"No model checkpoint found in {checkpoint_dir} for resume=True")

    logging.info(colored("Valid checkpoint found, resuming previous run", color="yellow", attrs=["bold"]))

    checkpoint_cfg_path = os.path.join(checkpoint_dir, "pretrained_model", "train_config.json")
    checkpoint_cfg = TrainRLServerPipelineConfig.from_pretrained(checkpoint_cfg_path)

    checkpoint_cfg.resume = True
    return checkpoint_cfg


def load_training_state(
    cfg: TrainRLServerPipelineConfig,
    optimizers: Optimizer | dict[str, Optimizer],
):
    """Loads the training state (optimizers, step count, etc.) from a checkpoint (simplified)."""
    if not cfg.resume:
        return None, None

    checkpoint_dir = os.path.join(cfg.output_dir, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK)

    try:
        step, optimizers, _ = utils_load_training_state(Path(checkpoint_dir), optimizers, None)

        training_state_path = os.path.join(checkpoint_dir, TRAINING_STATE_DIR, "training_state.pt")
        interaction_step = 0
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, weights_only=False) # nosec B614: Safe usage of torch.load
            interaction_step = training_state.get("interaction_step", 0)

        logging.info(f"Resuming from optimization step {step}")
        return step, interaction_step

    except Exception as e:
        logging.error(f"Failed to load training state: {e}")
        return None, None


def log_training_info(cfg: TrainRLServerPipelineConfig, model: nn.Module) -> None:
    """Log information about the training process."""
    num_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in model.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    logging.info(f"{cfg.env.task=}")
    logging.info(f"Max steps: {cfg.policy.online_steps}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")


def train_cli():
    global cfg

    # Use the job_name from the config
    train(cfg, job_name=cfg.job_name + "_classifier")

    logging.info("[CLASSIFIER] train_cli finished")


def train(cfg: TrainRLServerPipelineConfig, job_name: str | None = None):
    """
    Main training function that initializes and runs the training process.
    """
    cfg.validate()

    if job_name is None:
        job_name = cfg.job_name

    if job_name is None:
        raise ValueError("Job name must be specified either in config or as a parameter")

    # Create logs directory
    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"classifier_{job_name}.log")

    init_logging(log_file=log_file, display_pid=False)
    logging.info(f"Classifier logging initialized, writing to {log_file}")
    logging.info(pformat(cfg.to_dict()))

    # Setup WandB logging if enabled
    if cfg.wandb.enable and cfg.wandb.project:
        from lerobot.rl.wandb_utils import WandBLogger

        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    cfg = handle_resume_logic(cfg)
    set_seed(seed=cfg.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # START OF OFFLINE TRAINING
    run_classifier_training(
        cfg=cfg,
        wandb_logger=wandb_logger,
    )

if __name__ == "__main__":
    # NOTE: Placeholder for running the script outside a full system.
    train_cli()