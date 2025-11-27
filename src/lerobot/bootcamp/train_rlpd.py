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
Offline Learner Script for Residual RL Fine-Tuning.

This script implements an **offline** training loop (RLPD style) that samples
from static LeRobotDatasets (expert demonstrations + on-policy rollouts).
It removes all gRPC, queue management, and asynchronous actor communication,
simplifying the process for batch fine-tuning.
"""

import logging
import os
import time
from pathlib import Path
from pprint import pformat
from typing import TypedDict, Sequence
from collections.abc import Iterator
from pathlib import Path

import torch

from lerobot.envs import AlohaEnv
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.sac.configuration_sac import SACConfig
from termcolor import colored
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.rl.buffer import concatenate_batch_transitions
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_record import DatasetRecordConfig
from lerobot.share.configs import TrainRLServerPipelineConfig
from lerobot.utils.constants import (
    ACTION,
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    TRAINING_STATE_DIR,
    OBS_STATE,
    REWARD,
    DONE,
)
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state as utils_load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.transition import move_state_dict_to_device, move_transition_to_device
from lerobot.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    init_logging,
)
from lerobot.bootcamp.xvla_client import INSTRUCTION, EMBEDDING_DIM
from lerobot.bootcamp.utils import make_sac
from lerobot.bootcamp.preprocess_embeddings import VLA_ACTION_KEY

# --- Configuration Constants (Simplified) ---
cfg = TrainRLServerPipelineConfig(
    dataset=DatasetRecordConfig(
        single_task=INSTRUCTION,
        repo_id="lerobot/svla_so101_pickplace_vlm_states",
        root="/home/hk-project-pai00093/id_hqc2052/.cache/huggingface/lerobot"
    ),
    env=AlohaEnv(
        features={
            "observation.state": PolicyFeature(shape=(EMBEDDING_DIM,), type=FeatureType.STATE),
            "action": PolicyFeature(shape=(6,), type=FeatureType.ACTION),
        }
    ),
    policy=SACConfig(),
    output_dir="",
    resume=False,
    batch_size=16,
    job_name="test",
    num_workers=1
)
online_rollout_repo_id = "lerobot/svla_so101_pickplace_vlm_states"

# --- Type Definitions for Batched Transitions ---

class BatchTransition(TypedDict):
    state: dict[str, torch.Tensor]
    action: torch.Tensor
    vla_action: torch.Tensor
    reward: torch.Tensor
    next_state: dict[str, torch.Tensor]
    done: torch.Tensor
    truncated: torch.Tensor
    complementary_info: dict[str, torch.Tensor | float | int] | None


# --- Dataset Wrapper for RL Transitions ---

class RLDatasetWrapper(Dataset):
    """
    Wraps a LeRobotDataset to correctly yield RL transitions (s_t, a_t, r_t, s_{t+1}, done).
    
    This version is optimized to avoid looping over the dataset in __init__.
    """

    def __init__(self, dataset: LeRobotDataset, state_keys: Sequence[str]):
        self.dataset = dataset
        self.state_keys = state_keys
        self.num_frames = len(self.dataset)
        
        # Cache the episode indices for fast lookups in __getitem__
        # This is much faster than looping or indexing the dataset object repeatedly.
        try:
            self.episode_indices = self.dataset.hf_dataset["episode_index"]
            logging.info(f"RLDatasetWrapper initialized with {self.num_frames} total frames.")
        except Exception as e:
            logging.error(f"Failed to cache episode_index: {e}")
            logging.error("Falling back to slow, iterative lookup.")
            self.episode_indices = None

    def __len__(self):
        """
        The total number of transitions we can sample is equal to the
        total number of frames. Each frame is the start of one transition,
        either terminal or non-terminal.
        """
        return self.num_frames

    def __getitem__(self, idx: int) -> dict:
        """Returns a single transition dictionary (not batched)."""
        current_sample = self.dataset[idx]

        # 1. State (Images, Proprio) and Action
        current_state = {key: current_sample[key] for key in self.state_keys}
        action = current_sample[ACTION]
        vla_action = current_sample[VLA_ACTION_KEY]

        # 2. Reward, Done, Truncated
        reward = current_sample.get(REWARD, torch.tensor([0.0]))
        # We will determine 'done' and 'truncated' based on episode boundaries.
        
        # --- 3. Determine Next State (s_{t+1}) and terminal flag ---
        is_terminal_frame = False
        
        if idx == self.num_frames - 1:
            # This is the very last frame in the dataset.
            is_terminal_frame = True
        else:
            # Check if the next frame is in a different episode.
            if self.episode_indices:
                # Fast path: use cached indices
                if self.episode_indices[idx] != self.episode_indices[idx+1]:
                    is_terminal_frame = True
            else:
                # Slow path: index the dataset object
                if current_sample["episode_index"] != self.dataset[idx+1]["episode_index"]:
                    is_terminal_frame = True

        if not is_terminal_frame:
            # Normal transition: s_{t+1} is the state at index idx+1
            next_sample = self.dataset[idx + 1]
            next_state = {key: next_sample[key] for key in self.state_keys}
            done = current_sample.get(DONE, torch.tensor([False]))
            truncated = torch.tensor([False])
        else:
            # Terminal transition: s_{t+1} = s_t, done = True
            next_state = current_state
            done = torch.tensor([True])
            truncated = torch.tensor([True])

        # 4. Complementary Info
        comp_info_prefix = "complementary_info."
        complementary_info = {
            k[len(comp_info_prefix):]: v
            for k, v in current_sample.items()
            if k.startswith(comp_info_prefix)
        }
        complementary_info = complementary_info if complementary_info else None
        
        # 5. Language Instruction (Placeholder)
        # TODO: This must be loaded from your dataset if it changes per sample
        instruction = "Move the gripper to the target position"

        # 6. Unsqueeze tensors
        for key in current_state:
            current_state[key] = current_state[key]
            next_state[key] = next_state[key]
        action = action
        reward = reward
        done = done
        truncated = truncated

        return {
            "state": current_state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "truncated": truncated,
            "complementary_info": complementary_info,
            "instruction": instruction, # Added for VLA client
            "vla_action": vla_action
        }

# --- Collate Function ---

def rl_collate_fn(batch: list[dict]) -> BatchTransition:
    """
    Custom collate function to correctly batch list of transition dictionaries.
    Handles the nested dicts for 'state' and 'next_state'.
    """
    if not batch:
        return {}

    first_item = batch[0]

    # Collate state and next_state dictionaries
    def collate_nested_dict(key: str) -> dict:
        # Assumes all items in the batch have the same keys in their state/next_state dicts
        nested_dict = {}
        for state_key in first_item[key].keys():
            tensors = [item[key][state_key] for item in batch]
            nested_dict[state_key] = torch.stack(tensors, dim=0)
        return nested_dict

    batch_state = collate_nested_dict("state")
    batch_next_state = collate_nested_dict("next_state")

    # Collate simple tensor fields
    batch_action = torch.stack([item["action"] for item in batch], dim=0)
    batch_vla_action = torch.stack([item["vla_action"] for item in batch], dim=0)
    batch_reward = torch.stack([item["reward"] for item in batch], dim=0)
    batch_done = torch.stack([item["done"] for item in batch], dim=0)
    batch_truncated = torch.stack([item["truncated"] for item in batch], dim=0)

    # Collate complementary_info (if present and homogeneous)
    batch_complementary_info = None
    if first_item["complementary_info"] is not None:
        batch_complementary_info = {}
        info_keys = first_item["complementary_info"].keys()
        for key in info_keys:
            # Concatenate all tensors for this info key
            tensors = [item["complementary_info"][key] for item in batch]
            batch_complementary_info[key] = torch.stack(tensors, dim=0)

    return BatchTransition(
        state=batch_state,
        action=batch_action,
        vla_action=batch_vla_action,
        reward=batch_reward,
        next_state=batch_next_state,
        done=batch_done,
        truncated=batch_truncated,
        complementary_info=batch_complementary_info,
    )


# --- Core Algorithm Functions ---

def create_dataloader_iterator(dataset, batch_size, num_workers, shuffle=True) -> Iterator:
    """
    Creates a DataLoader and returns its infinite iterator.

    NOTE: The DataLoader now uses the custom RLDatasetWrapper and collate_fn.
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=rl_collate_fn, # Use custom collate function
    )
    return iter(dataloader)

def infinite_data_iterator(dataset: LeRobotDataset, batch_size: int, num_workers: int) -> Iterator:
    """
    Creates an infinite iterator over a LeRobotDataset by recreating the DataLoader
    iterator whenever it is exhausted.
    """
    # Wrap the LeRobotDataset to construct the RL transitions
    # Since we are in VLM embedding mode, we only expect 'observation.state' (VLM embedding)
    state_keys = ["observation.state"]
    wrapped_dataset = RLDatasetWrapper(dataset, state_keys=state_keys)

    iterator = create_dataloader_iterator(wrapped_dataset, batch_size, num_workers)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            logging.info(f"Restarting infinite iterator for dataset {dataset.repo_id}")
            # The wrapped dataset is already prepared, just recreate the DataLoader iterator
            iterator = create_dataloader_iterator(wrapped_dataset, batch_size, num_workers)


def load_static_datasets(
    cfg: TrainRLServerPipelineConfig
) -> tuple[
    tuple[Iterator, Iterator], # (offline_iterator, online_iterator)
    tuple[LeRobotDataset, LeRobotDataset] # (offline_dataset, online_rollout_dataset)
]:
    """
    Loads two separate LeRobotDatasets (Expert and On-Policy Rollouts)
     and prepares infinite iterators for 50/50 batch sampling.

    Args:
        cfg: Configuration object containing dataset information.

    Returns:
        tuple: ((offline_iterator, online_iterator), (offline_dataset, online_rollout_dataset))
    """
    offline_repo_id = cfg.dataset.repo_id

    if not online_rollout_repo_id:
        raise ValueError("Configuration must specify 'online_rollout_repo_id' for 50/50 training.")

    # 1. Load Datasets
    logging.info(f"Loading Expert Dataset: {offline_repo_id}")
    offline_dataset = LeRobotDataset(
        repo_id=offline_repo_id,
        root=Path(cfg.dataset.root) / offline_repo_id,
    )

    logging.info(f"Loading Rollout Dataset: {online_rollout_repo_id}")
    online_rollout_dataset = LeRobotDataset(
        repo_id=online_rollout_repo_id,
        root=Path(cfg.dataset.root) / online_rollout_repo_id,
    )

    batch_size_half = cfg.batch_size // 2
    num_workers = cfg.num_workers

    # 2. Create infinite iterators for 50/50 sampling
    offline_iterator = infinite_data_iterator(
        offline_dataset, batch_size_half, num_workers
    )
    online_iterator = infinite_data_iterator(
        online_rollout_dataset, batch_size_half, num_workers
    )

    return (offline_iterator, online_iterator), (offline_dataset, online_rollout_dataset)


def run_offline_training(
    cfg: TrainRLServerPipelineConfig,
    wandb_logger: WandBLogger | None,
):
    """
    Offline training loop using static LeRobotDatasets.
    """
    # Extract all configuration variables at the beginning
    device = get_safe_torch_device(try_device=cfg.policy.device, log=True)
    clip_grad_norm_value = cfg.policy.grad_clip_norm
    utd_ratio = cfg.policy.cta_ratio
    log_freq = cfg.log_freq
    save_freq = cfg.save_freq
    policy_update_freq = cfg.policy.policy_update_freq
    online_steps = cfg.policy.online_steps # Renamed to MAX_OPT_STEPS

    # --- OFFLINE DATA SETUP ---
    # Load datasets and iterators
    (offline_iterator, online_iterator), (offline_dataset, online_rollout_dataset) = load_static_datasets(cfg)

    total_dataset_size = len(offline_dataset) + len(online_rollout_dataset)
    min_samples_for_training = cfg.batch_size

    # Initialize policy and optimizers
    logging.info("Initializing policy")
    policy: SACPolicy = make_sac(sac_cfg=cfg.policy, ds_meta=offline_dataset.meta).to(device)
    policy.train()
    optimizers, _ = make_optimizers_and_scheduler(cfg=cfg, policy=policy)

    # Load training state if resuming
    resume_optimization_step, _ = load_training_state(cfg=cfg, optimizers=optimizers)
    optimization_step = resume_optimization_step if resume_optimization_step is not None else 0

    log_training_info(cfg=cfg, policy=policy)

    if total_dataset_size < min_samples_for_training:
        logging.error(f"[LEARNER] Not enough data. Need {min_samples_for_training} samples, found {total_dataset_size}.")
        return

    logging.info(f"[LEARNER] Starting offline training on {total_dataset_size} total samples.")
    logging.info(f"[LEARNER] Expert size: {len(offline_dataset)}, Rollout size: {len(online_rollout_dataset)}")

    # NOTE: THIS IS THE MAIN OFFLINE LOOP
    while optimization_step < online_steps:
        time_for_one_optimization_step = time.time()

        # UTD Loop for Critic/Target updates
        for _ in range(utd_ratio):
            # --- 1. Sample from both infinite iterators (50/50) ---

            # Sample 50% from Expert demos
            batch_expert = next(offline_iterator)
            # Sample 50% from Rollout data (on-policy)
            batch_rollout = next(online_iterator)

            # --- 2. Concatenate the two batches ---
            # This creates the full batch of size cfg.batch_size
            batch = concatenate_batch_transitions(
                left_batch_transitions=batch_expert,
                right_batch_transition=batch_rollout
            )

            # Move batch to device
            batch = move_transition_to_device(batch, device)

            # --- Common training steps ---
            actions = batch[ACTION]
            rewards = batch["reward"]
            observations = batch["state"]
            next_observations = batch["next_state"] # NOTE: next_state is correctly available here
            done = batch["done"]

            # Create a batch dictionary with all required elements for the forward method
            forward_batch = {
                ACTION: actions,
                "reward": rewards,
                "state": observations,
                "next_state": next_observations,
                "done": done.to(dtype=torch.int).squeeze(),
                "observation_feature": None,
                "next_observation_feature": None,
                "complementary_info": batch.get("complementary_info", {}),
                "vla_action": batch[VLA_ACTION_KEY]
            }

            # 1. Critic Optimization (UTD * N)
            critic_output = policy.forward(forward_batch, model="critic")
            loss_critic = critic_output["loss_critic"]
            optimizers["critic"].zero_grad()
            loss_critic.backward()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters=policy.critic_ensemble.parameters(), max_norm=clip_grad_norm_value
            )
            optimizers["critic"].step()

            # Discrete critic optimization (if available)
            if policy.config.num_discrete_actions is not None:
                discrete_critic_output = policy.forward(forward_batch, model="discrete_critic")
                loss_discrete_critic = discrete_critic_output["loss_discrete_critic"]
                optimizers["discrete_critic"].zero_grad()
                loss_discrete_critic.backward()
                torch.nn.utils.clip_grad_norm_(
                    parameters=policy.discrete_critic.parameters(), max_norm=clip_grad_norm_value
                )
                optimizers["discrete_critic"].step()

            # Update target networks (main and discrete)
            policy.update_target_networks()

            # 2. Actor and Temperature Optimization (Only once per policy_update_freq)
            if optimization_step % policy_update_freq == 0:
                # Actor optimization
                actor_output = policy.forward(forward_batch, model="actor")
                loss_actor = actor_output["loss_actor"]
                optimizers["actor"].zero_grad()
                loss_actor.backward()
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=policy.actor.parameters(), max_norm=clip_grad_norm_value
                ).item()
                optimizers["actor"].step()

                # Temperature optimization
                temperature_output = policy.forward(forward_batch, model="temperature")
                loss_temperature = temperature_output["loss_temperature"]
                optimizers["temperature"].zero_grad()
                loss_temperature.backward()
                torch.nn.utils.clip_grad_norm_(
                    parameters=[policy.log_alpha], max_norm=clip_grad_norm_value
                ).item()
                optimizers["temperature"].step()

                # Update temperature
                policy.update_temperature()

        # Log training metrics
        if optimization_step % log_freq == 0:
            training_infos = {
                "loss_critic": loss_critic.item(),
                "critic_grad_norm": critic_grad_norm.item(),
                "loss_actor": loss_actor.item(),
                "actor_grad_norm": actor_grad_norm,
                "loss_temperature": loss_temperature.item(),
                "temperature": policy.temperature,
                "dataset_size_total": total_dataset_size,
                "dataset_size_expert": len(offline_dataset),
                "dataset_size_rollout": len(online_rollout_dataset),
                "Optimization step": optimization_step,
            }
            if policy.config.num_discrete_actions is not None:
                training_infos["loss_discrete_critic"] = loss_discrete_critic.item()

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
            logging.info(f"[LEARNER] Step {optimization_step}, Loss: {loss_critic.item():.4f}, Freq: {frequency_for_one_optimization_step:.2f} Hz")

        # Save checkpoint at specified intervals
        if optimization_step % save_freq == 0 or optimization_step == online_steps - 1:
            save_training_checkpoint(
                cfg=cfg,
                optimization_step=optimization_step,
                online_steps=online_steps,
                policy=policy,
                optimizers=optimizers,
            )

        optimization_step += 1

    logging.info("[LEARNER] Offline training finished.")


# --- Simplified Setup and Utility Functions ---

def save_training_checkpoint(
    cfg: TrainRLServerPipelineConfig,
    optimization_step: int,
    online_steps: int,
    policy: nn.Module,
    optimizers: dict[str, Optimizer],
    # Removed interaction_message, replay_buffer, etc.
) -> None:
    """Save training checkpoint (simplified)."""
    logging.info(f"Checkpoint policy after step {optimization_step}")

    checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, online_steps, optimization_step)

    save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        step=optimization_step,
        cfg=cfg,
        policy=policy,
        optimizer=optimizers,
        scheduler=None,
    )

    # Save optimization step manually
    training_state_dir = os.path.join(checkpoint_dir, TRAINING_STATE_DIR)
    os.makedirs(training_state_dir, exist_ok=True)
    training_state = {"step": optimization_step}
    torch.save(training_state, os.path.join(training_state_dir, "training_state.pt"))

    # Update the "last" symlink
    update_last_checkpoint(checkpoint_dir)

    logging.info(f"Checkpoint saved to {checkpoint_dir}")


def make_optimizers_and_scheduler(cfg: TrainRLServerPipelineConfig, policy: nn.Module):
    """Creates and returns optimizers for the actor, critic, and temperature."""
    optimizer_actor = torch.optim.Adam(
        params=[
            p
            for n, p in policy.actor.named_parameters()
            if not policy.config.shared_encoder or not n.startswith("encoder")
        ],
        lr=cfg.policy.actor_lr,
    )
    optimizer_critic = torch.optim.Adam(params=policy.critic_ensemble.parameters(), lr=cfg.policy.critic_lr)

    if cfg.policy.num_discrete_actions is not None:
        optimizer_discrete_critic = torch.optim.Adam(
            params=policy.discrete_critic.parameters(), lr=cfg.policy.critic_lr
        )
    optimizer_temperature = torch.optim.Adam(params=[policy.log_alpha], lr=cfg.policy.critic_lr)

    optimizers = {
        "actor": optimizer_actor,
        "critic": optimizer_critic,
        "temperature": optimizer_temperature,
    }
    if cfg.policy.num_discrete_actions is not None:
        optimizers["discrete_critic"] = optimizer_discrete_critic

    return optimizers, None # No scheduler implemented


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


def log_training_info(cfg: TrainRLServerPipelineConfig, policy: nn.Module) -> None:
    """Log information about the training process."""
    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.policy.online_steps=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")


def train_cli():
    # Use the job_name from the config
    global cfg

    logging.info("[LEARNER] train_cli finished")
    train(cfg, job_name=cfg.job_name)


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
    log_file = os.path.join(log_dir, f"learner_{job_name}.log")

    init_logging(log_file=log_file, display_pid=False)
    logging.info(f"Learner logging initialized, writing to {log_file}")

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
    run_offline_training(
        cfg=cfg,
        wandb_logger=wandb_logger,
    )


if __name__ == "__main__":
    train_cli()
