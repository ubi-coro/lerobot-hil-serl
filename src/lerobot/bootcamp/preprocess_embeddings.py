#!/usr/bin/env python

# ------------------------------------------------------------------------------
# Script for Offline VLA Embedding and Action Precomputation.
# ------------------------------------------------------------------------------
"""
This script loads a raw LeRobotDataset, calls the external VLA server (xvla_server_batched)
in large batches to precompute VLM embeddings (z_s) and VLA prior actions (a_beta),
and saves these new features into the dataset columns: 'observation.state' and 'vla_action'.

The RL trainer can then load this processed dataset directly, skipping slow network calls.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from concurrent.futures import ThreadPoolExecutor
from copy import copy
import os # Included for NUM_CPU_WORKERS

import torch
import numpy as np
import requests
import json_numpy
import math
from tqdm import tqdm
from lerobot.utils.constants import (
    ACTION,
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    TRAINING_STATE_DIR,
    OBS_STATE,
    REWARD,
    DONE,
)

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import get_safe_torch_device
from lerobot.bootcamp.xvla_client import VLAClient, EMBEDDING_DIM, INSTRUCTION
from lerobot.configs.types import FeatureType, PolicyFeature

# --- CONFIGURATION CONSTANTS (MUST match your setup) ---
# NOTE: VLAClient is now assumed to be imported from the correct location
# We define the client parameters here, but the client implementation is external.
BATCH_SIZE = 64 # Use a large batch size for max throughput!

# Raw input keys expected in the source dataset
RAW_IMAGE_KEYS = ['observation.images.side', 'observation.images.up']
RAW_PROPRIO_KEY = 'observation.state'

# New output keys to be created in the processed dataset
VLM_EMBEDDING_KEY = 'observation.state'
VLA_ACTION_KEY = 'vla_action'

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Worker Config ---
NUM_CPU_WORKERS = os.cpu_count() or 4


# --- VLA Client for Precomputation ---
GLOBAL_VLA_CLIENT = VLAClient()

# --- Main Precomputation Logic ---

def run_precomputation(source_repo_id: str, target_repo_id: str, device: torch.device):
    """
    Main function to run the embedding and action precomputation.
    """
    log.info(f"Loading source dataset: {source_repo_id}")
    dataset = LeRobotDataset(repo_id=source_repo_id)
    num_transitions = len(dataset)
    num_batches = math.ceil(num_transitions / BATCH_SIZE)
    log.info(f"Total transitions: {num_transitions}. Processing in {num_batches} batches of size up to {BATCH_SIZE}.")

    action_dim = dataset.features[ACTION]["shape"][0]
    new_features = copy(dataset.features)
    new_features[OBS_STATE] = {"dtype": "float32", "shape": (EMBEDDING_DIM, )}
    new_features[VLA_ACTION_KEY] = {"dtype": "float32", "shape": (action_dim, )}
    for key in RAW_IMAGE_KEYS:
        del new_features[key]

    new_dataset = LeRobotDataset.create(
        repo_id=target_repo_id,
        root=str(dataset.root) + "_vlm_states",
        features=new_features,
        fps=dataset.fps,
    )

    # Use the globally initialized client instance
    vla_client = GLOBAL_VLA_CLIENT
    
    # Check for required keys
    for key in RAW_IMAGE_KEYS + [RAW_PROPRIO_KEY]:
        if key not in dataset.features:
            log.error(f"FATAL: Source dataset missing required key '{key}'. Aborting.")
            return

    log.info("Starting VLA inference for all transitions...")

    # Iterate through the dataset in batches
    prev_episode_idx = dataset[0]["episode_index"]
    for sample in dataset:

        if sample["episode_index"] != prev_episode_idx:
            new_dataset.save_episode()
            prev_episode_idx = sample["episode_index"]
            log.info(f"Processed episode {prev_episode_idx}")

        
        # Move relevant tensors to the device for VLA client
        processed_sample = {}
        for key in RAW_IMAGE_KEYS + [RAW_PROPRIO_KEY]:
             if key in sample:
                # Ensure tensors are float32/uint8 as expected, move to device
                processed_sample[key] = sample[key].to(device).unsqueeze(0)

        # 1. Call VLA server (batched)
        # We need the action prediction for the entire dataset, so skip_action=False
        vla_action, vlm_embedding = vla_client.get_actions_and_embeddings(
            {"state": processed_sample}, device, use_next_state=False
        )

        new_sample = {
            OBS_STATE: vlm_embedding.squeeze().cpu().numpy(),
            VLA_ACTION_KEY: vla_action.squeeze()[0, :action_dim].cpu().numpy(),
            ACTION: sample[ACTION],
            "task": INSTRUCTION
        }

        new_dataset.add_frame(new_sample)
    
    log.info(f"Successfully saved processed dataset to {target_repo_id}!")

# /home/hk-project-pai00093/id_hqc2052/.cache/huggingface/lerobot/lerobot/svla_so101_pickplace_vlm_states

# --- Example Execution ---
if __name__ == "__main__":
    SOURCE_REPO = "lerobot/svla_so101_pickplace"
    TARGET_REPO = "lerobot/svla_so101_pickplace_vlm_states"
    
    device = "cuda"

    log.warning("-------------------------------------------------------------------------------------------------")
    log.warning("REMINDER: Ensure the VLA server is running and the repo IDs are correct!")
    log.warning(f"  Source Repo (Raw Data): {SOURCE_REPO}")
    log.warning(f"  Target Repo (Processed Data): {TARGET_REPO}")
    log.warning("-------------------------------------------------------------------------------------------------")

    # To run this in a real environment, uncomment the call below:
    run_precomputation(SOURCE_REPO, TARGET_REPO, device)
    
    log.info("Script finished execution (Note: Precomputation step is commented out for safety).")
    log.info("Please set SOURCE_REPO and TARGET_REPO, and uncomment the run_precomputation call to execute.")