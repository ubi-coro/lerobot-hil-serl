#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
Hybrid Client Rollout Script for Residual RL Fine-Tuning.

This script executes the closed-loop residual policy: 
a_final = a_VLA + a_residual, where both a_VLA and R(s, a) are obtained 
from external VLA server/classifier via VLM embeddings.

MODIFIED: Implements chunked execution for VLA actions (30 steps open-loop).
"""

import gymnasium as gym
import torch
import numpy as np
import random
from collections import defaultdict
import requests
import json_numpy
import time
import cv2 # Used for image resizing/transposing
import logging
from typing import Dict, Any, Tuple, List
from os import PathLike

# --- RCS IMPORTS (From Provided Interface) ---
# NOTE: Assumes these are available in the running environment
import rcs
import rcs.hand.tilburg_hand
from rcs.camera.hw import HardwareCameraSet
from rcs.envs.base import (
    ControlMode,
    GripperWrapper,
    HandWrapper,
    MultiRobotWrapper,
    RelativeActionSpace,
    RelativeTo,
    RobotEnv,
)
from rcs.envs.creators import RCSHardwareEnvCreator
from rcs.hand.tilburg_hand import TilburgHand
from rcs_fr3 import hw
from rcs_fr3.envs import FR3HW
from rcs_fr3.utils import default_fr3_hw_gripper_cfg, default_fr3_hw_robot_cfg

# --- LEROBOT/POLICY IMPORTS ---
from lerobot.common.policies.sac.sac_policy import SACPolicy
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import push_to_hub
from lerobot.policies.sac.reward_model.modeling_classifier import Classifier
from lerobot.bootcamp.xvla_client import VLAClient # <-- IMPORTING EXTERNAL CLIENT
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# --- VLA Client and Constants (Matching your server/client logic) ---

# --- Server & Data Config ---
DATA_REPO_ID = "YOUR_HF_USERNAME/my-bootcamp-rollouts"
TASK_INSTRUCTION = "Pick up the something" # Task for the VLA
SERVER_TIMEOUT = 10 # Timeout for VLA server request

# --- Model Dimensions (MUST match server/env) ---
VLA_EMBED_DIM = 4096 
PROPRIO_DIM = 7 # Assumed proprio dim from environment observation vector (7 joints/end-effector pose)
PROPRIO_TARGET_DIM = 20 # VLA Server expectation (padding is critical)
IMAGE_SHAPE = (256, 256, 3) # H, W, C
TARGET_IMAGE_SIZE = (256, 256)
ACTION_DIM = 7 # Assumed continuous action dim (e.g., 6 DOF + 1 gripper)

# --- CHUNKED EXECUTION PARAMETER ---
VLA_ACTION_CHUNK_LENGTH = 30 # The VLA predicts 30 steps ahead

# --- Policy Checkpoints ---
RESIDUAL_POLICY_CHECKPOINT = "./models/sac_policy_epoch_k.pth"
REWARD_CLASSIFIER_CHECKPOINT = None 

# --- Exploration Parameters ---
NUM_ROLLOUTS = 50
EXPLORE_PROBABILITY = 0.2
DENOISING_STEPS = 10 
EXPLOIT_NOISE_STD = 0.05 

# --- RCS ENV SETUP CONSTANTS ---
ROBOT_IP = "192.168.1.10" # Placeholder IP
ENV_CONTROL_MODE = ControlMode.CARTESIAN_TRPY

def format_embedding_for_sac(embedding_np, device):
    """Formats the VLA embedding vector for the `lerobot` SAC policy."""
    embedding_tensor = torch.tensor(embedding_np, dtype=torch.float32, device=device).unsqueeze(0)
    obs_dict = {"observation.state": embedding_tensor}
    return obs_dict
    
def unwrap_action_from_policy(action_tensor):
    """Removes the batch dimension and converts to NumPy for gym."""
    return action_tensor.squeeze(0).cpu().numpy()

def get_proprio_from_obs(obs_dict, proprio_dim=PROPRIO_DIM):
    """
    Extracts the proprioceptive state from the RCS environment's observation dictionary.
    """
    if isinstance(obs_dict, dict) and 'robot_state' in obs_dict:
        # Example: Accessing a specific state feature from RCS
        proprio_vector = obs_dict['robot_state']['q'] # Assuming joint positions
        return proprio_vector[:proprio_dim]
    
    # Fallback if obs_dict is a flat array (like in gym.make)
    if isinstance(obs_dict, np.ndarray):
        return obs_dict[:proprio_dim]
    
    logging.warning("Could not find standard RCS proprio keys. Returning zeros.")
    return np.zeros(proprio_dim, dtype=np.float32)


def collect_rollout(env, residual_policy, reward_classifier, vla_client, device, dataset):
    """
    Collects a single episode rollout using the hybrid action strategy with VLA action chunking.
    """
    episode_data = []
    
    obs_dict, info = env.reset() 
    obs_image_np = obs_dict.get('external_camera.rgb') or env.render("rgb_array") 
    obs_proprio_np = get_proprio_from_obs(obs_dict)

    terminated = truncated = False
    
    is_explore_mode = random.random() < EXPLORE_PROBABILITY
    denoising_steps = DENOISING_STEPS
    
    # --- CHUNKED EXECUTION STATE ---
    vla_action_chunk = np.zeros((VLA_ACTION_CHUNK_LENGTH, ACTION_DIM), dtype=np.float32)
    action_chunk_index = VLA_ACTION_CHUNK_LENGTH # Force initial fetch
    # -------------------------------
    
    print(f"  Rollout mode: {'Explore' if is_explore_mode else 'Exploit'} (VLA steps={denoising_steps}, Chunk={VLA_ACTION_CHUNK_LENGTH})...")
        
    while not (terminated or truncated):
        
        # 1. --- CONDITIONAL VLA CALL (Fetch Action Chunk or only Embedding) ---

        processed_sample = {
            OBS_STATE: obs_proprio_np,
            "observation.images.left": obs_image_np
        }
        
        if action_chunk_index >= VLA_ACTION_CHUNK_LENGTH:
            # Time to fetch a new chunk and the current state embedding (s)
            
            # Fetch full action chunk (30 steps) and current embedding (z_s)
            action_chunk_result, embedding_zs = vla_client.get_actions_and_embeddings(
                {"state": processed_sample}, device, use_next_state=False
            )
            
            # Reset index and store the new chunk
            vla_action_chunk = action_chunk_result
            action_chunk_index = 0
            
            print(f"  > Fetched new VLA action chunk ({VLA_ACTION_CHUNK_LENGTH} steps).")
        else:
            # Only need the current VLM embedding (z_s)
            # The action_beta for this step will come from the stored chunk.
            
            # Fetch ONLY embedding (skip_action=True for server optimization)
            _ , embedding_zs = vla_client.get_output(
                {"state": processed_sample}, device, use_next_state=True
            )

        # 2. --- GET VLA PRIOR ACTION (a_beta) from the chunk ---
        action_beta = vla_action_chunk[action_chunk_index] # Get the open-loop action
        
        # 3. Format embedding for SAC policy (s -> z_s)
        sac_obs_dict = format_embedding_for_sac(embedding_zs, device)
        
        # 4. --- GET RESIDUAL ACTION (delta_a) ---
        with torch.no_grad():
            residual_action_tensor, _ = residual_policy.select_action(sac_obs_dict)
            residual_action = unwrap_action_from_policy(residual_action_tensor)
            
            # Final Action: a_final = a_beta + delta_a
            final_action = action_beta + residual_action
            
            # Add small exploration noise 
            if is_explore_mode or EXPLOIT_NOISE_STD > 0:
                noise = np.random.randn(*final_action.shape) * EXPLOIT_NOISE_STD
                final_action += noise

            # Prepare action tensor for reward/env step
            action_tensor_for_reward = torch.tensor(final_action, device=device).unsqueeze(0)

        # 5. Step the environment
        next_obs_dict, env_reward, terminated, truncated, info = env.step(final_action)
        
        next_obs_image_np = next_obs_dict.get('external_camera.rgb') or env.render("rgb_array")
        next_obs_proprio_np = get_proprio_from_obs(next_obs_dict)
        
        # 6. --- GET LEARNED REWARD ---
        reward = env_reward
        if reward_classifier is not None:
            with torch.no_grad():
                reward_tensor = reward_classifier.predict_reward(
                    sac_obs_dict, 
                    action=action_tensor_for_reward
                )
                reward = reward_tensor.squeeze().cpu().item() 
        
        
        # 8. Store the data frame (embeddings, not images)
        frame = {
            "observation.state": embedding_zs,          # <-- VLM embedding (s)
            "vla_action": action_beta,                  # <-- VLA Prior Action (a_beta)
            "action": final_action,                     # <-- Final Executed Action (a_final)
            "reward": np.array([reward], dtype=np.float32), 
            "terminated": np.array([terminated]),
            "truncated": np.array([truncated]),
        }
        dataset.add_frame(frame)
        
        # 9. Update state and chunk index for next loop
        obs_dict = next_obs_dict
        obs_image_np = next_obs_image_np
        obs_proprio_np = next_obs_proprio_np
        action_chunk_index += 1
        
    dataset.save_episode()
    print(f"  ...finished rollout after {len(episode_data)} steps.")
    return episode_data

def format_rollouts_for_lerobot_dataset(all_rollout_data):
    """Converts list-of-lists-of-dicts to dict-of-lists."""
    print("Formatting data for LeRobotDataset...")
    dataset_dict = defaultdict(list)
    for episode_data in all_rollout_data:
        # Use existing data to determine next index, preventing overlap
        episode_index = dataset_dict["episode_index"][-1] + 1 if dataset_dict["episode_index"] else 0
        for frame in episode_data:
            for key, value in frame.items():
                dataset_dict[key].append(value)
            dataset_dict["episode_index"].append(episode_index)
    return dict(dataset_dict)


def load_policy(checkpoint_path, cfg, device):
    """Loads a lightweight MLP SAC policy or Classifier trained on VLM embeddings."""
    
    cfg.policy.pretrained_path = checkpoint_path
    offline_dataset = LeRobotDataset(
        repo_id=offline_repo_id,
        root=Path(cfg.dataset.root) / cfg.dataset.repo_id,
    )
    policy = make_sac(cfg.policy, ds_meta=offline_dataset.meta)
    policy.eval()
    return policy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Rollout Client using device: {device}")
    
    # --- 1. Load Env & Define Spaces ---
    # Instantiate the environment creator
    env_creator = RCSFR3DefaultEnvCreator()
    
    # Create the environment instance using the creator and placeholder IP
    try:
        # NOTE: Assumes environment handles camera setup internally or external cameras are configured
        env = env_creator(
            robot_ip=ROBOT_IP,
            control_mode=ENV_CONTROL_MODE,
            delta_actions=True, # Uses RelativeActionSpace
            gripper=True
        )
    except Exception as e:
        print(f"FATAL: Could not instantiate RCS environment using IP {ROBOT_IP}.")
        print(f"Please ensure the robot is accessible and the RCS system is running. Error: {e}")
        return

    # Get action space from env
    act_space = env.action_space
    
    # CRITICAL: Define the observation space for your *MLP* policies
    # This space is the VLA's embedding (the output of the pre-processing server).
    obs_space = gym.spaces.Dict({
        "observation.state": gym.spaces.Box(
            -np.inf, np.inf, 
            shape=(VLA_EMBED_DIM,), 
            dtype=np.float32
        )
    })

    dataset = LeRobotDataset.create(
        repo_id=DATA_REPO_ID,
        fps=30,
        features={
            OBS_STATE: {"dtype": "float32", "shape": (PROPRIO_DIM, )},
            "observation.images.1": {"dtype": "float32", "shape": (PROPRIO_DIM, )},
            OBS_STATE: {"dtype": "float32", "shape": (PROPRIO_DIM, )},
        }
    )

    # --- 2. Load VLA Client and Policies ---
    
    # Initialize the VLA Client Wrapper (uses the imported VLAClient)
    vla_client = VLAClient()

    # Both policies must be configured to consume the VLA_EMBED_DIM state
    residual_policy = load_policy(cfg, RESIDUAL_POLICY_CHECKPOINT, device)
    
    reward_classifier = None
    if REWARD_CLASSIFIER_CHECKPOINT:
        try:
            reward_classifier = load_mlp_policy(
                REWARD_CLASSIFIER_CHECKPOINT, device, obs_space, act_space, policy_class=Classifier
            )
        except Exception as e:
            print(f"Error loading reward classifier: {e}")
            print("Defaulting to environment rewards.")

    # --- 3. Collect rollouts ---
    all_rollout_data = []
    for i in range(NUM_ROLLOUTS):
        print(f"--- Collecting rollout {i+1} / {NUM_ROLLOUTS} ---")
        episode_data = collect_rollout(env, residual_policy, reward_classifier, vla_client, device)
        all_rollout_data.append(episode_data)
        
    env.close()

    if not all_rollout_data:
        print("No data collected. Exiting.")
        return
        
    dataset_dict = format_rollouts_for_lerobot_dataset(all_rollout_data)
    
    # The dataset now contains VLM embeddings as the state
    dataset = LeRobotDataset(dataset_dict)
    
    print(f"\nPushing dataset to {DATA_REPO_ID}...")
    push_to_hub(
        repo_id=DATA_REPO_ID,
        dataset=dataset,
        hf_token=None, # Assumes `huggingface-cli login`
        commit_message="Add epoch K rollout data (X-VLA Embeddings + SAC Residual)"
    )
    print("Done!")

if __name__ == "__main__":
    main()