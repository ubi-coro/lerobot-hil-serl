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
Meta Quest Teleoperation Data Collection Script (RCS Gym Interface).

This script connects a VR headset (Meta Quest via assumed SDK bridge) to the 
RCS-FR3 hardware environment to collect human demonstrations for training VLAs.
"""

import gymnasium as gym
import torch
import numpy as np
import time
from collections import defaultdict
import logging
from typing import Dict, Any, Tuple, List, Optional
from os import PathLike

# --- RCS IMPORTS ---
import rcs
from rcs.envs.base import ControlMode
from rcs.envs.creators import RCSHardwareEnvCreator
from rcs_fr3.utils import default_fr3_hw_gripper_cfg, default_fr3_hw_robot_cfg
# NOTE: Assuming necessary RCS modules are available in the running environment
# like RCSFR3EnvCreator, etc.

# --- LEROBOT/DATA IMPORTS ---
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import push_to_hub

# --- Constants ---
DATA_REPO_ID = "YOUR_HF_USERNAME/meta-quest-teleop-demos"
ROBOT_IP = "192.168.1.10" 
ENV_CONTROL_MODE = ControlMode.CARTESIAN_TRPY

# --- Teleop Configuration ---
MAX_EPISODE_STEPS = 500
TELEOP_FREQ = 100 # Hz (Rate at which we read the controller and step the environment)
TELEOP_SCALE_TRANS = 0.005 # Max meters per second (5 mm per cycle)
TELEOP_SCALE_ROT = np.deg2rad(1.5) # Max radians per cycle (1.5 degrees)


# --- Mock VR Input Class (Hook this up to your actual Meta Quest SDK reader) ---
class MetaQuestInput:
    """
    MOCK CLASS: Simulates reading the Meta Quest controller state.
    In a real system, this would interface with OpenXR or a custom bridge (e.g., Unity/ROS).
    """
    def __init__(self):
        self.is_connected = True
        self.last_pose = np.zeros(7) # [x, y, z, qx, qy, qz, qw]
        self.is_gripping = False

    def get_controller_state(self) -> Tuple[np.ndarray, bool]:
        """
        Returns the current 6D pose (3 pos, 4 quat) and grip status.
        MOCK: Returns dummy pose movement for testing.
        """
        # Read the actual pose from the SDK here
        current_pose = self.last_pose + np.array([0, 0, 0, 0, 0, 0, 0]) * 0.001
        
        # Simulate button press based on time for 'save episode' trigger
        is_trigger_pressed = (time.time() % 5 < 0.1) 
        
        return current_pose, is_trigger_pressed
    
    def get_delta_from_last_state(self, current_pose: np.ndarray) -> np.ndarray:
        """Calculates 6D delta (3 linear, 3 angular) normalized by scale."""
        
        # MOCK LOGIC: We need real delta pose math here. 
        # Since we're collecting data for a delta action space (RelativeActionSpace),
        # we calculate the difference between the *current* pose and the *last known* pose.
        
        linear_delta = current_pose[:3] - self.last_pose[:3]
        # NOTE: Rotational delta (quaternion math) is non-trivial and environment-specific.
        # We assume simplified delta calculation for the purpose of this template.
        rotational_delta = np.zeros(3) # [roll, pitch, yaw]
        
        # Apply scaling and clamping to map VR movement to robot action limits
        delta_action = np.zeros(7, dtype=np.float32)
        delta_action[:3] = np.clip(linear_delta, -TELEOP_SCALE_TRANS, TELEOP_SCALE_TRANS)
        delta_action[3:6] = np.clip(rotational_delta, -TELEOP_SCALE_ROT, TELEOP_SCALE_ROT)

        # Gripper action (0 or 1) - assuming last dim is gripper
        # MOCK: Use grip button status
        delta_action[6] = 1.0 if self.is_gripping else -1.0 
        
        self.last_pose = current_pose
        return delta_action


# --- RCS Environment Creator (Requires definition or import) ---
# NOTE: Assuming RCSFR3EnvCreator is defined elsewhere and needed for the Default creator
class RCSFR3EnvCreator(RCSHardwareEnvCreator):
    # This class definition is truncated here, but assumed to exist
    # in the user's project to allow RCSFR3DefaultEnvCreator to work.
    pass

class RCSFR3DefaultEnvCreator(RCSHardwareEnvCreator):
    """Factory method for the FR3 hardware environment."""
    def __call__(  # type: ignore
        self,
        robot_ip: str,
        control_mode: ControlMode = ControlMode.CARTESIAN_TRPY,
        delta_actions: bool = True,
        camera_set: Any | None = None, # Use Any for HardwareCameraSet
        gripper: bool = True,
    ) -> gym.Env:
        # NOTE: Assuming RCSFR3EnvCreator is accessible.
        # The actual implementation logic is omitted but assumed correct.
        
        # MOCK Environment Creation (Actual logic is long and unnecessary here)
        print(f"Creating RCS environment for IP: {robot_ip}...")
        
        # Mock initialization to get a generic Gym env
        env = gym.make("Reacher-v4") 
        
        # In a real setup, this returns the configured RCS Env
        # return RCSFR3EnvCreator()(
        #     ip=robot_ip,
        #     camera_set=camera_set,
        #     control_mode=control_mode,
        #     robot_cfg=default_fr3_hw_robot_cfg(),
        #     collision_guard=None,
        #     gripper_cfg=default_fr3_hw_gripper_cfg() if gripper else None,
        #     max_relative_movement=(0.2, np.deg2rad(45)) if delta_actions else None,
        #     relative_to=RelativeTo.LAST_STEP,
        # )
        return env


# --- Main Data Collection Logic ---

def collect_episode(env: gym.Env, vr_input: MetaQuestInput, episode_idx: int):
    """
    Collects a single episode of teleoperated data.
    """
    episode_data = []
    
    # NOTE: Assuming the RCS env is configured to render an image array on demand
    obs_flat, info = env.reset()
    obs_image_np = env.render("rgb_array") 

    terminated = truncated = False
    step_count = 0
    start_time = time.time()
    
    # Initialize VR pose tracker
    initial_pose, _ = vr_input.get_controller_state()
    vr_input.last_pose = initial_pose # Set the initial reference point
    
    print(f"  Starting Episode {episode_idx} (Press trigger to start recording)...")

    while not (terminated or truncated) and step_count < MAX_EPISODE_STEPS:
        
        time_since_last_step = time.time() - start_time
        
        if time_since_last_step < 1.0 / TELEOP_FREQ:
            # Wait to maintain the desired frequency
            time.sleep(1.0 / TELEOP_FREQ - time_since_last_step)
        
        start_time = time.time() # Reset step timer

        # 1. Read human input
        current_pose, is_trigger_pressed = vr_input.get_controller_state()
        
        # The episode is saved when the trigger is released (for HIL training)
        if is_trigger_pressed:
            vr_input.is_gripping = True
        else:
            vr_input.is_gripping = False

        # 2. Convert VR pose delta to robot action (a_human)
        # This delta is relative to the last step, fitting the RCS RelativeActionSpace
        human_action = vr_input.get_delta_from_last_state(current_pose)

        # 3. Step the environment
        next_obs_flat, reward, terminated, truncated, info = env.step(human_action)
        next_obs_image_np = env.render("rgb_array")
        
        # 4. Store the data frame (This structure is compatible with DAgger/RLPD)
        frame = {
            # Observations (Image is required for VLA state z_s)
            "observation.image": obs_image_np,
            # Add other necessary state info from obs_flat here (e.g., proprio)
            "observation.state": obs_flat, 
            
            # Actions and Transitions
            "action": human_action,
            "next_observation.image": next_obs_image_np,
            "next_observation.state": next_obs_flat,
            "reward": np.array([reward], dtype=np.float32), 
            "terminated": np.array([terminated]),
            "truncated": np.array([truncated]),
            "is_human_intervention": np.array([True]), # Always true in teleop
        }
        episode_data.append(frame)
        
        # Update state for next loop
        obs_flat = next_obs_flat
        obs_image_np = next_obs_image_np
        step_count += 1
        
    print(f"  ...Episode {episode_idx} finished after {step_count} steps.")
    return episode_data

def format_rollouts_for_lerobot_dataset(all_rollout_data):
    """Converts list-of-lists-of-dicts to dict-of-lists."""
    dataset_dict = defaultdict(list)
    for episode_data in all_rollout_data:
        # Determine the next episode index
        episode_index = dataset_dict["episode_index"][-1] + 1 if dataset_dict["episode_index"] else 0
        for frame in episode_data:
            for key, value in frame.items():
                dataset_dict[key].append(value)
            dataset_dict["episode_index"].append(episode_index)
    return dict(dataset_dict)


def main():
    print("--- Meta Quest Teleoperation Data Collector ---")
    
    # 1. Initialize VR Input and Environment
    vr_input = MetaQuestInput()

    # Instantiate the environment creator
    env_creator = RCSFR3DefaultEnvCreator()
    
    try:
        # Create the environment instance using the creator and robot IP
        env = env_creator(
            robot_ip=ROBOT_IP,
            control_mode=ENV_CONTROL_MODE,
            delta_actions=True, 
            gripper=True
        )
    except Exception as e:
        print(f"FATAL: Could not instantiate RCS environment. Error: {e}")
        return

    # 2. Collect Episodes
    all_rollout_data = []
    for i in range(NUM_ROLLOUTS):
        # Stop collecting if VR is disconnected (in a real scenario)
        if not vr_input.is_connected:
            print("VR controller disconnected. Stopping collection.")
            break
            
        episode_data = collect_episode(env, vr_input, i)
        if episode_data:
            all_rollout_data.append(episode_data)
        
    env.close()

    if not all_rollout_data:
        print("No successful episodes collected. Exiting.")
        return
        
    # 3. Format and Save Data
    dataset_dict = format_rollouts_for_lerobot_dataset(all_rollout_data)
    
    # Create LeRobotDataset
    dataset = LeRobotDataset(dataset_dict)
    
    print(f"\nPushing {len(all_rollout_data)} episodes to {DATA_REPO_ID}...")
    push_to_hub(
        repo_id=DATA_REPO_ID,
        dataset=dataset,
        hf_token=None, 
        commit_message="Meta Quest Teleoperation Demos"
    )
    print("Teleoperation data collection complete and saved to Hugging Face Hub!")

if __name__ == "__main__":
    main()