#!/usr/bin/env python
"""
Script to print current joint positions of all 4 robots (Leader Left/Right, Follower Left/Right).
Useful for determining home positions.
"""

import sys
import os
import time
import logging

# Add backend directory to sys.path to import modules
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend_fastapi"))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from config_models import TeleopRequest
from config_resolver import resolve
from lerobot_adapter import to_lerobot_configs
from lerobot.robots.utils import make_robot_from_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    print("Initializing robot connections...")
    
    # Request bimanual configuration to get all arms
    req = TeleopRequest(
        operation_mode="bimanual",
        robot_type="bi_viperx",
        teleop_type="bi_widowx",
        cameras_enabled=False,  # No cameras needed for joint reading
        display_data=False,
        fps=30
    )

    try:
        # Resolve configuration
        robot_cfg, teleop_cfg, _ = resolve(req)
        robot_config, teleop_config = to_lerobot_configs(robot_cfg, teleop_cfg)

        # Create robot instances
        print("Connecting to Follower Arms (BiViperX)...")
        follower = make_robot_from_config(robot_config)
        follower.connect(calibrate=False) # Assume already calibrated or just reading raw values
        
        print("Connecting to Leader Arms (BiWidowX)...")
        leader = make_robot_from_config(teleop_config)
        leader.connect(calibrate=False)

        print("\n" + "="*60)
        print("ROBOT JOINT POSITION READER")
        print("="*60)
        print("Instructions:")
        print("1. Move the robots to the desired position manually.")
        print("2. Press ENTER to read and print the current joint positions.")
        print("3. Press Ctrl+C to exit.")
        print("="*60 + "\n")

        while True:
            input("Press ENTER to read positions...")
            
            # Read observations
            follower_obs = follower.get_observation()
            leader_obs = leader.get_observation()
            
            print("\n--- Current Joint Positions ---")
            
            # Helper to print arm joints
            def print_arm(name, obs, prefix):
                print(f"\n[{name}]")
                joints = [k for k in obs.keys() if k.startswith(prefix) and "pos" in k]
                # Sort for consistent order
                joints.sort()
                
                # Print as dictionary format for easy copy-pasting
                print(f"{name}_HOME = {{")
                for j in joints:
                    # Remove prefix for the key in the dict if desired, or keep it
                    # The SINGLE_ARM_HOME in robot.py uses keys like "waist.pos" (no prefix)
                    # But here we have "left_waist.pos" etc.
                    # Let's print both full key and stripped key for convenience
                    
                    clean_key = j.replace(prefix, "")
                    val = obs[j]
                    print(f'    "{clean_key}": {val:.4f},')
                print("}")

            print_arm("Follower Left", follower_obs, "left_")
            print_arm("Follower Right", follower_obs, "right_")
            print_arm("Leader Left", leader_obs, "left_")
            print_arm("Leader Right", leader_obs, "right_")
            
            print("\n" + "-"*60)

    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
    finally:
        print("Disconnecting...")
        try:
            if 'follower' in locals() and follower:
                follower.disconnect()
            if 'leader' in locals() and leader:
                leader.disconnect()
        except Exception as e:
            print(f"Error during disconnect: {e}")

if __name__ == "__main__":
    main()
