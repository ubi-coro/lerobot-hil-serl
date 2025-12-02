#!/usr/bin/env python
"""
Print current joint positions using experiment configs.

This script uses the same experiment-based configuration that `lerobot/share`
uses (via ExperimentConfigMapper), so ports, calibration, and processor setup
match your working teleop/record flows.

Usage examples:
  - Bimanual (left+right):
      python web/scripts/print_joint_positions.py --mode bimanual
  - Single left:
      python web/scripts/print_joint_positions.py --mode left
  - Single right:
      python web/scripts/print_joint_positions.py --mode right

It connects robots only (no cameras), reads positions when you press ENTER,
prints them in an easy copy-paste dict format, then you can rerun for another
mode if you prefer reading one arm at a time.
"""

import sys
import os
import time
import logging
import argparse

# Add backend directory to sys.path to import modules
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend_fastapi"))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from experiment_config_mapper import ExperimentConfigMapper
from lerobot.processor.hil_processor import (
    AddTeleopEventsAsInfoStep,
    AddTeleopActionAsComplimentaryDataStep,
    InterventionActionProcessorStep,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Print joint positions using experiments")
    parser.add_argument("--mode", choices=["bimanual","left","right"], default="bimanual",
                        help="Operation mode to use (matches GUI experiment mapping)")
    parser.add_argument("--source", choices=["follower","leader","both"], default="both",
                        help="Which hardware to read: follower robots (ViperX), leaders (WidowX), or both")
    args = parser.parse_args()

    print(f"Initializing environment for mode={args.mode}, source={args.source}...")

    try:
        # Create environment via experiments (no cameras)
        env, env_processor, action_processor, env_cfg, mapping = ExperimentConfigMapper.create_env_from_gui_selection(
            operation_mode=args.mode,
            demo_mode=False,
            policy_path_override=None,
            device="cpu",
            use_cameras=False,
        )

        # The environment exposes `robot_dict` with left/right entries when bimanual,
        # or a single robot when in single-arm mode. Similarly for teleoperators
        # controlling the leaders (WidowX), via `teleop_dict` or `teleoperator`.
        # We will read directly from these structures if present.
        robot_dict = getattr(env, "robot_dict", None)
        single_robot = None
        if robot_dict is None:
            # Fallback: experiments might expose `robot` as dict or single
            rob = getattr(env, "robot", None)
            if isinstance(rob, dict):
                robot_dict = rob
            else:
                single_robot = rob

        # Teleoperators are held inside action_processor steps; extract dict safely
        teleop_dict = None
        try:
            for step in getattr(action_processor, "steps", []):
                if isinstance(step, (AddTeleopEventsAsInfoStep, AddTeleopActionAsComplimentaryDataStep, InterventionActionProcessorStep)):
                    if getattr(step, "teleoperators", None):
                        teleop_dict = step.teleoperators
                        break
        except Exception:
            teleop_dict = None
        single_teleop = None

        print("\n" + "="*60)
        print("ROBOT JOINT POSITION READER (Experiments)")
        print("="*60)
        print("Instructions:")
        print("1. Move the robots/leaders to the desired position manually.")
        print("2. Press ENTER to read and print the current joint positions.")
        print("3. Press Ctrl+C to exit.")
        print("="*60 + "\n")

        while True:
            input("Press ENTER to read positions...")
            
            # Read observations from environment robots and teleoperators
            follower_obs_map = {}
            leader_obs_map = {}
            if args.source in ("follower", "both"):
                if robot_dict:
                    for name, rob in robot_dict.items():
                        try:
                            follower_obs_map[name] = rob.get_observation()
                        except Exception as e:
                            print(f"[Follower:{name}] read failed: {e}")
                elif single_robot is not None:
                    try:
                        follower_obs_map[args.mode] = single_robot.get_observation()
                    except Exception as e:
                        print(f"[Follower:robot] read failed: {e}")

            if args.source in ("leader", "both"):
                if teleop_dict:
                    for name, tel in teleop_dict.items():
                        try:
                            leader_obs_map[name] = tel.get_action()
                        except Exception as e:
                            print(f"[Leader:{name}] read failed: {e}")
                elif single_teleop is not None:
                    try:
                        leader_obs_map[args.mode] = single_teleop.get_action()
                    except Exception as e:
                        print(f"[Leader:teleop] read failed: {e}")

            print("\n--- Current Joint Positions ---")
            
            # Helper to print arm joints
            def print_arm(name, obs, prefix=None):
                print(f"\n[{name}]")
                if prefix is None:
                    # Unprefixed keys (single-arm robot like ViperX instances)
                    joints = [k for k in obs.keys() if k.endswith('.pos')]
                    joints.sort()
                    print(f"{name}_HOME = {{")
                    for j in joints:
                        val = obs[j]
                        print(f'    "{j}": {val:.4f},')
                    print("}")
                    return
                
                # Prefixed keys (from MultiRobot-style: left_/right_)
                joints = [k for k in obs.keys() if k.startswith(prefix) and "pos" in k]
                joints.sort()
                print(f"{name}_HOME = {{")
                for j in joints:
                    clean_key = j.replace(prefix, "")
                    val = obs[j]
                    print(f'    "{clean_key}": {val:.4f},')
                print("}")

            # Print followers (ViperX robots)
            if follower_obs_map:
                print("\n[Follower Robots]")
                for name, obs in follower_obs_map.items():
                    has_left = any(k.startswith("left_") for k in obs.keys())
                    has_right = any(k.startswith("right_") for k in obs.keys())
                    if has_left or has_right:
                        if has_left:
                            print_arm(f"{name} Left", obs, "left_")
                        if has_right:
                            print_arm(f"{name} Right", obs, "right_")
                    else:
                        print_arm(f"{name}", obs, None)

            # Print leaders (WidowX teleoperators)
            if leader_obs_map:
                print("\n[Leader Teleoperators]")
                for name, obs in leader_obs_map.items():
                    has_left = any(k.startswith("left_") for k in obs.keys())
                    has_right = any(k.startswith("right_") for k in obs.keys())
                    if has_left or has_right:
                        if has_left:
                            print_arm(f"{name} Left", obs, "left_")
                        if has_right:
                            print_arm(f"{name} Right", obs, "right_")
                    else:
                        print_arm(f"{name}", obs, None)
            
            print("\n" + "-"*60)

    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
    finally:
        print("Disconnecting...")
        try:
            if 'env' in locals() and env:
                env.close()
        except Exception as e:
            print(f"Error during env close: {e}")

if __name__ == "__main__":
    main()
