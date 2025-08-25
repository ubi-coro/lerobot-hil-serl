import multiprocessing
import time
from typing import Dict, Sequence, Union, Tuple, Optional

import gymnasium as gym
import numpy as np
import torch

from lerobot.common.robot_devices.motors import pyspacemouse


class SpaceMouseExpert:
    def __init__(self, device: Optional[str] = None):
        pyspacemouse.open(device=device)

        self.stop_event = multiprocessing.Event()

        # Manager to handle shared state between processes
        self.manager = multiprocessing.Manager()
        self.latest_data = self.manager.dict()
        self.latest_data["action"] = [0.0] * 6
        self.latest_data["buttons"] = [0, 0, 0, 0]

        self.process = multiprocessing.Process(
            target=self._read_spacemouse, name="SpaceMouseReader"
        )
        self.process.daemon = True
        self.process.start()

    def _read_spacemouse(self):
        while not self.stop_event.is_set():
            state = pyspacemouse.read_all()
            action = [0.0] * 6
            buttons = [0, 0, 0, 0]

            if len(state) == 2:
                action = [
                    -state[0].y, state[0].x, state[0].z,
                    -state[0].roll, -state[0].pitch, -state[0].yaw,
                    -state[1].y, state[1].x, state[1].z,
                    -state[1].roll, -state[1].pitch, -state[1].yaw
                ]
                buttons = state[0].buttons + state[1].buttons
            elif len(state) == 1:
                action = [
                    -state[0].y, state[0].x, state[0].z,
                    -state[0].roll, -state[0].pitch, -state[0].yaw
                ]
                buttons = state[0].buttons

            try:
                # If the manager/pipe is gone during shutdown, just exit quietly
                self.latest_data["action"] = action
                self.latest_data["buttons"] = buttons
            except (BrokenPipeError, EOFError, ConnectionResetError, OSError):
                break

            # be nice to a CPU core :)
            time.sleep(0.002)

    def get_action(self) -> Tuple[np.ndarray, list]:
        action = self.latest_data.get("action", [0.0] * 6)
        buttons = self.latest_data.get("buttons", [0, 0, 0, 0])
        return np.array(action), buttons

    def close(self):
        # Request the reader to stop
        self.stop_event.set()
        # Closing the HID usually unblocks read_all() if it’s waiting
        try:
            pyspacemouse.close()
        except Exception:
            pass

        # Give the process a moment to exit cleanly
        self.process.join(timeout=0.5)
        if self.process.is_alive():
            # Fall back to hard kill if needed
            self.process.terminate()
            self.process.join(timeout=0.2)

        # Now it’s safe to tear down the manager/IPC
        try:
            self.manager.shutdown()
        except Exception:
            pass


class SpaceMouseInterventionWrapper(gym.ActionWrapper):
    """
    A SpaceMouse-based intervention for multiple robots.

    On each step, reads from one SpaceMouseExpert per robot (listening on a specific HID device),
    scales its 6 axes by `action_scale`, maps buttons to gripper values, and overrides only
    the indices in `action_indices`. Falls back to policy action if the mouse is at rest.

    Args:
      env: gym.Env
      action_indices: dict robot_name -> list of ints (positions in global action vector)
      devices: dict robot_name -> device name
      action_scale: dict robot_name -> float or list of 6 floats
    """
    def __init__(
        self,
        env: gym.Env,
        devices: Dict[str, str],
        action_indices: Dict[str, Sequence[int]],
        action_scale: Dict[str, Union[float, Sequence[float]]],
        intercept_with_button: bool = False,
        device: str = "cuda"
    ):
        super().__init__(env)
        self.robot_names = list(env.unwrapped.robot.controllers.keys())
        self.intercept_with_button = intercept_with_button
        self.block_interventions = False
        self.device = device
        self.experts = {}
        self.action_indices = {}
        self.scales = {}
        self.gripper_enabled = {}

        # build spacemouse experts, indices and scales
        if devices is None:
            devices = {}
        if action_indices is None:
            action_indices = {}
        if action_scale is None:
            action_scale = {}

        for name in self.robot_names:
            self.experts[name] = SpaceMouseExpert(device=devices.get(name, None))
            self.gripper_enabled[name] = env.unwrapped.robot.controllers[name].config.use_gripper
            assert not (intercept_with_button and self.gripper_enabled[name])

            num_actions = 7 if self.gripper_enabled[name] else 6
            self.action_indices[name] = np.array(action_indices.get(name, [1] * num_actions), dtype=bool)
            assert len(self.action_indices[name]) == num_actions

            scale = np.array(action_scale.get(name, 1.0), dtype=float)
            if scale.size == 1:
                scale = np.full(6, scale)
            assert scale.size == 6
            self.scales[name] = scale

    def action(self, policy_action: torch.Tensor) -> Tuple:
        is_intervention = False
        intervention_action = policy_action.clone()

        if self.block_interventions:
            return policy_action, is_intervention, None

        idx_start = 0
        for name in self.robot_names:
            expert = self.experts[name]
            idc = self.action_indices[name]
            scale = self.scales[name]
            gripper_enabled = self.gripper_enabled[name]
            offset = sum(idc)

            # handle spacemouse
            spacemouse_action, buttons = expert.get_action()

            if self.intercept_with_button:
                moved = bool(buttons[0]) | bool(buttons[1])
            else:
                moved = np.linalg.norm(spacemouse_action) > 1e-3

            spacemouse_action = scale * spacemouse_action

            # handle gripper
            if gripper_enabled:
                # if we do not mask the gripper action out later, its current value
                # is the last entry in the policy action
                offset += 1
                gripper_value = policy_action[idx_start + offset] if idc[-1] else 0.0 # avoid out of bounds error

                close_gripper, open_gripper = bool(buttons[0]), bool(buttons[1])
                if close_gripper and not open_gripper:
                    gripper_value = 0.0
                    moved = True
                elif open_gripper and not close_gripper:
                    gripper_value = 1.0
                    moved = True

                # append gripper action
                spacemouse_action = np.concatenate([spacemouse_action, np.array([gripper_value])])

            if not moved:
                idx_start += offset
                continue

            # add filtered action to intervention action
            is_intervention = True
            intervention_action_slice = torch.tensor(spacemouse_action[idc]).to(device=self.device)
            intervention_action[idx_start: idx_start + offset] = intervention_action_slice
            idx_start += offset

        return policy_action, is_intervention, intervention_action

    def step(self, action):
        policy_action, is_intervention, intervention_action = self.action(action)

        if is_intervention:
            new_action = intervention_action
        else:
            new_action = policy_action

        obs, reward, terminated, truncated, info = self.env.step(new_action)

        info = info or {}
        info['is_intervention'] = is_intervention
        if is_intervention:
            info["action_intervention"] = intervention_action

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['is_intervention'] = False
        return obs, info

    def close(self):
        super().close()
        for e in self.experts.values(): e.close()
