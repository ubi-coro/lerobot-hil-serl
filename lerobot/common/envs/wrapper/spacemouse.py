from typing import Dict, Sequence, Union, Tuple, Optional

import gymnasium as gym
import numpy as np

from lerobot.common.robot_devices.motors import pyspacemouse


class SpaceMouseExpert:
    """
    Reads from a single SpaceMouse HID device.

    Args:
      device: string name of one supported device (from pyspacemouse.supported_devices)
    """
    def __init__(self, device: Optional[str] = None):
        pyspacemouse.open(device=device)

    def get_action(self) -> Tuple[np.ndarray, list]:
        """Returns a 6-vector of axes and a list of button states"""
        state = pyspacemouse.read_all()[0]
        axes = np.array([
            state.x, state.y, state.z,
            state.roll, state.pitch, state.yaw
        ], dtype=np.float32)
        buttons = list(state.buttons)
        return axes, buttons

    def close(self):
        pyspacemouse.close()

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
    ):
        super().__init__(env)
        self.robot_names = list(env.unwrapped.robot.controllers.keys())
        self.experts = {}
        self.action_indices = {}
        self.scales = {}

        # gripper flags
        self.gripper_enabled = {n: env.unwrapped.controllers[n].config.use_gripper for n in self.robot_names}
        self.action_space = env.action_space

        # build spacemouse experts, indices and scales
        if devices is None:
            devices = {}
        if action_indices is None:
            action_indices = {}
        if action_scale is None:
            action_scale = {}

        for name in self.robot_names:
            num_actions = 7 if self.gripper_enabled[name] else 6
            self.experts[name] = SpaceMouseExpert(device=devices.get(name, None))
            self.action_indices = action_indices.get(name, [1] * num_actions)

            scale = np.array(action_scale.get(name, 1.0), dtype=float)
            if scale.size == 1:
                scale = np.full(6, scale)
            assert scale.size == 6
            self.scales[name] = scale

    def action(self, policy_action: np.ndarray) -> Tuple:

        is_intervention = False
        intervention_action = policy_action.copy()
        idx_start = 0
        for name in self.robot_names:
            spacemouse_action, buttons = self.experts[name].get_action()
            moved = np.linalg.norm(spacemouse_action) < 1e-3
            spacemouse_action = spacemouse_action * self.scales[name]

            # handle gripper
            close_gripper, open_gripper = bool(buttons[0]), bool(buttons[1])
            gripper_value = None
            if self.gripper_enabled[name]:
                if close_gripper and not open_gripper:
                    gripper_value = 0.0
                    moved = True
                elif open_gripper and not close_gripper:
                    gripper_value = 1.0
                    moved = True
            if self.gripper_enabled[name]:
                spacemouse_action = np.concatenate([spacemouse_action, np.array([gripper_value])])

            # add filtered action to intervention action
            is_intervention = moved
            idc = self.action_indices[name]
            offset = len(idc)
            intervention_action[idx_start: idx_start + offset] = spacemouse_action[idc]

        return policy_action, is_intervention, intervention_action

    def step(self, action):
        policy_action, is_intervention, intervention_action = self.action(action)

        if is_intervention:
            new_action = intervention_action
        else:
            new_action = policy_action

        obs, reward, done, info = self.env.step(new_action)

        info = info or {}
        info['is_intervention'] = is_intervention
        if is_intervention:
            info["action_intervention"] = intervention_action

        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def close(self):
        super().close()
        for e in self.experts.values(): e.close()
