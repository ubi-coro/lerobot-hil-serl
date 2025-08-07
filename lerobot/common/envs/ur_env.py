from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from lerobot.common.robot_devices.robots.ur import UR


class UREnv(gym.Env):
    """
    Gym wrapper for UR robot:

    - Observation is a Dict with:
        * "state": concatenated non-image features
        * each image feature keyed separately
    - Action is a continuous Box from robot.features["action"]
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        robot: UR,
        display_cameras: bool = False
    ):
        super().__init__()
        
        self.robot = robot
        self.display_cameras = display_cameras
        self.current_step = 0

        if not self.robot.is_connected:
            self.robot.connect()

        # Build from robot.features
        feats = self.robot.features
        self.state_keys = [k for k in feats if k != "action" and "image" not in k.lower()]
        self.image_keys = [k for k in feats if "image" in k.lower()]
        obs_spaces: Dict[str, spaces.Space] = {}

        # Compute state dimension
        state_dim = 0
        for k in self.state_keys:
            state_dim += int(np.prod(feats[k]["shape"]))
        state_dim = state_dim
        obs_spaces["observation.state"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        
        # Images
        for k in self.image_keys:
            shape = tuple(feats[k]["shape"])
            # assume images are uint8 0–255
            obs_spaces[k] = spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
            
        self.observation_space = spaces.Dict(obs_spaces)

        # Action space
        a_spec = feats["action"]
        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=tuple(a_spec["shape"]),
            dtype=np.dtype(a_spec["dtype"]),
        )

        self._last_obs: Optional[Dict[str, np.ndarray]] = None

    def reset(self, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict]:
        obs = self._obs()
        self.current_step = 0
        self._last_obs = obs
        return obs, {}

    def step(self, action: Any):
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action.astype(np.float32))

        sent = self.robot.send_action(action)

        obs = self._obs()
        
        if self.display_cameras:
            self.render()
        
        self.current_step += 1
        self._last_obs = obs
        
        return obs, 0.0, False, False, {}

    def _obs(self) -> Dict[str, torch.Tensor]:
        raw = self.robot.capture_observation()
        #obs = {k: raw[k] for k in self.image_keys}
        raw["observation.state"] = torch.cat([raw[k] for k in self.state_keys])
        return raw

    def render(self, mode="human"):
        import cv2

        observation = self.robot.capture_observation()

        for key in self.image_keys:
            cv2.imshow(key, cv2.cvtColor(observation[key].numpy(), cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
