from typing import Any, Dict, Union

import gym
import numpy as np
import sapien
import torch
from mani_skill.agents.robots import Panda, Fetch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import SimConfig, GPUMemoryConfig, Array
from pynput import keyboard


@register_env("Reach1D-v0", max_episode_steps=50)
class Reach1DEnv(BaseEnv):
    """A minimal task: move robot's end-effector along X-axis to reach a target X position."""

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(self, target_x: float = 0.6, robot_uids="panda", robot_init_qpos_noise=0.0, **kwargs):
        self.target_x = target_x
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25,
                max_rigid_patch_count=2**18,
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([0.4, 0, 0.6], [0.6, 0, 0.2])
        return [CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 2)]

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        # Just a floor and table
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            # the initialization functions where you as a user place all the objects and initialize their properties
            # are designed to support partial resets, where you generate initial state for a subset of the environments.
            # this is done by using the env_idx variable, which also tells you the batch size
            b = len(env_idx)
            # when using scene builders, you must always call .initialize on them so they can set the correct poses of objects in the prebuilt scene
            # note that the table scene is built such that z=0 is the surface of the table.
            self.table_scene.initialize(env_idx)

    def evaluate(self):
        tcp_x = self.agent.tcp.pose.p[..., 0]
        distance = torch.abs(tcp_x - self.target_x)
        success = distance < 0.02  # within 2cm
        return {"success": success}

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            obs["target_x"] = torch.tensor([self.target_x], device=self.device)
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        tcp_x = self.agent.tcp.pose.p[..., 0]
        dist = torch.abs(tcp_x - self.target_x)
        reward = 1 - torch.tanh(10 * dist)
        reward[info["success"]] = 3
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        return self.compute_dense_reward(obs, action, info) / 3.0


class ActionMaskingWrapper(gym.ActionWrapper):
    """A wrapper to mask actions, allowing movement only in the target direction."""

    def step(self, action: Array):
        # Mask the action to allow movement only in the X direction
        tcp_x = self.env.agent.tcp.pose.p[..., 0]
        tcp_x[0] = action[0]  # Allow movement in the X direction
        return self.env.step(tcp_x)


class KeyboardControlWrapper(gym.Wrapper):
    """
    A wrapper for controlling the robot's end-effector using the keyboard.
    Arrow keys are used to move the end-effector along the X-axis.
    """

    def __init__(self, env: Reach1DEnv):
        super().__init__(env)
        self.action = 0.0  # Default action value
        self.done = False
        listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        listener.start()

    def _on_press(self, key):
        try:
            if key == keyboard.Key.right:
                self.action = 1.0  # Move right
            elif key == keyboard.Key.left:
                self.action = -1.0  # Move left
            elif key == keyboard.Key.esc:
                self.done = True  # Exit the loop
        except AttributeError:
            pass

    def _on_release(self, key):
        if key in [keyboard.Key.right, keyboard.Key.left]:
            self.action = 0.0  # Stop movement

    def step(self, action):
        """
        Run the environment loop with keyboard control.
        """
        overwrite_action = [self.action]

        obs, reward, done, info = self.env.step(overwrite_action)
        return obs, reward, self.done, info


if __name__ == "__main__":
    import time

    fps = 30

    env = Reach1DEnv(target_x=0.6, render_mode="human")
    #env = ActionMaskingWrapper(env)
    #env = KeyboardControlWrapper(env)

    done = False
    while not done:
        action = np.zeros(env.action_space.shape)
        obs, reward, done, _, info = env.step(action)
        env.render()
        time.sleep(1 / fps) # Control the frame rate

    env.close()
