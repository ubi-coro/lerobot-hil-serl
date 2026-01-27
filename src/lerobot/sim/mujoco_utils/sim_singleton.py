import gymnasium as gym
import numpy as np

from lerobot.envs.factory import make_env, make_env_config
from lerobot.sim.mujoco_utils.viewer import create_viewer, AbstractViewer


class SimSingleton:
    def __init__(self, sim_config):
        env_cfg = make_env_config(sim_config.type)
        gym_env = make_env(env_cfg)
        self.env: gym.Env = gym_env.get(sim_config.type).get(0)

        self.next_action: dict = {}
        self.observation: dict = {
        }

        self.action_order = []
        self.reset()

        physics = self.env.envs[0].unwrapped._env.physics
        model = physics.model.ptr
        data = physics.data.ptr

        viewer_kwargs = {
            "key": sim_config.viewer,
            "model": model,
            "data": data,
            "image_keys": []
        }
        viewer = create_viewer(**viewer_kwargs)

        self.viewer: AbstractViewer = viewer
        self.viewer.start()


    def step(self):
        action = self._format_next_action()
        obs, reward, terminated, truncated, events = self.env.step(action)
        self._format_observation(obs)
        self.viewer.sync(obs)  # TODO(jzilke) set framerate
        return self.observation, reward, terminated, truncated, events


    def _format_next_action(self):
        """Format next action to sim"""
        return np.array([[self.next_action[joint] for joint in self.action_order]])


    def _format_observation(self, obs):
        """Format observation from sim to real data format"""
        self.observation |= obs['pixels']
        pos = obs['agent_pos'][0]
        self.observation |= {
            self.action_order[i]: pos[i] for i in range(len(self.action_order))
        }


    def get_observation(self, name: str):
        return self.observation.get(name)


    def add_action(self, name, action):
        _action = {f"{name}.{key}": value for key, value in action.items()}
        self.next_action |= _action


    def reset(self, seed=None, options=None):
        obs, events = self.env.reset(seed=seed, options=options)
        self._format_observation(obs)
        return obs, events


__sim: SimSingleton | None = None


def init_sim(sim_cfg):
    global __sim
    if __sim is not None:
        raise Exception("Simulation is already initialized")
    __sim = SimSingleton(sim_cfg)
    return __sim


def get_sim():
    global __sim
    if __sim is None:
        raise Exception("Simulation is not initialized")
    return __sim
