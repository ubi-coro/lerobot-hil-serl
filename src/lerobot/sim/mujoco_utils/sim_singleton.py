import gymnasium as gym
import numpy as np

from lerobot.sim.mujoco_utils.viewer import Viewer, PassiveViewer, create_viewer, AbstractViewer


class SimSingleton:
    def __init__(self, env: gym.Env, viewer: AbstractViewer = None):
        self.env: gym.Env = env
        self.next_action: dict = {}
        self.observation: dict = {
            'left': {
                'waist.pos': 0.0,
                "shoulder.pos": 0.0,
                "elbow.pos": 0.0,
                "forearm_roll.pos":0.0,
                "wrist_angle.pos": 0.0,
                "wrist_rotate.pos": 0.0,
                "gripper.pos": 0.0
            },
            'right': {
                'waist.pos': 0.0,
                "shoulder.pos": 0.0,
                "elbow.pos": 0.0,
                "forearm_roll.pos": 0.0,
                "wrist_angle.pos": 0.0,
                "wrist_rotate.pos": 0.0,
                "gripper.pos": 0.0
            }
        }

        self.env.reset()

        physics = self.env.envs[0].unwrapped._env.physics
        model = physics.model.ptr
        data = physics.data.ptr

        if viewer is None:
            viewer_kwargs = {
                "key": "mujoco",
                "model": model,
                "data": data,
                "image_keys": []
            }
            viewer = create_viewer(**viewer_kwargs)

        self.viewer: AbstractViewer = viewer
        self.viewer.start()

    def step(self):
        temp_action = np.array([[*self.next_action, *self.next_action]]) # TODO(jzilke) remove with real action array
        obs = self.env.step(temp_action)
        self.observation = obs
        self.viewer.sync(obs)  # TODO(jzilke) set framerate

    def get_observation(self, name: str):
        return self.observation.get(name)

    def add_action(self, name: str, action: list):
        self.next_action[name] = action

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)


__sim: SimSingleton | None = None


def init_sim(env: gym.Env, viewer: AbstractViewer = None):
    global __sim
    if __sim is not None:
        raise Exception("Simulation is already initialized")
    __sim = SimSingleton(env, viewer)


def get_sim():
    global __sim
    if __sim is None:
        raise Exception("Simulation is not initialized")
    return __sim
