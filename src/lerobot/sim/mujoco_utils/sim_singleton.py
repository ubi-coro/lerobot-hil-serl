import gymnasium as gym
import numpy as np

from lerobot.envs.factory import make_env, make_env_config
from lerobot.sim.mujoco_utils.viewer import create_viewer, AbstractViewer


class SimSession:
    """Owns a single MuJoCo gym environment and its live viewer."""
    def __init__(self, sim_config):
        """Create env, viewer, and internal state."""
        env_cfg = make_env_config(sim_config.type)
        gym_env = make_env(env_cfg)
        self.env: gym.Env = gym_env.get(sim_config.type).get(0)

        self.next_action: dict = {}
        self.observation: dict = {}

        self.action_order = []
        self.reset()

        physics = self.env.envs[0].unwrapped._env.physics
        model = physics.model.ptr
        data = physics.data.ptr

        viewer_kwargs = {
            "key": sim_config.viewer,
            "model": model,
            "data": data,
            "image_keys": sim_config.image_keys,
        }
        viewer = create_viewer(**viewer_kwargs)

        self.viewer: AbstractViewer = viewer
        self.viewer.start()


    def step(self):
        """Step sim with staged action."""
        action = self._format_next_action()
        obs, reward, terminated, truncated, events = self.env.step(action)
        self._store_observation(obs)
        self.viewer.sync(obs)  # TODO(jzilke) set framerate
        return self.observation, reward, terminated, truncated, events


    def _format_next_action(self):
        """Convert staged joint commands to env action vector."""
        return np.array([[self.next_action.get(joint, -1.0) for joint in self.action_order]])


    def _store_observation(self, obs):
        """Cache env observation in flattened dict form."""
        new = {}

        new |= obs["pixels"]

        pos = obs["agent_pos"][0]
        new |= {self.action_order[i]: pos[i] for i in range(len(self.action_order))}

        for k, v in self.observation.items():
            if k not in new:
                new[k] = v # keep last

        self.observation = new


    def get_observation(self, name: str):
        """Return a single cached observation entry."""
        if name not in self.observation:
            raise KeyError(name)
        return self.observation[name]


    def add_action(self, name, action):
        """Stage joint commands."""
        _action = {f"{name}.{key}": value for key, value in action.items()}
        self.next_action |= _action


    def reset(self, seed=None, options=None):
        """Reset sim env and cached obsservation."""
        obs, events = self.env.reset(seed=seed, options=options)
        self._store_observation(obs)
        return obs, events


class SimManager:
    """Singleton access to the active simulation."""
    _sim: SimSession = None

    @classmethod
    def init(cls, cfg):
        """Create the global simulation instance."""
        if cls._sim is not None:
            raise RuntimeError("Simulation already initialized")
        cls._sim = SimSession(cfg)
        return cls._sim

    @classmethod
    def get(cls):
        """Return the active simulation instance."""
        if cls._sim is None:
            raise RuntimeError("Simulation not initialized")
        return cls._sim

