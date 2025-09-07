import logging
import time
from collections import deque
from typing import Annotated, Dict, Sequence, Tuple, Any

import gymnasium as gym
import numpy as np
import torch
import torchvision.transforms.functional as F  # noqa: N812
from gymnasium import spaces
from gymnasium.core import ObsType, WrapperObsType
from scipy.spatial.transform import Rotation as R
from torch import Tensor

from lerobot.common.robot_devices.control_utils import (
    busy_wait,
    reset_follower_position
)
from lerobot.common.utils.utils import log_say

logging.basicConfig(level=logging.INFO)


class AddJointVelocityToObservation(gym.ObservationWrapper):
    def __init__(self, env, joint_velocity_limits=100.0, fps=30, num_dof=6):
        super().__init__(env)

        # Extend observation space to include joint velocities
        old_low = self.observation_space["observation.state"].low
        old_high = self.observation_space["observation.state"].high
        old_shape = self.observation_space["observation.state"].shape

        self.last_joint_positions = np.zeros(num_dof)

        new_low = np.concatenate([old_low, np.ones(num_dof) * -joint_velocity_limits])
        new_high = np.concatenate([old_high, np.ones(num_dof) * joint_velocity_limits])

        new_shape = (old_shape[0] + num_dof,)

        self.observation_space["observation.state"] = gym.spaces.Box(
            low=new_low,
            high=new_high,
            shape=new_shape,
            dtype=np.float32,
        )

        self.dt = 1.0 / fps

    def observation(self, observation):
        joint_velocities = (observation["observation.state"] - self.last_joint_positions) / self.dt
        self.last_joint_positions = observation["observation.state"].clone()
        observation["observation.state"] = torch.cat(
            [observation["observation.state"], joint_velocities], dim=-1
        )
        return observation


class AddCurrentToObservation(gym.ObservationWrapper):
    def __init__(self, env, max_current=500, num_dof=6):
        super().__init__(env)

        # Extend observation space to include joint velocities
        old_low = self.observation_space["observation.state"].low
        old_high = self.observation_space["observation.state"].high
        old_shape = self.observation_space["observation.state"].shape

        new_low = np.concatenate([old_low, np.zeros(num_dof)])
        new_high = np.concatenate([old_high, np.ones(num_dof) * max_current])

        new_shape = (old_shape[0] + num_dof,)

        self.observation_space["observation.state"] = gym.spaces.Box(
            low=new_low,
            high=new_high,
            shape=new_shape,
            dtype=np.float32,
        )

    def observation(self, observation):
        present_current = (
            self.unwrapped.robot.follower_arms["main"].read("Present_Current").astype(np.float32)
        )
        observation["observation.state"] = torch.cat(
            [observation["observation.state"], torch.from_numpy(present_current)], dim=-1
        )
        return observation


class TimeLimitWrapper(gym.Wrapper):
    def __init__(self, env, control_time_s, fps):
        super().__init__(env)
        self.control_time_s = control_time_s
        self.fps = fps

        self.last_timestamp = 0.0
        self.episode_time_in_s = 0.0

        self.max_episode_steps = int(self.control_time_s * self.fps)

        self.current_step = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        time_since_last_step = time.perf_counter() - self.last_timestamp
        self.episode_time_in_s += time_since_last_step
        self.last_timestamp = time.perf_counter()
        self.current_step += 1
        # check if last timestep took more time than the expected fps
        if 1.0 / time_since_last_step < self.fps:
            logging.debug(f"Current timestep exceeded expected fps {self.fps}")

        if self.current_step >= self.max_episode_steps:
            terminated = True
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.episode_time_in_s = 0.0
        self.last_timestamp = time.perf_counter()
        self.current_step = 0
        return self.env.reset(seed=seed, options=options)


class ImageCropResizeWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        crop_params_dict: Dict[str, Tuple[int, int, int, int]],
        resize_size=None,
    ):
        super().__init__(env)
        self.env = env
        self.crop_params_dict = crop_params_dict
        for key_crop in crop_params_dict:
            if key_crop not in self.env.observation_space.keys():  # noqa: SIM118
                raise ValueError(f"Key {key_crop} not in observation space")
        for key in crop_params_dict:
            new_shape = (3, resize_size[0], resize_size[1])
            self.observation_space[key] = gym.spaces.Box(low=0, high=255, shape=new_shape)

        self.resize_size = resize_size
        if self.resize_size is None:
            self.resize_size = (128, 128)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        for k in self.crop_params_dict:
            device = obs[k].device
            if obs[k].dim() >= 3:
                # Reshape to combine height and width dimensions for easier calculation
                batch_size = obs[k].size(0)
                channels = obs[k].size(1)
                flattened_spatial_dims = obs[k].view(batch_size, channels, -1)

                # Calculate standard deviation across spatial dimensions (H, W)
                # If any channel has std=0, all pixels in that channel have the same value
                # This is helpful if one camera mistakenly covered or the image is black
                #std_per_channel = torch.std(flattened_spatial_dims, dim=2)
                #if (std_per_channel <= 0.02).any():
                #    logging.warning(
                #        f"Potential hardware issue detected: All pixels have the same value in observation {k}"
                #    )

            if device == torch.device("mps:0"):
                obs[k] = obs[k].cpu()

            obs[k] = F.crop(obs[k], *self.crop_params_dict[k])
            obs[k] = F.resize(obs[k], self.resize_size)
            # TODO (michel-aractingi): Bug in resize, it returns values outside [0, 1]
            obs[k] = obs[k].clamp(0.0, 1.0)
            obs[k] = obs[k].to(device)

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        for k in self.crop_params_dict:
            device = obs[k].device
            if device == torch.device("mps:0"):
                obs[k] = obs[k].cpu()
            obs[k] = F.crop(obs[k], *self.crop_params_dict[k])
            obs[k] = F.resize(obs[k], self.resize_size)
            obs[k] = obs[k].clamp(0.0, 1.0)
            obs[k] = obs[k].to(device)
        return obs, info


class ConvertToLeRobotObservation(gym.ObservationWrapper):
    def __init__(self, env, device: str = "cpu"):
        super().__init__(env)

        example_obs = env.unwrapped.robot.capture_observation()
        for key in env.observation_space:
            if "image" in key:
                env.observation_space[key] = gym.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=example_obs[key].permute(2, 0, 1).shape,
                    dtype=np.float32
                )
        self.device = torch.device(device)

    def observation(self, observation):
        for key in observation:
            observation[key] = observation[key].float()
            if "image" in key:
                h, w, c = observation[key].shape
                if c < h and c < w:
                    observation[key] = observation[key].permute(2, 0, 1)
                observation[key] /= 255.0
        observation = {
            key: observation[key].to(self.device, non_blocking=self.device.type == "cuda")
            for key in observation
        }

        return observation


class ResetWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        reset_pose: np.ndarray | None = None,
        reset_time_s: float = 5,
    ):
        super().__init__(env)
        self.reset_time_s = reset_time_s
        self.reset_pose = reset_pose
        self.robot = self.unwrapped.robot

    def reset(self, *, seed=None, options=None):
        start_time = time.perf_counter()
        if self.reset_pose is not None:
            log_say("Reset the environment.", play_sounds=True)
            reset_follower_position(self.robot.follower_arms["main"], self.reset_pose)
            log_say("Reset the environment done.", play_sounds=True)

            if len(self.robot.leader_arms) > 0:
                self.robot.leader_arms["main"].write("Torque_Enable", 1)
                log_say("Reset the leader robot.", play_sounds=True)
                reset_follower_position(self.robot.leader_arms["main"], self.reset_pose)
                log_say("Reset the leader robot done.", play_sounds=True)
        else:
            log_say(
                f"Manually reset the environment for {self.reset_time_s} seconds.",
                play_sounds=True,
            )
            start_time = time.perf_counter()
            while time.perf_counter() - start_time < self.reset_time_s:
                self.robot.teleop_step()

            log_say("Manual reset of the environment done.", play_sounds=True)

        busy_wait(self.reset_time_s - (time.perf_counter() - start_time))

        return super().reset(seed=seed, options=options)


class BatchCompatibleWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for key in observation:
            if "image" in key and observation[key].dim() == 3:
                observation[key] = observation[key].unsqueeze(0)
            if "state" in key and observation[key].dim() == 1:
                observation[key] = observation[key].unsqueeze(0)
            if "velocity" in key and observation[key].dim() == 1:
                observation[key] = observation[key].unsqueeze(0)
        return observation


class TorchBox(gym.spaces.Box):
    """A version of gym.spaces.Box that handles PyTorch tensors.

    This class extends gym.spaces.Box to work with PyTorch tensors,
    providing compatibility between NumPy arrays and PyTorch tensors.
    """

    def __init__(
        self,
        low: float | Sequence[float] | np.ndarray,
        high: float | Sequence[float] | np.ndarray,
        shape: Sequence[int] | None = None,
        np_dtype: np.dtype | type = np.float32,
        torch_dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        seed: int | np.random.Generator | None = None,
    ) -> None:
        super().__init__(low, high, shape=shape, dtype=np_dtype, seed=seed)
        self.torch_dtype = torch_dtype
        self.device = device

    def sample(self) -> torch.Tensor:
        arr = super().sample()
        return torch.as_tensor(arr, dtype=self.torch_dtype, device=self.device)

    def contains(self, x: torch.Tensor) -> bool:
        # Move to CPU/numpy and cast to the internal dtype
        arr = x.detach().cpu().numpy().astype(self.dtype, copy=False)
        return super().contains(arr)

    def seed(self, seed: int | np.random.Generator | None = None):
        super().seed(seed)
        return [seed]

    def __repr__(self) -> str:
        return (
            f"TorchBox({self.low_repr}, {self.high_repr}, {self.shape}, "
            f"np={self.dtype.name}, torch={self.torch_dtype}, device={self.device})"
        )


class TorchActionWrapper(gym.Wrapper):
    """
    The goal of this wrapper is to change the action_space.sample()
    to torch tensors.
    """

    def __init__(self, env: gym.Env, device: str = "cuda"):
        super().__init__(env)
        self.action_space = TorchBox(
            low=env.action_space.low,
            high=env.action_space.high,
            shape=env.action_space.shape,
            torch_dtype=torch.float32,
            device=device,
        )

    def step(self, action: torch.Tensor):
        if action.dim() == 2:
            action = action.squeeze(0)
        #action = action.detach().cpu().numpy()
        return self.env.step(action)


class StabilizingActionMaskingWrapper(gym.ActionWrapper):
    """
    A wrapper that:
    1. Restricts motion to a single axis (e.g., 'x').
    2. Stabilizes all other axes using a proportional controller.
    3. Expects a scalar action input corresponding to the selected axis.
    """

    def __init__(
        self,
        env,
        ax: list | float | None = None,
        ref_pose: np.ndarray = None,
        kp_pos: float = 0.1,
        kp_rot: float = 0.1,
    ):
        super().__init__(env)

        if ax is None:
            ax = 0
        if not isinstance(ax, list):
            ax = [ax]
        self.axes = ax

        self.kp_pos = kp_pos
        self.kp_rot = kp_rot

        # Reference pose: 3D position + 3D rotation as quaternion
        self.ref_pose = ref_pose  # np.ndarray with shape (7,)

        # Action space becomes 1D scalar for the selected axis
        low = [env.action_space[0].low[..., ax] for ax in self.axes]
        high = [env.action_space[0].high[..., ax] for ax in self.axes]
        self.action_space = gym.spaces.Tuple(
            spaces=(
                gym.spaces.Box(
                    low=np.array(low),
                    high=np.array(high),
                    shape=(len(self.axes),), dtype=np.float32
                ),
                gym.spaces.Discrete(2)  # keep teleop flag
            )
        )

    def step(self, action):
        if isinstance(action, tuple):
            action, telop = action
        else:
            if action.ndim > 1:
                action = action.squeeze(0)
            telop = 0

        # Initialize full action vector
        full_action = np.zeros_like(self.env.action_space[0].low).squeeze()

        if self.ref_pose is None:
            # Get current pose (position + quaternion) from environment
            current_pose = self.env.unwrapped.agent.tcp.pose.raw_pose.squeeze().numpy()
            self.ref_pose = current_pose.copy()

        # Decompose current and reference pose
        current_pos = self.env.unwrapped.agent.tcp.pose.p.squeeze().numpy()
        current_quat = self.env.unwrapped.agent.tcp.pose.q.squeeze().numpy()
        ref_pos = self.ref_pose[:3]
        ref_quat = self.ref_pose[3:]

        # --- Positional control ---
        delta_pos = np.zeros(3)

        # Stabilize other axes
        pos_error = ref_pos - current_pos
        pos_correction = self.kp_pos * pos_error
        delta_pos += pos_correction


        # --- Rotational control ---
        current_r = R.from_quat(current_quat)
        ref_r = R.from_quat(ref_quat)
        delta_r = (ref_r * current_r.inv()).as_rotvec()  # axis-angle difference
        delta_rot = self.kp_rot * delta_r

        full_action[:3] = delta_pos
        # full_action[3:6] = delta_rot

        for ax in self.axes:
            full_action[ax] = action[ax]

        return self.env.step((full_action, telop))


class FrameStackStateWrapper(gym.ObservationWrapper):
    """
    Stack last `n` frames for all observation keys whose name contains 'state' (case-insensitive),
    flatten them to 1-D, and return as torch.Tensors on the same device/dtype as the source values.

    - Nested Dict observations supported.
    - Non-'state' keys are passed through unchanged.
    - On reset, buffers are prefilled with the first observation (no zeros).
    """

    def __init__(self, env, n: int = 4):
        super().__init__(env)
        assert n >= 1
        self.n = n
        # per fully-qualified key: deque[Tensor], device, dtype, flat_dim
        self._buffers: Dict[Tuple[str, ...], deque] = {}
        self._meta: Dict[Tuple[str, ...], Dict[str, Any]] = {}

        # Build observation space: make state-like Boxes flat with length n * prod
        self.observation_space = self._build_obs_space(env.observation_space)

    # ---------- space helpers ----------
    def _build_obs_space(self, space):
        if isinstance(space, spaces.Dict):
            out = {}
            for k, v in space.spaces.items():
                if "state" in k.lower():
                    if isinstance(v, spaces.Box):
                        # Flatten & replicate bounds n times
                        low = np.ravel(v.low)
                        high = np.ravel(v.high)
                        low = np.tile(low, self.n)
                        high = np.tile(high, self.n)
                        out[k] = spaces.Box(low=low, high=high, dtype=v.dtype)
                    elif isinstance(v, spaces.Dict):
                        out[k] = self._build_obs_space(v)
                    else:
                        out[k] = v  # leave as-is if it's not a Box (rare)
                else:
                    out[k] = self._build_obs_space(v) if isinstance(v, spaces.Dict) else v
            return spaces.Dict(out)
        return space

    # ---------- device/dtype-safe conversion ----------
    @staticmethod
    def _as_tensor_like(x, ref: Tensor | None) -> Tensor:
        """
        Convert x (np.ndarray or Tensor) to a Tensor matching device/dtype of ref if provided;
        otherwise keep device of x if it's already a tensor, else default to CPU float32.
        """
        if isinstance(x, Tensor):
            return x  # already a tensor; keep its device/dtype
        t = torch.from_numpy(np.asarray(x))
        if ref is not None:
            return t.to(device=ref.device, dtype=ref.dtype, copy=False)
        return t  # CPU default

    @staticmethod
    def _flatten1d(t: Tensor) -> Tensor:
        return t.reshape(-1)

    # ---------- lifecycle ----------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._init_buffers(obs)
        return self.observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info

    # ---------- core ----------
    def observation(self, obs):
        return self._process(obs, prefix=())

    def _process(self, node, prefix):
        out = {}
        for k, v in node.items():
            kp = prefix + (k,)
            if isinstance(v, dict):
                out[k] = self._process(v, kp)
                continue

            if "state" in k.lower():
                # per-key meta (device/dtype) decided at reset from first obs
                meta = self._meta[kp]
                buf = self._buffers[kp]

                # ensure tensor on same device/dtype as initial
                t = self._as_tensor_like(v, meta["ref"])
                t = self._flatten1d(t)
                buf.append(t)  # append keeps only last n

                # concatenate in time order (oldest -> newest)
                stacked = torch.cat(list(buf), dim=0)
                out[k] = stacked
            else:
                out[k] = v  # unchanged (can be np or tensor; you control upstream)
        return out

    def _init_buffers(self, obs):
        self._buffers.clear()
        self._meta.clear()

        def _walk(d, prefix=()):
            for k, v in d.items():
                kp = prefix + (k,)
                if isinstance(v, dict):
                    _walk(v, kp)
                    continue

                if "state" in k.lower():
                    # decide device/dtype from the very first value (if tensor)
                    ref = v if isinstance(v, Tensor) else None
                    t0 = self._as_tensor_like(v, ref)
                    t0 = self._flatten1d(t0)
                    buf = deque(maxlen=self.n)
                    for _ in range(self.n):
                        buf.append(t0.clone())  # clone so future in-place ops won’t alias
                    self._buffers[kp] = buf
                    self._meta[kp] = {"ref": t0}  # remember device/dtype
        _walk(obs)

