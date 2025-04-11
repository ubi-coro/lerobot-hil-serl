import argparse
import time
from contextlib import nullcontext

import einops
import hydra
import numpy as np
import gymnasium as gym
import torch
from typing import Union

import sapien
from scipy.spatial.transform import Rotation as R
from omegaconf import DictConfig
from typing import Any
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.agents.robots import Panda, Fetch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.types import SimConfig, GPUMemoryConfig, Array
from pynput import keyboard
from torch import nn
from tqdm import trange

from lerobot.common.envs.factory import make_env
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import init_hydra_config, get_safe_torch_device, set_global_seed, inside_slurm
from lerobot.scripts.eval import get_pretrained_policy_path


@register_env("Reach1D-v0", max_episode_steps=50)
class Reach1DEnv(BaseEnv):
    """A minimal task: move robot's end-effector along X-axis to reach a target X position."""

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(self, target_x: float = -0.1, robot_uids="panda", robot_init_qpos_noise=0.0, **kwargs):
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
        pose = sapien_utils.look_at([0.4, 0.4, 0.6], [0.6, 0, 0.2])
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
        if success:
            x=3
        return {"success": success}

    def _get_obs_extra(self, info: dict):
        # some useful observation info for solving the task includes the pose of the tcp (tool center point) which is the point between the
        # grippers of the robot
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if self.obs_mode_struct.use_state:
            # if the observation mode requests to use state, we provide ground truth information about where the cube is.
            # for visual observation modes one should rely on the sensed visual data to determine where the cube is
            obs.update(
                goal_pos=self.goal_region.pose.p,
                obj_pose=self.obj.pose.raw_pose,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        tcp_x = self.agent.tcp.pose.p[..., 0]
        dist = torch.abs(tcp_x - self.target_x)
        reward = 1 - torch.tanh(10 * dist)
        reward[info["success"]] = 3
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        return self.compute_dense_reward(obs, action, info) / 3.0

def preprocess_maniskill_observation(
    observations: dict[str, np.ndarray],
) -> dict[str, torch.Tensor]:
    """Convert environment observation to LeRobot format observation.
    Args:
        observation: Dictionary of observation batches from a Gym vector environment.
    Returns:
        Dictionary of observation batches with keys renamed to LeRobot format and values as tensors.
    """
    # map to expected inputs for the policy
    return_observations = {}
    # TODO: You have to merge all tensors from agent key and extra key
    # You don't keep sensor param key in the observation
    # And you keep sensor data rgb
    q_pos = observations["agent"]["qpos"]
    q_vel = observations["agent"]["qvel"]
    tcp_pos = observations["extra"]["tcp_pose"]
    img = observations["sensor_data"]["base_camera"]["rgb"]

    _, h, w, c = img.shape
    assert c < h and c < w, f"expect channel last images, but instead got {img.shape=}"

    # sanity check that images are uint8
    assert img.dtype == torch.uint8, f"expect torch.uint8, but instead {img.dtype=}"

    # convert to channel first of type float32 in range [0,1]
    img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
    img = img.type(torch.float32)
    img /= 255

    state = torch.cat([q_pos, q_vel, tcp_pos], dim=-1)

    return_observations["observation.image"] = img
    return_observations["observation.state"] = state
    return return_observations


class ManiSkillObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, device: torch.device = "cuda"):
        super().__init__(env)
        self.device = device

    def observation(self, observation):
        observation = preprocess_maniskill_observation(observation)
        observation = {k: v.to(self.device) for k, v in observation.items()}
        return observation


class ManiSkillCompat(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        new_action_space_shape = env.action_space.shape[-1]
        new_low = np.squeeze(env.action_space.low, axis=0)
        new_high = np.squeeze(env.action_space.high, axis=0)
        self.action_space = gym.spaces.Box(
            low=new_low, high=new_high, shape=(new_action_space_shape,)
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        options = {}
        return super().reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = reward.item()
        terminated = terminated.item()
        truncated = truncated.item()
        return obs, reward, terminated, truncated, info


class ManiSkillActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Tuple(
            spaces=(env.action_space, gym.spaces.Discrete(2))
        )

    def action(self, action):
        action, telop = action
        return action


class ManiSkillMultiplyActionWrapper(gym.Wrapper):
    def __init__(self, env, multiply_factor: float = 1):
        super().__init__(env)
        self.multiply_factor = multiply_factor
        action_space_agent: gym.spaces.Box = env.action_space[0]
        action_space_agent.low = action_space_agent.low * multiply_factor
        action_space_agent.high = action_space_agent.high * multiply_factor
        self.action_space = gym.spaces.Tuple(
            spaces=(action_space_agent, gym.spaces.Discrete(2))
        )

    def step(self, action):
        if isinstance(action, tuple):
            action, telop = action
        else:
            telop = 0
        action = action / self.multiply_factor
        obs, reward, terminated, truncated, info = self.env.step((action, telop))
        return obs, reward, terminated, truncated, info


class ActionMaskingWrapper(gym.ActionWrapper):
    """A wrapper to mask actions, allowing movement only in the target direction."""

    def step(self, action: Array):
        if isinstance(action, tuple):
            action, telop = action
        else:
            telop = 0
        masked_action = np.zeros_like(action)
        masked_action[0] = action[0]
        return self.env.step((masked_action, telop))


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
        low = [env.action_space[0].low[..., ax][0] for ax in self.axes]
        high = [env.action_space[0].high[..., ax][0] for ax in self.axes]
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

class KeyboardControlWrapper(gym.Wrapper):
    """
    A wrapper for controlling the robot's end-effector using the keyboard.
    Arrow keys are used to move the end-effector along the X-axis.
    """

    def __init__(self, env: Reach1DEnv, ax: list | float | None = None):
        super().__init__(env)

        if ax is None:
            ax = 0
        if not isinstance(ax, list):
            ax = [ax]
        assert len(ax) <= 3, "Can only control three axes with KeyboardControlWrapper"
        self.axes = ax

        self.action = np.array([0.0, 0.0, 0.0])  # Default action value
        self.done = False
        self.intervention = False
        listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        listener.start()

    def _on_press(self, key):
        try:
            if key == keyboard.Key.right:
                self.action[0] = 1.0  # Move right (X-axis)
                self.intervention = True
            elif key == keyboard.Key.left:
                self.action[0] = -1.0  # Move left (X-axis)
                self.intervention = True
            elif key == keyboard.Key.up:
                self.action[1] = 1.0  # Move up (Y-axis)
                self.intervention = True
            elif key == keyboard.Key.down:
                self.action[1] = -1.0  # Move down (Y-axis)
                self.intervention = True
            elif key == keyboard.Key.page_up:
                self.action[2] = 1.0  # Move up (Z-axis) with Shift
                self.intervention = True
            elif key == keyboard.Key.page_down:
                self.action[2] = -1.0  # Move down (Z-axis) with Shift
                self.intervention = True
            elif key == keyboard.Key.esc:
                self.done = True  # Exit the loop
        except AttributeError:
            pass

    def _on_release(self, key):
        if key in [keyboard.Key.right, keyboard.Key.left]:
            self.action[0] = 0.0  # Stop movement along X-axis
        if key in [keyboard.Key.up, keyboard.Key.down]:
            self.action[1] = 0.0  # Stop movement along Y-axis
        if key in [keyboard.Key.page_up, keyboard.Key.page_down]:
            self.action[2] = 0.0  # Stop movement along Z-axis

        if not self.action[0] and not self.action[1]:
            self.intervention = False  # Stop intervention when no keys are pressed

    def step(self, action):
        """
        Run the environment loop with keyboard control.
        """
        if isinstance(action, tuple):
            action, telop = action
        else:
            telop = self.intervention

        if telop:
            for i, ax in enumerate(self.axes):
                action[ax] = self.action[i]


        obs, reward, done, vec, info = self.env.step(action)
        return obs, reward, self.done or done, vec, info


def make_maniskill(
    cfg: DictConfig,
    n_envs: int | None = None,
) -> gym.Env:
    """
    Factory function to create a ManiSkill environment with standard wrappers.

    Args:
        task: Name of the ManiSkill task
        obs_mode: Observation mode (rgb, rgbd, etc)
        control_mode: Control mode for the robot
        render_mode: Rendering mode
        sensor_configs: Camera sensor configurations
        n_envs: Number of parallel environments

    Returns:
        A wrapped ManiSkill environment
    """

    env = gym.make(
        cfg.env.task,
        obs_mode=cfg.env.obs,
        control_mode=cfg.env.control_mode,
        render_mode=cfg.env.render_mode,
        render_backend='sapien_cpu',
        sensor_configs={"width": cfg.env.image_size, "height": cfg.env.image_size},
        num_envs=n_envs,
    )

    if cfg.env.video_record.enabled:
        env = RecordEpisode(
            env,
            output_dir=cfg.env.video_record.record_dir,
            save_trajectory=True,
            trajectory_name=cfg.env.video_record.trajectory_name,
            save_video=True,
            video_fps=30,
        )
    env = ManiSkillObservationWrapper(env, device=cfg.env.device)
    env = ManiSkillVectorEnv(env, ignore_terminations=True, auto_reset=False)
    env._max_episode_steps = env.max_episode_steps = (
        50  # gym_utils.find_max_episode_steps_value(env)
    )
    env.unwrapped.metadata["render_fps"] = 20
    #env = ManiSkillCompat(env)
    env = ManiSkillActionWrapper(env)
    env = ManiSkillMultiplyActionWrapper(env, multiply_factor=0.03)
    #env = ActionMaskingWrapper(env)
    env = StabilizingActionMaskingWrapper(env, ref_pose=np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]))
    env = KeyboardControlWrapper(env)

    return env

def rollout(args):

    pretrained_policy_path = get_pretrained_policy_path(args.pretrained_policy_name_or_path)
    hydra_cfg = init_hydra_config(
        str(pretrained_policy_path / "config.yaml"), dict()
    )

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_global_seed(hydra_cfg.seed)

    policy = make_policy(
        hydra_cfg=hydra_cfg,
        pretrained_policy_name_or_path=str(pretrained_policy_path),
    )

    hydra_cfg.env.render_mode = "human"
    env = make_env(
        hydra_cfg,
        n_envs=1
    )

    assert isinstance(policy, nn.Module)
    policy.eval()

    with (
        torch.no_grad(),
        torch.autocast(device_type=get_safe_torch_device(hydra_cfg.device, log=True).type) if hydra_cfg.use_amp else nullcontext(),
    ):
        policy.reset()
        observation, info = env.reset(options=dict())
        done = False
        steps = 0
        while not done:
            with torch.inference_mode():
                action = policy.select_action(observation)

            # Convert to CPU / numpy.
            action = action.to("cpu").numpy()
            assert action.ndim == 2, "Action dimensions should be (batch, action_dim)"

            # Apply the next action.
            observation, reward, terminated, truncated, info = env.step(action)

            env.render()

            done = terminated | truncated | done

            time.sleep(0.05)

            steps += 1

    env.close()


def explore(args):

    # Initialize config
    with hydra.initialize(version_base=None, config_path="../../configs"):
        cfg = hydra.compose(config_name="env/maniskill_example.yaml")

    env = make_env(
        cfg,
        n_envs=1
    )

    print("env done")

    obs, info = env.reset(options=dict())
    while True:
        random_action = env.action_space.sample()
        zero_action = np.zeros_like(random_action[0])
        obs, reward, terminated, truncated, info = env.step(zero_action)
        print(reward)
        env.render()
        time.sleep(0.05)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--pretrained-policy-name-or-path",
        help=(
            "Either the repo ID of a model hosted on the Hub or a path to a directory containing weights "
            "saved using `Policy.save_pretrained`. If not provided, the policy is initialized from scratch "
            "(useful for debugging). This argument is mutually exclusive with `--config`."
        ),
    )

    explore(parser.parse_args())
