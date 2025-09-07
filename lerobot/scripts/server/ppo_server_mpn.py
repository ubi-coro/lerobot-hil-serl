import datetime
import os
import logging
import time
from pprint import pformat
import pprint

import gymnasium.spaces
import numpy as np
import ray
from gymnasium import Env
from ray.rllib.algorithms import ppo
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
import torch

import lerobot.experiments
from lerobot.common.robot_devices.robots.ur import UR
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.utils.utils import get_safe_torch_device
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.server.mp_nets import reset_mp_net, MPNetConfig

NUM_FRAME_STACKS = 2

# Paper hyperparameters:
PPO_LEARNING_RATE = 5e-5
PPO_BATCH_SIZE = 150
PPO_CLIP_PARAM = 0.2
PPO_HIDDEN = [256, 256]
PPO_OBS_NORM = "MeanStdFilter"
PPO_REWARD_NORM = True  # RLlib: "normalize_rewards": True

PPO_NUM_WORKERS = 1
PPO_LR = 5e-5
PPO_BATCH_SIZE = 1000
PPO_LAYER_SIZE = 256
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 1.0
PPO_NUM_GPUS = 1
PPO_EVAL_INTERVAL = 8
PPO_EVAL_EPISODES = 5
PPO_TRAIN_STEPS = 500

OBS_LOW = [-0.03, -0.03, -0.03, -0.5, -0.5, -0.5,  # v_x-c
                            -6.0, -6.0, -11.0, # f_x-z
                            -2.0, -2.0, -0.8, # f_a-c
                            -0.005,  # p_z
                            ] * NUM_FRAME_STACKS
OBS_HIGH = [0.03, 0.03, 0.03, 0.5, 0.5, 0.5,
                            6.0, 6.0, 3.0,
                            2.0, 2.0, 0.8,
                            1.1 * 0.032,
                            ] * NUM_FRAME_STACKS
ACTION_LOW = [-0.02, -0.02, -0.75]
ACTION_HIGH = [0.02, 0.02, 0.75]

OBS_LOW = [2 * o for o in OBS_LOW]
OBS_HIGH = [2 * o for o in OBS_HIGH]

def _maybe_resume_checkpoint(cfg, trainer):
    # Resume logic, similar to your off-policy code
    if cfg.resume and os.path.exists(cfg.output_dir):
        latest = ppo.PPOTrainer.latest_checkpoint(cfg.output_dir)
        if latest is not None:
            logging.info(f"Resuming from checkpoint {latest}")
            trainer.restore(latest)

class MPNetEnv(Env):
    def __init__(self, cfg: TrainPipelineConfig):
        self.mp_net = cfg.env
        self.mp_net.device = "cpu"
        assert isinstance(self.mp_net, MPNetConfig)

        # assert that there is only one primitive
        amp = None
        for name, primitive in self.mp_net.primitives.items():
            if primitive.is_adaptive:
                if amp is None:
                    amp = name
                else:
                    raise ValueError("On-policy can only be run for a single adaptive primitive")

        if amp is None:
            raise ValueError("No adaptive primitive to train")

        self.robot: UR = make_robot_from_config(self.mp_net.robot)
        self.mp_net.preload_envs = False

        self.preloaded_envs = {}
        if self.mp_net.preload_envs:
            self.preloaded_envs = {primitive_id: p.make(self.mp_net, self.robot)
                                   for primitive_id, p in self.mp_net.primitives.items()}

        self.current_primitive = self.mp_net.primitives[self.mp_net.start_primitive]
        self.online_env = self.preloaded_envs.get(self.current_primitive.id, self.current_primitive.make(self.mp_net, self.robot))

        self.prev_primitive = None
        self.preloaded_envs = None
        self.step_counter = self.mp_net.get_step_counter()
        self.device = get_safe_torch_device(self.mp_net.device, log=True)
        self.eval_mode = False
        self.episode_cnt = 0
        self.preloaded_envs = {}

        self.action_space = gymnasium.spaces.Box(
            low=np.array(ACTION_LOW),
            high=np.array(ACTION_HIGH),
            shape=(len(ACTION_LOW), ),
        )
        self.observation_space = gymnasium.spaces.Dict({
            "observation.state": gymnasium.spaces.Box(
                low=np.array(OBS_LOW),
                high=np.array(OBS_HIGH),
                shape=(len(OBS_LOW), ),
            ),
            "observation.image.main": gymnasium.spaces.Box(
                low=0.0,
                high=1.0,
                shape=[128, 128, 3],
                dtype=np.float32
            )
        })

        self._initialized = False

    def step(self, action):
        start_time = time.perf_counter()

        action = torch.tensor(action)

        # step primitive
        obs, reward, terminated, truncated, info = self.online_env.step(action)

        # check termination condition
        done = (terminated or truncated)  # and info.get("success", False)
        self.current_primitive = self.mp_net.check_transitions(self.current_primitive, obs, done)

        if self.prev_primitive != self.current_primitive:
            terminated = True

        if self.mp_net.fps is not None:
            dt_time = time.perf_counter() - start_time
            busy_wait(1 / self.mp_net.fps - dt_time)

        return self.observation(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        # initial setup
        if not self._initialized:
            # Full reset at the beginning of each sequence
            self._initialized = True

        self.current_primitive = self.mp_net.primitives[self.mp_net.start_primitive]
        obs, info = reset_mp_net(self.online_env, self.mp_net)

        while not self.current_primitive.is_adaptive:
            start_time = time.perf_counter()
            self.prev_primitive = self.current_primitive

            action = self.online_env.action_space.sample()
            obs, reward, terminated, truncated, info = self.online_env.step(action)

            # Check stop triggered by transition function
            done = (terminated or truncated)  # and info.get("success", False)
            self.current_primitive = self.mp_net.check_transitions(self.current_primitive, obs, done)

            if self.prev_primitive != self.current_primitive:
                if not self.mp_net.preload_envs:
                    self.online_env.close()

                do_hard_reset = False
                if self.current_primitive.is_terminal:
                    do_hard_reset = True
                    self.current_primitive = self.mp_net.primitives[self.mp_net.start_primitive]

                self.online_env = self.preloaded_envs.get(self.current_primitive.id, self.current_primitive.make(self.mp_net, self.robot))

                if do_hard_reset:
                    obs, info = reset_mp_net(self.online_env, self.mp_net)
                else:
                    obs, info = self.online_env.reset()

            if self.mp_net.fps is not None:
                dt_time = time.perf_counter() - start_time
                busy_wait(1 / self.mp_net.fps - dt_time)

        self.prev_primitive = self.current_primitive

        return self.observation(obs), info

    def close(self):
        self.online_env.close()

    def observation(self, obs):
        out = {}
        for k, v in obs.items():
            # to numpy
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            v = np.asarray(v)

            # squeeze leading batch dim if present
            if v.ndim >= 1 and v.shape[0] == 1:
                v = v[0]

            if k == "observation.image.main":
                # convert to HWC
                if v.ndim == 3 and v.shape[0] in (1, 3):  # CHW -> HWC
                    v = np.transpose(v, (1, 2, 0))
                elif v.ndim == 4 and v.shape[1] in (1, 3):  # BCHW -> BHWC (we already squeezed above, so unlikely)
                    v = np.transpose(v, (0, 2, 3, 1))
                    if v.shape[0] == 1:
                        v = v[0]
                # ensure dtype/scale matches space
                v = v.astype(np.float32)
                # if it’s 0..255, normalize to 0..1 (your Box is [0,1] float32)
                if v.max() > 1.5:
                    v = v / 255.0

            elif k == "observation.state":
                v = v.astype(np.float32)

            out[k] = v
        return out


def run_env(cfg):
    env = MPNetEnv(cfg)

    while True:
        env.reset()

        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated


@parser.wrap()
def train_cli(cfg: TrainPipelineConfig):
    """
    Synchronous PPO training using Ray RLlib.
    - Trains a PPO agent with a physical robot env.
    - Logging (wandb/local) and checkpointing matches SAC/learner setup for comparability.
    """
    #run_env(cfg)

    now = datetime.datetime.now()
    cfg.env.root = cfg.env.root.replace("rlpd", "ppo")
    cfg.output_dir = os.path.join(cfg.env.root, "run", f"learner-{now:%Y-%m-%d}-{now:%H-%M-%S}")

    # Setup logging
    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"ppo_trainer_{cfg.job_name or 'ppo'}.log")
    logging.basicConfig(filename=log_file, level=logging.INFO)
    logging.info("PPO trainer logging initialized")
    logging.info(pformat(cfg.to_dict()))

    # Setup wandb (if enabled)
    wandb_logger = None
    if cfg.wandb.enable and cfg.wandb.project:
        try:
            import wandb
            wandb.init(
                project=cfg.wandb.project,
                config=cfg.to_dict(),
                name=cfg.job_name or "ppo_run",
                dir=cfg.output_dir,
                mode="online" if cfg.wandb.enable else "disabled"
            )
            wandb_logger = wandb
        except ImportError:
            logging.warning("wandb enabled in config, but not installed.")

    register_env(cfg.env.type, lambda _: MPNetEnv(cfg))

    # Init Ray
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

    # Configure PPO
    config = (
        PPOConfig()
        .environment(env=cfg.env.type, env_config={})
        .env_runners(num_env_runners=0, batch_mode="complete_episodes", rollout_fragment_length="auto")  # single local runner for real robot
        .framework("torch")
        # 👇 turn OFF new API stack
        .api_stack(enable_rl_module_and_learner=False,
                   enable_env_runner_and_connector_v2=False)
        .training(
            lr=PPO_LR,
            train_batch_size=256,  # how many env-steps to gather before returning
            minibatch_size=256,  # one SGD chunk (can be <= train_batch_size)
            num_epochs=1,  # like num_sgd_iter
            gamma=PPO_GAMMA,
            lambda_=PPO_GAE_LAMBDA,
            clip_param=PPO_CLIP_PARAM,
            # Your Dict obs: {"observation.state": (26,), "observation.image.main": (3,128,128)}
            # Old stack can auto-handle Dicts: CNN on image + MLP on state -> concat.
            model={
                # CNN for the image branch
                "conv_filters": [
                    [32, 8, 4],
                    [64, 4, 2],
                    [64, 3, 1],
                ],
                "conv_activation": "relu",
                # IMPORTANT: your image is (C,H,W)=(3,128,128)
                #"conv_format": "NCHW",
                # MLP after concat
                "post_fcnet_hiddens": [256, 256],
                "vf_share_layers": True,
            },
        )
        .resources(num_gpus=PPO_NUM_GPUS)
        .evaluation(
            evaluation_interval=None,
            evaluation_num_workers=0,
            evaluation_duration=PPO_EVAL_EPISODES,
            evaluation_duration_unit="episodes",
            evaluation_config={"explore": False}
        )
    )

    # (optional) observation normalization
    config = config.update_from_dict({"observation_filter": "MeanStdFilter"})
    # Build trainer
    trainer = config.build()

    # Optionally resume from checkpoint
    _maybe_resume_checkpoint(cfg, trainer)

    # Training loop
    pp = pprint.PrettyPrinter(depth=4)
    episode_cnt = 0
    for i in range(PPO_TRAIN_STEPS):
        result = trainer.train()

        pp.pprint(result)

        step = result["info"]["num_env_steps_sampled"]
        episodes = result["env_runners"]["episodes_this_iter"]
        hist_episode_len = result["env_runners"]["hist_stats"]["episode_lengths"][-episodes:]
        max_episode_len = int(cfg.env.primitives["insert"].wrapper.control_time_s * cfg.env.fps)
        hist_success = [l < max_episode_len for l in hist_episode_len]

        success_rate = sum(hist_success) / len(hist_success)
        mean_cycle_time = sum(hist_episode_len) / len(hist_episode_len)
        episode_cnt += episodes

        logging.info(f"[PPO] Iter {i} | success={success_rate:.3f} | cycle={mean_cycle_time:.3f} steps")

        # Wandb logging
        if wandb_logger:
            wandb.log({
                "Iteration": i,
                "Episode": episode_cnt,
                "Success Rate": success_rate,
                "Cycle Time": mean_cycle_time,
                "Interaction step": step,
            })

        # Periodic checkpoint
        if cfg.save_checkpoint and (i % cfg.save_freq) == 0:
            checkpoint_path = trainer.save(cfg.output_dir)
            logging.info(f"[PPO] Saved checkpoint at {checkpoint_path}")

    ray.shutdown()
    if wandb_logger:
        wandb.finish()
    logging.info("PPO training complete.")

if __name__ == "__main__":
    train_cli()


