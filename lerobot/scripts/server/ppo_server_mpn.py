import os
import logging
from pprint import pformat

import gymnasium as gym
import ray
from ray.rllib.algorithms import ppo
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.env_context import EnvContext

from lerobot.common.robot_devices.robots.ur import UR
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.common.utils.utils import get_safe_torch_device
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.server.mp_nets import reset_mp_net, MPNetConfig

# Paper hyperparameters:
PPO_LEARNING_RATE = 5e-5
PPO_BATCH_SIZE = 150
PPO_CLIP_PARAM = 0.2
PPO_HIDDEN = [256, 256]
PPO_OBS_NORM = "MeanStdFilter"
PPO_REWARD_NORM = True  # RLlib: "normalize_rewards": True

PPO_NUM_WORKERS = 0
PPO_LR = 1e-5
PPO_BATCH_SIZE = 1000
PPO_LAYER_SIZE = 256
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_NUM_GPUS = 0
PPO_EVAL_INTERVAL = 8
PPO_EVAL_EPISODES = 5

def _maybe_resume_checkpoint(cfg, trainer):
    # Resume logic, similar to your off-policy code
    if cfg.resume and os.path.exists(cfg.output_dir):
        latest = ppo.PPOTrainer.latest_checkpoint(cfg.output_dir)
        if latest is not None:
            logging.info(f"Resuming from checkpoint {latest}")
            trainer.restore(latest)

class MPNetEnv(gym.Wrapper):
    def __init__(self, cfg: TrainPipelineConfig):
        self.mp_net = cfg.env
        assert isinstance(self.mp_net, MPNetConfig)

        self.robot_cfg = cfg.robot
        self.robot = None
        self.current_primitive = None
        self.prev_primitive = None
        self.policy = None
        self.online_env = None
        self.preloaded_envs = None
        self.step_counter = mp_net.get_step_counter()
        self.device = get_safe_torch_device(mp_net.device, log=True)
        self.eval_mode = False
        self.episode_cnt = 0
        self.preloaded_envs = {}

        # assert that there is only one primitive
        amp: str = None
        for name, primitive in self.mp_net.primitives.items():
            if primitive.is_adaptive:
                if amp is None:
                    amp = name
                else:
                    raise ValueError("On-policy can only be run for a single adaptive primitive")

        if amp is None:
            raise ValueError("No adaptive primitive to train")

        self._initialized = False

    def step(self, action):
        # step primitive
        obs, reward, terminated, truncated, info = self.online_env.step(action)

        # check termination condition
        done = (terminated or truncated)  # and info.get("success", False)
        self.current_primitive = self.mp_net.check_transitions(self.current_primitive, obs, done)

        if self.prev_primitive != self.current_primitive:
            terminated = True

        return obs, reward, terminated, truncated, info

    def reset(self):
        # initial setup
        if not self._initialized:
            self.policy = self.mp_net.make_policies()[0] # we assert that returns only one policy
            self.robot: UR = make_robot_from_config(self.robot_cfg)
            if self.mp_net.preload_envs:
                self.preloaded_envs = {primitive_id: p.make(self.mp_net, self.robot) for primitive_id, p in self.mp_net.primitives.items()}

            self.current_primitive = self.mp_net.primitives[self.mp_net.start_primitive]
            self.online_env = self.preloaded_envs.get(self.current_primitive.id, self.current_primitive.make(self.mp_net, self.robot))

            # Full reset at the beginning of each sequence
            obs, info = reset_mp_net(self.online_env, self.mp_net)

            self._initialized = True

        while not self.current_primitive.is_terminal:
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

        return obs, info

    def close(self):
        self.online_env.close()


@parser.wrap()
def train_cli(cfg: TrainPipelineConfig):
    """
    Synchronous PPO training using Ray RLlib.
    - Trains a PPO agent with a physical robot env.
    - Logging (wandb/local) and checkpointing matches SAC/learner setup for comparability.
    """
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

    # Init Ray
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

    # Configure PPO
    config = PPOConfig().environment(
        env=lambda: MPNetEnv(cfg),  # your env must be registered, or use a factory lambda
        env_config={},
    ).rollouts(
        num_rollout_workers=PPO_NUM_WORKERS or 0
    ).framework(
        framework="torch"
    ).training(
        lr=PPO_LR,
        train_batch_size=PPO_BATCH_SIZE,
        model={
            "fcnet_hiddens": [PPO_LAYER_SIZE]*2,
        },
        gamma=PPO_LAYER_SIZE,
        lambda_=PPO_GAE_LAMBDA
    ).resources(
        num_gpus=PPO_NUM_GPUS
    ).evaluation(
        evaluation_interval=PPO_EVAL_INTERVAL,
        evaluation_num_workers=0,
        evaluation_duration=PPO_EVAL_EPISODES,
        evaluation_duration_unit="episodes",
        evaluation_config={"explore": False}
    ).callbacks(
        {}  # can be used for custom hooks
    )

    # If you want observation normalization (optional)
    config = config.update_from_dict({"observation_filter": "MeanStdFilter"})

    # Build trainer
    trainer = config.build()

    # Optionally resume from checkpoint
    _maybe_resume_checkpoint(cfg, trainer)

    # Training loop
    for i in range(cfg.train_iterations):
        result = trainer.train()
        reward_mean = result["episode_reward_mean"]
        len_mean = result.get("episode_len_mean", None)

        logging.info(f"[PPO] Iter {i} | reward_mean={reward_mean} | len_mean={len_mean}")

        # Wandb logging
        if wandb_logger:
            wandb.log({
                "iteration": i,
                "episode_reward_mean": reward_mean,
                "episode_len_mean": len_mean,
                "timesteps_total": result.get("timesteps_total"),
            })

        # Periodic checkpoint
        if cfg.save_checkpoint and (i % cfg.save_freq == 0 or i == cfg.train_iterations - 1):
            checkpoint_path = trainer.save(cfg.output_dir)
            logging.info(f"[PPO] Saved checkpoint at {checkpoint_path}")
            if wandb_logger:
                wandb.log({"checkpoint_path": checkpoint_path, "iteration": i})

    ray.shutdown()
    if wandb_logger:
        wandb.finish()
    logging.info("PPO training complete.")

if __name__ == "__main__":
    train_cli()
