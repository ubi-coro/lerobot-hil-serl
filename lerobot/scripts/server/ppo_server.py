import os
import logging
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from my_base_env import MyBaseEnv  # replace with your actual env class

@parser.wrap()
def train_cli(cfg: TrainPipelineConfig):
    """
    Synchronous PPO training using Ray RLlib.
    """
    # Initialize Ray
    ray.init(ignore_reinit_error=True, include_dashboard=False)

    # Register custom Gym environment
    env_name = f"my_env_{cfg.job_name}"
    register_env(env_name, lambda _: MyBaseEnv(cfg))

    # Configure PPO
    ppo_config = {
        "env": env_name,
        "framework": "torch",
        "num_workers": cfg.ray.num_workers,
        "num_gpus": cfg.ray.num_gpus or 0,
        "rollout_fragment_length": cfg.policy.rollout_fragment_length,
        "train_batch_size": cfg.policy.train_batch_size,
        "sgd_minibatch_size": cfg.policy.sgd_minibatch_size,
        "batch_mode": "truncate_episodes",
        # Model architecture (if using vision): adjust as needed
        "model": {
            "conv_filters": cfg.policy.conv_filters,
            "fcnet_hiddens": cfg.policy.fcnet_hiddens,
        },
        # Logging
        "log_level": cfg.log_level,
    }

    # Initialize PPO trainer
    trainer = PPOTrainer(config=ppo_config)

    # Training loop
    for i in range(cfg.train_iterations):
        result = trainer.train()
        logging.info(f"Iteration {i}: reward_mean = {result['episode_reward_mean']}")
        # Optional: log to WandB if enabled in cfg
        if cfg.wandb.enable:
            import wandb
            wandb.log({
                'iteration': i,
                'episode_reward_mean': result['episode_reward_mean'],
                'episode_len_mean': result.get('episode_len_mean', None)
            })
        # Periodic checkpoint
        if i % cfg.save_freq == 0 and cfg.save_checkpoint:
            chkpt = trainer.save(cfg.output_dir)
            logging.info(f"Saved checkpoint at {chkpt}")

    # Shutdown Ray
    ray.shutdown()

if __name__ == "__main__":
    train_cli()
