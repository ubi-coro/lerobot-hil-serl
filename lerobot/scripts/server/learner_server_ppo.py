import os
import logging
import ray
from ray.rllib.algorithms import ppo
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.env_context import EnvContext
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig

# Paper hyperparameters:
PPO_LEARNING_RATE = 5e-5
PPO_BATCH_SIZE = 150
PPO_CLIP_PARAM = 0.2
PPO_HIDDEN = [256, 256]
PPO_OBS_NORM = "MeanStdFilter"
PPO_REWARD_NORM = True  # RLlib: "normalize_rewards": True

def _maybe_resume_checkpoint(cfg, trainer):
    # Resume logic, similar to your off-policy code
    if cfg.resume and os.path.exists(cfg.output_dir):
        latest = ppo.PPOTrainer.latest_checkpoint(cfg.output_dir)
        if latest is not None:
            logging.info(f"Resuming from checkpoint {latest}")
            trainer.restore(latest)

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
        env=cfg.env.env_id,  # your env must be registered, or use a factory lambda
        env_config=cfg.env.to_dict(),
    ).rollouts(
        num_rollout_workers=cfg.policy.num_workers or 0
    ).framework(
        framework="torch"
    ).training(
        lr=cfg.policy.learning_rate,
        train_batch_size=cfg.policy.train_batch_size or 1000,
        model={
            "fcnet_hiddens": [cfg.policy.neurons_per_layer]*2,
        },
        gamma=cfg.policy.gamma if hasattr(cfg.policy, "gamma") else 0.99,
        lambda_=cfg.policy.gae_lambda if hasattr(cfg.policy, "gae_lambda") else 0.95,
    ).resources(
        num_gpus=cfg.policy.num_gpus if hasattr(cfg.policy, "num_gpus") else 0
    ).evaluation(
        evaluation_interval=cfg.eval_interval or 1,
        evaluation_num_workers=0,
        evaluation_duration=cfg.eval_episodes or 5,
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
