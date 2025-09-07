# !/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import datetime
import logging
import os
import pathlib
import queue
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from pathlib import Path
from pprint import pformat
from typing import Optional

import grpc
import hilserl_pb2_grpc  # type: ignore
import torch
from termcolor import colored
from torch import nn
from torch.multiprocessing import Queue
from torch.optim.optimizer import Optimizer

from lerobot.common.constants import (
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    PRETRAINED_MODEL_DIR,
    TRAINING_STATE_DIR,
)
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.sac.configuration_sac import SACConfig
from lerobot.common.policies.sac.modeling_sac import SACPolicy
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.train_utils import (
    load_training_state as utils_load_training_state,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.experiments import *
from lerobot.scripts.server import learner_service
from lerobot.scripts.server.buffer import ReplayBuffer, concatenate_batch_transitions
from lerobot.scripts.server.mp_nets import MPNetConfig
from lerobot.scripts.server.network_utils import (
    bytes_to_python_object,
    bytes_to_transitions,
    state_to_bytes,
)
from lerobot.scripts.server.utils import (
    move_state_dict_to_device,
    move_transition_to_device,
    setup_process_handlers,
)

LOG_PREFIX = "[LEARNER]"

logging.basicConfig(level=logging.INFO)

torch._dynamo.config.suppress_errors = True

#################################################
# MAIN ENTRY POINTS AND CORE ALGORITHM FUNCTIONS #
#################################################


@parser.wrap()
def train_cli(cfg: TrainPipelineConfig):
    if not use_threads(cfg):
        import torch.multiprocessing as mp

        mp.set_start_method("spawn")

    # Use the job_name from the config
    train(
        cfg,
        job_name=cfg.job_name,
    )

    logging.info("[LEARNER] train_cli finished")


def train(cfg: TrainPipelineConfig, job_name: str | None = None):
    """
    Main training function that initializes and runs the training process.

    Args:
        cfg (TrainPipelineConfig): The training configuration
        job_name (str | None, optional): Job name for logging. Defaults to None.
    """

    cfg.validate()
    cfg.policy = SACConfig()

    if not cfg.resume:
        now = datetime.datetime.now()
        cfg.output_dir = os.path.join(cfg.env.root, "run", f"learner-{now:%Y-%m-%d}-{now:%H-%M-%S}")

    if job_name is None:
        job_name = cfg.job_name

    if job_name is None:
        raise ValueError("Job name must be specified either in config or as a parameter")

    # Create logs directory to ensure it exists
    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"learner_{job_name}.log")

    # Initialize logging with explicit log file
    init_logging(log_file=log_file)
    logging.info(f"Learner logging initialized, writing to {log_file}")

    # Setup WandB logging if enabled
    if cfg.wandb.enable and cfg.wandb.project:
        from lerobot.common.utils.wandb_utils import WandBLogger

        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    # Handle resume logic
    #cfg = handle_resume_logic(cfg)

    set_seed(seed=cfg.env.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    shutdown_event = setup_process_handlers(use_threads(cfg))

    start_learner_threads(
        cfg=cfg,
        wandb_logger=wandb_logger,
        shutdown_event=shutdown_event,
    )


def start_learner_threads(
    cfg: TrainPipelineConfig,
    wandb_logger: WandBLogger | None,
    shutdown_event: any,  # Event,
) -> None:
    """
    Start the learner threads for training.

    Args:
        cfg (TrainPipelineConfig): Training configuration
        wandb_logger (WandBLogger | None): Logger for metrics
        shutdown_event: Event to signal shutdown
    """
    # Create multiprocessing queues
    transition_queue = Queue(maxsize=10)
    interaction_message_queue = Queue(maxsize=10)
    parameters_queue = Queue(maxsize=1)

    concurrency_entity = None

    if use_threads(cfg):
        from threading import Thread

        concurrency_entity = Thread
    else:
        from torch.multiprocessing import Process

        concurrency_entity = Process

    communication_process = concurrency_entity(
        target=start_learner_server,
        args=(
            parameters_queue,
            transition_queue,
            interaction_message_queue,
            shutdown_event,
            cfg,
        ),
        daemon=True,
    )
    communication_process.start()

    add_actor_information_and_train(
        cfg=cfg,
        wandb_logger=wandb_logger,
        shutdown_event=shutdown_event,
        transition_queue=transition_queue,
        interaction_message_queue=interaction_message_queue,
        parameters_queue=parameters_queue,
    )
    logging.info("[LEARNER] Training process stopped")

    logging.info("[LEARNER] Closing queues")
    transition_queue.close()
    interaction_message_queue.close()
    parameters_queue.close()

    communication_process.join()
    logging.info("[LEARNER] Communication process joined")

    logging.info("[LEARNER] join queues")
    transition_queue.cancel_join_thread()
    interaction_message_queue.cancel_join_thread()
    parameters_queue.cancel_join_thread()

    logging.info("[LEARNER] queues closed")


#################################################
# Core algorithm functions #
#################################################


def add_actor_information_and_train(
    cfg: TrainPipelineConfig,
    wandb_logger: WandBLogger | None,
    shutdown_event: any,  # Event,
    transition_queue: Queue,
    interaction_message_queue: Queue,
    parameters_queue: Queue,
):
    """
    Handles data transfer from the actor to the learner, manages training updates,
    and logs training progress in an online reinforcement learning setup.

    This function continuously:
    - Transfers transitions from the actor to the replay buffer.
    - Logs received interaction messages.
    - Ensures training begins only when the replay buffer has a sufficient number of transitions.
    - Samples batches from the replay buffer and performs multiple critic updates.
    - Periodically updates the actor, critic, and temperature optimizers.
    - Logs training statistics, including loss values and optimization frequency.

    NOTE: This function doesn't have a single responsibility, it should be split into multiple functions
    in the future. The reason why we did that is the  GIL in Python. It's super slow the performance
    are divided by 200. So we need to have a single thread that does all the work.

    Args:
        cfg (TrainPipelineConfig): Configuration object containing hyperparameters.
        wandb_logger (WandBLogger | None): Logger for tracking training progress.
        shutdown_event (Event): Event to signal shutdown.
        transition_queue (Queue): Queue for receiving transitions from the actor.
        interaction_message_queue (Queue): Queue for receiving interaction messages from the actor.
        parameters_queue (Queue): Queue for sending policy parameters to the actor.
    """
    # Initialize logging for multiprocessing
    if not use_threads(cfg):
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"learner_train_process_{os.getpid()}.log")
        init_logging(log_file=log_file)
        logging.info("Initialized logging for actor information and training process")

    log_freq = cfg.log_freq
    save_freq = cfg.save_freq
    saving_checkpoint = cfg.save_checkpoint

    mp_net: MPNetConfig = cfg.env
    fps = mp_net.fps
    device = get_safe_torch_device(try_device=mp_net.device, log=True)
    storage_device = get_safe_torch_device(try_device=mp_net.storage_device)

    # Read parameters from policies
    policies = mp_net.make_policies(resume=cfg.resume, path=cfg.output_dir)
    scalers = {n: torch.cuda.amp.GradScaler(enabled=True) for n in policies}
    policy_parameters_push_frequency = cfg.env.get_policy_configs()[0].actor_learner_config.policy_parameters_push_frequency

    clip_grad_norm_value = {}
    training_starts = {}
    cta_ratio = {}
    policy_update_freq = {}
    bc_dagger_enable = {}
    noise_enable = {}
    noise_update_freq = {}
    noise_update_steps = {}
    online_steps = {}
    async_prefetch = {}
    optimization_step = {}
    interaction_step_shift = {}
    last_messages = {}
    for name, policy in policies.items():
        assert isinstance(policy, nn.Module)
        policy.train()

        clip_grad_norm_value[name] = policy.config.grad_clip_norm
        training_starts[name] = policy.config.training_starts
        cta_ratio[name] = policy.config.cta_ratio
        policy_update_freq[name] = policy.config.policy_update_freq
        bc_dagger_enable[name] = policy.config.use_bc_dagger
        noise_enable[name] = policy.config.noise_config.enable
        noise_update_freq[name] = policy.config.noise_config.update_freq
        noise_update_steps[name] = policy.config.noise_config.update_steps
        online_steps[name] = policy.config.online_steps
        async_prefetch[name] = policy.config.async_prefetch
        optimization_step[name] = 0
        interaction_step_shift[name] = 0
        last_messages[name] = None

    push_all_actor_policies_to_queue(parameters_queue=parameters_queue, policies=policies)

    last_time_policy_pushed = time.time()

    optimizers, lr_scheduler = make_optimizers_and_scheduler(policies=policies)

    # If we are resuming, we need to load the training state
    resume_optimization_step, resume_interaction_step = load_training_state(cfg=cfg, optimizers=optimizers)

    log_training_info(cfg=cfg, policies=policies)

    for name, policy in policies.items():
        cfg.dataset = DatasetConfig(repo_id=cfg.env.repo_id + f"-{name}")

    replay_buffers = initialize_replay_buffer(cfg, policies, device, storage_device)
    batch_size = cfg.batch_size

    offline_replay_buffers = {p: None for p in policies.keys()}
    has_offline_ds = os.path.exists(os.path.join(cfg.env.root, "offline-demos"))
    assert has_offline_ds or not (any(list(noise_enable.values())) or any(list(bc_dagger_enable.values()))), "Noise / BC batches need to sample from offline buffer"
    if has_offline_ds:
        offline_replay_buffers = initialize_offline_replay_buffer(cfg, policies, device, storage_device)
        batch_size: int = batch_size // 2  # We will sample from both replay buffers

    logging.info("Starting learner thread")
    optimization_step = resume_optimization_step if resume_optimization_step is not None else optimization_step
    interaction_step_shift = resume_interaction_step if resume_interaction_step is not None else interaction_step_shift
    for name, policy in policies.items():
        if isinstance(optimization_step, dict) and name in optimization_step:
            policy.set_opt_step(optimization_step[name])

    # Initialize iterators
    online_iterators = {n: None for n in policies}
    offline_iterators = {n: None for n in policies}
    noise_iterators = {n: None for n in policies}

    def prepare_batch(current_policy, batch):
        actions = batch["action"]
        rewards = batch["reward"]
        observations = batch["state"]
        next_observations = batch["next_state"]
        done = batch["done"]
        check_nan_in_transition(observations=observations, actions=actions, next_state=next_observations)

        observation_features, next_observation_features = get_observation_features(
            policy=current_policy, observations=observations, next_observations=next_observations
        )

        # Create a batch dictionary with all required elements for the forward method
        forward_batch = {
            "action": actions,
            "reward": rewards,
            "state": observations,
            "next_state": next_observations,
            "done": done,
            "observation_feature": observation_features,
            "next_observation_feature": next_observation_features,
            "complementary_info": batch["complementary_info"],
        }

        return forward_batch

    # NOTE: THIS IS THE MAIN LOOP OF THE LEARNER
    try:
        while True:
            # Exit the training loop if shutdown is requested
            if shutdown_event is not None and shutdown_event.is_set():
                logging.info("[LEARNER] Shutdown signal received. Exiting...")
                break

            time_for_one_optimization_step = time.time()

            # Process all available transitions to the replay buffer, send by the actor server
            process_transitions(
                transition_queue=transition_queue,
                replay_buffers=replay_buffers,
                offline_replay_buffers=offline_replay_buffers,
                device=device,
                shutdown_event=shutdown_event,
            )

            # Process all available interaction messages sent by the actor server
            process_interaction_messages(
                interaction_message_queue=interaction_message_queue,
                interaction_step_shift=interaction_step_shift,
                wandb_logger=wandb_logger,
                shutdown_event=shutdown_event,
                last_messages=last_messages
            )

            for name, policy in policies.items():

                # Wait until the replay buffer has enough samples to start training
                if len(replay_buffers[name]) < training_starts[name]:
                    continue

                if optimization_step[name] >= online_steps[name]:
                    continue

                if online_iterators[name] is None:
                    online_iterators[name] = replay_buffers[name].get_iterator(
                        batch_size=batch_size, async_prefetch=async_prefetch[name], queue_size=4
                    )

                if has_offline_ds and offline_iterators[name] is None:
                    offline_iterators[name] = offline_replay_buffers[name].get_iterator(
                        batch_size=batch_size, async_prefetch=async_prefetch[name], queue_size=4
                    )

                if noise_enable[name] and optimization_step[name] % noise_update_freq[name] == 0:
                    noise_iterators[name] = offline_replay_buffers[name].get_iterator(
                        batch_size=2 * batch_size, async_prefetch=async_prefetch[name], queue_size=4
                    )

                if bc_dagger_enable[name]:
                    batch = next(offline_iterators[name])

                    forward_batch = prepare_batch(policy, batch)

                    bc_output = policy.forward(forward_batch, model="bc")

                    # Main critic optimization
                    loss_bc = bc_output["loss_bc"]
                    optimizers[name]["critic"].zero_grad()
                    loss_bc.backward()
                    bc_grad_norm = torch.nn.utils.clip_grad_norm_(
                        parameters=policy.actor.parameters(), max_norm=clip_grad_norm_value[name]
                    )
                    optimizers[name]["actor"].step()

                    training_infos = bc_output["training_infos"]
                    training_infos["bc_grad_norm"] = bc_grad_norm

                else:
                    for _ in range(cta_ratio[name] - 1):
                        # Sample from the iterators
                        batch = next(online_iterators[name])

                        if has_offline_ds:
                            batch_offline = next(offline_iterators[name])
                            batch = concatenate_batch_transitions(
                                left_batch_transitions=batch, right_batch_transition=batch_offline
                            )

                        forward_batch = prepare_batch(policy, batch)

                        # Use the forward method for critic loss
                        critic_output = policy.forward(forward_batch, model="critic")

                        # Main critic optimization
                        loss_critic = critic_output["loss_critic"]
                        optimizers[name]["critic"].zero_grad()
                        loss_critic.backward()
                        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                            parameters=policy.critic_ensemble.parameters(), max_norm=clip_grad_norm_value[name]
                        )
                        optimizers[name]["critic"].step()

                        # Discrete critic optimization (if available)
                        if policy.config.num_discrete_actions is not None:
                            discrete_critic_output = policy.forward(forward_batch, model="discrete_critic")
                            loss_discrete_critic = discrete_critic_output["loss_discrete_critic"]
                            optimizers[name]["discrete_critic"].zero_grad()
                            loss_discrete_critic.backward()
                            discrete_critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                                parameters=policy.discrete_critic.parameters(), max_norm=clip_grad_norm_value[name]
                            )
                            optimizers[name]["discrete_critic"].step()

                        # Update target networks (main and discrete)
                        #policy.update_target_networks()

                    # Sample for the last update in the UTD ratio
                    batch = next(online_iterators[name])

                    if has_offline_ds:
                        batch_offline = next(offline_iterators[name])
                        batch = concatenate_batch_transitions(
                            left_batch_transitions=batch, right_batch_transition=batch_offline
                        )

                    forward_batch = prepare_batch(policy, batch)

                    critic_output = policy.forward(forward_batch, model="critic")

                    loss_critic = critic_output["loss_critic"]
                    optimizers[name]["critic"].zero_grad()
                    loss_critic.backward()
                    critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                        parameters=policy.critic_ensemble.parameters(), max_norm=clip_grad_norm_value[name]
                    )
                    optimizers[name]["critic"].step()

                    # Initialize training info dictionary
                    training_infos = critic_output["training_infos"]
                    training_infos["loss_critic"] = loss_critic
                    training_infos["critic_grad_norm"] = critic_grad_norm

                    # Discrete critic optimization (if available)
                    if policy.config.num_discrete_actions is not None:
                        discrete_critic_output = policy.forward(forward_batch, model="discrete_critic")
                        loss_discrete_critic = discrete_critic_output["loss_discrete_critic"]
                        optimizers[name]["discrete_critic"].zero_grad()
                        loss_discrete_critic.backward()
                        discrete_critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                            parameters=policy.discrete_critic.parameters(), max_norm=clip_grad_norm_value[name]
                        )
                        optimizers[name]["discrete_critic"].step()

                        # Add discrete critic info to training info
                        training_infos["loss_discrete_critic"] = loss_discrete_critic
                        training_infos["discrete_critic_grad_norm"] = discrete_critic_grad_norm

                    # Actor and temperature optimization (at specified frequency)
                    if optimization_step[name] % policy_update_freq[name] == 0:
                        for _ in range(policy_update_freq[name]):
                            # Actor optimization
                            actor_output = policy.forward(forward_batch, model="actor")
                            loss_actor = actor_output["loss_actor"]
                            optimizers[name]["actor"].zero_grad()
                            loss_actor.backward()
                            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                                parameters=policy.actor.parameters(), max_norm=clip_grad_norm_value[name]
                            )
                            optimizers[name]["actor"].step()

                            # Add actor info to training info
                            training_infos["loss_actor"] = loss_actor
                            training_infos["actor_grad_norm"] = actor_grad_norm

                            # Temperature optimization
                            temperature_output = policy.forward(forward_batch, model="temperature")
                            loss_temperature = temperature_output["loss_temperature"]
                            optimizers[name]["temperature"].zero_grad()
                            loss_temperature.backward()
                            temp_grad_norm = torch.nn.utils.clip_grad_norm_(
                                parameters=[policy.log_alpha], max_norm=clip_grad_norm_value[name]
                            )
                            optimizers[name]["temperature"].step()

                            # Add temperature info to training info
                            training_infos["loss_temperature"] = loss_temperature
                            training_infos["temperature_grad_norm"] = temp_grad_norm
                            training_infos["temperature"] = policy.temperature

                            # Update temperature
                            policy.update_temperature()

                    if noise_enable[name] and optimization_step[name] % noise_update_freq[name] == 0:
                        for noise_update_step in range(noise_update_steps[name]):
                            # Structured noise optimization
                            batch = next(noise_iterators[name])
                            forward_batch = prepare_batch(policy, batch)

                            noise_output = policy.forward(forward_batch, model="noise")
                            loss_noise = noise_output["loss_noise"]
                            optimizers[name]["noise"].zero_grad()
                            loss_noise.backward()
                            noise_grad_norm = torch.nn.utils.clip_grad_norm_(
                                parameters=policy.noise_net.parameters(), max_norm=clip_grad_norm_value[name]
                            )
                            optimizers[name]["noise"].step()

                            if noise_update_step % (log_freq / 5) == 0:
                                noise_infos = {
                                    "loss_noise": loss_noise.item(),
                                    "dgn_improvement_ratio": noise_output["dgn_improvement_ratio"],
                                    "noise_grad_norm": noise_grad_norm.item(),
                                    "Noise optimization step": optimization_step[name]
                                }

                                # Log training metrics
                                if wandb_logger:
                                    noise_infos = {f"{name}/{key}": value for key, value in noise_infos.items()}
                                    wandb_logger.log_dict(d=noise_infos, mode="train", custom_step_key=f"{name}/Noise optimization step")

                    # Update target networks (main and discrete)
                    policy.update_target_networks()

                # Save checkpoint at specified intervals
                if saving_checkpoint and (optimization_step[name] % save_freq == 0 or optimization_step[name] == online_steps[name]):
                    save_training_checkpoint(
                        primitive_id=name,
                        cfg=cfg,
                        optimization_step=optimization_step[name],
                        online_steps=online_steps[name],
                        interaction_message=last_messages[name],
                        policy=policy,
                        optimizers=optimizers[name],
                        replay_buffer=replay_buffers[name],
                        offline_replay_buffer=offline_replay_buffers[name],
                        fps=fps,
                    )

                optimization_step[name] += 1

                # Keep the policy-internal step in sync with learner time.
                # This makes epsilon decay independent of the actor's interaction step.
                policies[name].set_opt_step(optimization_step[name])

                # Log training metrics at specified intervals or when having trained the noise net
                if optimization_step[name] % log_freq == 0:
                    logging.info(f"[LEARNER] Number of optimization steps for {name}: {optimization_step[name]}")

                    # Log training metrics
                    if wandb_logger:

                        # Add step infos
                        training_infos["Optimization step"] = optimization_step[name]
                        training_infos["replay_buffer_size"] = len(replay_buffers[name])
                        if has_offline_ds:
                            training_infos["offline_replay_buffer_size"] = len(offline_replay_buffers[name])

                        # Clean and log
                        log_infos = {}
                        for key, value in training_infos.items():
                            key = f"{name}/{key}"
                            if isinstance(value, torch.Tensor):
                                value = value.item()
                            log_infos[key] = value
                        wandb_logger.log_dict(d=log_infos, mode="train", custom_step_key=f"{name}/Optimization step")


            # Skip frequency in the beginning
            if all([step == 0 for step in optimization_step.values()]):
                continue

            # Push policy to actors if needed
            if time.time() - last_time_policy_pushed > policy_parameters_push_frequency:
                push_all_actor_policies_to_queue(parameters_queue=parameters_queue, policies=policies)
                last_time_policy_pushed = time.time()

            # Calculate and log optimization frequency
            time_for_one_optimization_step = time.time() - time_for_one_optimization_step
            frequency_for_one_optimization_step = 1 / (time_for_one_optimization_step + 1e-9)

            logging.info(f"[LEARNER] Optimization frequency loop [Hz]: {frequency_for_one_optimization_step}")

            # Log optimization frequency
            if wandb_logger:
                wandb_logger.log_dict(
                    {
                        f"Optimization frequency loop [Hz]": frequency_for_one_optimization_step,
                        f"Global optimization step": sum(optimization_step.values()),
                    },
                    mode="train",
                    custom_step_key="Global optimization step",
                )

    except Exception as e:
        import traceback
        logging.error(f"[LEARNER CRASH] Step {optimization_step[name]} failed: {e}")
        traceback.print_exc()


def start_learner_server(
    parameters_queue: Queue,
    transition_queue: Queue,
    interaction_message_queue: Queue,
    shutdown_event: any,  # Event,
    cfg: TrainPipelineConfig,
):
    """
    Start the learner server for training.
    It will receive transitions and interaction messages from the actor server,
    and send policy parameters to the actor server.

    Args:
        parameters_queue: Queue for sending policy parameters to the actor
        transition_queue: Queue for receiving transitions from the actor
        interaction_message_queue: Queue for receiving interaction messages from the actor
        shutdown_event: Event to signal shutdown
        cfg: Training configuration
    """
    if not use_threads(cfg):
        # Create a process-specific log file
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"learner_server_process_{os.getpid()}.log")

        # Initialize logging with explicit log file
        init_logging(log_file=log_file)
        logging.info("Learner server process logging initialized")

        # Setup process handlers to handle shutdown signal
        # But use shutdown event from the main process
        # Return back for MP
        setup_process_handlers(False)

    service = learner_service.LearnerService(
        shutdown_event=shutdown_event,
        parameters_queue=parameters_queue,
        seconds_between_pushes=cfg.env.get_policy_configs()[0].actor_learner_config.policy_parameters_push_frequency,
        transition_queue=transition_queue,
        interaction_message_queue=interaction_message_queue,
    )

    server = grpc.server(
        ThreadPoolExecutor(max_workers=learner_service.MAX_WORKERS),
        options=[
            ("grpc.max_receive_message_length", learner_service.MAX_MESSAGE_SIZE),
            ("grpc.max_send_message_length", learner_service.MAX_MESSAGE_SIZE),
        ],
    )

    hilserl_pb2_grpc.add_LearnerServiceServicer_to_server(
        service,
        server,
    )

    host = cfg.env.get_policy_configs()[0].actor_learner_config.learner_host
    port = cfg.env.get_policy_configs()[0].actor_learner_config.learner_port

    server.add_insecure_port(f"{host}:{port}")
    server.start()
    logging.info("[LEARNER] gRPC server started")

    shutdown_event.wait()
    logging.info("[LEARNER] Stopping gRPC server...")
    server.stop(learner_service.SHUTDOWN_TIMEOUT)
    logging.info("[LEARNER] gRPC server stopped")


def save_training_checkpoint(
    primitive_id: str,
    cfg: TrainPipelineConfig,
    optimization_step: int,
    online_steps: int,
    interaction_message: dict | None,
    policy: nn.Module,
    optimizers: dict[str, Optimizer],
    replay_buffer: ReplayBuffer,
    offline_replay_buffer: ReplayBuffer | None = None,
    fps: int = 30,
) -> None:
    """
    Save training checkpoint and associated data.

    This function performs the following steps:
    1. Creates a checkpoint directory with the current optimization step
    2. Saves the policy model, configuration, and optimizer states
    3. Saves the current interaction step for resuming training
    4. Updates the "last" checkpoint symlink to point to this checkpoint
    5. Saves the replay buffer as a dataset for later use
    6. If an offline replay buffer exists, saves it as a separate dataset

    Args:
        cfg: Training configuration
        optimization_step: Current optimization step
        online_steps: Total number of online steps
        interaction_message: Dictionary containing interaction information
        policy: Policy model to save
        optimizers: Dictionary of optimizers
        replay_buffer: Replay buffer to save as dataset
        offline_replay_buffer: Optional offline replay buffer to save
        dataset_repo_id: Repository ID for dataset
        fps: Frames per second for dataset
    """
    logging.info(f"Checkpoint policy after step {optimization_step}")
    _num_digits = max(6, len(str(online_steps)))
    interaction_step = interaction_message[f"{primitive_id}/Interaction step"] if interaction_message is not None else 0

    # Create checkpoint directory
    checkpoint_dir = get_step_checkpoint_dir(Path(cfg.output_dir) / primitive_id, online_steps, optimization_step)

    # Save checkpoint
    save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        step=optimization_step,
        cfg=cfg,
        policy=policy,
        optimizer=optimizers,
        scheduler=None,
    )

    # Save interaction step manually
    training_state_dir = os.path.join(checkpoint_dir, TRAINING_STATE_DIR)
    os.makedirs(training_state_dir, exist_ok=True)
    training_state = {"step": optimization_step, "interaction_step": interaction_step}
    torch.save(training_state, os.path.join(training_state_dir, "training_state.pt"))

    # Update the "last" symlink
    update_last_checkpoint(checkpoint_dir)

    # TODO : temporary save replay buffer here, remove later when on the robot
    # We want to control this with the keyboard inputs
    dataset_dir = os.path.join(cfg.output_dir, primitive_id, "dataset")
    if os.path.exists(dataset_dir) and os.path.isdir(dataset_dir):
        shutil.rmtree(dataset_dir)

    # Save dataset
    # NOTE: Handle the case where the dataset repo id is not specified in the config
    # eg. RL training without demonstrations data
    repo_id_buffer_save = cfg.dataset.repo_id + f"-{primitive_id}"
    replay_buffer.to_lerobot_dataset(repo_id=repo_id_buffer_save, fps=fps, root=dataset_dir)

    if offline_replay_buffer is not None:
        dataset_offline_dir = os.path.join(cfg.output_dir, primitive_id, "dataset-offline")
        if os.path.exists(dataset_offline_dir) and os.path.isdir(dataset_offline_dir):
            shutil.rmtree(dataset_offline_dir)

        offline_replay_buffer.to_lerobot_dataset(
            cfg.dataset.repo_id + f"-{primitive_id}",
            fps=fps,
            root=dataset_offline_dir,
        )

    logging.info("Resume training")


def make_optimizers_and_scheduler(policies: dict[str, SACPolicy]):
    """
    Creates and returns optimizers for the actor, critic, and temperature components of a reinforcement learning policy.

    This function sets up Adam optimizers for:
    - The **actor network**, ensuring that only relevant parameters are optimized.
    - The **critic ensemble**, which evaluates the value function.
    - The **temperature parameter**, which controls the entropy in soft actor-critic (SAC)-like methods.

    It also initializes a learning rate scheduler, though currently, it is set to `None`.

    NOTE:
    - If the encoder is shared, its parameters are excluded from the actor's optimization process.
    - The policy's log temperature (`log_alpha`) is wrapped in a list to ensure proper optimization as a standalone tensor.

    Args:
        cfg: Configuration object containing hyperparameters.
        policies: dict[str, nn.Module] (nn.Module): The policy model containing the actor, critic, and temperature components.

    Returns:
        Tuple[Dict[str, torch.optim.Optimizer], Optional[torch.optim.lr_scheduler._LRScheduler]]:
        A tuple containing:
        - `optimizers`: A dictionary mapping component names ("actor", "critic", "temperature") to their respective Adam optimizers.
        - `lr_scheduler`: Currently set to `None` but can be extended to support learning rate scheduling.

    """
    optimizers = {}
    lr_scheduler = {}
    for name, policy in policies.items():

        optimizer_actor = torch.optim.Adam(
            params=[
                p
                for n, p in policy.actor.named_parameters()
                if not policy.config.shared_encoder or not n.startswith("encoder")
            ],
            lr=policy.config.actor_lr,
        )
        optimizer_critic = torch.optim.Adam(params=policy.critic_ensemble.parameters(), lr=policy.config.critic_lr)

        if policy.config.num_discrete_actions is not None:
            optimizer_discrete_critic = torch.optim.Adam(
                params=policy.discrete_critic.parameters(), lr=policy.config.critic_lr
            )
        optimizer_temperature = torch.optim.Adam(params=[policy.log_alpha], lr=policy.config.temperature_lr)

        if policy.config.noise_config.enable:
            optimizer_noise = torch.optim.Adam(
                params=policy.noise_net.parameters(), lr=policy.config.noise_lr
            )

        lr_scheduler[name] = None
        optimizers[name] = {
            "actor": optimizer_actor,
            "critic": optimizer_critic,
            "temperature": optimizer_temperature,
        }
        if policy.config.num_discrete_actions is not None:
            optimizers[name]["discrete_critic"] = optimizer_discrete_critic
        if policy.config.noise_config.enable:
            optimizers[name]["noise"] = optimizer_noise

    return optimizers, lr_scheduler


#################################################
# Training setup functions #
#################################################


def handle_resume_logic(cfg: TrainPipelineConfig) -> TrainPipelineConfig:
    """
    Handle the resume logic for training.

    If resume is True:
    - Verifies that a checkpoint exists
    - Loads the checkpoint configuration
    - Logs resumption details
    - Returns the checkpoint configuration

    If resume is False:
    - Checks if an output directory exists (to prevent accidental overwriting)
    - Returns the original configuration

    Args:
        cfg (TrainPipelineConfig): The training configuration

    Returns:
        TrainPipelineConfig: The updated configuration

    Raises:
        RuntimeError: If resume is True but no checkpoint found, or if resume is False but directory exists
    """
    out_dir = cfg.output_dir

    # Case 1: Not resuming, but need to check if directory exists to prevent overwrites
    if not cfg.resume:
        checkpoint_dir = os.path.join(out_dir, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK)
        if os.path.exists(checkpoint_dir):
            raise RuntimeError(
                f"Output directory {checkpoint_dir} already exists. Use `resume=true` to resume training."
            )
        return cfg

    # Case 2: Resuming training
    checkpoint_dir = os.path.join(out_dir, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK)
    if not os.path.exists(checkpoint_dir):
        raise RuntimeError(f"No model checkpoint found in {checkpoint_dir} for resume=True")

    # Log that we found a valid checkpoint and are resuming
    logging.info(
        colored(
            "Valid checkpoint found: resume=True detected, resuming previous run",
            color="yellow",
            attrs=["bold"],
        )
    )

    # Load config using Draccus
    checkpoint_cfg_path = os.path.join(checkpoint_dir, PRETRAINED_MODEL_DIR, "train_config.json")
    checkpoint_cfg = TrainPipelineConfig.from_pretrained(checkpoint_cfg_path)

    # Ensure resume flag is set in returned config
    checkpoint_cfg.resume = True
    return checkpoint_cfg


def load_training_state(
    cfg: TrainPipelineConfig,
    optimizers: Optimizer | dict[str, Optimizer],
):
    """
    Loads the training state (optimizers, step count, etc.) from a checkpoint.

    Args:
        cfg (TrainPipelineConfig): Training configuration
        optimizers (Optimizer | dict): Optimizers to load state into

    Returns:
        tuple: (optimization_step, interaction_step) or (None, None) if not resuming
    """
    if not cfg.resume:
        return None, None

    steps: dict[str, int] = {}
    interaction_steps: dict[str, int] = {}

    try:
        for primitive_id in optimizers:

            # Construct path to the last checkpoint directory
            checkpoint_dir = os.path.join(cfg.output_dir, primitive_id, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK)

            logging.info(f"Loading {primitive_id} training state from {checkpoint_dir}")


            # Use the utility function from train_utils which loads the optimizer state
            step, optimizers, _ = utils_load_training_state(Path(checkpoint_dir), optimizers, None)

            # Load interaction step separately from training_state.pt
            training_state_path = os.path.join(checkpoint_dir, TRAINING_STATE_DIR, "training_state.pt")
            interaction_step = 0
            if os.path.exists(training_state_path):
                training_state = torch.load(training_state_path, weights_only=False)  # nosec B614: Safe usage of torch.load
                interaction_step = training_state.get("interaction_step", 0)

            logging.info(f"Resuming {primitive_id} from step {step}, interaction step {interaction_step}")
            steps[primitive_id] = step
            interaction_steps[primitive_id] = interaction_step

    except Exception as e:
        logging.error(f"Failed to load training state: {e}")
        return None, None

    return steps, interaction_steps



def log_training_info(cfg: TrainPipelineConfig, policies: dict[str, nn.Module]) -> None:
    """
    Log information about the training process.

    Args:
        cfg (TrainPipelineConfig): Training configuration
        policy (dict): Policy model
    """
    logging.info(colored("[LEARNER] Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    logging.info(f"[LEARNER] {cfg.env.task=}")
    for name, policy in policies.items():
        num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        num_total_params = sum(p.numel() for p in policy.parameters())

        logging.info(f"--- {name} policy ---")
        logging.info(f"{policy.config.online_steps=}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")


def initialize_replay_buffer(cfg: TrainPipelineConfig, policies, device: str, storage_device: str) -> dict[str, ReplayBuffer]:
    """
    Initialize a replay buffer, either empty or from a dataset if resuming.

    Args:
        cfg (TrainPipelineConfig): Training configuration
        device (str): Device to store tensors on
        storage_device (str): Device for storage optimization

    Returns:
        ReplayBuffer: Initialized replay buffer
    """
    if not cfg.resume:
        return {
            name: ReplayBuffer(
                capacity=policy.config.online_buffer_capacity,
                device=device,
                state_keys=policy.config.input_features.keys(),
                storage_device=storage_device,
                optimize_memory=True,
            ) for name, policy in policies.items()
        }

    logging.info("Resume training load the online dataset")

    replay_buffers = {}
    for name, policy in policies.items():
        cfg.dataset = DatasetConfig(repo_id=cfg.env.repo_id + f"-{name}")

        dataset_path = os.path.join(cfg.output_dir, name, "dataset")

        # NOTE: In RL is possible to not have a dataset.
        repo_id = None
        if cfg.dataset is not None:
            repo_id = cfg.dataset.repo_id + f"-{name}"
        dataset = LeRobotDataset(
            repo_id=repo_id,
            root=dataset_path,
        )
        replay_buffers[name] = ReplayBuffer.from_lerobot_dataset(
            lerobot_dataset=dataset,
            capacity=policy.config.online_buffer_capacity,
            device=device,
            state_keys=policy.config.input_features.keys(),
            storage_device=storage_device,
            optimize_memory=True,
        )

    return replay_buffers


def initialize_offline_replay_buffer(
    cfg: TrainPipelineConfig,
    policies: dict[str, SACPolicy],
    device: str,
    storage_device: str,
) -> dict[str, ReplayBuffer]:
    """
    Initialize an offline replay buffer from a dataset.

    Args:
        cfg (TrainPipelineConfig): Training configuration
        policies (dict):
        device (str): Device to store tensors on
        storage_device (str): Device for storage optimization

    Returns:
        ReplayBuffer: Initialized offline replay buffer
    """
    offline_replay_buffers = {}
    for name, policy in policies.items():

        # Temporarily overwrite the top-level fields of the config to use utilities
        cfg.policy = policy.config
        cfg.dataset = DatasetConfig(
            root=str(Path(cfg.env.root) / "offline-demos" / name),
            repo_id=cfg.env.repo_id + f"-{name}"
        )

        if not cfg.resume:
            logging.info("make_dataset offline buffer")
            offline_dataset = make_dataset(cfg)
        else:
            logging.info("load offline dataset")
            dataset_offline_dir = os.path.join(cfg.output_dir, name, "dataset-offline")
            offline_dataset = LeRobotDataset(
                repo_id=cfg.dataset.repo_id,
                root=dataset_offline_dir
            )

        logging.info("Convert to a offline replay buffer")
        offline_replay_buffers[name] = ReplayBuffer.from_lerobot_dataset(
            offline_dataset,
            device=device,
            state_keys=policy.config.input_features.keys(),
            storage_device=storage_device,
            optimize_memory=True,
            capacity=policy.config.offline_buffer_capacity,
        )
    return offline_replay_buffers


#################################################
# Utilities/Helpers functions #
#################################################


def get_observation_features(
    policy: SACPolicy, observations: torch.Tensor, next_observations: torch.Tensor
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    Get observation features from the policy encoder. It act as cache for the observation features.
    when the encoder is frozen, the observation features are not updated.
    We can save compute by caching the observation features.

    Args:
        policy: The policy model
        observations: The current observations
        next_observations: The next observations

    Returns:
        tuple: observation_features, next_observation_features
    """

    if policy.config.vision_encoder_name is None or not policy.config.freeze_vision_encoder or not policy.actor.encoder.has_images:
        return None, None

    with torch.no_grad():
        observation_features = policy.actor.encoder.get_cached_image_features(observations, normalize=True)
        next_observation_features = policy.actor.encoder.get_cached_image_features(next_observations, normalize=True)

    return observation_features, next_observation_features


def use_threads(cfg: TrainPipelineConfig) -> bool:
    return cfg.env.get_policy_configs()[0].concurrency.actor == "threads"


def check_nan_in_transition(
    observations: torch.Tensor,
    actions: torch.Tensor,
    next_state: torch.Tensor,
    raise_error: bool = False,
) -> bool:
    """
    Check for NaN values in transition data.

    Args:
        observations: Dictionary of observation tensors
        actions: Action tensor
        next_state: Dictionary of next state tensors
        raise_error: If True, raises ValueError when NaN is detected

    Returns:
        bool: True if NaN values were detected, False otherwise
    """
    nan_detected = False

    # Check observations
    for key, tensor in observations.items():
        if torch.isnan(tensor).any():
            logging.error(f"observations[{key}] contains NaN values")
            nan_detected = True
            if raise_error:
                raise ValueError(f"NaN detected in observations[{key}]")

    # Check next state
    for key, tensor in next_state.items():
        if torch.isnan(tensor).any():
            logging.error(f"next_state[{key}] contains NaN values")
            nan_detected = True
            if raise_error:
                raise ValueError(f"NaN detected in next_state[{key}]")

    # Check actions
    if torch.isnan(actions).any():
        logging.error("actions contains NaN values")
        nan_detected = True
        if raise_error:
            raise ValueError("NaN detected in actions")

    return nan_detected


def push_all_actor_policies_to_queue(parameters_queue: Queue, policies: dict[str, nn.Module]):
    logging.debug("[LEARNER] Pushing all actor policies to the queue")

    # ensure we never block: keep only the latest payload per push cycle
    def _drain_one(q: Queue):
        try:
            _ = q.get_nowait()
        except Exception:
            pass

    for primitive_id, policy in policies.items():
        try:
            # make room if the queue is full
            if parameters_queue.full():
                _drain_one(parameters_queue)

            state_dict = move_state_dict_to_device(policy.state_dict(), device="cpu")
            state_dict["primitive_id"] = primitive_id
            state_bytes = state_to_bytes(state_dict)

            # non-blocking put; if still full, skip this push
            parameters_queue.put(state_bytes, block=False)
        except queue.Full:
            logging.warning("[LEARNER] parameters_queue full; skipping push for %s", primitive_id)
        except Exception as e:
            logging.error("[LEARNER] Failed to enqueue params for %s: %s", primitive_id, e)


def process_interaction_message(
    message, interaction_step_shift: dict[str, int], wandb_logger: WandBLogger | None = None
):
    """Process a single interaction message with consistent handling."""
    message = bytes_to_python_object(message)

    primitive_id = message['Primitive']

    # Shift interaction step for consistency with checkpointed state
    message["Interaction step"] += interaction_step_shift[primitive_id]

    # prepend primitive id
    del message['Primitive']
    message = {f"{primitive_id}/{key}": value for key, value in message.items()}

    # Log if logger available
    if wandb_logger:
        wandb_logger.log_dict(d=message, mode="train", custom_step_key=f"{primitive_id}/Interaction step")

    return primitive_id, message


def process_transitions(
    transition_queue: Queue,
    replay_buffers: dict[str, ReplayBuffer],
    offline_replay_buffers: dict[str, ReplayBuffer],
    device: str,
    shutdown_event: any,
):
    """Process all available transitions from the queue.

    Args:
        transition_queue: Queue for receiving transitions from the actor
        replay_buffers: Replay buffer to add transitions to
        offline_replay_buffers: Offline replay buffer to add transitions to
        device: Device to move transitions to
        dataset_repo_id: Repository ID for dataset
        shutdown_event: Event to signal shutdown
    """
    while not transition_queue.empty() and not shutdown_event.is_set():
        try:
            transition_list = transition_queue.get(timeout=5.0)
        except queue.Empty:
            print("[LEARNER] No transition received for 5s.")
            return
        except Exception as e:
            print(f"[LEARNER] Failed to receive transition: {e}")
            return

        transition_list = bytes_to_transitions(buffer=transition_list)
        for transition in transition_list:
            transition = move_transition_to_device(transition=transition, device=device)

            # Skip transitions with NaN values
            if check_nan_in_transition(
                observations=transition["state"],
                actions=transition["action"],
                next_state=transition["next_state"],
            ):
                logging.warning("[LEARNER] NaN detected in transition, skipping")
                continue

            # build a one‐off dict without "id"
            payload = {k: v for k, v in transition.items() if k != "id"}
            primitive_id = transition["id"]
            replay_buffers[primitive_id].add(**payload)

            # Add to offline buffer if it's an intervention
            if (offline_replay_buffers[primitive_id] is not None and
                transition.get("complementary_info", {}).get("is_intervention")):
                offline_replay_buffers[primitive_id].add(**payload)


def process_interaction_messages(
    interaction_message_queue: Queue,
    interaction_step_shift: dict[str, int],
    wandb_logger: WandBLogger | None,
    shutdown_event: any,
    last_messages: Optional[dict] = None
) -> dict | None:
    """Process all available interaction messages from the queue.

    Args:
        interaction_message_queue: Queue for receiving interaction messages
        interaction_step_shift: Amount to shift interaction step by
        wandb_logger: Logger for tracking progress
        shutdown_event: Event to signal shutdown

    Returns:
        dict | None: The last interaction message processed, or None if none were processed
    """
    if last_messages is None:
        last_messages = {}

    while not interaction_message_queue.empty() and not shutdown_event.is_set():
        try:
            message = interaction_message_queue.get(timeout=5.0)
        except queue.Empty:
            print("[LEARNER] No interaction messages received for 5s.")
            return last_messages
        except Exception as e:
            print(f"[LEARNER] Failed to receive interaction message: {e}")
            return last_messages

        primitive_id, last_message = process_interaction_message(
            message=message,
            interaction_step_shift=interaction_step_shift,
            wandb_logger=wandb_logger,
        )
        last_messages[primitive_id] = last_message

    return last_messages


if __name__ == "__main__":
    train_cli()
    logging.info("[LEARNER] main finished")
