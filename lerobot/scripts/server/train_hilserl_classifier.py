# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import numpy as np
import torch
import wandb
from termcolor import colored
from torch.amp import GradScaler
from torch.autograd import profiler
from torch.optim import Optimizer
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler, create_balanced_sampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.policies.reward_model.modeling_classifier import RewardClassifierConfig
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint, get_best_checkpoint_dir,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy
from lerobot.scripts.server.buffer import random_shift


def train_epoch(model, train_loader, optimizer, grad_scaler, device, logger, step, cfg):
    # Single epoch training loop with AMP support and progress tracking
    model.train()
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        start_time = time.perf_counter()
        new_batch = {img_key: batch[img_key].to(device) for img_key in model.image_keys}
        new_batch = {img_key: random_shift(new_batch[img_key], 4) for img_key in new_batch}
        new_batch["next.reward"] = batch["next.reward"].float().to(device)

        # Forward pass with optional AMP
        with torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
            loss, output_dict = model.forward(new_batch)

        # Backward pass with gradient scaling if AMP enabled
        optimizer.zero_grad()
        if cfg.policy.use_amp:
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()

        train_info = {
            "loss": loss.item(),
            "accuracy": output_dict["accuracy"],
            "dataloading_s": time.perf_counter() - start_time,
        }

        if logger is not None:
            logger.log_dict(train_info, step + batch_idx, mode="train")
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{output_dict['accuracy']:.2f}%"})


def validate(model, val_loader, device, logger, cfg):
    # Validation loop with metric tracking and sample logging
    model.eval()
    correct = 0
    total = 0
    batch_start_time = time.perf_counter()
    samples = []
    running_loss = 0
    inference_times = []

    with (
        torch.no_grad(),
        torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
    ):
        for batch in tqdm(val_loader, desc="Validation"):
            new_batch = {img_key: batch[img_key].to(device) for img_key in model.image_keys}
            labels = batch["next.reward"].float().to(device)
            new_batch["next.reward"] = labels

            loss, output_dict = model.forward(new_batch)

            correct += output_dict["accuracy"] / 100.0 * labels.size(0)
            total += labels.size(0)
            running_loss += loss.item()

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(val_loader)
    print(f"Average validation loss {avg_loss}, and accuracy {accuracy}")

    eval_info = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "eval_s": time.perf_counter() - batch_start_time,
        "eval/prediction_samples": wandb.Table(
            data=[list(s.values()) for s in samples],
            columns=list(samples[0].keys()),
        )
        if logger is not None
        else None,
    }

    if len(inference_times) > 0:
        eval_info["inference_time_avg"] = np.mean(inference_times)
        eval_info["inference_time_median"] = np.median(inference_times)
        eval_info["inference_time_std"] = np.std(inference_times)
        eval_info["inference_time_batch_size"] = val_loader.batch_size

        print(
            f"Inference mean time: {eval_info['inference_time_avg']:.2f} us, median: {eval_info['inference_time_median']:.2f} us, std: {eval_info['inference_time_std']:.2f} us, with {len(inference_times)} iterations on {device.type} device, batch size: {eval_info['inference_time_batch_size']}"
        )

    return accuracy, eval_info


def benchmark_inference_time(model, dataset, logger, cfg, device, step):
    if not cfg.training.profile_inference_time:
        return

    iters = cfg.training.profile_inference_time_iters
    inference_times = []

    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=cfg.training.num_workers,
        sampler=RandomSampler(dataset),
        pin_memory=True,
    )

    model.eval()
    with torch.no_grad():
        for _ in tqdm(range(iters), desc="Benchmarking inference time"):
            x = next(iter(loader))
            x = [x[img_key].to(device) for img_key in model.image_keys]

            # Warm up
            for _ in range(10):
                _ = model(x)

            # sync the device
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()

            with (
                profiler.profile(record_shapes=True) as prof,
                profiler.record_function("model_inference"),
            ):
                _ = model(x)

            inference_times.append(
                next(x for x in prof.key_averages() if x.key == "model_inference").cpu_time
            )

    inference_times = np.array(inference_times)
    avg, median, std = (
        inference_times.mean(),
        np.median(inference_times),
        inference_times.std(),
    )
    print(
        f"Inference time mean: {avg:.2f} us, median: {median:.2f} us, std: {std:.2f} us, with {iters} iterations on {device.type} device"
    )
    if logger._cfg.wandb.enable:
        logger.log_dict(
            {
                "inference_time_benchmark_avg": avg,
                "inference_time_benchmark_median": median,
                "inference_time_benchmark_std": std,
            },
            step + 1,
            mode="eval",
        )

    return avg, median, std


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)
    logging.info(f"Dataset size: {len(dataset)}")

    n_total = len(dataset)
    n_train = int(0.9 * len(dataset))
    train_dataset = torch.utils.data.Subset(dataset, range(0, n_train))
    val_dataset = torch.utils.data.Subset(dataset, range(n_train, n_total))

    sampler = create_balanced_sampler(train_dataset, cfg)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        sampler=sampler,
        pin_memory=device.type == "cuda",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.eval.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    # Resume training if requested
    step = 0
    best_val_acc = 0

    logging.info("Creating policy")
    model = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, model)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    # Log model parameters
    num_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Learnable parameters: {format_big_number(num_learnable_params)}")
    logging.info(f"Total parameters: {format_big_number(num_total_params)}")

    # Training loop with validation and checkpointing
    for epoch in range(cfg.steps):
        logging.info(f"\nEpoch {epoch + 1}/{cfg.steps}")

        train_epoch(
            model,
            train_loader,
            optimizer,
            grad_scaler,
            device,
            wandb_logger,
            step,
            cfg,
        )

        # Periodic validation
        if cfg.eval_freq > 0 and (epoch + 1) % cfg.eval_freq == 0:
            val_acc, eval_info = validate(
                model,
                val_loader,
                device,
                wandb_logger,
                cfg,
            )

            if wandb_logger is not None:
                wandb_logger.log_dict(eval_info, step + len(train_loader), mode="eval")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(checkpoint_dir, step, cfg, model, optimizer, None)

        # Periodic checkpointing
        if cfg.save_checkpoint and (epoch + 1) % cfg.save_freq == 0:
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(checkpoint_dir, step, cfg, model, optimizer, None)

        step += len(train_loader)

    #benchmark_inference_time(model, dataset, wandb_logger, cfg, device, step)

    logging.info("Training completed")


if __name__ == "__main__":
    train()
