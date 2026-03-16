import logging
import time
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from typing import Any, Optional

import numpy as np
import torch
from torch.autograd.profiler import record_function, profile, ProfilerActivity

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.envs.configs import ResetConfig, HILSerlProcessorConfig
from lerobot.envs.robot_env import RobotEnv
from lerobot.envs.utils import env_to_dataset_features
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotProcessorPipeline,
    create_transition,
    TransitionKey
)
from lerobot.processor.rename_processor import rename_stats
from lerobot.rl.gym_manipulator import step_env_and_process_transition
from lerobot.share.configs import RecordConfig
from lerobot.teleoperators import TeleopEvents
from lerobot.utils.constants import ACTION, REWARD, DONE, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK
from lerobot.utils.control_utils import (
    predict_action,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.transition import Transition
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
    log_say,
)
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from share.envs.manipulation_primitive_net.env_manipulation_primitive_net import ManipulationPrimitiveNet
from share.utils.control_utils import MPNetStepCounter
from share.utils.video_utils import MultiVideoEncodingManager

""" --------------- record_loop() data flow --------------------------
       [ Robot ]
           V
     [ robot.get_observation() ] ---> raw_obs
           V
     [ robot_observation_processor ] ---> processed_obs
           V
     .-----( ACTION LOGIC )------------------.
     V                                       V
     [ From Teleoperator ]                   [ From Policy ]
     |                                       |
     |  [teleop.get_action] -> raw_action    |   [predict_action]
     |          |                            |          |
     |          V                            |          V
     | [teleop_action_processor]             |          |
     |          |                            |          |
     '---> processed_teleop_action           '---> processed_policy_action
     |                                       |
     '-------------------------.-------------'
                               V
                  [ robot_action_processor ] --> robot_action_to_send
                               V
                    [ robot.send_action() ] -- (Robot Executes)
                               V
                    ( Save to Dataset )
                               V
                  ( Rerun Log / Loop Wait )
"""


def make_policies_and_datasets(cfg: RecordConfig):
    datasets = {}
    policies = {}
    preprocessors = {}
    postprocessors = {}
    for name, p in cfg.env.primitives.items():
        if p.is_adaptive:

            # 1) dataset
            root = Path(cfg.dataset.root) / name
            repo_id = f"{cfg.dataset.repo_id}-{name}"

            if cfg.resume:
                datasets[name] = LeRobotDataset(
                    repo_id,
                    root=root,
                    batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                    vcodec=cfg.dataset.vcodec,
                )
                datasets[name].start_image_writer(
                    num_processes=cfg.dataset.num_image_writer_processes,
                    num_threads=cfg.dataset.num_image_writer_threads_per_camera * p.num_cameras
                )

            else:
                datasets[name] = LeRobotDataset.create(
                    repo_id,
                    cfg.env.fps,
                    root=root,
                    features=p.features,
                    robot_type=cfg.env.type,
                    use_videos=cfg.dataset.video,
                    image_writer_processes=cfg.dataset.num_image_writer_processes,
                    image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * p.num_cameras,
                    batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                    vcodec=cfg.dataset.vcodec,
                )

            # 2) policy
            if p.policy.pretrained_path is not None:
                cli_overrides = parser.get_cli_overrides("policy")
                p.policy = PreTrainedConfig.from_pretrained(p.policy.pretrained_path)  # , cli_overrides=cli_overrides)

            policies[name] = make_policy(cfg=p.policy, ds_meta=datasets[name].meta)
            policies[name] = policies[name].eval()

            pre, post = make_pre_post_processors(
                policy_cfg=p.policy,
                pretrained_path=p.policy.pretrained_path,
                dataset_stats=rename_stats(datasets[name].meta.stats, cfg.dataset.rename_map),
                preprocessor_overrides={
                    "device_processor": {"device": p.policy.device},
                    "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
                },
            )
            preprocessors[name] = pre
            postprocessors[name] = post

    return datasets, policies, preprocessors, postprocessors


@safe_stop_image_writer
def record_loop(
    mp_net: ManipulationPrimitiveNet,
    datasets: dict[str, LeRobotDataset],
    policies: dict[str, PreTrainedPolicy],
    preprocessors: dict[str, PolicyProcessorPipeline[dict[str, Any], dict[str, Any]]],
    postprocessors: dict[str, PolicyProcessorPipeline[PolicyAction, PolicyAction]],
    single_task: str | None = None,
    display_data: bool = False,
    display_compressed_images: bool = False,
    interactive: bool = False
):
    transition = mp_net.reset()

    # check if we need to terminate early
    info = transition.get(TransitionKey.INFO, {})
    if info.get(TeleopEvents.STOP_RECORDING, False):
        return info

    sum_reward = 0.0
    while True:
        start_loop_t = time.perf_counter()
        obs = transition[TransitionKey.OBSERVATION]
        policy = policies.get(mp_net.active_primitive, None)
        dataset = datasets.get(mp_net.active_primitive, None)

        # (1) Decide and process action a_t
        if policy is not None:
            # noinspection PyTypeChecker
            action = predict_action(
                observation=obs,
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
                preprocessor=preprocessors[mp_net.active_primitive],
                postprocessor=postprocessors[mp_net.active_primitive],
                use_amp=policy.config.use_amp,
                task=single_task,
                robot_type=mp_net.config.type
            )
        else:
            # Permanently set the intervention flag to stay in control
            info[TeleopEvents.IS_INTERVENTION] = True

            # Dummy action, expected to be overwritten by teleop action
            action = torch.tensor([0.0] * mp_net.action_dim, dtype=torch.float32)

        # (2) Step environment
        new_transition = mp_net.step(action)

        action = new_transition[TransitionKey.ACTION]
        reward = new_transition[TransitionKey.REWARD]
        done = new_transition.get(TransitionKey.DONE, False)
        truncated = new_transition.get(TransitionKey.TRUNCATED, False)
        info = new_transition.get(TransitionKey.INFO, {})
        sum_reward += float(reward)

        # (3) Exit on episode end
        if info.get(TeleopEvents.INTERVENTION_COMPLETED, False):
            return info

        # (4) Store transition. When interactive, only store frames on interventions
        # store o_t, a_t, r_t+1
        if dataset is not None and (not interactive or info.get(TeleopEvents.IS_INTERVENTION, False)):
            # observations are batched and may contain other keys
            dataset_observation = {
                k: v.squeeze().cpu()
                for k, v in obs.items()
                if k in dataset.features
            }

            # store frame
            frame = {
                **dataset_observation,
                ACTION: action.squeeze().cpu(),
                REWARD: np.array([reward], dtype=np.float32),
                DONE: np.array([done], dtype=bool),
                "task": single_task
            }
            dataset.add_frame(frame)

            if display_data:
                rerun_obs = {k: v.numpy() for k, v in dataset_observation.items()}
                log_rerun_data(
                    observation=rerun_obs, action=action.squeeze().cpu(), compress_images=display_compressed_images
                )

        # (5) Update current observation
        transition = new_transition

        # 6) Handle done
        # Termination refers to whether we are in a terminal primitive
        # Only from here are we able to cleanly reset to starting conditions
        if (
            done or
            truncated or
            info.get(TeleopEvents.RERECORD_EPISODE, False)
        ):
            return info

        # (7) Handle frequency
        dt_load = time.perf_counter() - start_loop_t
        precise_sleep(1 / mp_net.config.fps - dt_load)
        dt_loop = time.perf_counter() - start_loop_t
        logging.info(
            f"dt_loop: {dt_loop * 1000:5.2f}ms ({1 / dt_loop:3.1f}hz), "
            f"dt_load: {dt_load * 1000:5.2f}ms ({1 / dt_load:3.1f}hz)"
        )

@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="recording", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    # make
    mp_net = ManipulationPrimitiveNet(cfg.env)
    datasets, policies, preprocessors, postprocessors = make_policies_and_datasets(cfg)

    with MultiVideoEncodingManager(datasets):
        while True:

            dataset = datasets.get(mp_net.active_primitive, None)

            info = record_loop(
                mp_net=mp_net,
                datasets=datasets,
                policies=policies,
                preprocessors=preprocessors,
                postprocessors=postprocessors,
                display_data=cfg.display_data,
                display_compressed_images=display_compressed_images,
                interactive=cfg.interactive
            )

            if info.get(TeleopEvents.STOP_RECORDING, False):
                break

            if dataset is None:
                continue

            # dataset ops, saving / clearing episode buffers
            if info.get(TeleopEvents.RERECORD_EPISODE, False):
                log_say("Re-record episode", cfg.play_sounds, blocking=True)
                dataset.clear_episode_buffer()
            elif dataset.episode_buffer["size"] > 0:
                log_say("Save episode", cfg.play_sounds, blocking=False)
                dataset.save_episode()
            else:
                log_say("Dataset is empty, continue execution", cfg.play_sounds, blocking=True)

    log_say("Stop recording", cfg.play_sounds, blocking=True)
    mp_net.close()



if __name__ == "__main__":
    import experiments
    record()