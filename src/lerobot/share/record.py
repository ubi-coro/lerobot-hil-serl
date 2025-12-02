import logging
import time
from dataclasses import asdict
from pprint import pformat
from typing import Any

import numpy as np
import torch
from torch.autograd.profiler import record_function, profile, ProfilerActivity

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.envs.configs import ResetConfig
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
from lerobot.utils.constants import ACTION, REWARD, DONE
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


@safe_stop_image_writer
def record_loop(
    env: RobotEnv,
    fps: int,
    action_dim: int,
    action_processor: RobotProcessorPipeline[Transition, Transition],
    env_processor: RobotProcessorPipeline[Transition, Transition],
    dataset: LeRobotDataset | None = None,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
    control_time_s: float | None = None,
    single_task: str | None = None,
    robot_type: str| None = None,
    display_data: bool = False,
    device: str = "cuda",
    interactive: bool = False
):
    if control_time_s is None:
        control_time_s = float("inf")

    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    has_policy = policy is not None and preprocessor is not None and postprocessor is not None
    teleoperate = not has_policy
    assert not interactive or has_policy, "Interactive recording requires a policy."

    # Reset policy and processor if they are provided
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    obs, info = env.reset()
    env_processor.reset()
    action_processor.reset()

    # Process initial observation
    transition = create_transition(observation=obs, info=info)
    transition = env_processor(data=transition)  # outputs valid transition

    intervention_occurred = False
    info: dict = transition[TransitionKey.INFO]
    episode_step = 0
    episode_start_time = time.perf_counter()
    while (time.perf_counter() - episode_start_time) < control_time_s:
        start_loop_t = time.perf_counter()

        # (1) Keep the PRE-STEP transition (this holds o_t) and reset info dict
        info = {}

        # (2) Handle intervention control flow
        if teleoperate:
            # Permanently set the intervention flag to stay in control
            info[TeleopEvents.IS_INTERVENTION] = True

        # (3) Decide and process action a_t
        if has_policy:
            policy_observation = {
                k: v for k, v in transition[TransitionKey.OBSERVATION].items() if k in policy.config.input_features
            }
            # noinspection PyTypeChecker
            action = predict_action(
                observation=policy_observation,
                policy=policy,
                device=get_safe_torch_device(device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=single_task,
                robot_type=robot_type
            )
        else:
            # Dummy action, expected to be overwritten by teleop action
            action = torch.tensor([0.0] * action_dim, dtype=torch.float32)

        new_transition = step_env_and_process_transition(
            env=env,
            action=action,
            env_processor=env_processor,
            action_processor=action_processor,
            info=info,
            exit_early_on_intervention_end=True  # we want skip stepping the environment if an intervention ends
        )

        action = new_transition[TransitionKey.ACTION]
        reward = new_transition[TransitionKey.REWARD]
        done = new_transition.get(TransitionKey.DONE, False)
        truncated = new_transition.get(TransitionKey.TRUNCATED, False)
        info = new_transition.get(TransitionKey.INFO, {})

        # exit on episode end
        if info.get(TeleopEvents.INTERVENTION_COMPLETED, False):
            episode_time = time.perf_counter() - episode_start_time
            logging.info(
                f"Intervention ended after {episode_step} steps in {episode_time:.1f}s with reward {transition[TransitionKey.REWARD]}"
            )
            env.stop()
            return info

        # (8) Store transition. When interactive, only store frames on interventions
        # store o_t, a_t, r_t+1
        if dataset is not None and (not interactive or info.get(TeleopEvents.IS_INTERVENTION, False)):

            # observations are batched and may contain other keys
            dataset_observation = {
                k: v.squeeze().cpu()
                for k, v in transition[TransitionKey.OBSERVATION].items()
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
                log_rerun_data(observation=rerun_obs, action=action)

        transition = new_transition
        episode_step += 1

        # (9) Handle done
        if (
                done or
                truncated or
                info.get(TeleopEvents.RERECORD_EPISODE, False) or
                info.get(TeleopEvents.TERMINATE_EPISODE, False) or
                info.get(TeleopEvents.PAUSE_RECORDING, False)
        ):
            episode_time = time.perf_counter() - episode_start_time
            logging.info(
                f"Episode ended after {episode_step} steps in {episode_time:.1f}s with reward {transition[TransitionKey.REWARD]}"
            )
            env.stop()
            return info

        # (10) Handle frequency
        dt_load = time.perf_counter() - start_loop_t
        precise_sleep(1 / fps - dt_load)
        dt_loop = time.perf_counter() - start_loop_t
        logging.info(
            f"dt_loop: {dt_loop * 1000:5.2f}ms ({1 / dt_loop:3.1f}hz), "
            f"dt_load: {dt_load * 1000:5.2f}ms ({1 / dt_load:3.1f}hz)"
        )

    else:
        env.stop()
        return info

@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="recording")

    # make env
    env, env_processor, action_processor = cfg.env.make(device="cpu" if cfg.policy is None else cfg.policy.device)

    # handle timing
    reset_cfg: ResetConfig = cfg.env.processor.reset
    if cfg.dataset.reset_time_s is not None:
        reset_cfg.teleop_on_reset = True
        reset_cfg.reset_time_s = cfg.dataset.reset_time_s

    # make dataset
    features = env_to_dataset_features(cfg.env.features)
    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )

        if hasattr(env, "cameras") and len(env.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.dataset.num_image_writer_processes,
                num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(env.cameras),
            )
        sanity_check_dataset_robot_compatibility(dataset, cfg.env.type, cfg.dataset.fps, features)
    else:
        # Create empty dataset or load existing saved episodes
        sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.env.fps,
            root=cfg.dataset.root,
            robot_type=cfg.env.type,
            features=features,
            use_videos=cfg.dataset.video,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(cfg.env.cameras),
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )

    # Load pretrained policy
    policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)
    preprocessor = None
    postprocessor = None
    if cfg.policy is not None:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            dataset_stats=rename_stats(dataset.meta.stats, cfg.dataset.rename_map),
            preprocessor_overrides={
                "device_processor": {"device": cfg.policy.device},
                "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
            },
        )

    info = {}
    with ((VideoEncodingManager(dataset))):
        recorded_episodes = 0
        paused = False
        while recorded_episodes < cfg.dataset.num_episodes and not info.get(TeleopEvents.STOP_RECORDING, False):

            # Execute a few seconds without recording to give time to manually reset the environment
            if reset_cfg.teleop_on_reset and not info.get(TeleopEvents.INTERVENTION_COMPLETED, False):
                log_say("Reset the environment", cfg.play_sounds, blocking=True)

                info = record_loop(
                    env=env,
                    fps=cfg.env.fps,
                    control_time_s=reset_cfg.reset_time_s,
                    action_dim=features[ACTION]["shape"][0],
                    action_processor=action_processor,
                    env_processor=env_processor,
                    policy=None,
                    dataset=None,
                    display_data=cfg.display_data,
                    interactive=False
                )

                # if we teleop on reset, env.reset should not perform the env reset
                # still call reset to reset internal states (state counter, buffers), reset again
                env.reset()
                if policy is not None and preprocessor is not None and postprocessor is not None:
                    policy.reset()
                    preprocessor.reset()
                    postprocessor.reset()

            if info.get(TeleopEvents.STOP_RECORDING, False):
                break

            log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds, blocking=True)
            info = record_loop(
                env=env,
                fps=cfg.env.fps,
                control_time_s=cfg.dataset.episode_time_s,
                action_dim=features[ACTION]["shape"][0],
                action_processor=action_processor,
                env_processor=env_processor,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                dataset=dataset,
                interactive=cfg.interactive,
                single_task=cfg.dataset.single_task,
                robot_type=cfg.env.type,
                display_data=cfg.display_data
            )

            if info.get(TeleopEvents.RERECORD_EPISODE, False):
                log_say("Re-record episode", cfg.play_sounds)
                dataset.clear_episode_buffer()
                continue

            if dataset.episode_buffer["size"] > 0:
                dataset.save_episode()
                recorded_episodes += 1

                # todo: check for pause and do it here
                if info.get(TeleopEvents.PAUSE_RECORDING, False):
                    log_say("Pause", cfg.play_sounds)
                continue



            if paused:
                # todo: shoul be in a loop that stays here until resume
                if info.get(TeleopEvents.RESUME_RECORDING, False):
                    log_say("Resume recording", cfg.play_sounds)
                    paused = False
                else:
                    continue

            if info.get(TeleopEvents.STOP_RECORDING, False):
                break

            if dataset.episode_buffer["size"] == 0:
                log_say("Dataset is empty, re-record episode", cfg.play_sounds, blocking=True)

    log_say("Stop recording", cfg.play_sounds, blocking=True)

    env.close()

    if cfg.dataset.push_to_hub:
        log_say("Uploading dataset", cfg.play_sounds, blocking=True)
        dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

    return dataset


if __name__ == "__main__":
    import experiments

    record()

    #sort_by_keyword = "cuda_time_total"
    #with profile(use_device="cuda", record_shapes=True) as prof:
    #    with record_function("record"):


    #print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=10))
