"""GUI Recording Worker (Phase 1)
=================================

Implements a Socket.IO-driven wrapper around LeRobot's new `record_loop()`
and dataset APIs without modifying upstream code.

Phase 1 goals:
 - Start/stop dataset recording from the GUI via Socket.IO
 - Minimal status emission (episodes, frames, fps, timing)
 - Support basic rerecord / skip / stop controls
 - Reuse already connected robot when possible (teleoperation/robot_service)

Deferred to later phases:
 - Policy/eval (interactive) runs
 - Intervention toggling
 - Fine‑grained per-frame latency metrics
 - Push-to-hub manual trigger beyond cfg.push_to_hub
 - Resume validation UI / advanced queue metrics

Design notes:
 - We call `lerobot.record.record_loop` for warmup/record/reset segments,
     passing our own thread-safe event adapter.
 - A background asyncio task emits status every 0.5s.
 - The heavy loop runs in a dedicated thread.
 - We compute dataset features from robot action/observation features,
     mirroring upstream logic in `lerobot.record`.
"""

from __future__ import annotations

import threading
import time
import logging
import asyncio
from pathlib import Path
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import ProcessorStep, TransitionKey, EnvTransition
from lerobot.share.record import record_loop as shared_record_loop
from lerobot.share.utils import get_pipeline_dataset_features
from lerobot.teleoperators import TeleopEvents
from lerobot.utils.control_utils import (
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.utils import log_say, get_safe_torch_device
from lerobot.processor.rename_processor import rename_stats

from ..experiment_config_mapper import ExperimentConfigMapper, ExperimentMapping

logger = logging.getLogger(__name__)


@dataclass
class RecordControlConfig:
    """Minimal config for GUI recording.

    Mirrors the fields we use from the UI; not tied to upstream dataclasses.
    """
    repo_id: str
    single_task: str
    fps: int
    warmup_time_s: float
    episode_time_s: float
    reset_time_s: float
    num_episodes: int
    video: bool = True
    push_to_hub: bool = False
    private: bool = False
    tags: Optional[list[str]] = None
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4
    video_encoding_batch_size: int = 1
    rename_map: Dict[str, str] = field(default_factory=dict)
    display_data: bool = False
    resume: bool = False
    root: Optional[str] = None
    play_sounds: bool = False
    mode: str = "recording"  # "recording" or "replay"
    policyPath: Optional[str] = None  # Path to pretrained_model for replay mode
    operation_mode: str = "bimanual"
    interactive: bool = False
    # Optional future fields we may ignore safely
    save_eval: bool = True


try:  # Optional imports for robot reuse
    from .aloha_teleoperation import aloha_state  # type: ignore
except Exception:  # pragma: no cover - optional
    aloha_state = None

try:
    # Import the module, not the variable, to avoid stale references.
    from . import robot as robot_module  # type: ignore
except Exception:  # pragma: no cover
    robot_module = None

try:
    import shared  # global Socket.IO accessor
except Exception as e:  # pragma: no cover
    shared = None  # type: ignore
    logger.warning(f"Shared socket module not available: {e}")


class ApiEventAdapter(dict):
    """Thread-safe event container replacing keyboard-driven ControlEvents.

    Flags used by upstream logic:
      exit_early, rerecord_episode, stop_recording, intervention (unused phase 1)
    """

    def __init__(self):
        super().__init__(
            {
                "exit_early": False,
                "rerecord_episode": False,
                "stop_recording": False,
                "intervention": False,
            }
        )
        self._lock = threading.Lock()

    def set_flag(self, key: str, value: bool = True):
        with self._lock:
            if key in self:
                self[key] = value

    def toggle(self, key: str):
        with self._lock:
            if key in self:
                self[key] = not self[key]

    def consume(self, key: str) -> bool:
        with self._lock:
            value = bool(self.get(key, False))
            if value:
                self[key] = False
            return value

    def reset(self):  # override to keep thread safety
        with self._lock:
            self["exit_early"] = False
            self["rerecord_episode"] = False
            # do not reset stop_recording here
            self["intervention"] = False

    def update(self):  # upstream calls this each loop; nothing required
        return


@dataclass
class ApiCommandActionProcessorStep(ProcessorStep):
    """Inject GUI-issued commands into the processor pipeline as teleop events."""

    events: ApiEventAdapter

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        info = dict(new_transition.get(TransitionKey.INFO, {}))
        should_terminate = False

        if self.events.consume("stop_recording"):
            info[TeleopEvents.STOP_RECORDING] = True
            should_terminate = True

        if self.events.consume("rerecord_episode"):
            info[TeleopEvents.RERECORD_EPISODE] = True
            info[TeleopEvents.TERMINATE_EPISODE] = True
            should_terminate = True

        if self.events.consume("exit_early"):
            info.setdefault(TeleopEvents.TERMINATE_EPISODE, True)
            should_terminate = True

        if should_terminate:
            new_transition[TransitionKey.DONE] = True
            new_transition[TransitionKey.TRUNCATED] = True

        if info:
            new_transition[TransitionKey.INFO] = info

        return new_transition


class RecordingWorkerState:
    def __init__(self):
        self.active: bool = False
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.events: Optional[ApiEventAdapter] = None
        self.cfg: Optional[RecordControlConfig] = None
        self.dataset: Optional[LeRobotDataset] = None
        self.env = None
        self.env_processor = None
        self.action_processor = None
        self.mapping: Optional[ExperimentMapping] = None
        self.episode_index: int = 0
        self.total_frames: int = 0
        self.episode_frames: int = 0
        self.episode_start_t: float | None = None
        # Phase tracking (for countdown progress): one of "idle","warmup","recording","resetting","transition"
        self.phase: str = "idle"
        self.phase_start_t: float | None = None
        self.phase_total_s: float | None = None
        self.status_lock = threading.Lock()
        self.last_status: Dict[str, Any] = {}
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.display_data_session_started = False
        # When resuming, capture how many episodes already existed before starting new session
        self.existing_dataset_episodes: int | None = None

    def snapshot(self) -> Dict[str, Any]:
        with self.status_lock:
            now = time.perf_counter()
            episode_elapsed = None
            if self.episode_start_t is not None:
                episode_elapsed = now - self.episode_start_t
            phase_elapsed = None
            if self.phase_start_t is not None:
                phase_elapsed = now - self.phase_start_t
            return {
                "active": self.active,
                "episode_index": self.episode_index,
                "total_episodes": getattr(self.cfg, "num_episodes", 0) or 0,
                # Episodes that were already present when starting (resume mode)
                "existing_episodes": self.existing_dataset_episodes,
                "episode_frames": self.episode_frames,
                "total_frames": self.total_frames,
                "episode_elapsed_s": episode_elapsed,
                "episode_duration_s": getattr(self.cfg, "episode_time_s", 0) or 0,
                "repo_id": getattr(self.cfg, "repo_id", None),
                "single_task": getattr(self.cfg, "single_task", None),
                "fps_target": getattr(self.cfg, "fps", 0) or 0,
                # fps_current simple estimate: frames / elapsed
                "fps_current": (
                    (self.episode_frames / episode_elapsed) if episode_elapsed and episode_elapsed > 0 else None
                ),
                # UI hint: if a re-record has been requested for the current episode
                "rerecord_pending": bool(self.events["rerecord_episode"]) if self.events is not None else False,
                # Phase & countdown support for UI
                "phase": self.phase,
                "phase_elapsed_s": phase_elapsed or 0,
                "phase_total_s": self.phase_total_s or 0,
                # Keep legacy 'state' for compatibility but align with phase
                "state": self.phase if self.active or self.phase != "idle" else "idle",
            }

    def _event_flag(self, key: str) -> bool:
        try:
            if self.events is None:
                return False
            # ApiEventAdapter supports dict-like access
            return bool(self.events[key])
        except Exception:
            return False


recording_worker = RecordingWorkerState()

def start_recording_via_api(config: Dict[str, Any]):
    if recording_worker.active:
        raise RuntimeError("Recording already active")

    required = ["repo_id", "single_task", "fps", "episode_time_s", "num_episodes"]
    missing = [field for field in required if field not in config]
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")

    def _pos_int(value, default=None):
        try:
            return int(value)
        except Exception:
            return default

    def _pos_float(value, default=None):
        try:
            return float(value)
        except Exception:
            return default

    warmup_time_s = _pos_float(config.get("warmup_time_s"), 10.0)
    episode_time_s = _pos_float(config.get("episode_time_s"), 30.0)
    reset_time_s = _pos_float(config.get("reset_time_s"), 10.0)

    fps_val = _pos_int(config.get("fps"), None)
    if fps_val is None or fps_val <= 0:
        raise ValueError(f"Invalid fps value: {config.get('fps')} (must be a positive integer)")

    num_img_writer_proc = _pos_int(config.get("num_image_writer_processes", 0), 0)
    num_img_writer_threads_per_cam = _pos_int(config.get("num_image_writer_threads_per_camera", 4), 4)
    video_batch_size = max(1, _pos_int(config.get("video_encoding_batch_size", 1), 1) or 1)

    rename_map_val = config.get("rename_map") or {}
    if not isinstance(rename_map_val, dict):
        rename_map_val = {}

    play_sounds = bool(config.get("play_sounds", False))

    num_episodes_val = _pos_int(config.get("num_episodes", 1), 1)
    if num_episodes_val < 1:
        num_episodes_val = 1

    operation_mode = config.get("operation_mode")
    if not operation_mode and aloha_state and aloha_state.get("config"):
        operation_mode = aloha_state["config"].get("operation_mode")
    normalized_mode = ExperimentConfigMapper.normalize_operation_mode(operation_mode)

    cfg = RecordControlConfig(
        repo_id=config["repo_id"],
        single_task=config["single_task"],
        fps=fps_val,
        warmup_time_s=warmup_time_s,
        episode_time_s=episode_time_s,
        reset_time_s=reset_time_s,
        num_episodes=num_episodes_val,
        video=config.get("video", True),
        push_to_hub=config.get("push_to_hub", False),
        private=config.get("private", False),
        tags=config.get("tags"),
        num_image_writer_processes=num_img_writer_proc,
        num_image_writer_threads_per_camera=num_img_writer_threads_per_cam,
        video_encoding_batch_size=video_batch_size,
        rename_map=rename_map_val,
        display_data=config.get("display_data", False),
        resume=config.get("resume", False),
        root=config.get("root"),
        play_sounds=play_sounds,
        mode=config.get("mode", "recording"),
        policyPath=config.get("policyPath"),
        operation_mode=normalized_mode,
        interactive=bool(config.get("interactive", False)),
    )

    cfg.num_episodes = cfg.num_episodes or 1
    cfg.episode_time_s = cfg.episode_time_s or 30.0
    cfg.warmup_time_s = cfg.warmup_time_s or 10.0
    cfg.reset_time_s = cfg.reset_time_s or 10.0
    cfg.video = True if cfg.video is None else cfg.video
    cfg.push_to_hub = bool(cfg.push_to_hub)
    cfg.private = bool(cfg.private)
    cfg.resume = bool(cfg.resume)
    cfg.display_data = bool(cfg.display_data)
    cfg.rename_map = cfg.rename_map or {}
    cfg.play_sounds = bool(cfg.play_sounds)

    demo_mode = cfg.mode == "replay"
    mapping = ExperimentConfigMapper.resolve_mapping(cfg.operation_mode, demo_mode, cfg.policyPath)
    if demo_mode and not (mapping.supports_policy or cfg.policyPath):
        raise ValueError(f"Replay mode for '{cfg.operation_mode}' requires a configured policy path.")
    if not cfg.policyPath and mapping.default_policy_path:
        cfg.policyPath = mapping.default_policy_path
    if cfg.policyPath and not cfg.policyPath.endswith("pretrained_model"):
        raise ValueError("policyPath should end with 'pretrained_model'")

    if aloha_state and aloha_state.get("active"):
        raise RuntimeError("Cannot start recording while teleoperation is active. Please stop teleoperation first.")

    events = ApiEventAdapter()
    recording_worker.stop_event.clear()

    recording_worker.active = True
    recording_worker.cfg = cfg
    recording_worker.events = events
    recording_worker.episode_index = 0
    recording_worker.total_frames = 0
    recording_worker.episode_frames = 0
    recording_worker.episode_start_t = None
    recording_worker.mapping = mapping
    with recording_worker.status_lock:
        recording_worker.phase = "transition"
        recording_worker.phase_start_t = None
        recording_worker.phase_total_s = None

    def _worker():
        env = None
        env_processor = None
        action_processor = None
        dataset: Optional[LeRobotDataset] = None
        policy: Optional[PreTrainedPolicy] = None
        preprocessor = None
        postprocessor = None
        policy_cfg: Optional[PreTrainedConfig] = None
        policy_device = get_safe_torch_device("cpu")
        use_amp = False
        last_info: Dict[str, Any] = {}
        borrowed_cameras = None
        
        try:
            # Borrow cameras from robot module if available
            try:
                from . import robot as robot_module
                borrowed_cameras = robot_module.borrow_cameras("recording_worker")
                if borrowed_cameras:
                    logger.info(f"Successfully borrowed {len(borrowed_cameras)} cameras from robot module for recording")
                else:
                    logger.warning("Cameras not available from robot module for recording")
            except RuntimeError as e:
                logger.error(f"Failed to borrow cameras: {e}")
                borrowed_cameras = None
            except Exception as e:
                logger.debug(f"Could not borrow cameras from robot module: {e}")
                borrowed_cameras = None
            
            root_path = Path(cfg.root) if cfg.root else None
            if root_path and root_path.exists() and not cfg.resume:
                if root_path.is_dir() and not any(root_path.iterdir()):
                    try:
                        root_path.rmdir()
                        logger.info("Removed empty dataset root to allow creation: %s", root_path)
                    except Exception:
                        pass
                else:
                    if (root_path / "meta" / "info.json").exists():
                        raise RuntimeError(
                            f"Existing dataset detected at {root_path}. Enable 'Resume' or pick a different root."
                        )
                    raise RuntimeError(
                        f"Dataset root exists and is not empty: {root_path}. Choose a different root or enable Resume."
                    )

            policy_cfg_for_name = None
            if demo_mode:
                resolved_policy_path = cfg.policyPath or mapping.default_policy_path
                if not resolved_policy_path:
                    raise RuntimeError("Replay mode active but no policy path available.")
                policy_cfg_for_name = PreTrainedConfig.from_pretrained(resolved_policy_path)
                policy_cfg_for_name.pretrained_path = resolved_policy_path
                cfg.policyPath = resolved_policy_path

            sanity_check_dataset_name(cfg.repo_id, policy_cfg_for_name)

            env, env_processor, action_processor, env_cfg, mapping_local = ExperimentConfigMapper.create_env_from_gui_selection(
                cfg.operation_mode,
                demo_mode,
                cfg.policyPath,
                use_cameras=False,  # Never create new camera connections; use borrowed ones if available
            )
            
            # If we successfully borrowed cameras, inject them into the environment
            if borrowed_cameras:
                logger.info("Injecting borrowed cameras into recording environment")
                if hasattr(env, 'cameras'):
                    env.cameras = borrowed_cameras
                # Also inject into robot_dict if it exists (for bimanual setups)
                if hasattr(env, 'robot_dict'):
                    for robot_name, robot_instance in env.robot_dict.items():
                        if hasattr(robot_instance, 'cameras'):
                            robot_instance.cameras = borrowed_cameras
                            logger.debug(f"Injected cameras into robot: {robot_name}")
            
            recording_worker.env = env
            recording_worker.env_processor = env_processor
            recording_worker.action_processor = action_processor
            recording_worker.mapping = mapping_local

            action_processor.steps.append(ApiCommandActionProcessorStep(events))

            dataset_features = get_pipeline_dataset_features(
                env=env,
                env_processor=env_processor,
                action_dim=env_cfg.action_dim,
                use_video=cfg.video,
            )

            camera_count = len(getattr(env, "cameras", {}) or {})

            if demo_mode:
                dataset = None
                recording_worker.existing_dataset_episodes = None
                recording_worker.dataset = None
            else:
                if cfg.resume:
                    dataset = LeRobotDataset(
                        cfg.repo_id,
                        root=cfg.root,
                        batch_encoding_size=cfg.video_encoding_batch_size,
                    )
                    if camera_count > 0:
                        dataset.start_image_writer(
                            num_processes=cfg.num_image_writer_processes,
                            num_threads=cfg.num_image_writer_threads_per_camera * camera_count,
                        )
                    sanity_check_dataset_robot_compatibility(dataset, env_cfg.type, cfg.fps, dataset_features)
                    try:
                        recording_worker.existing_dataset_episodes = int(getattr(dataset, "num_episodes", 0))
                    except Exception:
                        recording_worker.existing_dataset_episodes = None
                else:
                    dataset = LeRobotDataset.create(
                        repo_id=cfg.repo_id,
                        fps=cfg.fps,
                        root=cfg.root,
                        robot_type=env_cfg.type,
                        features=dataset_features,
                        use_videos=cfg.video,
                        image_writer_processes=cfg.num_image_writer_processes,
                        image_writer_threads=cfg.num_image_writer_threads_per_camera * camera_count,
                        batch_encoding_size=cfg.video_encoding_batch_size,
                    )
                    recording_worker.existing_dataset_episodes = 0

                recording_worker.dataset = dataset

                # Hook frame counting only when a dataset exists
                orig_add_frame = dataset.add_frame

                def add_frame_hook(frame):
                    with recording_worker.status_lock:
                        recording_worker.total_frames += 1
                        recording_worker.episode_frames += 1
                    return orig_add_frame(frame)

                dataset.add_frame = add_frame_hook  # type: ignore

            if demo_mode:
                policy_cfg = policy_cfg_for_name or PreTrainedConfig.from_pretrained(cfg.policyPath)
                policy_cfg.pretrained_path = cfg.policyPath
                policy_device = get_safe_torch_device(getattr(policy_cfg, "device", "cpu") or "cpu")
                policy_cfg.device = str(policy_device)
                use_amp = bool(getattr(policy_cfg, "use_amp", False))
                # In demo mode, avoid dataset dependency: infer features from env_cfg and load processors from pretrained
                policy = make_policy(policy_cfg, env_cfg=env_cfg)
                preprocessor, postprocessor = make_pre_post_processors(
                    policy_cfg=policy_cfg,
                    pretrained_path=policy_cfg.pretrained_path,
                    preprocessor_overrides={
                        "device_processor": {"device": str(policy_device)},
                        "rename_observations_processor": {"rename_map": cfg.rename_map},
                    },
                    postprocessor_overrides={
                        "device_processor": {"device": str(policy_device)},
                    },
                )
                policy.reset()

            try:
                warmup_timeout = int(cfg.warmup_time_s or 10)
                _preflight_cameras(env, timeout_s=min(10, max(3, warmup_timeout)))
            except Exception as cam_e:
                raise RuntimeError(
                    f"Camera preflight failed: {cam_e}. Ensure cameras are connected and not used elsewhere."
                ) from cam_e

            teleop_reset_cfg = getattr(getattr(env_cfg, "processor", None), "reset", None)
            teleop_on_reset = bool(getattr(teleop_reset_cfg, "teleop_on_reset", False))

            manager = VideoEncodingManager(dataset) if dataset is not None else nullcontext()
            with manager:
                if teleop_on_reset:
                    with recording_worker.status_lock:
                        recording_worker.phase = "warmup"
                        recording_worker.phase_total_s = float(cfg.warmup_time_s or 0)
                        recording_worker.phase_start_t = time.perf_counter()
                    last_info = shared_record_loop(
                        env=env,
                        fps=cfg.fps,
                        action_dim=env_cfg.action_dim,
                        action_processor=action_processor,
                        env_processor=env_processor,
                        dataset=None,
                        policy=None,
                        preprocessor=None,
                        postprocessor=None,
                        control_time_s=cfg.warmup_time_s,
                        single_task=cfg.single_task,
                        robot_type=env_cfg.type,
                        display_data=cfg.display_data,
                        device="cpu",
                        use_amp=False,
                        interactive=False,
                    )
                    events.reset()

                recorded_episodes = 0
                while recorded_episodes < cfg.num_episodes and not last_info.get(
                    TeleopEvents.STOP_RECORDING, False
                ):
                    with recording_worker.status_lock:
                        recording_worker.phase = "recording"
                        recording_worker.phase_total_s = float(cfg.episode_time_s or 0)
                        recording_worker.phase_start_t = time.perf_counter()
                        recording_worker.episode_frames = 0
                        recording_worker.episode_start_t = time.perf_counter()

                    last_info = shared_record_loop(
                        env=env,
                        fps=cfg.fps,
                        action_dim=env_cfg.action_dim,
                        action_processor=action_processor,
                        env_processor=env_processor,
                        dataset=(None if demo_mode else dataset),
                        policy=policy,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        control_time_s=cfg.episode_time_s,
                        single_task=cfg.single_task,
                        robot_type=env_cfg.type,
                        display_data=cfg.display_data,
                        device=str(policy_device),
                        use_amp=use_amp,
                        interactive=cfg.interactive,
                    )

                    if last_info.get(TeleopEvents.RERECORD_EPISODE, False):
                        log_say("Re-record episode", cfg.play_sounds)
                        if dataset is not None:
                            dataset.clear_episode_buffer()
                        events.reset()
                        continue

                    if demo_mode:
                        # In demo mode, we do not save any data; count an episode on completion
                        recorded_episodes += 1
                        recording_worker.episode_index = recorded_episodes
                    else:
                        episode_size = int(dataset.episode_buffer.get("size", 0))
                        if episode_size > 0:
                            with recording_worker.status_lock:
                                recording_worker.phase = "processing"
                                recording_worker.phase_total_s = None
                                recording_worker.phase_start_t = time.perf_counter()
                            dataset.save_episode()
                            recorded_episodes += 1
                            recording_worker.episode_index = recorded_episodes
                        else:
                            log_say("Dataset is empty, re-record episode", cfg.play_sounds)
                            events.reset()
                            continue

                    if recorded_episodes < cfg.num_episodes and not last_info.get(
                        TeleopEvents.STOP_RECORDING, False
                    ):
                        with recording_worker.status_lock:
                            recording_worker.phase = "resetting"
                            recording_worker.phase_total_s = float(cfg.reset_time_s or 0)
                            recording_worker.phase_start_t = time.perf_counter()
                        last_info = shared_record_loop(
                            env=env,
                            fps=cfg.fps,
                            action_dim=env_cfg.action_dim,
                            action_processor=action_processor,
                            env_processor=env_processor,
                            dataset=None,
                            policy=None,
                            preprocessor=None,
                            postprocessor=None,
                            control_time_s=cfg.reset_time_s,
                            single_task=cfg.single_task,
                            robot_type=env_cfg.type,
                            display_data=cfg.display_data,
                            device="cpu",
                            use_amp=False,
                            interactive=False,
                        )
                        events.reset()

                if last_info.get(TeleopEvents.STOP_RECORDING, False):
                    log_say("Stop recording", cfg.play_sounds)

            if not demo_mode and cfg.push_to_hub:
                with recording_worker.status_lock:
                    recording_worker.phase = "pushing"
                    recording_worker.phase_total_s = None
                    recording_worker.phase_start_t = time.perf_counter()
                dataset.push_to_hub(tags=cfg.tags, private=cfg.private)

        except Exception as exc:
            logger.error("Recording worker error: %s", exc, exc_info=True)
            sio = shared.get_socketio() if shared else None
            if sio:
                try:
                    coro = sio.emit("recording_error", {"error": str(exc)})
                    _schedule_coro(coro)
                except Exception:
                    pass
        finally:
            if hasattr(dataset, "close"):
                try:
                    dataset.close()
                except Exception:
                    logger.debug("Dataset close failed", exc_info=True)
            if hasattr(env, "close") and env is not None:
                try:
                    env.close()
                except Exception:
                    logger.debug("Env close failed", exc_info=True)
            
            # Return borrowed cameras to robot module
            if borrowed_cameras:
                try:
                    from . import robot as robot_module
                    robot_module.return_cameras("recording_worker")
                    logger.info("Cameras returned to robot module from recording")
                except Exception as e:
                    logger.warning(f"Could not return cameras to robot module: {e}")
            
            with recording_worker.status_lock:
                recording_worker.active = False
                recording_worker.phase = "idle"
                recording_worker.phase_start_t = None
                recording_worker.phase_total_s = None
            recording_worker.env = None
            recording_worker.env_processor = None
            recording_worker.action_processor = None
            recording_worker.mapping = None
            recording_worker.dataset = None
            recording_worker.existing_dataset_episodes = None
            events.set_flag("stop_recording", False)

            sio = shared.get_socketio() if shared else None
            if sio:
                try:
                    coro = sio.emit("recording_status", recording_worker.snapshot())
                    _schedule_coro(coro)
                except Exception:
                    pass

    thread = threading.Thread(target=_worker, name="RecordingWorker", daemon=True)
    recording_worker.thread = thread
    thread.start()


def stop_recording_via_api():
    if not recording_worker.active:
        return
    if recording_worker.events:
        recording_worker.events.set_flag("stop_recording", True)
        recording_worker.events.set_flag("exit_early", True)


## removed duplicate command_recording (see consolidated version below)


def _schedule_coro(coro):
    """Utility to schedule coroutine from threads."""
    try:
        loop = recording_worker.loop or asyncio.get_event_loop()
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, loop)
    except RuntimeError:
        pass


def _preflight_cameras(robot, timeout_s: int = 6):
    """Try to start camera streams and receive first frame within timeout.

    Raises with a clear error if frames don't arrive, to avoid cryptic thread errors later.
    """
    cams = getattr(robot, "cameras", {}) or {}
    if not cams:
        logger.warning("No cameras found on robot; proceeding without visual data.")
        return

    start = time.perf_counter()
    for name, cam in cams.items():
        try:
            # Ensure camera connected and fps set
            if not getattr(cam, "is_connected", False):
                cam.connect()
            # Kick off background reader and wait for first frame
            # Note: async_read() itself blocks until first frame or raises after ~1s*fps
            cam.async_read()
        except Exception as e:
            raise RuntimeError(f"Camera '{name}' failed to start: {e}")

    # All cams kicked; do a brief wait loop to ensure at least one frame arrived
    while time.perf_counter() - start < timeout_s:
        ready = True
        for name, cam in cams.items():
            if getattr(cam, "color_image", None) is None:
                ready = False
                break
        if ready:
            return
        time.sleep(0.1)

    # If here, at least one cam never produced a frame
    missing = [name for name, cam in cams.items() if getattr(cam, "color_image", None) is None]
    raise RuntimeError(f"No frames received within {timeout_s}s from cameras: {missing}")


async def _status_emitter_task(interval: float = 0.5):
    """Background task emitting status periodically."""
    sio = shared.get_socketio() if shared else None
    if not sio:
        logger.warning("Socket.IO not available for status emitter")
        return
    while True:
        try:
            if recording_worker.active:
                await sio.emit("recording_status", recording_worker.snapshot())
        except Exception:  # pragma: no cover
            pass
        await asyncio.sleep(interval)


def init_recording_worker(loop: asyncio.AbstractEventLoop):
    recording_worker.loop = loop
    # Start background status task once
    loop.create_task(_status_emitter_task())


def stop_recording_via_api():
    """Stop recording via API call"""
    if recording_worker.active and recording_worker.events:
        recording_worker.events.set_flag("stop_recording", True)
        logger.info("Recording stop requested via API")


def command_recording(action: str):
    """Handle recording commands via API"""
    if not recording_worker.active or not recording_worker.events:
        logger.warning(f"Recording command '{action}' ignored - no active recording")
        return
    
    if action == "exit_early":
        recording_worker.events.set_flag("exit_early", True)
        logger.info("Exit early command received")
    elif action == "rerecord_episode":
        recording_worker.events.set_flag("rerecord_episode", True)
        recording_worker.events.set_flag("exit_early", True)
        logger.info("Rerecord episode command received")
    elif action == "skip_episode":
        recording_worker.events.set_flag("exit_early", True)
        logger.info("Skip episode command received")
    elif action == "stop":
        recording_worker.events.set_flag("stop_recording", True)
        recording_worker.events.set_flag("exit_early", True)
        logger.info("Stop recording command received (exit_early also set)")
    else:
        logger.warning(f"Unknown recording command: {action}")


## removed duplicate helper and emitter definitions (see earlier implementations)


# Socket.IO event handler registration helpers
def register_socketio_handlers(sio):
    logger.info("Registering Socket.IO handlers for recording worker")
    @sio.event
    async def start_recording(sid, data):  # type: ignore
        try:
            logger.info(f"start_recording event received from {sid} with payload: {data}")
            start_recording_via_api(data or {})
            # Emit in two steps to pinpoint any serialization issues
            try:
                await sio.emit("recording_status", recording_worker.snapshot(), room=sid)
            except Exception as se:
                logger.error("start_recording status emit failed", exc_info=True)
                await sio.emit("recording_error", {"error": f"status_emit_failed: {se}"}, room=sid)
                return
            try:
                await sio.emit("recording_started", {"ok": True}, room=sid)
            except Exception as se:
                logger.error("start_recording started emit failed", exc_info=True)
                await sio.emit("recording_error", {"error": f"started_emit_failed: {se}"}, room=sid)
        except Exception as e:
            logger.error("start_recording error", exc_info=True)
            await sio.emit("recording_error", {"error": str(e)}, room=sid)

    @sio.event
    async def stop_recording(sid, data):  # type: ignore
        stop_recording_via_api()
        await sio.emit("recording_status", recording_worker.snapshot(), room=sid)

    @sio.event
    async def recording_command(sid, data):  # type: ignore
        try:
            action = (data or {}).get("action")
            logger.info(f"recording_command event '{action}' from {sid}")
            command_recording(action)
            await sio.emit("recording_status", recording_worker.snapshot())
        except Exception as e:
            logger.error(f"recording_command error: {e}")
            await sio.emit("recording_error", {"error": str(e)})
