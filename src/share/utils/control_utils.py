import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence, Tuple

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.utils import env_to_dataset_features
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor.rename_processor import rename_stats

from share.configs.record import RecordConfig
from share.envs.manipulation_primitive.config_manipulation_primitive import ManipulationPrimitiveConfig


def make_step_timing_hooks(
    pipeline_steps: Sequence["ProcessorStep"],
    label: str = "pipeline",
    log_every: int = 1,
    ema_alpha: float = 0.2,
    also_print: bool = False,
) -> Tuple:
    """
    Create before/after hooks that time each step in a DataProcessorPipeline.

    Args:
        pipeline_steps: the pipeline steps whose steps we are timing.
        label: a short label to identify this pipeline in logs (e.g., "env", "action").
        log_every: emit a summary every N pipeline passes.
        ema_alpha: smoothing factor for EMA timings (0..1].
        also_print: if True, print the summary in addition to logging.

    Returns:
        before_hook, after_hook callables suitable for pipeline.before_step_hooks / after_step_hooks.
    """
    step_names: Sequence[str] = [type(s).__name__ for s in pipeline_steps]
    n_steps = len(step_names)

    # Per-step timing state
    t_start = [0.0] * n_steps              # last start time per step
    last_ms = [0.0] * n_steps              # last measured dt per step (ms)
    ema_ms  = [0.0] * n_steps              # EMA per step (ms)

    # Per-pass timing
    pass_idx = 0
    pass_t0 = 0.0

    def _emit():
        # Compose a compact, single-line breakdown
        parts = [f"{step_names[i]}={last_ms[i]:.2f}ms(ema:{ema_ms[i]:.2f})"
                 for i in range(n_steps)]
        total_last = sum(last_ms)
        total_ema  = sum(ema_ms)
        msg = f"[{label}] total={total_last:.2f}ms(ema:{total_ema:.2f}) | " + ", ".join(parts)
        logging.info(msg)
        if also_print:
            print(msg)

    def before_hook(idx: int, _transition: "EnvTransition") -> None:
        nonlocal pass_t0
        # If first step, mark pipeline-pass start
        if idx == 0:
            pass_t0 = time.perf_counter()
        t_start[idx] = time.perf_counter()

    def after_hook(idx: int, _transition: "EnvTransition") -> None:
        nonlocal pass_idx
        dt_ms = (time.perf_counter() - t_start[idx]) * 1000.0
        last_ms[idx] = dt_ms
        # Update EMA
        ema_ms[idx] = dt_ms if ema_ms[idx] == 0.0 else (1.0 - ema_alpha) * ema_ms[idx] + ema_alpha * dt_ms

        # If last step, bump pass counter and (maybe) emit
        if idx == n_steps - 1:
            pass_idx += 1
            if log_every > 0 and (pass_idx % log_every == 0):
                # Optionally include the pipeline wall time (may differ slightly from sum of steps)
                pipe_ms = (time.perf_counter() - pass_t0) * 1000.0
                # Replace total with measured wall time if you prefer:
                # msg can include pipe_ms too; here we just keep it implicit to keep line short.
                _emit()

    return [before_hook], [after_hook]


class MPNetStepCounter:
    def __init__(self, primitives: dict[str, ManipulationPrimitiveConfig]):
        # initialize per-primitive step budgets and counters
        self._budget: dict[str, int] = {}
        self._count: dict[str, int] = {}
        self._last_finish_count: dict[str, int] = {}
        for name, p in primitives.items():
            self._last_finish_count[name] = 0
            self._count[name] = 0

    def __getitem__(self, item):
        return self._count[item]

    def increment(self, name: str, n: int = 1):
        """Call this every time the given primitive takes n interaction steps."""
        if name in self._count:
            self._count[name] += n

    def finish_episode(self, name: str):
        """True if this primitive is non-adaptive or has reached its online_steps."""
        if name in self._count:
            self._last_finish_count[name] = self._count[name]

    def episode_length(self, name: str) -> int:
        return self._count.get(name, 0) - self._last_finish_count.get(name, 0)

    @property
    def global_step(self):
        return sum(self._count.values())


def make_policies_and_datasets(cfg: RecordConfig):
    datasets = {}
    policies = {}
    preprocessors = {}
    postprocessors = {}
    for name, p in cfg.env.primitives.items():
        if p.is_adaptive and name != cfg.env.reset_primitive:

            # 1) dataset
            stats = None
            rename_map = {}
            if cfg.dataset is not None:
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
                        features=env_to_dataset_features(p.features),
                        robot_type=cfg.env.type,
                        use_videos=cfg.dataset.video,
                        image_writer_processes=cfg.dataset.num_image_writer_processes,
                        image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * p.num_cameras,
                        batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                        vcodec=cfg.dataset.vcodec,
                    )

                stats = rename_stats(datasets[name].meta.stats, cfg.dataset.rename_map)
                rename_map = cfg.dataset.rename_map

            # 2) policy
            if p.policy is None:
                continue

            if p.policy.pretrained_path is not None:
                policy_path = p.policy.pretrained_path
                p.policy = PreTrainedConfig.from_pretrained(policy_path)
                p.policy = replace(p.policy, **p.policy_overwrites)
                p.policy.pretrained_path = policy_path

            policies[name] = make_policy(cfg=p.policy, env_cfg=p)
            policies[name] = policies[name].eval()

            pre, post = make_pre_post_processors(
                policy_cfg=p.policy,
                pretrained_path=p.policy.pretrained_path,
                dataset_stats=stats,
                preprocessor_overrides={
                    "device_processor": {"device": p.policy.device},
                    "rename_observations_processor": {"rename_map": rename_map},
                },
            )
            preprocessors[name] = pre
            postprocessors[name] = post

    return datasets, policies, preprocessors, postprocessors
