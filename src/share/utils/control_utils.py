import logging
import time
from typing import Sequence, Tuple


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
