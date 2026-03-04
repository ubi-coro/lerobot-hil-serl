# Manipulation Primitive Net — Issue Board

This board translates the current implementation gaps and requested MP-Net capabilities into concrete, sequenced work items.

## General
Leave brief docstrings and keep them updated in all files you touch.

## Milestone Goal
Ship a minimal but production-usable `ManipulationPrimitiveNet` that:
- exposes a clean gym-like interface over chained primitives,
- executes explicit transition logic between primitives,
- supports reset paths from terminal states,
- is fully serializable/config-driven through draccus.

---

## Current Gap Snapshot

| Component | Status | Risk | Why this matters now |
|---|---|---|---|
| `ManipulationPrimitiveNet.step` | Missing | **High** | This is the orchestration entry point. Without it, primitives cannot be chained, transitions cannot trigger, and rollouts cannot progress beyond a single primitive. |
| `RelativeFrameActionProcessor` | Placeholder / no-op | **Medium** | Tasks that rely on relative-to-initial-grasp behavior will silently execute with incorrect semantics. |
| `manipulation_primitive_net` transition logic | Skeleton | **Medium** | Config plumbing exists, but no concrete transition policy means no automated primitive switching or reward shaping at the net level. |

---

## EPIC MPN-A — Runtime Orchestration (Core Env Contract)

### MPN-101: Implement `ManipulationPrimitiveNet.step`
**Priority:** P0  
**Status:** Done  
**Owner:** Unassigned

#### Problem
`step` is currently unimplemented, so the MP-Net cannot execute policy actions through the active primitive and evaluate transitions.

#### Scope
- Track active primitive state (`self._active_primitive`).
- Route action through active primitive env/action pipeline.
- Evaluate transition candidates for the active primitive after each low-level step.
- Apply transition outputs: next primitive, additional reward, terminated/truncated overrides.
- Return gymnasium-compatible `(obs, reward, terminated, truncated, info)`.

#### Acceptance Criteria
- MP-Net can run multi-step rollouts with primitive changes.
- Reward includes base primitive reward + transition reward terms.
- `info` contains transition diagnostics (`from`, `to`, `reason`, transition name/type).

#### Tests
- `test_mp_net_step_executes_active_primitive`
- `test_mp_net_step_switches_primitive_when_transition_fires`
- `test_mp_net_step_applies_transition_reward_and_done_flags`

---

### MPN-102: Implement coherent reset semantics across primitive chains
**Priority:** P0  
**Status:** Done  
**Owner:** Unassigned

#### Problem
Reset behavior is unclear when terminal primitives and reset primitives are introduced.

#### Scope
- Define reset policy in env contract:
  - Standard reset to `start_primitive`.
  - Optional terminal-to-start reset path via explicit reset primitives.
- Clear all per-episode MP-Net state (active primitive, elapsed steps, transition caches).
- Reset underlying primitive env(s) deterministically.

#### Acceptance Criteria
- `reset()` consistently returns the initial observation for configured start behavior.
- Terminal episodes can either end immediately or flow into reset primitives based on config.

#### Tests
- `test_mp_net_reset_starts_from_start_primitive`
- `test_mp_net_reset_path_via_reset_primitives`

---

## EPIC MPN-B — Transition System (Most Underspecified Area)

### MPN-201: Replace bare `MP_Transition` with typed transition dataclasses
**Priority:** P0  
**Status:** Done  
**Owner:** Unassigned

#### Problem
Current `MP_Transition` only exposes a boolean check and cannot express next primitive, reward shaping, or done-flag semantics.

#### Scope
Create a transition contract consistent with:
- `t(obs, info) -> condition_fulfilled, next_primitive, additional_reward, terminated`
- draccus-selectable transition types, e.g.:
  1. Observation-threshold transition (`obs[key] >= threshold` style),
  2. Time-limit transition,
  3. Reward/success-classifier transition.

#### Acceptance Criteria
- Each transition type is a dataclass registered via `ChoiceRegistry`.
- Transition output includes all required fields for MP-Net orchestration.
- Time-limit transitions can set episode-end semantics without forcing `terminated=True` (truncation-style behavior).

#### Tests
- `test_transition_threshold_triggers_and_returns_next_primitive`
- `test_transition_time_limit_sets_truncation_style_flags`
- `test_transition_classifier_adds_sparse_success_reward`

---

### MPN-202: Add transition graph validation
**Priority:** P1  
**Status:** Todo  
**Owner:** Unassigned

#### Problem
Misconfigured transition graphs (dangling primitive names, unreachable terminals, dead-ends) can fail only at runtime.

#### Scope
- Validate all source/target primitive references exist.
- Validate start primitive exists.
- Detect obvious dead-ends unless explicitly marked terminal.
- Validate reset paths if reset primitives are configured.

#### Acceptance Criteria
- Invalid graphs fail at config load with actionable errors.
- Valid graphs pass without runtime surprises.

#### Tests
- `test_mp_net_config_rejects_unknown_primitive_in_transition`
- `test_mp_net_config_rejects_missing_start_primitive`
- `test_mp_net_config_allows_intentional_terminal_dead_end`

---

## EPIC MPN-C — Primitive Lifecycle Semantics

## MP-Net Reset + Done Semantics (API Contract)

This project follows Gymnasium’s strict contract: if `step()` returns `terminated=True` or `truncated=True`, the caller MUST call `reset()` before calling `step()` again. We therefore separate **net-level episode termination** (Gym done flags) from **primitive-level segment boundaries** (reported via `info`).

Terminology:
- “MP-Net episode”: the Gymnasium episode of the `ManipulationPrimitiveNet` environment.
- “Primitive segment”: the contiguous slice of steps during which one learnable primitive is active. Segments are the unit of storage for per-primitive datasets/policies.

### Step signature and invariants

`ManipulationPrimitiveNet.step(action) -> (obs, reward, terminated, truncated, info)`

Hard invariants:
1) `terminated/truncated` signal MP-Net episode end only (Gym semantics).
   - If either is True, stepping again without `reset()` is invalid.
2) Primitive transitions and primitive “segment done” MUST NOT set Gym done flags unless they intentionally end the MP-Net episode.
3) Primitive segment boundaries are communicated via `info` only.

### Transition evaluation contract

Each configured transition is evaluated as:

`t.evaluate(obs, info) -> TransitionOutcome`

Where `TransitionOutcome` may contain:
- `condition_fulfilled: bool`
- `next_primitive: str | None`
- `additional_reward: float`
- `terminated: bool`  (net-level termination request)
- `truncated: bool`   (net-level truncation request)
- `reason: str | None`

Important:
- Transition `terminated/truncated` are **net-level** flags and are OR’ed into the MP-Net’s Gym done flags.
- Transitions used only for routing between learnable primitives should keep these False.

### Primitive env done handling

The active primitive env’s own `step()` returns `(obs, reward, prim_terminated, prim_truncated, prim_info)`.

Rules:
- `prim_terminated/prim_truncated` do NOT directly imply MP-Net termination.
- They are surfaced as *primitive-level* flags in `info`, and may optionally be mapped to MP-Net done flags by explicit transition logic (preferred) or by a deliberate policy.

Required behavior in MP-Net:
- Always expose primitive done flags in `info`:
  - `info["primitive_done"] = prim_terminated or prim_truncated`
  - `info["primitive_terminated"] = bool(prim_terminated)`
  - `info["primitive_truncated"] = bool(prim_truncated)`
  - `info["primitive_done_reason"]` (if available / derivable)
- The MP-Net `terminated/truncated` must only become True when the MP-Net episode ends.

### Primitive segment boundaries (data collection semantics)

A primitive segment ends when the MP-Net switches from one learnable primitive to another learnable primitive, or when a learnable primitive decides to “cut” the segment (e.g. per-primitive time limit) while continuing the MP-Net episode.

Segment boundary signals live in `info`:
- `info["segment_done"] : bool`
- `info["segment_from"] : str` (previous active primitive)
- `info["segment_to"] : str` (new active primitive; may equal from if segment cut without switching)
- `info["segment_reason"] : str` (e.g., `"transition_fired"`, `"primitive_time_limit"`, `"operator_abort"`)
- Optional:
  - `info["segment_additional_reward"]` (if you want to attribute shaping to segment boundary)

Collector contract:
- If `info["segment_done"]` is True, flush/save the segment buffer for `segment_from` under that primitive’s dataset/policy namespace.
- If `terminated or truncated` is True, flush/save the final segment (if not already flushed) and then call `reset()`.

### Terminal primitives (net-level episode end)

A primitive can be marked `is_terminal_primitive=True`. Terminal semantics must be explicit and Gym-correct:

- When the MP-Net is in a terminal primitive, the MP-Net episode is expected to end via either:
  1) an explicit transition outcome setting `terminated=True` or `truncated=True`, or
  2) a terminal-policy fallback rule (only if documented in code) that sets `terminated=True` when in a terminal primitive and no transition fires.

Do NOT rely on external code to “keep stepping after done”. If the task should continue into reset primitives, that should be expressed as transitions without ending the MP-Net episode, or as a `reset()` path (see below).

### Reset semantics

`ManipulationPrimitiveNet.reset()` establishes a new MP-Net episode and returns `(obs, info)`.

Rules:
- `reset()` must clear MP-Net episode state (active primitive, episode counters, per-episode caches).
- If configured, reset may start in a reset primitive and internally route back to `start_primitive` using the reset transition path, but this routing must occur inside `reset()` (or return once the start primitive is active). External callers should not need to call `step()` to complete reset.

### Required `info` fields (minimum)

Every MP-Net `step()` must include:
- `info["active_primitive"] : str`
- `info["transition"] : dict` with at least:
  - `from`, `to`, `reason`, `transition_name`, `transition_type`
- Primitive done flags:
  - `primitive_done`, `primitive_terminated`, `primitive_truncated`
- Segment boundary (when applicable):
  - `segment_done` plus `segment_from/segment_to/segment_reason` when `segment_done=True`

This contract is mandatory for any future refactor of `ManipulationPrimitiveNet.step`, transition types, and dataset/recording scripts.

### MPN-301: Formalize terminal and reset primitive roles
**Priority:** P1  
**Status:** Partially Done  
**Owner:** Unassigned

#### Problem
Terminal primitives and reset primitives are conceptually defined but not represented in config and runtime policy.

#### Scope
- Add primitive metadata:
  - `is_terminal: bool`
  - `is_reset: bool`
- Decide and document whether terminal/reset outgoing transitions are:
  1. part of standard transition list, or
  2. represented as dedicated optional transition lists.
- Implement one consistent model and enforce it.

#### Recommended design decision
Keep a **single transition list** for simplicity and serializability. Differentiate behavior by primitive metadata + transition type. This avoids split logic and supports static editing tools.

#### Acceptance Criteria
- Terminal and reset behavior is explicit in config and runtime.
- Transition execution rules are documented and tested.

#### Tests
- `test_terminal_primitive_ends_or_routes_based_on_transition`
- `test_reset_primitive_routes_back_to_start_domain`

---

### MPN-302: Enforce MP-Net step/reset API contract in runtime info
**Priority:** P0  
**Status:** Partially Done  
**Owner:** Unassigned

#### Progress update
- Gym-level done semantics now require reset-before-next-step once the MP-Net episode ends.
- Primitive-level done flags are surfaced in `info` without ending the MP-Net episode by default.
- Segment boundary markers are emitted in `info` for primitive switches (`segment_done`, `segment_from`, `segment_to`, `segment_reason`).

#### Remaining
- Add explicit segment-cut support when segment ends without primitive switch.
- Expand transition diagnostics with stricter validation for missing `next_primitive` on fired transitions.

---

## EPIC MPN-D — Config & Serialization (Draccus-First)

> Backlog note: Serialization-focused work is intentionally deprioritized for now while EPIC MPN-C runtime semantics are finalized.

### MPN-401: Make MP-Net fully serializable with typed primitive dictionaries
**Priority:** P0  
**Status:** Todo  
**Owner:** Unassigned

#### Problem
MP-Net needs static, editable representations with named primitives selected via `.type`, but dictionary-of-choice patterns are not yet established here.

#### Scope
- Ensure `ManipulationPrimitiveNetConfig` supports:
  - `primitives: dict[str, ManipulationPrimitiveConfig]` where each entry can use draccus `.type` selection,
  - typed transition objects in a serializable structure.
- Add round-trip load/save examples for YAML.
- Confirm nested choice-registry fields inside dictionaries are resolved correctly.

#### Acceptance Criteria
- Config can be authored entirely in YAML and round-tripped without loss.
- Primitive and transition types are recoverable after deserialize/serialize cycle.

#### Tests
- `test_mp_net_config_roundtrip_serialization`
- `test_mp_net_config_dict_of_typed_primitives`
- `test_mp_net_config_dict_of_typed_transitions`

---

### MPN-402: Provide runnable scripts with clear abstraction boundaries
**Priority:** P1  
**Status:** Todo  
**Owner:** Unassigned

#### Problem
A high-level interface is desired, but scripts should interact through simple, stable APIs and avoid leaking orchestration complexity.

#### Scope
- Add minimal scripts for:
  - instantiate MP-Net from draccus config,
  - rollout/eval loop,
  - optional data collection.
- Expose small utility methods (e.g., active primitive getter, transition stats snapshot) rather than coupling scripts to internals.

#### Acceptance Criteria
- Scripts run without direct access to private MP-Net internals.
- API is clear enough for policy and dataset tooling integration.

#### Tests
- script smoke test using mocked primitives/transitions.

---

## EPIC MPN-E — Relative Frame Action Support

### MPN-501: Implement `RelativeFrameActionProcessor` behavior
**Priority:** P1  
**Status:** Todo  
**Owner:** Unassigned

#### Problem
Current no-op behavior breaks tasks that require actions relative to an initial grasp/reference pose.

#### Scope
- Capture reference frame at episode/phase start.
- Transform incoming actions into the requested relative frame.
- Define reset behavior for reference frame on primitive transitions (configurable).

#### Acceptance Criteria
- Relative frame tasks produce expected transformed actions.
- Behavior is deterministic across resets/transitions.

#### Tests
- `test_relative_frame_processor_applies_transform_from_initial_reference`
- `test_relative_frame_processor_resets_reference_on_configured_transition`

---

## Suggested Delivery Plan

1. **Phase 1 (must-have):** MPN-101, MPN-102, MPN-201, MPN-401.  
2. **Phase 2 (stability):** MPN-202, MPN-301, MPN-501.  
3. **Phase 3 (ergonomics):** MPN-402 + examples/docs polish.

---

## Open Design Questions (to resolve before implementation lock)

1. Should transition evaluation be first-match, priority-based, or allow multiple rewards before selecting next primitive?  
2. Should additional transition reward be additive only, or support replacement/scaling of primitive reward?  
3. For terminal primitive episodes that hit time limit, should default mapping be `terminated=False, truncated=True`?  
4. Should reset primitives execute in the same episode context, or always begin a fresh episode boundary?

---

## Definition of Done for MP-Net v1

- MP-Net can execute and transition between named primitives at runtime.
- Transition types cover threshold, time-limit, and classifier-driven routing.
- Terminal/reset semantics are explicit, tested, and documented.
- Draccus config is fully serializable for primitive dictionaries and transition objects.
- Basic scripts demonstrate clean usage without internal coupling.
