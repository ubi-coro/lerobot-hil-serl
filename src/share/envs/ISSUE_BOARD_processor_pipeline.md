# Share-RL Env Processor Pipeline — Issue Board

This board tracks the finalization work for the `src/share/envs` processor pipeline before release.

## Milestone Goal
Ship a robust, test-backed action processing pipeline that faithfully implements the architecture in `AGENTS_envs.md` across:
- Learning-space action encoding,
- Task-frame command generation,
- Conditional conversion to joint-space commands.

---

## EPIC A — Configuration and Contract Enforcement

### ENV-101: Enforce teleoperator compatibility for learnable `VEL`/`FORCE` axes
**Priority:** P0  
**Status:** Done  
**Owner:** Unassigned

#### Problem
Current `ManipulationPrimitiveConfig.validate` does not yet enforce the documented constraint that non-delta teleoperators cannot command adaptive `VEL`/`FORCE` task-frame axes.

#### Scope
- In `validate`, iterate over each robot/task-frame pair.
- For each learnable axis (`policy_mode[i] is not None`), if `control_mode[i] in {VEL, FORCE}` and teleop is non-delta, raise a clear `ValueError`.
- Include robot/axis metadata in the error message.

#### Acceptance Criteria
- Invalid config fails fast with descriptive error.
- Valid configs (delta teleop for adaptive `VEL`/`FORCE`) pass.
- Unit tests cover pass/fail combinations.

#### Tests
- `test_validate_rejects_non_delta_teleop_for_adaptive_vel_force`
- `test_validate_allows_delta_teleop_for_adaptive_vel_force`

---

### ENV-102: Complete validation matrix for control-space and hardware compatibility
**Priority:** P0  
**Status:** Done  
**Owner:** Unassigned

#### Problem
Validation has TODO placeholders for key compatibility checks.

#### Scope
- Enforce JOINT-space axes are POS-only when applicable.
- Enforce non-task-frame robots reject unsupported non-POS modes.
- Validate FK/IK requirements are present when teleop/robot modality mismatch requires them.
- Normalize scalar-vs-dict processor config expansion robustly.

#### Acceptance Criteria
- All documented invalid combinations raise explicit errors.
- All expected valid combinations pass.
- Errors mention required remediation (e.g., enable kinematics).

#### Tests
- Parametrized compatibility matrix test over:
  - teleop type (delta vs absolute-joint),
  - target space (TASK vs JOINT),
  - control mode (POS/VEL/FORCE),
  - robot capability (task-frame vs joint-only).

---

## EPIC B — Learning Space Representation and Policy Interface

### ENV-201: Infer policy action dimension from `TaskFrame` manifold rules
**Priority:** P0  
**Status:** Done  
**Owner:** Unassigned

#### Problem
Action feature dimension is still placeholder-level and not fully inferred from adaptive task-frame specification.

#### Scope
- Implement action-dim inference:
  - Differential (`VEL`/`FORCE`) learnable axis: `+1` each.
  - Absolute rotational POS representation:
    - 3 learnable rotational axes -> `+6` (SO(3) 6D rep),
    - 2 -> `+3` (S2 vector),
    - 1 -> `+2` (S1 embedding).
- Integrate with `infer_features` to expose correct policy action shape.

#### Acceptance Criteria
- For representative task-frame configs, inferred action dims match spec.
- Policy features report stable and deterministic shape.

#### Tests
- Unit tests for action-dim calculator across mixed translational/rotational configurations.

---

## EPIC C — Action Pipeline Step Implementations

### ENV-301: Implement `MatchTeleopToPolicyActionProcessorStep` mapping matrix
**Priority:** P0  
**Status:** Todo  
**Owner:** Unassigned

#### Problem
The step is intended but needs complete behavior across teleop modality and target control space.

#### Scope
Implement logic for:
- Delta teleop:
  - direct map for differential targets,
  - integration for absolute POS targets,
  - IK path for JOINT-space mappings where required.
- Absolute joint teleop:
  - FK to task pose,
  - differentiation for relative policy modes,
  - direct pass-through where semantics match.

#### Acceptance Criteria
- Output learning-space action matches policy representation shape and semantics.
- Behavior is deterministic with virtual reference enabled/disabled.

#### Tests
- Unit tests with mocked kinematics + synthetic teleop streams.

---

### ENV-302: Implement `InterventionActionProcessorStep` projection + scatter/merge
**Priority:** P0  
**Status:** Todo  
**Owner:** Unassigned

#### Problem
Need canonical conversion from unconstrained learning-space vector to mixed-mode full 6-DoF task-frame command.

#### Scope
- Slice incoming action by inferred manifold layout.
- Apply projections:
  - normalization / orthogonalization for orientation reps,
  - bounded scaling for differential controls.
- Scatter projected values into learnable indices.
- Merge static `TaskFrame.target` values for non-learnable axes.

#### Acceptance Criteria
- Mixed adaptive/static task-frame target is always fully populated.
- Orientation projections are numerically stable.

#### Tests
- SO(3)/S2/S1 projection tests (shape + normalization + angle extraction sanity).
- Scatter/merge correctness tests.

---

### ENV-303: Implement conditional `ToJointActionProcessorStep`
**Priority:** P0  
**Status:** Todo  
**Owner:** Unassigned

#### Problem
Joint-only robots require integration, clamping, and IK conversion before command dispatch.

#### Scope
When `is_task_frame_robot=False`:
1. Integrate relative task-frame commands to absolute pose targets.
2. Clamp against task-frame limits.
3. Solve IK to joint config.
4. Replace task-frame action keys with robot joint keys.

#### Acceptance Criteria
- Joint-only path outputs valid joint dictionary/tensor shape for downstream robot send.
- Limit enforcement and IK failure handling are explicit.

#### Tests
- Integration/clamping unit tests.
- IK invocation and key remapping tests.

---

## EPIC D — Env Robustness and Wiring

### ENV-401: Complete `ManipulationPrimitive.reset` and clean env contract
**Priority:** P1  
**Status:** Todo  
**Owner:** Unassigned

#### Problem
`reset` is currently unimplemented; env lifecycle is incomplete for rollouts and integration tests.

#### Scope
- Implement minimal `reset` behavior and return `(obs, info)`.
- Ensure camera/robot state interactions are safe.
- Confirm `step` and `reset` satisfy gymnasium expectations.

#### Acceptance Criteria
- Environment can run a smoke rollout without `NotImplemented`/`pass` path.

#### Tests
- `test_env_reset_returns_obs_info`
- `test_env_step_after_reset_smoke`

---

### ENV-402: Normalize naming consistency (`min_pose`/`max_pose` vs docs naming)
**Priority:** P1  
**Status:** Todo  
**Owner:** Unassigned

#### Problem
Documentation references `min_target`/`max_target` while dataclass currently uses `min_pose`/`max_pose`.

#### Scope
- Decide canonical naming and apply consistently in docs + code.
- Add backward-compatible load behavior if needed.

#### Acceptance Criteria
- No ambiguity in config fields and processor usage.

#### Tests
- Serialization/deserialization compatibility test.

---

## EPIC E — Test & Release Readiness

### ENV-501: Build compatibility matrix tests for release confidence
**Priority:** P1  
**Status:** Todo  
**Owner:** Unassigned

#### Scope
- Add parametrized test suite covering modality and control compatibility.
- Ensure CI can run tests without hardware dependencies (mock interfaces).

#### Acceptance Criteria
- Matrix gives high confidence in configuration and pipeline branching logic.

---

### ENV-502: Add one end-to-end pipeline smoke test
**Priority:** P2  
**Status:** Todo  
**Owner:** Unassigned

#### Scope
- Single primitive configuration with mocked teleop and robot.
- Exercise env processor + action processor + one `step` call.

#### Acceptance Criteria
- Demonstrates end-to-end wiring does not regress.

---


## EPIC F — Mocking Infrastructure for Robots, Kinematics, and Teleoperators

### ENV-601: Add reusable mock entities for pipeline modality coverage
**Priority:** P0  
**Status:** Done  
**Owner:** Unassigned

#### Problem
Pipeline tickets depend on deterministic, hardware-free fixtures that model both robot capabilities and teleoperator modalities.

#### Scope
- Add reusable mock entities that represent:
  - task-frame-capable robot,
  - joint-only robot,
  - delta teleoperator,
  - absolute-joint teleoperator,
  - deterministic FK/IK mock solver.
- Keep the mocks lightweight and independent from hardware backends.

#### Acceptance Criteria
- Mocks can be instantiated in unit tests with no hardware dependencies.
- Teleoperator and robot modality helpers can distinguish each mock correctly.

#### Tests
- `test_mock_robots_cover_task_frame_and_joint_only_modalities`
- `test_mock_teleoperators_cover_delta_and_absolute_joint_modalities`

---

### ENV-602: Validate deterministic mock kinematics behavior for FK/IK-dependent tickets
**Priority:** P0  
**Status:** Done  
**Owner:** Unassigned

#### Problem
Upcoming processor-step tests require a deterministic and invertible kinematics stub to validate FK/IK branching logic.

#### Scope
- Provide deterministic FK mapping from joint dictionary to 6D task-frame pose.
- Provide deterministic IK mapping that round-trips back to the original joint dictionary.
- Add explicit tests for repeatability and numerical sanity.

#### Acceptance Criteria
- FK output is deterministic and stable for fixed joint input.
- IK(FK(q)) returns the original joint targets for test fixtures.

#### Tests
- `test_mock_kinematics_solver_is_deterministic_for_fk_and_ik`

---

## Suggested execution order
1. ENV-101  
2. ENV-102  
3. ENV-201  
4. ENV-302  
5. ENV-301  
6. ENV-303  
7. ENV-401  
8. ENV-402  
9. ENV-501  
10. ENV-502
11. ENV-601
12. ENV-602

## Definition of Done (overall)
- All P0 tickets merged.
- Unit tests green for validation + manifold projections + conversion steps.
- One end-to-end smoke test green.
- Documentation and config naming aligned.
