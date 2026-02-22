# Share-RL Architecture: Action Processing Pipeline & Task Frames

This document defines the exact mathematical pipeline for handling hybrid human-in-the-loop task frames in `share-rl`. It outlines the responsibilities of the data processors that translate between the "Learning Space" (policy outputs and standardized teleop actions) and the "Control Space" (Task Frame API or Joint Commands).

## 1. Core Philosophy: The Three Mathematical Spaces

Because 3D rotations belong to non-linear manifolds ($SO(3)$), the system strictly decouples representations across the pipeline:

1. **Learning Space (Policy & Standardized Teleop):** Unconstrained continuous vectors ($\mathbb{R}^n$) to ensure smooth gradient flow without singularities or gimbal lock.
2. **Task Frame Space (API):** A human-readable 6-vector where absolute orientations are explicitly defined as **Extrinsic Euler Angles**. This allows masking specific axes independently (e.g., lock X and Y, vary Z).
3. **Joint Space (Hardware):** Absolute joint angles ($q$), used when the underlying robot does not natively support Cartesian task-frame control.

---

## 2. Configuration Validation: Teleoperator Compatibility

Before any processing occurs, `ManipulationPrimitiveConfig.validate` must enforce strict hardware-to-math compatibility using the `check_delta_teleoperator` utility. 

**The Rule:** If *any* learnable axis in the `TaskFrame` has a `control_mode` of `VEL` or `FORCE`, the teleoperator **must** be a delta teleoperator.
* *Why:* Absolute joint teleoperators (e.g., leader arms) only provide absolute positional states. They cannot natively or safely command pure task-frame velocities or wrenches without an external admittance model.
* *Enforcement:* If `is_delta_teleoperator == False` and `control_mode` $\in$ `{VEL, FORCE}`, raise a configuration error.

---

## 3. The Learning Space: Action Dimension & Representation

The `DataProcessorPipeline` must dynamically infer the policy's action dimension from the `TaskFrame` config. For every axis $i \in \{0..5\}$ where `policy_mode[i] != None` (`is_adaptive == True`), the representation is:

### A. Differential Constraints (`VEL`, `FORCE` mode)
* **Dimension added:** $+1$ (Raw scalar, $\mathbb{R}^1$).
* **Mapping:** Bounded via `tanh` and scaled by physical limits.

### B. Absolute Constraints (`POS` mode)
For absolute rotation targets, we project them to continuous manifolds:
* **3 Axes Learnable ($SO(3)$):** $+6$ (6D Continuous Representation, $\mathbb{R}^6$).
* **2 Axes Learnable ($S^2$ Pointing Task):** $+3$ (3D Vector, $\mathbb{R}^3$).
* **1 Axis Learnable ($S^1$ Unit Circle):** $+2$ (2D Vector, $\mathbb{R}^2$).

---

## 4. Pipeline Step 1: `MatchTeleopToPolicyActionProcessorStep`

**Goal:** Transform raw teleoperation inputs into the exact "Learning Space" representation expected by the policy. This guarantees that HIL RL datasets contain semantically equivalent actions regardless of the human input device.

The processor must branch its logic based on the teleoperator type and the target `ControlSpace`:

### Type A: Delta Teleoperators (SpaceMouse, Keyboard, Gamepad)
These devices natively output Cartesian velocities / deltas.
* **To `VEL` / `FORCE` / `RELATIVE POS`:** Map the deltas directly (with appropriate scaling).
* **To `ABSOLUTE POS`:** Numerically integrate the delta against a virtual setpoint to generate an absolute Cartesian pose, then apply the manifold projections (6D/3D/2D representations).
* **To `ControlSpace.JOINT`:** Require **Inverse Kinematics (IK)** to translate Cartesian deltas into Joint angle deltas.


### Type B: Absolute Joint Teleoperators (Leader Arms)
These devices natively output absolute joint angles ($q_{leader}$). *Note: Guaranteed by validation to only target `POS` control modes.*
* **If `TaskFrame.space == ControlSpace.TASK`:** Require **Forward Kinematics (FK)** to compute the absolute Cartesian pose of the leader arm.
  * For `ABSOLUTE POS`: Apply the manifold projections (e.g., extract the 6D rotation representation from the FK rotation matrix).
  * For `RELATIVE POS`: Differentiate the FK Cartesian pose against the previous step to compute a delta.
* **If `TaskFrame.space == ControlSpace.JOINT`:** Trivial 1:1 mapping for absolute joints, or simple differentiation for relative joint deltas.

---

## 5. Pipeline Step 2: `InterventionActionProcessorStep`

**Goal:** Convert the active Learning Space action (either Policy or standard Teleop) into the **Task Frame Space** (partial), and then **scatter** it into the full 6-DoF command.

**Execution Flow:**
1. **Slice & Project:** Slice the unconstrained vector based on the rules in Section 3. Apply projections (Gram-Schmidt orthogonalization, $L_2$ normalization, `atan2`) to convert them into Extrinsic Euler angles or scaled velocity/force scalars.
2. **Scatter:** The resulting projected action only contains values for `learnable_axis_indices`. The processor must "scatter" these values into a full 6-vector `target`.
3. **Merge:** For non-learnable axes (`policy_mode == None`), inject the static values from `TaskFrame.target`.

The output of this step is a populated dictionary/tensor representing the full, mixed-mode 6-DoF Task Frame command.

---

## 6. Pipeline Step 3: `ToJointActionProcessorStep` (Conditional)

**Goal:** Handle the hardware disparity defined by `is_task_frame_robot`. 

If `is_task_frame_robot == True`:
* The pipeline stops. The 6-DoF Task Frame command is passed directly to the low-level controller, which handles integration, pose limits (`min_target`, `max_target`), and IK natively in its $SO(3)$ control loop.

If `is_task_frame_robot == False`:
* The hardware only accepts joint angles. This processor must perform the heavy lifting:
  1. **Integration:** If `policy_mode == RELATIVE`, numerically integrate the delta commands into absolute Cartesian setpoints using the current observation.
  2. **Pose Limits:** Clamp the absolute Cartesian setpoint against `TaskFrame.min_target` and `TaskFrame.max_target`.
  3. **Inverse Kinematics (IK):** Solve IK (`kinematics_solver`) to convert the bounded Cartesian 6-vector into absolute joint configurations ($q$).
  4. Replace the Task Frame dictionary keys with Joint names.
