# Changelog

## [0.4.0]

### Added
- `MotionRetargeter(enable_procedural_transforms=...)`, threaded through
  `FitContext` to `SOMALayer`. py-soma-x defaults to its expanded twist-joint
  rig, whose procedural transform definition the public `nvidia/soma-x`
  snapshot does not ship — so the default fit was unreachable for anyone using
  stock assets, with no way to opt out (`fit_kwargs` reach
  `PoseInversion.fit`, not the layer). Passing `False` fits against the public
  78-joint rig. Both modes expose the same 78 public joints, so `joint_names`
  and the returned arrays are unchanged in shape and meaning; the fitted
  rotations differ, because procedural mode refines the skin the fit is matched
  against. The kwarg is appended to the signature, so positional callers are
  unaffected.
- With procedural transforms on, the missing definition is now reported before
  the layer is built, naming the file, the data root, and the kwarg that opts
  out, instead of surfacing part-way through `SOMALayer.__init__`.

### Changed
- The `[fit]` extra now requires `py-soma-x>=0.2` (was `>=0.1.0`). 0.1.0 has no
  procedural-transform rig and its `SOMALayer` does not accept
  `enable_procedural_transforms`.

## [0.3.0]

### Added
- Multi-robot retargeting targets. A robot is defined by a
  `configs/<robot>/soma_to_<robot>_retargeter_config.json` bundle; the new
  `robot_model` config section declares how its Newton model is loaded
  (`type: mjcf|urdf`, optional `newton_asset`/`relpath` for download-backed
  models, `floating` for URDF bases). `available_target_robots()` lists the
  bundled robots.
- `limx_oli` target (LimX Oli / HU_D04, 31-DoF humanoid): full config bundle
  (retargeter, scaler, feet stabilizer) mapping the SOMA rig onto the Oli URDF.
  The URDF itself is not bundled — pass it via the new
  `MotionRetargeter(robot_model_path=...)` kwarg (also threaded through
  `NewtonPipeline` / `FeetStabilizer`).
- `soma_retargeter.pipelines.robot_model` — shared robot-model loading
  (spec validation, path resolution, mjcf/urdf dispatch) and model-derived
  DOF naming, used by the pipeline, the feet stabilizer, and the CLI app.

### Changed
- Output `joint_names` are now derived from the loaded robot model's joint
  labels (`joint_q` order) instead of a hand-authored list; for `unitree_g1`
  the derived names are asserted equal to the legacy
  `unitree_g1_joint_names()` order, and outputs are unchanged (verified
  bit-identical). `MotionRetargeter.joint_names` is now a lazy property
  (first access builds the IK pipeline).
- The `TargetType` enum is gone; target robots are validated strings.
  `get_target_type_from_str` / `get_target_str_from_type` keep working with
  string semantics and now list the available robots on error.
- G1's retargeter config renamed to the filename convention:
  `soma_to_unitree_g1_retargeter_config.json` (content-identical apart from
  the new `robot_model` section).
- Config errors fail fast with actionable messages: unknown robots list the
  available ones, bad `ik_map` / feet-effector / smoothing-mask body names
  list the model's bodies, and robots without a bundled model name the
  `robot_model_path=` kwarg at construction time.
- `MotionRetargeter.__init__` gained `robot_model_path` (after `robot`);
  callers passing the chunk options positionally must update.

## [0.2.0]

### Added
- `soma_retargeter.retargeting` — a high-level, in-process retargeting API.
  `MotionRetargeter` retargets SMPL / SMPL-H / SMPL-X (via py-soma-x fitting)
  and native SOMA motion (arrays, `save_soma_npz` files, or SOMA-skeleton BVH)
  to Unitree G1 qpos, with per-clip `retarget*` methods and a batched
  `retarget_batch`. Heavy state is built once and reused, memory stays flat
  over unlimited clips, and there is no temp-file / subprocess round-trip.
- `[fit]` optional-dependency extra (`py-soma-x`, `smplx`, `torch`, `chumpy`).
  The core BVH→CSV path and the `soma-bvh-to-csv` script keep working without
  it (the fit dependencies load lazily).

### Changed
- Docs and headers aligned with the fork's identity: README retitled to
  `eden-soma-retargeter` (unofficial fork, not affiliated with NVIDIA),
  SECURITY.md now points to the fork maintainers instead of NVIDIA PSIRT,
  fork-authored files carry fork copyright headers, and fork-modified upstream
  files carry Apache-2.0 change notices.

### Fixed
- Host-memory leak in the Newton retargeting pipeline: `FeetStabilizer.solve`,
  `JointLimitClamper.apply`, and `HumanToRobotScaler`'s
  `compute_effectors_from_buffer` / `compute_effectors_from_skeleton` defined
  `@wp.kernel`s inside methods invoked per frame / per clip, so Warp accumulated
  a redefinition on every call. The kernels are hoisted to module scope
  (behaviour-preserving), making RSS flat across a streamed batch.

## [0.1.0] - 2026-03-16

Initial public release.