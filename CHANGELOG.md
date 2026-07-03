# Changelog

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