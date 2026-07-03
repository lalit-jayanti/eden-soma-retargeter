# SPDX-FileCopyrightText: Copyright (c) 2026 The eden-soma-retargeter authors (unofficial fork of NVIDIA soma-retargeter)
# SPDX-License-Identifier: Apache-2.0

"""Array-level bridges between motion data and the core retargeting pipeline.

The reference app drives the pipeline from ``.bvh`` files. These helpers build
the same ``AnimationBuffer`` inputs directly from rotation arrays (skipping the
BVH text round-trip and its Euler %.6f quantization) and read the pipeline's
``CSVAnimationBuffer`` output directly (skipping the cm/deg CSV round-trip).

Conventions match the reference BVH path exactly: buffers are Y-up (BVH world),
and the pipeline lifts them to Z-up via the Mujoco ``qx(+90)`` root offset.
"""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

_TEMPLATE_CACHE: dict = {}


def template_bvh_path() -> Path:
    """Path of the SOMA template skeleton BVH bundled with this package."""
    import soma_retargeter

    p = Path(soma_retargeter.__file__).parent / "configs" / "soma" / "soma_zero_frame0.bvh"
    if not p.exists():
        raise FileNotFoundError(f"SOMA template BVH not found at {p}")
    return p


def load_template_skeleton(path: str | Path | None = None):
    """Load (and cache) the SOMA template ``Skeleton``."""
    from soma_retargeter.assets.bvh import BVHImporter

    p = Path(path) if path is not None else template_bvh_path()
    key = str(p.resolve())
    skeleton = _TEMPLATE_CACHE.get(key)
    if skeleton is None:
        skeleton, _ = BVHImporter().create_skeleton(str(p))
        _TEMPLATE_CACHE[key] = skeleton
    return skeleton


def mujoco_root_offset():
    """The qx(+90 deg) root transform the reference app applies to Y-up BVH buffers."""
    import warp as wp

    return wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(1, 0, 0), wp.radians(90.0)))


# qx(-90 deg) takes the Z-up SMPL / SOMA world to the Y-up BVH world; the
# pipeline's Mujoco root offset (qx(+90)) lifts it back to Z-up at solve time.
_QX_NEG90 = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=np.float32,
)


def zup_to_yup(rotations: np.ndarray, root_trans: np.ndarray, hips_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Apply the Z-up to Y-up world change of basis to Hips rotation and root translation.

    Inner joints store local-to-parent rotations and pass through unchanged
    under a world rotation; only the world-anchored Hips rotation and the root
    translation are touched.

    Args:
        rotations: (T, J, 3, 3) absolute local rotation matrices, Z-up.
        root_trans: (T, 3) Hips translation in metres, Z-up.
        hips_idx: Column of the Hips joint in ``rotations``.
    """
    qx_neg90 = _QX_NEG90.astype(rotations.dtype)
    rot = rotations.copy()
    rot[:, hips_idx] = qx_neg90 @ rot[:, hips_idx]
    trans = root_trans @ qx_neg90.T
    return rot, trans


def buffer_from_local_rotations(
    skeleton,
    rotations: np.ndarray,
    root_trans_m: np.ndarray,
    fps: float,
    source_joint_names: list[str],
):
    """Build an ``AnimationBuffer`` on ``skeleton`` from per-joint local rotations.

    Produces exactly what ``load_bvh`` would return for a motion written on the
    template hierarchy: joint positions from the skeleton bind offsets, the
    Hips position carrying the root translation, and per-frame joint rotations
    mapped by name (joints absent from ``source_joint_names`` keep their bind
    rotation).

    Args:
        skeleton: Target ``Skeleton`` (normally :func:`load_template_skeleton`).
        rotations: (T, J_src, 3, 3) local rotation matrices, Y-up BVH world.
        root_trans_m: (T, 3) Hips translation in metres, Y-up BVH world.
        fps: Sample rate stamped on the buffer.
        source_joint_names: Names labelling the ``J_src`` columns of ``rotations``.
    """
    from soma_retargeter.animation.animation_buffer import AnimationBuffer

    T = rotations.shape[0]
    J = skeleton.num_joints
    local = np.zeros((T, J, 7), dtype=np.float32)
    local[:] = skeleton.reference_local_transforms[None]

    name_to_src = {n: i for i, n in enumerate(source_joint_names)}
    # The virtual Root may be absent (SOMA npz files store J=77 without it);
    # its bind rotation is identity, which is what the pipeline expects.
    missing = [n for n in skeleton.joint_names if n not in name_to_src and n != "Root"]
    if missing:
        raise ValueError(f"Skeleton joints missing from source rotations: {missing}")

    for j, name in enumerate(skeleton.joint_names):
        src = name_to_src.get(name)
        if src is None:
            continue
        local[:, j, 3:7] = Rotation.from_matrix(rotations[:, src]).as_quat().astype(np.float32)

    root_idx = 0  # template root ("Root") carries no translation
    local[:, root_idx, 0:3] = 0.0
    hips_idx = skeleton.joint_index("Hips")
    if hips_idx == -1:
        raise ValueError("Skeleton has no 'Hips' joint")
    local[:, hips_idx, 0:3] = np.asarray(root_trans_m, dtype=np.float32)

    return AnimationBuffer(skeleton, T, fps, local)


def resample_buffer(buffer, target_fps: float):
    """Resample an ``AnimationBuffer`` to ``target_fps`` via its own interpolator."""
    from soma_retargeter.animation.animation_buffer import AnimationBuffer

    if buffer.sample_rate == target_fps:
        return buffer
    T_out = int((buffer.num_frames - 1) * target_fps / buffer.sample_rate) + 1
    if T_out < 2:
        raise ValueError(
            f"resampled sequence would have {T_out} frame(s); "
            f"input was {buffer.num_frames} @ {buffer.sample_rate} fps -> target {target_fps} fps")
    local = np.stack([np.asarray(buffer.sample(i / target_fps)) for i in range(T_out)])
    return AnimationBuffer(buffer.skeleton, T_out, target_fps, local.astype(np.float32))


def csv_buffer_to_arrays(buffer) -> dict:
    """Read a ``CSVAnimationBuffer``'s raw joint state into numpy arrays.

    Buffer rows are raw Newton free-joint states
    ``[root_pos(3) m, root_quat xyzw(4), dof(...) rad]`` in the Z-up world.

    Returns:
        dict with ``root_pos`` (T, 3) float32 metres, ``root_quat`` (T, 4)
        float32 **wxyz**, ``dof_pos`` (T, N) float32 radians, and ``fps``.
    """
    data = np.stack(buffer.data).astype(np.float32)
    root_pos = data[:, 0:3]
    root_quat = data[:, [6, 3, 4, 5]]  # xyzw -> wxyz
    dof_pos = data[:, 7:]
    return {"root_pos": root_pos, "root_quat": root_quat, "dof_pos": dof_pos, "fps": float(buffer.sample_rate)}


def unitree_g1_joint_names() -> list[str]:
    """The 29 G1 actuated-joint names in pipeline output column order.

    Sourced from ``UnitreeG129DOF_CSVConfig.csv_header`` (the '_dof' suffix is
    stripped to match URDF joint names) so the order always tracks what the
    installed core emits.
    """
    from soma_retargeter.assets.csv import UnitreeG129DOF_CSVConfig

    names = []
    for col in UnitreeG129DOF_CSVConfig.csv_header[7:]:
        if not col.endswith("_dof"):
            raise RuntimeError(
                f"Unexpected CSV joint column {col!r}: expected '_dof' suffix. "
                "The core csv_header layout may have changed.")
        names.append(col[: -len("_dof")])
    return names


def resample_rotvec_tracks(
    poses: np.ndarray,
    trans: np.ndarray,
    source_fps: float,
    target_fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample axis-angle joint tracks (per-joint SLERP) and translation (linear).

    Args:
        poses: (T_in, J, 3) or (T_in, 3*J) axis-angle.
        trans: (T_in, 3) metres.

    Returns:
        (poses, trans) at ``target_fps`` with the same poses rank as the input;
        ``T_out = int((T_in - 1) * target_fps / source_fps) + 1``. Exact integer
        downsample ratios use a stride fast path.
    """
    T_in = poses.shape[0]
    if T_in < 2:
        raise ValueError(f"input has {T_in} frames; need at least 2 to resample")

    if source_fps == target_fps:
        return poses, trans

    T_out = int((T_in - 1) / source_fps * target_fps) + 1
    if T_out < 2:
        raise ValueError(
            f"resampled sequence would have {T_out} frame(s); "
            f"input was {T_in} @ {source_fps} fps -> target {target_fps} fps")

    downsample_rate = int(source_fps // target_fps)
    if downsample_rate >= 1 and downsample_rate * target_fps == source_fps:
        return poses[::downsample_rate].copy(), trans[::downsample_rate].copy()

    old_time = np.arange(T_in) / source_fps
    new_time = np.arange(T_out) / target_fps
    # Guard against a ULP overshoot at the last sample (source/target divisors
    # differ) so scipy Slerp never rejects an out-of-range interpolation time.
    new_time = np.minimum(new_time, old_time[-1])

    trans_out = np.empty((T_out, 3), dtype=np.float32)
    for i in range(3):
        trans_out[:, i] = np.interp(x=new_time, xp=old_time, fp=trans[:, i])

    flat = poses.ndim == 2
    poses_3 = poses.reshape(T_in, -1, 3)
    n_joints = poses_3.shape[1]
    poses_out = np.empty((T_out, n_joints, 3), dtype=np.float32)
    for j in range(n_joints):
        slerp = Slerp(times=old_time, rotations=Rotation.from_rotvec(poses_3[:, j, :]))
        poses_out[:, j, :] = slerp(new_time).as_rotvec()

    if flat:
        poses_out = poses_out.reshape(T_out, n_joints * 3)
    return poses_out, trans_out
