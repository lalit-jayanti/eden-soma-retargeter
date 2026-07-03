# SPDX-FileCopyrightText: Copyright (c) 2026 The eden-soma-retargeter authors (unofficial fork of NVIDIA soma-retargeter)
# SPDX-License-Identifier: Apache-2.0

"""Numpy-only helpers for the SOMA motion format (as written by ``soma.io.save_soma_npz``).

py-soma-x 0.1.0 ships only the writer; this module provides the matching
reader plus the pose-convention conversions needed to feed SOMA motions into
the retargeting pipeline without importing torch.
"""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

_UNIT_TO_METERS = {"meters": 1.0, "centimeters": 0.01, "millimeters": 0.001}


def load_soma_npz(path: str | Path) -> dict:
    """Load a SOMA motion ``.npz`` (the ``save_soma_npz`` format) into plain arrays.

    Returns:
        dict with keys ``poses`` ((T, J, 3) rotvec or (T, J, 3, 3) matrix),
        ``transl`` (T, 3), ``joint_names`` (list[str]), ``rotation_repr``,
        ``absolute_pose`` (bool), ``unit`` (str), ``keep_root`` (bool), and
        ``joint_orient`` ((J, 3, 3) or None; present iff poses are
        T-pose-relative).
    """
    with np.load(path, allow_pickle=False) as d:
        missing = [k for k in ("poses", "transl", "joint_names") if k not in d.files]
        if missing:
            raise ValueError(f"{path} is not a SOMA motion npz: missing keys {missing}")
        out = {
            "poses": np.asarray(d["poses"], dtype=np.float32),
            "transl": np.asarray(d["transl"], dtype=np.float32),
            "joint_names": [str(n) for n in d["joint_names"]],
            "rotation_repr": str(d["rotation_repr"]) if "rotation_repr" in d.files else None,
            "absolute_pose": bool(d["absolute_pose"]) if "absolute_pose" in d.files else True,
            "unit": str(d["unit"]) if "unit" in d.files else "meters",
            "keep_root": bool(d["keep_root"]) if "keep_root" in d.files else False,
            "joint_orient": np.asarray(d["joint_orient"], dtype=np.float32) if "joint_orient" in d.files else None,
        }
    return out


def transl_to_meters(transl: np.ndarray, unit: str) -> np.ndarray:
    try:
        scale = _UNIT_TO_METERS[unit]
    except KeyError:
        raise ValueError(f"Unknown unit {unit!r}; expected one of {sorted(_UNIT_TO_METERS)}") from None
    transl = np.asarray(transl, dtype=np.float32)
    return transl if scale == 1.0 else transl * scale


def poses_to_matrices(poses: np.ndarray) -> np.ndarray:
    """Convert (T, J, 3) rotvec or (T, J, 3, 3) matrix poses to (T, J, 3, 3) matrices."""
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim == 4 and poses.shape[-2:] == (3, 3):
        return poses
    if poses.ndim == 3 and poses.shape[-1] == 3:
        T, J = poses.shape[:2]
        return Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(T, J, 3, 3).astype(np.float32)
    raise ValueError(f"poses shape {poses.shape} is neither (T, J, 3) rotvec nor (T, J, 3, 3) matrix")


def relative_to_absolute(
    poses_mat: np.ndarray,
    joint_orient: np.ndarray,
    joint_names: list[str],
    parent_names: dict[str, str | None],
) -> np.ndarray:
    """Convert T-pose-relative local rotations to absolute local rotations.

    Numpy port of ``soma.geometry.rig_utils.apply_joint_orient_local``:
    ``R_abs[j] = orient[parent(j)].T @ R_rel[j] @ orient[j]`` (identity orient
    for joints whose parent is not in the file, e.g. Hips when Root is stripped).

    Args:
        poses_mat: (T, J, 3, 3) relative local rotations.
        joint_orient: (J, 3, 3) per-joint world orientation stored in the npz.
        joint_names: J names labelling the pose columns.
        parent_names: joint name -> parent joint name (None for the root),
            from the skeleton hierarchy.
    """
    J = poses_mat.shape[1]
    if joint_orient.shape != (J, 3, 3):
        raise ValueError(f"joint_orient shape {joint_orient.shape} != ({J}, 3, 3)")

    name_to_col = {n: i for i, n in enumerate(joint_names)}
    orient_parent_T = np.empty_like(joint_orient)
    for j, name in enumerate(joint_names):
        parent = parent_names.get(name)
        parent_col = name_to_col.get(parent) if parent is not None else None
        if parent_col is None:
            orient_parent_T[j] = np.eye(3, dtype=joint_orient.dtype)
        else:
            orient_parent_T[j] = joint_orient[parent_col].T

    return orient_parent_T[None] @ poses_mat @ joint_orient[None]
