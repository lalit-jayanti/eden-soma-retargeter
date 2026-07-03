# SPDX-FileCopyrightText: Copyright (c) 2026 The eden-soma-retargeter authors (unofficial fork of NVIDIA soma-retargeter)
# SPDX-License-Identifier: Apache-2.0

"""Body-model asset resolution for the fitting stage."""

import os
from pathlib import Path

_SMPL_FAMILY = ("smpl", "smplh", "smplx")


def resolve_body_model_path(body_model: str, body_model_path, gender: str) -> Path:
    """Resolve the body-model file for a given gender.

    Args:
        body_model: "smpl" | "smplh" | "smplx".
        body_model_path: One of
            - a model file (``.npz``/``.pkl``) used for every gender,
            - a directory containing ``<MODEL>_<GENDER>.{npz,pkl}`` files,
            - a mapping ``{gender: path}``.
        gender: "neutral" | "male" | "female".
    """
    if body_model not in _SMPL_FAMILY:
        raise ValueError(f"body_model {body_model!r} has no model file (expected one of {_SMPL_FAMILY})")
    if body_model_path is None:
        raise ValueError(f"body_model_path is required for body_model={body_model!r}")

    if isinstance(body_model_path, dict):
        try:
            return Path(body_model_path[gender]).expanduser()
        except KeyError:
            raise FileNotFoundError(
                f"No model path configured for gender={gender!r}; available: {sorted(body_model_path)}") from None

    p = Path(body_model_path).expanduser()
    if p.is_file():
        return p
    if p.is_dir():
        stem = f"{body_model.upper()}_{gender.upper()}"
        for ext in (".npz", ".pkl"):
            candidate = p / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No {stem}.npz/.pkl under {p} for gender={gender!r}")
    raise FileNotFoundError(f"body_model_path {p} does not exist")


def ensure_smplh_data_root(base_data_root=None) -> Path:
    """Build a SOMA-X data_root usable with ``identity_model_type='smplh'``.

    The upstream nvidia/soma-x asset snapshot ships topology-transfer wrap
    meshes for ``SMPL/`` and ``SMPLX/`` but not ``SMPLH/``. SMPL-H shares
    SMPL's 6890-vertex mesh topology, so SMPL's ``base_body.obj`` /
    ``SOMA_wrap.obj`` are valid for it. This creates (once) a symlink farm that
    mirrors the real data_root and adds ``SMPLH -> SMPL``.

    Args:
        base_data_root: The real SOMA-X assets dir; defaults to
            ``soma.assets.get_assets_dir()`` (auto-downloads on first use).

    Returns:
        Path of the shimmed data_root.
    """
    if base_data_root is None:
        from soma.assets import get_assets_dir

        base_data_root = get_assets_dir()
    base = Path(base_data_root).resolve()
    if (base / "SMPLH").exists():
        return base
    if not (base / "SMPL").exists():
        raise FileNotFoundError(
            f"SMPL-H support borrows SMPL's wrap meshes, but neither SMPLH/ nor SMPL/ exists under {base}. "
            "Provide a soma-x data_root containing SMPL/ (base_body.obj + SOMA_wrap.obj)."
        )

    shim = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "eden-soma-retargeter" / "smplh_data_root"
    marker = shim / ".base_data_root"
    smplh_link = shim / "SMPLH"
    # Rebuild unless the marker matches this base AND the SMPLH link still points
    # at base/SMPL (a moved snapshot or a broken link forces a rebuild).
    if (
        marker.exists()
        and marker.read_text().strip() == str(base)
        and smplh_link.is_symlink()
        and smplh_link.resolve() == (base / "SMPL").resolve()
    ):
        return shim

    shim.mkdir(parents=True, exist_ok=True)
    for entry in list(shim.iterdir()):
        if entry.is_symlink() or entry.name == ".base_data_root":
            entry.unlink()
    for entry in base.iterdir():
        (shim / entry.name).symlink_to(entry)
    smplh_link.symlink_to(base / "SMPL")
    marker.write_text(str(base))
    return shim
