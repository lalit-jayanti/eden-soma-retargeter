# SPDX-FileCopyrightText: Copyright (c) 2026 The eden-soma-retargeter authors (unofficial fork of NVIDIA soma-retargeter)
# SPDX-License-Identifier: Apache-2.0

"""Config-driven robot model loading and model-derived DOF names.

A target robot is described by the ``robot_model`` section of its
``configs/<robot>/soma_to_<robot>_retargeter_config.json``::

    "robot_model": {
        "type": "mjcf" | "urdf",                 // which ModelBuilder loader to use
        "newton_asset": "unitree_g1",            // optional: newton.utils.download_asset() bundle
        "relpath": "mjcf/g1_29dof_rev_1_0.xml",  // model file inside the downloaded bundle
        "floating": true                         // urdf only: free-joint the base (default true)
    }

A spec with a ``newton_asset`` is self-contained (the model file downloads on
demand). A spec without one describes a robot whose model file ships outside
this package (e.g. ``limx_oli``'s ``HU_D04_01.urdf``) and must be supplied by
the caller via ``robot_model_path=``.

Shared by ``NewtonPipeline``, ``FeetStabilizer``, and the CLI app so every
consumer builds the target robot from the same spec and the same file.
"""

from pathlib import Path

_MODEL_TYPES = ("mjcf", "urdf")
_SPEC_KEYS = ("type", "newton_asset", "relpath", "floating")


def validate_robot_model_spec(spec: dict, robot_type: str = "") -> dict:
    """Fail fast on a malformed ``robot_model`` config section."""
    where = f"'robot_model' section for robot {robot_type!r}"
    if not isinstance(spec, dict):
        raise ValueError(f"{where} must be a mapping, got {type(spec).__name__}")
    unknown = sorted(set(spec) - set(_SPEC_KEYS))
    if unknown:
        raise ValueError(f"{where} has unknown keys {unknown}; allowed keys: {list(_SPEC_KEYS)}")
    model_type = spec.get("type")
    if model_type not in _MODEL_TYPES:
        raise ValueError(f"{where} needs 'type' set to one of {_MODEL_TYPES}, got {model_type!r}")
    if "relpath" in spec and "newton_asset" not in spec:
        raise ValueError(f"{where}: 'relpath' is only meaningful together with 'newton_asset'")
    if "floating" in spec:
        if model_type != "urdf":
            raise ValueError(f"{where}: 'floating' only applies to type 'urdf' (mjcf reads <freejoint>)")
        if not isinstance(spec["floating"], bool):
            raise ValueError(f"{where}: 'floating' must be a bool, got {spec['floating']!r}")
    return spec


def spec_requires_model_path(spec: dict) -> bool:
    """Whether the spec depends on a caller-supplied ``robot_model_path``."""
    return "newton_asset" not in spec


def default_robot_model_spec(robot_type: str) -> dict:
    """The ``robot_model`` section of the robot's bundled retargeter config."""
    import soma_retargeter.pipelines.utils as pipeline_utils

    config = pipeline_utils.get_retargeter_config("soma", robot_type)
    spec = config.get("robot_model")
    if spec is None:
        raise ValueError(
            f"retargeter config for robot {robot_type!r} has no 'robot_model' section; add one "
            "(see configs/unitree_g1/soma_to_unitree_g1_retargeter_config.json)")
    return validate_robot_model_spec(spec, robot_type)


def resolve_robot_model_path(spec: dict, path_override=None, robot_type: str = "") -> Path:
    """Resolve the model file: explicit path > ``newton_asset`` download > error."""
    if path_override is not None:
        path = Path(path_override)
        if not path.exists():
            raise FileNotFoundError(f"robot_model_path for robot {robot_type!r} does not exist: {path}")
        return path
    if "newton_asset" in spec:
        import newton

        return newton.utils.download_asset(spec["newton_asset"]) / spec.get("relpath", "")
    raise ValueError(
        f"robot {robot_type!r} needs an explicit model file: its config bundles no downloadable "
        f"'newton_asset'. Pass robot_model_path=<path to the robot's {spec['type']} file>, e.g. "
        f"MotionRetargeter(robot={robot_type!r}, robot_model_path='/path/to/robot.{spec['type']}')")


def load_robot_model(builder, spec: dict, path=None, robot_type: str = "") -> Path:
    """Load the robot into ``builder`` per the spec; returns the file used."""
    spec = validate_robot_model_spec(spec, robot_type)
    path = resolve_robot_model_path(spec, path, robot_type)
    if spec["type"] == "mjcf":
        builder.add_mjcf(path)
    else:
        builder.add_urdf(str(path), floating=spec.get("floating", True))
    return path


def derive_joint_names(builder, robot_type: str = "") -> list[str]:
    """Actuated DOF names in ``joint_q`` order — the ``dof_pos`` column labels.

    Pipeline output rows are raw free-joint states ``[root(7), dof...]``, so
    the builder must hold exactly one FREE joint (the root, declared first) and
    every other coordinate-bearing joint must be 1-DOF so one name labels one
    column. FIXED joints (frame decorations) carry no state and are skipped.
    """
    from newton import JointType

    from soma_retargeter.utils.newton_utils import get_name_from_label

    names = []
    free_joint_indices = []
    for i, (jtype, label, dof_dim) in enumerate(zip(builder.joint_type, builder.joint_label, builder.joint_dof_dim)):
        jtype = JointType(jtype)
        _, coord_count = jtype.dof_count(sum(dof_dim))
        if jtype == JointType.FREE:
            free_joint_indices.append(i)
            continue
        if coord_count == 0:
            continue
        if coord_count != 1:
            raise ValueError(
                f"robot {robot_type!r}: joint {get_name_from_label(label)!r} has {coord_count} position "
                "coordinates; only 1-DOF joints (plus the free root) are supported")
        names.append(get_name_from_label(label))
    if free_joint_indices != [0]:
        raise ValueError(
            f"robot {robot_type!r}: expected exactly one FREE root joint declared first, found FREE joints "
            f"at indices {free_joint_indices}; the [root(7), dof...] output layout does not hold")
    if builder.joint_coord_count != 7 + len(names):
        raise ValueError(
            f"robot {robot_type!r}: joint_coord_count={builder.joint_coord_count} != 7 + {len(names)} named "
            "DOFs; the model contains unsupported joint state")
    return names


def find_body_index(body_names: list[str], name: str, context: str, robot_type: str = "") -> int:
    """``body_names.index(name)`` with an error that names the offender and the options."""
    try:
        return body_names.index(name)
    except ValueError:
        raise ValueError(
            f"robot {robot_type!r}: {context} refers to body {name!r}, which is not a body of the loaded "
            f"model. Available bodies: {', '.join(body_names)}") from None
