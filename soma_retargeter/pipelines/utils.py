# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified by the eden-soma-retargeter fork: target robots are validated strings discovered from
# configs/<robot>/ directories (the hardcoded TargetType enum is gone), and the usd import is lazy.

from enum import IntEnum, auto

import soma_retargeter.utils.io_utils as io_utils


class SourceType(IntEnum):
    """Enumeration of supported source model types."""
    SOMA = auto()


_SOURCE_TYPE_TO_STR = {
    SourceType.SOMA : "soma"
}
_STR_TO_SOURCE_TYPE = {s : t for t, s in _SOURCE_TYPE_TO_STR.items()}


def get_source_str_from_type(source: SourceType) -> str:
    """
    Get the string name associated with a given source type.

    Args:
        source (SourceType): The source type enum value.

    Returns:
        str: The string representation of the source type.
    """
    return _SOURCE_TYPE_TO_STR[source]


def get_source_type_from_str(source: str) -> SourceType:
    """
    Convert a string to its corresponding SourceType enum value.

    Args:
        source (str): The string representation of a source.

    Returns:
        SourceType: The corresponding source type enum.

    Raises:
        ValueError: If the provided string does not correspond to a valid source type.
    """
    try:
        return _STR_TO_SOURCE_TYPE[source]
    except KeyError:
        allowed = ", ".join(_STR_TO_SOURCE_TYPE.keys())
        raise ValueError(f"Unknown source type: [{source}]. Allowed values: {allowed}") from None


def available_target_robots() -> list[str]:
    """Target robots with a bundled ``configs/<robot>/soma_to_<robot>_retargeter_config.json``."""
    robots = []
    for entry in sorted(io_utils.get_configs_dir().iterdir()):
        if entry.is_dir() and (entry / f"soma_to_{entry.name}_retargeter_config.json").is_file():
            robots.append(entry.name)
    return robots


def get_target_str_from_type(target) -> str:
    """Return the target robot name as a string (identity; kept for API compatibility)."""
    return str(target)


def get_target_type_from_str(target: str) -> str:
    """
    Validate a target robot name against the bundled configs.

    A robot is supported when ``configs/<robot>/soma_to_<robot>_retargeter_config.json``
    exists, so adding a robot means adding a config directory, not editing code.

    Args:
        target (str): The target robot name (e.g. "unitree_g1", "limx_oli").

    Returns:
        str: The validated target robot name, unchanged.

    Raises:
        ValueError: If no bundled config exists for the robot, listing the available ones.
    """
    robots = available_target_robots()
    if target not in robots:
        allowed = ", ".join(robots)
        raise ValueError(
            f"Unknown target type: [{target}]. Allowed values: {allowed} (a robot is supported when "
            f"soma_retargeter/configs/<robot>/soma_to_<robot>_retargeter_config.json exists)")
    return target


def get_source_model_mesh(source: SourceType, skeleton) -> dict:
    """
    Retrieve model mesh for a given source type.

    Args:
        source (SourceType): The source type for which properties should be retrieved.
        skeleton: The skeleton associated with the source model, used for loading the mesh.

    Returns:
        SkeletalMesh: The skeleton mesh for the given source type.

    Raises:
        ValueError: If the source type is not recognized.
    """
    if source == SourceType.SOMA:
        import soma_retargeter.assets.usd as usd_utils

        return usd_utils.load_skeletal_mesh_from_usd(
            str(io_utils.get_config_file('soma', 'soma_base_skel_minimal.usd')),
            skeleton,
            '/OUTPUT/c_geometry_grp',
            '/OUTPUT/c_skeleton_grp/Root')

    raise ValueError(f"Unknown source type {source}.")


def get_retargeter_config(source, target: str) -> dict:
    """
    Load the retargeter configuration between a specific source and target.

    Args:
        source (SourceType | str): The source type.
        target (str): The target robot name.

    Returns:
        dict: The loaded JSON configuration for the retargeter.

    Raises:
        ValueError: If the source or target type is not supported.
    """
    source_str = source if isinstance(source, str) else get_source_str_from_type(source)
    get_source_type_from_str(source_str)
    target = get_target_type_from_str(target)
    return io_utils.load_json(
        io_utils.get_config_file(target, f"{source_str}_to_{target}_retargeter_config.json")
    )
