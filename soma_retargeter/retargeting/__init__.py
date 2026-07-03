# SPDX-FileCopyrightText: Copyright (c) 2026 The eden-soma-retargeter authors (unofficial fork of NVIDIA soma-retargeter)
# SPDX-License-Identifier: Apache-2.0

"""High-level in-process retargeting API.

Lazy module: importing ``soma_retargeter.retargeting`` (or the base package)
pulls no heavy dependencies; torch/smplx/py-soma-x load only when the
SMPL-family fit path is actually used.
"""

__all__ = ["MotionRetargeter", "CachedNewtonPipeline"]


def __getattr__(name):
    if name == "MotionRetargeter":
        from soma_retargeter.retargeting.retargeter import MotionRetargeter

        return MotionRetargeter
    if name == "CachedNewtonPipeline":
        from soma_retargeter.retargeting.pipeline_cache import CachedNewtonPipeline

        return CachedNewtonPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
