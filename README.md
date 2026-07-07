# eden-soma-retargeter

> **Note:** This is an unofficial fork of [NVIDIA/soma-retargeter](https://github.com/NVIDIA/soma-retargeter). It is not affiliated with or endorsed by NVIDIA.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

![SOMA Retargeter Banner](assets/docs/banner.gif)

Convert [SOMA](https://github.com/NVlabs/SOMA-X) human motion captures into humanoid robot joint animation. Takes BVH motion files as input and produces robot-playable CSV joint data as output using GPU-optimized inverse kinematics via [Newton](https://github.com/newton-physics/newton) and high-performance computation with [NVIDIA Warp](https://github.com/NVIDIA/warp).

The retargeting pipeline handles proportional human-to-robot scaling, multi-objective IK solving with joint limits, feet stabilization to maintain ground contact, and per-DOF joint limit clamping. Currently supports SOMA as the input skeleton, with Unitree G1 (29 DOF) and LimX Oli (HU_D04, 31 DOF) as output robots. Robots are defined by a `soma_retargeter/configs/<robot>/` config bundle — adding one requires no dispatch-code changes (see the `robot_model` config section and `soma_retargeter.pipelines.robot_model`).

> **Note:** This project is in active development. The API may change between releases as the design is refined.

## Requirements

- **Python:** 3.10, 3.11, or 3.12 (upstream tested on 3.12; 3.10/3.11 supported via relaxed numpy/scipy pins)
- **Git LFS:** Installed and initialized for asset downloads (only needed when working from source)
- **OS:** Windows (x86-64) and Linux (x86-64, aarch64)
- **GPU:** NVIDIA GPU (Maxwell or newer), driver 545+ (CUDA 12). No local CUDA Toolkit installation required.

## Installation

### From PyPI (recommended)

```bash
pip install eden-soma-retargeter
soma-bvh-to-csv --help
```

This installs the `soma_retargeter` Python package and the `soma-bvh-to-csv` console script. The default retargeting config and the SOMA zero-frame BVH ship inside the wheel, so no LFS checkout is needed for the standard pipeline.

Runtime requirements: Python 3.10/3.11/3.12, NVIDIA GPU (Maxwell or newer), driver 545+.

<details>

<summary>From source (for contributors)</summary>

LFS only needs to be pulled if you plan to use the sample motions in `assets/motions/` or rebuild the docs in `assets/docs/`. The wheel itself bundles only the default config and the SOMA zero-frame BVH that already live under `soma_retargeter/configs/`.

### Method 1 (conda + pip)

```bash
conda create -n soma-retargeter python=3.12 -y
conda activate soma-retargeter
git lfs pull   # sample motions / docs assets
pip install .
```

### Method 2 (uv)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git lfs pull
uv sync
```

`uv sync` creates an isolated `.venv` inside the project directory, installs the correct Python version and resolves all dependencies.

### Platform-specific notes

**Note (Linux):** For the GUI viewer to work, install `tkinter`

```bash
sudo apt-get install python3.12-tk
```

**Note (Windows):** If `imgui-bundle` fails to install, the Microsoft Visual C++ Redistributables may be missing. Download from the [official Microsoft documentation](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).

</details>

## Motion Data

This repo includes 10 sample BVH/CSV pairs in `assets/motions/` for immediate testing.

For large-scale motion data, see the [SEED dataset](https://huggingface.co/datasets/bones-studio/seed) (Skeletal Everyday Embodiment Dataset) published by [Bones Studio](https://huggingface.co/bones-studio). SEED provides a large-scale collection of human motions on the SOMA uniform-proportion skeleton, which is the expected input format for this tool. The G1 robot motion data included in SEED was retargeted using the upstream SOMA Retargeter.

## Quick Start

> When using **uv**, replace `soma-bvh-to-csv` with `uv run soma-bvh-to-csv` in the commands below.

### Interactive viewer (OpenGL)

```bash
soma-bvh-to-csv --viewer gl
```

This uses the bundled default config. To override:

```bash
soma-bvh-to-csv --config ./my_config.json --viewer gl
```

![Interactive viewer interface](assets/docs/interactive-viewer-screenshot.png)

The viewer displays the source SOMA motion alongside the retargeted robot in a 3D viewport. Use the right panel to load BVH files, run retargeting, and save CSV output. Playback controls at the bottom allow scrubbing, speed adjustment, and looping. Toggle visibility of the skinned mesh, skeleton, joint axes, and positioning gizmos.

### Batch conversion (headless)

Process a folder of BVH files without a display. Set `import_folder` and `export_folder` in the config file, then run:

```bash
soma-bvh-to-csv --config ./my_config.json --viewer null
```

Batch mode recursively finds all `.bvh` files in the import folder, processes them in configurable batch sizes, and writes CSV files to the export folder mirroring the input directory structure.

The same entry point is also reachable as a Python module, useful for embedding in other tools:

```bash
python -m soma_retargeter.app.bvh_to_csv_converter --viewer null
```

### In-process Python API (`soma_retargeter.retargeting`)

For embedding retargeting directly in a Python program — with no temp files or
subprocesses, and flat memory across an unbounded stream of clips — use
`MotionRetargeter`. It also fits **SMPL / SMPL-H / SMPL-X** motion into the SOMA
rig on the way in (via [py-soma-x](https://github.com/NVlabs/SOMA-X)), so you can
retarget raw AMASS-style body-model motion, not just SOMA BVH.

```python
from soma_retargeter.retargeting import MotionRetargeter

# SMPL-X -> Unitree G1 (fitting requires the [fit] extra)
rt = MotionRetargeter(body_model="smplx", body_model_path="/path/to/SMPLX_NEUTRAL.npz",
                      target_fps=30.0, device="cuda")
out = rt.retarget(poses, trans, source_fps=120.0, betas=betas)   # AMASS layout arrays
# out = {"root_pos" (T,3) m, "root_quat" (T,4) wxyz, "dof_pos" (T,29) rad,
#        "joint_names" [29], "fps"}

# Native SOMA motion needs no body model and no torch:
rt_soma = MotionRetargeter(body_model="soma", target_fps=30.0)
out = rt_soma.retarget_bvh("motion.bvh")                 # SOMA-skeleton BVH (e.g. BONES-SEED)
out = rt_soma.retarget_soma_npz("motion.npz", source_fps=30.0)   # save_soma_npz format

# SOMA -> LimX Oli (31 DOF). The Oli URDF is not bundled: pass your own copy
# (e.g. HU_D04_01.urdf from the robot's asset distribution).
rt_oli = MotionRetargeter(body_model="soma", robot="limx_oli",
                          robot_model_path="/path/to/HU_D04_01.urdf")
out = rt_oli.retarget_bvh("motion.bvh")                  # dof_pos (T, 31)
```

Install the fit dependencies with the extra:

```bash
pip install "eden-soma-retargeter[fit]"
```

`retarget_batch([...])` retargets many clips in one IK solve (one env per clip).
See the `MotionRetargeter` docstring for the full option set (per-call gender,
chunk sizes, IK overrides, …).

## Code Overview

### `soma_retargeter/`

| Module | Description |
|--------|-------------|
| `app/bvh_to_csv_converter.py` | Main entry point (also exposed as the `soma-bvh-to-csv` console script). Drives both interactive and headless batch retargeting modes. |
| `animation/` | Core data structures for skeletons, animation buffers, IK, and skinned meshes. |
| `assets/` | File I/O for BVH, CSV, and USD formats. |
| `pipelines/` | Retargeting pipeline: IK solving, feet stabilization, and joint limit clamping. |
| `retargeting/` | Fork-added in-process Python API (`MotionRetargeter`): SMPL-family fitting + native SOMA input. |
| `robotics/` | Human-to-robot scaling and robot output formatting. |
| `renderers/` | Visualization for the interactive viewer. |
| `utils/` | Math, pose, coordinate conversion, Newton and Warp helpers. |
| `configs/` | JSON configuration for retargeting, scaling, and feet stabilization parameters, plus the default `default_bvh_to_csv_converter_config.json` shipped with the wheel. |

## Related Work & Acknowledgments

Upstream SOMA-ecosystem projects for humanoid motion data:

* [SOMA Body Model](https://github.com/NVlabs/SOMA-X) - Parametric human body model with standardized skeleton, mesh, and shape parameters
* [GEM-X](https://github.com/NVlabs/GEM-X) - Human motion estimation from video
* [Kimodo](https://github.com/nv-tlabs/kimodo) - Kinematic motion diffusion model for text and constraint-driven 3D human and robot motion generation
* [ProtoMotions](https://github.com/NVlabs/ProtoMotions) - GPU-accelerated simulation and learning framework for training physically simulated digital humans and humanoid robots
* [SONIC](https://nvlabs.github.io/GEAR-SONIC/) - Whole-body control for humanoid robots, training locomotion and interaction policies

The upstream project also draws on [GMR](https://github.com/YanjieZe/GMR) (General Motion Retargeting) and [PyRoki](https://pyroki-toolkit.github.io/) (A Modular Toolkit for Robot Kinematic Optimization).

## License

Licensed under [Apache-2.0](LICENSE), same as the upstream project. Installation pulls in third-party open-source dependencies; review their license terms before use.
