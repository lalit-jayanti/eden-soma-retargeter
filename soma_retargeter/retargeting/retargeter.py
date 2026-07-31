# SPDX-FileCopyrightText: Copyright (c) 2026 The eden-soma-retargeter authors (unofficial fork of NVIDIA soma-retargeter)
# SPDX-License-Identifier: Apache-2.0

"""High-level, in-process human-to-robot motion retargeter.

Construct one :class:`MotionRetargeter` and stream an unbounded number of
clips through it: all heavy state (Newton model, IK solver, CUDA graphs, body
models, SOMA layers, PoseInversion) is built lazily, cached, and reused, so
memory stays flat and per-clip cost is only the fit + IK solve. No
subprocesses, no temporary files, no caller-side garbage collection.
"""

import gc
from pathlib import Path

import numpy as np

from soma_retargeter.retargeting.animation_io import (
    buffer_from_local_rotations,
    csv_buffer_to_arrays,
    load_template_skeleton,
    mujoco_root_offset,
    resample_buffer,
    resample_rotvec_tracks,
    zup_to_yup,
)
from soma_retargeter.retargeting.soma_format import (
    load_soma_npz,
    poses_to_matrices,
    relative_to_absolute,
    transl_to_meters,
)

_SMPL_FAMILY = ("smpl", "smplh", "smplx")
_BODY_MODELS = _SMPL_FAMILY + ("soma",)
_GENDERS = ("neutral", "male", "female")


class MotionRetargeter:
    """Retarget human motion (SMPL / SMPL-H / SMPL-X / SOMA) to robot qpos arrays.

    Args:
        body_model: "smpl" | "smplh" | "smplx" (fitted via py-soma-x) or
            "soma" (already on the SOMA rig; torch-free).
        body_model_path: Model file (``.npz``/``.pkl``), a directory of
            ``<MODEL>_<GENDER>.{npz,pkl}`` files, or a ``{gender: path}``
            mapping. Required for the smpl family; ignored for "soma".
        gender: Default gender; a per-call ``gender=`` overrides it. Fit
            stacks are built lazily and cached per gender.
        target_fps: Every clip is resampled to this rate before retargeting.
        device: Torch/warp device for the fit stage ("cuda", "cuda:1", "cpu").
        robot: Target robot; any robot with a bundled
            ``configs/<robot>/soma_to_<robot>_retargeter_config.json``
            (e.g. "unitree_g1", "limx_oli"). See
            ``soma_retargeter.pipelines.utils.available_target_robots()``.
        robot_model_path: Robot model file (MJCF/URDF) overriding the config's
            ``robot_model`` source. Required for robots whose config bundles no
            downloadable ``newton_asset`` (e.g. "limx_oli", whose
            ``HU_D04_01.urdf`` ships with the robot's asset distribution).
        forward_chunk: Body-model forward batch size (bounds peak VRAM).
        fit_chunk: PoseInversion fit batch size.
        pad_fit_chunks: Pad the tail chunk to ``fit_chunk`` frames (repeating
            the last frame) so every fit call sees constant shapes.
        reuse_pose_inversion: Reuse one PoseInversion per gender across clips.
            Faster, but its cached GPU state is corrupted whenever another soma
            context (a second gender / model, or external soma use) exists in
            the process, so it defaults to False. The shared low-LOD SOMA layer
            already avoids the dominant per-clip rebuild cost, so a fresh
            PoseInversion per clip stays cheap and memory-flat. Only enable this
            when exactly one FitContext will ever be live.
        num_betas: Shape-coefficient width (None = model default). Input betas
            are truncated / zero-padded to this width.
        flat_hand_mean: Passed to ``smplx.create``.
        ik_config_overrides: dict merged over the bundled
            ``soma_to_<robot>_retargeter_config.json`` (top-level keys replaced).
        fit_kwargs: Forwarded to ``PoseInversion.fit`` (e.g. ``body_iters``).
        soma_data_root: SOMA-X assets dir override (None = auto-download).
        gc_interval: If > 0, run ``gc.collect()`` (+ CUDA cache trim) every N
            clips as an internal backstop. 0 disables it.
        verbose: Enable the core pipeline's progress output.
        enable_procedural_transforms: Forwarded to ``SOMALayer``. True
            (py-soma-x's default) fits against its expanded twist-joint rig,
            which requires procedural assets the public ``nvidia/soma-x``
            snapshot does not ship — construction then fails, naming this kwarg.
            False fits against the public 78-joint rig, which is what a stock
            asset download supports. Both modes expose the same 78 public
            joints, so ``joint_names`` and the returned arrays keep their shape
            and meaning; the fitted rotations themselves differ, because
            procedural mode refines the skin the fit is matched against.

    Every ``retarget*`` method returns (per clip) a dict::

        {"root_pos": (T, 3) float32 metres,   # Z-up world
         "root_quat": (T, 4) float32 wxyz,
         "dof_pos":  (T, N) float32 radians,
         "joint_names": [N str],              # dof_pos column order
         "fps": float}
    """

    def __init__(self, body_model: str = "smplx", body_model_path=None, gender: str = "neutral",
                 target_fps: float = 30.0, device: str = "cuda", robot: str = "unitree_g1",
                 robot_model_path=None,
                 forward_chunk: int = 64, fit_chunk: int = 256, pad_fit_chunks: bool = True,
                 reuse_pose_inversion: bool = False, num_betas: int | None = None,
                 flat_hand_mean: bool = True, ik_config_overrides: dict | None = None,
                 fit_kwargs: dict | None = None, soma_data_root=None,
                 gc_interval: int = 0, verbose: bool = False,
                 enable_procedural_transforms: bool = True):
        from soma_retargeter.pipelines.robot_model import (
            resolve_robot_model_path,
            spec_requires_model_path,
            validate_robot_model_spec,
        )
        from soma_retargeter.pipelines.utils import available_target_robots, get_retargeter_config

        if body_model not in _BODY_MODELS:
            raise ValueError(f"body_model must be one of {_BODY_MODELS}, got {body_model!r}")
        if gender not in _GENDERS:
            raise ValueError(f"gender must be one of {_GENDERS}, got {gender!r}")
        robots = available_target_robots()
        if robot not in robots:
            raise ValueError(
                f"robot {robot!r} is not supported; available robots: {robots} (a robot is supported when "
                "soma_retargeter/configs/<robot>/soma_to_<robot>_retargeter_config.json exists)")
        if body_model in _SMPL_FAMILY and body_model_path is None:
            raise ValueError(f"body_model_path is required for body_model={body_model!r}")

        # Fail fast on the robot-model source: the config's spec must be well-formed, and robots
        # without a downloadable asset need robot_model_path= now, not at first retarget.
        self._retarget_config = get_retargeter_config("soma", robot)
        if ik_config_overrides:
            self._retarget_config.update(dict(ik_config_overrides))
        spec = self._retarget_config.get("robot_model")
        if spec is None:
            raise ValueError(
                f"retargeter config for robot {robot!r} has no 'robot_model' section; add one "
                "(see configs/unitree_g1/soma_to_unitree_g1_retargeter_config.json)")
        validate_robot_model_spec(spec, robot)
        if robot_model_path is None and spec_requires_model_path(spec):
            resolve_robot_model_path(spec, None, robot)  # raises, naming the kwarg
        if robot_model_path is not None and not Path(robot_model_path).exists():
            raise FileNotFoundError(f"robot_model_path for robot {robot!r} does not exist: {robot_model_path}")

        self.body_model = body_model
        self.body_model_path = body_model_path
        self.gender = gender
        self.target_fps = float(target_fps)
        self.device = device
        self.robot = robot
        self.robot_model_path = robot_model_path
        self.forward_chunk = forward_chunk
        self.fit_chunk = fit_chunk
        self.pad_fit_chunks = pad_fit_chunks
        self.reuse_pose_inversion = reuse_pose_inversion
        self.num_betas = num_betas
        self.flat_hand_mean = flat_hand_mean
        self.ik_config_overrides = dict(ik_config_overrides) if ik_config_overrides else None
        self.fit_kwargs = dict(fit_kwargs) if fit_kwargs else None
        self.soma_data_root = soma_data_root
        self.gc_interval = gc_interval
        self.verbose = verbose
        self.enable_procedural_transforms = enable_procedural_transforms

        self._template_skeleton = load_template_skeleton()
        self._template_parent_names = {
            name: (self._template_skeleton.joint_names[p] if p != -1 else None)
            for name, p in zip(self._template_skeleton.joint_names, self._template_skeleton.parent_indices)
        }
        self._pipeline = None
        self._fit_contexts = {}
        self._clips_done = 0

    # ------------------------------------------------------------------ public

    @property
    def joint_names(self) -> list[str]:
        """``dof_pos`` column names, derived from the robot model's joint labels
        (``joint_q`` order, free root skipped). First access builds the IK pipeline."""
        return list(self._ensure_pipeline().joint_names)

    def retarget(self, poses, trans, *, source_fps: float, betas=None, gender: str | None = None) -> dict:
        """Retarget one SMPL-family clip.

        Args:
            poses: (T, D) or (T, J, 3) axis-angle, AMASS layout for the
                configured body model (D = 72 smpl / 156 smplh / 165 smplx).
            trans: (T, 3) root translation in metres, Z-up.
            source_fps: Frame rate of the input arrays.
            betas: (K,) shape coefficients (None = zeros); truncated / padded
                to the model width.
            gender: Override the default gender for this clip.
        """
        buffer = self._buffer_from_smpl(poses, trans, source_fps, betas, gender)
        return self._finish(self._run_ik([buffer]))[0]

    def retarget_soma(self, poses, transl, *, source_fps: float, joint_names: list[str] | None = None,
                      absolute_pose: bool = True, joint_orient=None, unit: str = "meters") -> dict:
        """Retarget one SOMA-format clip (torch-free).

        Args:
            poses: (T, J, 3) axis-angle or (T, J, 3, 3) rotation matrices,
                local joint rotations on the SOMA rig (Z-up world).
            transl: (T, 3) Hips translation in ``unit``.
            source_fps: Frame rate of the input arrays.
            joint_names: Names for the J pose columns (default: the template
                rig order without the virtual Root, i.e. ``save_soma_npz``'s
                J=77 layout).
            absolute_pose: Whether poses are absolute local rotations. If
                False, ``joint_orient`` must be given (T-pose-relative input).
            joint_orient: (J, 3, 3) per-joint orientation for relative poses.
            unit: Unit of ``transl`` ("meters" / "centimeters" / "millimeters").
        """
        buffer = self._buffer_from_soma(poses, transl, source_fps, joint_names, absolute_pose, joint_orient, unit)
        return self._finish(self._run_ik([buffer]))[0]

    def retarget_soma_npz(self, path, *, source_fps: float) -> dict:
        """Retarget a ``save_soma_npz`` motion file (torch-free).

        ``source_fps`` is required because the SOMA npz format stores no frame
        rate.
        """
        d = load_soma_npz(path)
        return self.retarget_soma(
            d["poses"], d["transl"], source_fps=source_fps, joint_names=d["joint_names"],
            absolute_pose=d["absolute_pose"], joint_orient=d["joint_orient"], unit=d["unit"])

    def retarget_bvh(self, bvh_path, *, resample: bool = True) -> dict:
        """Retarget a SOMA-skeleton BVH file (e.g. the BONES-SEED dataset; torch-free).

        The clip's own frame rate is read from the file; with ``resample`` it
        is resampled to ``target_fps`` first.
        """
        buffer = self._buffer_from_bvh(bvh_path, resample)
        return self._finish(self._run_ik([buffer]))[0]

    def retarget_batch(self, clips: list[dict]) -> list[dict]:
        """Retarget many clips in one batched IK solve (one env per clip).

        Each clip dict holds the keyword arguments of one of the single-clip
        methods, dispatched by key: ``bvh_path`` -> :meth:`retarget_bvh`,
        ``transl`` -> :meth:`retarget_soma`, otherwise :meth:`retarget`.

        The IK solve runs all clips as parallel Newton environments and costs
        ``max(frames)`` iterations, so batching clips of similar length is
        fastest. A batch size not seen before builds (and caches) one new
        solver + CUDA graph.
        """
        if not clips:
            return []
        buffers = []
        for clip in clips:
            clip = dict(clip)
            if "bvh_path" in clip:
                buffers.append(self._buffer_from_bvh(clip.pop("bvh_path"), clip.pop("resample", True)))
            elif "transl" in clip:
                buffers.append(self._buffer_from_soma(
                    clip.pop("poses"), clip.pop("transl"), clip.pop("source_fps"),
                    clip.pop("joint_names", None), clip.pop("absolute_pose", True),
                    clip.pop("joint_orient", None), clip.pop("unit", "meters")))
            else:
                buffers.append(self._buffer_from_smpl(
                    clip.pop("poses"), clip.pop("trans"), clip.pop("source_fps"),
                    clip.pop("betas", None), clip.pop("gender", None)))
            if clip:
                raise TypeError(f"Unexpected clip keys: {sorted(clip)}")
        return self._finish(self._run_ik(buffers))

    def close(self):
        """Release cached solvers, CUDA graphs, and fit stacks."""
        if self._pipeline is not None:
            self._pipeline.release()
            self._pipeline = None
        self._fit_contexts.clear()
        self._collect(force=True)

    # ----------------------------------------------------------------- buffers

    def _buffer_from_smpl(self, poses, trans, source_fps, betas, gender):
        from soma_retargeter.retargeting.fitting import pose_dim

        if self.body_model not in _SMPL_FAMILY:
            raise ValueError(
                f"retarget() fits {_SMPL_FAMILY} motion, but this retargeter was built with "
                f"body_model={self.body_model!r}; use retarget_soma()/retarget_bvh() instead.")

        poses = np.asarray(poses, dtype=np.float32)
        trans = np.asarray(trans, dtype=np.float32)
        if poses.ndim == 3:
            poses = poses.reshape(poses.shape[0], -1)
        dim = pose_dim(self.body_model)
        if poses.ndim != 2 or poses.shape[1] != dim:
            raise ValueError(f"poses: expected (T, {dim}) for {self.body_model}, got {poses.shape}")
        if trans.shape != (poses.shape[0], 3):
            raise ValueError(f"trans: expected ({poses.shape[0]}, 3), got {trans.shape}")

        poses, trans = resample_rotvec_tracks(poses, trans, float(source_fps), self.target_fps)

        ctx = self._fit_context(gender or self.gender)
        rotations, root_trans = ctx.fit_clip(
            poses, trans, betas,
            fit_chunk=self.fit_chunk, forward_chunk=self.forward_chunk,
            pad_chunks=self.pad_fit_chunks, fit_kwargs=self.fit_kwargs)

        rotations, root_trans = zup_to_yup(rotations, root_trans, ctx.hips_index)
        return buffer_from_local_rotations(
            self._template_skeleton, rotations, root_trans, self.target_fps, ctx.joint_names)

    def _buffer_from_soma(self, poses, transl, source_fps, joint_names, absolute_pose, joint_orient, unit):
        if joint_names is None:
            joint_names = [n for n in self._template_skeleton.joint_names if n != "Root"]
        joint_names = [str(n) for n in joint_names]

        poses = np.asarray(poses, dtype=np.float32)
        transl = transl_to_meters(transl, unit)
        if poses.ndim == 3 and poses.shape[-1] == 3:
            poses, transl = resample_rotvec_tracks(poses, transl, float(source_fps), self.target_fps)
            poses_mat = poses_to_matrices(poses)
        else:
            poses_mat = poses_to_matrices(poses)
            if float(source_fps) != self.target_fps:
                T = poses_mat.shape[0]
                from scipy.spatial.transform import Rotation

                rotvec = Rotation.from_matrix(poses_mat.reshape(-1, 3, 3)).as_rotvec()
                rotvec = rotvec.reshape(T, -1, 3).astype(np.float32)
                rotvec, transl = resample_rotvec_tracks(rotvec, transl, float(source_fps), self.target_fps)
                poses_mat = poses_to_matrices(rotvec)

        if not absolute_pose:
            if joint_orient is None:
                raise ValueError("absolute_pose=False requires joint_orient")
            poses_mat = relative_to_absolute(
                poses_mat, np.asarray(joint_orient, dtype=np.float32), joint_names, self._template_parent_names)

        if "Hips" not in joint_names:
            raise ValueError("SOMA poses must include a 'Hips' column")
        poses_mat, transl = zup_to_yup(poses_mat, transl, joint_names.index("Hips"))
        return buffer_from_local_rotations(
            self._template_skeleton, poses_mat, transl, self.target_fps, joint_names)

    def _buffer_from_bvh(self, bvh_path, resample):
        from soma_retargeter.assets.bvh import load_bvh

        path = Path(bvh_path)
        if not path.exists():
            raise FileNotFoundError(f"BVH file not found: {path}")
        _, buffer = load_bvh(str(path), self._template_skeleton)
        if resample:
            buffer = resample_buffer(buffer, self.target_fps)
        return buffer

    # ---------------------------------------------------------------- internal

    def _fit_context(self, gender: str):
        from soma_retargeter.retargeting.assets import resolve_body_model_path
        from soma_retargeter.retargeting.fitting import FitContext

        if gender not in _GENDERS:
            raise ValueError(f"gender must be one of {_GENDERS}, got {gender!r}")
        ctx = self._fit_contexts.get(gender)
        if ctx is None:
            ctx = FitContext(
                body_model=self.body_model,
                model_path=resolve_body_model_path(self.body_model, self.body_model_path, gender),
                gender=gender,
                device=self._torch_device(),
                num_betas=self.num_betas,
                flat_hand_mean=self.flat_hand_mean,
                soma_data_root=self.soma_data_root,
                reuse_pose_inversion=self.reuse_pose_inversion,
                enable_procedural_transforms=self.enable_procedural_transforms,
            )
            self._fit_contexts[gender] = ctx
        return ctx

    def _torch_device(self):
        import torch

        if str(self.device).startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(self.device)

    def _ensure_pipeline(self):
        if self._pipeline is None:
            from soma_retargeter.retargeting.pipeline_cache import CachedNewtonPipeline

            self._pipeline = CachedNewtonPipeline(
                self._template_skeleton, source_type="soma", robot_type=self.robot,
                retarget_config=dict(self._retarget_config),
                robot_model_path=self.robot_model_path, verbose=self.verbose)
        return self._pipeline

    def _run_ik(self, buffers):
        import warp as wp

        pipeline = self._ensure_pipeline()
        pipeline.clear()
        pipeline.add_input_motions(buffers, [mujoco_root_offset()] * len(buffers), True)
        results = pipeline.execute()
        pipeline.clear()
        # Barrier so the IK's warp work finishes before the next clip's fit
        # (or the caller reading outputs) reuses warp's memory pool.
        if wp.get_device().is_cuda:
            wp.synchronize_device()
        return results

    def _finish(self, csv_buffers):
        out = []
        for buf in csv_buffers:
            arrays = csv_buffer_to_arrays(buf)
            arrays["joint_names"] = list(self.joint_names)
            out.append(arrays)
        self._clips_done += len(out)
        self._collect()
        return out

    def _collect(self, force: bool = False):
        if not force and (self.gc_interval <= 0 or self._clips_done % self.gc_interval != 0):
            return
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
