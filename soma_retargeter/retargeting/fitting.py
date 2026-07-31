# SPDX-FileCopyrightText: Copyright (c) 2026 The eden-soma-retargeter authors (unofficial fork of NVIDIA soma-retargeter)
# SPDX-License-Identifier: Apache-2.0

"""SMPL-family -> SOMA fitting stage (requires the ``fit`` extra: torch, smplx, py-soma-x).

A :class:`FitContext` holds the per-(body_model, gender) stack — the smplx
forward model, the SOMA layer, and one reusable ``PoseInversion`` — and turns
axis-angle body-model motion into absolute local rotations on the SOMA rig.

Everything is constructed and fitted under a CPU default torch device:
``smplx.create`` and soma's index-tensor factories break when the process-wide
default device is CUDA (which Genesis sets on ``gs.init(backend=gs.gpu)``).
Tensors that must live on the GPU are placed there explicitly.
"""

from pathlib import Path

import numpy as np

# AMASS-style axis-angle layout per body model: name -> (total_dim, forward-kwarg slices)
_POSE_LAYOUT = {
    "smpl": (72, {"global_orient": (0, 3), "body_pose": (3, 72)}),
    "smplh": (156, {"global_orient": (0, 3), "body_pose": (3, 66),
                    "left_hand_pose": (66, 111), "right_hand_pose": (111, 156)}),
    "smplx": (165, {"global_orient": (0, 3), "body_pose": (3, 66),
                    "jaw_pose": (66, 69), "leye_pose": (69, 72), "reye_pose": (72, 75),
                    "left_hand_pose": (75, 120), "right_hand_pose": (120, 165)}),
}


# Genuine fits are near-orthonormal (smplx ~2e-4, smplh ~1.3e-2 from its coarser
# topology transfer); a torn matrix from a GPU race is ~O(0.5-1). This threshold
# sits well above the former and well below the latter.
_ORTHO_TOL = 0.1
_MAX_FIT_ATTEMPTS = 3   # bounded retries to heal transient GPU stream races


def pose_dim(body_model: str) -> int:
    """Total axis-angle pose width for a body model (72 / 156 / 165)."""
    try:
        return _POSE_LAYOUT[body_model][0]
    except KeyError:
        raise ValueError(f"Unknown body model {body_model!r}; expected one of {sorted(_POSE_LAYOUT)}") from None


def _orthonormality_error(rotations_np: np.ndarray) -> float:
    """max ||R Rᵀ − I|| over all rotation matrices (0 for perfect rotations)."""
    rrt = np.einsum("...ab,...cb->...ac", rotations_np, rotations_np)
    return float(np.abs(rrt - np.eye(3, dtype=rotations_np.dtype)).max())


class FitContext:
    """Per-(body_model, gender) fit stack: body model + SOMALayer + PoseInversion.

    Args:
        body_model: "smpl" | "smplh" | "smplx".
        model_path: Resolved body-model file for this gender.
        gender: Gender passed to ``smplx.create``.
        device: Torch device for the forward pass and the fit.
        num_betas: Shape-space width for both the forward model and the SOMA
            identity model (None = smplx default, 10).
        flat_hand_mean: Passed to ``smplx.create``.
        soma_data_root: SOMA-X assets dir override (None = auto-download;
            SMPL-H automatically routes through the SMPLH data_root shim).
        reuse_pose_inversion: Reuse one ``PoseInversion`` across clips
            (per-clip cost is then only ``prepare_identity``). Unsafe when
            another soma context coexists — its cached GPU buffers get
            corrupted — so callers default this to False. The shared low-LOD
            layer already removes the expensive per-clip rebuild, so a fresh
            instance per clip stays cheap.
        enable_procedural_transforms: Passed to ``SOMALayer``. True (py-soma-x's
            default) keeps the expanded twist-joint rig, which needs procedural
            assets the public SOMA-X snapshot omits; False fits against the
            public 78-joint rig. Either way the layer exposes the same 78 public
            joints, so ``joint_names`` and everything downstream are unchanged —
            but the fitted rotations differ, since procedural mode refines the
            skin the fit is matched against.
    """

    def __init__(self, body_model: str, model_path: Path, gender: str, device,
                 num_betas: int | None, flat_hand_mean: bool,
                 soma_data_root, reuse_pose_inversion: bool,
                 enable_procedural_transforms: bool = True):
        import smplx
        import torch
        from soma.pose_inversion import PoseInversion
        from soma.soma import SOMALayer

        from soma_retargeter.retargeting.assets import (
            check_procedural_transform_definition,
            ensure_smplh_data_root,
            resolve_soma_data_root,
        )

        if body_model not in _POSE_LAYOUT:
            raise ValueError(f"Unknown body model {body_model!r}; expected one of {sorted(_POSE_LAYOUT)}")

        self.body_model = body_model
        self._device = torch.device(device)
        self._reuse_pose_inversion = reuse_pose_inversion
        self._PoseInversion = PoseInversion
        self.procedural_transforms_enabled = enable_procedural_transforms

        model_path = Path(model_path)
        if body_model == "smplh":
            soma_data_root = ensure_smplh_data_root(soma_data_root)
        if enable_procedural_transforms:
            # Resolve first so the check reads the directory the layer will use.
            soma_data_root = resolve_soma_data_root(soma_data_root)
            check_procedural_transform_definition(soma_data_root)

        identity_kwargs = {"model_path": str(model_path), "gender": gender}
        model_kwargs = {}
        if num_betas is not None:
            identity_kwargs["num_betas"] = num_betas
            model_kwargs["num_betas"] = num_betas

        with torch.device("cpu"):
            self._body_model = smplx.create(
                model_path=str(model_path),
                model_type=body_model,
                gender=gender,
                ext=model_path.suffix.lstrip("."),
                use_pca=False,
                flat_hand_mean=flat_hand_mean,
                batch_size=1,
                **model_kwargs,
            )
            # low_lod=True so PoseInversion reuses this layer directly instead
            # of silently rebuilding a second, low-LOD SOMALayer per instance.
            # (Were it to rebuild, py-soma-x copies the procedural setting off
            # this layer, so the flag below holds for the fit either way.)
            self._soma_layer = SOMALayer(
                data_root=str(soma_data_root) if soma_data_root is not None else None,
                low_lod=True,
                identity_model_type=body_model,
                device=self._device,
                mode="warp",
                identity_model_kwargs=identity_kwargs,
                enable_procedural_transforms=enable_procedural_transforms,
            )
        self._body_model = self._body_model.to(self._device)

        self.joint_names = list(self._soma_layer.rig_data["joint_names"])
        self.hips_index = self.joint_names.index("Hips")
        self.num_betas = int(self._body_model.num_betas)

        self._warp_device = None
        if self._device.type == "cuda":
            import warp as wp

            self._warp_device = wp.get_device(str(self._device))

        self._inv = None
        if reuse_pose_inversion:
            with torch.device("cpu"):
                self._inv = PoseInversion(self._soma_layer, low_lod=True)

    def _forward_vertices(self, poses_t, trans_t, betas_t, chunk: int):
        """Chunked body-model forward -> (B, V, 3) vertices (bounds peak VRAM)."""
        import torch

        _, slices = _POSE_LAYOUT[self.body_model]
        B = poses_t.shape[0]
        out = []
        extra = {}
        if self.body_model == "smplx":
            n_expr = getattr(self._body_model, "num_expression_coeffs", 10)
            extra["expression"] = torch.zeros(1, n_expr, dtype=poses_t.dtype, device=poses_t.device)
        for s in range(0, B, chunk):
            e = min(s + chunk, B)
            kwargs = {name: poses_t[s:e, lo:hi] for name, (lo, hi) in slices.items()}
            kwargs.update({k: v.expand(e - s, -1) for k, v in extra.items()})
            result = self._body_model(
                transl=trans_t[s:e],
                betas=betas_t.expand(e - s, -1),
                **kwargs,
            )
            out.append(result.vertices)
        return torch.cat(out, dim=0)

    def fit_clip(self, poses: np.ndarray, trans: np.ndarray, betas: np.ndarray | None,
                 *, fit_chunk: int, forward_chunk: int, pad_chunks: bool,
                 fit_kwargs: dict | None) -> tuple[np.ndarray, np.ndarray]:
        """Fit one clip -> (rotations (T, J, 3, 3), root translation (T, 3)), Z-up, metres.

        Each chunk's result is copied straight into preallocated numpy output,
        so peak GPU memory is one chunk regardless of clip length. With
        ``pad_chunks`` the tail chunk repeats the last frame up to
        ``fit_chunk`` so every warp/torch launch sees the same shapes.
        """
        import torch

        T = poses.shape[0]
        dim = pose_dim(self.body_model)
        if poses.shape != (T, dim):
            raise ValueError(f"poses: expected shape ({T}, {dim}) for {self.body_model}, got {poses.shape}")
        if trans.shape != (T, 3):
            raise ValueError(f"trans: expected shape ({T}, 3), got {trans.shape}")

        betas_t = torch.zeros(1, self.num_betas, dtype=torch.float32, device=self._device)
        if betas is not None:
            betas_in = np.asarray(betas, dtype=np.float32).reshape(-1)
            n = min(self.num_betas, betas_in.size)
            betas_t[0, :n] = torch.from_numpy(betas_in[:n]).to(self._device)

        poses_t = torch.from_numpy(np.ascontiguousarray(poses, dtype=np.float32)).to(self._device)
        trans_t = torch.from_numpy(np.ascontiguousarray(trans, dtype=np.float32)).to(self._device)

        # The fit interleaves torch ops with warp kernels sharing memory
        # zero-copy across the two engines' CUDA streams. Stream discipline
        # (below) removes the common races, but a rare transient can still tear
        # a rotation matrix; _MAX_FIT_ATTEMPTS re-runs after a full device
        # barrier. A genuinely bad fit fails the orthonormality check every
        # attempt and raises, so this heals races without masking real errors.
        last_err = 0.0
        for attempt in range(_MAX_FIT_ATTEMPTS):
            rotations_np, trans_np = self._fit_once(
                poses_t, trans_t, betas_t, T, fit_chunk, forward_chunk, pad_chunks, fit_kwargs or {})
            last_err = _orthonormality_error(rotations_np)
            if last_err <= _ORTHO_TOL:
                return rotations_np, trans_np

        raise RuntimeError(
            f"SOMA fit produced non-orthonormal rotations after {_MAX_FIT_ATTEMPTS} attempts "
            f"(max ||RRᵀ − I|| = {last_err:.3e} > {_ORTHO_TOL:.0e}). This usually indicates a "
            "corrupt input pose or a persistent GPU issue, not a transient race.")

    def _fit_once(self, poses_t, trans_t, betas_t, T, fit_chunk, forward_chunk, pad_chunks, fit_kwargs):
        import torch
        import warp as wp

        rotations_np = None
        trans_np = np.empty((T, 3), dtype=np.float32)

        # Fence all prior GPU work — the fit's torch/warp kernels share the
        # warp memory pool with anything else on the device (a host Genesis
        # sim, the downstream Newton IK, other soma users). A device-wide
        # barrier before and after keeps those from overlapping and tearing
        # the fit's zero-copy warp<->torch buffers. (Binding warp to torch's
        # stream instead of a hard barrier collides with Genesis's own CUDA
        # context, so a barrier is used here.)
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

        with torch.device("cpu"):
            inv = self._inv
            if inv is None:
                inv = self._PoseInversion(self._soma_layer, low_lod=True)
            inv.prepare_identity(betas_t)

            for s in range(0, T, fit_chunk):
                e = min(s + fit_chunk, T)
                chunk_poses, chunk_trans = poses_t[s:e], trans_t[s:e]
                if pad_chunks and e - s < fit_chunk:
                    pad = fit_chunk - (e - s)
                    chunk_poses = torch.cat([chunk_poses, chunk_poses[-1:].expand(pad, -1)], dim=0)
                    chunk_trans = torch.cat([chunk_trans, chunk_trans[-1:].expand(pad, -1)], dim=0)
                with torch.no_grad():
                    verts = self._forward_vertices(chunk_poses, chunk_trans, betas_t, forward_chunk)
                result = inv.fit(verts, **fit_kwargs)
                rot = result["rotations"].detach()
                tr = result["root_translation"].detach()
                if rotations_np is None:
                    rotations_np = np.empty((T, rot.shape[1], 3, 3), dtype=np.float32)
                if self._device.type == "cuda":
                    torch.cuda.synchronize(self._device)
                rotations_np[s:e] = rot[: e - s].cpu().numpy()
                trans_np[s:e] = tr[: e - s].cpu().numpy()

        if self._device.type == "cuda":
            wp.synchronize_device(self._warp_device)

        return rotations_np, trans_np
