# SPDX-FileCopyrightText: Copyright (c) 2026 The eden-soma-retargeter authors (unofficial fork of NVIDIA soma-retargeter)
# SPDX-License-Identifier: Apache-2.0

"""Reusable NewtonPipeline: per-env-count cached model state, IK solver, and CUDA graph.

``NewtonPipeline.execute()`` rebuilds a ``num_envs``-sized Newton model, all IK
objectives, an ``IKSolver`` and a CUDA graph capture on every call, even though
none of that state depends on the motions being retargeted — only on the number
of parallel environments. ``CachedNewtonPipeline`` hoists that setup into a
per-``num_envs`` cache so repeated ``execute()`` calls (the per-clip loop of a
long retargeting run) cost only the joint-state reset and the frame loop.
"""

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import warp as wp

from soma_retargeter.pipelines.newton_pipeline import NewtonPipeline
from soma_retargeter.robotics.csv_animation_buffer import CSVAnimationBuffer


@dataclass
class _EnvResources:
    """IK-solve state valid for one specific number of parallel environments."""

    default_joint_q: wp.array  # (num_envs, joint_coord_count) initial pose
    joint_q: wp.array          # (num_envs, joint_coord_count) working state
    position_objectives: list
    rotation_objectives: list
    smooth_joint_filter_objective: object
    ik_solver: object
    graph: object | None       # captured CUDA graph of one solver step (None on CPU)


class CachedNewtonPipeline(NewtonPipeline):
    """NewtonPipeline whose ``execute()`` reuses per-env-count solver state.

    Behaviourally identical to the base class (same objectives, same solve
    order, same output) except that:

    - the num_envs model / objectives / solver / CUDA graph are built once per
      distinct env count and cached (LRU, ``max_cached_env_counts`` entries),
    - the feet stabilizer is only re-setup when the env count changes
      (it tracks its own ``num_envs``),
    - progress prints are gated behind ``verbose``,
    - an empty input list returns ``[]`` instead of ``None``.
    """

    def __init__(self, skeleton, source_type="soma", robot_type="unitree_g1",
                 retarget_config: dict = None, max_cached_env_counts: int = 2,
                 verbose: bool = False):
        super().__init__(skeleton, source_type=source_type, robot_type=robot_type,
                         retarget_config=retarget_config)
        self.verbose = verbose
        self._max_cached_env_counts = max(1, max_cached_env_counts)
        self._env_resources: OrderedDict[int, _EnvResources] = OrderedDict()

    def release(self):
        """Drop all cached per-env-count solver state (frees GPU buffers/graphs)."""
        self._env_resources.clear()

    def _resources_for(self, num_envs: int) -> _EnvResources:
        res = self._env_resources.get(num_envs)
        if res is not None:
            self._env_resources.move_to_end(num_envs)
            return res

        # Replicates the per-call setup of NewtonPipeline.execute(): the
        # num_envs model exists only to seed default body poses / joint_q and
        # is dropped once the objectives and solver (bound to the 1-env
        # ik_model) are built.
        import newton.ik as ik

        model = self._build_model(num_envs)
        state = model.state()

        (
            position_objectives,
            rotation_objectives,
            joint_limit_objective,
            smooth_joint_filter_objective,
        ) = self._create_ik_objectives(num_envs, model, state)

        active_objectives = [*position_objectives, *rotation_objectives]
        if self.joint_limit_weight > 0.0:
            active_objectives.append(joint_limit_objective)
        if self.smooth_joint_filter_weight > 0.0:
            active_objectives.append(smooth_joint_filter_objective)

        ik_solver = ik.IKSolver(
            model=self.ik_model,
            n_problems=num_envs,
            objectives=active_objectives,
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC)

        default_joint_q = wp.empty(shape=(num_envs, self.ik_model.joint_coord_count))
        wp.copy(default_joint_q, model.joint_q)
        joint_q = wp.clone(default_joint_q)

        ik_solver.reset()

        graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as cap:
                ik_solver.step(joint_q, joint_q, iterations=self.ik_iterations)
            graph = cap.graph

        res = _EnvResources(
            default_joint_q=default_joint_q,
            joint_q=joint_q,
            position_objectives=position_objectives,
            rotation_objectives=rotation_objectives,
            smooth_joint_filter_objective=smooth_joint_filter_objective,
            ik_solver=ik_solver,
            graph=graph)

        self._env_resources[num_envs] = res
        while len(self._env_resources) > self._max_cached_env_counts:
            self._env_resources.popitem(last=False)
        return res

    def execute(self):
        num_envs = len(self.input_targets)
        if num_envs == 0:
            self.retargeted_motions = []
            return []

        # Same weight clamping as the base class.
        self.ik_iterations = max(1, self.ik_iterations)
        self.joint_limit_weight = max(0.0, self.joint_limit_weight)
        self.smooth_joint_filter_weight = max(0.0, self.smooth_joint_filter_weight)

        res = self._resources_for(num_envs)

        if self.post_processing_enabled:
            if self.feet_stabilizer.num_envs != num_envs:
                self.feet_stabilizer.setup_num_envs(num_envs)
            env_feet_tx = np.empty((num_envs, len(self.feet_effector_indices), 7), dtype=np.float32)

        # Per-call reset: joint state back to the model default, solver
        # internals re-initialized. Both are in-place on the arrays the
        # captured graph references, so the graph stays valid.
        wp.copy(res.joint_q, res.default_joint_q)
        res.ik_solver.reset()

        position_objectives = res.position_objectives
        rotation_objectives = res.rotation_objectives
        smooth_joint_filter_objective = res.smooth_joint_filter_objective
        joint_q = res.joint_q

        num_frames_to_remove = self.num_initialization_frames + self.num_stabilization_frames
        joint_q_data = [np.full((len(self.input_targets[i]),), None) for i in range(num_envs)]

        # With no init/stabilization frames there is nothing to ramp over, so the
        # smoothing filter is set to full weight up front (the base pipeline would
        # divide by zero here). Otherwise the per-frame ramp below handles it.
        if num_frames_to_remove == 0:
            smooth_joint_filter_objective.set_weight(self.smooth_joint_filter_weight)

        frames = range(self.max_frames)
        if self.verbose:
            from tqdm import trange
            frames = trange(self.max_frames, desc="[INFO] Retargeting Motions")

        for frame in frames:
            if num_frames_to_remove > 0 and frame <= num_frames_to_remove:
                smooth_joint_filter_objective.set_weight(
                    self.smooth_joint_filter_weight * (frame / float(num_frames_to_remove)))

            for env in range(num_envs):
                if frame > (len(self.input_targets[env]) - 1):
                    continue
                frame_targets = self.input_targets[env][frame]
                for i, target in enumerate(frame_targets):
                    position_objectives[i].set_target_position(env, wp.vec3(*target[0:3]))
                    rotation_objectives[i].set_target_rotation(env, wp.quat(*target[3:7]))

            if res.graph is not None:
                wp.capture_launch(res.graph)
            else:
                res.ik_solver.step(joint_q, joint_q, iterations=self.ik_iterations)

            if self.post_processing_enabled:
                self.feet_stabilizer.reset_state(joint_q)

                for env in range(num_envs):
                    if frame > (len(self.input_targets[env]) - 1):
                        env_feet_tx[env] = np.asarray(self.input_targets[env][-1][self.feet_effector_indices])
                    else:
                        env_feet_tx[env] = np.asarray(self.input_targets[env][frame][self.feet_effector_indices])

                self.feet_stabilizer.solve(env_feet_tx)
                data = self.joint_limit_clamper.apply(self.feet_stabilizer.current_state()).numpy()
            else:
                data = self.joint_limit_clamper.apply(joint_q).numpy()

            for env in range(num_envs):
                if frame > (len(self.input_targets[env]) - 1):
                    continue
                joint_q_data[env][frame] = data[env]

        return [
            CSVAnimationBuffer.create_from_raw_data(joint_q_data[i][num_frames_to_remove:], self.input_sample_rates[i])
            for i in range(num_envs)]
