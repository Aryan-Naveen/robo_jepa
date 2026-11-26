# Copyright (c) 2026 RoboJEPA experiment contributors.
# Licensed under the LICENSE file in the root directory of this source tree.

"""Planner overhead fixes that leave the numerics unchanged (enabled by default, ``--no-fast-planner`` disables).

1. ``WorldModel.step`` keeps the rollout state on the GPU. The stock step ends with ``features.cpu()``, and the
   planner moves the state back on the next call: a ~64 MB pageable copy per step at batch 16, which was about half
   of the step's GPU time.
2. ``mpc_utils.cem`` calls ``gc.collect()`` after every candidate batch (~0.16 s each). Skipping it does not raise
   peak memory.

Both costs grow with the number of candidates, so they would hide the predictor speed-ups we want to compare.
"""

import torch

from app.robo_jepa.embodiment_preprocessing import (get_action_preprocessor, get_state_preprocessor,
                                                    resolve_embodiment_name)
from app.robo_jepa.world_model import WorldModel, WorldModelState, compute_new_pose

_orig_step = WorldModel.step


def step_gpu(self, action, state, use_cache=False, no_grad=True, batch=100, device="cuda",
             pos_embedding_resize_t=None):
    if not (use_cache and no_grad):
        return _orig_step(self, action, state, use_cache=use_cache, no_grad=no_grad, batch=batch, device=device,
                          pos_embedding_resize_t=pos_embedding_resize_t)
    actions_history, feats = state.actions, state.features
    embodiment_ids = state.embodiment_ids
    name = resolve_embodiment_name(embodiment_ids, getattr(self.model, "embodiment_list", None))
    action_preproc, state_preproc = get_action_preprocessor(name), get_state_preprocessor(name)
    if action.shape[-1] != self.model.action_dim:
        action = action_preproc(action, self.model.action_dim)
    if actions_history.shape[-1] != self.model.action_dim:
        actions_history = torch.zeros([actions_history.shape[0], 0, self.model.action_dim], device=device)
    actions = torch.cat([actions_history.to(device), action.to(device)], dim=1)
    poses = state.states
    if poses is not None and poses.shape[-1] != self.model.state_dim:
        poses = state_preproc(poses, self.model.state_dim)
    camera_extrinsics = None
    if state.camera_extrinsics is not None:
        camera_extrinsics = state.camera_extrinsics.repeat(1, feats.shape[1], 1)
    if state.cache is not None:
        cache, position, cache_size = state.cache, state.feature_positions, state.cache.pointer[0]
    else:
        cache, position, cache_size = None, 0, 0
    out, _, _, cache = self.forward(
        feats.to(device), actions, use_cache=True, states=poses, camera_extrinsics=camera_extrinsics, dt=state.dt,
        view_map=state.view_map, cache_size=cache_size, embodiment_ids=embodiment_ids, cache=cache,
        input_position=position, pos_embedding_resize_t=pos_embedding_resize_t)
    position += feats.shape[1]
    features = out[:, -1:]
    if self.model.state_tokens > 0:
        new_pose = compute_new_pose(poses[:, -1:], action, embodiment_ids=embodiment_ids,
                                    embodiment_list=getattr(self.model, "embodiment_list", None))
        if poses.shape[-1] != new_pose.shape[-1]:
            new_pose = state_preproc(new_pose, self.model.state_dim)
        poses = new_pose  # use_cache: only the newest pose is kept
    return out, WorldModelState(actions=actions[:, :0], features=features, cache=cache, feature_positions=position,
                                camera_extrinsics=state.camera_extrinsics, embodiment_ids=embodiment_ids,
                                dt=state.dt, states=poses, view_map=state.view_map)


class _NoGC:
    @staticmethod
    def collect(*args, **kwargs):
        return 0


def apply():
    from app.robo_jepa.mpc import mpc_utils
    WorldModel.step = step_gpu
    mpc_utils.gc = _NoGC()
