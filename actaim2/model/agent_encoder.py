import pdb
from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

# from helpers import utils
# from helpers.utils import visualise_voxel, stack_on_channel
# from voxel.voxel_grid import VoxelGrid
# from voxel.augmentation import apply_se3_augmentation
from einops import rearrange
from helpers.clip.core.clip import build_model, load_clip, tokenize
from typing import Union, List

import transformers
from helpers.optim.lamb import Lamb

from torch.nn.parallel import DistributedDataParallel as DDP

class QFunction(nn.Module):

    def __init__(self,
                 perceiver_encoder: nn.Module,
                 # bounds_offset: float,
                 rotation_resolution: float,
                 device,
                 training):
        super(QFunction, self).__init__()
        self._rotation_resolution = rotation_resolution
        # self._bounds_offset = bounds_offset
        self._qnet = perceiver_encoder.to(device)
        self.device = device

        # TODO add distributed training
        # if training:
        #     self._qnet = DDP(self._qnet, device_ids=[device])

        # load CLIP for encoding language goals during evaluation
        model, _ = load_clip("RN50", jit=False)
        self._clip_rn50 = build_model(model.state_dict())
        self._clip_rn50 = self._clip_rn50.float().to(device)
        self._clip_rn50.eval()
        del model

    def _argmax_3d(self, tensor_orig):
        b, c, d, h, w = tensor_orig.shape  # c will be one
        idxs = tensor_orig.view(b, c, -1).argmax(-1)
        indices = torch.cat([((idxs // h) // d), (idxs // h) % w, idxs % w], 1)
        return indices

    def _argmax_2d(self, tensor_orig):
        b, c, h, w = tensor_orig.shape  # c will be one
        idxs = tensor_orig.view(b, c, -1).argmax(-1)
        indices = torch.cat([(idxs // h) % w, idxs % w], 1)
        return indices


    def choose_highest_action(self, q_trans, q_rot_grip, q_collision):
        if len(q_trans.shape) == 5:
            coords = self._argmax_3d(q_trans)
        else:
            # TODO don't output ind but the whole dist
            coords = q_trans
            # coords = self._argmax_2d(q_trans)
        rot_and_grip_indicies = None
        ignore_collision = None
        if q_rot_grip is not None:
            q_rot = torch.stack(torch.split(
                q_rot_grip[:, :-2],
                int(360 // self._rotation_resolution),
                dim=1), dim=1)
            rot_and_grip_indicies = torch.cat(
                [q_rot[:, 0:1].argmax(-1),
                 q_rot[:, 1:2].argmax(-1),
                 q_rot[:, 2:3].argmax(-1),
                 q_rot_grip[:, -2:].argmax(-1, keepdim=True)], -1)
            ignore_collision = q_collision[:, -2:].argmax(-1, keepdim=True)
        return coords, rot_and_grip_indicies, ignore_collision


    def choose_highest_rotation_grip(self, q_rot_grip, q_collision):
        rot_and_grip_indicies = None
        ignore_collision = None
        if q_rot_grip is not None:
            q_rot = torch.stack(torch.split(
                q_rot_grip[:, :-2],
                int(360 // self._rotation_resolution),
                dim=1), dim=1)
            rot_and_grip_indicies = torch.cat(
                [q_rot[:, 0:1].argmax(-1),
                 q_rot[:, 1:2].argmax(-1),
                 q_rot[:, 2:3].argmax(-1),
                 q_rot_grip[:, -2:].argmax(-1, keepdim=True)], -1)
            ignore_collision = q_collision[:, -2:].argmax(-1, keepdim=True)
        return rot_and_grip_indicies, ignore_collision


    def tokenize(self, x: Union[str, List[str]]):
        x = tokenize(x)
        return x.to(self.device)

    def encode_text(self, x: Union[str, List[str]]):
        x = self.tokenize(x)
        with torch.no_grad():
            text_feat, text_emb = self._clip_rn50.encode_text_with_embeddings(x)

        text_feat = text_feat.detach()
        text_emb = text_emb.detach()
        text_mask = torch.where(x==0, x, 1)  # [1, max_token_len]
        return text_feat, text_emb

    def forward(self, voxel_grid, proprio, lang_goal_emb, lang_token_embs, y_t, task_embed, _ts, traj_step, context_mask, pos_cond,
                bounds=None, prev_bounds=None, prev_layer_voxel_grid=None):

        # forward pass
        (q_trans, \
        q_rot_and_grip, \
        q_ignore_collisions,
        q_task_pred) = self._qnet(voxel_grid,
                                         proprio,
                                         lang_goal_emb,
                                         lang_token_embs,
                                         y_t,
                                         task_embed,
                                         _ts,
                                         traj_step,
                                         context_mask,
                                         pos_cond,
                                         prev_layer_voxel_grid,
                                         bounds,
                                         prev_bounds)

        return q_trans, q_rot_and_grip, q_ignore_collisions, q_task_pred


    def get_num_rot_classes(self):
        return self._qnet.num_rotation_classes

    def get_qnet_type(self):
        return self._qnet.model_name




