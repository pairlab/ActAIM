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
from new_scripts import utils

import pytorch3d.transforms as transforms
import math
import pdb


class PerActDecoder:

    def __init__(self,
                 device,
                 batch_size: int,
                 loss_type: str,   # which types of loss are using, ce or mse
                 voxel_size: int,  # N voxels per side (size: N*N*N)
                 num_rotation_classes=72,  # 5 degree increments (5*72=360) for each of the 3-axis
                 trans_loss_weight: float = 1.0,
                 rot_loss_weight: float = 1.0,
                 grip_loss_weight: float = 1.0,
                 collision_loss_weight: float = 1.0,
                 task_loss_weight: float = 20.0,
                 ):
        super(PerActDecoder, self).__init__()
        self._loss_type = loss_type
        self._batch_size = batch_size
        self._voxel_size = voxel_size
        self._num_rotation_classes = num_rotation_classes

        self.device = device

        # TODO give an larger operation bounding box!
        # voxel_scale = 1.5
        # [[-1.5, 1.5], [-1., 1.], [0, 2]]
        # self.voxel_bnd = torch.tensor([[-1.2, 0.4], [-0.8, 0.8], [0, 1.6]]).to(device)
        self.voxel_bnd = torch.tensor([[-0.9, 0.3], [-0.6, 0.6], [0.0, 1.2]]).to(device)
        # np.array([[-0.9, 0.3], [-0.6, 0.6], [0.0, 1.2]])

        self.voxel_bnd_len = (self.voxel_bnd[:, 1] - self.voxel_bnd[:, 0])

        self.lower_bnd = self.voxel_bnd[:, 0]
        self.upper_bnd = self.voxel_bnd[:, 1] - (self.voxel_bnd_len / self._voxel_size)

        # CE loss
        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        self._mse_loss = nn.MSELoss(reduce=False, reduction='mean')
        self.cos_sim_loss = nn.CosineEmbeddingLoss(reduce=False, reduction='mean')

        # loss weight
        self._trans_loss_weight = trans_loss_weight
        self._rot_loss_weight = rot_loss_weight
        self._grip_loss_weight = grip_loss_weight
        self._collision_loss_weight = collision_loss_weight
        self._task_loss_weight = task_loss_weight

        # one-hot zero tensors
        self._action_trans_one_hot_zeros = torch.zeros((self._batch_size,
                                                        1,
                                                        self._voxel_size,
                                                        self._voxel_size,
                                                        self._voxel_size),
                                                       dtype=int,
                                                       device=device)
        self._action_rot_x_one_hot_zeros = torch.zeros((self._batch_size,
                                                        self._num_rotation_classes),
                                                       dtype=int,
                                                       device=device)
        self._action_rot_y_one_hot_zeros = torch.zeros((self._batch_size,
                                                        self._num_rotation_classes),
                                                       dtype=int,
                                                       device=device)
        self._action_rot_z_one_hot_zeros = torch.zeros((self._batch_size,
                                                        self._num_rotation_classes),
                                                       dtype=int,
                                                       device=device)
        self._action_grip_one_hot_zeros = torch.zeros((self._batch_size,
                                                       2),
                                                      dtype=int,
                                                      device=device)
        self._action_grip_one_hot_zeros = torch.zeros((self._batch_size,
                                                       2),
                                                      dtype=int,
                                                      device=device)

        # for 2D image
        # TODO maybe hyper params

        self._num_cam = 5
        self._img_width = 320
        self._img_height = 320
        self._action_2d_trans_one_hot_zeros = torch.zeros((self._batch_size,
                                                        self._num_cam,
                                                        self._img_width,
                                                        self._img_height),
                                                       dtype=int,
                                                       device=device)
        # fpr rvt generate novel view
        self.renderer = None
    def get_voxel_bnd_size(self):
        return self.voxel_bnd, self.voxel_bnd_len

    def set_device(self, device):
        self.voxel_bnd = self.voxel_bnd.to(device)
        self.voxel_bnd_len = (self.voxel_bnd[:, 1] - self.voxel_bnd[:, 0])

        self._cross_entropy_loss = self._cross_entropy_loss.to(device)
        self._action_trans_one_hot_zeros = self._action_trans_one_hot_zeros.to(device)
        self._action_rot_x_one_hot_zeros = self._action_rot_x_one_hot_zeros.to(device)
        self._action_rot_y_one_hot_zeros = self._action_rot_y_one_hot_zeros.to(device)
        self._action_rot_z_one_hot_zeros = self._action_rot_z_one_hot_zeros.to(device)
        self._action_grip_one_hot_zeros = self._action_grip_one_hot_zeros.to(device)
        self._action_2d_trans_one_hot_zeros = self._action_2d_trans_one_hot_zeros.to(device)
        self.device = device

    def set_renderer(self, renderer):
        self.renderer = renderer


    def compute_gt_y(self, pos, rotations, gripper_open_close):
        bs = pos.shape[0]

        rot_y  = torch.Tensor().to(self.device)
        pos_y = torch.Tensor().to(self.device)

        for b in range(bs):
            pos_i = pos[b]
            rotation_i = rotations[b]

            voxel_bnd_min = self.voxel_bnd[:, 0]
            pos_i_scale = (pos_i - voxel_bnd_min) / self.voxel_bnd_len

            rotation_i = rotation_i[0]

            angle_xyz = transforms.quaternion_to_axis_angle(rotation_i)
            angle_xyz = angle_xyz % (2.0 * math.pi)
            angle_xyz_scale = angle_xyz / (math.pi * 2.0)

            pos_y = torch.cat((pos_y, pos_i_scale.unsqueeze(0)), 0)
            rot_y = torch.cat((rot_y, angle_xyz_scale.unsqueeze(0)), 0)

        return pos_y, rot_y, gripper_open_close.unsqueeze(-1)


    def compute_gt_grid(self, pos, rotations, gripper_open_close):
        bs = pos.shape[0]  # batch size
        action_trans = self._action_trans_one_hot_zeros.clone()
        action_rot_x = self._action_rot_x_one_hot_zeros.clone()
        action_rot_y = self._action_rot_x_one_hot_zeros.clone()
        action_rot_z = self._action_rot_x_one_hot_zeros.clone()
        action_grip = self._action_grip_one_hot_zeros.clone()

        gt_scale = 1.0

        for b in range(bs):
            pos_i = pos[b]
            # TODO different device error
            # pos_i = torch.max(torch.min(pos_i, self.upper_bnd), self.lower_bnd)

            rotation_i = rotations[b]
            grip_i = gripper_open_close[b]

            voxel_bnd_min = self.voxel_bnd[:, 0]
            pos_i_scale = (pos_i - voxel_bnd_min) / self.voxel_bnd_len

            pos_grid = (pos_i_scale * self._voxel_size).int().cpu().detach().tolist()
            action_trans[b, :, pos_grid[0], pos_grid[1], pos_grid[2]] = gt_scale

            # TODO Need to think of how to convert quaternion to rotation space
            # TODO Now, only takes the first quaternion
            rotation_i = rotation_i[0]

            angle_xyz = transforms.quaternion_to_axis_angle(rotation_i)
            angle_xyz = angle_xyz % (2.0 * math.pi)
            angle_xyz_scale = angle_xyz / (math.pi * 2.0)
            angle_xyz_grid = (angle_xyz_scale * self._num_rotation_classes).int().cpu().detach().tolist()

            action_rot_x[b, angle_xyz_grid[0]] = gt_scale
            action_rot_y[b, angle_xyz_grid[1]] = gt_scale
            action_rot_z[b, angle_xyz_grid[2]] = gt_scale

            grip_i = int(grip_i.detach().cpu())
            if grip_i > 0:
                action_grip[b, 1] = gt_scale
            else:
                action_grip[b, 0] = gt_scale

        return action_trans.float(), action_rot_x.float(), action_rot_y.float(), action_rot_z.float(), action_grip.float()

    def compute_pos_xy_rvt(self, pos):
        bs = pos.shape[0]
        pos = pos.unsqueeze(1)
        pt_img = self.renderer.get_pt_loc_on_img(
            pos, fix_cam=True, dyn_cam_info=None
        )
        pt_img = pt_img.view(bs, 5, 2).int()
        return pt_img

    def compute_pos_xy(self, pos, view_matrices):
        num_cam = view_matrices.shape[0]
        pos_xy_cam = torch.Tensor().to(self.device)
        # camera intrinsics params
        fu = 2.0
        fv = 2.0
        for i in range(num_cam):
            vin = view_matrices[i]
            add_one = torch.ones(1).to(self.device)
            pos_one = torch.cat((pos, add_one)).unsqueeze(0)

            big_mat_inv = torch.matmul(pos_one, vin)
            proj_u_inv = big_mat_inv[:, 0]
            proj_v_inv = big_mat_inv[:, 1]
            depth_img_inv = big_mat_inv[:, 2]

            u_range_inv = proj_u_inv / (fu * depth_img_inv)
            v_range_inv = proj_v_inv / (fv * depth_img_inv)

            x_inv = torch.round(self._img_width * (-u_range_inv)) + self._img_width * 0.5
            y_inv = torch.round(self._img_height * v_range_inv) + self._img_height * 0.5

            pos_xy_i = torch.cat((x_inv, y_inv))
            pos_xy_cam = torch.cat((pos_xy_cam, pos_xy_i.unsqueeze(0)), dim=0)

        return pos_xy_cam


    def compute_gt_grid_rvt(self, pos, rotations, gripper_open_close, view_matrices):
        bs = pos.shape[0]  # batch size
        action_trans_2d = self._action_2d_trans_one_hot_zeros.clone()
        action_rot_x = self._action_rot_x_one_hot_zeros.clone()
        action_rot_y = self._action_rot_x_one_hot_zeros.clone()
        action_rot_z = self._action_rot_x_one_hot_zeros.clone()
        action_grip = self._action_grip_one_hot_zeros.clone()

        gt_scale = 1.0

        # if its sequence data
        bs_view_matrices = len(view_matrices)
        is_seq = bs > bs_view_matrices

        # transfer pos to pixel loc
        pos = self.compute_pos_xy_rvt(pos)

        pos = pos.reshape(-1, 2)

        hm = utils.generate_hm_from_pt(pos, res=(self._img_height, self._img_width), sigma=1.5, thres_sigma_times=3)
        hm = hm.reshape(bs, self._num_cam, self._img_height, self._img_width)
        action_trans_2d = hm.clone()

        for b in range(bs):
            # pos_i = pos[b]
            # TODO different device error
            # pos_i = torch.max(torch.min(pos_i, self.upper_bnd), self.lower_bnd)

            if is_seq:
                seq_horizon = bs / bs_view_matrices
                view_matrices_bs_index = int(b // seq_horizon)
                view_matrices_i = view_matrices[view_matrices_bs_index]
            else:
                view_matrices_i = view_matrices[b]

            # pos_xy = pos_i

            rotation_i = rotations[b]
            grip_i = gripper_open_close[b]

            # for cam_i in range(self._num_cam):
            #     pos_xy_grid = pos_xy[cam_i].int().cpu().detach().tolist()
            #     # TODO not sure whether the grid is inside bound
            #     assert pos_xy_grid[0] >= 0 and pos_xy_grid[0] < self._img_width
            #     assert pos_xy_grid[1] >= 0 and pos_xy_grid[1] < self._img_height

            #     action_trans_2d[b, cam_i, pos_xy_grid[0], pos_xy_grid[1]] = gt_scale

            rotation_i = rotation_i[0]

            angle_xyz = transforms.quaternion_to_axis_angle(rotation_i)
            angle_xyz = angle_xyz % (2.0 * math.pi)
            angle_xyz_scale = angle_xyz / (math.pi * 2.0)
            angle_xyz_grid = (angle_xyz_scale * self._num_rotation_classes).int().cpu().detach().tolist()

            action_rot_x[b, angle_xyz_grid[0]] = gt_scale
            action_rot_y[b, angle_xyz_grid[1]] = gt_scale
            action_rot_z[b, angle_xyz_grid[2]] = gt_scale

            grip_i = int(grip_i.detach().cpu())
            if grip_i > 0:
                action_grip[b, 1] = gt_scale
            else:
                action_grip[b, 0] = gt_scale

        return action_trans_2d.float(), action_rot_x.float(), action_rot_y.float(), action_rot_z.float(), action_grip.float()


    def compute_gt_grid_2d(self, pos, rotations, gripper_open_close, view_matrices):
        bs = pos.shape[0]  # batch size
        action_trans_2d = self._action_2d_trans_one_hot_zeros.clone()
        action_rot_x = self._action_rot_x_one_hot_zeros.clone()
        action_rot_y = self._action_rot_x_one_hot_zeros.clone()
        action_rot_z = self._action_rot_x_one_hot_zeros.clone()
        action_grip = self._action_grip_one_hot_zeros.clone()

        gt_scale = 1.0

        # if its sequence data
        bs_view_matrices = len(view_matrices)
        is_seq = bs > bs_view_matrices


        for b in range(bs):
            pos_i = pos[b]
            # TODO different device error
            # pos_i = torch.max(torch.min(pos_i, self.upper_bnd), self.lower_bnd)

            if is_seq:
                seq_horizon = bs / bs_view_matrices
                view_matrices_bs_index = int(b // seq_horizon)
                view_matrices_i = view_matrices[view_matrices_bs_index]
            else:
                view_matrices_i = view_matrices[b]
            pos_xy = self.compute_pos_xy(pos_i, view_matrices_i)

            rotation_i = rotations[b]
            grip_i = gripper_open_close[b]

            for cam_i in range(self._num_cam):
                pos_xy_grid = pos_xy[cam_i].int().cpu().detach().tolist()
                # TODO not sure whether the grid is inside bound
                assert pos_xy_grid[0] >= 0 and pos_xy_grid[0] < self._img_width
                assert pos_xy_grid[1] >= 0 and pos_xy_grid[1] < self._img_height

                action_trans_2d[b, cam_i, pos_xy_grid[0], pos_xy_grid[1]] = gt_scale

            # pos_grid = (pos_i_scale * self._voxel_size).int().cpu().detach().tolist()

            rotation_i = rotation_i[0]

            angle_xyz = transforms.quaternion_to_axis_angle(rotation_i)
            angle_xyz = angle_xyz % (2.0 * math.pi)
            angle_xyz_scale = angle_xyz / (math.pi * 2.0)
            angle_xyz_grid = (angle_xyz_scale * self._num_rotation_classes).int().cpu().detach().tolist()

            action_rot_x[b, angle_xyz_grid[0]] = gt_scale
            action_rot_y[b, angle_xyz_grid[1]] = gt_scale
            action_rot_z[b, angle_xyz_grid[2]] = gt_scale

            grip_i = int(grip_i.detach().cpu())
            if grip_i > 0:
                action_grip[b, 1] = gt_scale
            else:
                action_grip[b, 0] = gt_scale

        return action_trans_2d.float(), action_rot_x.float(), action_rot_y.float(), action_rot_z.float(), action_grip.float()


    def _celoss(self, pred, labels):
        # TODO CELoss does not work well
        # TODO Use traditional MSE loss instead

        # return self._mse_loss(pred, labels).mean(dim=1)
        return self._cross_entropy_loss(pred, labels.argmax(-1))

    def _mseloss(self, pred, labels):
        return self._mse_loss(pred, labels).mean(dim=1)


    def compute_loss(self, q_trans, q_rot_grip, q_collision, q_task, pos, rotations, gripper_open_close, task):
        bs = q_trans.shape[0]
        target = torch.ones(bs).to(self.device)
        # task_pred_loss = self.cos_sim_loss(q_task, task.detach(), target)
        task_pred_loss = self._mseloss(q_task, task.detach())
        task_pred_loss = self._task_loss_weight * task_pred_loss.mean()

        if len(rotations.shape) > 2:
            # seq action prediction
            seq_len = rotations.shape[1]

            total_loss = 0
            q_trans_loss_mean = 0
            q_rot_loss_mean = 0
            q_grip_loss_mean = 0

            q_rotations = q_rot_grip[:, :self._num_rotation_classes * 3]

            for step_i in range(seq_len):
                curr_grip_ind = self._num_rotation_classes * 3 + 2 * step_i
                step_i_grip = q_rot_grip[:, curr_grip_ind : curr_grip_ind + 2]
                q_rot_grip_i = torch.cat((q_rotations, step_i_grip), 1)

                if q_trans.shape[-2] == self._img_width and q_trans.shape[-1] == self._img_height:
                    q_trans_i = q_trans[:, :, step_i, :, :]
                    pos_i = pos[:, step_i, :]
                else:
                    q_trans_i = q_trans[:, step_i, :].unsqueeze(1)
                    pos_i = pos[:, step_i, :].unsqueeze(1)

                total_loss_i, q_trans_loss_mean_i, q_rot_loss_mean_i, q_grip_loss_mean_i \
                    = self.compute_loss_step(q_trans_i, q_rot_grip_i,
                                             q_collision[:, :], pos_i,
                                             rotations[:, step_i, :], gripper_open_close[:, step_i, :])

                total_loss += total_loss_i
                q_trans_loss_mean += q_trans_loss_mean_i
                q_rot_loss_mean += q_rot_loss_mean_i
                q_grip_loss_mean += q_grip_loss_mean_i

        else:
            # tuple actio prediction
            # q_trans               [bs, 1, 40, 40, 40]
            # q_rot_grip            [bs, 218]
            # q_collision           [bs, 2]
            # pos                   [bs, 1, 40, 40, 40]
            # rotations             [bs, 218]
            # gripper_open_close    [bs, 2]
            total_loss, q_trans_loss_mean, q_rot_loss_mean, q_grip_loss_mean = self.compute_loss_step(q_trans, q_rot_grip, q_collision, pos, rotations, gripper_open_close)

        return total_loss, q_trans_loss_mean, q_rot_loss_mean, q_grip_loss_mean, task_pred_loss


    def compute_loss_step_ce(self, q_trans, q_rot_grip, q_collision, pos, rotations, gripper_open_close):
        bs = q_trans.shape[0]  # batch size

        # compute gt grid representation
        # gt_action_trans, gt_action_rot_x, gt_action_rot_y, gt_action_rot_z, gt_action_grip = self.compute_gt_grid(pos, rotations, gripper_open_close)

        gt_action_trans = pos
        gt_action_rot_x = rotations[:, self._num_rotation_classes * 0:self._num_rotation_classes * 1]
        gt_action_rot_y = rotations[:, self._num_rotation_classes * 1:self._num_rotation_classes * 2]
        gt_action_rot_z = rotations[:, self._num_rotation_classes * 2:self._num_rotation_classes * 3]

        gt_action_grip = rotations[:, self._num_rotation_classes * 3:]

        q_trans_loss, q_rot_loss, q_grip_loss, q_collision_loss = 0., 0., 0., 0.

        # translation loss
        if q_trans.shape[-2] == self._img_width and q_trans.shape[-1] == self._img_height:
            q_trans_flat = q_trans.view(bs * self._num_cam, -1)
            gt_action_trans_flat = gt_action_trans.reshape(bs * self._num_cam, -1)
            q_trans_loss_cam = self._celoss(q_trans_flat, gt_action_trans_flat)
            q_trans_loss_cam = q_trans_loss_cam.view(bs, self._num_cam, -1)
            q_trans_loss = torch.sum(q_trans_loss_cam, dim=1).squeeze()

        else:
            q_trans_flat = q_trans.view(bs, -1)
            gt_action_trans_flat = gt_action_trans.view(bs, -1)
            q_trans_loss = self._celoss(q_trans_flat, gt_action_trans_flat)

        # TODO debugging gt and predict
        # gt_trans_ind = torch.argmax(gt_action_trans_flat, dim=1)
        # q_trans_ind = torch.argmax(q_trans_flat, dim=1)
        # print("gt_ind: ", gt_trans_ind)
        # print("predict_ind: ", q_trans_ind)


        # flatten predictions
        q_rot_x_flat = q_rot_grip[:, 0 * self._num_rotation_classes:1 * self._num_rotation_classes]
        q_rot_y_flat = q_rot_grip[:, 1 * self._num_rotation_classes:2 * self._num_rotation_classes]
        q_rot_z_flat = q_rot_grip[:, 2 * self._num_rotation_classes:3 * self._num_rotation_classes]
        q_grip_flat = q_rot_grip[:, 3 * self._num_rotation_classes:]
        # q_collisions_loss = self._celoss(q_collision, q_collision)

        # rot loss
        q_rot_loss += self._celoss(q_rot_x_flat, gt_action_rot_x)
        q_rot_loss += self._celoss(q_rot_y_flat, gt_action_rot_y)
        q_rot_loss += self._celoss(q_rot_z_flat, gt_action_rot_z)

        # TODO debugging gt and predict
        # gt_rot_x_ind = torch.argmax(gt_action_rot_x, dim=1)
        # q_rot_x_indv = torch.argmax(q_rot_x_flat, dim=1)
        # print("gt_rot_x: ", gt_rot_x_ind)
        # print("q_rot_x_ind: ", q_rot_x_ind)

        # gt_rot_y_ind = torch.argmax(gt_action_rot_y, dim=1)
        # q_rot_y_ind = torch.argmax(q_rot_y_flat, dim=1)
        # print("gt_rot_y: ", gt_rot_y_ind)
        # print("q_rot_y_ind: ", q_rot_y_ind)

        # gt_rot_z_ind = torch.argmax(gt_action_rot_z, dim=1)
        # q_rot_z_ind = torch.argmax(q_rot_z_flat, dim=1)
        # print("gt_rot_z: ", gt_rot_z_ind)
        # print("q_rot_z_ind: ", q_rot_z_ind)

        # gripper loss
        q_grip_loss += self._celoss(q_grip_flat, gt_action_grip)

        combined_losses = (q_trans_loss * self._trans_loss_weight) + \
                          (q_rot_loss * self._rot_loss_weight) + \
                          (q_grip_loss * self._grip_loss_weight) + \
                          (q_collision_loss * self._collision_loss_weight)
        total_loss = combined_losses.mean()

        q_trans_loss_mean = (q_trans_loss * self._trans_loss_weight).mean()
        q_rot_loss_mean = (q_rot_loss * self._rot_loss_weight).mean()
        q_grip_loss_mean = (q_grip_loss * self._grip_loss_weight).mean()
        # q_collision_loss_mean = (q_collision_loss * self._collision_loss_weight).mean()

        return total_loss, q_trans_loss_mean, q_rot_loss_mean, q_grip_loss_mean

    def compute_loss_step_mse(self, q_trans, q_rot_grip, q_collision, pos, rotations, gripper_open_close):
        bs = q_trans.shape[0]  # batch size

        q_trans_loss, q_rot_loss, q_grip_loss, q_collision_loss = 0., 0., 0., 0.

        # compute gt grid representation
        # gt_action_trans, gt_action_rot_x, gt_action_rot_y, gt_action_rot_z, gt_action_grip = self.compute_gt_grid(pos, rotations, gripper_open_close)

        gt_action_trans_label = pos

        # gt_action_rot_x = rotations[:, self._num_rotation_classes * 0:self._num_rotation_classes * 1]
        # gt_action_rot_y = rotations[:, self._num_rotation_classes * 1:self._num_rotation_classes * 2]
        # gt_action_rot_z = rotations[:, self._num_rotation_classes * 2:self._num_rotation_classes * 3]

        # gt_action_grip = rotations[:, self._num_rotation_classes * 3:]
        # gt_action_trans_ind = torch.nonzero(gt_action_trans.squeeze())[:, 1:]
        # gt_action_trans_label = ((gt_action_trans_ind / self._voxel_size) - 0.5) * 2.0

        # TODO how to fix this in diffusion model?
        # TODO cannot get noise nonzero

        q_trans_loss = self._mseloss(q_trans, gt_action_trans_label)

        # gt_action_rot_x_ind = torch.nonzero(gt_action_rot_x)[:, 1:]
        # gt_action_rot_y_ind = torch.nonzero(gt_action_rot_y)[:, 1:]
        # gt_action_rot_z_ind = torch.nonzero(gt_action_rot_z)[:, 1:]

        # gt_action_rot_ind = torch.cat((gt_action_rot_x_ind, gt_action_rot_y_ind, gt_action_rot_z_ind), dim=1)
        # gt_action_rot = ((gt_action_rot_ind / self._num_rotation_classes) - 0.5) * 2.0
        # gt_action_grip_ind = torch.nonzero(gt_action_grip.squeeze())[:, 1:]
        # gt_action_grip = (gt_action_grip_ind - 0.5) * 2.0
        # gt_action_rit_grip = torch.cat((gt_action_rot, gt_action_grip), dim=1)

        q_rot = q_rot_grip[:, :3]
        q_grip = q_rot_grip[:, -1:]
        gt_action_rot = rotations[:, :3]
        gt_action_grip = rotations[:, -1:]

        q_rot_loss = self._mseloss(q_rot, gt_action_rot)
        q_grip_loss = self._mseloss(q_grip, gt_action_grip)

        combined_losses = (q_trans_loss * self._trans_loss_weight) + \
                          (q_rot_loss * self._rot_loss_weight) + \
                          (q_grip_loss * self._grip_loss_weight) + \
                          (q_collision_loss * self._collision_loss_weight)
        total_loss = combined_losses.mean()

        q_trans_loss_mean = (q_trans_loss * self._trans_loss_weight).mean()
        q_rot_loss_mean = (q_rot_loss * self._rot_loss_weight).mean()
        q_grip_loss_mean = (q_grip_loss * self._grip_loss_weight).mean()
        # q_collision_loss_mean = (q_collision_loss * self._collision_loss_weight).mean()

        return total_loss, q_trans_loss_mean, q_rot_loss_mean, q_grip_loss_mean


    def compute_loss_step(self, q_trans, q_rot_grip, q_collision, pos, rotations, gripper_open_close):

        if self._loss_type == "ce":
            total_loss, q_trans_loss_mean, q_rot_loss_mean, q_grip_loss_mean = self.compute_loss_step_ce(q_trans,
                                                                                                         q_rot_grip,
                                                                                                         q_collision,
                                                                                                         pos, rotations,
                                                                                                         gripper_open_close)
        elif self._loss_type == "mse":
            total_loss, q_trans_loss_mean, q_rot_loss_mean, q_grip_loss_mean = self.compute_loss_step_mse(q_trans,
                                                                                                         q_rot_grip,
                                                                                                         q_collision,
                                                                                                         pos, rotations,
                                                                                                         gripper_open_close)
        else:
            print("wrong loss type")
            exit()

        return total_loss, q_trans_loss_mean, q_rot_loss_mean, q_grip_loss_mean