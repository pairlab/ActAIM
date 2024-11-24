import torch
import torch.nn as nn
import numpy as np
from math import pi, log
import pytorch3d.transforms as transforms


import pdb
################################################################
class Model_Afford_Transformer(nn.Module):
    def __init__(self, q_func, decoder, betas, n_T, device, drop_prob=0.1, guide_w=0.0):
        super(Model_Afford_Transformer, self).__init__()
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.q_func = q_func
        self.decoder = decoder
        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()
        self.guide_w = guide_w


    def loss_on_batch(self, x_batch, y_batch, task_embed, task_label, traj_step, view_matrices=None):
        obs_input, franka_proprio, lang_prompt = x_batch
        pos, rotation, gripper_open_close = y_batch
        bs = pos.shape[0]

        # encode language if exists
        # TODO add option to encode language
        text_feat, text_emb = self.q_func.encode_text(lang_prompt)
        # q_trans, q_rot_and_grip, q_ignore_collisions = self.q_func(obs_input, franka_proprio, text_feat, text_emb)

        # convert y_label to distribution
        # q_trans shape [B, 1, 40, 40, 40]
        # q_rot_and_grip [B, 218]
        # q_ignore_collisions [B, 2]
        # pos [B, 3]
        # rotation [B, 2, 4]
        # gripper_open_close [B]

        # TODO compute gt grid based on action type
        model_type = self.q_func.get_qnet_type()
        if model_type == "rvt":
            gt_action_trans, gt_action_rot_x, gt_action_rot_y, gt_action_rot_z, gt_action_grip = self.decoder.compute_gt_grid_rvt(
                pos, rotation, gripper_open_close, view_matrices)
            # gt_action_trans, gt_action_rot_x, gt_action_rot_y, gt_action_rot_z, gt_action_grip = self.decoder.compute_gt_grid_2d(
            #     pos, rotation, gripper_open_close, view_matrices)
        else:
            gt_action_trans, gt_action_rot_x, gt_action_rot_y, gt_action_rot_z, gt_action_grip = self.decoder.compute_gt_grid(pos, rotation, gripper_open_close)

        y_pos = gt_action_trans
        y_rot = torch.cat((gt_action_rot_x, gt_action_rot_y, gt_action_rot_z, gt_action_grip), dim=1)
        y_grip = gt_action_grip

        # noise
        _ts = torch.zeros((bs, 1)).to(self.device)

        # dropout context with some probability
        # TODO testing CFG, uneccessary here
        # TODO but maybe could be used for language embedding
        context_mask = torch.bernoulli(torch.zeros(bs) + self.drop_prob).to(self.device)
        pos_cond = pos.view(bs, -1)

        # y_t_pos = torch.zeros_like(y_pos).to(self.device)
        # y_t_rotation = torch.zeros_like(y_rot).to(self.device)
        # y_t_grip = torch.zeros_like(y_grip).to(self.device)

        y_t_pos = torch.zeros((bs, 3)).to(self.device)
        y_t_rotation = torch.zeros((bs, 3)).to(self.device)
        y_t_grip = torch.zeros((bs, 1)).to(self.device)

        y_t = (y_t_pos, y_t_rotation, y_t_grip)

        # TODO modify the model so it takes in
        # use nn model to predict noise
        pred_q_trans, pred_q_rot_and_grip, pred_q_ignore_collisions, pred_task = self.q_func(obs_input, franka_proprio,
                                                                                                    text_feat, text_emb, y_t, task_embed, _ts / self.n_T, traj_step, context_mask, pos_cond)


        # return mse between predicted and true noise
        # self.decoder.compute_loss(pred_q_trans, pred_q_rot_and_grip, pred_q_ignore_collisions, pred_task, y_pos, y_rot, y_grip, task_label)
        # y_gt = (y_pos, y_rot, y_grip)
        # y_pred = (pred_q_trans, pred_q_rot_and_grip, pred_q_ignore_collisions)
        # action_gt = self.decode_action(y_gt)
        # action_pred = self.decode_action(y_pred)

        bc_loss, q_trans_loss, q_rot_loss, q_grip_loss, q_task_loss = self.decoder.compute_loss(pred_q_trans, pred_q_rot_and_grip, pred_q_ignore_collisions, pred_task, y_pos, y_rot, y_grip, task_label)

        # TODO test renderer reverse
        # from new_scripts import utils
        # torch.cuda.empty_cache()
        # selected_pc = utils.get_wpt(self.decoder.renderer, gt_action_trans)
        # predicted_pc = utils.get_wpt(self.decoder.renderer, pred_q_trans)

        return bc_loss, q_trans_loss, q_rot_loss, q_grip_loss, q_task_loss, pred_q_trans


    def loss_on_batch_seq(self, x_batch, y_batch, task_embed, task_label, traj_step, view_matrices=None):
        obs_input, franka_proprio, lang_prompt = x_batch
        pos, rotation, gripper_open_close = y_batch
        bs = pos.shape[0]
        seq_len = pos.shape[1]

        # encode language if exists
        # TODO add option to encode language
        text_feat, text_emb = self.q_func.encode_text(lang_prompt)
        # q_trans, q_rot_and_grip, q_ignore_collisions = self.q_func(obs_input, franka_proprio, text_feat, text_emb)

        # convert y_label to distribution
        # q_trans shape [B, 1, 40, 40, 40]
        # q_rot_and_grip [B, 218]
        # q_ignore_collisions [B, 2]
        # pos [B, 3]
        # rotation [B, 2, 4]
        # gripper_open_close [B]

        pos = pos.view(bs * seq_len, *pos.shape[2:])
        rotation = rotation.view(bs * seq_len, *rotation.shape[2:])
        gripper_open_close = gripper_open_close.view(bs * seq_len, *gripper_open_close.shape[2:])

        model_type = self.q_func.get_qnet_type()
        if model_type == "rvt":
            # compute_gt_grid_2d
            gt_action_trans, gt_action_rot_x, gt_action_rot_y, gt_action_rot_z, gt_action_grip = self.decoder.compute_gt_grid_rvt(
                pos, rotation, gripper_open_close, view_matrices)
        else:
            gt_action_trans, gt_action_rot_x, gt_action_rot_y, gt_action_rot_z, gt_action_grip = self.decoder.compute_gt_grid(
                pos, rotation, gripper_open_close)

        y_pos = gt_action_trans.view(bs, seq_len, *gt_action_trans.shape[1:]).squeeze(2)
        y_rot = torch.cat((gt_action_rot_x, gt_action_rot_y, gt_action_rot_z, gt_action_grip), dim=1)
        y_rot = y_rot.view(bs, -1)
        y_grip = gt_action_grip.view(bs, -1)

        # noise
        _ts = torch.zeros((bs, 1)).to(self.device)

       # dropout context with some probability
        # TODO testing CFG, uneccessary here
        # TODO but maybe could be used for language embedding
        context_mask = torch.bernoulli(torch.zeros(bs) + self.drop_prob).to(self.device)

        y_t_pos = torch.zeros((bs, 3)).to(self.device)
        y_t_rotation = torch.zeros((bs, 3)).to(self.device)
        y_t_grip = torch.zeros((bs, 1)).to(self.device)

        y_t = (y_t_pos, y_t_rotation, y_t_grip)

        # conditional input for predicting rotation
        pos_cond = pos.view(bs, -1)

        # TODO modify the model so it takes in
        # use nn model to predict noise
        pred_q_trans, pred_q_rot_and_grip, pred_q_ignore_collisions, pred_task = self.q_func(obs_input, franka_proprio, text_feat, text_emb, y_t, task_embed, _ts/self.n_T, traj_step, context_mask, pos_cond)

        y_rot = y_rot.view(bs, seq_len, -1)
        y_grip = y_grip.view(bs, seq_len, -1)

        bc_loss, q_trans_loss, q_rot_loss, q_grip_loss, q_task_loss = self.decoder.compute_loss(pred_q_trans, pred_q_rot_and_grip, pred_q_ignore_collisions, pred_task, y_pos, y_rot, y_grip, task_label)

        # return mse between predicted and true noise
        return bc_loss, q_trans_loss, q_rot_loss, q_grip_loss, q_task_loss, pred_q_trans


    def sample_extra(self, x_batch, task_embed, traj_step, extra_steps=4, return_y_trace=False):
        obs_input, franka_proprio, lang_prompt = x_batch

        # compute lang feature
        # used for model
        text_feat, text_emb = self.q_func.encode_text(lang_prompt)

        # also use this as a shortcut to avoid doubling batch when guide_w is zero
        is_zero = False
        if self.guide_w > -1e-3 and self.guide_w < 1e-3:
            is_zero = True

        # how many noisy actions to begin with
        n_sample = obs_input.shape[0]
        voxel_len = obs_input.shape[-1]

        '''
        # sample initial noise, y_0 ~ N(0, 1),
        y_i_pos = torch.randn(n_sample, 1, voxel_len, voxel_len, voxel_len).to(self.device)

        rot_dim = (self.q_func.get_num_rot_classes() * 3) + 2
        y_i_rot = torch.randn(n_sample, rot_dim).to(self.device) # 2 direction for quat
        y_i_collide = torch.randn(n_sample, 2).to(self.device)

        y_i = (y_i_pos, y_i_rot, y_i_collide)

        y_pos = torch.zeros_like(y_i_pos).to(self.device)
        y_rotation = torch.zeros_like(y_i_rot).to(self.device)
        y_grip = torch.zeros_like(y_i_collide).to(self.device)

        y_i = (y_pos, y_rotation, y_grip)
        '''

        y_i_pos = torch.zeros(n_sample, 3).to(self.device)
        # rot_dim = (self.q_func.get_num_rot_classes() * 3) + 2
        y_i_rot = torch.zeros(n_sample, 3).to(self.device)  # 2 direction for quat
        y_i_collide = torch.zeros(n_sample, 1).to(self.device)

        y_i = (y_i_pos, y_i_rot, y_i_collide)

        _ts = torch.zeros((n_sample, 1)).to(self.device)


        if not is_zero:
            if len(obs_input.shape) > 2:
                # repeat obs_input twice, so can use guided diffusion
                obs_input = obs_input.repeat(2, 1, 1, 1)
            else:
                # repeat obs_input twice, so can use guided diffusion
                obs_input = obs_input.repeat(2, 1)
            # half of context will be zero
            context_mask = torch.zeros(obs_input.shape[0]).to(self.device)
            context_mask[n_sample:] = 1.0  # makes second half of batch context free
        else:
            # context_mask = torch.zeros_like(obs_input[:,0]).to(self.device)
            context_mask = torch.zeros(obs_input.shape[0]).to(self.device)

        # run denoising chain
        # for i_dummy in range(self.n_T, 0, -1):


        pseudo_pos_cond = torch.zeros(n_sample, 3).to(self.device)

        # traj_step = torch.zeros(bs).to(self.device)
        q_trans_, q_rot_grip_, q_collide_, q_task_ = self.q_func(obs_input, franka_proprio, text_feat, text_emb, y_i,
                                                        task_embed, _ts, traj_step, context_mask, pseudo_pos_cond)

        model_type = self.q_func.get_qnet_type()
        if model_type == "rvt":
            q_trans = q_trans_
            q_rot_grip = q_rot_grip_
            q_collide = q_collide_
            q_task = q_task_
        else:
            q_cond_pos = self.q_func._argmax_3d(q_trans_)
            q_cond_pos = q_cond_pos / voxel_len
            q_trans, q_rot_grip, q_collide, q_task = self.q_func(obs_input, franka_proprio, text_feat, text_emb, y_i,
                                                        task_embed, _ts, traj_step, context_mask, q_cond_pos)


        y_i = (q_trans, q_rot_grip, q_collide)
        return y_i, q_task


    def sample_extra_seq(self, x_batch, task_embed, extra_steps=4, return_y_trace=False):
        # TODO fix seq_horizon here
        seq_horizon = 4
        obs_input, franka_proprio, lang_prompt = x_batch
        bs = obs_input.shape[0]

        # compute lang feature
        # used for model
        text_feat, text_emb = self.q_func.encode_text(lang_prompt)

        # also use this as a shortcut to avoid doubling batch when guide_w is zero
        is_zero = False
        if self.guide_w > -1e-3 and self.guide_w < 1e-3:
            is_zero = True

        # how many noisy actions to begin with
        n_sample = obs_input.shape[0]
        voxel_len = obs_input.shape[-1]

        y_i_pos = torch.zeros(n_sample, 3).to(self.device)
        # rot_dim = (self.q_func.get_num_rot_classes() * 3) + 2
        y_i_rot = torch.zeros(n_sample, 3).to(self.device)  # 2 direction for quat
        y_i_collide = torch.zeros(n_sample, 1).to(self.device)

        y_i = (y_i_pos, y_i_rot, y_i_collide)

        _ts = torch.zeros((n_sample, 1)).to(self.device)


        if not is_zero:
            if len(obs_input.shape) > 2:
                # repeat obs_input twice, so can use guided diffusion
                obs_input = obs_input.repeat(2, 1, 1, 1)
            else:
                # repeat obs_input twice, so can use guided diffusion
                obs_input = obs_input.repeat(2, 1)
            # half of context will be zero
            context_mask = torch.zeros(obs_input.shape[0]).to(self.device)
            context_mask[n_sample:] = 1.0  # makes second half of batch context free
        else:
            # context_mask = torch.zeros_like(obs_input[:,0]).to(self.device)
            context_mask = torch.zeros(obs_input.shape[0]).to(self.device)

        # run denoising chain
        # for i_dummy in range(self.n_T, 0, -1):

        pseudo_pos_cond = torch.zeros(n_sample, 3 * seq_horizon).to(self.device)

        traj_step = torch.zeros(bs).to(self.device)
        q_trans_, q_rot_grip_, q_collide_, q_task_ = self.q_func(obs_input, franka_proprio, text_feat, text_emb, y_i,
                                                        task_embed, _ts, traj_step, context_mask, pseudo_pos_cond)

        q_cond_pos = self.q_func._argmax_3d(q_trans_)
        q_cond_pos = q_cond_pos / voxel_len

        q_trans, q_rot_grip, q_collide, q_task = self.q_func(obs_input, franka_proprio, text_feat, text_emb, y_i,
                                                     task_embed, _ts, traj_step, context_mask, q_cond_pos)


        model_type = self.q_func.get_qnet_type()
        is_rvt = True if model_type == "rvt" else False
        if is_rvt:
            q_trans = q_trans.permute(0, 2, 1, 3, 4)
        else:
            # q_rot_grip = q_rot_grip.reshape(n_sample, seq_horizon * rot_dim)
            # q_collide = q_collide.reshape(n_sample, seq_horizon * 2)
            q_trans = q_trans.reshape(n_sample, seq_horizon, voxel_len, voxel_len, voxel_len)

        y_i = (q_trans, q_rot_grip, q_collide)
        return y_i

    def decode_voxel(self, trans):
        voxel_bnd = self.decoder.voxel_bnd
        voxel_bnd_len = self.decoder.voxel_bnd_len
        voxel_size = self.decoder._voxel_size

        voxel_bnd_min = voxel_bnd[:, 0]
        voxel_bnd_min = voxel_bnd_min.unsqueeze(0).repeat(bs, 1)


    def decode_action_robot(self, y):
        # just for real robot decoding
        y_pos, y_rot, y_collide = y
        seq_horizon = 2

        rot_num_classes = self.decoder._num_rotation_classes
        y_rotation = y_rot[:, :rot_num_classes * 3]
        step_grip = y_rot[:, rot_num_classes * 3: rot_num_classes * 3 + 2]
        exit()


    def decode_action(self, y):
        y_pos, y_rot, y_collide = y
        seq_horizon = y_pos.shape[1]
        bs = y_pos.shape[0]

        pdb.set_trace()

        rot_num_classes = self.decoder._num_rotation_classes
        trans_size = y_pos.shape[-1]

        if seq_horizon > 1 and trans_size == self.decoder._voxel_size:
            y_rotation = y_rot[:, :rot_num_classes * 3]
            # y_rot = y_rot.view(bs, seq_horizon, -1)
            # y_collide = y_collide.view(bs, seq_horizon, -1)

            actions = []
            for step_i in range(seq_horizon):
                step_grip = y_rot[:, rot_num_classes * 3 + step_i * 2 : rot_num_classes * 3 + step_i * 2 + 2]
                y_rot_grip_step = torch.cat((y_rotation, step_grip), 1)
                y_i = (y_pos[:, step_i, :].unsqueeze(1), y_rot_grip_step, y_collide)
                actions.append(self.decode_action_step(y_i).unsqueeze(1))

            actions = torch.cat(actions, dim=1)
            return actions

        elif seq_horizon > 1 and trans_size != self.decoder._voxel_size:
            # TODO rvt case
            return self.decode_action_rvt(y)
        else:
            return self.decode_action_step(y)



    def decode_action_rvt_(self, y):
        # action tuple decode
        y_pos, y_rot, y_collide = y
        bs = y_rot.shape[0]
        rot_num_classes = self.decoder._num_rotation_classes

        rot_and_grip_indicies, ignore_collision = self.q_func.choose_highest_rotation_grip(y_rot, y_collide)

        rot_indicies = rot_and_grip_indicies[:, :3]
        rot_num_classes = self.decoder._num_rotation_classes
        rot_scale = torch.ones_like(rot_indicies) * pi * 2.0 / rot_num_classes
        rot = rot_indicies * rot_scale
        quat = transforms.axis_angle_to_quaternion(rot)
        is_grip = rot_and_grip_indicies[:, 3:]
        action = torch.cat((quat, is_grip), dim=-1)

        return (y_pos, action)



    def decode_action_rvt(self, y):
        # TODO
        # this rvt decoder only works for sequence
        y_pos, y_rot, y_collide = y
        seq_horizon = y_pos.shape[1]
        bs = y_rot.shape[0]
        rot_num_classes = self.decoder._num_rotation_classes
        coords_seq =  torch.Tensor().to(y_pos.device)
        actions = []

        # tuple y_pos size torch.Size([1, 5, 320, 320])
        # seq y_pos size torch.Size([1, 4, 5, 320, 320])
        if seq_horizon == 4:
            for step_i in range(seq_horizon):
                y_pos_i = y_pos[:, step_i, ...]
                y_rot_i = y_rot[:, :rot_num_classes * 3]
                grip_i = y_rot[:, rot_num_classes * 3 + step_i * 2: rot_num_classes * 3 + step_i * 2 + 2]
                y_rot_grip_i = torch.cat((y_rot_i, grip_i), 1)
                y_i = (y_pos_i, y_rot_grip_i, y_collide)
                action, coords = self.decode_action_step(y_i)
                actions.append(action)
                coords_seq = torch.cat((coords_seq, coords.unsqueeze(1)), 1)
            actions = torch.cat(actions, dim=0)
            return (actions, coords_seq)
        else:
            return self.decode_action_step(y)


    def decode_action_step(self, y):
        y_pos, y_rot, y_collide = y
        bs = y_rot.shape[0]

        coords, rot_and_grip_indicies, ignore_collision = self.q_func.choose_highest_action(y_pos, y_rot, y_collide)
        trans_size = y_pos.shape[-1]
        is_rvt = False if trans_size == self.decoder._voxel_size else True

        voxel_bnd = self.decoder.voxel_bnd
        voxel_bnd_len = self.decoder.voxel_bnd_len
        voxel_size = self.decoder._voxel_size

        voxel_bnd_min = voxel_bnd[:, 0]
        voxel_bnd_min = voxel_bnd_min.unsqueeze(0).repeat(bs, 1)
        grid_size = (voxel_bnd_len / voxel_size).unsqueeze(0).repeat(bs, 1)

        if is_rvt:
            pos = torch.Tensor().to(y_pos.device)
        else:
            pos = voxel_bnd_min + grid_size * coords
        is_grip = rot_and_grip_indicies[:, 3:]

        rot_indicies = rot_and_grip_indicies[:, :3]
        rot_num_classes = self.decoder._num_rotation_classes
        rot_scale = torch.ones_like(voxel_bnd_min) * pi * 2.0 / rot_num_classes
        rot = rot_indicies * rot_scale

        quat = torch.zeros_like(rot_and_grip_indicies).float()

        for b in range(bs):
            rot_i = rot[b, :]
            quat_i = transforms.axis_angle_to_quaternion(rot_i)
            quat[b] = quat_i

        action = torch.cat((pos, quat, is_grip), dim=-1)

        if is_rvt:
            return (action, coords)
        else:
            return action

class Model_Afford_Diffusion(nn.Module):
    def __init__(self, q_func, decoder, betas, n_T, device, drop_prob=0.1, guide_w=0.0):
        super(Model_Afford_Diffusion, self).__init__()
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.q_func = q_func
        self.decoder = decoder
        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()
        self.guide_w = guide_w

    def loss_on_batch(self, x_batch, y_batch, task_embed, task_label, step, view_matrices=None):
        obs_input, franka_proprio, lang_prompt = x_batch
        pos, rotation, gripper_open_close = y_batch
        bs = pos.shape[0]

        # encode language if exists
        # TODO add option to encode language
        text_feat, text_emb = self.q_func.encode_text(lang_prompt)
        # q_trans, q_rot_and_grip, q_ignore_collisions = self.q_func(obs_input, franka_proprio, text_feat, text_emb)

        # convert y_label to distribution
        # q_trans shape [B, 1, 40, 40, 40]
        # q_rot_and_grip [B, 218]
        # q_ignore_collisions [B, 2]

        gt_action_trans, gt_action_rot, gt_action_grip = self.decoder.compute_gt_y(pos, rotation, gripper_open_close)

        y_pos = gt_action_trans
        y_rot = gt_action_rot
        y_grip = gt_action_grip

        # noise
        _ts = torch.randint(1, self.n_T + 1, (bs, 1)).to(self.device)

        # dropout context with some probability
        # TODO testing CFG, uneccessary here
        # TODO but maybe could be used for language embedding
        context_mask = torch.bernoulli(torch.zeros(bs) + self.drop_prob).to(self.device)

        # randomly sample some noise, noise ~ N(0, 1)
        noise_pos = torch.randn_like(y_pos).to(self.device)
        noise_rotation = torch.randn_like(y_rot).to(self.device)
        noise_grip = torch.randn_like(y_grip).to(self.device)

        # add noise to clean target actions
        y_t_pos = self.sqrtab[_ts] * y_pos + self.sqrtmab[_ts] * noise_pos
        y_t_rotation = self.sqrtab[_ts] * y_rot + self.sqrtmab[_ts] * noise_rotation
        y_t_grip = self.sqrtab[_ts] * y_grip + self.sqrtmab[_ts] * noise_grip

        y_t = (y_t_pos, y_t_rotation, y_t_grip)

        # TODO modify the model so it takes in
        # use nn model to predict noise
        noise_pred_q_trans, noise_pred_q_rot_and_grip, noise_pred_q_ignore_collisions = self.q_func(obs_input, franka_proprio,
                                                                                                    text_feat, text_emb, y_t, task_embed, _ts / self.n_T, step, context_mask)

        # return mse between predicted and true noise
        return self.decoder.compute_loss(noise_pred_q_trans, noise_pred_q_rot_and_grip, noise_pred_q_ignore_collisions, noise_pos, noise_rotation, noise_grip)

    def loss_on_batch_seq(self, x_batch, y_batch, task_embed, task_label, step, view_matrices=None):
        obs_input, franka_proprio, lang_prompt = x_batch
        pos, rotation, gripper_open_close = y_batch
        bs = pos.shape[0]
        seq_len = pos.shape[1]

        # encode language if exists
        # TODO add option to encode language
        text_feat, text_emb = self.q_func.encode_text(lang_prompt)
        # q_trans, q_rot_and_grip, q_ignore_collisions = self.q_func(obs_input, franka_proprio, text_feat, text_emb)

        # convert y_label to distribution
        # q_trans shape [B, 1, 40, 40, 40]
        # q_rot_and_grip [B, 218]
        # q_ignore_collisions [B, 2]

        pos = pos.view(bs * seq_len, *pos.shape[2:])
        rotation = rotation.view(bs * seq_len, *rotation.shape[2:])
        gripper_open_close = gripper_open_close.view(bs * seq_len, *gripper_open_close.shape[2:])

        gt_action_trans, gt_action_rot_x, gt_action_rot_y, gt_action_rot_z, gt_action_grip = self.decoder.compute_gt_grid(pos, rotation, gripper_open_close)

        y_pos = gt_action_trans.view(bs, seq_len, *gt_action_trans.shape[1:]).squeeze()
        y_rot = torch.cat((gt_action_rot_x, gt_action_rot_y, gt_action_rot_z, gt_action_grip), dim=1)
        y_rot = y_rot.view(bs, -1)
        y_grip = gt_action_grip.view(bs, -1)

        # noise
        _ts = torch.randint(1, self.n_T + 1, (bs, 1)).to(self.device)

        # dropout context with some probability
        # TODO testing CFG, uneccessary here
        # TODO but maybe could be used for language embedding
        context_mask = torch.bernoulli(torch.zeros(bs) + self.drop_prob).to(self.device)

        # randomly sample some noise, noise ~ N(0, 1)
        noise_pos = torch.randn_like(y_pos).to(self.device)
        noise_rotation = torch.randn_like(y_rot).to(self.device)
        noise_grip = torch.randn_like(y_grip).to(self.device)

        # add noise to clean target actions
        y_t_pos = self.sqrtab[_ts][:, None, None, None] * y_pos + self.sqrtmab[_ts][:, None, None, None] * noise_pos
        y_t_rotation = self.sqrtab[_ts] * y_rot + self.sqrtmab[_ts] * noise_rotation
        y_t_grip = self.sqrtab[_ts] * y_grip + self.sqrtmab[_ts] * noise_grip

        y_t = (y_t_pos, y_t_rotation, y_t_grip)

        # TODO modify the model so it takes in
        # use nn model to predict noise
        noise_pred_q_trans, noise_pred_q_rot_and_grip, noise_pred_q_ignore_collisions = self.q_func(obs_input, franka_proprio,
                                                                                                    text_feat, text_emb, y_t, task_embed, _ts / self.n_T, step, context_mask)

        noise_rotation = noise_rotation.view(bs, seq_len, -1)
        noise_grip = noise_grip.view(bs, seq_len, -1)

        # return mse between predicted and true noise
        return self.decoder.compute_loss(noise_pred_q_trans, noise_pred_q_rot_and_grip, noise_pred_q_ignore_collisions, noise_pos, noise_rotation, noise_grip)

    def sample(self, x_batch, return_y_trace=False, extract_embedding=False):
        # also use this as a shortcut to avoid doubling batch when guide_w is zero
        is_zero = False
        if self.guide_w > -1e-3 and self.guide_w < 1e-3:
            is_zero = True

        # how many noisy actions to begin with
        n_sample = x_batch.shape[0]

        y_shape = (n_sample, self.y_dim)

        # sample initial noise, y_0 ~ N(0, 1),
        y_i = torch.randn(y_shape).to(self.device)

        if not is_zero:
            if len(x_batch.shape) > 2:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1, 1, 1)
            else:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1)

            # half of context will be zero
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)
            context_mask[n_sample:] = 1.0  # makes second half of batch context free
        else:
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)

        if extract_embedding:
            x_embed = self.nn_model.embed_context(x_batch)

        # run denoising chain
        y_i_store = []  # if want to trace how y_i evolved
        for i in range(self.n_T, 0, -1):
            t_is = torch.tensor([i / self.n_T]).to(self.device)
            t_is = t_is.repeat(n_sample, 1)

            if not is_zero:
                # double batch
                y_i = y_i.repeat(2, 1)
                t_is = t_is.repeat(2, 1)

            z = torch.randn(y_shape).to(self.device) if i > 1 else 0

            # split predictions and compute weighting
            if extract_embedding:
                eps = self.nn_model(y_i, x_batch, t_is, context_mask, x_embed)
            else:
                eps = self.nn_model(y_i, x_batch, t_is, context_mask)
            if not is_zero:
                eps1 = eps[:n_sample]
                eps2 = eps[n_sample:]
                eps = (1 + self.guide_w) * eps1 - self.guide_w * eps2
                y_i = y_i[:n_sample]
            y_i = self.oneover_sqrta[i] * (y_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
            if return_y_trace and (i % 20 == 0 or i == self.n_T or i < 8):
                y_i_store.append(y_i.detach().cpu().numpy())

        if return_y_trace:
            return y_i, y_i_store
        else:
            return y_i

    def sample_update(self, x_batch, betas, n_T, return_y_trace=False):
        original_nT = self.n_T

        # set new schedule
        self.n_T = n_T
        for k, v in ddpm_schedules(betas[0], betas[1], self.n_T).items():
            self.register_buffer(k, v.to(self.device))

        # also use this as a shortcut to avoid doubling batch when guide_w is zero
        is_zero = False
        if self.guide_w > -1e-3 and self.guide_w < 1e-3:
            is_zero = True

        # how many noisy actions to begin with
        n_sample = x_batch.shape[0]

        y_shape = (n_sample, self.y_dim)

        # sample initial noise, y_0 ~ N(0, 1),
        y_i = torch.randn(y_shape).to(self.device)

        if not is_zero:
            if len(x_batch.shape) > 2:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1, 1, 1)
            else:
                # repeat x_batch twice, so can use guided diffusion
                x_batch = x_batch.repeat(2, 1)
            # half of context will be zero
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)
            context_mask[n_sample:] = 1.0  # makes second half of batch context free
        else:
            context_mask = torch.zeros(x_batch.shape[0]).to(self.device)

        # run denoising chain
        y_i_store = []  # if want to trace how y_i evolved
        for i in range(self.n_T, 0, -1):
            t_is = torch.tensor([i / self.n_T]).to(self.device)
            t_is = t_is.repeat(n_sample, 1)

            if not is_zero:
                # double batch
                y_i = y_i.repeat(2, 1)
                t_is = t_is.repeat(2, 1)

            # I'm a bit confused why we are adding noise during denoising?
            z = torch.randn(y_shape).to(self.device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(y_i, x_batch, t_is, context_mask)
            if not is_zero:
                eps1 = eps[:n_sample]
                eps2 = eps[n_sample:]
                eps = (1 + self.guide_w) * eps1 - self.guide_w * eps2
                y_i = y_i[:n_sample]
            y_i = self.oneover_sqrta[i] * (y_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
            if return_y_trace and (i % 20 == 0 or i == self.n_T or i < 8):
                y_i_store.append(y_i.detach().cpu().numpy())

        # reset original schedule
        self.n_T = original_nT
        for k, v in ddpm_schedules(betas[0], betas[1], self.n_T).items():
            self.register_buffer(k, v.to(self.device))

        if return_y_trace:
            return y_i, y_i_store
        else:
            return y_i

    def sample_extra(self, x_batch, task_embed, step, extra_steps=4, return_y_trace=False):
        obs_input, franka_proprio, lang_prompt = x_batch

        # compute lang feature
        # used for model
        text_feat, text_emb = self.q_func.encode_text(lang_prompt)

        # also use this as a shortcut to avoid doubling batch when guide_w is zero
        is_zero = False
        if self.guide_w > -1e-3 and self.guide_w < 1e-3:
            is_zero = True

        # how many noisy actions to begin with
        n_sample = obs_input.shape[0]
        voxel_len = obs_input.shape[-1]

        # sample initial noise, y_0 ~ N(0, 1),
        y_i_pos = torch.randn(n_sample, 3).to(self.device)
        # rot_dim = (self.q_func.get_num_rot_classes() * 3) + 2
        y_i_rot = torch.randn(n_sample, 3).to(self.device) # 2 direction for quat
        y_i_collide = torch.randn(n_sample, 1).to(self.device)

        y_i = (y_i_pos, y_i_rot, y_i_collide)

        if not is_zero:
            if len(obs_input.shape) > 2:
                # repeat obs_input twice, so can use guided diffusion
                obs_input = obs_input.repeat(2, 1, 1, 1)
            else:
                # repeat obs_input twice, so can use guided diffusion
                obs_input = obs_input.repeat(2, 1)
            # half of context will be zero
            context_mask = torch.zeros(obs_input.shape[0]).to(self.device)
            context_mask[n_sample:] = 1.0  # makes second half of batch context free
        else:
            # context_mask = torch.zeros_like(obs_input[:,0]).to(self.device)
            context_mask = torch.zeros(obs_input.shape[0]).to(self.device)

        # run denoising chain
        y_i_store = []  # if want to trace how y_i evolved
        # for i_dummy in range(self.n_T, 0, -1):
        for i_dummy in range(self.n_T, -extra_steps, -1):

            i = max(i_dummy, 1)
            t_is = torch.tensor([i / self.n_T]).to(self.device)
            t_is = t_is.repeat(n_sample, 1)

            if not is_zero:
                # double batch
                y_i = (y_i_pos.repeat(2, 1), y_i_rot.repeat(2, 1), y_i_collide.repeat(2, 1))
                t_is = t_is.repeat(2, 1)

            z_pos = torch.randn(y_i_pos.shape).to(self.device) if i > 1 else 0
            z_rot = torch.randn(y_i_rot.shape).to(self.device) if i > 1 else 0
            z_collide = torch.randn(y_i_collide.shape).to(self.device) if i > 1 else 0

            # split predictions and compute weighting
            # eps = self.nn_model(y_i, x_batch, t_is, context_mask)

            # TODO no_grad to save space
            with torch.no_grad():
                q_trans, q_rot_grip, q_collide = self.q_func(obs_input, franka_proprio, text_feat, text_emb, y_i,
                                                             task_embed, t_is, step, context_mask)

            if not is_zero:
                q_trans1 = q_trans[:n_sample]
                q_trans2 = q_trans[n_sample:]
                q_rot1 = q_rot_grip[:n_sample]
                q_rot2 = q_rot_grip[n_sample:]
                q_collide1 = q_collide[:n_sample]
                q_collide2 = q_collide[n_sample:]

                q_trans = (1 + self.guide_w) * q_trans1 - self.guide_w * q_trans2
                q_rot_grip = (1 + self.guide_w) * q_rot1 - self.guide_w * q_rot2
                q_collide = (1 + self.guide_w) * q_collide1 - self.guide_w * q_collide2

                y_i = (y_i_pos[:n_sample], y_i_rot[:n_sample], y_i_collide[:n_sample])
            # y_i = self.oneover_sqrta[i] * (y_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z

            q_collide = q_rot_grip[:, -1:]
            q_rot_grip = q_rot_grip[:, :3]

            y_i_pos = self.oneover_sqrta[i] * (y_i_pos - q_trans * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z_pos
            y_i_rot = self.oneover_sqrta[i] * (y_i_rot - q_rot_grip * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z_rot
            y_i_collide = self.oneover_sqrta[i] * (y_i_collide - q_collide * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z_collide

            y_i = (y_i_pos, y_i_rot, y_i_collide)
            if return_y_trace and (i % 20 == 0 or i == self.n_T or i < 8):
                y_i_np = (y_i_pos.detach().cpu().numpy(), y_i_rot.detach().cpu().numpy(), y_i_collide.detach().cpu().numpy())
                y_i_store.append(y_i_np)
            # print(torch.cuda.memory_allocated())

        if return_y_trace:
            return y_i, y_i_store
        else:
            return y_i

    def sample_extra_seq(self, x_batch, task_embed, extra_steps=4, return_y_trace=False):
        # TODO set seq_horizon here
        seq_horizon = 4
        obs_input, franka_proprio, lang_prompt = x_batch

        # compute lang feature
        # used for model
        text_feat, text_emb = self.q_func.encode_text(lang_prompt)

        # also use this as a shortcut to avoid doubling batch when guide_w is zero
        is_zero = False
        if self.guide_w > -1e-3 and self.guide_w < 1e-3:
            is_zero = True

        # how many noisy actions to begin with
        n_sample = obs_input.shape[0]
        voxel_len = obs_input.shape[-1]

        # sample initial noise, y_0 ~ N(0, 1),
        y_i_pos = torch.randn(n_sample, seq_horizon, voxel_len, voxel_len, voxel_len).to(self.device)

        rot_dim = (self.q_func.get_num_rot_classes() * 3) + 2
        y_i_rot = torch.randn(n_sample, seq_horizon, rot_dim).to(self.device) # 2 direction for quat
        y_i_collide = torch.randn(n_sample, seq_horizon, 2).to(self.device)

        # fix y_i_rot and y_i_collide shape
        y_i_rot = y_i_rot.view(n_sample, seq_horizon * rot_dim)
        y_i_collide = y_i_collide.view(n_sample, seq_horizon * 2)

        y_i = (y_i_pos, y_i_rot, y_i_collide)

        if not is_zero:
            if len(obs_input.shape) > 2:
                # repeat obs_input twice, so can use guided diffusion
                obs_input = obs_input.repeat(2, 1, 1, 1)
            else:
                # repeat obs_input twice, so can use guided diffusion
                obs_input = obs_input.repeat(2, 1)
            # half of context will be zero
            context_mask = torch.zeros(obs_input.shape[0]).to(self.device)
            context_mask[n_sample:] = 1.0  # makes second half of batch context free
        else:
            # context_mask = torch.zeros_like(obs_input[:,0]).to(self.device)
            context_mask = torch.zeros(obs_input.shape[0]).to(self.device)

        # run denoising chain
        y_i_store = []  # if want to trace how y_i evolved
        # for i_dummy in range(self.n_T, 0, -1):
        for i_dummy in range(self.n_T, -extra_steps, -1):

            i = max(i_dummy, 1)
            t_is = torch.tensor([i / self.n_T]).to(self.device)
            t_is = t_is.repeat(n_sample, 1)

            if not is_zero:
                # double batch
                y_i = (y_i_pos.repeat(2, 1), y_i_rot.repeat(2, 1), y_i_collide.repeat(2, 1))
                t_is = t_is.repeat(2, 1)

            # split predictions and compute weighting
            # eps = self.nn_model(y_i, x_batch, t_is, context_mask)

            # TODO no_grad to save space
            with torch.no_grad():
                q_trans, q_rot_grip, q_collide = self.q_func(obs_input, franka_proprio, text_feat, text_emb, y_i,
                                                             task_embed, t_is, context_mask)

            if not is_zero:
                q_trans1 = q_trans[:n_sample]
                q_trans2 = q_trans[n_sample:]
                q_rot1 = q_rot_grip[:n_sample]
                q_rot2 = q_rot_grip[n_sample:]
                q_collide1 = q_collide[:n_sample]
                q_collide2 = q_collide[n_sample:]

                q_trans = (1 + self.guide_w) * q_trans1 - self.guide_w * q_trans2
                q_rot_grip = (1 + self.guide_w) * q_rot1 - self.guide_w * q_rot2
                q_collide = (1 + self.guide_w) * q_collide1 - self.guide_w * q_collide2

                y_i = (y_i_pos[:n_sample], y_i_rot[:n_sample], y_i_collide[:n_sample])

            y_i_rot = y_i_rot.view(n_sample, seq_horizon, rot_dim)
            y_i_collide = y_i_collide.view(n_sample, seq_horizon, 2)

            z_pos = torch.randn(y_i_pos.shape).to(self.device) if i > 1 else 0
            z_rot = torch.randn(y_i_rot.shape).to(self.device) if i > 1 else 0
            z_collide = torch.randn(y_i_collide.shape).to(self.device) if i > 1 else 0

            y_i_pos = self.oneover_sqrta[i] * (y_i_pos - q_trans * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z_pos
            y_i_rot = self.oneover_sqrta[i] * (y_i_rot - q_rot_grip * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z_rot
            y_i_collide = self.oneover_sqrta[i] * (y_i_collide - q_collide * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z_collide

            # fix y_i_rot and y_i_collide shape
            y_i_rot = y_i_rot.view(n_sample, seq_horizon * rot_dim)
            y_i_collide = y_i_collide.view(n_sample, seq_horizon * 2)

            y_i = (y_i_pos, y_i_rot, y_i_collide)
            if return_y_trace and (i % 20 == 0 or i == self.n_T or i < 8):
                y_i_np = (y_i_pos.detach().cpu().numpy(), y_i_rot.detach().cpu().numpy(), y_i_collide.detach().cpu().numpy())
                y_i_store.append(y_i_np)
            # print(torch.cuda.memory_allocated())

        if return_y_trace:
            return y_i, y_i_store
        else:
            return y_i


    def decode_action_value(self, y):
        y_pos, y_rot, y_collide = y
        bs = y_pos.shape[0]

        is_grip = torch.where(y_collide > 0.5, torch.ones_like(y_collide), torch.zeros_like(y_collide))
        voxel_bnd = self.decoder.voxel_bnd
        voxel_bnd_len = self.decoder.voxel_bnd_len

        voxel_bnd_min = voxel_bnd[:, 0]
        voxel_bnd_min = voxel_bnd_min.unsqueeze(0).repeat(bs, 1)
        voxel_bnd_len = voxel_bnd_len.unsqueeze(0).repeat(bs, 1)
        pos = voxel_bnd_min + voxel_bnd_len * y_pos

        quat = torch.zeros((bs, 4)).float().to(self.device)
        y_rot = y_rot * 2.0 * pi

        for b in range(bs):
            rot_i = y_rot[b, :]
            quat_i = transforms.axis_angle_to_quaternion(rot_i)
            quat[b] = quat_i

        action = torch.cat((pos, quat, is_grip), dim=-1)
        return action

    def decode_action(self, y):
        y_pos, y_rot, y_collide = y
        seq_horizon = y_pos.shape[1]
        bs = y_pos.shape[0]
        if seq_horizon > 1:
            y_rot = y_rot.view(bs, seq_horizon, -1)
            y_collide = y_collide.view(bs, seq_horizon, -1)
            actions = []
            for step_i in range(seq_horizon):
                y_i = (y_pos[:, step_i, :].unsqueeze(1), y_rot[:, step_i, :], y_collide[:, step_i, :])
                actions.append(self.decode_action_step(y_i).unsqueeze(1))

            actions = torch.cat(actions, dim=1)
            return actions
        else:
            return self.decode_action_step(y)


    def decode_action_step(self, y):
        y_pos, y_rot, y_collide = y
        bs = y_rot.shape[0]
        coords, rot_and_grip_indicies, ignore_collision = self.q_func.choose_highest_action(y_pos, y_rot, y_collide)

        voxel_bnd = self.decoder.voxel_bnd
        voxel_bnd_len = self.decoder.voxel_bnd_len
        voxel_size = self.decoder._voxel_size

        voxel_bnd_min = voxel_bnd[:, 0]
        voxel_bnd_min = voxel_bnd_min.unsqueeze(0).repeat(bs, 1)
        grid_size = (voxel_bnd_len / voxel_size).unsqueeze(0).repeat(bs, 1)

        pos = voxel_bnd_min + grid_size * coords
        is_grip = rot_and_grip_indicies[:, 3:]

        rot_indicies = rot_and_grip_indicies[:, :3]
        rot_num_classes = self.decoder._num_rotation_classes
        rot_scale = torch.ones_like(voxel_bnd_min) * pi * 2.0 / rot_num_classes
        rot = rot_indicies * rot_scale

        quat = torch.zeros_like(rot_and_grip_indicies).float()

        for b in range(bs):
            rot_i = rot[b, :]
            quat_i = transforms.axis_angle_to_quaternion(rot_i)
            quat[b] = quat_i

        action = torch.cat((pos, quat, is_grip), dim=-1)

        return action



################################################################

def ddpm_schedules(beta1, beta2, T, is_linear=True):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    # beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    # beta_t = (beta2 - beta1) * torch.arange(-1, T + 1, dtype=torch.float32) / T + beta1
    if is_linear:
        beta_t = (beta2 - beta1) * torch.arange(-1, T, dtype=torch.float32) / (T - 1) + beta1
    else:
        beta_t = (beta2 - beta1) * torch.square(torch.arange(-1, T, dtype=torch.float32)) / torch.max(torch.square(torch.arange(-1, T, dtype=torch.float32))) + beta1
    beta_t[0] = beta1  # modifying this so that beta_t[1] = beta1, and beta_t[n_T]=beta2, while beta[0] is never used
    # this is as described in Denoising Diffusion Probabilistic Models paper, section 4
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }
