# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

from math import ceil

import torch
import torch.nn.functional as F

from torch import nn
from einops import rearrange, repeat
import pdb
from new_scripts.helpers.network_utils import SpatialSoftmax3D, Conv3DBlock


from new_scripts.model.rvt.attn import (
    Conv2DBlock,
    Conv2DUpsampleBlock,
    PreNorm,
    Attention,
    cache_fn,
    DenseBlock,
    FeedForward,
)

class TimeSiren(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(TimeSiren, self).__init__()
        # just a fully connected NN with sin activations
        self.lin1 = nn.Linear(input_dim, emb_dim, bias=False)
        self.lin2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        x = torch.sin(self.lin1(x))
        x = self.lin2(x)
        return x


class MVT(nn.Module):
    def __init__(
        self,
        loss_type,  # which types of action output due to loss type
        act_type,  # which types of action should be the input and output
        add_lang,
        num_img=5,
        depth=8,
        img_size=320,
        add_proprio=True,
        proprio_dim=9,
        lang_dim=512,
        lang_len=77,
        img_feat_dim=3,
        feat_dim=(72*3)+2+2,
        im_channels=64,
        attn_dim=512,
        attn_heads=8,
        attn_dim_head=64,
        activation="lrelu",
        weight_tie_layers=False,
        attn_dropout=0.1,
        decoder_dropout=0.0,
        img_patch_size=16,
        final_dim=64,
        self_cross_ver=1,
        add_corr=True,
        add_pixel_loc=False,
        add_depth=True,
        pe_fix=True,
        renderer=None,
    ):
        """MultiView Transfomer

        :param depth: depth of the attention network
        :param img_size: number of pixels per side for rendering
        :param renderer_device: device for placing the renderer
        :param add_proprio:
        :param proprio_dim:
        :param add_lang:
        :param lang_dim:
        :param lang_len:
        :param img_feat_dim:
        :param feat_dim:
        :param im_channels: intermediate channel size
        :param attn_dim:
        :param attn_heads:
        :param attn_dim_head:
        :param activation:
        :param weight_tie_layers:
        :param attn_dropout:
        :param decoder_dropout:
        :param img_patch_size: intial patch size
        :param final_dim: final dimensions of features
        :param self_cross_ver:
        :param add_corr:
        :param add_pixel_loc:
        :param add_depth:
        :param pe_fix: matter only when add_lang is True
            Either:
                True: use position embedding only for image tokens
                False: use position embedding for lang and image token
        """

        super().__init__()
        self.loss_type = loss_type
        self.act_type = act_type
        self.depth = depth
        self.img_feat_dim = img_feat_dim
        self.img_size = img_size
        self.add_proprio = add_proprio
        self.proprio_dim = proprio_dim
        self.add_lang = add_lang
        self.lang_dim = lang_dim
        self.lang_len = lang_len
        self.im_channels = im_channels
        self.img_patch_size = img_patch_size
        self.final_dim = final_dim
        self.attn_dropout = attn_dropout
        self.decoder_dropout = decoder_dropout
        self.self_cross_ver = self_cross_ver
        self.add_corr = add_corr
        self.add_pixel_loc = add_pixel_loc
        self.add_depth = add_depth
        self.pe_fix = pe_fix

        self.model_name = "rvt"

        print(f"MVT Vars: {vars(self)}")
        self.num_img = num_img

        # patchified input dimensions
        spatial_size = img_size // self.img_patch_size  # 128 / 8 = 16

        if self.add_proprio:
            # 64 img features + 64 proprio features
            self.input_dim_before_seq = self.im_channels * 8
        else:
            self.input_dim_before_seq = self.im_channels * 6

        # learnable positional encoding
        if add_lang:
            lang_emb_dim, lang_max_seq_len = lang_dim, lang_len
        else:
            lang_emb_dim, lang_max_seq_len = 0, 0
        self.lang_emb_dim = lang_emb_dim
        self.lang_max_seq_len = lang_max_seq_len

        if self.pe_fix:
            num_pe_token = spatial_size**2 * self.num_img
        else:
            num_pe_token = lang_max_seq_len + (spatial_size**2 * self.num_img)

        self.pos_encoding = nn.Parameter(
            torch.randn(
                1,
                num_pe_token,
                self.input_dim_before_seq,
            )
        )

        inp_img_feat_dim = self.img_feat_dim
        if self.add_corr:
            inp_img_feat_dim += 3
        if self.add_pixel_loc:
            inp_img_feat_dim += 3
            self.pixel_loc = torch.zeros(
                (self.num_img, 3, self.img_size, self.img_size)
            )
            self.pixel_loc[:, 0, :, :] = (
                torch.linspace(-1, 1, self.num_img).unsqueeze(-1).unsqueeze(-1)
            )
            self.pixel_loc[:, 1, :, :] = (
                torch.linspace(-1, 1, self.img_size).unsqueeze(0).unsqueeze(-1)
            )
            self.pixel_loc[:, 2, :, :] = (
                torch.linspace(-1, 1, self.img_size).unsqueeze(0).unsqueeze(0)
            )
        if self.add_depth:
            inp_img_feat_dim += 1

        # img input preprocessing encoder
        self.input_preprocess = Conv2DBlock(
            inp_img_feat_dim,
            self.im_channels,
            kernel_sizes=1,
            strides=1,
            norm=None,
            activation=activation,
        )
        inp_pre_out_dim = self.im_channels


        self.y_pos_embed_nn = DenseBlock(3, self.im_channels, norm=None, activation=activation, )
        self.y_rotation_embed_nn = DenseBlock(3, self.im_channels, norm=None, activation=activation, )
        self.y_grip_embed_nn = DenseBlock(1, self.im_channels, norm=None, activation=activation, )

        self.task_embed_nn = DenseBlock(128*5, self.im_channels, norm=None, activation=activation,)
        # task masking
        self.task_mask_nn = Conv3DBlock(
            self.im_channels, self.im_channels, kernel_sizes=1, strides=1,
            norm=None, activation=activation,
        )
        # batch norm for patchify params
        self.patchify_batchnorm = nn.BatchNorm3d(self.im_channels)
        self.task_ln = nn.LayerNorm([self.im_channels, self.num_img, 20, 20])

        # embed t
        self.t_embed_nn = TimeSiren(1, self.im_channels)

        # embed traj step
        self.traj_step_embed_nn = TimeSiren(1, self.im_channels)
        self.traj_step_ln = nn.LayerNorm([self.im_channels, self.num_img, 20, 20])

        if self.add_proprio:
            # proprio preprocessing encoder
            self.proprio_preprocess = DenseBlock(
                self.proprio_dim,
                self.im_channels,
                norm="group",
                activation=activation,
            )

        self.patchify = Conv2DBlock(
            inp_pre_out_dim,
            self.im_channels,
            kernel_sizes=self.img_patch_size,
            strides=self.img_patch_size,
            norm="group",
            activation=activation,
            padding=0,
        )

        # patchify conv
        # self.patchify3D = Conv3DBlock(
        #     self.im_channels,
        #     self.im_channels,
        #     kernel_sizes=self.img_patch_size // 2,
        #     strides=self.img_patch_size,
        #     norm=None,
        #     activation=activation,
        #     padding=0,
        # )

        # lang preprocess
        if self.add_lang:
            self.lang_preprocess = DenseBlock(
                lang_emb_dim,
                self.im_channels * 8,
                norm="group",
                activation=activation,
            )

        self.fc_bef_attn = DenseBlock(
            self.input_dim_before_seq,
            attn_dim,
            norm=None,
            activation=None,
        )
        self.fc_aft_attn = DenseBlock(
            attn_dim,
            self.input_dim_before_seq,
            norm=None,
            activation=None,
        )

        get_attn_attn = lambda: PreNorm(
            attn_dim,
            Attention(
                attn_dim,
                heads=attn_heads,
                dim_head=attn_dim_head,
                dropout=attn_dropout,
            ),
        )
        get_attn_ff = lambda: PreNorm(attn_dim, FeedForward(attn_dim))
        get_attn_attn, get_attn_ff = map(cache_fn, (get_attn_attn, get_attn_ff))
        # self-attention layers
        self.layers = nn.ModuleList([])
        cache_args = {"_cache": weight_tie_layers}
        attn_depth = depth

        for _ in range(attn_depth):
            self.layers.append(
                nn.ModuleList([get_attn_attn(**cache_args), get_attn_ff(**cache_args)])
            )

        self.up0 = Conv2DUpsampleBlock(
            self.input_dim_before_seq,
            self.im_channels,
            kernel_sizes=self.img_patch_size - 1,
            strides=self.img_patch_size,
            norm=None,
            activation=activation,
        )

        final_inp_dim = self.im_channels + inp_pre_out_dim

        # final layers
        self.final = Conv2DBlock(
            final_inp_dim,
            self.im_channels,
            kernel_sizes=3,
            strides=1,
            norm=None,
            activation=activation,
        )

        # TODO training robot seq = 2
        self.seq_horizon = 2
        # self.seq_horizon = 4

        if "seq" in self.act_type:
            trans_channel_dim = self.seq_horizon
        else:
            trans_channel_dim = 1

        self.trans_decoder = Conv2DBlock(
            self.final_dim,
            trans_channel_dim,
            kernel_sizes=3,
            strides=1,
            norm=None,
            activation=None,
        )

        if "mse" in self.loss_type:
            self.trans_ind_decoder0 = Conv2DBlock(self.num_img, self.seq_horizon, kernel_sizes=5, strides=5, activation=activation)
            self.trans_ind_decoder1 = Conv2DBlock(self.seq_horizon, self.seq_horizon, kernel_sizes=5, strides=5, activation=activation)
            self.trans_ind_fc = DenseBlock(13 * 13, 3, norm=None, activation="tanh",)

        feat_out_size = feat_dim
        feat_fc_dim = 0
        feat_fc_dim += self.input_dim_before_seq
        feat_fc_dim += self.final_dim

        if "seq" in self.act_type:
            feat_out_size += 2 * 3

        self.feat_fc = nn.Sequential(
            nn.Linear(self.num_img * feat_fc_dim, feat_fc_dim),
            nn.ReLU(),
            nn.Linear(feat_fc_dim, feat_fc_dim // 2),
            nn.ReLU(),
            nn.Linear(feat_fc_dim // 2, feat_out_size),
        )

        self.task_dense = DenseBlock(self.num_img * feat_fc_dim, 256, None, activation)
        self.task_pred_decoder = DenseBlock(256, 640, None,  activation="tanh")


    def forward(
            self,
            img,
            proprio,
            lang_goal_emb,
            lang_token_embs,
            y_t,                # diffusion model
            task_embed,         # Task indicator
            t,                  # diffusion model
            traj_step,          # current step of the trajectory
            context_mask,       # CFG diffusion model
            pos_cond,           # Ground-Truth Conditional var for predicting rotation
            prev_layer_voxel_grid=None,
            bounds=None,
            prev_layer_bounds=None,

    ):
        """
        :param img: tensor of shape (bs, num_img, img_feat_dim, h, w)
        :param proprio: tensor of shape (bs, priprio_dim)
        :param lang_emb: tensor of shape (bs, lang_len, lang_dim)
        :param img_aug: (float) magnitude of augmentation in rgb image
        """

        bs, num_img, img_feat_dim, h, w = img.shape
        num_pat_img = h // self.img_patch_size
        assert num_img == self.num_img
        # assert img_feat_dim == self.img_feat_dim
        assert h == w == self.img_size

        img = img.view(bs * num_img, img_feat_dim, h, w)
        # preprocess
        # (bs * num_img, im_channels, h, w)
        d0 = self.input_preprocess(img)

        # (bs * num_img, im_channels, h, w) ->
        # (bs * num_img, im_channels, h / img_patch_strid, w / img_patch_strid) patches
        ins = self.patchify(d0)
        # (bs, im_channels, num_img, h / img_patch_strid, w / img_patch_strid) patches
        ins = (
            ins.view(
                bs,
                num_img,
                self.im_channels,
                num_pat_img,
                num_pat_img,
            )
            .transpose(1, 2)
            .clone()
        )

        # concat proprio
        _, _, _d, _h, _w = ins.shape
        if self.add_proprio:
            p = self.proprio_preprocess(proprio)  # [B,4] -> [B,64]
            p = p.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, _d, _h, _w)
            ins = torch.cat([ins, p], dim=1)  # [B, 128, num_img, np, np]

        # add conditional
        # concat y and t for diffusion model
        y_t_pos, y_t_rotation, y_t_grip = y_t
        y_t_pos_embed = self.y_pos_embed_nn(y_t_pos)  # [B, 64, 5, 5, 5]
        # y_t_pos_embed = self.patchify(y_t_pos_embed)
        y_t_rotation_embed = self.y_rotation_embed_nn(y_t_rotation)
        y_t_grip_embed = self.y_grip_embed_nn(y_t_grip)
        t_embed = self.t_embed_nn(t)

        # TODO should not be that early
        # concat task embed
        task_embed = self.task_embed_nn(task_embed)
        task_embed = task_embed.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, _d, _h, _w)
        task_embed_mask = self.task_mask_nn(task_embed)

        # task_embed_mask = self.patchify_batchnorm(task_embed_mask)
        task_embed_mask = self.task_ln(task_embed_mask)

        # TODO batchnorm layer before cross-attention
        y_t_embed = torch.cat([y_t_pos_embed, y_t_rotation_embed, y_t_grip_embed, t_embed], dim=1)
        y_t_embed = y_t_embed.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, _d, _h, _w)

        ins = torch.cat([ins, y_t_embed, task_embed_mask], dim=1)  # [B, 448, 5, 5, 5]

        # concat current step
        traj_step = traj_step.unsqueeze(-1)
        traj_step_embed = self.traj_step_embed_nn(traj_step)
        traj_step_embed = traj_step_embed.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, _d, _h, _w)
        traj_step_embed = self.traj_step_ln(traj_step_embed)

        ins = torch.cat([ins, traj_step_embed], dim=1)

        # channel last
        ins = rearrange(ins, "b d ... -> b ... d")  # [B, num_img, np, np, 128]

        # save original shape of input for layer
        ins_orig_shape = ins.shape

        # flatten patches into sequence
        ins = rearrange(ins, "b ... d -> b (...) d")  # [B, num_img * np * np, 128]
        # add learable pos encoding

        # only added to image tokens
        if self.pe_fix:
            ins += self.pos_encoding

        # append language features as sequence
        num_lang_tok = 0
        if self.add_lang:
            lang_emb = lang_token_embs
            l = self.lang_preprocess(
                lang_emb.view(bs * self.lang_max_seq_len, self.lang_emb_dim)
            )
            l = l.view(bs, self.lang_max_seq_len, -1)
            num_lang_tok = l.shape[1]
            ins = torch.cat((l, ins), dim=1)  # [B, num_img * np * np + 77, 128]


        # add learable pos encoding
        if not self.pe_fix:
            ins = ins + self.pos_encoding

        x = self.fc_bef_attn(ins)
        if self.self_cross_ver == 0:
            # self-attention layers
            for self_attn, self_ff in self.layers:
                x = self_attn(x) + x
                x = self_ff(x) + x

        elif self.self_cross_ver == 1:
            lx, imgx = x[:, :num_lang_tok], x[:, num_lang_tok:]

            # within image self attention
            imgx = imgx.reshape(bs * num_img, num_pat_img * num_pat_img, -1)
            for self_attn, self_ff in self.layers[: len(self.layers) // 2]:
                imgx = self_attn(imgx) + imgx
                imgx = self_ff(imgx) + imgx

            imgx = imgx.view(bs, num_img * num_pat_img * num_pat_img, -1)
            x = torch.cat((lx, imgx), dim=1)
            # cross attention
            for self_attn, self_ff in self.layers[len(self.layers) // 2 :]:
                x = self_attn(x) + x
                x = self_ff(x) + x

        else:
            assert False

        # append language features as sequence
        if self.add_lang:
            # throwing away the language embeddings
            x = x[:, num_lang_tok:]
        x = self.fc_aft_attn(x)

        # reshape back to orginal size
        x = x.view(bs, *ins_orig_shape[1:-1], x.shape[-1])  # [B, num_img, np, np, 128]
        x = rearrange(x, "b ... d -> b d ...")  # [B, 128, num_img, np, np]

        feat = []
        _feat = torch.max(torch.max(x, dim=-1)[0], dim=-1)[0]
        _feat = _feat.view(bs, -1)
        feat.append(_feat)

        x = (
            x.transpose(1, 2)
            .clone()
            .view(
                bs * self.num_img, self.input_dim_before_seq, num_pat_img, num_pat_img
            )
        )

        u0 = self.up0(x)
        u0 = torch.cat([u0, d0], dim=1)
        u = self.final(u0)

        # translation decoder
        if "seq" in self.act_type:
            trans = self.trans_decoder(u).view(bs, self.num_img, self.seq_horizon, h, w)
            hm = F.softmax(trans.detach().view(bs, self.num_img * self.seq_horizon, h * w), 2).view(
                bs * self.num_img, self.seq_horizon, h, w
            )
            repeat_dim = int(self.final_dim / self.seq_horizon)
            hm = hm.repeat(1, repeat_dim, 1, 1)

        else:
            trans = self.trans_decoder(u).view(bs, self.num_img, h, w)
            hm = F.softmax(trans.detach().view(bs, self.num_img, h * w), 2).view(
                bs * self.num_img, 1, h, w
            )

        _feat = torch.sum(hm * u, dim=[2, 3])
        _feat = _feat.view(bs, -1)
        feat.append(_feat)
        feat = torch.cat(feat, dim=-1)
        task_feat = self.task_dense(feat)
        task_pred = self.task_pred_decoder(task_feat)

        feat_final = self.feat_fc(feat)
        rot_and_grip_out = feat_final[:, :-2]
        collision_out = feat_final[:, -2:]

        if "mse" in self.loss_type:
            trans_ind0 = self.trans_ind_decoder0(trans)
            trans_ind1 = self.trans_ind_decoder1(trans_ind0)
            trans_ind1 = trans_ind1.view(bs * self.seq_horizon, -1)
            trans_ind = self.trans_ind_fc(trans_ind1)
            trans = trans_ind.view(bs, self.seq_horizon, -1)

        return trans, rot_and_grip_out, collision_out, task_pred


    def get_wpt(self, out, dyn_cam_info, y_q=None):
        """
        Estimate the q-values given output from mvt
        :param out: output from mvt
        """
        nc = self.num_img
        h = w = self.img_size
        bs = out["trans"].shape[0]

        q_trans = out["trans"].view(bs, nc, h * w)
        hm = torch.nn.functional.softmax(q_trans, 2)
        hm = hm.view(bs, nc, h, w)

        if dyn_cam_info is None:
            dyn_cam_info_itr = (None,) * bs
        else:
            dyn_cam_info_itr = dyn_cam_info


        return

