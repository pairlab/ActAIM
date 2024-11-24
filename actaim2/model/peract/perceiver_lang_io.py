# Perceiver IO implementation adpated for manipulation
# Source: https://github.com/lucidrains/perceiver-pytorch
# License: https://github.com/lucidrains/perceiver-pytorch/blob/main/LICENSE

from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat, reduce
from einops.layers.torch import Reduce

from new_scripts.helpers.network_utils import DenseBlock, SpatialSoftmax3D, Conv3DBlock, Conv3DUpsampleBlock
import pdb
# helpers

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn

def fourier_encode(x, max_freq, num_bands = 4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    x = torch.cat((x, orig_x), dim = -1)
    return x

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        # dropout
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


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


# PerceiverIO adapted for 6-DoF manipulation
class PerceiverVoxelLangEncoder(nn.Module):
    def __init__(
            self,
            loss_type,                # which types of action output due to loss type
            act_type,                 # which types of action should be the input and output
            depth,                    # number of self-attention layers
            iterations,               # number cross-attention iterations (PerceiverIO uses just 1)
            voxel_size,               # N voxels per side (size: N*N*N)
            initial_dim,              # 10 dimensions - dimension of the input sequence to be encoded
            low_dim_size,             # 4 dimensions - proprioception: {gripper_open, left_finger, right_finger, timestep}
            layer=0,
            num_rotation_classes=72,  # 5 degree increments (5*72=360) for each of the 3-axis
            num_grip_classes=2,       # open or not open
            num_collision_classes=2,  # collisions allowed or not allowed
            input_axis=3,             # 3D tensors have 3 axes
            num_latents=512,          # number of latent vectors
            im_channels=64,           # intermediate channel size
            latent_dim=512,           # dimensions of latent vectors
            cross_heads=1,            # number of cross-attention heads
            latent_heads=8,           # number of latent heads
            cross_dim_head=64,
            latent_dim_head=64,
            activation='relu',
            weight_tie_layers=False,
            pos_encoding_with_lang=True,
            input_dropout=0.1,
            attn_dropout=0.1,
            decoder_dropout=0.0,
            lang_fusion_type='seq',
            voxel_patch_size=9,
            voxel_patch_stride=10,
            no_skip_connection=False,
            no_perceiver=False,
            add_lang=False,
            final_dim=64,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.act_type = act_type
        self.depth = depth
        self.layer = layer
        self.init_dim = int(initial_dim)
        self.iterations = iterations
        self.input_axis = input_axis
        self.voxel_size = voxel_size
        self.low_dim_size = low_dim_size
        self.im_channels = im_channels
        self.pos_encoding_with_lang = pos_encoding_with_lang
        self.lang_fusion_type = lang_fusion_type
        self.voxel_patch_size = voxel_patch_size
        self.voxel_patch_stride = voxel_patch_stride
        self.num_rotation_classes = num_rotation_classes
        self.num_grip_classes = num_grip_classes
        self.num_collision_classes = num_collision_classes
        self.final_dim = final_dim
        self.input_dropout = input_dropout
        self.attn_dropout = attn_dropout
        self.decoder_dropout = decoder_dropout
        self.no_skip_connection = no_skip_connection
        self.no_perceiver = no_perceiver
        self.no_language = not add_lang

        self.model_name = "peract"

        # patchified input dimensions
        spatial_size = voxel_size // self.voxel_patch_stride  # 100/5 = 20

        # 64 voxel features + 64 proprio features (+ 64 lang goal features if concattenated)
        self.input_dim_before_seq = self.im_channels * 3 + 1 if self.lang_fusion_type == 'concat' else self.im_channels * 7 + 1

        # CLIP language feature dimensions
        lang_feat_dim, lang_emb_dim, lang_max_seq_len = 1024, 512, 77

        # learnable positional encoding
        if self.pos_encoding_with_lang:
            self.pos_encoding = nn.Parameter(torch.randn(1,
                                                         lang_max_seq_len + spatial_size ** 3,
                                                         self.input_dim_before_seq))
        else:
            # assert self.lang_fusion_type == 'concat', 'Only concat is supported for pos encoding without lang.'
            self.pos_encoding = nn.Parameter(torch.randn(1,
                                                         spatial_size, spatial_size, spatial_size,
                                                         self.input_dim_before_seq))

        # voxel input preprocessing 1x1 conv encoder
        self.input_preprocess = Conv3DBlock(
            self.init_dim, self.im_channels, kernel_sizes=1, strides=1,
            norm=None, activation=activation,
        )

        # softmax
        self.softmax = nn.Softmax(dim=1)

        # patchify conv
        self.patchify = Conv3DBlock(
            self.input_preprocess.out_channels, self.im_channels,
            kernel_sizes=self.voxel_patch_size, strides=self.voxel_patch_stride,
            norm=None, activation=activation)

        # language preprocess
        if self.lang_fusion_type == 'concat':
            self.lang_preprocess = nn.Linear(lang_feat_dim, self.im_channels)

        elif self.lang_fusion_type == 'seq':
            # diffsuion model concat y and t
            self.lang_preprocess = nn.Linear(lang_emb_dim, self.input_dim_before_seq)

        # proprioception
        if self.low_dim_size > 0:
            self.proprio_preprocess = DenseBlock(
                self.low_dim_size, self.im_channels, norm=None, activation=activation,
            )

        # embed t
        self.t_embed_nn = TimeSiren(1, self.im_channels)


        if "seq" in self.act_type:
            self.seq_horizon = 4
            pos_channel = self.seq_horizon
            rotation_dim = self.seq_horizon * 218
            grip_dim = self.seq_horizon * 2

            self.num_layers = 2
            self.hidden_size = 64
            self.lstm = nn.LSTM(self.final_dim, self.hidden_size, self.num_layers, batch_first=True)
            lstm_output_dim = self.num_rotation_classes * 3 + self.num_grip_classes + self.num_collision_classes

            self.lstm_fc = nn.Linear(self.final_dim, lstm_output_dim)
            self.pos_cond_dim = 3  * self.seq_horizon

        else:
            pos_channel = 1
            rotation_dim = 218
            grip_dim = 2
            self.pos_cond_dim = 3

        # TODO heatmap diffusion is not working
        # embed y
        # self.y_pos_embed_nn = Conv3DBlock(
        #     pos_channel,
        #     self.im_channels,
        #      kernel_sizes=3,
        #      strides=1, norm=None, activation=activation)
        # self.y_rotation_embed_nn = DenseBlock(rotation_dim, self.im_channels, norm=None, activation=activation, )
        # self.y_grip_embed_nn = DenseBlock(grip_dim, self.im_channels, norm=None, activation=activation, )

        self.y_pos_embed_nn = DenseBlock(3, self.im_channels, norm=None, activation=activation, )
        self.y_rotation_embed_nn = DenseBlock(3, self.im_channels, norm=None, activation=activation, )
        self.y_grip_embed_nn = DenseBlock(1, self.im_channels, norm=None, activation=activation, )

        self.task_embed_nn = DenseBlock(128*5, self.im_channels, norm=None, activation=activation,)
        # task masking
        self.task_mask_nn = Conv3DBlock(
            self.im_channels, self.im_channels, kernel_sizes=1, strides=1,
            norm=None, activation=activation,
        )

        # batch norm for pachify params
        self.patchify_batchnorm = nn.BatchNorm3d(self.im_channels)

        # pooling functions
        self.local_maxp = nn.MaxPool3d(3, 2, padding=1)
        self.global_maxp = nn.AdaptiveMaxPool3d(1)

        # 1st 3D softmax
        self.ss0 = SpatialSoftmax3D(
            self.voxel_size, self.voxel_size, self.voxel_size, self.im_channels)
        flat_size = self.im_channels * 4

        # latent vectors (that are randomly initialized)
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        # encoder cross attention
        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(latent_dim, Attention(latent_dim,
                                          self.input_dim_before_seq,
                                          heads=cross_heads,
                                          dim_head=cross_dim_head,
                                          dropout=input_dropout),
                    context_dim=self.input_dim_before_seq),
            PreNorm(latent_dim, FeedForward(latent_dim))
        ])

        get_latent_attn = lambda: PreNorm(latent_dim,
                                          Attention(latent_dim, heads=latent_heads,
                                                    dim_head=latent_dim_head, dropout=attn_dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        # self attention layers
        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        # decoder cross attention
        self.decoder_cross_attn = PreNorm(self.input_dim_before_seq, Attention(self.input_dim_before_seq,
                                                                               latent_dim,
                                                                               heads=cross_heads,
                                                                               dim_head=cross_dim_head,
                                                                               dropout=decoder_dropout),
                                                                               context_dim=latent_dim)

        # upsample conv
        self.up0 = Conv3DUpsampleBlock(
            self.input_dim_before_seq, self.final_dim,
            kernel_sizes=self.voxel_patch_size, strides=self.voxel_patch_stride,
            norm=None, activation=activation,
        )

        # 2nd 3D softmax
        self.ss1 = SpatialSoftmax3D(
            spatial_size, spatial_size, spatial_size,
            self.input_dim_before_seq)

        flat_size += self.input_dim_before_seq * 4

        # final 3D softmax
        self.final = Conv3DBlock(
            self.im_channels if (self.no_perceiver or self.no_skip_connection) else self.im_channels * 2,
            self.im_channels,
            kernel_sizes=3,
            strides=1, norm=None, activation=activation)

        self.trans_decoder = Conv3DBlock(
            self.final_dim, pos_channel, kernel_sizes=3, strides=1,
            norm=None, activation=None,
        )

        self.trans_ind_decoder = DenseBlock(voxel_size * voxel_size * voxel_size, 3, activation=None)
        self.rot_grip_ind_decoder = DenseBlock(self.final_dim, 4, None, activation=None)
        self.pos_cond_encoder = DenseBlock(self.pos_cond_dim, self.im_channels, None,  activation=None)
        self.task_pred_decoder = DenseBlock(256, 640, None,  activation=None)


        # rotation, gripper, and collision MLP layers
        if self.num_rotation_classes > 0:
            self.ss_final = SpatialSoftmax3D(
                self.voxel_size, self.voxel_size, self.voxel_size,
                self.im_channels)

            flat_size += self.im_channels * (4 + 1)

            self.dense0 =  DenseBlock(flat_size, 256, None, activation)
            self.dense1 = DenseBlock(256, self.final_dim, None, activation)

            # predict task embed
            self.dense2 = DenseBlock(flat_size + 4 * self.im_channels, 256, None, activation)

            if "seq" in self.act_type:
                rot_grip_output_dim = self.num_rotation_classes * 3 + self.num_grip_classes * 4 + self.num_collision_classes
            else:
                rot_grip_output_dim = self.num_rotation_classes * 3 + self.num_grip_classes + self.num_collision_classes

            self.rot_grip_collision_ff = DenseBlock(self.final_dim,
                                                    rot_grip_output_dim,
                                                    None, None)

    def init_hidden(self, batch_size, device):
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        )

    def forward(
            self,
            ins,
            proprio,
            lang_goal_emb,
            lang_token_embs,
            y_t,                # diffusion model
            task_embed,         # Task indicator
            t,                  # diffusion model
            traj_step,          # current step of the trajectory
            context_mask,       # CFG diffusion model
            pos_cond,           # Ground-Truth Conditional var for predicting rotation

            # noise_a_trans,
            # nosie_a_rot,
            # noise_a_grip,
            prev_layer_voxel_grid=None,
            bounds=None,
            prev_layer_bounds=None,
    ):

        # preprocess input
        d0 = self.input_preprocess(ins)                       # [B,10,100,100,100] -> [B,64,100,100,100]
        y_t_pos, y_t_rotation, y_t_grip = y_t

        # aggregated features from 1st softmax and maxpool for MLP decoders
        feats = [self.ss0(d0.contiguous()), self.global_maxp(d0).view(ins.shape[0], -1)]

        # patchify input (5x5x5 patches)
        ins = self.patchify(d0)                               # [B,64,100,100,100] -> [B,64,20,20,20]
        # ins = self.patchify_batchnorm(ins)

        b, c, d, h, w, device = *ins.shape, ins.device
        axis = [d, h, w]
        assert len(axis) == self.input_axis, 'input must have the same number of axis as input_axis'

        # concat proprio
        if self.low_dim_size > 0:
            p = self.proprio_preprocess(proprio)              # [B,4] -> [B,64]
            p = p.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, d, h, w)
            ins = torch.cat([ins, p], dim=1)                  # [B,128,20,20,20]

        # concat y and t for diffusion model
        y_t_pos_embed = self.y_pos_embed_nn(y_t_pos)    # [B, 64, 5, 5, 5]
        # y_t_pos_embed = self.patchify(y_t_pos_embed)
        y_t_rotation_embed = self.y_rotation_embed_nn(y_t_rotation)
        y_t_grip_embed = self.y_grip_embed_nn(y_t_grip)
        t_embed = self.t_embed_nn(t)

        # TODO should not be that early
        # concat task embed
        task_embed = self.task_embed_nn(task_embed)
        task_embed = task_embed.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.voxel_size, self.voxel_size, self.voxel_size)
        task_embed_mask = self.task_mask_nn(task_embed)
        task_embed_mask = self.patchify(task_embed_mask)
        task_embed_mask = self.patchify_batchnorm(task_embed_mask)

        # TODO batchnorm layer before cross-attention
        y_t_embed = torch.cat([y_t_pos_embed, y_t_rotation_embed, y_t_grip_embed, t_embed], dim=1)
        y_t_embed = y_t_embed.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, d, h, w)

        ins = torch.cat([ins, y_t_embed, task_embed_mask], dim=1)                # [B, 448, 5, 5, 5]

        # concat current step
        traj_step = traj_step.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, d, h, w)
        ins = torch.cat([ins, traj_step], dim=1)

        # language ablation
        if self.no_language:
            lang_goal_emb = torch.zeros_like(lang_goal_emb)
            lang_token_embs = torch.zeros_like(lang_token_embs)

        # option 1: tile and concat lang goal to input
        if self.lang_fusion_type == 'concat':
            lang_emb = lang_goal_emb
            lang_emb = lang_emb.to(dtype=ins.dtype)
            l = self.lang_preprocess(lang_emb)
            l = l.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, d, h, w)
            ins = torch.cat([ins, l], dim=1)

        # channel last
        ins = rearrange(ins, 'b d ... -> b ... d')

        # concat to channels of and flatten axis
        queries_orig_shape = ins.shape

        # rearrange input to be channel last
        ins = rearrange(ins, 'b ... d -> b (...) d')
        ins_wo_prev_layers = ins                            # [B, 125, 448]

        # option 2: add lang token embs as a sequence
        if self.lang_fusion_type == 'seq':
            l = self.lang_preprocess(lang_token_embs)         # [B,77,512] -> [B,77,128]
            ins = torch.cat((l, ins), dim=1)                  # [B,8077,128]


        # add pos encoding to language + flattened grid (the recommended way)
        if self.pos_encoding_with_lang:
            ins = ins + self.pos_encoding
        # batchify latents
        x = repeat(self.latents, 'n d -> b n d', b=b)

        cross_attn, cross_ff = self.cross_attend_blocks


        for it in range(self.iterations):
            # encoder cross attention
            x = cross_attn(x, context=ins, mask=None) + x
            x = cross_ff(x) + x

            # self-attention layers
            for self_attn, self_ff in self.layers:
                x = self_attn(x) + x
                x = self_ff(x) + x

        # decoder cross attention
        latents = self.decoder_cross_attn(ins, context=x)

        # crop out the language part of the output sequence
        if self.lang_fusion_type == 'seq':
            latents = latents[:, l.shape[1]:]

        # reshape back to voxel grid
        latents = latents.view(b, *queries_orig_shape[1:-1], latents.shape[-1]) # [B,20,20,20,64]
        latents = rearrange(latents, 'b ... d -> b d ...')                      # [B,64,20,20,20]

        # aggregated features from 2nd softmax and maxpool for MLP decoders
        feats.extend([self.ss1(latents.contiguous()), self.global_maxp(latents).view(b, -1)])

        # upsample
        u0 = self.up0(latents)

        # ablations
        if self.no_skip_connection:
            u = self.final(u0)
        elif self.no_perceiver:
            u = self.final(d0)
        else:
            u = self.final(torch.cat([d0, u0], dim=1))

        # latent u shape: [B, 64, 40, 40, 40]

        # TODO task embedding is only a mask
        # Train the model and then train the task embedding

        # concat task embed

        # task_embed = self.task_embed_nn(task_embed)
        # task_embed = task_embed.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.voxel_size, self.voxel_size, self.voxel_size)
        # task_embed_mask = self.task_mask_nn(task_embed)

        # apply task embed mask
        # u = u * task_embed_mask
        bs = u.shape[0]

        pos_cond_feature = self.pos_cond_encoder(pos_cond)

        if self.loss_type == "ce":
            # translation decoder
            trans = self.trans_decoder(u)
            rot_and_grip_out, collision_out = self.predict_rot_heatmap(u, feats, pos_cond_feature)

            task_pred = self.predict_task(u, feats)
        elif self.loss_type == "mse":
            # translation decoder
            trans = self.trans_decoder(u)
            trans = self.trans_ind_decoder(trans.view(bs, -1))

            rot_and_grip_out, collision_out = self.predict_rot_ind(u, feats, pos_cond_feature)
            task_pred = self.predict_task(u, feats)
        else:
            print("wrong loss type")
            exit()

        return  trans, rot_and_grip_out, collision_out, task_pred

    def predict_rot_ind(self, u, feats, pos_cond_feature):
        bs = u.shape[0]
        if self.num_rotation_classes > 0:
            feats.extend([self.ss_final(u.contiguous()), self.global_maxp(u).view(bs, -1), pos_cond_feature])
            dense0 = self.dense0(torch.cat(feats, dim=1))
            dense1 = self.dense1(dense0)
            rot_and_grip_collision_out = self.rot_grip_ind_decoder(dense1)

        return rot_and_grip_collision_out, None

    def predict_task(self, u, feats):
        bs = u.shape[0]
        feats.extend([self.ss_final(u.contiguous()), self.global_maxp(u).view(bs, -1)])
        dense0 = self.dense2(torch.cat(feats, dim=1))
        task_pred = self.task_pred_decoder(dense0)

        return task_pred


    def predict_rot_heatmap(self, u, feats, pos_cond_feature):
        bs = u.shape[0]
        # rotation, gripper, and collision MLPs
        rot_and_grip_out = None
        if self.num_rotation_classes > 0:
            feats.extend([self.ss_final(u.contiguous()), self.global_maxp(u).view(bs, -1),  pos_cond_feature])

            dense0 = self.dense0(torch.cat(feats, dim=1))
            dense1 = self.dense1(dense0)

            if "seq" in self.act_type:
                # TODO using LSTM may not predict seq action
                '''
                # Initialize the hidden state and cell state
                hidden = self.init_hidden(b, device)
                # Pass the input through the LSTM
                lstm_out, hidden = self.lstm(dense1.unsqueeze(1), hidden)
                outputs = [lstm_out[:, -1].unsqueeze(1)]
                # Generate future predictions based on previous predictions
                for _ in range(self.seq_horizon - 1):
                    lstm_out, hidden = self.lstm(outputs[-1], hidden)
                    outputs.append(lstm_out[:, -1].unsqueeze(1))

                outputs = torch.cat(outputs, dim=1)
                rot_and_grip_collision_out = self.lstm_fc(outputs.view(b * self.seq_horizon, -1)).view(bs, self.seq_horizon, -1)
                rot_and_grip_out = rot_and_grip_collision_out[:, :, :-self.num_collision_classes]
                collision_out = rot_and_grip_collision_out[:, :, -self.num_collision_classes:]
                '''

                rot_and_grip_collision_out = self.rot_grip_collision_ff(dense1)
                rot_and_grip_out = rot_and_grip_collision_out[:, :-self.num_collision_classes]
                collision_out = rot_and_grip_collision_out[:, -self.num_collision_classes:]


            else:
                rot_and_grip_collision_out = self.rot_grip_collision_ff(dense1)
                rot_and_grip_out = rot_and_grip_collision_out[:, :-self.num_collision_classes]
                collision_out = rot_and_grip_collision_out[:, -self.num_collision_classes:]

        # Add softmax in voxel prediction
        # (b, c, d, h, w) = trans.shape
        # trans = trans.view(b * c, d * h * w)
        # trans = self.softmax(trans)
        # trans = trans.view((b, c, d, h, w))

        return rot_and_grip_out, collision_out

