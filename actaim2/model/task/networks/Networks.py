"""
---------------------------------------------------------------------
-- Author: Jhosimar George Arias Figueroa
---------------------------------------------------------------------

Gaussian Mixture Variational Autoencoder Networks

"""
import torch
import torch.nn.init as init
from torch import nn
from torch.nn import functional as F
from new_scripts.model.task.networks.Layers import *
from new_scripts.model.task.multi_view_encoder import MultiViewEncoder
from new_scripts.model.task.multi_view_encoder import ResnetBlockFC


# Inference Network
class InferenceNet(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim):
        super(InferenceNet, self).__init__()

        self.cond_x_x_layer = ResnetBlockFC(x_dim * 2, x_dim)
        self.actn = nn.ReLU()


        # q(y|x)
        self.inference_qyx = torch.nn.ModuleList([
            ResnetBlockFC(x_dim, 512),
            nn.ReLU(),
            ResnetBlockFC(512, 512),
            nn.ReLU(),
            GumbelSoftmax(512, y_dim)
        ])

        # q(z|y,x)
        self.inference_qzyx = torch.nn.ModuleList([
            ResnetBlockFC(x_dim + y_dim, 512),
            nn.ReLU(),
            ResnetBlockFC(512, 512),
            nn.ReLU(),
            Gaussian(512, z_dim)
        ])

    # q(y|x)
    def qyx(self, x, temperature, hard):
        num_layers = len(self.inference_qyx)

        for i, layer in enumerate(self.inference_qyx):
            if i == num_layers - 1:
                # last layer is gumbel softmax
                x = layer(x, temperature, hard)
            else:
                x = layer(x)
        return x

    # q(z|x,y)
    def qzxy(self, x, y):
        concat = torch.cat((x, y), dim=1)
        for layer in self.inference_qzyx:
            concat = layer(concat)
        return concat

    def forward(self, x, cond_x, temperature=1.0, hard=0):
        # x = Flatten(x)

        cond_x_x = torch.cat((x, cond_x), dim=-1)
        x = self.cond_x_x_layer(cond_x_x)
        x = self.actn(x)

        # q(y|x, c)
        logits, prob, y = self.qyx(x, temperature, hard)

        # q(z|x,y, c)
        mu, var, z = self.qzxy(x, y)

        output = {'mean': mu, 'var': var, 'gaussian': z,
                  'logits': logits, 'prob_cat': prob, 'categorical': y}

        return output


# Generative Network
class GenerativeNet(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim):
        super(GenerativeNet, self).__init__()

        self.channel_num = 5

        self.z_cond_x_layer = ResnetBlockFC(z_dim + x_dim // self.channel_num, z_dim)

        # p(z|y)
        self.y_mu = nn.Linear(y_dim, z_dim)
        self.y_var = nn.Linear(y_dim, z_dim)

        # TODO attenion
        attn_layers_dim = z_dim
        mlp_dim = attn_layers_dim * 4,
        attn_layers_heads = 2
        max_seq_len = 5
        with_cross_attention = True
        attn_layers_depth = 2


        self.generative_pxz_transformer = torch.nn.ModuleList([
            TransformerBlock(
                hidden_size=attn_layers_dim,
                mlp_dim=attn_layers_dim * 4,
                num_heads=attn_layers_heads,
                dropout_rate=0.0,
                qkv_bias=False,
                causal=True,
                sequence_length=max_seq_len,
                with_cross_attention=with_cross_attention,
            )
            for _ in range(attn_layers_depth)
        ])


        self.generaive_pxz_fc = torch.nn.ModuleList([
            ResnetBlockFC(self.channel_num * z_dim, x_dim),
        ])

    # p(z|y)
    def pzy(self, y):
        y_mu = self.y_mu(y)
        y_var = F.softplus(self.y_var(y))
        return y_mu, y_var

    # p(x|z)
    def pxz(self, z):
        bs = z.shape[0]
        for layer in self.generative_pxz_transformer:
            z = layer(z)
        z = z.view(bs, -1)
        for layer in self.generaive_pxz_fc:
            z = layer(z)

        return z

    def pxz_cond(self, z, cond_x):
        bs = cond_x.shape[0]
        z = z.unsqueeze(1).repeat(1, self.channel_num, 1)

        z_cond_x = torch.cat((z, cond_x), dim=-1)

        z_cond_x = z_cond_x.view(bs * self.channel_num, -1)
        z_cond_x = self.z_cond_x_layer(z_cond_x)
        z_cond_x = z_cond_x.view(bs, self.channel_num, -1)

        # p(x|z)
        x_rec = self.pxz(z_cond_x)

        return x_rec


    def forward(self, z, y, cond_x):
        bs = cond_x.shape[0]
        cond_x = cond_x.view(bs, self.channel_num, -1)
        z = z.unsqueeze(1).repeat(1, self.channel_num, 1)

        z_cond_x = torch.cat((z, cond_x), dim=-1)

        z_cond_x = z_cond_x.view(bs * self.channel_num, -1)
        z_cond_x = self.z_cond_x_layer(z_cond_x)
        z_cond_x = z_cond_x.view(bs, self.channel_num, -1)

        # p(z|y)
        y_mu, y_var = self.pzy(y)

        # p(x|z)
        x_rec = self.pxz(z_cond_x)

        output = {'y_mean': y_mu, 'y_var': y_var, 'x_rec': x_rec}
        return output


# GMVAE Network
class GMVAENet(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim):
        super(GMVAENet, self).__init__()

        self.vision_encoder = MultiViewEncoder()

        self.inference = InferenceNet(x_dim, z_dim, y_dim)
        self.generative = GenerativeNet(x_dim, z_dim, y_dim)

        # weight initialization
        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None and m.bias.data is not None:
                    init.constant_(m.bias, 0)

        self.bn1 = nn.BatchNorm1d(5)

    def process_raw_image(self, x):
        x = self.vision_encoder(x)
        return x


    def forward(self, curr_x, final_x, temperature=1.0, hard=0):
        bs = curr_x.shape[0]
        curr_x = self.vision_encoder(curr_x)
        final_x = self.vision_encoder(final_x)

        x = (curr_x - final_x)
        # add batchnorm here
        x = self.bn1(x)
        cond_x = curr_x

        x = x.view(x.size(0), -1)
        cond_x = cond_x.view(cond_x.size(0), -1)

        out_inf = self.inference(x, cond_x, temperature, hard)
        z, y = out_inf['gaussian'], out_inf['categorical']
        out_gen = self.generative(z, y, cond_x)

        # merge output
        output = out_inf
        for key, value in out_gen.items():
            output[key] = value
        output["data"] = x.detach()
        output["final_x"] = final_x.detach()
        # output["data_train"] = x
        return output


    def inference_data(self, curr_x, final_x, temperature=1.0, hard=0):
        bs = curr_x.shape[0]
        curr_x = self.vision_encoder(curr_x)
        final_x = self.vision_encoder(final_x)

        x = (curr_x - final_x)

        cond_x = curr_x

        x = x.view(x.size(0), -1)
        cond_x = cond_x.view(cond_x.size(0), -1)

        out_inf = self.inference(x, cond_x, temperature, hard)

        return out_inf