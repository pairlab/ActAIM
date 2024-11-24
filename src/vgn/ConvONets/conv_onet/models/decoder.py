import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from vgn.ConvONets.layers import ResnetBlockFC
from vgn.ConvONets.common import normalize_coordinate, normalize_3d_coordinate, map2local
from vgn.ipdf.models import ImplicitSO3
import pdb


class FCDecoder(nn.Module):
    """Decoder.
        Instead of conditioning on global features, on plane/volume local features.
    Args:
    dim (int): input dimension
    c_dim (int): dimension of latent conditioned code c
    out_dim (int): dimension of latent conditioned code c
    leaky (bool): whether to use leaky ReLUs
    sample_mode (str): sampling feature strategy, bilinear|nearest
    padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    """

    def __init__(self, dim=3, c_dim=128, out_dim=1, leaky=False, sample_mode="bilinear", padding=0.1):
        super().__init__()
        self.c_dim = c_dim

        self.fc = nn.Linear(dim + c_dim, out_dim)
        self.sample_mode = sample_mode
        self.padding = padding

    def sample_plane_feature(self, p, c, plane="xz"):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding)  # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode="border", align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)  # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0  # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = (
            F.grid_sample(c, vgrid, padding_mode="border", align_corners=True, mode=self.sample_mode)
            .squeeze(-1)
            .squeeze(-1)
        )
        return c

    def forward(self, p, c_plane):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if "grid" in plane_type:
                c += self.sample_grid_feature(p, c_plane["grid"])
            if "xz" in plane_type:
                c += self.sample_plane_feature(p, c_plane["xz"], plane="xz")
            if "xy" in plane_type:
                c += self.sample_plane_feature(p, c_plane["xy"], plane="xy")
            if "yz" in plane_type:
                c += self.sample_plane_feature(p, c_plane["yz"], plane="yz")
            c = c.transpose(1, 2)

        net = self.fc(torch.cat((c, p), dim=2)).squeeze(-1)

        return net


class LocalDecoder(nn.Module):
    """Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    """

    def __init__(
        self,
        dim=3,
        c_dim=128,
        hidden_size=256,
        n_blocks=5,
        out_dim=1,
        leaky=False,
        sample_mode="bilinear",
        padding=0.1,
        concat_feat=False,
        no_xyz=False,
        add_noise=False,
    ):
        super().__init__()

        self.concat_feat = concat_feat
        if concat_feat:
            c_dim *= 3
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.no_xyz = no_xyz
        self.hidden_size = hidden_size
        self.add_noise = add_noise

        # add latent dim
        self.latent_dim = 8
        if self.add_noise:
            c_dim += self.latent_dim

        # TODO add latent z here
        c_dim += 256

        if c_dim != 0:
            self.fc_c = nn.ModuleList([nn.Linear(c_dim, hidden_size) for i in range(n_blocks)])

        if not no_xyz:
            self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([ResnetBlockFC(hidden_size) for i in range(n_blocks)])

        self.fc_out = nn.Linear(hidden_size, out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding

    def sample_plane_feature(self, p, c, plane="xz"):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding)  # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode="border", align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)  # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0  # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = (
            F.grid_sample(c, vgrid, padding_mode="border", align_corners=True, mode=self.sample_mode)
            .squeeze(-1)
            .squeeze(-1)
        )
        return c

    # TODO add latent z here
    def forward(self, p, c_plane, z, **kwargs):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            if self.concat_feat:
                c = []
                if "grid" in plane_type:
                    c = self.sample_grid_feature(p, c_plane["grid"])
                if "xz" in plane_type:
                    c.append(self.sample_plane_feature(p, c_plane["xz"], plane="xz"))
                if "xy" in plane_type:
                    c.append(self.sample_plane_feature(p, c_plane["xy"], plane="xy"))
                if "yz" in plane_type:
                    c.append(self.sample_plane_feature(p, c_plane["yz"], plane="yz"))
                c = torch.cat(c, dim=1)
                c = c.transpose(1, 2)
            else:
                c = 0
                if "grid" in plane_type:
                    c += self.sample_grid_feature(p, c_plane["grid"])
                if "xz" in plane_type:
                    c += self.sample_plane_feature(p, c_plane["xz"], plane="xz")
                if "xy" in plane_type:
                    c += self.sample_plane_feature(p, c_plane["xy"], plane="xy")
                if "yz" in plane_type:
                    c += self.sample_plane_feature(p, c_plane["yz"], plane="yz")
                c = c.transpose(1, 2)

        p = p.float()
        if self.no_xyz:
            net = torch.zeros(p.size(0), p.size(1), self.hidden_size).to(p.device)
        else:
            net = self.fc_p(p)

        if self.add_noise:
            batch_size = c.shape[0]
            gaussian_noise = torch.empty((batch_size, 1, self.latent_dim)).normal_(mean=0, std=0.1).to(p.device)
            c = torch.cat((c, gaussian_noise), -1)

        z = z[:, None, :]
        c = torch.cat((c, z), -1)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out

    def query_feature(self, p, c_plane):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if "grid" in plane_type:
                c += self.sample_grid_feature(p, c_plane["grid"])
            if "xz" in plane_type:
                c += self.sample_plane_feature(p, c_plane["xz"], plane="xz")
            if "xy" in plane_type:
                c += self.sample_plane_feature(p, c_plane["xy"], plane="xy")
            if "yz" in plane_type:
                c += self.sample_plane_feature(p, c_plane["yz"], plane="yz")
            c = c.transpose(1, 2)
        return c

    def compute_out(self, p, c):
        p = p.float()
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out

class LocalDecoderFeature(nn.Module):
    """Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    """

    def __init__(
        self,
        dim=3,
        c_dim=128,
        hidden_size=256,
        n_blocks=5,
        out_dim=1,
        leaky=False,
        sample_mode="bilinear",
        padding=0.1,
        concat_feat=False,
        no_xyz=False,
        add_noise=False,
    ):
        super().__init__()

        self.concat_feat = concat_feat
        if concat_feat:
            c_dim *= 3
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.no_xyz = no_xyz
        self.hidden_size = hidden_size
        self.add_noise = add_noise

        self.sample_mode = sample_mode
        self.padding = padding

    def sample_plane_feature(self, p, c, plane="xz"):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding)  # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode="border", align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)  # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0  # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = (
            F.grid_sample(c, vgrid, padding_mode="border", align_corners=True, mode=self.sample_mode)
            .squeeze(-1)
            .squeeze(-1)
        )
        return c

    def forward(self, p, c_plane):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            if self.concat_feat:
                c = []
                if "grid" in plane_type:
                    c = self.sample_grid_feature(p, c_plane["grid"])
                if "xz" in plane_type:
                    c.append(self.sample_plane_feature(p, c_plane["xz"], plane="xz"))
                if "xy" in plane_type:
                    c.append(self.sample_plane_feature(p, c_plane["xy"], plane="xy"))
                if "yz" in plane_type:
                    c.append(self.sample_plane_feature(p, c_plane["yz"], plane="yz"))
                c = torch.cat(c, dim=1)
                c = c.transpose(1, 2)
            else:
                c = 0
                if "grid" in plane_type:
                    c += self.sample_grid_feature(p, c_plane["grid"])
                if "xz" in plane_type:
                    c += self.sample_plane_feature(p, c_plane["xz"], plane="xz")
                if "xy" in plane_type:
                    c += self.sample_plane_feature(p, c_plane["xy"], plane="xy")
                if "yz" in plane_type:
                    c += self.sample_plane_feature(p, c_plane["yz"], plane="yz")
                c = c.transpose(1, 2)

        return c



class LocalDecoderScore(nn.Module):
    """Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    """

    def __init__(
        self,
        dim=3,
        c_dim=128,
        hidden_size=256,
        n_blocks=5,
        out_dim=1,
        leaky=False,
        sample_mode="bilinear",
        padding=0.1,
        concat_feat=False,
        no_xyz=False,
        add_noise=False,
    ):
        super().__init__()

        self.concat_feat = concat_feat
        if concat_feat:
            c_dim *= 3
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.no_xyz = no_xyz
        self.hidden_size = hidden_size
        self.add_noise = add_noise

        # add latent dim
        self.latent_dim = 8
        if self.add_noise:
            c_dim += self.latent_dim

        # TODO add latent z here
        c_dim += 256

        if c_dim != 0:
            self.fc_c = nn.ModuleList([nn.Linear(c_dim, hidden_size) for i in range(n_blocks)])

        if not no_xyz:
            self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([ResnetBlockFC(hidden_size) for i in range(n_blocks)])

        self.fc_out = nn.Linear(hidden_size, out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding

    def sample_plane_feature(self, p, c, plane="xz"):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding)  # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode="border", align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)  # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0  # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = (
            F.grid_sample(c, vgrid, padding_mode="border", align_corners=True, mode=self.sample_mode)
            .squeeze(-1)
            .squeeze(-1)
        )
        return c

    # TODO add latent z here
    def forward(self, p, c_plane, z, **kwargs):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            if self.concat_feat:
                c = []
                if "grid" in plane_type:
                    c = self.sample_grid_feature(p, c_plane["grid"])
                if "xz" in plane_type:
                    c.append(self.sample_plane_feature(p, c_plane["xz"], plane="xz"))
                if "xy" in plane_type:
                    c.append(self.sample_plane_feature(p, c_plane["xy"], plane="xy"))
                if "yz" in plane_type:
                    c.append(self.sample_plane_feature(p, c_plane["yz"], plane="yz"))
                c = torch.cat(c, dim=1)
                c = c.transpose(1, 2)
            else:
                c = 0
                if "grid" in plane_type:
                    c += self.sample_grid_feature(p, c_plane["grid"])
                if "xz" in plane_type:
                    c += self.sample_plane_feature(p, c_plane["xz"], plane="xz")
                if "xy" in plane_type:
                    c += self.sample_plane_feature(p, c_plane["xy"], plane="xy")
                if "yz" in plane_type:
                    c += self.sample_plane_feature(p, c_plane["yz"], plane="yz")
                c = c.transpose(1, 2)

        p = p.float()
        if self.no_xyz:
            net = torch.zeros(p.size(0), p.size(1), self.hidden_size).to(p.device)
        else:
            net = self.fc_p(p)

        if self.add_noise:
            batch_size = c.shape[0]
            gaussian_noise = torch.empty((batch_size, 1, self.latent_dim)).normal_(mean=0, std=0.1).to(p.device)
            c = torch.cat((c, gaussian_noise), -1)

        pdb.set_trace()

        z = z[:, None, :]
        c = torch.cat((c, z), -1)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out

    def query_feature(self, p, c_plane):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if "grid" in plane_type:
                c += self.sample_grid_feature(p, c_plane["grid"])
            if "xz" in plane_type:
                c += self.sample_plane_feature(p, c_plane["xz"], plane="xz")
            if "xy" in plane_type:
                c += self.sample_plane_feature(p, c_plane["xy"], plane="xy")
            if "yz" in plane_type:
                c += self.sample_plane_feature(p, c_plane["yz"], plane="yz")
            c = c.transpose(1, 2)
        return c

    def compute_out(self, p, c):
        p = p.float()
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out

class LocalDecoderImage(nn.Module):
    """Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    """

    def __init__(
        self,
        dim=2,
        c_dim=128,
        hidden_size=256,
        n_blocks=5,
        out_dim=1,
        leaky=False,
        sample_mode="bilinear",
        padding=0.1,
        concat_feat=False,
        no_xyz=False,
        add_noise=False,
    ):
        super().__init__()

        self.concat_feat = concat_feat
        if concat_feat:
            c_dim *= 1
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.no_xyz = no_xyz
        self.hidden_size = hidden_size
        self.add_noise = add_noise

        # add latent dim
        self.latent_dim = 8
        if self.add_noise:
            c_dim += self.latent_dim

        # TODO add latent z here
        c_dim += 256

        if c_dim != 0:
            self.fc_c = nn.ModuleList([nn.Linear(c_dim, hidden_size) for i in range(n_blocks)])

        if not no_xyz:
            self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([ResnetBlockFC(hidden_size) for i in range(n_blocks)])

        self.fc_out = nn.Linear(hidden_size, out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding

    def sample_plane_feature(self, p, c, plane="xz"):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding)  # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0  # normalize to (-
        c = F.grid_sample(c, vgrid, padding_mode="border", align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)  # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0  # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = (
            F.grid_sample(c, vgrid, padding_mode="border", align_corners=True, mode=self.sample_mode)
            .squeeze(-1)
            .squeeze(-1)
        )
        return c

    # TODO add latent z here
    def forward(self, p, c_plane, z, **kwargs):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            if self.concat_feat:
                c = []
                if "grid" in plane_type:
                    c = self.sample_grid_feature(p, c_plane["grid"])
                if "xz" in plane_type:
                    c.append(self.sample_plane_feature(p, c_plane["xz"], plane="xz"))
                if "xy" in plane_type:
                    c.append(self.sample_plane_feature(p, c_plane["xy"], plane="xy"))
                if "yz" in plane_type:
                    c.append(self.sample_plane_feature(p, c_plane["yz"], plane="yz"))
                c = torch.cat(c, dim=1)
                c = c.transpose(1, 2)
            else:
                c = 0
                if "grid" in plane_type:
                    c += self.sample_grid_feature(p, c_plane["grid"])
                if "xz" in plane_type:
                    c += self.sample_plane_feature(p, c_plane["xz"], plane="xz")
                if "xy" in plane_type:
                    c += self.sample_plane_feature(p, c_plane["xy"], plane="xy")
                if "yz" in plane_type:
                    c += self.sample_plane_feature(p, c_plane["yz"], plane="yz")
                c = c.transpose(1, 2)

        p = p.float()

        if self.no_xyz:
            net = torch.zeros(p.size(0), p.size(1), self.hidden_size).to(p.device)
        else:
            net = self.fc_p(p)

        if self.add_noise:
            batch_size = c.shape[0]
            gaussian_noise = torch.empty((batch_size, 1, self.latent_dim)).normal_(mean=0, std=0.1).to(p.device)
            c = torch.cat((c, gaussian_noise), -1)

        z = z[:, None, :]
        c = torch.cat((c, z), -1)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out

    def query_feature(self, p, c_plane):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if "grid" in plane_type:
                c += self.sample_grid_feature(p, c_plane["grid"])
            if "xz" in plane_type:
                c += self.sample_plane_feature(p, c_plane["xz"], plane="xz")
            if "xy" in plane_type:
                c += self.sample_plane_feature(p, c_plane["xy"], plane="xy")
            if "yz" in plane_type:
                c += self.sample_plane_feature(p, c_plane["yz"], plane="yz")
            c = c.transpose(1, 2)
        return c

    def compute_out(self, p, c):
        p = p.float()
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out


class LocalDecoderIpdf(nn.Module):
    """Decoder.
        decode both rotation and force vector

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    """

    def __init__(
        self,
        dim=3,
        c_dim=128,
        hidden_size=256,
        n_blocks=5,
        out_dim=1,
        leaky=False,
        sample_mode="bilinear",
        padding=0.1,
        concat_feat=False,
        no_xyz=False,
    ):
        super().__init__()

        self.concat_feat = concat_feat
        if concat_feat:
            c_dim *= 3
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.no_xyz = no_xyz
        self.hidden_size = hidden_size

        # Define ImplicitSO3 params
        number_fourier_components = 1
        head_network_specs = 256
        so3_sampling_mode = "grid"
        # memory would explode if 2**12
        number_train_queries_rotation = 2**8
        number_train_queries_xyz = 2**6
        number_eval_queries = 2**16

        self.ipdf_model = ImplicitSO3(
            hidden_size,
            number_fourier_components,
            head_network_specs,
            so3_sampling_mode,
            number_train_queries_rotation,
            number_train_queries_xyz,
            number_eval_queries,
        )

        if c_dim != 0:
            self.fc_c = nn.ModuleList([nn.Linear(c_dim, hidden_size) for i in range(n_blocks)])

        if not no_xyz:
            self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([ResnetBlockFC(hidden_size) for i in range(n_blocks)])

        self.fc_out = nn.Linear(hidden_size, out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding

    def sample_plane_feature(self, p, c, plane="xz"):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding)  # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode="border", align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)  # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0  # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = (
            F.grid_sample(c, vgrid, padding_mode="border", align_corners=True, mode=self.sample_mode)
            .squeeze(-1)
            .squeeze(-1)
        )
        return c

    def forward(self, p, c_plane, r, f, **kwargs):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            if self.concat_feat:
                c = []
                if "grid" in plane_type:
                    c = self.sample_grid_feature(p, c_plane["grid"])
                if "xz" in plane_type:
                    c.append(self.sample_plane_feature(p, c_plane["xz"], plane="xz"))
                if "xy" in plane_type:
                    c.append(self.sample_plane_feature(p, c_plane["xy"], plane="xy"))
                if "yz" in plane_type:
                    c.append(self.sample_plane_feature(p, c_plane["yz"], plane="yz"))
                c = torch.cat(c, dim=1)
                c = c.transpose(1, 2)
            else:
                c = 0
                if "grid" in plane_type:
                    c += self.sample_grid_feature(p, c_plane["grid"])
                if "xz" in plane_type:
                    c += self.sample_plane_feature(p, c_plane["xz"], plane="xz")
                if "xy" in plane_type:
                    c += self.sample_plane_feature(p, c_plane["xy"], plane="xy")
                if "yz" in plane_type:
                    c += self.sample_plane_feature(p, c_plane["yz"], plane="yz")
                c = c.transpose(1, 2)
        p = p.float()
        if self.no_xyz:
            net = torch.zeros(p.size(0), p.size(1), self.hidden_size).to(p.device)
        else:
            net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        # TODO apply implicit pdf here
        prob = self.ipdf_model.predict_probability(net, r, f)
        out = prob[:, None]

        return out

    def query_feature(self, p, c_plane):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if "grid" in plane_type:
                c += self.sample_grid_feature(p, c_plane["grid"])
            if "xz" in plane_type:
                c += self.sample_plane_feature(p, c_plane["xz"], plane="xz")
            if "xy" in plane_type:
                c += self.sample_plane_feature(p, c_plane["xy"], plane="xy")
            if "yz" in plane_type:
                c += self.sample_plane_feature(p, c_plane["yz"], plane="yz")
            c = c.transpose(1, 2)
        return c

    def compute_out(self, p, c):
        p = p.float()
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out


class LocalDecoderPair(nn.Module):
    """Decoder.
        decode both rotation and force vector

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    """

    def __init__(
        self,
        dim=3,
        c_dim=128,
        hidden_size=256,
        n_blocks=5,
        out_dim=1,
        leaky=False,
        sample_mode="bilinear",
        padding=0.1,
        concat_feat=False,
        no_xyz=False,
    ):
        super().__init__()

        self.concat_feat = concat_feat
        if concat_feat:
            c_dim *= 3
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.no_xyz = no_xyz
        self.hidden_size = hidden_size

        # TODO add rotation and force encoder
        self.r_encode = nn.Linear(8, hidden_size)
        self.f_encode = nn.Linear(3, hidden_size)

        self.r_res = ResnetBlockFC(hidden_size)
        self.f_res = ResnetBlockFC(hidden_size)

        self.joint_encode = nn.Linear(3 * hidden_size, hidden_size)

        if c_dim != 0:
            self.fc_c = nn.ModuleList([nn.Linear(c_dim, hidden_size) for i in range(n_blocks)])

        if not no_xyz:
            self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([ResnetBlockFC(hidden_size) for i in range(n_blocks)])

        self.fc_out = nn.Linear(hidden_size, out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding
        self.sigmoid = nn.Sigmoid()

    def sample_plane_feature(self, p, c, plane="xz"):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding)  # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode="border", align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)  # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0  # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = (
            F.grid_sample(c, vgrid, padding_mode="border", align_corners=True, mode=self.sample_mode)
            .squeeze(-1)
            .squeeze(-1)
        )
        return c

    def forward(self, p, c_plane, r, f, **kwargs):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            if self.concat_feat:
                c = []
                if "grid" in plane_type:
                    c = self.sample_grid_feature(p, c_plane["grid"])
                if "xz" in plane_type:
                    c.append(self.sample_plane_feature(p, c_plane["xz"], plane="xz"))
                if "xy" in plane_type:
                    c.append(self.sample_plane_feature(p, c_plane["xy"], plane="xy"))
                if "yz" in plane_type:
                    c.append(self.sample_plane_feature(p, c_plane["yz"], plane="yz"))
                c = torch.cat(c, dim=1)
                c = c.transpose(1, 2)
            else:
                c = 0
                if "grid" in plane_type:
                    c += self.sample_grid_feature(p, c_plane["grid"])
                if "xz" in plane_type:
                    c += self.sample_plane_feature(p, c_plane["xz"], plane="xz")
                if "xy" in plane_type:
                    c += self.sample_plane_feature(p, c_plane["xy"], plane="xy")
                if "yz" in plane_type:
                    c += self.sample_plane_feature(p, c_plane["yz"], plane="yz")
                c = c.transpose(1, 2)
        p = p.float()
        if self.no_xyz:
            net = torch.zeros(p.size(0), p.size(1), self.hidden_size).to(p.device)
        else:
            net = self.fc_p(p)

        f = f.unsqueeze(-2)
        batch_size = net.shape[0]
        r = r.view(batch_size, 1, -1)
        r = self.r_encode(r)
        f = self.f_encode(f)

        r_ft = self.r_res(r)
        f_ft = self.f_res(f)

        net = torch.cat((net, r_ft, f_ft), dim=-1)

        net = self.joint_encode(net)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = self.sigmoid(out)
        out = out.squeeze(-1)

        return out

    def query_feature(self, p, c_plane):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if "grid" in plane_type:
                c += self.sample_grid_feature(p, c_plane["grid"])
            if "xz" in plane_type:
                c += self.sample_plane_feature(p, c_plane["xz"], plane="xz")
            if "xy" in plane_type:
                c += self.sample_plane_feature(p, c_plane["xy"], plane="xy")
            if "yz" in plane_type:
                c += self.sample_plane_feature(p, c_plane["yz"], plane="yz")
            c = c.transpose(1, 2)
        return c

    def compute_out(self, p, c):
        p = p.float()
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = self.sigmoid(out)
        out = out.squeeze(-1)

        return out


class PatchLocalDecoder(nn.Module):
    """Decoder adapted for crop training.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        local_coord (bool): whether to use local coordinate
        unit_size (float): defined voxel unit size for local system
        pos_encoding (str): method for the positional encoding, linear|sin_cos
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]

    """

    def __init__(
        self,
        dim=3,
        c_dim=128,
        hidden_size=256,
        leaky=False,
        n_blocks=5,
        sample_mode="bilinear",
        local_coord=False,
        pos_encoding="linear",
        unit_size=0.1,
        padding=0.1,
    ):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        if c_dim != 0:
            self.fc_c = nn.ModuleList([nn.Linear(c_dim, hidden_size) for i in range(n_blocks)])

        # self.fc_p = nn.Linear(dim, hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.blocks = nn.ModuleList([ResnetBlockFC(hidden_size) for i in range(n_blocks)])

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode

        if local_coord:
            self.map2local = map2local(unit_size, pos_encoding=pos_encoding)
        else:
            self.map2local = None

        if pos_encoding == "sin_cos":
            self.fc_p = nn.Linear(60, hidden_size)
        else:
            self.fc_p = nn.Linear(dim, hidden_size)

    def sample_feature(self, xy, c, fea_type="2d"):
        if fea_type == "2d":
            xy = xy[:, :, None].float()
            vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
            c = F.grid_sample(c, vgrid, padding_mode="border", align_corners=True, mode=self.sample_mode).squeeze(-1)
        else:
            xy = xy[:, :, None, None].float()
            vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
            c = (
                F.grid_sample(c, vgrid, padding_mode="border", align_corners=True, mode=self.sample_mode)
                .squeeze(-1)
                .squeeze(-1)
            )
        return c

    def forward(self, p, c_plane, **kwargs):
        p_n = p["p_n"]
        p = p["p"]

        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if "grid" in plane_type:
                c += self.sample_feature(p_n["grid"], c_plane["grid"], fea_type="3d")
            if "xz" in plane_type:
                c += self.sample_feature(p_n["xz"], c_plane["xz"])
            if "xy" in plane_type:
                c += self.sample_feature(p_n["xy"], c_plane["xy"])
            if "yz" in plane_type:
                c += self.sample_feature(p_n["yz"], c_plane["yz"])
            c = c.transpose(1, 2)

        p = p.float()
        if self.map2local:
            p = self.map2local(p)

        net = self.fc_p(p)
        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)
            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out


class LocalPointDecoder(nn.Module):
    """Decoder for PointConv Baseline.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        n_blocks (int): number of blocks ResNetBlockFC layers
        sample_mode (str): sampling mode  for points
    """

    def __init__(self, dim=3, c_dim=128, hidden_size=256, leaky=False, n_blocks=5, sample_mode="gaussian", **kwargs):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        if c_dim != 0:
            self.fc_c = nn.ModuleList([nn.Linear(c_dim, hidden_size) for i in range(n_blocks)])

        self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([ResnetBlockFC(hidden_size) for i in range(n_blocks)])

        self.fc_out = nn.Linear(hidden_size, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        if sample_mode == "gaussian":
            self.var = kwargs["gaussian_val"] ** 2

    def sample_point_feature(self, q, p, fea):
        # q: B x M x 3
        # p: B x N x 3
        # fea: B x N x c_dim
        # p, fea = c
        if self.sample_mode == "gaussian":
            # distance betweeen each query point to the point cloud
            dist = -(((p.unsqueeze(1).expand(-1, q.size(1), -1, -1) - q.unsqueeze(2)).norm(dim=3) + 10e-6) ** 2)
            weight = (dist / self.var).exp()  # Guassian kernel
        else:
            weight = 1 / ((p.unsqueeze(1).expand(-1, q.size(1), -1, -1) - q.unsqueeze(2)).norm(dim=3) + 10e-6)

        # weight normalization
        weight = weight / weight.sum(dim=2).unsqueeze(-1)

        c_out = weight @ fea  # B x M x c_dim

        return c_out

    def forward(self, p, c, **kwargs):
        n_points = p.shape[1]

        if n_points >= 30000:
            pp, fea = c
            c_list = []
            for p_split in torch.split(p, 10000, dim=1):
                if self.c_dim != 0:
                    c_list.append(self.sample_point_feature(p_split, pp, fea))
            c = torch.cat(c_list, dim=1)

        else:
            if self.c_dim != 0:
                pp, fea = c
                c = self.sample_point_feature(p, pp, fea)

        p = p.float()
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out


class PointDecoder(nn.Module):
    """Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    """

    def __init__(
        self,
        dim=3,
        c_dim=128,
        hidden_size=256,
        n_blocks=5,
        out_dim=3,
        leaky=False,
        padding=0.1,
        no_xyz=False,
    ):
        super().__init__()

        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.no_xyz = no_xyz
        self.hidden_size = hidden_size

        # TODO add latent z here
        c_dim += 256

        if c_dim != 0:
            self.fc_c = nn.ModuleList([nn.Linear(c_dim, hidden_size) for i in range(n_blocks)])


        self.fc_z = nn.Linear(256, hidden_size)

        self.blocks = nn.ModuleList([ResnetBlockFC(hidden_size) for i in range(n_blocks)])

        self.fc_out = nn.Linear(hidden_size, out_dim)

        self.conv_in0 = nn.Conv3d(3, 3, 5)
        self.conv_in1 = nn.Conv3d(3, 3, 5)
        self.conv_in2 = nn.Conv3d(3, 1, 5, stride = 2)

        self.fc_c_pre = nn.Linear(1960, 128)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.padding = padding

    # TODO add latent z here
    def forward(self, c_plane, z, **kwargs):
        batch_size = z.shape[0]
        c_plane_xy = c_plane['xy']
        c_plane_xz = c_plane['xz']
        c_plane_yz = c_plane['yz']
        c = torch.stack((c_plane_xy, c_plane_yz, c_plane_xz), dim=1)

        c = self.actvn(self.conv_in0(c))
        c = self.actvn(self.conv_in1(c))
        c = self.actvn(self.conv_in2(c)).view(batch_size, -1)
        c = self.actvn(self.fc_c_pre(c))


        z = z.float()
        net = self.fc_z(z)
        c = torch.cat((c, z), -1)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)
        out = torch.sigmoid(out) - 0.5
        return out
