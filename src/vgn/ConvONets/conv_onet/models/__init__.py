import pdb

import torch
import torch.nn as nn
from torch import distributions as dist
from vgn.ConvONets.conv_onet.models import decoder
from vgn.ConvONets.layers import ResnetBlockFC

# Decoder dictionary
decoder_dict = {
    "simple_fc": decoder.FCDecoder,
    "simple_local": decoder.LocalDecoder,
    "simple_local_crop": decoder.PatchLocalDecoder,
    "simple_local_point": decoder.LocalPointDecoder,
    "simple_local_pair": decoder.LocalDecoderPair,
    "simple_local_ipdf": decoder.LocalDecoderIpdf,
    "simple_local_image": decoder.LocalDecoderImage,
    "point_decoder": decoder.PointDecoder,
    "simple_local_score": decoder.LocalDecoderScore,
    "simple_local_feature": decoder.LocalDecoderFeature,

}


class ConvolutionalOccupancyNetwork(nn.Module):
    """Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    """

    def __init__(self, decoders, encoder=None, device=None, detach_tsdf=False):
        super().__init__()

        self.decoder_qual = decoders[0].to(device)
        self.decoder_rot = decoders[1].to(device)
        self.decoder_force = decoders[2].to(device)

        if len(decoders) == 4:
            self.decoder_tsdf = decoders[3].to(device)

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

        self._device = device

        self.detach_tsdf = detach_tsdf

        self.point_predict = True
        if self.point_predict:
            self.point_decoder = decoder.PointDecoder().to(device)

    # TODO add latent z here
    def forward(self, inputs, p, z, p_tsdf=None, sample=True, **kwargs):
        """Performs a forward pass through the network.

        Args:
            p (tensor): sampled points, B*N*C
            inputs (tensor): conditioning input, B*N*3
            sample (bool): whether to sample for z
            p_tsdf (tensor): tsdf query points, B*N_P*3
        """
        #############
        if isinstance(p, dict):
            batch_size = p["p"].size(0)
        else:
            batch_size = p.size(0)
        c = self.encode_inputs(inputs)
        # feature = self.query_feature(p, c)
        # qual, rot, force = self.decode_feature(p, feature)

        # TODO add latent z here
        qual, rot, force = self.decode(p, c, z)

        if self.point_predict:
            qual = self.point_decoder(c, z)

        if p_tsdf is not None:
            if self.detach_tsdf:
                for k, v in c.items():
                    c[k] = v.detach()
            tsdf = self.decoder_tsdf(p_tsdf, c, **kwargs)

            return qual, rot, force, tsdf
        else:
            return qual, rot, force

    def infer_geo(self, inputs, p_tsdf, **kwargs):
        c = self.encode_inputs(inputs)
        tsdf = self.decoder_tsdf(p_tsdf, c, **kwargs)
        return tsdf

    def encode_inputs(self, inputs):
        """Encodes the input.

        Args:
            input (tensor): the input
        """

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            # Return inputs?
            c = torch.empty(inputs.size(0), 0)

        return c

    def query_feature(self, p, c):
        return self.decoder_qual.query_feature(p, c)

    def decode_feature(self, p, feature):
        qual = self.decoder_qual.compute_out(p, feature)
        qual = torch.sigmoid(qual)
        rot = self.decoder_rot.compute_out(p, feature)
        rot = nn.functional.normalize(rot, dim=2)
        force = self.decoder_force.compute_out(p, feature)
        force = torch.sigmoid(force)
        return qual, rot, force

    def decode_occ(self, p, c, **kwargs):
        """Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        """

        logits = self.decoder_tsdf(p, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def decode(self, p, c, z, **kwargs):
        """Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        """

        qual = self.decoder_qual(p, c, z, **kwargs)
        qual = torch.sigmoid(qual)

        rot = self.decoder_rot(p, c, z, **kwargs)
        rot = nn.functional.normalize(rot, dim=2)
        force = self.decoder_force(p, c, z, **kwargs)
        return qual, rot, force

    def to(self, device):
        """Puts the model to the device.

        Args:
            device (device): pytorch device
        """
        model = super().to(device)
        model._device = device
        return model

    def grad_refine(self, x, pos, bound_value=0.0125, lr=1e-6, num_step=1):
        pos_tmp = pos.clone()
        l_bound = pos - bound_value
        u_bound = pos + bound_value
        pos_tmp.requires_grad = True
        optimizer = torch.optim.SGD([pos_tmp], lr=lr)
        self.eval()
        for p in self.parameters():
            p.requres_grad = False
        for _ in range(num_step):
            optimizer.zero_grad()
            qual_out, _, _ = self.forward(x, pos_tmp)
            # print(qual_out)
            loss = -qual_out.sum()
            loss.backward()
            optimizer.step()
            # print(qual_out.mean().item())
        with torch.no_grad():
            # print(pos, pos_tmp)
            pos_tmp = torch.maximum(torch.minimum(pos_tmp, u_bound), l_bound)
            qual_out, rot_out, force_out = self.forward(x, pos_tmp)
            # print(pos, pos_tmp, qual_out)
            # print(qual_out.mean().item())
        # import pdb; pdb.set_trace()
        # self.train()
        for p in self.parameters():
            p.requres_grad = True
        # import pdb; pdb.set_trace()
        return qual_out, pos_tmp, rot_out, force_out


class ConvolutionalOccupancyNetworkGeometry(nn.Module):
    def __init__(self, decoder, encoder=None, device=None):
        super().__init__()

        self.decoder_tsdf = decoder.to(device)

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

        self._device = device

    def forward(self, inputs, p, p_tsdf, sample=True, **kwargs):
        """Performs a forward pass through the network.

        Args:
            p (tensor): sampled points, B*N*C
            inputs (tensor): conditioning input, B*N*3
            sample (bool): whether to sample for z
            p_tsdf (tensor): tsdf query points, B*N_P*3
        """
        #############
        if isinstance(p, dict):
            batch_size = p["p"].size(0)
        else:
            batch_size = p.size(0)
        c = self.encode_inputs(inputs)
        tsdf = self.decoder_tsdf(p_tsdf, c, **kwargs)
        return tsdf

    def infer_geo(self, inputs, p_tsdf, **kwargs):
        c = self.encode_inputs(inputs)
        tsdf = self.decoder_tsdf(p_tsdf, c, **kwargs)
        return tsdf

    def encode_inputs(self, inputs):
        """Encodes the input.

        Args:
            input (tensor): the input
        """

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            # Return inputs?
            c = torch.empty(inputs.size(0), 0)

        return c

    def decode_occ(self, p, c, **kwargs):
        """Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        """

        logits = self.decoder_tsdf(p, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r


class ConvolutionalOccupancyNetworkPair(nn.Module):
    """Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    """

    def __init__(self, decoders, encoder=None, device=None, detach_tsdf=False):
        super().__init__()

        self.decoder_qual = decoders[0].to(device)

        # self.channel_res = ResnetBlockFC(4)
        # self.channel_encode = nn.Linear(4, 1)
        if len(decoders) == 4:
            self.decoder_tsdf = decoders[3].to(device)

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

        self._device = device

        self.detach_tsdf = detach_tsdf


    def forward(self, inputs, p, p_tsdf=None, sample=True, **kwargs):
        """Performs a forward pass through the network.

        Args:
            p (tensor): sampled points, B*N*C
            inputs (tensor): conditioning input, B*N*3
            sample (bool): whether to sample for z
            p_tsdf (tensor): tsdf query points, B*N_P*3
        """
        #############

        pc, rotations, force = inputs

        if isinstance(p, dict):
            batch_size = p["p"].size(0)
        else:
            batch_size = p.size(0)
        c = self.encode_inputs(pc)

        # feature = self.query_feature(p, c)
        # qual, rot, force = self.decode_feature(p, feature)

        qual = self.decode(p, c, rotations, force)

        if p_tsdf is not None:
            if self.detach_tsdf:
                for k, v in c.items():
                    c[k] = v.detach()
            tsdf = self.decoder_tsdf(p_tsdf, c, **kwargs)

            return qual
        else:
            return qual

    def infer_geo(self, inputs, p_tsdf, **kwargs):
        c = self.encode_inputs(inputs)
        tsdf = self.decoder_tsdf(p_tsdf, c, **kwargs)
        return tsdf

    def encode_inputs(self, inputs):
        """Encodes the input.

        Args:
            input (tensor): the input
        """

        if inputs.shape[-1] == 4:
            # inputs = self.channel_res(inputs)
            # inputs = self.channel_encode(inputs)
            inputs = inputs.squeeze()

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            # Return inputs?
            c = torch.empty(inputs.size(0), 0)

        return c

    def query_feature(self, p, c):
        return self.decoder_qual.query_feature(p, c)

    def decode_feature(self, p, feature):
        qual = self.decoder_qual.compute_out(p, feature)
        qual = torch.sigmoid(qual)
        rot = self.decoder_rot.compute_out(p, feature)
        rot = nn.functional.normalize(rot, dim=2)
        force = self.decoder_force.compute_out(p, feature)
        force = torch.sigmoid(force)
        return qual, rot, force

    def decode_occ(self, p, c, **kwargs):
        """Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        """

        logits = self.decoder_tsdf(p, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def decode(self, p, c, r, f, **kwargs):
        """Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        """

        qual = self.decoder_qual(p, c, r, f, **kwargs)

        # TODO for ipdf, we don't want sigmoid here
        # qual = torch.sigmoid(qual)

        return qual

    def to(self, device):
        """Puts the model to the device.

        Args:
            device (device): pytorch device
        """
        model = super().to(device)
        model._device = device
        return model

    def grad_refine(self, x, pos, bound_value=0.0125, lr=1e-6, num_step=1):
        pos_tmp = pos.clone()
        l_bound = pos - bound_value
        u_bound = pos + bound_value
        pos_tmp.requires_grad = True
        optimizer = torch.optim.SGD([pos_tmp], lr=lr)
        self.eval()
        for p in self.parameters():
            p.requres_grad = False
        for _ in range(num_step):
            optimizer.zero_grad()
            qual_out, _, _ = self.forward(x, pos_tmp)
            # print(qual_out)
            loss = -qual_out.sum()
            loss.backward()
            optimizer.step()
            # print(qual_out.mean().item())
        with torch.no_grad():
            # print(pos, pos_tmp)
            pos_tmp = torch.maximum(torch.minimum(pos_tmp, u_bound), l_bound)
            qual_out, rot_out, force_out = self.forward(x, pos_tmp)
            # print(pos, pos_tmp, qual_out)
            # print(qual_out.mean().item())
        # import pdb; pdb.set_trace()
        # self.train()
        for p in self.parameters():
            p.requres_grad = True
        # import pdb; pdb.set_trace()
        return qual_out, pos_tmp, rot_out, force_out

class ConvolutionalOccupancyNetworkScore(nn.Module):
    """Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    """

    def __init__(self, decoders, encoder=None, device=None, detach_tsdf=False):
        super().__init__()

        self.decoder = decoders[0].to(device)

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

        self._device = device

        self.detach_tsdf = detach_tsdf

        self.point_predict = True
        if self.point_predict:
            self.point_decoder = decoder.PointDecoder().to(device)

    # TODO add latent z here
    def forward(self, inputs, p, z, p_tsdf=None, sample=True, **kwargs):
        """Performs a forward pass through the network.

        Args:
            p (tensor): sampled points, B*N*C
            inputs (tensor): conditioning input, B*N*3
            sample (bool): whether to sample for z
            p_tsdf (tensor): tsdf query points, B*N_P*3
        """
        #############
        if isinstance(p, dict):
            batch_size = p["p"].size(0)
        else:
            batch_size = p.size(0)
        c = self.encode_inputs(inputs)
        # feature = self.query_feature(p, c)
        # qual, rot, force = self.decode_feature(p, feature)

        # TODO add latent z here
        qual, rot, force = self.decode(p, c, z)

        # TODO
        exit()


        if self.point_predict:
            qual = self.point_decoder(c, z)

        if p_tsdf is not None:
            if self.detach_tsdf:
                for k, v in c.items():
                    c[k] = v.detach()
            tsdf = self.decoder_tsdf(p_tsdf, c, **kwargs)

            return qual, rot, force, tsdf
        else:
            return qual, rot, force

    def infer_geo(self, inputs, p_tsdf, **kwargs):
        c = self.encode_inputs(inputs)
        tsdf = self.decoder_tsdf(p_tsdf, c, **kwargs)
        return tsdf

    def encode_inputs(self, inputs):
        """Encodes the input.

        Args:
            input (tensor): the input
        """

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            # Return inputs?
            c = torch.empty(inputs.size(0), 0)

        return c

    def query_feature(self, p, c):
        return self.decoder_qual.query_feature(p, c)

    def decode_feature(self, p, feature):
        qual = self.decoder_qual.compute_out(p, feature)
        qual = torch.sigmoid(qual)
        rot = self.decoder_rot.compute_out(p, feature)
        rot = nn.functional.normalize(rot, dim=2)
        force = self.decoder_force.compute_out(p, feature)
        force = torch.sigmoid(force)
        return qual, rot, force

    def decode_occ(self, p, c, **kwargs):
        """Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        """

        logits = self.decoder_tsdf(p, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def decode(self, p, c, z, **kwargs):
        """Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        """

        qual = self.decoder(p, c, z, **kwargs)
        qual = torch.sigmoid(qual)

        rot = self.decoder(p, c, z, **kwargs)
        rot = nn.functional.normalize(rot, dim=2)
        force = self.decoder(p, c, z, **kwargs)
        return qual, rot, force

    def to(self, device):
        """Puts the model to the device.

        Args:
            device (device): pytorch device
        """
        model = super().to(device)
        model._device = device
        return model

    def grad_refine(self, x, pos, bound_value=0.0125, lr=1e-6, num_step=1):
        pos_tmp = pos.clone()
        l_bound = pos - bound_value
        u_bound = pos + bound_value
        pos_tmp.requires_grad = True
        optimizer = torch.optim.SGD([pos_tmp], lr=lr)
        self.eval()
        for p in self.parameters():
            p.requres_grad = False
        for _ in range(num_step):
            optimizer.zero_grad()
            qual_out, _, _ = self.forward(x, pos_tmp)
            # print(qual_out)
            loss = -qual_out.sum()
            loss.backward()
            optimizer.step()
            # print(qual_out.mean().item())
        with torch.no_grad():
            # print(pos, pos_tmp)
            pos_tmp = torch.maximum(torch.minimum(pos_tmp, u_bound), l_bound)
            qual_out, rot_out, force_out = self.forward(x, pos_tmp)
            # print(pos, pos_tmp, qual_out)
            # print(qual_out.mean().item())
        # import pdb; pdb.set_trace()
        # self.train()
        for p in self.parameters():
            p.requres_grad = True
        # import pdb; pdb.set_trace()
        return qual_out, pos_tmp, rot_out, force_out

class ConvolutionalOccupancyNetworkFeature(nn.Module):
    """Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    """

    def __init__(self, decoders, encoder=None, device=None, detach_tsdf=False):
        super().__init__()

        self.decoder = decoders[0].to(device)

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

        self._device = device

        self.detach_tsdf = detach_tsdf

        self.point_predict = True
        if self.point_predict:
            self.point_decoder = decoder.PointDecoder().to(device)

        self.eval = False
        self.save_feature = None

    def forward(self, inputs, p, p_tsdf=None, sample=True, **kwargs):
        """Performs a forward pass through the network.

        Args:
            p (tensor): sampled points, B*N*C
            inputs (tensor): conditioning input, B*N*3
            sample (bool): whether to sample for z
            p_tsdf (tensor): tsdf query points, B*N_P*3
        """
        #############
        if isinstance(p, dict):
            batch_size = p["p"].size(0)
        else:
            batch_size = p.size(0)
        c = self.encode_inputs(inputs)
        if self.eval and self.save_feature is None:
            self.save_feature = c.detach()

        if self.eval and self.save_feature is not None:
            feature = self.deoder(p, self.save_feature)
        else:
            feature = self.decoder(p, c)

        return feature

    def infer_geo(self, inputs, p_tsdf, **kwargs):
        c = self.encode_inputs(inputs)
        tsdf = self.decoder_tsdf(p_tsdf, c, **kwargs)
        return tsdf

    def encode_inputs(self, inputs):
        """Encodes the input.

        Args:
            input (tensor): the input
        """

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            # Return inputs?
            c = torch.empty(inputs.size(0), 0)

        return c

    def query_feature(self, p, c):
        return self.decoder_qual.query_feature(p, c)


    def decode_occ(self, p, c, **kwargs):
        """Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        """

        logits = self.decoder_tsdf(p, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def to(self, device):
        """Puts the model to the device.

        Args:
            device (device): pytorch device
        """
        model = super().to(device)
        model._device = device
        return model

    def grad_refine(self, x, pos, bound_value=0.0125, lr=1e-6, num_step=1):
        pos_tmp = pos.clone()
        l_bound = pos - bound_value
        u_bound = pos + bound_value
        pos_tmp.requires_grad = True
        optimizer = torch.optim.SGD([pos_tmp], lr=lr)
        self.eval()
        for p in self.parameters():
            p.requres_grad = False
        for _ in range(num_step):
            optimizer.zero_grad()
            qual_out, _, _ = self.forward(x, pos_tmp)
            # print(qual_out)
            loss = -qual_out.sum()
            loss.backward()
            optimizer.step()
            # print(qual_out.mean().item())
        with torch.no_grad():
            # print(pos, pos_tmp)
            pos_tmp = torch.maximum(torch.minimum(pos_tmp, u_bound), l_bound)
            qual_out, rot_out, force_out = self.forward(x, pos_tmp)
            # print(pos, pos_tmp, qual_out)
            # print(qual_out.mean().item())
        # import pdb; pdb.set_trace()
        # self.train()
        for p in self.parameters():
            p.requres_grad = True
        # import pdb; pdb.set_trace()
        return qual_out, pos_tmp, rot_out, force_out
