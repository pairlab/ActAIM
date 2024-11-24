import pdb

from affordance.cvae.cvae_encoder import *
from vgn.dataset_my import DatasetVoxel
from vgn.networks import get_network, load_network

import argparse
from pathlib import Path
from datetime import datetime

import torch
from torch.utils import tensorboard
import torch.nn.functional as F
import pdb

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def main(args):
    LOSS_KEYS = ["loss_all", "loss_qual", "loss_rot", "loss_force", "loss_elbo"]
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    if args.savedir == "":
        # create log directory
        time_stamp = datetime.now().strftime("%y-%m-%d-%H-%M")
        description = "augment={},net={},batch_size={},lr={:.0e},{}".format(
            time_stamp,
            args.augment,
            args.net,
            args.batch_size,
            args.lr,
            args.description,
        ).strip(",")
        logdir = args.logdir / description
    else:
        logdir = Path(args.savedir)

    # create data loaders
    train_loader, val_loader = create_train_val_loaders(args.batch_size, args.val_split, args.augment, kwargs)
    print("------finish loading dataset")

    # TODO merge class_num with encoder
    class_num = 5

    decoder = TestDecoder().to(device)  # p(I|z,o)
    obs_encoder = DepthEncoder().to(device)
    prior_m = CatEncoder().to(device)  # p(m|o)
    posterior_m = ModeEncode().to(device)  # p(m|I,o)
    posterior_z = LatentEncode().to(device)  # p(z|I, o)
    prior_z_m = ModeLatentEncode().to(device)  #

    # define optimizer and metrics
    params = (
        list(decoder.parameters())
        + list(obs_encoder.parameters())
        + list(prior_m.parameters())
        + list(posterior_m.parameters())
        + list(posterior_z.parameters())
        + list(prior_z_m.parameters())
    )
    # params = decoder.parameters()
    optimizer = torch.optim.Adam(params, lr=args.lr)

    # TODO cuda out of memory
    loss_values = []
    for epoch in range(100):
        optimizer.zero_grad()
        epoch_kl_z_loss = 0
        epoch_kl_m_loss = 0
        epoch_dir_loss = 0
        epoch_reconstruct_loss = 0
        epoch_loss_all = 0

        for i, data in enumerate(train_loader):
            x, y, pos = prepare_batch(data, device)
            label, rotations, force, depth = y
            rotations = rotations[:, 0, :]
            pos = pos[:, 0, :]
            inter_params = torch.cat((rotations, force, pos), dim=-1)
            obs_encode = obs_encoder(depth)

            obs_inter = torch.cat((obs_encode, inter_params), dim=-1)
            z_mean, z_log_var = posterior_z(obs_inter)
            mu, log_sigma = prior_z_m(obs_encode)

            z = reparameterize(z_mean, z_log_var)
            obs_z = torch.cat((obs_encode, z), dim=-1)
            recon_rotations, recon_point, recon_force = decoder(obs_z)
            reconstruct_loss = torch.mean(
                F.mse_loss(recon_rotations, rotations) + F.mse_loss(recon_point, pos) + F.mse_loss(recon_force, force)
            )

            m = prior_m(obs_encode)
            m_I = posterior_m(obs_inter)

            kl_loss_z_all = torch.tensor([0.0]).to(device)
            kl_loss_m_all = torch.tensor([0.0]).to(device)

            dir_loss_m_all = torch.tensor([0.0]).to(device)

            for j in range(0, class_num):
                # kl-divergence between q(z|I, o) and p(z|m,o)
                kl_loss_z = -0.5 * (1 + z_log_var - torch.square(z_mean - mu[:, j, :]) - torch.exp(z_log_var))
                kl_loss_z = torch.mean(kl_loss_z)
                kl_loss_z_all = kl_loss_z_all + kl_loss_z

                # kl-divergence between q(m|I, o) and p(m|o)
                m_I_mean = torch.mean(m_I[:, j], dim=-1)
                m_mean = torch.mean(m[:, j], dim=-1)
                kl_cat = m_I_mean * torch.log(m_I_mean) - m_I_mean * torch.log(m_mean)
                kl_loss_m_all = kl_loss_m_all + torch.mean(kl_cat)

                # kl-divergence in Dir dist
                dir_prior = -0.0001 * torch.log(m_mean)
                dir_loss_m_all = dir_loss_m_all + torch.mean(dir_prior)

            loss_all = kl_loss_z_all + kl_loss_m_all + dir_loss_m_all + reconstruct_loss
            loss_all.backward()
            optimizer.step()

            epoch_kl_z_loss += kl_loss_z_all.detach().item()
            epoch_kl_m_loss += kl_loss_m_all.detach().item()
            epoch_dir_loss += dir_loss_m_all.detach().item()
            epoch_reconstruct_loss += reconstruct_loss.detach().item()
            epoch_loss_all += loss_all.detach().item()

        optimizer.zero_grad()

        print("epoch: ", epoch)
        print("epoch_kl_z_loss: ", epoch_kl_z_loss)
        print("epoch_kl_m_loss: ", epoch_kl_m_loss)
        print("epoch_dir_loss: ", epoch_dir_loss)
        print("epoch_reconstruct_loss: ", epoch_reconstruct_loss)
        print("epoch_loss_all: ", epoch_loss_all)
        print("__________________________________________________________")

    for i, data in enumerate(train_loader):
        x, y, pos = prepare_batch(data, device)
        label, rotations, force, depth = y
        rotations = rotations[:, 0, :]
        pos = pos[:, 0, :]
        inter_params = torch.cat((rotations, force, pos), dim=-1)
        obs_encode = obs_encoder(depth)

        obs_inter = torch.cat((obs_encode, inter_params), dim=-1)
        z_mean, z_log_var = posterior_z(obs_inter)
        z_mean_test = torch.cat((z_mean_test, z_mean.detach()))

        z = reparameterize(z_mean, z_log_var)
        obs_z = torch.cat((obs_encode, z), dim=-1)
        recon_rotations, recon_point, recon_force = decoder(obs_z)
        if i == 20:
            break

    z_mean_test = z_mean_test.detach().cpu().numpy()
    X_embedded = TSNE(n_components=2).fit_transform(z_mean_test)
    pdb.set_trace()
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
    plt.title("Latent Space Representation Estimated by tGM-VAE")
    plt.show()

    pdb.set_trace()

    # torch.save(pack_model.state_dict(), "./pretrain_pack.pth")


def create_train_val_loaders(batch_size, val_split, augment, kwargs):
    # load the dataset
    dataset = DatasetVoxel(pair=False, augment=augment, cvae=True)
    # split into train and validation sets
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    # create loaders for both datasets
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
    return train_loader, val_loader


def prepare_batch(batch, device):
    pc, (label, rotations, force, depth), pos = batch
    pc = pc.float().to(device)
    depth = depth.float().to(device)
    label = label.float().to(device)
    rotations = rotations.float().to(device)
    force = force.float().to(device)
    pos.unsqueeze_(1)  # B, 1, 3
    pos = pos.float().to(device)
    return pc, (label, rotations, force, depth), pos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", default="giga_aff")
    parser.add_argument("--logdir", type=Path, default="data/runs")
    parser.add_argument("--description", type=str, default="cvae")
    parser.add_argument("--savedir", type=str, default="")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--visual-dim", type=int, default=32)

    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--silence", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--load-path", type=str, default="")
    args = parser.parse_args()
    print(args)
    main(args)
