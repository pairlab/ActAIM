from affordance.cvae.cvae_encoder import *
from affordance.cvae.model import CVAEModel
from vgn.dataset_my import DatasetVoxel
from vgn.networks import get_network, load_network

import argparse
from pathlib import Path
from datetime import datetime

import torch
from torch.utils import tensorboard
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time

import pdb


def save_training_plot(x, y, name):
    plt.plot(x, y)
    plt.xlabel("Iteration")
    plt.ylabel(name + " Loss")
    training_plot_name = "CVAE_" + name + "_train_plot.png"
    plt.savefig(training_plot_name)
    plt.clf()


def rot_loss_fn(pred, target):
    loss0 = quat_loss_fn(pred, target[:, 0])
    loss1 = quat_loss_fn(pred, target[:, 1])
    # loss0 = F.mse_loss(pred, target[:, 0])
    # loss1 = F.mse_loss(pred, target[:, 1])

    return torch.min(loss0, loss1)


def quat_loss_fn(pred, target):
    return 1.0 - torch.abs(torch.sum(pred * target, dim=1))


def force_loss_fn(pred, target):
    pred_norm = F.normalize(pred, dim=1)
    target_norm = F.normalize(target, dim=1)
    return 1.0 - torch.abs(torch.sum(pred_norm * target_norm, dim=1))


def frange_cycle_linear(n_iter, start=0.0, stop=1.0, n_cycle=4, ratio=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter / n_cycle
    step = (stop - start) / (period * ratio)  # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i + c * period) < n_iter):
            L[int(i + c * period)] = v
            v += step
            i += 1
    return L


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

    model = CVAEModel().to(device)
    # model.load_state_dict(torch.load('cvae_gaussian_model.pt'))

    # TODO merge class_num with encoder
    class_num = 8
    # define optimizer and metrics
    # params = list(decoder.parameters()) + list(obs_encoder.parameters()) + list(prior_m.parameters()) + list(posterior_m.parameters()) + list(posterior_z.parameters()) + list(prior_z_m.parameters())
    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=1e-5)

    z_mean_test = torch.Tensor().to(device)
    z_mean_dof = torch.Tensor()

    # TODO cuda out of memory
    loss_recon_values = []
    loss_kl_values = []
    loss_all_values = []
    epoch_values = []

    total_epoch = 200
    zero_betas = np.zeros(total_epoch // 3)
    betas = frange_cycle_linear(total_epoch - total_epoch // 3)
    betas = np.concatenate((zero_betas, betas))

    for epoch in range(total_epoch):
        optimizer.zero_grad()

        epoch_kl_z_loss = 0
        epoch_reconstruct_loss = 0
        epoch_loss_all = 0
        epoch_reconstruct_rotation = 0
        epoch_reconstruct_point = 0
        epoch_reconstruct_force = 0

        start_time = time.time()

        for i, data in enumerate(train_loader):
            x, y, pos, dof, object_id = prepare_batch(data, device)
            label, rotations, force, depth = y
            # check two directions
            # rotations = rotations[:, 0, :]
            pos = pos[:, 0, :]
            inter_params = torch.cat((rotations.view(args.batch_size, 8), force, pos), dim=-1)

            kl_loss_z, (z_mean, z_log_var), (recon_rotations, recon_point, recon_force), z = model(inter_params, depth)

            rotation_recon_loss = torch.mean(rot_loss_fn(recon_rotations, rotations))
            point_recon_loss = torch.mean(F.mse_loss(recon_point, pos)) * 10
            force_recon_loss = torch.mean(F.mse_loss(recon_force, force))
            # force_recon_loss = torch.mean(force_loss_fn(recon_force, force))

            reconstruct_loss = rotation_recon_loss + point_recon_loss + force_recon_loss

            kl_loss_z_all = torch.tensor([0.0]).to(device)

            kl_loss_z = torch.mean(kl_loss_z)
            kl_loss_z_all = kl_loss_z_all * 0.1 + kl_loss_z

            beta = betas[epoch]

            loss_all = beta * kl_loss_z_all * 0.1 + reconstruct_loss
            loss_all.backward()

            optimizer.step()

            epoch_kl_z_loss += kl_loss_z_all.detach().item()
            epoch_reconstruct_loss += reconstruct_loss.detach().item()
            epoch_loss_all += loss_all.detach().item()

            epoch_reconstruct_rotation += rotation_recon_loss.detach().item()
            epoch_reconstruct_force += force_recon_loss.detach().item()
            epoch_reconstruct_point += point_recon_loss.detach().item()

        optimizer.zero_grad()

        print("epoch: ", epoch)
        print("epoch_kl_z_loss: ", epoch_kl_z_loss)
        print("epoch_reconstruct_loss: ", epoch_reconstruct_loss)
        print("epoch_loss_all: ", epoch_loss_all)

        print("epoch_reconstruction_rotations: ", epoch_reconstruct_rotation)
        print("epoch_reconstruction_point: ", epoch_reconstruct_point)
        print("epoch_reconstruction_force: ", epoch_reconstruct_force)

        loss_recon_values.append(epoch_reconstruct_loss)
        loss_kl_values.append(epoch_kl_z_loss)
        loss_all_values.append(epoch_loss_all)
        epoch_values.append(epoch)

        if epoch % 10 == 0:
            save_training_plot(epoch_values, loss_recon_values, "reconstruction")
            save_training_plot(epoch_values, loss_kl_values, "kl")
            save_training_plot(epoch_values, loss_all_values, "total")
            model_name = "cvae_gaussian_model.pt"
            torch.save(model.state_dict(), str(model_name))

        print("__________________________________________________________")
        end_time = time.time()
        print("train one epoch cost time: ", end_time - start_time)

    # torch.save(pack_model.state_dict(), "./pretrain_pack.pth")

    for i, data in enumerate(train_loader):
        x, y, pos, dof, _ = prepare_batch(data, device)
        label, rotations, force, depth = y
        # rotations = rotations[:, 0, :]
        pos = pos[:, 0, :]
        inter_params = torch.cat((rotations.view(args.batch_size, 8), force, pos), dim=-1)

        kl_loss_z, (z_mean, z_log_var), (recon_rotations, recon_point, recon_force), _ = model(inter_params, depth)

        z_mean_test = torch.cat((z_mean_test, z_mean.detach()))
        z_mean_dof = torch.cat((z_mean_dof, dof.detach()))

        if i == 200:
            break

    print("reconstruct rotations: ", recon_rotations)
    print("reconstruct force: ", recon_force)
    print("reconstruct pc: ", recon_point)

    z_mean_test = z_mean_test.detach().cpu().numpy()
    z_mean_dof = z_mean_dof.detach().cpu().numpy()
    X_embedded = TSNE(n_components=2).fit_transform(z_mean_test)

    z_mean_dof[z_mean_dof == 10] = 3
    z_mean_dof[z_mean_dof == 20] = 4
    z_mean_dof[z_mean_dof == 11] = 5

    pdb.set_trace()
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=z_mean_dof)
    plt.title("Latent Space Representation Estimated by cVAE")
    plt.show()

    pdb.set_trace()


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
    pc, (label, rotations, force, depth), pos, dof, object_id = batch
    pc = pc.float().to(device)
    depth = depth.float().to(device)
    label = label.float().to(device)
    rotations = rotations.float().to(device)
    force = force.float().to(device)
    pos.unsqueeze_(1)  # B, 1, 3
    pos = pos.float().to(device)
    return pc, (label, rotations, force, depth), pos, dof, object_id


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

    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--silence", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--load-path", type=str, default="")
    args = parser.parse_args()
    print(args)
    main(args)
