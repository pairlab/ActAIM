import argparse
from pathlib import Path
from datetime import datetime

from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Average, Accuracy, Precision, Recall
import torch
from torch.utils import tensorboard
import torch.nn.functional as F
from affordance.vqvae.vq import Model
from affordance.vqvae.encoder import *

from vgn.dataset_my import DatasetVoxel
from vgn.networks import get_network, load_network
import matplotlib.pyplot as plt
import pdb
import time


def main(args):
    LOSS_KEYS = ["loss_all", "loss_qual", "loss_rot", "loss_force"]
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

    # if train pair data, then
    if args.pair or args.ipdf:
        LOSS_KEYS = ["loss_all"]

    # create data loaders
    train_loader, val_loader = create_train_val_loaders(
        args.batch_size, args.val_split, args.augment, args.pair or args.ipdf, kwargs
    )
    print("------finish loading dataset")

    if args.pair and not args.ipdf:
        args.net = args.net + "_pair"
    if args.ipdf:
        args.net = args.net + "_ipdf"

    # build the network or load
    if args.load_path == "":
        net = get_network(args.net).to(device)
    else:
        net = load_network(args.load_path, device, args.net)

    # Training params

    num_hiddens = 64
    num_residual_hiddens = 32
    num_residual_layers = 2
    embedding_dim = 32
    num_embeddings = 128
    commitment_cost = 0.25
    decay = 0.99
    learning_rate = 1e-5

    vqvae_model = Model(
        num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost, decay
    ).to(device)
    # define optimizer and metrics
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    vqvae_optimizer = torch.optim.Adam(vqvae_model.parameters(), lr=learning_rate)

    # TODO cuda out of memory
    loss_recon_values = []
    loss_kl_values = []
    loss_all_values = []
    epoch_values = []

    total_epoch = args.epochs
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
            net.train()
            optimizer.zero_grad()
            vqvae_optimizer.zero_grad()

            # forward
            x, y, pos, dof, _ = prepare_batch(data, device, False)
            label, rotations, force, depth = y

            inter_params = torch.cat((rotations.view(args.batch_size, 8), force, pos[:, 0, :]), dim=-1)

            vq_loss, data_recon, perplexity, quantized = vqvae_model(depth, inter_params)

            recon_rotations, recon_point, recon_force = data_recon

            y_pred = select(net(x, pos, quantized), False)
            _, rotation_pred, force_pred = y_pred

            rotation_recon_loss = torch.mean(rot_loss_fn(rotation_pred, rotations))
            point_recon_loss = torch.mean(F.mse_loss(recon_point, pos))
            force_recon_loss = torch.mean(F.mse_loss(force_pred, force))

            beta = betas[epoch]
            reconstruct_loss = rotation_recon_loss + point_recon_loss + force_recon_loss
            loss = beta * vq_loss + reconstruct_loss

            # backward
            loss.backward()
            optimizer.step()
            vqvae_optimizer.step()

            epoch_kl_z_loss += vq_loss.detach().item()
            epoch_reconstruct_loss += reconstruct_loss.detach().item()
            epoch_loss_all += loss.detach().item()

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
            save_training_plot(epoch_values, loss_recon_values, "giga_reconstruction")
            save_training_plot(epoch_values, loss_kl_values, "giga_kl")
            save_training_plot(epoch_values, loss_all_values, "giga_total")
            model_name = "multi_vqvae_giga_model.pt"
            torch.save(vqvae_model.state_dict(), str(model_name))
            model_name = "multi_vqvae_giga_decoder_model.pt"
            torch.save(net.state_dict(), str(model_name))

        print("__________________________________________________________")
        end_time = time.time()
        print("train one epoch cost time: ", end_time - start_time)


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


def rot_loss_fn(pred, target):
    loss0 = quat_loss_fn(pred, target[:, 0])
    loss1 = quat_loss_fn(pred, target[:, 1])
    # loss0 = F.mse_loss(pred, target[:, 0])
    # loss1 = F.mse_loss(pred, target[:, 1])

    return torch.min(loss0, loss1)


def quat_loss_fn(pred, target):
    return 1.0 - torch.abs(torch.sum(pred * target, dim=1))


def save_training_plot(x, y, name):
    plt.plot(x, y)
    plt.xlabel("Iteration")
    plt.ylabel(name + " Loss")
    training_plot_name = "vqvae_" + name + "_train_plot.png"
    plt.savefig(training_plot_name)
    plt.clf()


def create_train_val_loaders(batch_size, val_split, augment, pair, kwargs):
    # load the dataset
    # dataset = DatasetVoxel(pair, augment=augment)
    dataset = DatasetVoxel(pair=False, augment=augment, cvae=True)
    # split into train and validation sets
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    # create loaders for both datasets
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
    return train_loader, val_loader


def prepare_batch(batch, device, pair):
    if pair:
        (pc, rotations, force), label, pos, dof, object_id = batch
    else:
        pc, (label, rotations, force, depth), pos, dof, object_id = batch
    pc = pc.float().to(device)
    label = label.float().to(device)
    rotations = rotations.float().to(device)
    force = force.float().to(device)
    pos.unsqueeze_(1)  # B, 1, 3
    pos = pos.float().to(device)
    if pair:
        return (pc, rotations, force), label, pos, dof, object_id
    else:
        depth = depth.float().to(device)
        return pc, (label, rotations, force, depth), pos, dof, object_id


def select(out, is_pair):
    if is_pair:
        return out.squeeze(-1)
    else:
        qual_out, rot_out, force_out = out
        rot_out = rot_out.squeeze(1)
        return qual_out.squeeze(-1), rot_out, force_out.squeeze(-2)


def loss_pair(y_pred, y):
    loss_qual = _qual_loss_fn(y_pred, y)
    loss_dict = {"loss_all": loss_qual.mean()}
    return loss_qual.mean(), loss_dict


def loss_fn(y_pred, y):
    label_pred, rotation_pred, force_pred = y_pred
    label, rotations, force = y
    loss_qual = _qual_loss_fn(label_pred, label)
    loss_rot = _rot_loss_fn(rotation_pred, rotations)
    loss_force = _force_loss_fn(force_pred, force)
    loss = loss_qual + label * (loss_rot + loss_force)
    loss_dict = {
        "loss_qual": loss_qual.mean(),
        "loss_rot": loss_rot.mean(),
        "loss_force": loss_force.mean(),
        "loss_all": loss.mean(),
    }
    return loss.mean(), loss_dict


def _qual_loss_fn(pred, target):
    # TODO re-weight the BCE loss
    # weight = torch.ones_like(target) + 2 * target.detach().clone()
    return F.binary_cross_entropy(pred, target, reduction="none")


def _rot_loss_fn(pred, target):
    loss0 = _quat_loss_fn(pred, target[:, 0])
    loss1 = _quat_loss_fn(pred, target[:, 1])
    return torch.min(loss0, loss1)


def _quat_loss_fn(pred, target):
    return 1.0 - torch.abs(torch.sum(pred * target, dim=1))


def _force_loss_fn(pred, target):
    pred_norm = F.normalize(pred, dim=1)
    target_norm = F.normalize(target, dim=1)
    return 1.0 - torch.abs(torch.sum(pred_norm * target_norm, dim=1))
    # return F.mse_loss(4 * pred, 4 * target, reduction="none")


def create_trainer(net, vqvae_model, vqvae_optimizer, optimizer, loss_fn, metrics, device):
    def _update(_, batch):
        net.train()
        optimizer.zero_grad()
        vqvae_optimizer.zero_grad()

        # forward
        x, y, pos, dof, _ = prepare_batch(batch, device, False)
        label, rotations, force, depth = y

        inter_params = torch.cat((rotations.view(args.batch_size, 8), force, pos[:, 0, :]), dim=-1)

        kl_loss_z, (z_mean, z_log_var), _ = vqvae_model(inter_params, depth)
        z = reparameterize(z_mean, z_log_var)

        y_pred = select(net(x, pos, z), False)
        pdb.set_trace()
        loss, loss_dict = loss_fn(y_pred, kl_loss_z, y)

        # backward
        loss.backward()
        optimizer.step()
        vqvae_optimizer.step()

        return x, y_pred, y, loss_dict

    trainer = Engine(_update)

    for name, metric in metrics.items():
        metric.attach(trainer, name)

    return trainer


def create_evaluator(net, loss_fn, metrics, device):
    def _inference(_, batch):
        net.eval()
        with torch.no_grad():
            x, y, pos = prepare_batch(batch, device, False)
            y_pred = select(net(x, pos), False)
            loss, loss_dict = loss_fn(y_pred, y)
        return x, y_pred, y, loss_dict

    evaluator = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator


def create_trainer_pair(net, optimizer, loss_fn, metrics, device):
    def _update(_, batch):
        net.train()
        optimizer.zero_grad()

        # forward
        x, y, pos = prepare_batch(batch, device, True)
        y_pred = select(net(x, pos), True)
        loss, loss_dict = loss_fn(y_pred, y)

        # backward
        loss.backward()
        optimizer.step()

        return x, y_pred, y, loss_dict

    trainer = Engine(_update)

    for name, metric in metrics.items():
        metric.attach(trainer, name)

    return trainer


def create_evaluator_pair(net, loss_fn, metrics, device):
    def _inference(_, batch):
        net.eval()
        with torch.no_grad():
            x, y, pos = prepare_batch(batch, device, True)
            y_pred = select(net(x, pos), True)
            loss, loss_dict = loss_fn(y_pred, y)
        return x, y_pred, y, loss_dict

    evaluator = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator


def create_summary_writers(net, device, log_dir):
    train_path = log_dir / "train"
    val_path = log_dir / "validation"

    train_writer = tensorboard.SummaryWriter(train_path, flush_secs=60)
    val_writer = tensorboard.SummaryWriter(val_path, flush_secs=60)

    return train_writer, val_writer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", default="giga_aff")
    parser.add_argument("--logdir", type=Path, default="data/runs")
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--savedir", type=str, default="")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--silence", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--load-path", type=str, default="")
    parser.add_argument("--pair", dest="pair", default=False)
    parser.add_argument("--ipdf", dest="ipdf", default=False)
    args = parser.parse_args()
    print(args)
    main(args)
