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
from affordance.cvae.model import CVAEModel
from affordance.cvae.cvae_encoder import *

from vgn.dataset_my import DatasetVoxel
from affordance.unet.model import UNet
from affordance.utils.train2d import *
import matplotlib.pyplot as plt
import pdb
import time
from tqdm import tqdm



def main(args):
    LOSS_KEYS = ["loss_all", "loss_qual", "loss_rot", "loss_force"]
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    if args.vanilla:
        args.description += "_vanilla"

    if args.savedir == "":
        # create log directory
        time_stamp = datetime.now().strftime("%y-%m-%d-%H-%M")
        description = "net={},time={},batch_size={},lr={:.0e},{}".format(
            args.net,
            time_stamp,
            args.batch_size,
            args.lr,
            args.description,
        ).strip(",")
        logdir = args.logdir / description
    else:
        logdir = Path(args.savedir)

    logdir.mkdir(parents=True, exist_ok=True)
    logdir = str(logdir)

    # if train pair data, then
    if args.pair or args.ipdf:
        LOSS_KEYS = ["loss_all"]

    # create data loaders
    train_loader, val_loader = create_train_val_loaders(
        args.batch_size, args.val_split, args.augment, args.pair or args.ipdf, args.dataset, kwargs
    )
    print("------finish loading dataset")

    if args.pair and not args.ipdf:
        args.net = args.net + "_pair"
    if args.ipdf:
        args.net = args.net + "_ipdf"


    model = UNet(1, depth=5, merge_mode="concat", in_channels=4, start_filts=32).to(device)
    # path = "./data/unet_model.pt"
    # model.load_state_dict(torch.load(path))

    # define optimizer and metrics
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # TODO cuda out of memory
    loss_recon_values = []
    loss_kl_values = []
    loss_all_values = []
    epoch_values = []

    total_epoch = args.epochs
    zero_betas = np.zeros(total_epoch // 4)
    betas = frange_cycle_linear(total_epoch - total_epoch // 4)
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

        for i, data in enumerate(tqdm(train_loader)):
            model.train()
            optimizer.zero_grad()

            # forward
            x, y, pos, dof, object_id, camera_id = prepare_batch(data, device, False)
            label, rotations, force, depth = y
            point_pred = model(depth)

            # labels = torch.Tensor().to(device)
            pixels_feature = torch.Tensor().to(device)
            for index in range(pos.shape[0]):
                pixel_xy = G2P(pos[index], camera_id[index]).long()
                feature = point_pred[index, pixel_xy[0, 0], pixel_xy[0, 1]]
                pixels_feature = torch.cat((pixels_feature, feature.unsqueeze(0)), dim=0)


                # gt = torch.zeros_like(depth[0, 0, ...]).to(device)
                # gt[pixel_xy[0, 0], pixel_xy[0, 1]] = 1
                # gt = gt[None, ...]
                # labels = torch.cat((labels, gt), dim=0)


            # inter_params = torch.cat((rotations.view(args.batch_size, 8), force, pos[:, 0, :]), dim=-1)

            # evaluation and visualization
            # for j in range(len(object_id)):
            #     if object_id[i] == '19179_11':
            #         print("here")
            #         from torchvision.utils import save_image
            #         img1 = point_pred[i] * 255
            #         save_image(img1, 'img1.png')
            #         exit()

            loss = criterion(pixels_feature, label)

            # backward
            loss.backward()
            optimizer.step()
            epoch_loss_all += loss.detach().item()

        optimizer.zero_grad()

        print("epoch: ", epoch)
        print("loss all: ", epoch_loss_all)

        loss_all_values.append(epoch_loss_all)
        epoch_values.append(epoch)

        if epoch % 10 == 0:
            save_training_plot(epoch_values, loss_all_values, "unet_total", logdir)

            model_name = logdir + "/unet_model.pt"
            torch.save(model.state_dict(), str(model_name))

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


def save_training_plot(x, y, name, dir):
    plt.plot(x, y)
    plt.xlabel("Iteration")
    plt.ylabel(name + " Loss")
    plt.grid(True)
    training_plot_name = str(dir) + "/CVAE_" + name + "_train_plot.png"
    plt.savefig(training_plot_name)
    plt.clf()


def create_train_val_loaders(batch_size, val_split, augment, pair, dataset, kwargs):
    # load the dataset
    # dataset = DatasetVoxel(pair, augment=augment)
    dataset = DatasetVoxel(pair=False, augment=augment, cvae=True, root=dataset)
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
        (pc, rotations, force), label, pos, dof, object_id, camera_id = batch
    else:
        pc, ((label, rotations, force), depth), pos, dof, object_id, camera_id = batch
    pc = pc.float().to(device)
    label = label.float().to(device)
    rotations = rotations.float().to(device)
    force = force.float().to(device)
    pos.unsqueeze_(1)  # B, 1, 3
    pos = pos.float().to(device)
    if pair:
        return (pc, rotations, force), label, pos, dof, object_id, camera_id
    else:
        depth = depth.float().to(device)
        return pc, (label, rotations, force, depth), pos, dof, object_id, camera_id


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





def create_summary_writers(net, device, log_dir):
    train_path = log_dir / "train"
    val_path = log_dir / "validation"

    train_writer = tensorboard.SummaryWriter(train_path, flush_secs=60)
    val_writer = tensorboard.SummaryWriter(val_path, flush_secs=60)

    return train_writer, val_writer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", default="unet")
    # parser.add_argument("--net", default="giga_image")

    parser.add_argument("--logdir", type=Path, default="data/runs")
    parser.add_argument("--description", type=str, default="single_gpu")
    parser.add_argument("--savedir", type=str, default="")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--silence", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--load-path", type=str, default="")
    parser.add_argument("--pair", dest="pair", default=False)
    parser.add_argument("--ipdf", dest="ipdf", default=False)
    parser.add_argument("--vanilla", action="store_true", default=False)
    parser.add_argument("--dataset", default="dataset_19179_new")
    args = parser.parse_args()
    print(args)
    main(args)
