import argparse
from pathlib import Path
from datetime import datetime

from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Average, Accuracy, Precision, Recall
import torch
from torch.utils import tensorboard
import torch.nn as nn
import torch.nn.functional as F
from affordance.utils.train2d import G2P, P2G, norm2world
from affordance.cvae.model import CVAEModel, TaskCVAEModel

import pdb


from vgn.dataset_my import DatasetVoxel
from vgn.networks import get_network, load_network

cam_width, cam_height = 320, 320


class Wrapper(nn.Module):
    def __init__(self, args):
        super().__init__()

        # build the network or load
        if args.load_path == "":
            self.net = get_network(args.net)
        else:
            self.net = load_network(args.load_path, args.net)

        self.task_encode = args.task
        if self.task_encode:
            self.cvae_model = TaskCVAEModel()
        else:
            self.cvae_model = CVAEModel()

        self.which_net = args.net
        self.eval = args.eval

    def forward(self, x, y, pos, obs, camera_id):
        label, rotations, force, depth = y

        inter_params = torch.cat((rotations.view(args.batch_size, 8), force, pos[:, 0, :]), dim=-1)

        if self.task_encode:
            kl_loss_z, (z_mean, z_log_var), (recon_rotations, recon_point, recon_force), z = self.cvae_model(
                inter_params, depth, obs, self.eval
            )
        else:
            kl_loss_z, (z_mean, z_log_var), (recon_rotations, recon_point, recon_force), z = self.cvae_model(
                inter_params, depth, self.eval
            )

        if self.which_net == "giga_image":
            world_pos = norm2world(pos[:, 0, :])
            pixel = self.G2P_batch(world_pos, camera_id)
            pixel[:, 0] /= cam_width
            pixel[:, 1] /= cam_height
            # TODO should normalize it to [-0.5,0.5]
            pixel -= 0.5

            y_pred = select(self.net(depth, pixel.unsqueeze(1), z))
        else:
            y_pred = select(self.net(x, pos, z))

        point_pred, rotation_pred, force_pred = y_pred

        return kl_loss_z, point_pred, rotation_pred, force_pred

    def G2P_batch(self, world_pos, camera_id):
        batch_size = world_pos.shape[0]
        pixel_batch = torch.Tensor().cuda()
        for i in range(batch_size):
            pixel = G2P(world_pos[i].unsqueeze(0), int(camera_id[i]))
            pixel_batch = torch.cat((pixel_batch, pixel), 0)
        return pixel_batch

    def P2G_batch(self, pixel, negative_depth, camera_id):
        batch_size = pixel.shape[0]
        point_batch = torch.Tensor().cuda()
        for i in range(batch_size):
            pixel = P2G(pixel[i].unsqueeze(0), negative_depth, int(camera_id[i]))
            point_batch = torch.cat((point_batch, pixel), 0)
        return point_batch

    def evaluation(self):
        self.eval = True

    def is_train(self):
        self.eval = False

def main(args):
    LOSS_KEYS = ["loss_all", "loss_point", "loss_rot", "loss_force", "loss_kl"]
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    if args.savedir == "":
        # create log directory
        time_stamp = datetime.now().strftime("%y-%m-%d-%H-%M")
        description = "augment={},net={},batch_size={},lr={:.0e},dataset={},{}".format(
            time_stamp,
            args.augment,
            args.net,
            args.batch_size,
            args.lr,
            args.dataset,
            args.description,
        ).strip(",")
        logdir = args.logdir / description
    else:
        logdir = Path(args.savedir)

    # create data loaders
    train_loader, val_loader = create_train_val_loaders(
        args.batch_size, args.val_split, args.augment, args.dataset, kwargs
    )
    print("------finish loading dataset")

    # build the network or load
    model = Wrapper(args).cuda()


    # define optimizer and metrics
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    metrics = {}
    for k in LOSS_KEYS:
        metrics[k] = Average(lambda out, sk=k: out[3][sk])

    # create ignite engines for training and validation
    trainer = create_trainer(model, optimizer, loss_fn, metrics, device)
    evaluator = create_evaluator(model, loss_fn, metrics, device)


    '''
    loss_values = []
    for epoch in range(1000):
        net.train()
        optimizer.zero_grad()
        for i, data in enumerate(train_loader):
            x, y, pos, dof, object_id, camera_id = prepare_batch(data, device, True)
            y_pred = select(net(x, pos), True)
            loss, loss_dict = loss_pair(y_pred, y)
            import pdb; pdb.set_trace()
            # backward
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()
        # print("epoch: ", epoch)
        # print("pack_loss: ", pack_loss)
        # loss_values.append(pack_loss)
    # torch.save(pack_model.state_dict(), "./pretrain_pack.pth")
    '''

    # log training progress to the terminal and tensorboard
    ProgressBar(persist=True, ascii=True, dynamic_ncols=True, disable=args.silence).attach(trainer)

    train_writer, val_writer = create_summary_writers(model, device, logdir)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_train_results(engine):
        epoch, metrics = trainer.state.epoch, trainer.state.metrics
        for k, v in metrics.items():
            train_writer.add_scalar(k, v, epoch)

        msg = "Train"
        for k, v in metrics.items():
            msg += f" {k}: {v:.4f}"
        print(msg)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        epoch, metrics = trainer.state.epoch, evaluator.state.metrics
        for k, v in metrics.items():
            val_writer.add_scalar(k, v, epoch)

        msg = 'Val'
        for k, v in metrics.items():
            msg += f' {k}: {v:.4f}'
        print(msg)

    def default_score_fn(engine):
        score = -engine.state.metrics["loss_all"]
        return score

    # checkpoint model
    checkpoint_handler = ModelCheckpoint(
        logdir,
        "vgn",
        n_saved=1,
        require_empty=True,
    )
    best_checkpoint_handler = ModelCheckpoint(
        logdir,
        "best_aff",
        n_saved=1,
        score_name="val_acc",
        score_function=default_score_fn,
        require_empty=True,
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), checkpoint_handler, {args.net: model})
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED, best_checkpoint_handler, {args.net: model}
    )

    # run the training loop
    trainer.run(train_loader, max_epochs=args.epochs)


def create_train_val_loaders(batch_size, val_split, augment, dataset, kwargs):
    # load the dataset
    dataset = DatasetVoxel(False, augment=augment, cvae=True, root=dataset)
    # split into train and validation sets
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    # create loaders for both datasets
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
    return train_loader, val_loader


def prepare_batch(batch, device):
    pc, ((label, rotations, force), depth), pos, dof, obs, object_id, camera_id = batch
    pc = pc.float().to(device)
    label = label.float().to(device)
    rotations = rotations.float().to(device)
    force = force.float().to(device)
    pos.unsqueeze_(1)  # B, 1, 3
    pos = pos.float().to(device)
    depth = depth.float().to(device)
    obs = obs.float().to(device)

    return pc, (label, rotations, force, depth), pos, dof, obs, object_id, camera_id


def select(out):
    qual_out, rot_out, force_out = out
    rot_out = rot_out.squeeze(1)
    return qual_out.squeeze(-1), rot_out, force_out.squeeze(-2)


def loss_pair(y_pred, y):
    loss_qual = _qual_loss_fn(y_pred, y)
    loss_dict = {"loss_all": loss_qual.mean()}
    return loss_qual.mean(), loss_dict


def loss_fn(y_pred, y, pos):
    pos = pos.squeeze(1)
    kl_loss_z, point_pred, rotation_pred, force_pred = y_pred
    label, rotations, force, depth = y

    loss_point = _point_loss_fn(point_pred, pos) * label * 10
    loss_rot = _rot_loss_fn(rotation_pred, rotations) * label
    loss_force = _force_loss_fn(force_pred, force) * label * 10
    loss = (loss_rot + loss_force + loss_point) + kl_loss_z
    loss_dict = {
        "loss_all": loss.mean(),
        "loss_point": loss_point.mean(),
        "loss_rot": loss_rot.mean(),
        "loss_force": loss_force.mean(),
        "loss_kl": kl_loss_z.mean(),
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

def _point_loss_fn(pred, target):
    return F.mse_loss(pred, target)

def _force_loss_fn(pred, target):
    pred_norm = F.normalize(pred, dim=1)
    target_norm = F.normalize(target, dim=1)
    return 1.0 - torch.abs(torch.sum(pred_norm * target_norm, dim=1))
    # return F.mse_loss(4 * pred, 4 * target, reduction="none")


def create_trainer(net, optimizer, loss_fn, metrics, device):
    def _update(_, batch):
        net.is_train()
        optimizer.zero_grad()

        # forward
        x, y, pos, dof, obs, object_id, camera_id = prepare_batch(batch, device)
        y_pred = net(x, y, pos, obs, camera_id)
        loss, loss_dict = loss_fn(y_pred, y, pos)

        # backward
        loss.backward()
        optimizer.step()

        return x, y_pred, y, loss_dict

    trainer = Engine(_update)

    for name, metric in metrics.items():
        metric.attach(trainer, name)

    return trainer


def create_evaluator(net, loss_fn, metrics, device):
    def _inference(_, batch):
        net.evaluation()
        with torch.no_grad():
            x, y, pos, dof, obs, object_id, camera_id = prepare_batch(batch, device)
            y_pred = net(x, y, pos, obs, camera_id)
            loss, loss_dict = loss_fn(y_pred, y, pos)
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
    parser.add_argument("--description", type=str, default="heatmap_pair")
    parser.add_argument("--savedir", type=str, default="")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--silence", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--load-path", type=str, default="")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--task", action="store_true", default=False)
    parser.add_argument("--eval", action="store_true", default=False)

    args = parser.parse_args()
    print(args)
    main(args)
