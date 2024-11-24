import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Average, Accuracy, Precision, Recall
import torch
from torch.utils import tensorboard
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

from affordance.utils.train2d import G2P, P2G, norm2world
from affordance.cvae.model import CVAEModel, TaskCVAEModel2, CPCModel, CURLModel, TaskGMCVAEModel
from affordance.models.decoder import PointCritic, ScoreCritic, ActionPolicy, StochasticActionPolicy, VAEActionPolicy
from affordance.cvae.cvae_encoder import DepthEncoder
from affordance.metric.encoder import AutoEncoder
from affordance.models.pointnet import PointNet2SemSegSSG

import pdb

from vgn.dataset_my import DatasetVoxel
from vgn.networks import get_network, load_network

cam_width, cam_height = 320, 320
total_epoch = 80

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

zero_betas_len = int(total_epoch // 2)
ones_beta_len = int(total_epoch // 5)

zero_betas = np.zeros(zero_betas_len)
ones_betas = np.ones(ones_beta_len)
betas = frange_cycle_linear(int(total_epoch) - zero_betas_len - ones_beta_len + 1)
# betas = np.concatenate((zero_betas, betas, ones_betas))
betas = np.concatenate((zero_betas, betas, ones_betas))
betas = np.ones_like(betas)

class Wrapper(nn.Module):
    def __init__(self, args):
        super().__init__()

        # build the network or load
        if args.pointnet:
            self.net = PointNet2SemSegSSG({'feat_dim': 96})
        else:
            self.net = get_network(args.net)
        self.use_metric = args.use_metric
        self.is_stochastic = args.stochastic
        self.discrete = args.discrete
        self.category = int(args.category)
        self.finetune = args.finetune

        self.gmvae = args.gmvae
        if self.gmvae:
            self.encoder_model = TaskGMCVAEModel(self.category)
        else:
            self.encoder_model = TaskCVAEModel2()

        # self.r_decode = RotPolicy()
        # self.f_decode = ForcePolicy()

        if self.is_stochastic:
            self.policy = StochasticActionPolicy()
        else:
            self.policy = ActionPolicy()

        if self.discrete:
            self.point_critic = PointCritic(output_dim=1, latent_dim=self.category)
            # self.score_decode = ScoreCritic(output_dim=1, latent_dim=self.category)
        else:
            self.point_critic = PointCritic(output_dim=1, latent_dim=64)
        self.score_decode = ScoreCritic(output_dim=1, latent_dim=64)
        self.obs_encoder = DepthEncoder(c_dim=128, in_channel=1)

        self.which_net = args.net
        self.eval = args.eval

    def forward(self, x, y, pos, obs, camera_id, metric_model=None):
        label, rotations, force, depth = y
        batch_size = x.shape[0]

        depth = depth[:, 0, ...] # depth image only
        kl_loss, out_net, task, recon_loss = self.encoder_model.forward(depth, obs, metric_model, self.finetune)

        point_feature = select(self.net(x, pos))


        rot_pred, force_pred = self.policy.forward(task, depth, point_feature, self.obs_encoder)

        if self.discrete:
            category = out_net['categorical']
            # critic_pred = self.score_decode.forward(category, point_feature, rotations, force)
            point_score = self.point_critic.forward(category, point_feature)
            # point_score_label = self.compute_point_score_label(category.detach(), point_feature.detach())
        else:
            point_score = self.point_critic.forward(task, point_feature)

        critic_pred = self.score_decode.forward(task, point_feature, rotations, force)
        point_score_label = self.compute_point_score_label(task.detach(), point_feature.detach())

        return kl_loss, recon_loss, point_score, rot_pred, force_pred, critic_pred, point_score_label

    def compute_point_score_label(self, skill, point_feature):
        point_score_label = torch.zeros(skill.shape[0]).to(skill.device)
        avg_N = 200
        for i in range(point_feature.shape[0]):
            current_point_feature = point_feature[i]
            current_skill = skill[i]

            repeat_point_feature = current_point_feature.unsqueeze(0).repeat(avg_N, 1)
            repeat_skill = current_skill.unsqueeze(0).repeat(avg_N, 1)

            sample_rotation = Rotation.random(avg_N).as_quat()

            rotations = np.empty((avg_N, 2, 4), dtype=np.single)
            rotation_direction = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])
            rotations[:, 0, :] = sample_rotation
            rotations[:, 1, :] = (Rotation.from_quat(sample_rotation) * rotation_direction).as_quat()

            rotations = torch.tensor(rotations, device=skill.device).float()
            forces = torch.from_numpy(np.random.rand(avg_N, 3)).to(device=skill.device).float()

            current_point_score = self.score_decode.forward(repeat_skill, repeat_point_feature, rotations, forces)
            point_score_label[i] = current_point_score.max().item()

        return point_score_label.detach()


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
    LOSS_KEYS = ["loss_all", "loss_critic", "loss_rotation", "loss_force", "loss_score", "loss_kl", "loss_transition", "loss_eval"]
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    if args.pointnet == True:
        args.net = "pointnet++"

    if args.savedir == "":
        # create log directory
        time_stamp = datetime.now().strftime("%y-%m-%d-%H-%M")
        description = "time={},augment={},net={},batch_size={},lr={:.0e},dataset={},gmvae={},finetune={},{}".format(
            time_stamp,
            args.augment,
            args.net,
            args.batch_size,
            args.lr,
            args.dataset,
            args.gmvae,
            args.finetune,
            args.description,
        ).strip(",")
        logdir = args.logdir / description
    else:
        logdir = Path(args.savedir)

    # create data loaders
    train_loader, val_loader = create_train_val_loaders(
        args.batch_size, args.val_split, args.augment, args.dataset, args.pointnet, kwargs
    )
    print("------finish loading dataset")

    # build the network or load
    model = Wrapper(args).cuda()

    # TODO set finetune code
    if args.finetune:
        path = "./data/runs/time=22-08-29-01-44,augment=False,net=giga_feature,batch_size=32,lr=5e-05,dataset=dataset,gmvae=False,finetune=False,task-cvae/best_aff_giga_feature_val_acc=-2.8252.pt"
        path = "./model/best_aff_giga_feature_val_acc=-4.0746.pt"

        model.load_state_dict(torch.load(path))

    # resume training
    # path = "./data/runs/time=22-08-31-19-07,augment=False,net=giga_feature,batch_size=32,lr=1e-05,dataset=dataset,gmvae=False,finetune=False,task-cvae/best_aff_giga_feature_val_acc=-4.0746.pt"
    # model.load_state_dict(torch.load(path))

    # load metric model to get the latent
    metric_model = AutoEncoder().to(device)
    path = "./autoencoder/metric.pt"
    metric_model.load_state_dict(torch.load(path))
    metric_model.eval()

    # define optimizer and metrics
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    metrics = {}
    for k in LOSS_KEYS:
        metrics[k] = Average(lambda out, sk=k: out[3][sk])

    # create ignite engines for training and validation
    trainer = create_trainer(model, metric_model, optimizer, loss_fn, metrics, device)
    evaluator = create_evaluator(model, metric_model, loss_fn, metrics, device)

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
        score = -engine.state.metrics["loss_eval"]
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

def create_train_val_loaders(batch_size, val_split, augment, dataset, is_pc, kwargs):
    # load the dataset
    dataset = DatasetVoxel(False, augment=augment, cvae=True, root=dataset, pc=is_pc)
    # split into train and validation sets
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    # create loaders for both datasets
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
    return train_loader, val_loader

def prepare_batch(batch, device):
    pc, ((label, rotations, force), depth), pos, dof, obs, object_id, camera_id, augment = batch
    pc = pc.float().to(device)
    label = label.float().to(device)
    augment = augment.float().to(device)
    rotations = rotations.float().to(device)
    force = force.float().to(device)
    pos.unsqueeze_(1)  # B, 1, 3
    pos = pos.float().to(device)
    depth = depth.float().to(device)
    obs = obs.float().to(device)

    return pc, (label, rotations, force, depth), pos, dof, obs, object_id, camera_id, augment

def select(out):
    return out.squeeze(1)

def loss_pair(y_pred, y):
    loss_qual = _qual_loss_fn(y_pred, y)
    loss_dict = {"loss_all": loss_qual.mean()}
    return loss_qual.mean(), loss_dict

def loss_fn(y_pred, y, augment, beta):
    ignore_augment = torch.where(augment > 0, torch.zeros_like(augment), torch.ones_like(augment))

    kl_loss, recon_loss, score_pred, rot_pred, force_pred, critic_pred, point_score_label = y_pred

    # adjust augment point_score_label
    # point_score_label = point_score_label + augment
    point_score_label = torch.where(augment > 0, torch.zeros_like(point_score_label), point_score_label)

    label, rotations, force, depth = y
    loss_score = _qual_loss_fn(score_pred.squeeze(), point_score_label)
    loss_rot = _rot_loss_fn(rot_pred, rotations)
    loss_force = _force_loss_fn(force_pred, force)
    loss_critic = _qual_loss_fn(critic_pred.squeeze(), label)
    # loss = loss_score + kl_loss_z * 0.2 + (loss_rot + loss_force) * label * 5 + loss_critic
    # loss = kl_loss_z * 0.1 + trans_loss * 0.8 + loss_critic + (loss_rot + loss_force) * label * 5 + loss_score

    if kl_loss:
        # loss = loss_critic * 4 + (loss_rot + loss_force) * label * 5 + loss_score * 2 + kl_loss * 0.05 * beta + recon_loss
        # loss_eval = loss_critic * 4 + (loss_rot + loss_force) * label * 5 + loss_score * 2 + kl_loss * 0.06 + recon_loss
        loss = loss_critic * 4 + (loss_rot + loss_force) * label * 15 + loss_score * 2 + kl_loss * 0.05 * beta + recon_loss
        loss_eval = loss_critic * 4 + (loss_rot + loss_force) * label * 15 + loss_score * 2 + kl_loss * 0.06 + recon_loss
    else:
        loss = loss_critic * 4 + (loss_rot + loss_force) * label * 5 + loss_score * 2 + recon_loss
        loss_eval = loss_critic * 4 + (loss_rot + loss_force) * label * 5 + loss_score * 2 + recon_loss
        kl_loss = torch.zeros(1)

    loss_dict = {
        "loss_all": loss.mean().item(),
        "loss_score": loss_score.mean().item(),
        "loss_kl": (kl_loss).mean().item() * 0.06,
        "loss_critic": loss_critic.mean().item() * 2,
        "loss_force": (loss_force * label).mean().item() * 5,
        "loss_rotation": (loss_rot * label).mean().item() * 5,
        "loss_transition": (recon_loss).mean().item(),
        "loss_eval": (loss_eval).mean().item(),
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
    # return 1.0 - torch.abs(torch.sum(pred_norm * target_norm, dim=1))
    return torch.norm(target_norm - pred_norm, dim=1)

def create_trainer(net, metric_model, optimizer, loss_fn, metrics, device):
    def _update(this_engine, batch):
        net.is_train()
        optimizer.zero_grad()

        current_epoch = this_engine.state.epoch
        beta = betas[current_epoch]

        # forward
        x, y, pos, dof, obs, object_id, camera_id, augment = prepare_batch(batch, device)
        y_pred = net(x, y, pos, obs, camera_id, metric_model)
        loss, loss_dict = loss_fn(y_pred, y, augment, beta)

        # backward
        loss.backward()
        optimizer.step()

        return x, y_pred, y, loss_dict

    trainer = Engine(_update)

    for name, metric in metrics.items():
        metric.attach(trainer, name)

    return trainer

def create_evaluator(net, metric_model, loss_fn, metrics, device):
    def _inference(this_engine, batch):
        net.evaluation()
        with torch.no_grad():
            current_epoch = this_engine.state.epoch
            beta = betas[current_epoch]

            x, y, pos, dof, obs, object_id, camera_id, augment = prepare_batch(batch, device)
            y_pred = net(x, y, pos, obs, camera_id, metric_model)
            loss, loss_dict = loss_fn(y_pred, y, augment, beta)
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
    parser.add_argument("--net", default="giga_feature")
    parser.add_argument("--logdir", type=Path, default="data/runs")
    parser.add_argument("--description", type=str, default="task-cvae")
    parser.add_argument("--savedir", type=str, default="")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    # parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--silence", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--load-path", type=str, default="")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--task", action="store_true", default=True)
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--use-metric", default=True, action="store_false")
    parser.add_argument("--stochastic", default=False, action="store_true")
    parser.add_argument("--gmvae", default=False, action="store_true")
    parser.add_argument("--category", default=4)
    parser.add_argument("--discrete", default=False, action="store_true")
    parser.add_argument("--finetune", default=False, action="store_true")
    parser.add_argument("--pointnet", default=False, action="store_true")

    args = parser.parse_args()
    print(args)
    main(args)
