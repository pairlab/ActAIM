import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils import tensorboard
import torch.nn as nn
import os
from tqdm import tqdm

# TODO this is stupid
# TODO torchvision and torch versions are incompatible, need to call vgg first
vgg_model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19_bn', pretrained=True)

import random
import pdb
import wandb
from new_scripts import utils
from new_scripts.dataset.dataset import DatasetSeq, DatasetTuple
from new_scripts.model.peract.perceiver_lang_io import PerceiverVoxelLangEncoder
from new_scripts.model.rvt.mvt_single import MVT
from new_scripts.model.agent_encoder import QFunction
from new_scripts.model.diffusion_models import Model_Afford_Diffusion, Model_Afford_Transformer
from new_scripts.model.agent_decoder import PerActDecoder
from new_scripts.model.task.emvn.model import ResNetMVGCNN
from new_scripts.model.task.vq import TaskModel
from new_scripts.model.task.gt import GTaskModel, ZeroTaskModel, GoalTaskModel
from new_scripts.model.task.gmvae import GmvaeTaskModel
from new_scripts.helpers.optim.lamb import Lamb
from new_scripts.model.rvt.renderer import BoxRenderer

from new_scripts.model.task.multi_view_encoder import MultiViewEncoder
from affordance.utils.gmvae import CGMVAENet
from affordance.utils.LossFunctions import LossFunctions
from new_scripts.model.task.multi_view_encoder import ResnetBlockFC

wandb_checkpoint_file = "checkpoint.tar"


# Wrapper model for training
class Wrapper(nn.Module):
    def __init__(self, batch_size, voxel_size, device, net="transformer_tuple_ce", model_type="peract",
                 encoder="vqvae"):
        super().__init__()

        self.device = device

        latent_dim = 512
        self.multi_view_encode = MultiViewEncoder(latent_dim)
        self.visual_encode_len = 640
        self.y_dim = latent_dim // 2
        self.num_class = 10

        self.cgmvae = CGMVAENet(self.visual_encode_len, self.visual_encode_len, self.y_dim, self.num_class)
        self.w_rec = 10
        self.w_gauss = 2
        self.w_cat = 50

        self.softmax = nn.Softmax(dim=1)
        self.losses = LossFunctions()
        self.recon_loss = torch.nn.MSELoss()

        self.cond_decode = ResnetBlockFC(self.visual_encode_len * 2, self.visual_encode_len)


    def train_task(self, curr_color_img, final_color_img):

        curr_x = curr_color_img
        final_x = final_color_img

        bs = curr_x.shape[0]
        curr_x = self.multi_view_encode(curr_x).view(bs, -1)
        final_x = self.multi_view_encode(final_x).view(bs, -1)

        # define task representation as the difference between initial and final obs
        task_label = final_x - curr_x
        out_net = self.cgmvae.forward(task_label, curr_x)

        z, final_x_predict, skill = out_net['gaussian'], out_net['x_rec'], out_net['skill']
        logits, prob_cat = out_net['logits'], out_net['prob_cat']
        y_mu, y_var = out_net['y_mean'], out_net['y_var']
        mu, var = out_net['mean'], out_net['var']


        # reconstruction loss task label recon
        loss_rec_task = self.recon_loss(task_label.detach(), skill)

        # gaussian loss
        loss_gauss = self.losses.gaussian_loss(z, mu, var, y_mu, y_var)

        # categorical loss
        loss_cat = -self.losses.entropy(logits, prob_cat) - np.log(0.1)

        # total loss
        kl_loss = self.w_gauss * loss_gauss + self.w_cat * loss_cat
        trans_loss = self.w_rec * loss_rec_task

        total_loss = kl_loss + trans_loss

        embed_cond_x = torch.cat((skill, curr_x), dim=-1)
        final_x_pred = self.cond_decode(embed_cond_x)

        # reconstruction loss final_x
        loss_rec_final_pred = self.recon_loss(final_x.detach(), final_x_pred)

        total_loss = self.w_rec * loss_rec_final_pred + total_loss

        loss_dict = {
            "loss_total": total_loss,
            "loss_recon_task": self.w_rec * loss_rec_task,
            "loss_gaussian": self.w_gauss * loss_gauss,
            "loss_categorical": self.w_cat * loss_cat,
            "loss_kl": kl_loss,
            "loss_recon_final": self.w_rec * loss_rec_final_pred,
        }

        return total_loss, loss_dict


def convert_parallel_dict(save_model_dict):
    new_state_dict = {}
    for k, v in save_model_dict.items():
        if 'module' in k:
            k = k.replace('module.', '')
        new_state_dict[k] = v
    return new_state_dict


# Define your training function
def train(rank, world_size, model, device, train_loader, valid_loader, start_epoch, num_epochs, logdir, learning_rate,
          description, slurm_id):
    # Define your optimizer
    optimizer = Lamb(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.00001,
        betas=(0.9, 0.999),
        adam=False,
    )

    # Initialize wandb run
    if rank == 0:
        description_list = description.split(",")
        description_list.pop(1)
        new_description = ",".join(description_list)

        wandb.init(name=new_description, project="PerActAim", id=slurm_id)
        if slurm_id is not None:
            dir_name = logdir.split("/")[-1]
            checkpoint_dir = logdir.replace(dir_name, "")
            checkpoint_list = os.listdir(os.path.join(checkpoint_dir))

            find_checkpoint = utils.find_earliest_checkpoint(checkpoint_list, description)
            if find_checkpoint is not None:
                checkpoint_file = checkpoint_dir + find_checkpoint + "/" + wandb_checkpoint_file
                if Path(checkpoint_file).exists():
                    # load wandb weight
                    checkpoint = torch.load(checkpoint_file)

                    new_state_dict = convert_parallel_dict(checkpoint['model_state_dict'])
                    model.load_state_dict(new_state_dict)
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    start_epoch = checkpoint['epoch']

                    # delete the newly-made directory
                    os.system(f'rm -rf {logdir}')

                    # update new description
                    description = find_checkpoint
                    logdir = checkpoint_dir + find_checkpoint

    # Set the device based on the process rank
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Configure distributed training
    # torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)

    # Set the model to the correct device
    model = model.to(device)

    # model = nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)

    # Example training loop
    for epoch in range(start_epoch, num_epochs):
        # Set model to training mode
        model.train()
        save_loss_dict = {}

        # TODO delete
        # save image to track training
        save_image_per_epoch = 0
        max_save_image_per_epoch = 5

        # Example: Iterate over the training data loader
        for step, data in enumerate(tqdm(train_loader)):
            # print(f"training with step {step} in epoch {epoch}")

            x_batch, y_batch, curr_color_img, final_color_img, dof, traj_step, view_matrices = utils.preprocess_data_task(
                    data, device)
            # print(traj_step)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            loss, loss_dict = model.train_task(curr_color_img, final_color_img)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # save training loss
            utils.add_loss_dict(save_loss_dict, loss_dict, step, mode="Train_")
            # print(loss_dict)

        # Set model to evaluation mode
        model.eval()

        # Example: Iterate over the validation data loader
        with torch.no_grad():
            # save image to track training
            save_image_per_epoch = 0
            max_save_image_per_epoch = 0

            for step, data in enumerate(valid_loader):
                # Move data to the device
                x_batch, y_batch, curr_color_img, final_color_img, dof, traj_step, view_matrices = utils.preprocess_data_task(
                    data, device)

                eval_loss, eval_loss_dict = model.train_task(curr_color_img, final_color_img)


                utils.add_loss_dict(save_loss_dict, eval_loss_dict, step, mode="Validate_")

        # Save the model checkpoint every 10 steps
        if rank == 0:
            if (epoch + 1) % 20 == 0:
                torch.save(model.state_dict(), logdir + f"/model_checkpoint_{epoch}.pth")

            if (epoch + 1) % 1 == 0 and slurm_id is not None:
                # save resume point
                save_file = logdir + "/" + wandb_checkpoint_file
                torch.save({  # Save our checkpoint loc
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, save_file)

        if rank == 0:
            # Example: Compute and log the validation loss
            for loss_name, loss_value in save_loss_dict.items():
                wandb.log({loss_name: loss_value}, step=epoch)
            print("----train epoch :", epoch)
            print("train epoch loss: ", save_loss_dict)

    if rank == 0:
        # Save the final model checkpoint
        torch.save(model.state_dict(), logdir + "/final_model.pth")
        wandb.finish()


def main(args):
    # LOSS_KEYS = ["loss_all", "loss_critic", "loss_rotation", "loss_force", "loss_score", "loss_kl", "loss_transition", "loss_eval"]
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": args.gpu, "pin_memory": True} if use_cuda else {}

    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

    if args.slurm_id is not None:
        SLURM_JOB_ID = args.slurm_id
    else:
        SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID')

    time_stamp = datetime.now().strftime("%y-%m-%d-%H-%M")
    description = "task,time={},batch_size={},lr={:.0e},gpu_num={},slurm_id={}".format(
        time_stamp,
        args.batch_size,
        args.lr,
        args.gpu,
        SLURM_JOB_ID,
    ).strip(",")

    if args.savedir == "":
        # create log directory
        logdir = args.logdir / description
    else:
        logdir = Path(args.savedir)

    logdir.mkdir(parents=True, exist_ok=True)
    logdir = str(logdir)
    # create data loaders
    train_loader, val_loader = create_train_val_loaders(
        args, kwargs
    )
    print("------finish loading dataset")
    # Data loader for action sequence
    # TODO for debugging

    start_epoch = 0

    # save_model = "net=transformer_tuple_ce,model=rvt,encoder=vqvae,time=24-01-19-20-15,batch_size=4,lr=6e-04,gpu_num=1,slurm_id=None/model_checkpoint_159.pth"
    # save_model = "net=transformer_tuple_ce,model=rvt,encoder=vqvae,time=24-01-23-00-43,batch_size=4,lr=6e-04,gpu_num=1,slurm_id=None/model_checkpoint_59.pth"
    # net_type_index = save_model.index(",")
    # net_type = save_model[4:net_type_index]

    # save_file = args.logdir / save_model
    # save_model_dict = torch.load(save_file)

    # new_state_dict = convert_parallel_dict(save_model_dict)

    model = Wrapper(args.batch_size, args.voxel_size, device)
    # model.load_state_dict(new_state_dict)

    # Set the number of processes to use
    num_processes = args.gpu

    # TODO this is for debugging
    train(0, num_processes, model, device, train_loader, val_loader, start_epoch, args.epochs, logdir, args.lr,
          description, SLURM_JOB_ID)

    # Spawn multiple processes for training using torch.multiprocessing.spawn
    # mp.spawn(train, args=(num_processes, model, device, train_loader, val_loader, start_epoch, args.epochs, logdir, args.lr, description, SLURM_JOB_ID), nprocs=num_processes, join=True)


def create_train_val_loaders(args, kwargs):
    batch_size = args.batch_size
    val_split = args.val_split
    partial_ratio = args.partial

    dataset = DatasetTuple(is_rvt = True)

    # split into train and validation sets
    val_size = int(val_split * len(dataset) * partial_ratio)
    train_size = int(len(dataset) * partial_ratio) - val_size
    train_set, val_set, _ = torch.utils.data.random_split(dataset,
                                                          [train_size, val_size, len(dataset) - train_size - val_size])
    # create loaders for both datasets
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
    return train_loader, val_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slurm-id", type=str, default=None, help="for continue training")
    parser.add_argument("--logdir", type=Path, default="data/runs")
    parser.add_argument("--description", type=str, default="task-cvae")
    parser.add_argument("--savedir", type=str, default="")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0004)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--silence", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--load-path", type=str, default="")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--partial", type=float, default=1.0)
    parser.add_argument("--voxel-size", type=int, default=100)

    parser.add_argument("--master-addr", default="localhost")
    parser.add_argument("--master-port", default="29500")

    # parser.add_argument("--use-metric", default=True, action="store_false")
    # parser.add_argument("--stochastic", default=False, action="store_true")
    # parser.add_argument("--gmvae", default=False, action="store_true")
    # parser.add_argument("--category", default=4)
    # parser.add_argument("--discrete", default=False, action="store_true")
    # parser.add_argument("--finetune", default=False, action="store_true")
    # parser.add_argument("--pointnet", default=False, action="store_true")

    args = parser.parse_args()
    print(args)
    main(args)
