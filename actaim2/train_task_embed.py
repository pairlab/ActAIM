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

import pdb
import wandb
from new_scripts.dataset.dataset import DatasetSeq, DatasetTuple
from new_scripts.model.peract.perceiver_lang_io import PerceiverVoxelLangEncoder
from new_scripts.model.agent_encoder import QFunction
from new_scripts.model.diffusion_models import Model_Afford_Diffusion, Model_Afford_Transformer
from new_scripts.model.agent_decoder import PerActDecoder
from new_scripts.model.task.emvn.model import ResNetMVGCNN
from new_scripts.model.task.multi_view_encoder import MultiViewEncoder
from new_scripts.model.task.vq import TaskModel, ImageDecoder
from new_scripts.helpers.optim.lamb import Lamb



# Wrapper model for training
class Wrapper(nn.Module):
    def __init__(self, batch_size, device, net="diffusion_tuple"):
        super().__init__()

        self.device = device

        self.net = net

        self.task_encoder = TaskModel().to(device)
        self.img_decoder = ImageDecoder().to(device)

        self.img_decoder.to(device)
        self.task_encoder.to(device)

        # set extra_steps sampling to achieve higher-likelihood regions
        self.extra_diffusion_step = 16

        # ratio to set task and bc training loss
        self.bc_loss_weight = 1.0
        self.task_embed_loss_weight = 2.0

    def forward(self, x_batch, y_batch, curr_color_img, final_color_img):
        total_loss, clip_loss, vec_recon_loss, task_embed, quantized, task_label = self.task_encoder.test_task_embed(curr_color_img, final_color_img)
        bs = quantized.shape[0]
        num_view = quantized.shape[1]

        # quantized = quantized.view(bs * num_view, -1)
        # img_recon_loss, recon_img = self.img_decoder(quantized, final_color_img)

        loss_dict = {
            "total_loss": total_loss,
            "clip_loss": clip_loss,
            "vec_recon_loss": vec_recon_loss,
        }

        return total_loss, loss_dict

    def set_decoder_device(self, device):
        self.model.decoder.set_device(device)


    def sample(self, x_batch, curr_color_img, final_color_img):
        # TODO need to write functio to sample task_embed instead of using final_color_img
        task_loss, task_embed, perplexity = self.task_encoder(curr_color_img, final_color_img)
        y_predict = self.model.sample_extra(x_batch, task_embed, extra_steps=self.extra_diffusion_step)

        # TODO argmax y_predict
        return y_predict


    def evaluation(self):
        self.eval = True

    def is_train(self):
        self.eval = False

# Define your training function
def train(rank, world_size, model, train_loader, valid_loader, num_epochs, logdir, description):
    # Initialize wandb run
    if rank == 0:
        wandb.init(name=description, project="PerActAim")

    # Set the device based on the process rank
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Configure distributed training
    torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)

    # Set the model to the correct device
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)

    # Define your optimizer

    optimizer = Lamb(
        model.parameters(),
        lr=0.0005,
        weight_decay=0.000001,
        betas=(0.9, 0.999),
        adam=False,
    )

    # Example training loop
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()

        save_loss_dict = {}

        # Example: Iterate over the training data loader
        for step, data in enumerate(tqdm(train_loader)):
            # print(f"training with step {step} in epoch {epoch}")

            x_batch, y_batch, curr_color_img, final_color_img = preprocess_data(data, device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            loss, loss_dict = model(x_batch, y_batch, curr_color_img, final_color_img)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # save training loss
            add_loss_dict(save_loss_dict, loss_dict, step, mode="Train_")
            # print(loss_dict)

        # Set model to evaluation mode
        model.eval()

        # Example: Iterate over the validation data loader
        with torch.no_grad():
            for step, data in enumerate(valid_loader):
                # Move data to the device
                x_batch, y_batch, curr_color_img, final_color_img = preprocess_data(data, device)

                # Forward pass and Compute loss
                eval_loss, eval_loss_dict = model(x_batch, y_batch, curr_color_img, final_color_img)

                add_loss_dict(save_loss_dict, eval_loss_dict, step, mode="Validate_")

        # Save the model checkpoint every 10 steps
        if rank == 0:
            if (epoch + 1) % 20 == 0:
                torch.save(model.state_dict(), logdir + f"/model_checkpoint_{epoch}.pth")

        if rank == 0:
            # Example: Compute and log the validation loss
            for loss_name, loss_value in save_loss_dict.items():
                wandb.log({loss_name: loss_value}, step=epoch)
            print("----train epoch :", epoch)
            print("train epoch loss: ", save_loss_dict)

    if rank == 0:
        # Save the final model checkpoint
        torch.save(model.state_dict(), logdir+"/final_model.pth")
        wandb.finish()


def add_loss_dict(save_loss_dict, loss_dict, step, mode:str):
    for loss_name, loss_value in loss_dict.items():
        save_loss_name = mode + loss_name

        if save_loss_name not in save_loss_dict.keys():
            save_loss_dict[save_loss_name] = loss_value.item()
        else:
            current_value = save_loss_dict[save_loss_name]
            save_loss_dict[save_loss_name] = ((current_value * step) + loss_value.item()) / (step + 1)

def main(args):
    # LOSS_KEYS = ["loss_all", "loss_critic", "loss_rotation", "loss_force", "loss_score", "loss_kl", "loss_transition", "loss_eval"]
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": args.gpu, "pin_memory": True} if use_cuda else {}

    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

    time_stamp = datetime.now().strftime("%y-%m-%d-%H-%M")
    description = "net={},time={},batch_size={},lr={:.0e},dataset={},{}".format(
        args.net,
        time_stamp,
        args.batch_size,
        args.lr,
        args.dataset,
        args.description,
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
    # Data loader for action sequeence
    # object_cate, object_id, init_state, dof_change, lang_prompt, positions, rotations, color_img, depth_img, color_voxel, depth_voxel = next(iter(train_loader))

    model = Wrapper(args.batch_size, device, args.net)

    # Set the number of processes to use
    num_processes = args.gpu

    # TODO this is for debugging
    # train(1, num_processes, model, train_loader, val_loader, args.epochs, logdir, description)


    # Spawn multiple processes for training using torch.multiprocessing.spawn
    mp.spawn(train, args=(num_processes, model, train_loader, val_loader, args.epochs, logdir, description), nprocs=num_processes, join=True)


def preprocess_data(data, device):
    # Testing dataloader
    object_cate, object_id, init_state, dof_change, lang_prompt, pos, rotations, gripper_open_close, \
    curr_color_voxel, curr_depth_voxel, curr_color_img, curr_depth_img, \
    final_color_voxel, final_depth_voxel, final_color_img, final_depth_img, franka_proprio = data

    curr_color_voxel = curr_color_voxel.to(device).float()
    curr_depth_voxel = curr_depth_voxel.to(device).float()
    franka_proprio = franka_proprio.to(device).float()

    pos = pos.to(device).float()
    rotations = rotations.to(device).float()
    gripper_open_close = gripper_open_close.to(device).float()

    # concat voxel input
    curr_depth_voxel = curr_depth_voxel.unsqueeze(-1).permute(0, 4, 1, 2, 3)
    curr_color_voxel = curr_color_voxel.permute(0, 4, 1, 2, 3)

    voxel_input = torch.cat((curr_color_voxel, curr_depth_voxel), dim=1)

    # determine input and output
    x_batch = (voxel_input, franka_proprio, lang_prompt)
    y_batch = (pos, rotations, gripper_open_close)

    curr_color_img = preprocess_image(curr_color_img, device)
    final_color_img = preprocess_image(final_color_img, device)

    return x_batch, y_batch, curr_color_img, final_color_img

def preprocess_image(images, device):
    shp = images.shape # [B, v, w,h,c]
    images = images.view((shp[0] * shp[1], *shp[2:]))
    images = images.permute((0, 3, 1, 2))

    # preprocess image size to (1, 3, 224, 224)
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # TODO continue from here
    # need to iterate over to convert PIL
    image_tensor = torch.Tensor().to(device)

    for i in range(shp[0]*shp[1]):
        image_i = images[i]
        image_i = image_i[:3, :, :]
        image_i_tensor = preprocess(image_i)
        image_i_tensor = image_i_tensor.unsqueeze(0).to(device)
        image_tensor = torch.cat((image_tensor, image_i_tensor), dim=0)

    images_tensor = image_tensor.view(shp[0], shp[1], shp[-1] - 1, 224, 224)
    return images_tensor

def create_train_val_loaders(args, kwargs):
    batch_size = args.batch_size
    val_split = args.val_split
    partial_ratio = args.partial

    # load the dataset

    if "tuple" in args.net:
        dataset = DatasetTuple()
    elif "sequence" in args.net:
        dataset = DatasetSeq()
    else:
        print("invalid data type")
        exit()

    # split into train and validation sets
    val_size = int(val_split * len(dataset) * partial_ratio)
    train_size = int(len(dataset) * partial_ratio) - val_size
    train_set, val_set, _ = torch.utils.data.random_split(dataset, [train_size, val_size, len(dataset)-train_size-val_size])
    # create loaders for both datasets
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
    return train_loader, val_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str, default="task_embed_sequence")
    parser.add_argument("--logdir", type=Path, default="data/runs")
    parser.add_argument("--description", type=str, default="task-cvae")
    parser.add_argument("--savedir", type=str, default="")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--silence", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--load-path", type=str, default="")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--task", action="store_true", default=True)
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--partial", type=float, default=1.0)

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
