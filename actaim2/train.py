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

import sys
sys.path.append("./")

from new_scripts import utils
from new_scripts.dataset.dataset import DatasetSeq, DatasetTuple
from new_scripts.model.peract.perceiver_lang_io import PerceiverVoxelLangEncoder
from new_scripts.model.rvt.mvt_single import MVT
from new_scripts.model.agent_encoder import QFunction
from new_scripts.model.diffusion_models import Model_Afford_Diffusion, Model_Afford_Transformer
from new_scripts.model.agent_decoder import PerActDecoder
from new_scripts.model.task.emvn.model import ResNetMVGCNN
from new_scripts.model.task.multi_view_encoder import MultiViewEncoder
from new_scripts.model.task.vq import TaskModel
from new_scripts.model.task.cgmvae import CGmvaeTransformer
from new_scripts.model.task.gt import GTaskModel, ZeroTaskModel, GoalTaskModel
from new_scripts.model.task.gmvae import GmvaeTaskModel
from new_scripts.helpers.optim.lamb import Lamb
from new_scripts.model.rvt.renderer import BoxRenderer


wandb_checkpoint_file = "checkpoint.tar"

# for phoenix dataset
phoenix_dataset_prefix = "/storage/home/hcoda1/8/lwang831/scratch/"

# Wrapper model for training
class Wrapper(nn.Module):
    def __init__(self, batch_size, voxel_size, device, net="transformer_tuple_ce", model_type="peract", encoder="vqvae"):
        super().__init__()

        self.device = device

        self.net = net

        self.model_type = model_type

        self.encoder = encoder
        self.add_lang = False

        if self.encoder == "vqvae":
            self.task_encoder = TaskModel().to(device)

        elif self.encoder == "gt":
            self.task_encoder = GTaskModel().to(device)

        elif self.encoder == "goal":
            self.task_encoder = GoalTaskModel().to(device)

        elif self.encoder == "lang":
            self.task_encoder = ZeroTaskModel().to(device)
            self.add_lang = True

        elif self.encoder == "gmvae":
            self.task_encoder = GmvaeTaskModel().to(device)

        elif self.encoder == "cgmvae":
            self.task_encoder = CGmvaeTransformer().to(device)

        else:
            print("wrong encoder input")
            print("valid choices: vqvae, gt, goal, lang, gmvae")
            exit()

        seq_horizon = 4

        # TODO training robot seq = 2
        seq_horizon = 2

        if "seq" in self.net:
            self.act_type = "seq"
            decoder_batch_size = batch_size * seq_horizon
        elif "tuple" in self.net:
            self.act_type = "tuple"
            decoder_batch_size = batch_size
        else:
            self.act_type = None
            print("specify wrong prediction type")
            exit()

        if "ce" in self.net:
            self.loss_type = "ce"
        elif "mse" in self.net:
            self.loss_type = "mse"
        else:
            self.loss_type = None
            print("wrong loss type")
            exit()

        if "diffusion" in self.net:
            # ce loss does not work in diffusion model
            self.loss_type = "mse"

        # TODO change to voxel_size 100
        if self.model_type == "rvt":
            transformer_encoder = MVT(loss_type=self.loss_type, act_type=self.act_type, add_lang=self.add_lang)
        else:
            transformer_encoder = PerceiverVoxelLangEncoder(loss_type=self.loss_type, act_type=self.act_type, depth=3, iterations=1, voxel_size=voxel_size, initial_dim=4, low_dim_size=9, add_lang=self.add_lang)


        transformer_encoder = transformer_encoder.to(device)

        q_func = QFunction(transformer_encoder, rotation_resolution=5.0, device=device, training=True)
        decoder = PerActDecoder(device, decoder_batch_size, loss_type=self.loss_type, voxel_size=voxel_size, num_rotation_classes=72)

        # diffusion model params
        n_T = 50
        drop_prob = 0.0

        # Test final diffusion model
        if "diffusion" in self.net:
            self.model = Model_Afford_Diffusion(
                q_func,
                decoder,
                betas=(1e-4, 0.02),
                n_T=n_T,
                device=device,
                drop_prob=drop_prob,
                guide_w=0.0,
            )
        elif "transformer" in self.net:
            self.model = Model_Afford_Transformer(
                q_func,
                decoder,
                betas=(1e-4, 0.02),
                n_T=n_T,
                device=device,
                drop_prob=drop_prob,
                guide_w=0.0,
            )

        self.model = self.model.to(device)

        # set extra_steps sampling to achieve higher-likelihood regions
        self.extra_diffusion_step = 16

        # ratio to set task and bc training loss
        self.bc_loss_weight = 1.0
        self.task_embed_loss_weight = 0.6


        # TODO never do the pretraining here
        if self.encoder == "vqvae" or self.encoder == "gmvae" or self.encoder == "cgmvae":
            self.pretrain_epoch = 0
        else:
            self.pretrain_epoch = 0


    def train_task(self, x_batch, y_batch, curr_color_img, final_color_img, step, dof=None, view_matrices=None, epoch=9999):
        if self.encoder == "vqvae" or self.encoder == "gmvae" or self.encoder == "cgmvae":
            task_loss, task_embed, perplexity, task_label, final_x = self.task_encoder(curr_color_img, final_color_img)
        elif self.encoder == "gt" or self.encoder == "lang" or self.encoder == "goal":
            task_loss, task_embed, task_label, final_x = self.task_encoder(curr_color_img, final_color_img, dof)

        loss_dict = {
            "loss_all": task_loss,
        }

        return task_loss, loss_dict, None


    def forward(self, x_batch, y_batch, curr_color_img, final_color_img, step, dof=None, view_matrices=None, epoch=9999):
        if self.encoder == "vqvae" or self.encoder == "gmvae":
            task_loss, task_embed, perplexity, task_label, final_x = self.task_encoder(curr_color_img, final_color_img)
        elif self.encoder == "gt" or self.encoder == "lang" or self.encoder == "goal":
            task_loss, task_embed, task_label, final_x = self.task_encoder(curr_color_img, final_color_img, dof)
        elif self.encoder == "cgmvae":
            task_loss, task_embed, perplexity, task_label, final_x = self.task_encoder(curr_color_img, final_color_img, epoch)
        else:
            print("invalid encoder")
            exit()

        # curr_color_img shape: [bs, num_cam, num_channel, img_height, img_width]
        # task_embed shape: [bs, 640]

        if epoch < self.pretrain_epoch:
            task_embed = task_label

        # Blocking the gradient
        task_embed = task_embed.detach()

        if "tuple" in self.net:
            bc_loss, q_trans_loss, q_rot_loss, q_grip_loss, q_task_loss, q_trans = self.model.loss_on_batch(x_batch, y_batch, task_embed, task_label, step, view_matrices)  # BC loss
        elif "seq" in self.net:
            bc_loss, q_trans_loss, q_rot_loss, q_grip_loss, q_task_loss, q_trans = self.model.loss_on_batch_seq(x_batch, y_batch, task_embed, task_label, step, view_matrices)  # BC loss

        total_loss = self.bc_loss_weight * bc_loss + self.task_embed_loss_weight * task_loss + self.bc_loss_weight * q_task_loss * 0.1
        # total_loss = self.task_embed_loss_weight * task_loss + self.bc_loss_weight * q_task_loss

        loss_dict = {
            "loss_all": total_loss,
            "loss_bc": self.bc_loss_weight * bc_loss,
            "loss_task": self.task_embed_loss_weight * task_loss,
            "loss_trans": self.bc_loss_weight * q_trans_loss,
            "loss_rot": self.bc_loss_weight * q_rot_loss,
            "loss_grip": self.bc_loss_weight * q_grip_loss,
            "loss_task_pred": self.bc_loss_weight * q_task_loss * 0.1,
        }

        return total_loss, loss_dict, q_trans

    def set_decoder_device(self, device):
        self.model.decoder.set_device(device)

    def get_model_type(self):
        return self.model_type

    def get_model_act_type(self):
        return self.act_type

    def sample(self, x_batch, curr_color_img, task_embed, step, device):
        # TODO need to write functio to sample task_embed instead of using final_color_img
        task_ind = None
        if task_embed is None:
            task_embed, task_ind = self.task_encoder.sample(curr_color_img, device)

        if "seq" in self.act_type:
            y_predict = self.model.sample_extra_seq(x_batch, task_embed, extra_steps=self.extra_diffusion_step)

            # y_trans, y_rot, y_collide = y_predict
            # plot_voxel(y_trans.squeeze())

            action = self.model.decode_action(y_predict)
        else:
            y_predict = self.model.sample_extra(x_batch, task_embed, step, extra_steps=self.extra_diffusion_step)
            # y_trans, y_rot, y_collide = y_predict
            # plot_voxel(y_trans.squeeze())

            if self.loss_type == "mse":
                action = self.model.decode_action_value(y_predict)
            elif self.loss_type == "ce":
                action = self.model.decode_action(y_predict)
            else:
                exit()
        # TODO argmax y_predict
        return action, task_embed, task_ind


    def evaluation(self):
        self.eval = True

    def is_train(self):
        self.eval = False

    def set_renderer(self, renderer):
        self.model.decoder.set_renderer(renderer)

def convert_parallel_dict(save_model_dict):
    new_state_dict = {}
    for k, v in save_model_dict.items():
        if 'module' in k:
            k = k.replace('module.', '')
        new_state_dict[k] = v
    return new_state_dict

# Define your training function
def train(rank, world_size, model, device, train_loader, valid_loader, args, description, slurm_id, save_policy_only=False):
    # unpack values from args
    learning_rate = args.lr
    start_epoch = args.start_epoch
    num_epochs = args.epochs
    logdir = args.logdir

    # Load the trained model if finetuning
    if args.finetune and args.encoder == "cgmvae" and "tuple" in args.net:
        print("load cgmvae finetuning")
        # load task embedding model

        if args.cluster == "phoenix":
            save_model_file_encoder = "/storage/home/hcoda1/8/lwang831/scratch/trained_model/task_encoder.pth"
        else:
            save_model_file_encoder = "../trained_model/model/task_encoder.pth"
        save_model_dict_encoder = torch.load(save_model_file_encoder)
        save_model_dict_encoder = convert_parallel_dict(save_model_dict_encoder)
        model.task_encoder.network.load_state_dict(save_model_dict_encoder)

    if args.finetune and args.model == "rvt":
        print("load policy finetuning")
        # load pretrained policy network

        if args.cluster == "phoenix":
            save_model_file_policy = "/storage/home/hcoda1/8/lwang831/scratch/trained_model/policy.pth"
        else:
            save_model_file_policy = "../trained_model/model/policy.pth"
        save_model_dict_policy = torch.load(save_model_file_policy)
        save_model_dict_policy = convert_parallel_dict(save_model_dict_policy)
        model.model.load_state_dict(save_model_dict_policy)


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
        time_stamp_index = 6
        description_list.pop(time_stamp_index) # pop time
        new_description = ",".join(description_list)
        new_id = ",".join(description_list[1:time_stamp_index])

        wandb.init(name=new_description, project="PerActAim", resume=True, id=new_id)
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
        print("save directory: ", logdir)
        start_epoch = wandb.run.step

    model_type = model.get_model_type()
    model_act_type = model.get_model_act_type()
    # Set the device based on the process rank
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Configure distributed training
    # torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)

    # device = torch.device(f'cuda:{rank}')
    device = torch.device(f'cuda:0')

    # Set the model to the correct device
    model = model.to(device)
    model.set_decoder_device(device)

    if model_type == "rvt":
        # initialize renderer
        img_height = 320
        img_width = 320
        add_depth = True

        renderer = BoxRenderer(
            device=device,
            img_size=(img_height, img_width),
            with_depth=add_depth,
        )
        model.set_renderer(renderer)

    # model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

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

            train_task_embed_only = False
            if model_type == "rvt" and not train_task_embed_only and "tuple" in args.net:
                x_batch, y_batch, curr_color_img, final_color_img, dof, traj_step, view_matrices = utils.preprocess_data_rvt(data, renderer, device, args.dataset)
            elif model_type == "rvt" and not train_task_embed_only and "seq" in args.net: # TODO real robot pretrain
                x_batch, y_batch, curr_color_img, final_color_img, dof, traj_step, view_matrices = utils.preprocess_data_rvt_robot(
                    data, renderer, device, args.dataset)
            elif train_task_embed_only:
                x_batch, y_batch, curr_color_img, final_color_img, dof, traj_step, view_matrices = utils.preprocess_data_task(data, device)
            else:
                x_batch, y_batch, curr_color_img, final_color_img, dof, traj_step, view_matrices = utils.preprocess_data(data, device)
            # print(traj_step)


            # Zero the gradients
            optimizer.zero_grad()
                
            # Forward pass
            loss, loss_dict, q_trans = model(x_batch, y_batch, curr_color_img, final_color_img, traj_step, dof, view_matrices, epoch)
            # loss, loss_dict, q_trans = model.train_task(x_batch, y_batch, curr_color_img, final_color_img, traj_step, dof, view_matrices, epoch)

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
                if model_type == "rvt" and "tuple" in args.net:
                    x_batch, y_batch, curr_color_img, final_color_img, dof, traj_step, view_matrices = utils.preprocess_data_rvt(
                        data, renderer, device)

                elif model_type == "rvt" and "seq" in args.net:  # TODO real robot pretrain
                    x_batch, y_batch, curr_color_img, final_color_img, dof, traj_step, view_matrices = utils.preprocess_data_rvt_robot(
                        data, renderer, device, args.dataset)

                else:
                    x_batch, y_batch, curr_color_img, final_color_img, dof, traj_step, view_matrices = utils.preprocess_data(
                        data, device)

                # Forward pass and Compute loss
                eval_loss, eval_loss_dict, q_trans = model(x_batch, y_batch, curr_color_img, final_color_img, traj_step, dof, view_matrices, epoch)
                # eval_loss, eval_loss_dict, q_trans = model.train_task(x_batch, y_batch, curr_color_img, final_color_img, traj_step, dof, view_matrices, epoch)

                # TODO render the output results here
                if rank == 0 and "tuple" in model_act_type:
                    rand_num = random.uniform(0, 1)
                    image_epoch_root = "./image_epoch"
                    Path(image_epoch_root).mkdir(parents=True, exist_ok=True)

                    # random the input
                    if rand_num < 0.2 and save_image_per_epoch < max_save_image_per_epoch:
                        img_input, _, _ = x_batch
                        bs_ind = 0  # pick batch index
                        img_rgb_input = img_input[bs_ind, :, 3:-1, :, :]
                        q_trans_i = q_trans[bs_ind]
                        utils.draw_epoch_img_tensor(img_rgb_input, q_trans_i, image_epoch_root, epoch,
                                                    save_image_per_epoch)
                        save_image_per_epoch += 1

                utils.add_loss_dict(save_loss_dict, eval_loss_dict, step, mode="Validate_")

        # Save the model checkpoint every 10 steps
        if rank == 0:
            if save_policy_only:
                save_model_state_dict = model.model.state_dict()
            else:
                save_model_state_dict = model.state_dict()
            if (epoch + 1) % 4 == 0:
                torch.save(save_model_state_dict, logdir + f"/model_checkpoint_{epoch}.pth")

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
        torch.save(model.state_dict(), logdir+"/final_model.pth")
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

    if args.save_policy_only is True:
        args.encoder = 'goal'

    time_stamp = datetime.now().strftime("%y-%m-%d-%H-%M")
    description = "net={},model={},encoder={},cluster={},dataset={},without_init={},time={},batch_size={},lr={:.0e},save_policy_only={},gpu_num={},slurm_id={}".format(
        args.net,
        args.model,
        args.encoder,
        args.cluster,
        args.dataset,
        args.without_init,
        time_stamp,
        args.batch_size,
        args.lr,
        args.save_policy_only,
        args.gpu,
        SLURM_JOB_ID,
    ).strip(",")

    if args.savedir == "":
        # create log directory
        logdir = args.logdir / description
    else:
        logdir = Path(args.savedir)

    logdir_str = str(logdir)
    if args.cluster == "phoenix":
        logdir_str = phoenix_dataset_prefix + logdir_str
        logdir = Path(logdir_str)
        args.dataset = phoenix_dataset_prefix + args.dataset

    logdir.mkdir(parents=True, exist_ok=True)
    # args.logdir = str(logdir)

    # create data loaders
    train_loader, val_loader = create_train_val_loaders(
        args, kwargs
    )
    print("------finish loading dataset")
    # Data loader for action sequeence
    # TODO for debugging

    save_model = "net=transformer_tuple_ce,model=rvt,encoder=cgmvae,cluster=utm,dataset=dataset_all,time=24-02-25-22-31,batch_size=4,lr=4e-04,save_policy_only=False,gpu_num=1,slurm_id=None/model_checkpoint_19.pth"
    # save_model = "net=transformer_tuple_ce,model=rvt,encoder=vqvae,time=24-01-23-00-43,batch_size=4,lr=6e-04,gpu_num=1,slurm_id=None/model_checkpoint_59.pth"
    # net_type_index = save_model.index(",")
    # net_type = save_model[4:net_type_index]
    new_state_dict = None
    if args.encoder == "cgmvae" and not args.without_init:
        args.finetune = False
        if args.without_init:
            if args.cluster == "phoenix":
                save_model = "/storage/home/hcoda1/8/lwang831/scratch/trained_model/cgmvae_pretrain_without_init.pth"
            else:
                save_model = "../trained_model/model/cgmvae_pretrain_without_init.pth"

                # run real robot mdoel
                save_model = "../trained_model/model/model_checkpoint_47.pth"
        else:
            if args.cluster == "phoenix":
                save_model = "/storage/home/hcoda1/8/lwang831/scratch/trained_model/cgmvae_pretrain.pth"
            else:
                save_model = "../trained_model/model/cgmvae_pretrain.pth"

        save_model_dict = torch.load(save_model)
        new_state_dict = convert_parallel_dict(save_model_dict)

    model = Wrapper(args.batch_size, args.voxel_size, device, net=args.net, model_type=args.model, encoder=args.encoder)
    if args.encoder == "cgmvae" and args.net == "transformer_tuple_ce":
        model.load_state_dict(new_state_dict)

    # Set the number of processes to use
    num_processes = args.gpu
    args.logdir = str(logdir)

    # TODO this is for debugging
    train(1, num_processes, model, device, train_loader, val_loader, args, description, SLURM_JOB_ID, save_policy_only=args.save_policy_only)

    # Spawn multiple processes for training using torch.multiprocessing.spawn
    mp.spawn(train, args=(num_processes, model, device, train_loader, val_loader, args, description, SLURM_JOB_ID), nprocs=num_processes, join=True)


def create_train_val_loaders(args, kwargs):
    batch_size = args.batch_size
    val_split = args.val_split
    partial_ratio = args.partial

    is_rvt = True if args.model=="rvt" else False

    # load the dataset

    if "tuple" in args.net:
        dataset = DatasetTuple(is_rvt=is_rvt, root=args.dataset, without_init=args.without_init)
    elif "seq" in args.net:
        dataset = DatasetSeq(is_rvt=is_rvt, root=args.dataset, without_init=args.without_init)
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
    parser.add_argument("--net", type=str, default="transformer_tuple_ce", help="form: {net_arch}_{output_type}_{loss_type}")
    parser.add_argument("--encoder", type=str, default="cgmvae", help="select from {gt} {lang} {vqvae} {gmvae} {goal} {cgmvae}")
    parser.add_argument("--slurm-id", type=str, default=None, help="for continue training")
    parser.add_argument("--model", type=str, default="rvt", help="PerAct or RVT")
    parser.add_argument("--logdir", type=Path, default="data/runs")
    parser.add_argument("--description", type=str, default="task-cvae")
    parser.add_argument("--savedir", type=str, default="")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--start-epoch", type=int, default=0)

    parser.add_argument("--cluster", type=str, default="utm", help="form: [utm, vector, phoenix, shi]")

    parser.add_argument("--finetune", action="store_true", default=False, help="whether to use pretrained module when training cgmvae")

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--silence", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--load-path", type=str, default="")
    parser.add_argument("--dataset", default="dataset_all")
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--partial", type=float, default=1.0)
    parser.add_argument("--voxel-size", type=int, default=100)
    parser.add_argument("--save-policy-only", action="store_true", default=False)
    parser.add_argument("--without-init", action="store_true", default=False, help="whether to train the model with or without initial state")

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
