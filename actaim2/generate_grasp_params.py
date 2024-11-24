from pathlib import Path
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import torchgeometry as tgm

import pdb
import os

import math
import numpy as np
import torch
import yaml
from time import time
import random
import argparse

from affordance.utils.config import (
    set_np_formatting,
    set_seed,
    get_args,
    parse_sim_params,
    load_cfg,
    parse_vis_sim_params,
)
from affordance.utils.parse_task import parse_task
# from affordance.envs.franka_object import FrankaObject
# from affordance.envs.franka_robot_object import FrankaRobotObject

from affordance.envs.franka_affordance import FrankaAffordance

# franka base computation to set franka workspace
workspace_sphere_r = 0.5
workspace_sphere_R = 1.5
device = "cuda:0"


def main():
    pass

def perform_action(env, action, total_step):
    for j in range(total_step):
        env.render()
        env.set_local_step(j)
        env.step(action)


def read_object_id_cate(file_path):
    # Open the file in read mode
    object_id_list = []
    object_cate_list = []
    with open(file_path, 'r') as file:
        # Read the file line by line
        for line in file:
            # Process each line
            object_id, object_cate = line.strip().split(',')[0], line.strip().split(',')[1]
            object_id_list.append(object_id)
            object_cate_list.append(object_cate.capitalize())
    return object_id_list, object_cate_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_primitives", default=True, help='Whether to use simpler action primitives')
    parser.add_argument("--dataset", type=str, default="./dataset_scale")
    # table 20411
    parser.add_argument("--object_id", default=20411, type=str, help='which object to use')
    parser.add_argument("--random_init", default=True, dest="random_init", action="store_true", help="whether collect with random initial pose or fixed pose")
    parser.add_argument("--is_cluster", dest="is_cluster", default=True)
    parser.add_argument("--cluster_method", type=str, default="gmm")
    parser.add_argument("--savedata", type=str, default="dataset")
    parser.add_argument("--multi_obj", action="store_true", default=False)
    parser.add_argument("--obj", type=int, default=0, help="which line of object is using in file table_id.txt")
    args = parser.parse_args()

    set_np_formatting()
    with open("./config/franka_object.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config["env"]["datasetPath"] = args.dataset

    if not args.multi_obj:
        config["env"]["objectId"] = args.object_id
        object_id_file = None
        object_cate = "Table"
    else:
        # object_id_file = "./dataset_scale/all_objects.txt"
        object_id_file = "./dataset_scale/table_id.txt"
        object_id_list, object_cate_list = read_object_id_cate(object_id_file)
        obj_index = args.obj if args.obj < len(object_id_list) else random.randint(0, len(object_id_list) - 1)
        config["env"]["objectId"] = object_id_list[obj_index]
        object_cate = object_cate_list[obj_index]

    print("#" * 30)
    print("current index: ", args.obj)
    print("#" * 30)

    config["env"]["cate"] = object_cate

    camera_id = 0
    print(config)

    num_envs = 2
    config["env"]["numEnvs"] = num_envs
    physics_engine = gymapi.SIM_PHYSX
    graphics_device = 0

    sim_params = parse_vis_sim_params(config)
    # Using new isaacgym
    # sim_params.use_gpu_pipeline = False

    env = FrankaAffordance(config, sim_params, physics_engine, graphics_device, device)
    env.set_camera_id(int(camera_id))

    env.set_object_cate(object_cate)

    save_grasp_params_dir = args.savedata + "/" + "grasp_params"
    Path(save_grasp_params_dir).mkdir(parents=True, exist_ok=True)

    # List all files in the directory
    all_files = os.listdir(save_grasp_params_dir)
    # Get all files for current object
    obj_file = [file.split(".")[0] for file in all_files if file.split("_")[0] == args.object_id]
    obj_existing_state = [obj_state["-1"] for obj_state in obj_file]

    # start with random initial pose
    obj_init_state = env.multi_pose_init(args.random_init)

    attempt = 0
    while obj_init_state in obj_existing_state and attempt > 100:
        obj_init_state = env.multi_pose_init(args.random_init)
        attempt += 1


    # Doing for reset
    env.reset(torch.arange(env.num_envs, device=env.device))
    observation = env.compute_observations()
    for _ in range(2):
        env.render()
        env.step(-1)

    env.viewer_camera(0, False)

    env.generate_graspnet_pc()
    env.generate_contact_graspnet_params(save_grasp_params_dir)
