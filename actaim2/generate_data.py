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

def find_obj_cate(object_id):
    # find object category from reading the all_objects.txt file
    file_path = "./dataset_scale/all_objects.txt"
    with open(file_path, 'r') as file:
        for line in file:
            curr_object_id, curr_object_cate = line.strip().split(',')[0], line.strip().split(',')[1]
            if object_id == curr_object_id:
                return curr_object_cate
    return None

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
    parser.add_argument("--cgn", default=False, action="store_true", help="whether to use contact graspnet to predict grasp during data collection")
    args = parser.parse_args()

    set_np_formatting()
    with open("./config/franka_object.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config["env"]["datasetPath"] = args.dataset

    object_init_state = None
    print("#" * 30)
    print("current index: ", args.obj)
    print("#" * 30)

    if not args.cgn:
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
    else:
        # read object and object pose from grasps
        save_grasp_dir = "./dataset/grasps"
        grasps_list = os.listdir(save_grasp_dir)

        print("--------num grasps: ", len(grasps_list))

        picked_obj_cate = grasps_list[args.obj] if args.obj < len(grasps_list) else random.randint(0, len(grasps_list) - 1)
        object_id = picked_obj_cate.split("_")[0]
        config["env"]["objectId"] = object_id
        object_init_state = (picked_obj_cate.strip().split("_")[1]).split(".")[0]

        object_cate = find_obj_cate(object_id)
        if object_cate is None:
            print("grasp object does not exist!")
            exit()


    config["env"]["cate"] = object_cate

    camera_id = 0
    print(config)

    num_envs = 150
    config["env"]["numEnvs"] = num_envs
    physics_engine = gymapi.SIM_PHYSX
    graphics_device = 0

    sim_params = parse_vis_sim_params(config)
    # Using new isaacgym
    # sim_params.use_gpu_pipeline = False

    env = FrankaAffordance(config, sim_params, physics_engine, graphics_device, device)
    env.set_camera_id(int(camera_id))

    env.set_object_cate(object_cate)

    # start with random initial pose
    env.multi_pose_init(args.random_init, object_init_state)
    print("finish init env")

    # vis env here to compute the candidate point cloud
    # distroy camera after it done
    start_time = time()
    env.init_vision(False)

    # iteration for collecting data
    num_iter = 40

    # steps num for manipulation
    init_step = 240
    reach_step = 100
    close_gripper_step = 50
    force_step = 100
    steps = [init_step, reach_step, close_gripper_step, force_step]

    # initialize cem
    franka_base_pos = env.get_franka_base_pos()

    weights_dim = 6 + 3 + 3 # 6 initial pose, 3 reach to grasp point, 3 push or pull
    mean = 0.0
    std = 2.0
    n_elite_perc = 0.13
    n_elite = int(num_envs * n_elite_perc)

    reward_list = []
    reward_std = []
    # stores successful action during CEM
    success_actions = [torch.Tensor().to(device) for k in range(4)]
    success_threshold = 100


    # np.random.seed(0)
    total_collected_traj = 0


    for i in range(num_iter):
        print("iter: ", i)
        weights_pop = [mean + std * np.random.randn(weights_dim) for i_weight in range(num_envs)]

        '''
        if args.use_primitives:
            rot, init_pos, grasp_pos, move_pos = weight2action_primitives(weights_pop, env, franka_base_pos)
        else:
            rot, init_pos, grasp_pos, move_pos = weight2action(weights_pop, franka_base_pos)
        action_0, action_1, action_2, action_3 = action_primitives_tensor(rot, init_pos, grasp_pos, move_pos, env.device)
        actions = [action_0, action_1, action_2, action_3]
        '''


        # env.test_action(pc, rotation, force, init_poses)
        env.reset(torch.arange(env.num_envs, device=env.device))
        env.sample_action(args.is_cluster, args.cluster_method, i)
        action_0, action_1, action_2, action_3 = env.get_action_seq()
        actions = [action_0, action_1, action_2, action_3]
        stages = ["init", "reach", "grasp", "manipulate"]

        for j in range(4):
            env.set_task_state(j)
            perform_action(env, actions[j], steps[j])
            print(f"----stage {stages[j]} complete")

            # Save multi camera observation here
            # TODO create a new code to save image
            # TODO saving image obsevation is space/time-consuming
            # env.save_obs()

        # env.check_metric(args.use_metric, metric_model, args.metric_obs)
        env.check_hit_rate(eval=True)
        save_num = env.save_action_data(dataset_root=args.savedata, is_clustering=True, cluster_method="dof")
        total_collected_traj += save_num
        if total_collected_traj > 200:
            exit()
        print("collected traj: ", total_collected_traj)
        # print("------------------object cate: ", object_cate)