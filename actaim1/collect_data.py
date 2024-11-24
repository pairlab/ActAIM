import pdb

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import glob
import math
import numpy as np
import torch
import yaml
from time import time
import argparse
import random
import csv

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
from affordance.envs.franka_object_origin import FrankaObject
from affordance.metric.encoder import VoxelEncoder, PixelEncoder, DepthEncoder, AutoEncoder

import timeout_decorator
@timeout_decorator.timeout(50, timeout_exception=StopIteration)
def env_generate(
    env, num_traj, save_dataset, is_rgb, is_clustering, metric_model, use_metric, metric_obs, cluster_method, eval, augment, pull, object_cate
):
    if eval == True:
        augment = False
    # vis env here
    start_time = time()

    reach_step = 80
    close_gripper_step = 60
    force_step = 80

    if object_cate is not None:
        env.set_object_cate(object_cate)
    for i in range(num_traj):
        # is_clustering = (i != 0) and is_clustering
        # TODO debug
        env.sample_action(is_clustering, cluster_method, pull, i)
        env.reset(torch.arange(env.num_envs, device=env.device))

        for j in range(reach_step):
            env.render()
            env.step(0)
        print("reaching complete")

        for j in range(close_gripper_step):
            env.render()
            env.step(1)
        print("gripper close")
        env.filter_action()

        for j in range(force_step):
            env.render()
            env.step(2)
        print("force complete")

        env.check_metric(use_metric, metric_model, metric_obs)
        env.check_hit_rate(eval=eval, model_eval=None)
        env.save_hit("./" + save_dataset)

        env.save_data("./" + save_dataset, is_rgb, is_clustering, use_metric, cluster_method, eval, augment, i)
        # env.report_replay_buf()

if __name__ == "__main__":
    start_time = time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi", dest="multi", action="store_true")
    parser.add_argument("--pose", dest="multi_pose", action="store_true")
    parser.add_argument("--dataset", type=str, default="./data_sample/where2act_original_sapien_dataset")
    parser.add_argument("--rgb", dest="rgb", action="store_true")
    parser.add_argument("--first", dest="first", action="store_true")
    parser.add_argument("--cluster", dest="cluster", default=True, action="store_false")
    parser.add_argument("--cluster_method", type=str, default="gmm")
    parser.add_argument("--use_metric", dest="use_metric", default=True)
    parser.add_argument("--metric_obs", type=str, default="depth")
    parser.add_argument("--obj", default=19898)
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--augment", action="store_true", default=False)
    parser.add_argument("--pull", default=False, action="store_true")
    parser.add_argument("--savedata", type=str,default="dataset")

    # TODO test save data
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--test_id", type=int, default=0)

    parser.set_defaults(first=False)
    parser.set_defaults(rgb=False)
    parser.set_defaults(multi=False)
    parser.set_defaults(multi_pose=False)
    args = parser.parse_args()

    set_np_formatting()
    with open("./config/franka_object.yaml", "r") as f:
        config = yaml.safe_load(f)

    object_id = None
    object_cate = None
    print("=======================Process Line: ", int(args.test_id), "==============================")
    if args.test:
        # more objects
        config["env"]["datasetPath"] = "./sapien/where2act_original_sapien_dataset"
        line_index = int(args.test_id)
        # This is for selecting "available data"
        # with open('./available_object.txt', 'r') as csvfile:
        with open('./eval_object.txt', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i == line_index:
                    which_object = row
                    break
        object_id = int(which_object[0].split(' ')[0])
        object_cate = which_object[0].split(' ')[1]
        config["env"]["objectId"] = object_id
    else:
        config["env"]["datasetPath"] = args.dataset
        config["env"]["objectId"] = args.obj
    physics_engine = gymapi.SIM_PHYSX
    graphics_device = 0
    device = "cuda:0"
    sim_params = parse_vis_sim_params(config)

    metric_model = None
    if args.use_metric:
        if args.metric_obs == "tsdf":
            metric_model = VoxelEncoder().to(device)
        elif args.metric_obs == "rgbd":
            metric_model = PixelEncoder().to(device)
        elif args.metric_obs == "depth":
            metric_model = DepthEncoder(c_dim=2).to(device)

        metric_model = AutoEncoder().to(device)

        # path = './data/metric/metric,time=21-11-06-15-25,obs=depth,loss=marginnet=autoencoder,batch_size=32,lr=1e-06/metric.pt' # single object metric learning model
        # path = './data/metric/metric,time=22-01-17-16-05,obs=depth,loss=marginnet=autoencoder,batch_size=32,lr=1e-06/metric.pt'
        path = "./autoencoder/metric.pt"
        metric_model.load_state_dict(torch.load(path))
        metric_model.eval()

    num_traj = 8
    if args.eval:
        num_traj = 1

    if args.multi:
        object_file_list = glob.glob(args.dataset + "/*")
        object_id_list = [int(id.split("/")[-1]) for id in object_file_list]
        random.shuffle(object_id_list)
        config["env"]["objectId"] = object_id_list[0]
        print("object: ", object_id_list[0])
        env = FrankaObject(config, sim_params, physics_engine, graphics_device, device)
        env.multi_pose_init(args.multi_pose)
        env.init_vision(args.first)
        env_generate(
            env,
            num_traj,
            args.savedata,
            args.rgb,
            args.cluster,
            metric_model,
            args.use_metric,
            args.metric_obs,
            args.cluster_method,
            args.eval,
            args.augment,
            args.pull,
            object_cate
        )
    else:
        print("object: ", config["env"]["objectId"])
        env = FrankaObject(config, sim_params, physics_engine, graphics_device, device)
        env.multi_pose_init(args.multi_pose)
        env.init_vision(args.first)
        env_generate(
            env,
            num_traj,
            args.savedata,
            args.rgb,
            args.cluster,
            metric_model,
            args.use_metric,
            args.metric_obs,
            args.cluster_method,
            args.eval,
            args.augment,
            args.pull,
            object_cate
        )
    end_time = time()
    # print("generate traj takes time: ", end_time - start_time)

    '''
    if args.eval and args.test:
        from pathlib import Path
        csv_path = './available_object.txt'
        csv_path = Path(csv_path)
        with csv_path.open("a") as f:
            f.write(
                " ".join(
                    [
                        str(object_id),
                        object_cate
                    ]
                )
            )
            f.write("\n")
    '''



