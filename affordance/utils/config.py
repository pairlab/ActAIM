# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
import yaml

from isaacgym import gymapi
from isaacgym import gymutil

import numpy as np
import random
import torch


def set_np_formatting():
    np.set_printoptions(
        edgeitems=30,
        infstr="inf",
        linewidth=4000,
        nanstr="nan",
        precision=2,
        suppress=False,
        threshold=10000,
        formatter=None,
    )


def warn_task_name():
    raise Exception(
        "Unrecognized task!\nTask should be one of: [AntRun, BallBalance, Cartpole, CartpoleYUp, Humanoid, FrankaCabinet, ShadowHand]"
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def retrieve_cfg(args, use_rlg_config=False):
    if use_rlg_config:
        if args.task == "BallBalance":
            return os.path.join(args.logdir, "ball_balance"), "cfg/train/rlg_base.yaml", "cfg/ball_balance.yaml"
        elif args.task == "Cartpole":
            return os.path.join(args.logdir, "cartpole"), "cfg/train/rlg_base.yaml", "cfg/cartpole.yaml"
        elif args.task == "CartpoleYUp":
            return os.path.join(args.logdir, "cartpole_y_up"), "cfg/train/rlg_base.yaml", "cfg/cartpole_y_up.yaml"
        elif args.task == "Ant":
            return os.path.join(args.logdir, "ant"), "cfg/train/rlg_ant.yaml", "cfg/ant.yaml"
        elif args.task == "Humanoid":
            return os.path.join(args.logdir, "humanoid"), "cfg/train/rlg_humanoid.yaml", "cfg/humanoid.yaml"
        elif args.task == "FrankaCabinet":
            return (
                os.path.join(args.logdir, "franka_cabinet"),
                "cfg/train/rlg_franka_cabinet.yaml",
                "cfg/franka_cabinet.yaml",
            )
        elif args.task == "ShadowHand":
            return os.path.join(args.logdir, "shadow_hand"), "cfg/train/rlg_shadow_hand.yaml", "cfg/shadow_hand.yaml"
        elif args.task == "FrankaGrasp":
            return os.path.join(args.logdir, "franka_grasp"), "cfg/train/rlg_franka_grasp.yaml", "cfg/franka_grasp.yaml"
        elif args.task == "FrankaHammer":
            return (
                os.path.join(args.logdir, "franka_hammer"),
                "cfg/train/rlg_franka_hammer.yaml",
                "cfg/franka_hammer.yaml",
            )
        elif args.task == "FrankaPush":
            return os.path.join(args.logdir, "franka_push"), "cfg/train/rlg_franka_push.yaml", "cfg/franka_push.yaml"
        elif args.task == "FrankaInsert":
            return (
                os.path.join(args.logdir, "franka_insert"),
                "cfg/train/rlg_franka_insert.yaml",
                "cfg/franka_insert.yaml",
            )
        elif args.task == "FrankaObject":
            return (
                os.path.join(args.logdir, "franka_object"),
                "cfg/train/rlg_franka_object.yaml",
                "cfg/franka_object.yaml",
            )
        elif args.task == "FrankaHook":
            return os.path.join(args.logdir, "franka_hook"), "cfg/train/rlg_franka_hook.yaml", "cfg/franka_hook.yaml"
        elif args.task == "FrankaReach":
            return os.path.join(args.logdir, "franka_reach"), "cfg/train/rlg_franka_reach.yaml", "cfg/franka_reach.yaml"
        else:
            warn_task_name()
    else:
        if args.task == "BallBalance":
            return (
                os.path.join(args.logdir, "ball_balance"),
                "cfg/train/pytorch_ppo_ball_balance.yaml",
                "cfg/ball_balance.yaml",
            )
        elif args.task == "Cartpole":
            return os.path.join(args.logdir, "cartpole"), "cfg/train/pytorch_ppo_cartpole.yaml", "cfg/cartpole.yaml"
        elif args.task == "CartpoleYUp":
            return (
                os.path.join(args.logdir, "cartpole_y_up"),
                "cfg/train/pytorch_ppo_cartpole.yaml",
                "cfg/cartpole_y_up.yaml",
            )
        elif args.task == "Ant":
            return os.path.join(args.logdir, "ant"), "cfg/train/pytorch_ppo_ant.yaml", "cfg/ant.yaml"
        elif args.task == "Humanoid":
            return os.path.join(args.logdir, "humanoid"), "cfg/train/pytorch_ppo_humanoid.yaml", "cfg/humanoid.yaml"
        elif args.task == "FrankaCabinet":
            return (
                os.path.join(args.logdir, "franka_cabinet"),
                "cfg/train/pytorch_ppo_franka_cabinet.yaml",
                "cfg/franka_cabinet.yaml",
            )
        elif args.task == "ShadowHand":
            return (
                os.path.join(args.logdir, "shadow_hand"),
                "cfg/train/pytorch_ppo_shadow_hand.yaml",
                "cfg/shadow_hand.yaml",
            )
        elif args.task == "FrankaGrasp":
            return (
                os.path.join(args.logdir, "franka_grasp"),
                "cfg/train/pytorch_ppo_franka_grasp.yaml",
                "cfg/franka_grasp.yaml",
            )
        elif args.task == "FrankaHammer":
            return (
                os.path.join(args.logdir, "franka_hammer"),
                "cfg/train/pytorch_ppo_franka_hammer.yaml",
                "cfg/franka_hammer.yaml",
            )
        elif args.task == "FrankaPush":
            return (
                os.path.join(args.logdir, "franka_push"),
                "cfg/train/pytorch_ppo_franka_push.yaml",
                "cfg/franka_push.yaml",
            )
        elif args.task == "FrankaInsert":
            return (
                os.path.join(args.logdir, "franka_insert"),
                "cfg/train/pytorch_ppo_franka_insert.yaml",
                "cfg/franka_insert.yaml",
            )
        elif args.task == "FrankaObject":
            return (
                os.path.join(args.logdir, "franka_object"),
                "cfg/train/pytorch_ppo_franka_object.yaml",
                "cfg/franka_object.yaml",
            )
        elif args.task == "FrankaHook":
            return (
                os.path.join(args.logdir, "franka_hook"),
                "cfg/train/pytorch_ppo_franka_hook.yaml",
                "cfg/franka_hook.yaml",
            )
        elif args.task == "FrankaReach":
            return (
                os.path.join(args.logdir, "franka_reach"),
                "cfg/train/pytorch_ppo_franka_reach.yaml",
                "cfg/franka_reach.yaml",
            )
        else:
            warn_task_name()


def load_cfg(args, use_rlg_config=False):
    with open(os.path.join(os.getcwd(), args.cfg_train), "r") as f:
        cfg_train = yaml.load(f, Loader=yaml.SafeLoader)

    with open(os.path.join(os.getcwd(), args.cfg_env), "r") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    # Override number of environments if passed on the command line
    if args.num_envs > 0:
        cfg["env"]["numEnvs"] = args.num_envs

    if args.episode_length > 0:
        cfg["env"]["episodeLength"] = args.episode_length

    # Set physics domain randomization
    if "task" in cfg:
        if "randomize" not in cfg["task"]:
            cfg["task"]["randomize"] = args.randomize
        else:
            cfg["task"]["randomize"] = args.randomize or cfg["task"]["randomize"]
    else:
        cfg["task"] = {"randomize": False}

    logdir = args.logdir
    if use_rlg_config:
        exp_name = cfg_train["params"]["config"]["name"]
        if args.experiment_name != "Base":
            exp_name = "{}_{}_{}_{}".format(
                args.experiment_name, args.task_type, args.device, str(args.physics_engine).split("_")[-1]
            )

        if cfg["task"]["randomize"]:
            exp_name += "_DR"

        # Override config name
        cfg_train["params"]["config"]["name"] = exp_name

        if args.resume > 0:
            cfg_train["params"]["load_checkpoint"] = True

        # Set maximum number of training iterations (epochs)
        if args.max_iterations > 0:
            cfg_train["params"]["config"]["max_epochs"] = args.max_iterations

        cfg_train["params"]["config"]["num_actors"] = cfg["env"]["numEnvs"]

        seed = cfg_train.get("seed", 42)
        if args.seed > 0:
            seed = args.seed
        cfg["seed"] = seed
        cfg_train["seed"] = seed

        cfg["args"] = args
    else:
        # Override seed if passed on the command line
        if args.seed > 0:
            cfg_train["seed"] = args.seed

        log_id = args.logdir
        if args.experiment_name != "Base":
            log_id = args.logdir + "_{}_{}_{}_{}".format(
                args.experiment_name, args.task_type, args.device, str(args.physics_engine).split("_")[-1]
            )
            if cfg["task"]["randomize"]:
                log_id += "_DR"

        logdir = os.path.realpath(log_id)
        os.makedirs(logdir, exist_ok=True)

    return cfg, cfg_train, logdir


def parse_vis_sim_params(config):
    # initialize sim
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.num_client_threads = 0
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = 4
    sim_params.physx.use_gpu = True
    sim_params.physx.num_subscenes = 0
    sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024
    sim_params.use_gpu_pipeline = True

    # if sim options are provided in cfg, parse them and update/override above:
    gymutil.parse_sim_config(config["sim"], sim_params)
    # Override num_threads if passed on the command line
    sim_params.physx.num_threads = 0
    return sim_params


def parse_sim_params(args, cfg, cfg_train):
    # initialize sim
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.num_client_threads = args.slices

    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device == "GPU":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
        sim_params.flex.shape_collision_margin = 0.01
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 10
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = 4
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024

    if args.device == "GPU":
        sim_params.use_gpu_pipeline = True

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params


def get_args(benchmark=False, use_rlg_config=False):
    custom_parameters = [
        {"name": "--test", "action": "store_true", "default": False, "help": "Run trained policy, no training"},
        {
            "name": "--play",
            "action": "store_true",
            "default": False,
            "help": "Run trained policy, the same as test, can be used only by rl_games RL library",
        },
        {"name": "--resume", "type": int, "default": 0, "help": "Resume training or start testing from a checkpoint"},
        {
            "name": "--checkpoint",
            "type": str,
            "default": "",
            "help": "Path to the saved weights, only for rl_games RL library",
        },
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--task", "type": str, "default": "FrankaObject", "help": "FrankaObject"},
        {"name": "--task_type", "type": str, "default": "Python", "help": "Choose Python or C++"},
        {"name": "--device", "type": str, "default": "GPU", "help": "Choose CPU or GPU device for running physics"},
        {
            "name": "--ppo_device",
            "type": str,
            "default": "GPU",
            "help": "Choose CPU or GPU device for inferencing PPO network",
        },
        {"name": "--logdir", "type": str, "default": "logs/"},
        {"name": "--experiment_name", "type": str, "default": "Base"},
        {"name": "--cfg_train", "type": str, "default": "Base"},
        {"name": "--cfg_env", "type": str, "default": "Base"},
        {"name": "--seed", "type": int, "default": -1, "help": "Random seed"},
        {"name": "--max_iterations", "type": int, "default": 0, "help": "Set a maximum number of training iterations"},
        {
            "name": "--num_envs",
            "type": int,
            "default": 0,
            "help": "Number of environments to create - override config file",
        },
        {
            "name": "--episode_length",
            "type": int,
            "default": 0,
            "help": "Episode length, by default is read from yaml config",
        },
        {"name": "--randomize", "action": "store_true", "default": False, "help": "Apply physics domain randomization"},
    ]

    if benchmark:
        custom_parameters += [
            {"name": "--num_proc", "type": int, "default": 1, "help": "Number of child processes to launch"},
            {
                "name": "--random_actions",
                "action": "store_true",
                "help": "Run benchmark with random actions instead of inferencing",
            },
            {"name": "--bench_len", "type": int, "default": 10, "help": "Number of timing reports"},
            {"name": "--bench_file", "action": "store", "help": "Filename to store benchmark results"},
        ]

    # parse arguments
    args = gymutil.parse_arguments(description="RL Policy", custom_parameters=custom_parameters)

    if args.test:
        args.play = args.test
        args.train = False
    elif args.play:
        args.train = False
    else:
        args.train = True

    logdir, cfg_train, cfg_env = retrieve_cfg(args, use_rlg_config)

    # use custom parameters if provided by user
    if args.logdir == "logs/":
        args.logdir = logdir

    if args.cfg_train == "Base":
        args.cfg_train = cfg_train

    if args.cfg_env == "Base":
        args.cfg_env = cfg_env

    return args
