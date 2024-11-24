# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import os
import glob
import torch
import torchgeometry as tgm
import math
import open3d as o3d
import open3d.core as o3c

import math
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import collections
import time
import pickle
import uuid
import moviepy.video.io.ImageSequenceClip
from scipy.stats import entropy
import random
import pytorch3d.transforms

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from scipy.spatial.distance import cdist
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

from affordance.utils.torch_jit_utils import *
from affordance.envs.base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym.torch_utils import *

# from affordance.vision.perception import *
from affordance.vision.fusion import *
import umap
import pdb


# used for generating language prompt
nth = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth"
    # etc
}


class FrankaAffordance(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, graphics_device, device, save_video=False):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.graphics_device = graphics_device
        self.device = device
        self.save_video = save_video

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.noise_ = self.cfg["env"]["startRotationNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.reward_scale = self.cfg["env"]["rewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.pushDist = self.cfg["env"]["pushDist"]
        self.aggregate_mode = 3

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.distX_offset = 0.04
        self.dt = 1 / 60.0
        self.box_size = self.cfg["env"]["boxSize"]
        self.spacing = self.cfg["env"]["envSpacing"]

        self.object_id = self.cfg["env"]["objectId"]
        self.dataset_path = self.cfg["env"]["datasetPath"]
        self.init_pose_num = self.cfg["env"]["initPose"]
        self.object_cate = self.cfg["env"]["cate"]
        if self.object_cate == "Fridge":
            self.init_pose_num = 4

        self.is_multi_pose = False
        self.multi_pose_str = ""

        # distance from the eef point to gripper center point
        # depends on the gripper, 0.0584 is standard for franka gripper
        # self.eef_hand_dist = 0.0584 # floating gripper size
        # TODO add a small shifting
        self.eef_hand_dist = 0.0876 + 0.055 + 0.005  # robot gripper size
        self.eef_hand_dist = 0.0876 - 0.004
        # distance from interaction point to initial position
        self.init_dist = 0.25
        if self.object_cate == "Switch":
            self.init_dist = self.init_dist * 0.25

        if self.object_cate == "Faucet":
            self.init_dist = self.init_dist * 0.5


        # prop dimensions
        self.prop_width = 0.08
        self.prop_height = 0.08
        self.prop_length = 0.08
        self.prop_spacing = 0.09
        self.video_camera_handle = None

        # TODO add reach reward to envourage gripper reaching the object when reaching
        self.reach_reward = None
        self.unreachable_penalty = None
        self.interaction_points = None

        num_obs = 18
        num_acts = 9

        # the bound of object
        # self.tsdf_vol_bnds = np.array([[-1.5, 0.5], [-1.0, 1.0], [0, 2]])
        # self.tsdf_vol_bnds = np.array([[-1.2, 0.4], [-0.8, 0.8], [0, 1.6]])
        self.tsdf_vol_bnds = np.array([[-0.9, 0.3], [-0.6, 0.6], [0.0, 1.2]])

        # camera parameters
        self.num_camera = 4
        self.camera_id = random.randrange(self.num_camera)
        # self.camera_id = 2
        self.camera_view_id = (self.camera_id - 2 + self.num_camera) % self.num_camera

        pi = math.pi
        self.camera_height = 0.3
        # using 1.3
        self.camera_radius = 1.1
        self.view_angle = pi * (2 / 6)

        self.image_width = 256
        self.image_height = 256

        # graspnet camera property
        # self.num_graspnet_camera = 5
        # self.graspnet_image_width = 640
        # self.graspne_image_height = 640
        #
        # self.graspnet_camera_radius = 0.9
        # self.graspnet_view_angle = pi * (1 / 3)
        # self.graspnet_camera_height = 0.35

        # TODO checking the best grasping camera params
        self.num_graspnet_camera = 4
        self.graspnet_image_width = 640
        self.graspne_image_height = 640

        self.graspnet_camera_radius = 1.6
        self.graspnet_view_angle = pi * (1 / 3) * 3
        self.graspnet_camera_height = 0.7


        self.top_k_grasps = 20

        # save predicted grasp
        self.grasps = None
        self.heuristic_grasp_pc = None

        # save voxel param
        self.num_voxel_per_len = 100

        super().__init__(
            num_obs=num_obs,
            num_acts=num_acts,
            num_envs=self.cfg["env"]["numEnvs"],
            # num_envs=32,
            graphics_device=graphics_device,
            device=device,
        )

        # define state of the task
        # -1: initialize
        #  0: reaching
        #  1: closing gripper
        #  2: taking action
        self.task_state = -1
        self.pre_task_state = -2
        self.pre_target_pose = None

        self.epsilon = 0.7  # portion of random sample
        self.epsilon_grasp = 0.7
        self.epsilon_heuristic_grasp = 0.4

        self.start_fit = 24  # when to start GMM fitting
        self.success_threshold = 0.25

        self.select_points = torch.Tensor().to(self.device)
        self.mask = None
        self.tsdf = None
        self.local_step = -1
        self.video_frames = []
        self.curr_obs = None
        self.curr_rgb = None
        self.object_cate = None
        self.reward = None
        self.success = None
        self.lang_prompt = None

        # For computing camera intrinsics
        self.fu_list = None
        self.fv_list = None
        self.view_matrices = []
        self.view_matrices_rvt = []
        self.is_finetune = False
        self.data_generation = True


        # For storing visual observation
        self.rgba_tensor = torch.Tensor().to(self.device)
        self.depth_tensor = torch.Tensor().to(self.device)
        self.seg_tensor = torch.Tensor().to(self.device)

        # For storing visual observation but different camera pos
        self.rgba_rvt_tensor = torch.Tensor().to(self.device)
        self.depth_rvt_tensor = torch.Tensor().to(self.device)

        # Saving franka state
        self.franka_proprio = torch.Tensor().to(self.device)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # create some wrapper tensors for different slices
        self.franka_default_dof_pos = to_torch([0, 0, 0, -0.50, 0, 1.0000, 0, 0.04, 0.04], device=self.device)
        self.franka_default_dof_pos = to_torch([0, 0, 0, -1.17, 0, 1.8675, 0, 0.04, 0.04], device=self.device)

        self.franka_default_dof_pos = to_torch([0, -0.4, 0, -1.37, 0, 1.9675, 0, 0.04, 0.04], device=self.device)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.franka_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, : self.num_franka_dofs]
        self.franka_dof_pos = self.franka_dof_state[..., 0]
        self.franka_dof_vel = self.franka_dof_state[..., 1]

        self.dof_force = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, -1)
        self.contact = gymtorch.wrap_tensor(contact_tensor).view(self.num_envs, -1, 3)

        self.object_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_franka_dofs:]
        self.object_dof_pos = self.object_dof_state[..., 0]
        self.object_dof_vel = self.object_dof_state[..., 1]

        # TODO define new action primitives
        self.rotation = torch.zeros_like(self.franka_dof_pos[:, :3])
        self.interact_pos = torch.zeros_like(self.franka_dof_pos[:, :3])
        self.init_pos = torch.zeros_like(self.franka_dof_pos[:, :3])
        self.manipulate_pos = torch.zeros_like(self.franka_dof_pos[:, :3])
        self.init_grasp_dist = torch.zeros(self.num_envs)

        self.select_points = torch.zeros_like(self.franka_dof_pos[:, :3])

        self.global_indices = torch.arange(self.num_envs * 2, dtype=torch.int32, device=self.device).view(
            self.num_envs, -1
        )
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        scale = 0.5
        for i in range(len(self.object_dof_lower_limits)):
            self.multi_pose_str += str(1)

        object_dof_lower_limits_tensor = to_torch(self.object_dof_lower_limits, device=self.device)
        object_dof_upper_limits_tensor = to_torch(self.object_dof_upper_limits, device=self.device)

        object_dof_upper_limits_valid = torch.where(
            object_dof_upper_limits_tensor > 1000,
            torch.ones_like(object_dof_upper_limits_tensor) * math.pi,
            object_dof_upper_limits_tensor,
        )
        object_dof_lower_limits_valid = torch.where(
            object_dof_lower_limits_tensor < -1000,
            -torch.ones_like(object_dof_lower_limits_tensor) * math.pi,
            object_dof_lower_limits_tensor,
        )

        self.object_init_dof_pos = (
                object_dof_lower_limits_valid * scale + object_dof_upper_limits_valid * (1 - scale)
        ).repeat((self.num_envs, 1))

        self.object_dof_num = self.object_init_dof_pos.shape[-1]
        self.n_components = 2 * self.object_dof_num + 1  # number of modes + nothing happens mode

        self.obj_dof_change = torch.zeros((self.num_envs, self.object_dof_num)).to(self.device)
        self.obj_dof_change_str = []

        self.object_dof_pos_move_scale = torch.zeros((self.num_envs, self.object_dof_num)).to(self.device)
        self.pose_scale = torch.zeros((self.num_envs, self.object_dof_num)).to(self.device)

        # set downward direction vector
        self.down_dir = torch.Tensor([0, 0, -1]).to(self.device).view(1, 3)
        self.cluster_fit_data = None
        self.replay_buf_models = None
        self.success_data = torch.Tensor().to(self.device)

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        # print("actor_root_state_tensor shape: ", self.root_state_tensor.shape)
        # print("rigid_body_tensor shape: ", self.rigid_body_states.shape)
        # print("dof_state_tensor shape: ", self.dof_state.shape)
        # print("root_state_tensor shape: ", self.root_state_tensor.shape)
        self.object_actor_state = self.root_state_tensor[self.object_actor_idxs].clone()

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.franka_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.force = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.norm_force = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.valid_init_state = self.franka_init_state

        # TODO create replay buffer for sampling
        self.replay_buf = {}
        self.cluster_model = None

        # self.reset(torch.arange(self.num_envs, device=self.device))

    def set_finetune(self):
        self.is_finetune = True

    def set_task_state(self, task_state):
        self.pre_task_state = self.task_state
        self.task_state = task_state

    def set_local_step(self, step):
        self.local_step = step

    def set_camera_id(self, camera_id):
        self.camera_id = camera_id

    def get_object_dof_num(self):
        return self.object_dof_num

    def get_success(self):
        return self.success

    def get_hand_pose_np(self):
        hand_pose = torch.cat((self.hand_pos, self.hand_rot), dim=1).squeeze()
        hand_pose_np = hand_pose.cpu().numpy()

        return hand_pose_np

    def get_eef_pose_np(self):
        franka_eef_pos = (self.franka_lfinger_pos + self.franka_rfinger_pos) * 0.5
        eef_pose = torch.cat((franka_eef_pos, self.hand_rot), dim=1).squeeze()
        eef_pose_np = eef_pose.cpu().numpy()

        return eef_pose_np

    def generate_task_prompt(self):
        # generate task prompt based on final results
        if len(self.obj_dof_change_str) == 0:
            print("Not knowing object final state")
            exit()

        # assume that only one env
        assert len(self.obj_dof_change_str) == 1
        obj_change_str = self.obj_dof_change_str[0]

        if self.object_cate == "Table" or self.object_cate == "Storage":
            articulated_part_str = "drawer"
        elif self.object_cate == "Switch":
            articulated_part_str = "switch"
        elif self.object_cate == "Door":
            articulated_part_str = "door"
        elif self.object_cate == "Fridge":
            articulated_part_str = "door"
        elif self.object_cate == "Faucet":
            articulated_part_str = "faucet"
        elif self.object_cate == "Window":
            articulated_part_str = "window"
        elif self.object_cate == "Safe":
            articulated_part_str = "door"
        else:
            articulated_part_str = "part"


        if "1" in obj_change_str and "2" in obj_change_str:
            print("cannot open and close at the same time!")
            exit()

        if "1" in obj_change_str:
            action_prompt = "close the"
            if self.object_cate == "Faucet" or self.object_cate == "Switch":
                action_prompt = "turn off the"
            dof_change_str = "1"
        elif "2" in obj_change_str:
            action_prompt = "open the"
            if self.object_cate == "Faucet" or self.object_cate == "Switch":
                action_prompt = "turn on the"
            dof_change_str = "2"
        else:
            print("Unsuccessful manipulation")
            exit()

        manipulated_index = []
        for i in range(self.object_dof_num):
            if obj_change_str[i] == dof_change_str:
                manipulated_index.append(i)

        nth_drawer = ""
        num_manipulated_part = len(manipulated_index)
        for j in range(num_manipulated_part):
            nth_drawer += (" " + nth[manipulated_index[j] + 1])

            if len(manipulated_index) > 1 and j != num_manipulated_part - 1:
                nth_drawer += " and"

        self.lang_prompt = action_prompt + nth_drawer + " " + articulated_part_str

        return self.lang_prompt

    def multi_pose_init_all(self):
        object_init_dof_pos_rand = to_torch(self.object_dof_lower_limits, device=self.device).clone()
        multi_pose_index = to_torch(
            np.random.choice(self.init_pose_num, (self.num_envs, self.object_dof_num), replace=True), device=self.device
        )

        object_dof_upper_limits_tensor = to_torch(self.object_dof_upper_limits, device=self.device)
        object_dof_lower_limits_tensor = to_torch(self.object_dof_lower_limits, device=self.device)

        object_dof_upper_limits_valid = torch.where(
            object_dof_upper_limits_tensor > 1000,
            torch.ones_like(object_dof_upper_limits_tensor) * math.pi,
            object_dof_upper_limits_tensor,
        )
        object_dof_lower_limits_valid = torch.where(
            object_dof_lower_limits_tensor < -1000,
            -torch.ones_like(object_dof_lower_limits_tensor) * math.pi,
            object_dof_lower_limits_tensor,
        )

        multi_pose_scale = multi_pose_index / (self.init_pose_num - 1)
        object_dof_range = (object_dof_upper_limits_valid - object_dof_lower_limits_valid).unsqueeze(0)
        object_init_dof_pos_rand = object_init_dof_pos_rand.repeat((self.num_envs, 1))
        object_init_dof_pos_rand += multi_pose_scale * object_dof_range

        self.object_init_dof_pos = object_init_dof_pos_rand
        self.pose_scale = multi_pose_scale

    # function to collect different modes based on initial pose
    def multi_pose_modes(self):
        self.pose_scale = torch.zeros_like(self.pose_scale).to(self.device)

        # self.object_init_dof_pos stores the initial pose
        # self.rgbd stores the initial observation
        self.metric_init_dof = self.object_init_dof_pos[0].clone()
        object_init_dof_pos_rand = self.object_init_dof_pos.clone()
        object_dof_upper_limits_tensor = to_torch(self.object_dof_upper_limits, device=self.device)
        object_dof_lower_limits_tensor = to_torch(self.object_dof_lower_limits, device=self.device)

        object_dof_upper_limits_valid = torch.where(
            object_dof_upper_limits_tensor > 1000,
            torch.ones_like(object_dof_upper_limits_tensor) * math.pi,
            object_dof_upper_limits_tensor,
        )
        object_dof_lower_limits_valid = torch.where(
            object_dof_lower_limits_tensor < -1000,
            -torch.ones_like(object_dof_lower_limits_tensor) * math.pi,
            object_dof_lower_limits_tensor,
        )

        object_dof_range = object_dof_upper_limits_valid - object_dof_lower_limits_valid

        # 10% change of dof is successful interaction
        success_bound_index = self.init_pose_num // 10
        max_change_index_bound = self.init_pose_num // 3

        for i in range(self.num_envs):
            change_index = np.random.choice(self.object_dof_num, 1)
            if np.random.rand() >= 0.2:
                change_scale = np.random.randint(low=success_bound_index, high=max_change_index_bound)
            else:
                change_scale = np.random.randint(low=0, high=success_bound_index)

            change_scale /= self.init_pose_num
            # 50% of chance increase the dof value
            if np.random.rand() >= 0.5:
                object_init_dof_pos_rand[i, change_index] += change_scale * object_dof_range[change_index]
            # 50% of change decrease the dof value
            else:
                object_init_dof_pos_rand[i, change_index] -= change_scale * object_dof_range[change_index]

        object_dof_upper_limits_tensor_clamp = object_dof_upper_limits_tensor.unsqueeze(0).repeat(self.num_envs, 1)
        object_dof_lower_limits_tensor_clamp = object_dof_lower_limits_tensor.unsqueeze(0).repeat(self.num_envs, 1)
        self.object_init_dof_pos = tensor_clamp(
            object_init_dof_pos_rand, object_dof_lower_limits_tensor_clamp, object_dof_upper_limits_tensor_clamp
        )

        metric_init_dof_tensor = self.metric_init_dof.unsqueeze(0).repeat(self.num_envs, 1)
        metric_dof_change = self.object_init_dof_pos - metric_init_dof_tensor
        metric_dof_change_scale = metric_dof_change / object_dof_range

        self.pose_scale = torch.where(
            metric_dof_change_scale > 0.1, torch.ones_like(self.pose_scale).to(self.device), self.pose_scale
        )
        self.pose_scale = torch.where(
            metric_dof_change_scale < -0.1, -1 * torch.ones_like(self.pose_scale).to(self.device), self.pose_scale
        )

        # save rgbd as initial pose
        self.original_rgbd = self.rgbd.copy()
        self.original_tsdf = self.tsdf

    def test_action_tensor(self, points, rotation, force, init_poses=None):
        # TODO pre-define the num_envs as the number of action to be visualized
        self.select_points = points.float()
        self.norm_force = force.float()
        rotation = rotation.float()
        force_norm = torch.norm(self.norm_force - 0.5, dim=-1).unsqueeze(-1)
        self.force = ((self.norm_force - 0.5) / force_norm) * 0.5 * self.init_dist

        # TODO what if we only has push action
        # self.force = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1)) * 0.5 * self.init_dist
        self.force[:, 2] += self.init_dist

        z_axis = to_torch([0, 0, -1], device=self.device).repeat((self.num_envs, 1))
        init_pos = self.select_points + quat_apply(rotation, z_axis * (self.init_dist + self.eef_hand_dist))

        valid_state = torch.cat((init_pos, rotation), 1)
        init_vel = torch.zeros((self.num_envs, 6), device=self.device)
        self.valid_init_state = torch.cat((valid_state, init_vel), 1)

        # init starting state of the object
        if init_poses is not None:
            self.multi_pose_str = init_poses[0]
            for i in range(self.num_envs):
                for j in range(self.object_dof_num):
                    ij_dof_scale = int(init_poses[i][j]) / (self.init_pose_num - 1)
                    self.object_init_dof_pos[i, j] = ij_dof_scale

        self.object_init_dof_pos = (
                to_torch(self.object_dof_lower_limits, device=self.device) * (1 - self.object_init_dof_pos)
                + to_torch(self.object_dof_upper_limits, device=self.device) * self.object_init_dof_pos
        )

    def multi_pose_init(self, is_multi_pose, init_pose_str=None):
        self.is_multi_pose = is_multi_pose
        if self.is_multi_pose:
            object_init_dof_pos_rand = to_torch(self.object_dof_lower_limits, device=self.device).clone()

            if init_pose_str is None:
                rand_dof_array = np.random.choice(self.init_pose_num, self.object_dof_num, replace=True)
                # TODO test fully closed drawer
                # rand_dof_array = np.array([1.0] * self.object_dof_num)

                multi_pose_index = to_torch(
                    rand_dof_array, device=self.device
                )

                self.multi_pose_str = ""
                for i in range(multi_pose_index.shape[0]):
                    self.multi_pose_str += str(int(multi_pose_index[i].detach().cpu().numpy()))

            else:
                if init_pose_str == "0":
                    init_pose_str = "0" * self.object_dof_num
                assert len(init_pose_str) == self.object_dof_num
                self.multi_pose_str = init_pose_str

                multi_pose_index_np = np.array([float(dof_i) for dof_i in init_pose_str])
                multi_pose_index = to_torch(
                    multi_pose_index_np, device=self.device
                )

            object_dof_upper_limits_tensor = to_torch(self.object_dof_upper_limits, device=self.device)
            object_dof_lower_limits_tensor = to_torch(self.object_dof_lower_limits, device=self.device)

            object_dof_upper_limits_valid = torch.where(
                object_dof_upper_limits_tensor > 1000,
                torch.ones_like(object_dof_upper_limits_tensor) * math.pi,
                object_dof_upper_limits_tensor,
            )
            object_dof_lower_limits_valid = torch.where(
                object_dof_lower_limits_tensor < -1000,
                -torch.ones_like(object_dof_lower_limits_tensor) * math.pi,
                object_dof_lower_limits_tensor,
            )

            object_dof_range = object_dof_upper_limits_valid - object_dof_lower_limits_valid

            object_init_dof_pos_rand += (multi_pose_index / (self.init_pose_num - 1)) * object_dof_range

            # object_init_dof_pos_rand = torch.where(object_init_dof_pos_rand > 1000, torch.zeros_like(object_init_dof_pos_rand), object_init_dof_pos_rand)
            # object_init_dof_pos_rand = torch.where(object_init_dof_pos_rand < -1000, torch.zeros_like(object_init_dof_pos_rand), object_init_dof_pos_rand)
            self.object_init_dof_pos = object_init_dof_pos_rand.repeat((self.num_envs, 1))

        print("finish multi pose")
        return self.multi_pose_str

    def init_vision(self, is_first, is_metric=False, obs="tsdf", modes=False):
        # call viewer camera to construct point cloud for sampling
        # reconstruct tsdf from multi camera

        self.reset(torch.arange(self.num_envs, device=self.device))
        observation = self.compute_observations()
        for _ in range(2):
            self.render()
            self.step(-1)
        if is_metric:
            for i in range(self.num_envs):
                self.viewer_camera(i, is_first, is_metric)
                self.save_metric(i, obs, False, modes)
        else:
            self.viewer_camera(0, is_first)

            if not self.is_finetune:
                self.generate_graspnet_pc()
                self.generate_grasps()
        print("--------------------------Finish Initialize Vision")

    def find_object_asset_file(self, object_id):
        return os.path.join(self.dataset_path, str(object_id), "mobility_vhacd.urdf")

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = self.gym.create_sim(0, self.graphics_device, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.segmentation_id = 0
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = "./affordance/envs/assets"
        franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"

        # load franka gripper asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.armature = 0.01
        # asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        # asset_options.thickness = 0.005
        # asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        # get link index of panda hand, which we will use as end effector
        franka_link_dict = self.gym.get_asset_rigid_body_dict(franka_asset)
        self.franka_hand_index = franka_link_dict["panda_hand"]

        franka_dof_stiffness = to_torch([400, 400, 400, 400, 400, 400, 400, 1.0e6, 1.0e6], dtype=torch.float,
                                        device=self.device)
        franka_dof_damping = to_torch([80, 80, 80, 80, 80, 80, 80, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)

        # set franka dof properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []

        self.controller = "osc"
        # self.controller = "osc"
        # use position drive for all dofs
        if self.controller == "ik":
            franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
            franka_dof_props["stiffness"][:7].fill(400.0)
            franka_dof_props["damping"][:7].fill(40.0)
        else:  # osc
            franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
            franka_dof_props["stiffness"][:7].fill(0.0)
            franka_dof_props["damping"][:7].fill(0.0)
        # grippers
        franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
        franka_dof_props["stiffness"][7:].fill(800.0)
        franka_dof_props["damping"][7:].fill(40.0)


        for i in range(self.num_franka_dofs):
            '''
            franka_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props["stiffness"][i] = 1000
                franka_dof_props["damping"][i] = 100
            else:
                franka_dof_props["stiffness"][i] = 7000.0
                franka_dof_props["damping"][i] = 50.0
            '''
            self.franka_dof_lower_limits.append(franka_dof_props["lower"][i])
            self.franka_dof_upper_limits.append(franka_dof_props["upper"][i])

        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self.default_franka_dof_pos_tensor = (self.franka_dof_lower_limits + self.franka_dof_upper_limits) / 2

        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[[7, 8]] = 0.2
        # franka_dof_props["effort"][7] = 2000
        # franka_dof_props["effort"][8] = 2000

        franka_start_pose = gymapi.Transform()
        # Franka robot on the table
        franka_start_pose.p = gymapi.Vec3(-1.2, 0.0, 0.0)
        franka_start_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0)

        self.franka_start_pose = franka_start_pose

        # TODO attractor does not work on new isaacgym gpu pipeline
        '''
        # create attractor
        attractor_handles = []
        attractor_properties = gymapi.AttractorProperties()
        attractor_properties.stiffness = 5e5
        attractor_properties.damping = 5e3

        # Make attractor in all axes
        attractor_properties.axes = gymapi.AXIS_ALL
        attractor_pose = gymapi.Transform()
        attractor_pose.p = gymapi.Vec3(-1.2150, 0.0, 0.6418)
        attractor_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), math.pi / 2)
        '''


        # create object asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        # asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.armature = 0.001
        # asset_options.thickness = 0.005
        object_asset = self.gym.load_asset(
            self.sim, os.path.join(self.dataset_path, str(self.object_id)), "mobility_vhacd.urdf", asset_options
        )

        self.num_object_bodies = self.gym.get_asset_rigid_body_count(object_asset)
        self.num_object_dofs = self.gym.get_asset_dof_count(object_asset)

        # set object dof properties
        object_dof_props = self.gym.get_asset_dof_properties(object_asset)
        self.object_dof_lower_limits = []
        self.object_dof_upper_limits = []
        stiff_scale = 0.5
        if self.object_cate[0].upper() + self.object_cate[1:] == "Table":
            stiff_scale = .9 / 3

        if self.object_cate[0].upper() + self.object_cate[1:] == "Door":
            stiff_scale = .1 / 4

        if self.object_cate[0].upper() + self.object_cate[1:] == "Faucet":
            stiff_scale = .1 / 3

        if self.object_cate[0].upper() + self.object_cate[1:] == "Fridge":
            stiff_scale = 0.5 / 3

        if self.object_cate[0].upper() + self.object_cate[1:] == "Safe":
            stiff_scale = 0.3 / 3

        for i in range(self.num_object_dofs):
            object_dof_props["driveMode"][i] = gymapi.DOF_MODE_NONE
            object_dof_props["stiffness"][i] = 0.0

            object_dof_props["damping"][i] = 1e3 * stiff_scale
            object_dof_props["friction"][i] = 1e3 * stiff_scale
            object_dof_props["effort"][i] = 3e2 * stiff_scale
            object_dof_props["armature"][i] = 1e1 * stiff_scale

            # object_dof_props["damping"][i] = 9e3
            # object_dof_props["friction"][i] = 5e4
            # object_dof_props["effort"][i] = 5e3
            # object_dof_props["armature"][i] = 5e2

            self.object_dof_lower_limits.append(object_dof_props["lower"][i])
            self.object_dof_upper_limits.append(object_dof_props["upper"][i])

        # print("object_dof_lower_limits: ", self.object_dof_lower_limits)
        # print("object_dof_upper_limits: ", self.object_dof_upper_limits)
        # print("num franka bodies: ", self.num_franka_bodies)
        # print("num franka dofs: ", self.num_franka_dofs)
        # print("num object bodies: ", self.num_object_bodies)
        # print("num object dofs: ", self.num_object_dofs)

        # TODO set obejct transformation
        object_pose = gymapi.Transform()
        if self.object_cate == "Switch":
            object_pose.p = gymapi.Vec3(-0.2, 0.0, 0.4)
        elif self.object_cate == "Door":
            object_pose.p = gymapi.Vec3(0.0, 0.0, 1.2)
        elif self.object_cate == "Fridge":
            object_pose.p = gymapi.Vec3(0.2, 0.0, 0.8)
        elif self.object_cate == "Faucet":
            object_pose.p = gymapi.Vec3(-0.15, 0.0, 0.3)
        elif self.object_cate == "Window":
            object_pose.p = gymapi.Vec3(-0.1, 0.0, 0.6)
        else:
            object_pose.p = gymapi.Vec3(0.3, 0.0, 0.6)

        # add sphere to lead coordinate
        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = False
        asset_options.density = 2000
        asset_sphere_low = self.gym.create_sphere(self.sim, 0.1, asset_options)

        self.frankas = []
        self.objects = []

        self.hand_idxs = []
        self.franka_actor_idxs = []
        self.object_actor_idxs = []

        self.init_pos = []
        self.init_rot = []

        self.object_init_state = []

        self.franka_init_state = []
        self.prop_start = []
        self.envs = []

        self.object_dof_handle = []
        self.franka_dof_handle = []

        self.save_obs_camera_handles = []
        self.save_rvt_obs_camera_handles = []

        # add camera handle to save more image
        self.camera_handles = {}

        # add camera used for metric learning later
        self.metric_cameras = []

        # attractor handle list
        self.attractor_handles = []

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        num_object_bodies = self.gym.get_asset_rigid_body_count(object_asset)
        num_object_shapes = self.gym.get_asset_rigid_shape_count(object_asset)
        max_agg_bodies = num_franka_bodies + num_object_bodies
        max_agg_shapes = num_franka_shapes + num_object_shapes

        camera_properties = gymapi.CameraProperties()
        cam_width = self.image_width
        cam_height = self.image_height
        camera_properties.width = cam_width
        camera_properties.height = cam_height
        camera_properties.enable_tensors = True

        # Create helper geometry used for visualization
        # Create an wireframe axis
        self.axes_geom = gymutil.AxesGeometry(0.1)
        # Create an wireframe sphere
        sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
        sphere_pose = gymapi.Transform(r=sphere_rot)
        self.sphere_geom = gymutil.WireframeSphereGeometry(0.03, 24, 24, sphere_pose, color=(1, 0, 0))
        self.sphere_interaction_point = gymutil.WireframeSphereGeometry(0.03, 24, 24, sphere_pose,
                                                                        color=(1, 0.5, 1))  # pink pc
        self.sphere_grasp_point = gymutil.WireframeSphereGeometry(0.03, 24, 24, sphere_pose,
                                                                  color=(0, 0.5, 1))  # blue grasp pos
        self.sphere_init_point = gymutil.WireframeSphereGeometry(0.03, 24, 24, sphere_pose,
                                                                 color=(0, 1, 0.3))  # green init pos
        self.sphere_manipulate_point = gymutil.WireframeSphereGeometry(0.03, 24, 24, sphere_pose,
                                                                       color=(1, 0.5, 0))  # red manipulate pos

        self.sphere_voxel_bnd = gymutil.WireframeSphereGeometry(0.1, 24, 24, sphere_pose, color=(0.8, 1., 1))

        # Create an wireframe sphere to visualize working space
        self.workspace_sphere_r = 0.5
        self.workspace_sphere_R = 1.4
        self.sphere_work_geom_R = gymutil.WireframeSphereGeometry(self.workspace_sphere_R, 48, 48, sphere_pose,
                                                                  color=(0, 0, 1))
        self.sphere_work_geom_r = gymutil.WireframeSphereGeometry(self.workspace_sphere_r, 24, 24, sphere_pose,
                                                                  color=(0, 1, 0))



        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            franka_actor = self.gym.create_actor(
                env_ptr, franka_asset, franka_start_pose, "franka", i, 0, segmentationId=9
            )
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            franka_actor_idx = self.gym.get_actor_index(env_ptr, franka_actor, gymapi.DOMAIN_SIM)

            self.franka_actor_idxs.append(franka_actor_idx)

            hand_idx = self.gym.find_actor_rigid_body_index(env_ptr, franka_actor, "panda_hand", gymapi.DOMAIN_SIM)
            self.hand_idxs.append(hand_idx)

            self.franka_init_state.append(
                [
                    franka_start_pose.p.x,
                    franka_start_pose.p.y,
                    franka_start_pose.p.z,
                    franka_start_pose.r.x,
                    franka_start_pose.r.y,
                    franka_start_pose.r.z,
                    franka_start_pose.r.w,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )
            # Initialize the attractor
            body_dict = self.gym.get_actor_rigid_body_dict(env_ptr, franka_actor)
            hand_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_hand")
            props = self.gym.get_actor_rigid_body_states(env_ptr, franka_actor, gymapi.STATE_POS)


            # TODO attractor does not work on new isaacgym gpu pipeline
            '''
            attractor_properties.target = props["pose"][:][body_dict["panda_hand"]]
            attractor_properties.target.p.y -= 0.1
            attractor_properties.target.p.z = 0.1
            attractor_properties.rigid_handle = hand_handle

            attractor_handle = self.gym.create_rigid_body_attractor(env_ptr, attractor_properties)
            self.attractor_handles.append(attractor_handle)
            '''

            # add object
            object_pose.r = gymapi.Quat(0, 0, 0)
            object_actor = self.gym.create_actor(env_ptr, object_asset, object_pose, "object", i, 0, segmentationId=2)
            self.gym.set_actor_dof_properties(env_ptr, object_actor, object_dof_props)
            self.object_actor_env = object_actor

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            self.gym.enable_actor_dof_force_sensors(env_ptr, object_actor)

            rigid_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_actor)
            new_rigid_props = []
            for rigid_prop in rigid_props:
                rigid_prop.friction = 4e2
                new_rigid_props.append(rigid_prop)
            self.gym.set_actor_rigid_shape_properties(env_ptr, object_actor, new_rigid_props)

            # get box actor index
            object_actor_idx = self.gym.get_actor_index(env_ptr, object_actor, gymapi.DOMAIN_SIM)
            self.object_actor_idxs.append(object_actor_idx)
            # get initial hand pose
            hand_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_hand")
            hand_pose = self.gym.get_rigid_transform(env_ptr, hand_handle)
            self.init_pos.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
            self.init_rot.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

            # set box initial state
            self.object_init_state.append(
                [
                    object_pose.p.x,
                    object_pose.p.y,
                    object_pose.p.z,
                    object_pose.r.x,
                    object_pose.r.y,
                    object_pose.r.z,
                    object_pose.r.w,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )

            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)
            self.objects.append(object_actor)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # [[-1.21  0.6 ]
            #  [-0.65  065]
            #  [ 0.34  1.63]]

            h1 = self.gym.create_camera_sensor(env_ptr, camera_properties)
            camera_position = gymapi.Vec3(-2.5, 0.0, 1.5)
            camera_target = gymapi.Vec3(0.0, 0.0, 1.0)
            self.gym.set_camera_location(h1, env_ptr, camera_position, camera_target)
            self.metric_cameras.append(h1)

            # create camera actor
            if self.save_video:
                # Sensor camera properties
                cam_pos = gymapi.Vec3(-0.6, -1.0, 1.2)
                cam_target = gymapi.Vec3(-0.6, -0.0, 0.8)
                cam_props = gymapi.CameraProperties()

                self.resolution_width = 960
                self.resolution_height = 960

                cam_props.width = self.resolution_width
                cam_props.height = self.resolution_height

                self.video_camera_handle = self.gym.create_camera_sensor(env_ptr, cam_props)
                self.gym.set_camera_location(self.video_camera_handle, env_ptr, cam_pos, cam_target)

        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(
            self.num_envs, 13
        )
        self.hand_idxs = to_torch(self.hand_idxs, dtype=torch.long, device=self.device)

        self.franka_actor_idxs = to_torch(self.franka_actor_idxs, dtype=torch.long, device=self.device)
        self.object_actor_idxs = to_torch(self.object_actor_idxs, dtype=torch.long, device=self.device)

        self.hand_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_hand")
        self.lfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_leftfinger")
        self.rfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_rightfinger")

        self.franka_init_state = to_torch(self.franka_init_state, device=self.device, dtype=torch.float).view(
            self.num_envs, 13
        )

        # initial hand position and orientation tensors
        self.init_pos = torch.Tensor(self.init_pos).view(num_envs, 3).to(self.device)
        self.init_rot = torch.Tensor(self.init_rot).view(num_envs, 4).to(self.device)

        # get jacobian tensor
        # for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)

        # jacobian entries corresponding to franka hand
        self.j_eef = jacobian[:, self.franka_hand_index - 1, :, :7]

        # get mass matrix tensor
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        self.mm = gymtorch.wrap_tensor(_massmatrix)


    def get_franka_base_pos(self):
        return np.array([self.franka_start_pose.p.x, self.franka_start_pose.p.y, self.franka_start_pose.p.z])

    def update_interaction_point(self, interaction_point):
        self.interaction_points = interaction_point

    def save_hit(self, dataset_root):
        if self.is_multi_pose:
            npz_path = dataset_root + "/" + str(self.object_id) + "_" + self.multi_pose_str + "_hit.npz"
        else:
            npz_path = dataset_root + "/" + str(self.object_id) + "_hit.npz"
        npz_path = Path(npz_path)

        if npz_path.exists():
            self.pc_hit = np.load(npz_path)["pc"]
        else:
            self.pc_hit = np.zeros(self.candidate_points.shape[0])

        for i in range(len(self.valid_index)):
            if self.mask[i]:
                self.pc_hit[self.valid_index[i]] += 1
        np.savez_compressed(npz_path, pc=self.pc_hit)

    def check_metric(self, use_metric, metric_model, metric_obs):
        # use metric learning model to calculate latent z
        if not use_metric:
            return

        # TODO move gripper away
        self.move_gripper_away(torch.arange(self.num_envs, device=self.device))
        for _ in range(10):
            self.render()
            self.step(-1)

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        #
        # User code to digest tensors
        #
        metric_obs_tensor = torch.Tensor().to(self.device)
        rgb_obs_tensor = torch.Tensor().to(self.device)
        for i in range(self.num_envs):
            rgba_camera_tensor = self.gym.get_camera_image_gpu_tensor(
                self.sim, self.envs[i], self.metric_cameras[i], gymapi.IMAGE_COLOR
            )
            rgba_camera_tensor = gymtorch.wrap_tensor(rgba_camera_tensor)
            rgba_camera_tensor = rgba_camera_tensor.float().permute(2, 0, 1)
            rgba_camera_tensor /= 255

            depth_camera_tensor = self.gym.get_camera_image_gpu_tensor(
                self.sim, self.envs[i], self.metric_cameras[i], gymapi.IMAGE_DEPTH
            )
            depth_camera_tensor = gymtorch.wrap_tensor(depth_camera_tensor)
            depth_camera_tensor = depth_camera_tensor.float()
            depth_camera_tensor[depth_camera_tensor < -5] = 0
            depth_camera_tensor = -depth_camera_tensor

            # compute coord
            # TODO also add coord layer for curr_obs
            # end coord

            obs_tensor = torch.Tensor().to(self.device)
            rgb_tensor = torch.Tensor().to(self.device)
            if metric_obs == "depth":
                obs_tensor = depth_camera_tensor
                rgb_tensor = rgba_camera_tensor[:3, :, :]
                rgb_tensor = rgb_tensor.unsqueeze(0)
            elif metric_obs == "rgbd":
                obs_tensor = torch.zeros_like(rgba_camera_tensor)
                obs_tensor[:3, :, :] = rgba_camera_tensor[:3, :, :]
                obs_tensor[-1, :, :] = depth_camera_tensor
            obs_tensor = obs_tensor.unsqueeze(0)
            metric_obs_tensor = torch.cat((metric_obs_tensor, obs_tensor), 0)
            rgb_obs_tensor = torch.cat((rgb_obs_tensor, rgb_tensor), 0)
            if i == 0:
                rgb_filename = "after_rgb_env_cam%d.png" % (self.camera_id)
                # self.gym.write_camera_image_to_file(self.sim, self.envs[i], self.metric_cameras[i], gymapi.IMAGE_COLOR, rgb_filename)

        init_obs = torch.Tensor().to(self.device)
        rgbd_tensor = to_torch(self.rgbd, device=self.device)
        if metric_obs == "depth":
            init_obs = rgbd_tensor[-1, :, :]
        elif metric_obs == "rgbd":
            init_obs = rgbd_tensor

        init_obs = init_obs.unsqueeze(0)
        init_feature = metric_model.encode(init_obs).detach()

        self.curr_obs = metric_obs_tensor
        self.curr_rgb = rgb_obs_tensor

        batch_size = 5 if (self.num_envs % 5 == 0) else 1

        feature_z = torch.Tensor().to(self.device)
        for i in range(self.num_envs // batch_size):
            feature_z_batch = metric_model.encode(
                metric_obs_tensor[i * batch_size: (i + 1) * batch_size, :, :]
            ).detach()
            feature_z = torch.cat((feature_z, feature_z_batch), 0)

        # save feature_change vector to determine whether dof change happens
        self.feature_change = feature_z - init_feature

        # TODO save feature_z and init_feature for testing
        self.feature_z = feature_z
        self.init_feature = init_feature

        self.gym.end_access_image_tensors(self.sim)

    def check_hit_rate(self, eval=False, model_eval=None):
        self.gym.refresh_dof_state_tensor(self.sim)
        object_init_dof_pos = self.object_init_dof_pos
        object_dof_pos_move = object_init_dof_pos - self.object_dof_pos

        object_dof_upper_limits_tensor = to_torch(self.object_dof_upper_limits)
        object_dof_lower_limits_tensor = to_torch(self.object_dof_lower_limits, device=self.device)

        object_dof_upper_limits_valid = torch.where(
            object_dof_upper_limits_tensor > 1000,
            torch.ones_like(object_dof_upper_limits_tensor) * math.pi,
            object_dof_upper_limits_tensor,
        )
        object_dof_lower_limits_valid = torch.where(
            object_dof_lower_limits_tensor < -1000,
            -torch.ones_like(object_dof_lower_limits_tensor) * math.pi,
            object_dof_lower_limits_tensor,
        )

        object_dof_range = object_dof_upper_limits_valid - object_dof_lower_limits_valid

        self.object_dof_pos_move_scale = object_dof_pos_move / (
            object_dof_range
        )

        mask = to_torch([False], device=self.device).repeat(self.num_envs)

        # if dof move 10% then it is successful
        success_threshold = self.success_threshold
        if eval:
            success_threshold = 0.2

        for i in range(self.object_dof_num):
            mask = torch.logical_or(mask, (torch.abs(self.object_dof_pos_move_scale[:, i]) > success_threshold))

        # compute unreachable mask
        false_mask = to_torch([False], device=self.device).repeat(self.num_envs)
        true_mask = to_torch([True], device=self.device).repeat(self.num_envs)
        unreachable_penalty_mask = torch.where(self.unreachable_penalty < 0, false_mask, true_mask)
        print("successfully manipulated: ", int(mask.sum().cpu()))
        print("successfully reached: ", int(unreachable_penalty_mask.sum().cpu()))

        mask = torch.logical_and(mask, unreachable_penalty_mask)

        self.obj_dof_change = torch.zeros((self.num_envs, self.object_dof_num)).to(self.device)
        self.obj_dof_change = torch.where(
            self.object_dof_pos_move_scale > success_threshold,
            torch.ones_like(self.obj_dof_change).to(self.device),
            self.obj_dof_change,
        )
        self.obj_dof_change = torch.where(
            self.object_dof_pos_move_scale < -success_threshold,
            2 * torch.ones_like(self.obj_dof_change).to(self.device),
            self.obj_dof_change,
        )

        # print("obj dof change: ", self.obj_dof_change)

        # print("current successful num: ", int((self.obj_dof_change > 0).sum().cpu()))
        # print("mask num: ", int(mask.sum().cpu()))


        # TODO ECCV testing
        # for single env evaluation
        # ---------------------------
        # for i in range(self.num_envs):
        #     dof_str = ""
        #     for j in range(self.object_dof_num):
        #         dof_str += str(int(self.obj_dof_change[i, j].detach().cpu().numpy()))
        #     self.obj_dof_change_str.append(dof_str)
        # self.obj_dof_change_str

        # modes = {}
        # for el in self.obj_dof_change_str:
        #     if el not in modes.keys():
        #         modes[el] = 1
        #     else:
        #         modes[el] += 1

        # return modes
        # ---------------------------


        # TODO return the current results to check the success rate
        # return int((self.obj_dof_change).sum().cpu()), int(mask.sum().cpu())

        if not eval:
            success_metric = torch.norm(self.feature_change, dim=-1)
            class_predict = success_metric >= self.margin

            correct_predict = 0
            for i in range(self.num_envs):
                if class_predict[i].item() and mask[i].item():
                    correct_predict += 1
                elif not class_predict[i].item() and not mask[i].item():
                    correct_predict += 1
            # print("successfully predict: ", correct_predict / self.num_envs)

            # Use autoencoder instead of groundtruth
            self.mask = class_predict.long()

        else:
            self.mask = mask.long()

        # reset mask based on reset buffer
        self.mask = torch.where(self.reset_buf > 0, torch.zeros_like(self.mask), self.mask)

        hit_rate = self.mask.sum().cpu().numpy() / self.num_envs
        print("++++++++++++++++++++++++++++++++++++++++")

        # print("object id: ", self.object_id)
        # print("pose id: ", self.multi_pose_str)
        # print("camera id: ", self.camera_id)
        print("hit rate: ", "%.4f" % hit_rate)

        if self.object_cate is not None:
            row = self.object_cate + "," + str(self.object_id) + "," + str(self.multi_pose_str) + "," + str(
                self.camera_id) + "," + str("%.4f" % hit_rate)
        else:
            row = "unknown" + "," + str(self.object_id) + "," + str(self.multi_pose_str) + "," + str(
                self.camera_id) + "," + str("%.4f" % hit_rate)

        self.success = self.mask.long().unsqueeze(-1)

        # correct the obj_dof_change
        success_dof = self.success.repeat((1, self.object_dof_num))
        self.obj_dof_change = torch.where(success_dof > 0, self.obj_dof_change,
                                          torch.zeros_like(self.obj_dof_change).to(self.device))

        for i in range(self.num_envs):
            dof_str = ""
            for j in range(self.object_dof_num):
                dof_str += str(int(self.obj_dof_change[i, j].detach().cpu().numpy()))
            self.obj_dof_change_str.append(dof_str)

        # Check output dof change/modes
        # print(self.obj_dof_change)
        num_eval = self.obj_dof_change.shape[0]

        num_pull = 0
        for i in range(num_eval):
            if (self.obj_dof_change[i] == 2).sum().item() > 0:
                num_pull += 1
        print("pull rate: ", "%.4f" % (num_pull / num_eval))
        row += "," + str("%.4f" % (num_pull / num_eval))
        # print(self.obj_dof_change_str)
        max_mode = compute_max_mode(self.multi_pose_str)
        # compute_total_entropy(self.obj_dof_change_str)

        # check dof change (how many modes discovered
        modes = list(dict.fromkeys(self.obj_dof_change_str))
        num_mode_discover = len(modes) - 1

        print("number of modes discovered: ", num_mode_discover)
        row += "," + str(num_mode_discover)

        if num_mode_discover > max_mode:
            max_mode = num_mode_discover
        print("number of valid modes: ", max_mode)
        row += "," + str(max_mode)
        entropy = compute_entropy(self.obj_dof_change_str)
        max_entropy = compute_max_entropy(max_mode)

        if max_entropy == 0:
            max_entropy = 1
            entropy = 1

        row += "," + ("%.4f" % entropy)
        row += "," + ("%.4f" % max_entropy)

        modes_dict = {}
        for mode in modes:
            mode_num = self.obj_dof_change_str.count(mode)
            modes_dict[mode] = mode_num
        print("modes: ", modes_dict)
        row += "," + str(modes_dict)
        print("++++++++++++++++++++++++++++++++++++++++")

        if eval:
            if model_eval is None:
                csv_path = "./random_test.csv"
            else:
                csv_path = "./" + model_eval + ".csv"

            csv_path = Path(csv_path)

            # save csv file
            if not csv_path.exists():
                with csv_path.open("w") as f:
                    f.write(
                        ",".join(
                            [
                                "object_cate",
                                "object_id",
                                "pose_id",
                                "camera_id",
                                "ssr",
                                "pull_rate",
                                "num_modes",
                                "total_modes",
                                "success_entropy",
                                "max_entropy",
                                "mode"
                            ]
                        )
                    )
                    f.write("\n")

            with csv_path.open("a") as f:
                f.write(row)
                f.write("\n")

    def save_replay_buf(self, traj_data, use_metric, cluster_method):
        # TODO use KNN first, assume that we know the k cluster
        key_list = self.obj_dof_change_str
        # self.object_dof_pos_move_scale
        if cluster_method == "gmm" or cluster_method == "kmean":
            dof_change_data = self.object_dof_pos_move_scale

            max_modes_num = compute_max_mode(self.multi_pose_str)
            n_components = max_modes_num if self.num_envs > 9 else (self.num_envs // 2)

            if cluster_method == "gmm":
                self.cluster_model = GMM(n_components=n_components)
            elif cluster_method == "kmean":
                self.cluster_model = KMeans(n_clusters=n_components)

            # dim of traj_data = 19 = 4 + 1 + 3 + 3 + 3 + 3 + 1 + 1
            # traj_data = (rotation, init_grasp_dist, init_pos, interact_pos, manipulate_pos, select_points, success, dof_change)

            self.success_data = traj_data[self.success]
            # gt_label = data[:, -1]
            # gt_label = gt_label.cpu().numpy()

            dof_change_data = dof_change_data.cpu().numpy()
            # self.success_data = torch.Tensor().to(self.device)
            success_metric = torch.norm(self.feature_change, dim=-1)
            margin = 1.0

            j = 0
            for i in range(self.num_envs):
                # TODO use metric learning instead of label
                if success_metric[i] < margin:
                    # if data[i, -2] == 0:
                    dof_change_data = np.delete(dof_change_data, j, 0)
                    gt_label = np.delete(gt_label, j, 0)
                else:
                    self.success_data = torch.cat((self.success_data, traj_data[i, :].unsqueeze(0)), 0)
                    j += 1

            if self.cluster_fit_data is not None:
                self.cluster_fit_data = np.concatenate((self.cluster_fit_data, dof_change_data), axis=0)
            else:
                self.cluster_fit_data = dof_change_data.copy()

            # rng = np.random.RandomState(17)
            # dof_change_data = np.dot(dof_change_data, rng.randn(2, 2))

            # if cluster_method == 'gmm':
            #     plot_gmm(self.cluster_model, dof_change_data, gt_label)
            # elif cluster_method == 'kmean':
            #     plot_kmeans(self.cluster_model, dof_change_data)

            predict_cluster = self.cluster_model.fit(self.cluster_fit_data).predict(self.cluster_fit_data).tolist()
            key_list = predict_cluster

            # TODO empty replay buffer
            self.replay_buf = {}
            for i in range(self.cluster_fit_data.shape[0]):
                if key_list[i] not in self.replay_buf.keys():
                    self.replay_buf[key_list[i]] = torch.Tensor().to(self.device)
                save_data = self.success_data[i, :].unsqueeze(0)
                self.replay_buf[key_list[i]] = torch.cat((self.replay_buf[key_list[i]], save_data), 0)

        """
        for i in range(self.num_envs):
            if key_list[i] not in self.replay_buf.keys():
                self.replay_buf[key_list[i]] = torch.Tensor().to(self.device)
            save_data = data[i, :].unsqueeze(0)
            self.replay_buf[key_list[i]] = torch.cat((self.replay_buf[key_list[i]], save_data), 0)
        """
        if cluster_method == "dof":
            # use gt to represent cluster
            success_index = self.success.squeeze() > 0
            self.success_data = traj_data[success_index, :]

            # dim of traj_data = 19 = 4 + 1 + 3 + 3 + 3 + 3 + 1
            # traj_data = (rotation, init_grasp_dist, select_points, manipulate_pos, init_pos, interact_pos, dof_change)

            success_data_num = self.success_data.shape[0]
            self.replay_buf = {}
            for i in range(success_data_num):
                dof_change_str = str(int((self.success_data[i, -1]).cpu()))
                if dof_change_str not in self.replay_buf.keys():
                    self.replay_buf[dof_change_str] = torch.Tensor().to(self.device)
                success_data_i = self.success_data[i, :(4 + 1 + 3 + 3)]
                self.replay_buf[dof_change_str] = torch.cat(
                    (self.replay_buf[dof_change_str], success_data_i.unsqueeze(0)), 0)

        self.fit_replay_buf()

    def set_object_cate(self, object_cate):
        self.object_cate = object_cate

    def save_metric(self, index, obs="tsdf", is_rgb=False, modes=False):
        modes_str = "modes_" if modes else ""
        dataset_root = "./" + modes_str + obs + "_dataset"
        Path(dataset_root).mkdir(parents=True, exist_ok=True)
        object_obs_dir = dataset_root + "/" + str(self.object_id)
        Path(object_obs_dir).mkdir(parents=True, exist_ok=True)

        existing_obs = os.listdir(object_obs_dir)
        # self.num_envs observation + dof file + initial_obs
        modes_index = len(existing_obs) // (self.num_envs + 2)

        dof_file = str(modes_index) + "_dof.npz" if modes else "dof.npz"
        dof_npz_path_str = object_obs_dir + "/" + dof_file
        dof_npz_path = Path(dof_npz_path_str)
        dof_scale = self.pose_scale.cpu().numpy()

        if not dof_npz_path.exists():
            np.savez_compressed(dof_npz_path, dof=dof_scale)
        else:
            if index == 0:
                dof = np.load(dof_npz_path_str)["dof"]
                save_dof = np.concatenate((dof, dof_scale), axis=0)
                np.savez_compressed(dof_npz_path, dof=save_dof)

        existing_obs = os.listdir(object_obs_dir)

        npz_index = str(modes_index) + "_" + str(index) if modes else str(len(existing_obs) - 1)
        npz_path = object_obs_dir + "/" + npz_index + ".npz"
        npz_path = Path(npz_path)

        if not npz_path.exists():
            if obs == "tsdf":
                voxel_grid, color_grid = self.tsdf.get_volume()
                if is_rgb:
                    voxel_grid_channel = voxel_grid.unsqueeze(-1)
                    voxel_grid = torch.cat((voxel_grid_channel, color_grid), dim=-1)
                voxel_grid = voxel_grid.cpu().numpy()
                np.savez_compressed(npz_path, grid=voxel_grid)
            elif obs == "rgbd":
                np.savez_compressed(npz_path, grid=self.rgbd)
            elif obs == "depth":
                depth = self.rgbd[-1, :, :]
                np.savez_compressed(npz_path, grid=depth)

        if modes:
            init_obs_npz = "init_" + str(modes_index) + ".npz"
            init_obs_npz_path = object_obs_dir + "/" + init_obs_npz
            init_obs_npz_path = Path(init_obs_npz_path)

            if not init_obs_npz_path.exists():
                if obs == "tsdf":
                    voxel_grid, color_grid = self.original_tsdf.get_volume()
                    if is_rgb:
                        voxel_grid_channel = voxel_grid.unsqueeze(-1)
                        voxel_grid = torch.cat((voxel_grid_channel, color_grid), dim=-1)
                    voxel_grid = voxel_grid.cpu().numpy()
                    np.savez_compressed(init_obs_npz_path, grid=voxel_grid)
                elif obs == "rgbd":
                    np.savez_compressed(init_obs_npz_path, grid=self.original_rgbd)
                elif obs == "depth":
                    depth = self.original_rgbd[-1, :, :]
                    np.savez_compressed(init_obs_npz_path, grid=depth)

    def save_action_seq_finetune(self, dataset_root, state_hand_pose, traj_id):
        # Savc csv file if sequence interactions
        Path(dataset_root).mkdir(parents=True, exist_ok=True)

        csv_path = dataset_root + "/" + "action_seq_finetune.csv"
        csv_path = Path(csv_path)

        # save csv file
        if not csv_path.exists():
            with csv_path.open("w") as f:
                f.write(
                    ",".join(
                        ["object_cate", "object_id", "init_state", "traj_id", "dof", "lang_prompt",
                         "0_px", "0_py", "0_pz", "0_qx", "0_qy", "0_qz", "0_qw",
                         "1_px", "1_py", "1_pz", "1_qx", "1_qy", "1_qz", "1_qw",
                         "2_px", "2_py", "2_pz", "2_qx", "2_qy", "2_qz", "2_qw",
                         "3_px", "3_py", "3_pz", "3_qx", "3_qy", "3_qz", "3_qw",
                         ]
                    )
                )
                f.write("\n")

        state_hand_pose = state_hand_pose.tolist()

        with csv_path.open("a") as f:
            line = state_hand_pose
            action = ",".join([str(data) for data in line])

            row = str(self.object_cate) + "," + str(self.object_id) + "," \
                  + self.multi_pose_str + "," + traj_id + "," \
                  + self.obj_dof_change_str[0] + "," + self.lang_prompt + "," + action

            f.write(row)
            f.write("\n")

        print("save action seq data")

    def save_action_tuple_finetune(self, dataset_root, state_hand_pose, traj_id):
        # Savc csv file if sequence interactions
        Path(dataset_root).mkdir(parents=True, exist_ok=True)

        csv_path = dataset_root + "/" + "action_tuple_finetune.csv"
        csv_path = Path(csv_path)

        # save csv file
        if not csv_path.exists():
            with csv_path.open("w") as f:
                f.write(
                    ",".join(
                        ["object_cate", "object_id", "init_state", "traj_id", "dof", "lang_prompt", "state_i",
                         "px", "py", "pz", "qx", "qy", "qz", "qw", "jaw",
                         "robot_dof_0", "robot_dof_1", "robot_dof_2", "robot_dof_3", "robot_dof_4",
                         "robot_dof_5", "robot_dof_6", "robot_dof_7", "robot_dof_8"
                         ]
                    )
                )
                f.write("\n")

        total_state = 4  # 4 steps "init, reach, grasp, manipulate"
        action_dim = 7  # 3 xyz + 4 quat

        state_hand_pose = state_hand_pose.tolist()
        # only save a_1, a_2, a_3

        with csv_path.open("a") as f:
            for state_i in range(total_state):
                line = state_hand_pose[state_i * action_dim: (state_i + 1) * action_dim]
                action = ",".join([str(data) for data in line])

                gripper_open_close = "1" if state_i > 1 else "0"
                pre_franka_proprio = self.franka_proprio[state_i].cpu().numpy().tolist()
                pre_franka_proprio = ",".join([str(data) for data in pre_franka_proprio])

                row = str(self.object_cate) + "," + str(self.object_id) + "," \
                      + self.multi_pose_str + "," + traj_id + "," \
                      + self.obj_dof_change_str[0] + "," + self.lang_prompt + "," \
                      + str(state_i) + "," + action + "," + str(gripper_open_close) + "," \
                      + pre_franka_proprio

                f.write(row)
                f.write("\n")

        print("save action tuple data")

    def save_action_data(self, dataset_root, is_clustering=True, cluster_method="dof"):
        # Save cvs file of different interactions
        # Save tsdf file for convOnet
        # Save point cloud for pc sampling
        # Save depth for metric learning and encoding

        Path(dataset_root).mkdir(parents=True, exist_ok=True)

        # if len(self.valid_index) < self.num_envs:
        #     self.valid_index = [0 for i in range(self.num_envs)]

        # index_tensor = to_torch(self.valid_index).unsqueeze(-1)

        # TODO check whether our clusters have meaningful representation
        dof_change = torch.zeros_like(self.success)
        for i in range(len(self.obj_dof_change_str)):
            dof_change[i] = int(self.obj_dof_change_str[i])

        # dim of traj_data = 18 = 4 + 1 + 3 + 3 + 3 + 3 + 1
        # traj_data = (rotation, init_grasp_dist, select_points, manipulate_pos, init_pos, interact_pos, dof_change)
        traj_data = torch.cat((self.rotation, self.init_grasp_dist, self.select_points, self.manipulate_pos,
                               self.init_pos, self.interact_pos, dof_change), 1)

        if is_clustering:
            self.save_replay_buf(traj_data, False, cluster_method)
        else:
            if len(self.replay_buf) == 0:
                self.replay_buf[0] = torch.Tensor().to(self.device)
            self.replay_buf[0] = torch.cat((self.replay_buf[0], traj_data), 0)

        traj_data_np = traj_data.cpu().numpy()

        # csv_path = dataset_root + "/" + str(self.object_id) + "_inter.csv"
        csv_path = dataset_root + "/" + "action_seq.csv"
        csv_path = Path(csv_path)

        # save csv file
        if not csv_path.exists():
            with csv_path.open("w") as f:
                f.write(
                    ",".join(
                        ["object_cate", "object_id", "init_state", "traj_id",
                         "qx", "qy", "qz", "qw", "d_init",
                         "point_x", "point_y", "point_z",
                         "manipulate_x", "manipulate_y", "manipulate_z",
                         "init_x", "init_y", "init_z",
                         "interact_x", "interact_y", "interact_z",
                         "dof"]
                    )
                )
                f.write("\n")

        with csv_path.open("a") as f:
            for i in range(traj_data_np.shape[0]):
                line = traj_data_np[i].tolist()
                # TODO CVAE testing, we want positive data only
                if line[-1] > 0:
                    row = ",".join([str(data) for data in line])

                    # TODO add trajectory id for image tracing
                    # TODO need to double-check no duplicate traj_id
                    traj_id = random_string(8)
                    row = str(self.object_cate) + "," + str(
                        self.object_id) + "," + self.multi_pose_str + "," + traj_id + "," + row
                    # row += "," + self.obj_dof_change_str[i]
                    f.write(row)
                    f.write("\n")

        return  (traj_data_np[:, -1] > 0).sum()

        # save tsdf file
        if self.is_multi_pose:
            npz_path = dataset_root + "/" + str(self.object_id) + "_" + self.multi_pose_str + ".npz"
        else:
            npz_path = dataset_root + "/" + str(self.object_id) + ".npz"
        npz_path = Path(npz_path)

        if not npz_path.exists():
            voxel_grid, color_grid = self.tsdf.get_volume()
            if is_rgb:
                voxel_grid_channel = voxel_grid.unsqueeze(-1)
                voxel_grid = torch.cat((voxel_grid_channel, color_grid), dim=-1)
            voxel_grid = voxel_grid.cpu().numpy()
            np.savez_compressed(npz_path, grid=voxel_grid)

        # save point cloud

        if self.is_multi_pose:
            pc_npz_path = dataset_root + "/" + str(self.object_id) + "_" + self.multi_pose_str + "_pc.npz"
        else:
            pc_npz_path = dataset_root + "/" + str(self.object_id) + "_pc.npz"
        pc_npz_path = Path(pc_npz_path)

        if not pc_npz_path.exists():
            np.savez_compressed(pc_npz_path, grid=self.candidate_points)

        # save depth image
        if self.is_multi_pose:
            pc_npz_path = dataset_root + "/" + str(self.object_id) + "_" + self.multi_pose_str + "_depth.npz"
        else:
            pc_npz_path = dataset_root + "/" + str(self.object_id) + "_depth.npz"
        pc_npz_path = Path(pc_npz_path)

        if not pc_npz_path.exists():
            depth_img = self.rgbd[-1, :, :]
            # TODO concatenate
            coord = True
            if coord == True:
                depth_img = depth_img[None, :, :]
                coord = np.transpose(self.coord, (2, 0, 1))
                depth_img = np.concatenate((depth_img, coord), axis=0)
            np.savez_compressed(pc_npz_path, grid=depth_img)

    def compute_grasp_pos(self, rot, pos):
        # compute the grasping position given rotation and interaction point

        if isinstance(rot, np.ndarray):
            rot_tensor = torch.tensor(rot).to(self.device)
            # rot = tgm.angle_axis_to_quaternion(rot_tensor)
            rot = pytorch3d.transforms.matrix_to_quaternion(
                pytorch3d.transforms.euler_angles_to_matrix(rot_tensor, "XYZ"))

        z_axis = to_torch([0.0, 0.0, -1.0], device=self.device).repeat((self.num_envs, 1))
        interact_pos = pos + quat_apply(rot.float(), z_axis * self.eef_hand_dist)

        return interact_pos

    def compute_init_pos(self, rot, pos):
        if isinstance(rot, np.ndarray):
            rot_tensor = torch.tensor(rot).to(self.device)
            # rot = tgm.angle_axis_to_quaternion(rot_tensor)
            rot = pytorch3d.transforms.matrix_to_quaternion(
                pytorch3d.transforms.euler_angles_to_matrix(rot_tensor, "XYZ"))

        z_axis = to_torch([0.0, 0.0, -1.0], device=self.device).repeat((self.num_envs, 1))
        interact_pos = pos + quat_apply(rot.float(), z_axis * (self.init_dist + self.eef_hand_dist))

        return interact_pos

    def set_action(self):
        self.force = torch.Tensor().to(self.device)
        self.norm_force = torch.Tensor().to(self.device)

        valid_action_num = 0
        sample_scale = 1
        self.valid_init_state = None

        interact_points = torch.Tensor().to(self.device)
        self.valid_index = []

        candidate_rotate = to_torch(R.random(self.num_envs * sample_scale).as_quat(), device=self.device)

        # randomly sample point to interact
        index = np.random.choice(self.pointcloud.shape[0], self.num_envs * sample_scale, replace=True)
        index = np.random.choice(self.candidate_points.shape[0], self.num_envs * sample_scale, replace=True)

        selected_pc_index = to_torch(index, device=self.device)

        selected_point = to_torch(self.pointcloud[index], device=self.device)
        selected_point = to_torch(self.candidate_points[index], device=self.device)
        # selected_point = to_torch([-1.2, 0, 1.18], device=self.device).repeat((self.num_envs * sample_scale, 1))
        selected_point = to_torch([-0.6098, -0.1710, 1.0607], device=self.device).repeat(
            (self.num_envs * sample_scale, 1))
        qx = gymapi.Quat.from_euler_zyx(0.5 * math.pi + random.uniform(0, 1) * 0.5,
                                        0.5 * math.pi + random.uniform(0, 1) * 0.5,
                                        0.5 * math.pi + random.uniform(0, 1) * 0.5)
        candidate_rotate = to_torch([qx.x, qx.y, qx.z, qx.w], device=self.device).repeat(
            (self.num_envs * sample_scale, 1))

        z_axis = to_torch([0, 0, -1], device=self.device).repeat((self.num_envs * sample_scale, 1))
        y_axis = to_torch([0, 1, 0], device=self.device).repeat((self.num_envs * sample_scale, 1))

        interact_pos = selected_point + quat_apply(candidate_rotate, z_axis * self.eef_hand_dist)
        init_pos = selected_point + quat_apply(candidate_rotate, z_axis * (self.init_dist + self.eef_hand_dist))

        valid_init_rotate = candidate_rotate
        valid_init_pos = init_pos
        valid_interact_pos = interact_pos
        valid_state = torch.cat((valid_init_pos, valid_init_rotate), 1)
        self.valid_init_state = valid_state
        self.interact_pos = valid_interact_pos

        valid_selected_pc_index = selected_pc_index
        valid_selected_pc_index_list = valid_selected_pc_index.detach().cpu().numpy().tolist()
        self.valid_index.extend(valid_selected_pc_index_list)
        self.select_points = interact_points
        self.valid_init_state = self.valid_init_state
        self.interact_pos = self.interact_pos
        self.valid_index = self.valid_index
        self.valid_index = [int(x) for x in self.valid_index]

        init_vel = torch.zeros((self.num_envs, 6), device=self.device)
        self.valid_init_state = torch.cat((self.valid_init_state, init_vel), 1)

        # sample random force vector
        # force_sample = to_torch(np.random.rand(num_sample, 3), device=self.device) - 0.5
        self.norm_force = torch.cat((self.norm_force, to_torch(np.random.rand(1, 3), device=self.device)))
        force_norm = torch.norm(self.norm_force - 0.5, dim=-1).unsqueeze(-1)

        self.force = torch.cat((self.force, ((self.norm_force - 0.5) / force_norm) * 0.5 * self.init_dist))

        # TODO what if we only has push action
        self.norm_force = to_torch([0.5, 0.5, 1.0], device=self.device).repeat(1, 1)
        self.force = to_torch([2, 0, 0], device=self.device).repeat((1, 1)) * self.init_dist
        self.pc_hit = np.zeros(self.candidate_points.shape[0])

    def sample_action(self, is_clustering, cluster_method="dof", iter=0):
        # Remember to clear some tensor to avoid duplicate sampling
        self.force = torch.Tensor().to(self.device)
        self.norm_force = torch.Tensor().to(self.device)
        if self.grasps is None:
            self.epsilon_heuristic_grasp = 1.0

        if self.heuristic_grasp_pc is None:
            self.epsilon_grasp = self.epsilon_grasp * (1 - self.epsilon_heuristic_grasp)
            self.epsilon_heuristic_grasp = 0.0

        if self.grasps is None and self.heuristic_grasp_pc is None:
            self.epsilon_grasp = 0.0

        if is_clustering:
            # init cluster model if None
            if self.cluster_model is None:
                # we use fixed number of cluster
                if cluster_method == "gmm":
                    self.cluster_model = GMM(n_components=9)
                elif cluster_method == "kmean":
                    self.cluster_model = KMeans(n_clusters=9)

            # if cluster model exist, remove the cluster with fewer data
            if cluster_method != "gmm" and cluster_method != "kmean":
                threshold = 5
                remove = [k for k in self.replay_buf if self.replay_buf[k].shape[0] < threshold]
                for k in remove:
                    del self.replay_buf[k]

            # if data already save in the replay buffer
            if len(self.replay_buf):
                # epsilon greedy sampling
                # TODO do cluster sampling if we have enough data
                if iter > self.start_fit:
                    print("--------------start adaptive fitting---------------")
                    epsilon = self.epsilon
                else:
                    epsilon = 1.0

            else:
                epsilon = 1.0

            random_sample_num = int(self.num_envs * epsilon)
            uniform_action_num = int(random_sample_num * (1 - self.epsilon_grasp))
            grasp_action_num = random_sample_num - uniform_action_num
            grasp_heuristic_action_num = int(grasp_action_num * self.epsilon_heuristic_grasp)
            graspnet_action_num = grasp_action_num - grasp_heuristic_action_num

            self.uniform_action(uniform_action_num)
            self.grasp_action(graspnet_action_num)
            self.heuristic_handle_grasp_action(grasp_heuristic_action_num)
            self.sample_cluster(self.num_envs - random_sample_num)
        else:
            uniform_action_num = int(self.num_envs * (1 - self.epsilon_grasp))
            grasp_action_num = self.num_envs - uniform_action_num
            grasp_heuristic_action_num = int(grasp_action_num * self.epsilon_heuristic_grasp)
            graspnet_action_num = grasp_action_num - grasp_heuristic_action_num

            self.uniform_action(uniform_action_num)
            self.grasp_action(graspnet_action_num)
            self.heuristic_handle_grasp_action(grasp_heuristic_action_num)

    def report_replay_buf(self):
        # TODO save replay buffer to check the availability
        with open("saved_dictionary.pkl", "wb") as f:
            pickle.dump(self.replay_buf, f)

        modes = [0, 1, 10, 2, 20, 11]
        modes_num = [0, 0, 0, 0, 0, 0]

        for key in self.replay_buf.keys():
            cluster_num = self.replay_buf[key].shape[0]
            print("Replay cluster {} has {} data".format(key, cluster_num))
            for i in range(len(modes)):
                mode = modes[i]
                mode_num = (self.replay_buf[key][:, -1] == mode).sum()
                print("Within cluster {}, it has {:4.2f}% modes {}".format(key, (mode_num / cluster_num) * 100, mode))
                modes_num[i] += mode_num

        list_sum = sum(modes_num)
        for i in range(len(modes)):
            print("Modes {} contain {:4.2f}% samples".format(modes[i], 100 * modes_num[i] / list_sum))
        cluster_num = self.cluster_model.n_components
        fig, axs = plt.subplots(3, cluster_num // 3)
        index = 0
        labels = ["nothing", "push top", "push bottom", "pull top", "pull bottom", "push both"]
        colors = ["yellow", "blue", "red", "green", "black", "magenta"]
        for key in self.replay_buf.keys():
            cluster_num = self.replay_buf[key].shape[0]
            pie_chart_list = []
            for i in range(len(modes)):
                mode = modes[i]
                mode_num = (self.replay_buf[key][:, -1] == mode).sum()
                pie_chart_list.append(int(mode_num.cpu().numpy()))
            axs[index % 3, index // 3].pie(pie_chart_list, colors=colors)
            axs[index % 3, index // 3].set_title("cluster size {}".format(cluster_num))
            index += 1
        print("Currently {} data is clustered".format(list_sum))
        plt.legend(labels)
        plt.show()

        # TODO check successful data
        # data = torch.cat((rotation, self.select_points, self.norm_force, index_tensor, self.success, dof_change), 1)

    def fit_replay_buf(self):
        # create GMM for each cluster to perform sampling
        self.replay_buf_models = {}
        for key in self.replay_buf.keys():
            replay_buf_data = self.replay_buf[key]
            replay_buf_data_fit = replay_buf_data.clone()[:, :]
            n_cluster = 3 if replay_buf_data_fit.shape[0] > 3 else replay_buf_data_fit.shape[0]

            data_dim = replay_buf_data_fit.shape[1]
            self.replay_buf_models[key] = GMM(n_components=n_cluster)
            if replay_buf_data_fit.shape[0] > 1:
                self.replay_buf_models[key].fit(replay_buf_data_fit.cpu().numpy())
            else:
                replay_buf_data_fit_duplicate = replay_buf_data_fit.repeat((5, 1))
                replay_buf_data_fit_duplicate = replay_buf_data_fit_duplicate + (0.1 ** 0.5) * torch.randn(5,
                                                                                                           data_dim).to(
                    self.device
                )
                self.replay_buf_models[key].fit(replay_buf_data_fit_duplicate.cpu().numpy())

    def sample_cluster(self, num_sample):
        if num_sample == 0:
            return
        print(self.replay_buf.keys())
        # "qx", "qy", "qz", "qw", "x", "y", "z", "fx", "fy", "fz", "id"
        sampling = "model_sample"
        if sampling == "prob_sampling":
            cluster_size_inv = [1.0 / self.replay_buf[key].shape[0] for key in self.replay_buf.keys()]
            cluster_prob = [cluster_size_inv[i] / sum(cluster_size_inv) for i in range(len(cluster_size_inv))]
            sample_cluster = np.random.choice(list(self.replay_buf.keys()), num_sample, replace=True, p=cluster_prob)
            counter = collections.Counter(sample_cluster)
            for key in counter.keys():
                self.add_noise_cluster_sample(key, counter[key])
        elif sampling == "smaller":
            key_list = list(self.replay_buf.keys())
            cluster_num = [self.replay_buf[key].shape[0] for key in key_list]
            max_index = cluster_num.index(max(cluster_num))

            del key_list[max_index]
            del cluster_num[max_index]

            """
            max_index = cluster_num.index(max(cluster_num))
            del key_list[max_index]
            del cluster_num[max_index]
            """
            sample_num = [57, 57, 57, 57, 57, 57, 58]
            for i in range(7):
                self.add_noise_cluster_sample(key_list[i], sample_num[i])
        elif sampling == "uniform":
            key_list = list(self.replay_buf.keys())
            key_num = len(key_list)
            sample_num_cluster = []
            cluster_each = num_sample // key_num
            left = num_sample % key_num
            for i in range(key_num):
                if left > 0:
                    sample_each_cluster = cluster_each + 1
                    left -= 1
                else:
                    sample_each_cluster = cluster_each
                sample_num_cluster.append(sample_each_cluster)

            for i in range(key_num):
                self.add_noise_cluster_sample(key_list[i], sample_num_cluster[i])

        elif sampling == "model_sample" and self.replay_buf_models is not None:
            key_list = list(self.replay_buf.keys())
            key_num = len(key_list)
            sample_num_cluster = []
            cluster_each = num_sample // key_num
            left = num_sample % key_num
            for i in range(key_num):
                # make each cluster sampling equal
                if left > 0:
                    sample_each_cluster = cluster_each + 1
                    left -= 1
                else:
                    sample_each_cluster = cluster_each
                sample_num_cluster.append(sample_each_cluster)

            for i in range(key_num):
                self.sample_replay_buf(key_list[i], sample_num_cluster[i])

    def sample_replay_buf(self, key, num_sample):
        origin_sample_num = self.valid_init_state.shape[0]
        sample_model = self.replay_buf_models[key]
        if sample_model.weights_[0] > 1:
            sample_model.weights_[0] = 1

        # rotation, init_grasp_dist, select_points, manipulate_pos

        valid_action_num = 0
        interact_points = torch.Tensor().to(self.device)
        valid_init_state_pos = torch.Tensor().to(self.device)
        valid_force = torch.Tensor().to(self.device)
        sample_scale = 5

        sample_iter = 0

        while valid_action_num <= num_sample:
            sample_iter += 1
            if sample_iter > 30:
                exit()

            sample_data_x, sample_data_y = sample_model.sample(num_sample * sample_scale)
            sample_rotation = to_torch(sample_data_x[:, :4], device=self.device)
            sample_dist = to_torch(sample_data_x[:, 4], device=self.device)
            sample_point = to_torch(sample_data_x[:, 5:8], device=self.device)
            sample_manipulate_pos = to_torch(sample_data_x[:, 8:], device=self.device)

            candidate_points = to_torch(self.candidate_points, device=self.device)
            sample_candidate_dist = torch.cdist(candidate_points, sample_point)
            min_dist_index = torch.argmin(sample_candidate_dist, dim=0)
            selected_point = candidate_points[min_dist_index]

            # normalize the rotation quaternion
            candidate_rotate = torch.nn.functional.normalize(sample_rotation, dim=1)

            z_axis = to_torch([0, 0, -1], device=self.device).repeat((num_sample * sample_scale, 1))
            y_axis = to_torch([0, 1, 0], device=self.device).repeat((num_sample * sample_scale, 1))

            # interaction pos for gripper to grasp the objects
            interact_pos = selected_point + quat_apply(candidate_rotate, z_axis * self.eef_hand_dist)

            # moving distance for reaching and manipulating
            dist_min = self.init_dist * 0.4
            dist_max = self.init_dist * 1.0

            dist_min = self.init_dist * 0.5
            dist_max = self.init_dist * 0.5

            init_grasp_dist = torch.clamp(sample_dist, min=dist_min, max=dist_max).unsqueeze(1)
            dist = init_grasp_dist.repeat((1, 3))
            init_pos = interact_pos + quat_apply(candidate_rotate, z_axis * dist)

            manipulate_pos = sample_manipulate_pos

            # rejection sampling to filtering action
            mask = self.rejection_sampling_mask(candidate_rotate, init_pos, interact_pos, manipulate_pos, is_grasp=True)

            valid_rotate = candidate_rotate[mask]
            valid_init_pos = init_pos[mask]
            valid_interact_pos = interact_pos[mask]
            valid_manipulate_pos = manipulate_pos[mask]
            valid_state = torch.cat((valid_init_pos, valid_rotate), 1)
            valid_init_dist = init_grasp_dist[mask]
            valid_select_points = selected_point[mask]

            valid_num = int(mask.sum())
            init_vel = torch.zeros((valid_num, 6), device=self.device)
            valid_state = torch.cat((valid_state, init_vel), 1)

            self.valid_init_state = torch.cat((self.valid_init_state, valid_state), 0)
            self.interact_pos = torch.cat((self.interact_pos, valid_interact_pos), 0)
            self.rotation = torch.cat((self.rotation, valid_rotate), 0)
            self.init_pos = torch.cat((self.init_pos, valid_init_pos), 0)
            self.manipulate_pos = torch.cat((self.manipulate_pos, valid_manipulate_pos), 0)
            self.init_grasp_dist = torch.cat((self.init_grasp_dist, valid_init_dist), 0)
            self.select_points = torch.cat((self.select_points, valid_select_points), 0)

            interact_points = torch.cat((interact_points, selected_point[mask]))
            valid_action_num += mask.sum().cpu().numpy()

            # add valid index for heatmap visualization
            valid_selected_pc_index = min_dist_index[mask]
            valid_selected_pc_index_list = valid_selected_pc_index.detach().cpu().numpy().tolist()
            self.valid_index.extend(valid_selected_pc_index_list)

        data_collected_num = origin_sample_num + num_sample

        # TODO save the valid interaction
        self.select_points = self.select_points[:data_collected_num, :]
        self.valid_init_state = self.valid_init_state[:data_collected_num, :]
        self.interact_pos = self.interact_pos[:data_collected_num, :]
        self.valid_index = self.valid_index[:data_collected_num]
        self.rotation = self.rotation[:data_collected_num, :]
        self.init_pos = self.init_pos[:data_collected_num, :]
        self.manipulate_pos = self.manipulate_pos[:data_collected_num, :]
        self.init_grasp_dist = self.init_grasp_dist[:data_collected_num, :]
        self.valid_index = [int(x) for x in self.valid_index]

        self.pc_hit = np.zeros(self.candidate_points.shape[0])

    def add_noise_cluster_sample(self, key, num_sample):
        sample_candidate = self.replay_buf[key]
        # rotation, self.select_points, self.norm_force, index_tensor
        point_index_bound = self.candidate_points.shape[0] - 1
        valid_action_num = 0
        original_valid_len = len(self.valid_index)
        interact_points = torch.Tensor().to(self.device)
        valid_init_state_pos = torch.Tensor().to(self.device)
        valid_force = torch.Tensor().to(self.device)

        while valid_action_num <= num_sample:
            print("within while")
            index = np.random.choice(sample_candidate.shape[0], self.num_envs, replace=True)
            sample_candidate_sample = sample_candidate[index]
            quaternion = sample_candidate_sample[:, 0:4]
            force = sample_candidate_sample[:, 7:10]
            pc_id = sample_candidate_sample[:, -1]

            pc_id_noise = torch.randint(-10, 10, (self.num_envs,)).to(self.device)
            force_noise = torch.randn(self.num_envs, 3).to(self.device) * 0.1
            euler_noise = torch.randn(self.num_envs, 3).to(self.device) * 0.1
            quaternion_noise = quat_from_euler_xyz(euler_noise[:, 0], euler_noise[:, 1], euler_noise[:, 2]).to(
                self.device
            )

            pc_id += pc_id_noise
            pc_id = tensor_clamp(pc_id, torch.zeros_like(pc_id), torch.ones_like(pc_id) * point_index_bound)
            selected_pc_index = pc_id
            pc_id = pc_id.cpu().numpy().tolist()
            pc_id = [int(x) for x in pc_id]
            force += force_noise
            force = tensor_clamp(force, torch.zeros_like(force), torch.ones_like(force))
            candidate_rotate = quat_mul(quaternion, quaternion_noise)
            selected_point = to_torch(self.candidate_points[pc_id], device=self.device)

            z_axis = to_torch([0, 0, -1], device=self.device).repeat((self.num_envs, 1))
            y_axis = to_torch([0, 1, 0], device=self.device).repeat((self.num_envs, 1))

            self.interact_pos = selected_point + quat_apply(candidate_rotate, z_axis * self.eef_hand_dist)
            init_pos = selected_point + quat_apply(candidate_rotate, z_axis * (self.init_dist + self.eef_hand_dist))
            inter_pos = selected_point + quat_apply(
                candidate_rotate, z_axis * (self.init_dist * 0.5 + self.eef_hand_dist)
            )

            z_bias = quat_apply(candidate_rotate, z_axis * self.eef_hand_dist * 0.5)
            init_pos_y = (
                    selected_point + quat_apply(candidate_rotate,
                                                y_axis * (self.init_dist + self.eef_hand_dist)) + z_bias
            )
            init_pos_y_ = (
                    selected_point - quat_apply(candidate_rotate,
                                                y_axis * (self.init_dist + self.eef_hand_dist)) + z_bias
            )

            constrain_point = torch.stack([init_pos, inter_pos, init_pos_y, init_pos_y_], dim=2)
            constrain_point = torch.stack([init_pos, init_pos_y, init_pos_y_], dim=2)
            constrain_point = torch.stack([init_pos], dim=2)

            mask = to_torch([False], device=self.device).repeat(self.num_envs)

            for j in range(constrain_point.shape[-1]):
                gripper_point_inside_bound = to_torch([True], device=self.device).repeat(self.num_envs)
                for i in range(3):
                    gripper_point_inside_bound = torch.logical_and(
                        gripper_point_inside_bound,
                        (constrain_point[:, i, j] > self.action_bounding_box[i, 0] - 0.2 * self.init_dist),
                    )
                    gripper_point_inside_bound = torch.logical_and(
                        gripper_point_inside_bound,
                        (constrain_point[:, i, j] < self.action_bounding_box[i, 1] + 0.2 * self.init_dist),
                    )
                mask = torch.logical_or(gripper_point_inside_bound, mask)

            mask = ~mask

            for j in range(constrain_point.shape[-1]):
                mask = torch.logical_and(mask, (constrain_point[:, 2, j] > self.action_bounding_box[2, 0] + 0.0))

            # mask = to_torch([True], device=self.device).repeat(self.num_envs * sample_scale)
            mask = torch.gt(mask, 0)

            valid_init_rotate = candidate_rotate[mask]
            valid_init_pos = init_pos[mask]
            valid_state = torch.cat((valid_init_pos, valid_init_rotate), 1)
            valid_init_state_pos = torch.cat((valid_init_state_pos, valid_state), 0)
            valid_force = torch.cat((valid_force, force), 0)

            interact_points = torch.cat((interact_points, selected_point[mask]))
            valid_action_num += mask.sum().cpu().numpy()

            valid_selected_pc_index = selected_pc_index[mask]
            valid_selected_pc_index_list = valid_selected_pc_index.detach().cpu().numpy().tolist()
            self.valid_index.extend(valid_selected_pc_index_list)

        valid_interact_points = interact_points[:(num_sample), :]
        self.select_points = torch.cat((self.select_points, valid_interact_points))
        valid_init_state_pos = valid_init_state_pos[:num_sample, :]

        init_vel = torch.zeros((num_sample, 6), device=self.device)
        _valid_init_state = torch.cat((valid_init_state_pos, init_vel), 1)
        self.valid_init_state = torch.cat((self.valid_init_state, _valid_init_state), 0)

        self.valid_index = self.valid_index[: (num_sample + original_valid_len)]
        self.valid_index = [int(x) for x in self.valid_index]

        # sample random force vector
        # force_sample = to_torch(np.random.rand(num_sample, 3), device=self.device) - 0.5
        norm_force = valid_force[:num_sample, :]
        self.norm_force = torch.cat((self.norm_force, norm_force), 0)
        force_norm_devide = torch.norm(norm_force - 0.5, dim=-1).unsqueeze(-1)

        force_exe = ((norm_force - 0.5) / force_norm_devide) * 0.5 * self.init_dist
        force_exe[:, :2] += self.init_dist

        self.force = torch.cat((self.force, force_exe), 0)

    def sample_move_pos(self, dist_min, dist_max, num_sample):
        # Func samples random position given range of (dist_min, dist_max)

        # sample distance [0,1)
        dist_norm = torch.rand(num_sample).to(self.device)

        dist = dist_norm * (dist_max - dist_min) + dist_min
        direction_rand = torch.rand(num_sample, 3).to(self.device)
        direction_rand -= 0.5

        direction_rand_norm = torch.norm(direction_rand, dim=1).unsqueeze(-1)
        direction_rand_norm = direction_rand_norm.repeat((1, 3))

        direction_norm = direction_rand / direction_rand_norm

        sample_pos = direction_norm * dist.unsqueeze(-1)
        return sample_pos

    def compute_far_from_pc_mask(self, points, constrain_dist):
        # return mask of the same shape as points
        # points are defined as [num_sample, num_points_type, dim=3]
        num_sample = points.shape[0]
        num_point_types = points.shape[1]
        num_points = num_sample * num_point_types
        points_batch = points.view((-1, 3))

        pc_sparse_tensor = to_torch(self.pointcloud_sparse, device=self.device)
        num_pc_sparse = pc_sparse_tensor.shape[0]

        pc_sparse_repeat = pc_sparse_tensor[None, :, :].repeat((num_points, 1, 1))
        points_batch_repeat = points_batch[:, None, :].repeat((1, num_pc_sparse, 1))

        points_to_pc = points_batch_repeat - pc_sparse_repeat
        points_to_pc_dist = torch.norm(points_to_pc, dim=-1)

        min_dist, index = torch.min(points_to_pc_dist, dim=1)

        # needs to be far away above the threshold
        mask = min_dist > constrain_dist
        mask = mask.view((num_sample, num_point_types))

        mask_batch = self.compute_batch_mask(mask)
        return mask_batch

    def compute_batch_mask(self, mask):
        # mask with shape [num_sample, num_points_type]
        # logical_and all types of point
        # output mask shape [num_sample]
        num_sample = mask.shape[0]
        num_types = mask.shape[1]

        mask_batch = mask[:, 0]
        for i in range(1, num_types):
            mask_batch = torch.logical_and(mask_batch, mask[:, i])
        return mask_batch

    def compute_above_ground_mask(self, points):
        num_sample = points.shape[0]
        num_point_types = points.shape[1]
        num_points = num_sample * num_point_types
        points_batch = points.view((-1, 3))

        points_above_ground = points_batch[:, 2] > 0
        points_above_ground = points_above_ground.view(num_sample, num_point_types)
        mask_batch = self.compute_batch_mask(points_above_ground)
        return mask_batch

    def compute_inside_workspace_mask(self, points):
        # return mask of the same shape as points
        # points are defined as [num_sample, num_points_type, dim=3]
        num_sample = points.shape[0]
        num_point_types = points.shape[1]
        num_points = num_sample * num_point_types
        points_batch = points.view((-1, 3))

        franka_base_tensor = torch.tensor(
            [self.franka_start_pose.p.x, self.franka_start_pose.p.y, self.franka_start_pose.p.z]).to(self.device)
        points_to_base_dist = torch.norm(points_batch - franka_base_tensor, dim=-1)

        points_not_too_close_mask = points_to_base_dist > self.workspace_sphere_r
        points_not_too_far_mask = points_to_base_dist < self.workspace_sphere_R

        points_not_too_close_mask = points_not_too_close_mask.view(num_sample, num_point_types)
        points_not_too_far_mask = points_not_too_far_mask.view(num_sample, num_point_types)

        # print("close mask num: ", points_not_too_close_mask.sum())
        # print("far mask num: ", points_not_too_far_mask.sum())

        mask = torch.cat((points_not_too_far_mask, points_not_too_close_mask), dim=-1)
        mask_batch = self.compute_batch_mask(mask)

        return mask_batch

    def compute_inside_voxel_bnd_mask(self, points):
        # return mask of the same shape as points
        # points are defined as [num_sample, num_points_type, dim=3]
        num_sample = points.shape[0]
        num_point_types = points.shape[1]
        num_points = num_sample * num_point_types
        points_batch = points.view((-1, 3))

        # self.tsdf_vol_bnds
        mask = None
        for i in range(3):
            points_r_mask = (points_batch[:, i] < self.tsdf_vol_bnds[i, 1]).view(num_sample, num_point_types)
            points_l_mask = (points_batch[:, i] > self.tsdf_vol_bnds[i, 0]).view(num_sample, num_point_types)

            mask_in = torch.cat((points_r_mask, points_l_mask), dim=-1)
            if mask is None:
                mask = mask_in
            else:
                mask = torch.cat((mask, mask_in), dim=-1)
        mask_batch = self.compute_batch_mask(mask)

        return mask_batch

    def rejection_sampling_mask(self, candidate_rotate, init_pos, interact_pos, manipulate_pos, is_grasp=False):
        # print("----------------rejection sampling----------------")
        num_sample = candidate_rotate.shape[0]
        z_axis = to_torch([0, 0, -1], device=self.device).repeat((num_sample, 1))
        y_axis = to_torch([0, 0.7, 0], device=self.device).repeat((num_sample, 1))

        x_axis = to_torch([-1, 0, 0], device=self.device).repeat((num_sample, 1))

        constrain_dist_max = self.init_dist * 0.4
        constrain_dist_min = self.eef_hand_dist * 0.4
        constrain_dist = constrain_dist_max - constrain_dist_min

        # use rejection sampling to compute compute the mask
        z_bias = quat_apply(candidate_rotate, z_axis * self.eef_hand_dist * 0.5)
        init_pos_y = (
                init_pos + quat_apply(candidate_rotate, y_axis * (self.eef_hand_dist)) + z_bias
        )
        init_pos_y_ = (
                init_pos - quat_apply(candidate_rotate, y_axis * (self.eef_hand_dist)) + z_bias
        )

        interact_pos_y = (
                interact_pos + quat_apply(candidate_rotate, y_axis * (self.eef_hand_dist)) + z_bias
        )
        interact_pos_y_ = (
                interact_pos - quat_apply(candidate_rotate, y_axis * (self.eef_hand_dist)) + z_bias
        )

        manipulate_pos_y = (
                manipulate_pos + quat_apply(candidate_rotate, y_axis * (self.eef_hand_dist)) + z_bias
        )
        manipulate_pos_y_ = (
                manipulate_pos - quat_apply(candidate_rotate, y_axis * (self.eef_hand_dist)) + z_bias
        )

        # contrain_point_far stores points cannot be too close to the object
        constrain_init_point_far = torch.stack([init_pos, init_pos_y, init_pos_y_], dim=1)
        constrain_interact_point_far = torch.stack([interact_pos, interact_pos_y, interact_pos_y_], dim=1)

        # to void collision, we compute trajectory and moving path
        reaching_path_step = 20
        constrain_init_to_interact = constrain_interact_point_far - constrain_init_point_far

        far_from_pc_mask = torch.tensor([True]).repeat(num_sample).to(self.device)
        if not is_grasp:
            constrain_step = reaching_path_step * 6 // 10
        else:
            constrain_step = reaching_path_step * 6 // 30

        for i in range(constrain_step):
            step_scale = i / reaching_path_step
            constrain_dist = step_scale * constrain_dist + constrain_dist_min
            constrain_point = constrain_init_point_far + step_scale * constrain_init_to_interact
            # far_away_dist = self.eef_hand_dist * 1.2
            far_from_pc_mask_i = self.compute_far_from_pc_mask(constrain_point, constrain_dist)
            # print("step mask: ", far_from_pc_mask_i.sum())
            far_from_pc_mask = torch.logical_and(far_from_pc_mask, far_from_pc_mask_i)

            # print(far_from_pc_mask.sum())

        # constrain_point_workspace stores points need to be inside the workspace
        constrain_point_workspace = torch.stack([init_pos, manipulate_pos, interact_pos], dim=1)

        inside_workspace_mask = self.compute_inside_workspace_mask(constrain_point_workspace)

        mask = torch.logical_and(far_from_pc_mask, inside_workspace_mask)
        # print("workspace mask num: ", mask.sum().item())

        # constrain point above the ground
        constrain_point_above_ground = torch.stack([init_pos, manipulate_pos, interact_pos,
                                                    init_pos_y, init_pos_y_, interact_pos_y, interact_pos_y_,
                                                    manipulate_pos_y, manipulate_pos_y_], dim=1)
        mask_above_ground = self.compute_above_ground_mask(constrain_point_above_ground)
        mask = torch.logical_and(mask, mask_above_ground)

        # print("far mask num: ", mask.sum().item())

        # constrain init_pos to interact_pos vector cannot point backward
        interact_init_vec = interact_pos - init_pos
        interact_init_vec_norm = torch.norm(interact_init_vec, dim=1).unsqueeze(-1)
        uni_interact_init_vec = interact_init_vec / interact_init_vec_norm

        dot_x = torch.sum(uni_interact_init_vec * x_axis, dim=1)
        dot_z = torch.sum(uni_interact_init_vec * -z_axis, dim=1)

        # TODO grasping is hard to satisified this
        mask_correct_direction = torch.logical_and(dot_x < 0.1, dot_z < 5.0)
        # print("mask")
        # print((dot_x < 0.5).sum())
        # print((dot_z < 1.0).sum())
        # print(mask_correct_direction.sum())
        mask = torch.logical_and(mask, mask_correct_direction)
        # print("correct direction mask num: ", mask.sum().item())

        # constrain point inside voxel bounding box
        mask_inside_voxel_bnd = self.compute_inside_voxel_bnd_mask(constrain_point_workspace)
        mask = torch.logical_and(mask, mask_inside_voxel_bnd)

        # print("inside voxel bnd mask: ", mask.sum().item())

        # print("final mask num: ", mask.sum())
        return mask

    def generate_contact_graspnet_params(self, save_dir):
        assert self.pointcloud.shape[0] > 0
        assert len(self.graspnet_pc) > 0
        assert self.graspnet_pc[0].shape[0] > 600

        if self.grasps is not None:
            return

        all_graspnet_pc = np.concatenate(self.graspnet_pc)
        all_graspnet_pc_color = np.concatenate(self.graspnet_pc_rgb)

        # TODO do workspace detection here
        ############################
        valid_points_tensor = to_torch(all_graspnet_pc, device=self.device)
        points_inside_mask = self.compute_inside_workspace_mask(valid_points_tensor.unsqueeze(1)).squeeze()
        points_inside_mask = points_inside_mask.cpu().numpy()
        all_graspnet_pc = all_graspnet_pc[points_inside_mask]
        all_graspnet_pc_color = all_graspnet_pc_color[points_inside_mask]
        ############################

        pcd = o3d.open3d.geometry.PointCloud()
        pcd.points = o3d.open3d.utility.Vector3dVector(all_graspnet_pc)
        pcd.colors = o3d.open3d.utility.Vector3dVector(all_graspnet_pc_color)
        # o3d.visualization.draw_geometries([pcd])

        downpcd = pcd.voxel_down_sample(voxel_size=0.001)
        downpcd_plane, ind = downpcd.remove_radius_outlier(nb_points=64, radius=0.02)

        # test contact grasp here
        ###################################################################
        plane_model, inliers = downpcd.segment_plane(distance_threshold=0.01,
                                                     ransac_n=3,
                                                     num_iterations=1000,
                                                     probability=0.9999)

        if self.object_cate != "Faucet" and self.object_cate != "Switch":
            downpcd = downpcd.select_by_index(inliers, invert=True)

        render_colors = np.asarray(downpcd.colors)

        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(
                downpcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))

        points = np.asarray(downpcd.points)

        # TODO visualization here
        max_label = max(labels)
        # Testing cluster heuristic
        print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        downpcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        # o3d.visualization.draw_geometries([downpcd])

        valid = labels >= 0
        valid_points = points[valid]
        labels = labels[valid]
        render_colors = render_colors[valid, :]

        exist_labels = np.unique(labels)

        grasp_points = np.empty([0, 3])
        grasp_points_color = np.empty([0, 3])

        pc_seg = {}
        num_pc_threshold = 100
        for label_i in exist_labels:
            pc_ind = (labels == label_i)
            num_pc_ind = pc_ind.sum()

            if num_pc_ind >= num_pc_threshold:
                pc_i = valid_points[pc_ind]
                pc_i_color = render_colors[pc_ind]

                grasp_points = np.concatenate([grasp_points, pc_i], axis=0)
                grasp_points_color = np.concatenate([grasp_points_color, pc_i_color], axis=0)
                pc_seg[label_i] = pc_i

                print("cluster ", label_i, " has num pc: ", num_pc_ind)


        pc_seg["pcd"] = grasp_points
        pc_seg["color"] = grasp_points_color

        save_file = save_dir + f"/{str(self.object_id)}_{self.multi_pose_str}" + ".pkl"
        with open(save_file, 'wb') as f:
            pickle.dump(pc_seg, f)

    def generate_grasps(self):

        assert self.pointcloud.shape[0] > 0
        assert len(self.graspnet_pc) > 0
        assert self.graspnet_pc[0].shape[0] > 600

        if self.grasps is not None:
            return

        all_graspnet_pc = np.concatenate(self.graspnet_pc)
        all_graspnet_pc_color = np.concatenate(self.graspnet_pc_rgb)



        # TODO do workspace detection here
        ############################
        valid_points_tensor = to_torch(all_graspnet_pc, device=self.device)
        points_inside_mask = self.compute_inside_workspace_mask(valid_points_tensor.unsqueeze(1)).squeeze()
        points_inside_mask = points_inside_mask.cpu().numpy()
        all_graspnet_pc = all_graspnet_pc[points_inside_mask]
        all_graspnet_pc_color = all_graspnet_pc_color[points_inside_mask]
        ############################


        pcd = o3d.open3d.geometry.PointCloud()
        pcd.points = o3d.open3d.utility.Vector3dVector(all_graspnet_pc)
        pcd.colors = o3d.open3d.utility.Vector3dVector(all_graspnet_pc_color)
        # o3d.visualization.draw_geometries([pcd])

        downpcd = pcd.voxel_down_sample(voxel_size=0.001)
        downpcd_plane, ind = downpcd.remove_radius_outlier(nb_points=64, radius=0.02)

        plane_model, inliers = downpcd_plane.segment_plane(distance_threshold=0.01,
                                                           ransac_n=3,
                                                           num_iterations=1000,
                                                           probability=0.9999)

        # Get the plane's normal and centroid
        plane_normal = plane_model[:3]
        plane_centroid = np.mean(np.asarray(downpcd_plane.points)[inliers], axis=0)

        plane_longest_edge = compute_largest_xyz_bnd(np.asarray(downpcd_plane.points))

        downpcd = downpcd_plane.select_by_index(inliers, invert=True)

        # Compute the distance from each point to the plane
        distances = np.dot(np.asarray(downpcd.points) - plane_centroid, plane_normal)

        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(
                downpcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))

        max_label = labels.max()

        ############################################
        ############ Vis labels
        # print(f"point cloud has {max_label + 1} clusters")
        # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        # colors[labels < 0] = 0
        # downpcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        # print("labels: ", np.unique(labels))
        # o3d.visualization.draw_geometries([downpcd])
        ############################################

        # estimate handle position
        lower_bnd = 0.02
        upper_bnd = 0.1

        threshold = 100
        for label_idx in range(max_label+1):
            cluster_points = np.asarray(downpcd.points)[labels == label_idx, :]
            num_cluster_points = cluster_points.shape[0]

            ############################################
            ############ Vis labels
            # print("num pts: ", num_cluster_points)
            # pcd = o3d.open3d.geometry.PointCloud()
            # pcd.points = o3d.open3d.utility.Vector3dVector(cluster_points)
            # o3d.visualization.draw_geometries([pcd])
            ############################################

            if num_cluster_points >= threshold:
                # cluster_points_rgb = np.asarray(downpcd.colors)[labels == label_idx, :]
                cluster_points_dist = distances[labels == label_idx]
                cluster_mean_dist = cluster_points_dist.mean()
                is_heuristic_handle = abs(cluster_mean_dist) > lower_bnd and abs(cluster_mean_dist) < upper_bnd

                max_bnd_len = compute_largest_xyz_bnd(cluster_points)

                x_range = cluster_points[:, 0].max() - cluster_points[:, 0].min()
                y_range = cluster_points[:, 1].max() - cluster_points[:, 1].min()
                z_range = cluster_points[:, 2].max() - cluster_points[:, 2].min()

                # Find the longest range
                longest_range = max(x_range, y_range, z_range)

                if is_heuristic_handle and max_bnd_len < 0.5 * plane_longest_edge and longest_range < 0.5:
                    if self.heuristic_grasp_pc is None:
                        self.heuristic_grasp_pc = cluster_points
                    else:
                        self.heuristic_grasp_pc = np.concatenate((self.heuristic_grasp_pc, cluster_points), 0)

        ############################################
        ############ Vis labels
        # pcd = o3d.open3d.geometry.PointCloud()
        # pcd.points = o3d.open3d.utility.Vector3dVector(self.heuristic_grasp_pc)
        # o3d.visualization.draw_geometries([pcd])
        ############################################

        # down-sample heuristic grasp pc
        if self.heuristic_grasp_pc is not None:
            downpcd_heuristic = o3d.open3d.geometry.PointCloud()
            downpcd_heuristic.points = o3d.open3d.utility.Vector3dVector(self.heuristic_grasp_pc)
            downpcd_heuristic = downpcd_heuristic.voxel_down_sample(voxel_size=0.005)

            self.heuristic_grasp_pc = np.asarray(downpcd_heuristic.points)

            # print(self.heuristic_grasp_pc.shape[0])
            # o3d.visualization.draw_geometries([downpcd_heuristic])

        # Testing cluster heuristic
        # print(f"point cloud has {max_label + 1} clusters")
        # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        # colors[labels < 0] = 0
        # downpcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        # o3d.visualization.draw_geometries([downpcd])

        graspnet_input_pc = np.asarray(downpcd_plane.points)
        graspnet_input_pc_rgb = np.asarray(downpcd_plane.colors)
        self.grasp = None

        # extract pre_saved grasp
        grasp_save_dir = "./dataset/grasps"
        if not Path(grasp_save_dir).exists():
            self.grasp = None
            return


        grasps_list = os.listdir(grasp_save_dir)
        save_grasp_file = f"{str(self.object_id)}_{self.multi_pose_str}.pkl"
        if not save_grasp_file in grasps_list:
            self.grasp = None
            return

        grasp_file = grasp_save_dir + "/" + save_grasp_file
        with open(grasp_file, 'rb') as f:
            grasps = pickle.load(f)


        if grasps["grasps"].shape[0] == 0:
            self.graps = None
            return

        self.grasps = grasps

        # # import graspnet to predict grasp
        # from new_scripts.graspnet.graspnet import predict_grasp, vis_grasps, collision_detection
        #
        # gg, cloud = predict_grasp(graspnet_input_pc, graspnet_input_pc_rgb, self.device)
        # # gg.sort_by_score()
        #
        # if self.grasps is None:
        #     self.grasps = gg
        # else:
        #     self.grasps.add(gg)
        #
        # gg_all_collide = collision_detection(self.grasps, all_graspnet_pc)
        # gg_all_collide.sort_by_score()
        #
        # if len(gg_all_collide) > self.top_k_grasps:
        #     self.grasps = gg_all_collide[:self.top_k_grasps]  # pick 100 grasps
        # # vis_grasps(gg_all_collide[:100], outlier_cloud)
        # # vis_grasps(gg_all_collide[:self.top_k_grasps], pcd)
        # if len(self.grasps) == 0:
        #     self.grasps = None

    def heuristic_handle_grasp_action(self, num_sample):
        if num_sample == 0:
            return

        assert self.heuristic_grasp_pc is not None

        # print("+++++++++++generate grasp action: ", num_sample)
        origin_sample_num = self.valid_init_state.shape[0]

        # rotation, init_grasp_dist, select_points, manipulate_pos

        valid_action_num = 0
        interact_points = torch.Tensor().to(self.device)
        valid_init_state_pos = torch.Tensor().to(self.device)
        valid_force = torch.Tensor().to(self.device)
        unit_force = torch.Tensor([
#            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1]
        ]).to(self.device)

        sample_scale = unit_force.shape[0]

        sample_iter = 0

        while valid_action_num <= num_sample:
            sample_iter += 1
            if sample_iter > 50:
                print("sample heuristic action many times !!!!!!!!!!!!!!!")
                self.grasp_action(num_sample)
                return

            # randomly sample rotation matrix
            sample_rotation = to_torch(R.random(num_sample).as_quat(), device=self.device)

            # TODO test pull
            sample_rotation = torch.Tensor().to(self.device)
            for i in range(num_sample):
                rand_z_rot = math.pi * random.uniform(0, 1)
                qx = gymapi.Quat.from_euler_zyx(0.5 * math.pi + random.uniform(0, 1) * 0.1, rand_z_rot,
                                                0.5 * math.pi + random.uniform(0, 1) * 0.1)
                rotation = to_torch([qx.x, qx.y, qx.z, qx.w], device=self.device).unsqueeze(0)
                sample_rotation = torch.cat((sample_rotation, rotation), 0)
                # sample_rotation = to_torch([qx.x, qx.y, qx.z, qx.w], device=self.device).repeat((num_sample, 1))

            # randomly sample point to interact
            index = np.random.choice(self.heuristic_grasp_pc.shape[0], num_sample, replace=True)
            selected_point = to_torch(self.heuristic_grasp_pc[index], device=self.device)

            z_axis = to_torch([0, 0, -1], device=self.device).repeat((num_sample, 1))

            dist_min = self.init_dist * 0.4
            dist_max = self.init_dist * 1.0

            dist_min = self.init_dist * 0.5
            dist_max = self.init_dist * 0.5

            sample_grasp_pos = selected_point + quat_apply(sample_rotation, z_axis * self.eef_hand_dist)

            candidate_points = to_torch(self.candidate_points, device=self.device)
            sample_candidate_dist = torch.cdist(candidate_points, selected_point)
            min_dist_index = torch.argmin(sample_candidate_dist, dim=0)

            init_dist_mean = (dist_min + dist_max) / 2.5  # fix the init dist
            init_pos = sample_grasp_pos + quat_apply(sample_rotation, z_axis * init_dist_mean)

            # repeat rotation and grasp
            selected_point = selected_point.repeat(sample_scale, 1)
            candidate_rotate = sample_rotation.repeat(sample_scale, 1)
            interact_pos = sample_grasp_pos.repeat(sample_scale, 1)
            init_pos = init_pos.repeat(sample_scale, 1)
            min_dist_index = min_dist_index.repeat(sample_scale)

            unit_force_repeat = unit_force.repeat(num_sample, 1) * dist_min * 2.0
            manipulate_move_noise = self.sample_move_pos(dist_min * 0.01, dist_min * 0.05, num_sample * sample_scale)

            manipulate_move_pos = unit_force_repeat + manipulate_move_noise

            manipulate_pos = manipulate_move_pos + interact_pos

            # rejection sampling to filtering action
            mask = self.rejection_sampling_mask(candidate_rotate, init_pos, interact_pos, manipulate_pos, is_grasp=True)

            valid_rotate = candidate_rotate[mask]
            valid_init_pos = init_pos[mask]
            valid_interact_pos = interact_pos[mask]
            valid_manipulate_pos = manipulate_pos[mask]
            valid_state = torch.cat((valid_init_pos, valid_rotate), 1)

            valid_num = valid_rotate.shape[0]
            valid_init_dist = torch.Tensor([init_dist_mean]).repeat(valid_num).unsqueeze(-1).to(self.device)
            valid_select_points = selected_point[mask]

            valid_num = int(mask.sum())
            init_vel = torch.zeros((valid_num, 6), device=self.device)
            valid_state = torch.cat((valid_state, init_vel), 1)

            self.valid_init_state = torch.cat((self.valid_init_state, valid_state), 0)
            self.interact_pos = torch.cat((self.interact_pos, valid_interact_pos), 0)
            self.rotation = torch.cat((self.rotation, valid_rotate), 0)
            self.init_pos = torch.cat((self.init_pos, valid_init_pos), 0)
            self.manipulate_pos = torch.cat((self.manipulate_pos, valid_manipulate_pos), 0)
            self.init_grasp_dist = torch.cat((self.init_grasp_dist, valid_init_dist), 0)
            self.select_points = torch.cat((self.select_points, valid_select_points), 0)

            interact_points = torch.cat((interact_points, selected_point[mask]))
            valid_action_num += mask.sum().cpu().numpy()

            # add valid index for heatmap visualization
            valid_selected_pc_index = min_dist_index[mask]
            valid_selected_pc_index_list = valid_selected_pc_index.detach().cpu().numpy().tolist()
            self.valid_index.extend(valid_selected_pc_index_list)

        data_collected_num = origin_sample_num + num_sample

        # TODO save the valid interaction
        self.select_points = self.select_points[:data_collected_num, :]
        self.valid_init_state = self.valid_init_state[:data_collected_num, :]
        self.interact_pos = self.interact_pos[:data_collected_num, :]
        self.valid_index = self.valid_index[:data_collected_num]
        self.rotation = self.rotation[:data_collected_num, :]
        self.init_pos = self.init_pos[:data_collected_num, :]
        self.manipulate_pos = self.manipulate_pos[:data_collected_num, :]
        self.init_grasp_dist = self.init_grasp_dist[:data_collected_num, :]
        self.valid_index = [int(x) for x in self.valid_index]

        self.pc_hit = np.zeros(self.candidate_points.shape[0])

    def grasp_action(self, num_sample):
        if num_sample == 0:
            return

        assert self.grasps is not None

        # print("+++++++++++generate grasp action: ", num_sample)
        origin_sample_num = self.valid_init_state.shape[0]

        # rotation, init_grasp_dist, select_points, manipulate_pos

        valid_action_num = 0
        interact_points = torch.Tensor().to(self.device)
        valid_init_state_pos = torch.Tensor().to(self.device)
        valid_force = torch.Tensor().to(self.device)
        unit_force = torch.Tensor([
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ]).to(self.device)

        sample_scale = unit_force.shape[0]

        sample_iter = 0

        while valid_action_num <= num_sample:
            sample_iter += 1
            if sample_iter > 50:
                print("sample grasp action many times")
                exit()

            num_grasps = self.grasps["grasps"].shape[0]
            # pick grasps
            grasps_idxs = np.random.choice(num_grasps, num_sample,
                                           replace=True)  # noticed we only takes num_sample
            selected_grasps = self.grasps["contact_pts"][grasps_idxs]

            sample_rotation = torch.Tensor().to(self.device)
            sample_grasp_pos = torch.Tensor().to(self.device)

            for i in range(num_sample):
                # grasp_i = selected_grasps[i]
                # grasp_trans = grasp_i.translation
                grasp_trans = torch.Tensor(selected_grasps[i]).to(self.device)

                # grasp_rot = grasp_i.rotation_matrix
                # grasp_rot = torch.Tensor(grasp_rot).to(self.device)
                # grasp_quat = pytorch3d.transforms.matrix_to_quaternion(grasp_rot)

                # rotate in z axis
                # grasp_quat_ = to_torch(R.random(1).as_quat(), device=self.device).squeeze()

                # [[ 0.7847,  0.6176, -0.0515, -0.0141],
                #  [ 0.6176, -0.7847, -0.0141,  0.0515]]

                rand_z_rot = math.pi * random.uniform(0, 1)
                qx = gymapi.Quat.from_euler_zyx(0.5 * math.pi + random.uniform(0, 1) * 0.1, rand_z_rot,
                                                    0.5 * math.pi + random.uniform(0, 1) * 0.1)
                rotation = to_torch([qx.x, qx.y, qx.z, qx.w], device=self.device).unsqueeze(0)
                sample_rotation = torch.cat((sample_rotation, rotation), 0)

                sample_grasp_pos = torch.cat((sample_grasp_pos, grasp_trans.unsqueeze(0)), 0)
                # sample_rotation = torch.cat((sample_rotation, grasp_quat_.unsqueeze(0)), 0)


            # Find the closest point on the point cloud instead
            candidate_points = to_torch(self.candidate_points, device=self.device)
            sample_candidate_dist = torch.cdist(candidate_points, sample_grasp_pos)
            min_dist_index = torch.argmin(sample_candidate_dist, dim=0)
            selected_point = candidate_points[min_dist_index]

            z_axis = to_torch([0, 0, -1], device=self.device).repeat((num_sample, 1))
            y_axis = to_torch([0, 1, 0], device=self.device).repeat((num_sample, 1))

            dist_min = self.init_dist * 0.4
            dist_max = self.init_dist * 1.0

            dist_min = self.init_dist * 0.5
            dist_max = self.init_dist * 0.5
            # selected_point = sample_grasp_pos + quat_apply(sample_rotation, -z_axis * self.eef_hand_dist)
            # selected_point = sample_grasp_pos
            sample_grasp_pos = selected_point + quat_apply(sample_rotation, z_axis * self.eef_hand_dist)

            candidate_points = to_torch(self.candidate_points, device=self.device)
            sample_candidate_dist = torch.cdist(candidate_points, selected_point)
            min_dist_index = torch.argmin(sample_candidate_dist, dim=0)

            init_dist_mean = (dist_min + dist_max) / 2  # fix the init dist
            init_pos = sample_grasp_pos + quat_apply(sample_rotation, z_axis * init_dist_mean)

            # repeat rotation and grasp
            selected_point = selected_point.repeat(sample_scale, 1)
            candidate_rotate = sample_rotation.repeat(sample_scale, 1)
            interact_pos = sample_grasp_pos.repeat(sample_scale, 1)
            init_pos = init_pos.repeat(sample_scale, 1)
            min_dist_index = min_dist_index.repeat(sample_scale)

            unit_force_ = unit_force.repeat(num_sample, 1) * dist_min * 2.0
            manipulate_move_noise = self.sample_move_pos(dist_min * 0.01, dist_min * 0.05, num_sample * sample_scale)

            manipulate_move_pos = unit_force_ + manipulate_move_noise

            manipulate_pos = manipulate_move_pos + interact_pos

            # rejection sampling to filtering action
            mask = self.rejection_sampling_mask(candidate_rotate, init_pos, interact_pos, manipulate_pos, is_grasp=True)

            valid_rotate = candidate_rotate[mask]
            valid_init_pos = init_pos[mask]
            valid_interact_pos = interact_pos[mask]
            valid_manipulate_pos = manipulate_pos[mask]
            valid_state = torch.cat((valid_init_pos, valid_rotate), 1)

            valid_num = valid_rotate.shape[0]
            valid_init_dist = torch.Tensor([init_dist_mean]).repeat(valid_num).unsqueeze(-1).to(self.device)
            valid_select_points = selected_point[mask]

            valid_num = int(mask.sum())
            init_vel = torch.zeros((valid_num, 6), device=self.device)
            valid_state = torch.cat((valid_state, init_vel), 1)

            self.valid_init_state = torch.cat((self.valid_init_state, valid_state), 0)
            self.interact_pos = torch.cat((self.interact_pos, valid_interact_pos), 0)
            self.rotation = torch.cat((self.rotation, valid_rotate), 0)
            self.init_pos = torch.cat((self.init_pos, valid_init_pos), 0)
            self.manipulate_pos = torch.cat((self.manipulate_pos, valid_manipulate_pos), 0)
            self.init_grasp_dist = torch.cat((self.init_grasp_dist, valid_init_dist), 0)
            self.select_points = torch.cat((self.select_points, valid_select_points), 0)

            interact_points = torch.cat((interact_points, selected_point[mask]))
            valid_action_num += mask.sum().cpu().numpy()

            # add valid index for heatmap visualization
            valid_selected_pc_index = min_dist_index[mask]
            valid_selected_pc_index_list = valid_selected_pc_index.detach().cpu().numpy().tolist()
            self.valid_index.extend(valid_selected_pc_index_list)

        data_collected_num = origin_sample_num + num_sample

        # TODO save the valid interaction
        self.select_points = self.select_points[:data_collected_num, :]
        self.valid_init_state = self.valid_init_state[:data_collected_num, :]
        self.interact_pos = self.interact_pos[:data_collected_num, :]
        self.valid_index = self.valid_index[:data_collected_num]
        self.rotation = self.rotation[:data_collected_num, :]
        self.init_pos = self.init_pos[:data_collected_num, :]
        self.manipulate_pos = self.manipulate_pos[:data_collected_num, :]
        self.init_grasp_dist = self.init_grasp_dist[:data_collected_num, :]
        self.valid_index = [int(x) for x in self.valid_index]

        self.pc_hit = np.zeros(self.candidate_points.shape[0])

    def uniform_action(self, num_sample):
        """
        uniform_action must be the first method to collect action
        """

        self.valid_index = []
        if num_sample == 0:
            self.valid_init_state = torch.Tensor().to(self.device)
            self.interact_pos = torch.Tensor().to(self.device)
            self.rotation = torch.Tensor().to(self.device)
            self.init_pos = torch.Tensor().to(self.device)
            self.manipulate_pos = torch.Tensor().to(self.device)
            self.init_grasp_dist = torch.Tensor().to(self.device)
            self.select_points = torch.Tensor().to(self.device)
            self.pc_hit = np.zeros(self.candidate_points.shape[0])

            return

        print("generate uniform action: ", num_sample)
        valid_action_num = 0
        # use sample_scale to perform over-saturated sampling
        sample_scale = 21
        self.valid_init_state = None

        interact_points = torch.Tensor().to(self.device)

        sample_iter = 0
        while valid_action_num <= num_sample:
            sample_iter += 1
            if sample_iter > 50:
                exit()

            # randomly sample rotation matrix
            candidate_rotate = to_torch(R.random(self.num_envs * sample_scale).as_quat(), device=self.device)

            # TODO test pull
            # qx = gymapi.Quat.from_euler_zyx(0.5 * math.pi + random.uniform(0, 1) * 0.1, 0.0 * math.pi+ random.uniform(0, 1) * 0.1, 0.5 * math.pi+ random.uniform(0, 1) * 0.1)
            # candidate_rotate = to_torch([qx.x, qx.y, qx.z, qx.w], device=self.device).repeat((self.num_envs * sample_scale, 1))

            # randomly sample point to interact
            index = np.random.choice(self.candidate_points.shape[0], self.num_envs * sample_scale, replace=True)

            selected_pc_index = to_torch(index, device=self.device)
            selected_point = to_torch(self.candidate_points[index], device=self.device)

            z_axis = to_torch([0, 0, -1], device=self.device).repeat((self.num_envs * sample_scale, 1))
            y_axis = to_torch([0, 1, 0], device=self.device).repeat((self.num_envs * sample_scale, 1))

            # interaction pos for gripper to grasp the objects
            # mul 1.1 not grasp the surface
            interact_pos = selected_point + quat_apply(candidate_rotate, z_axis * self.eef_hand_dist * 1.1)

            # moving distance for reaching and manipulating
            dist_min = self.init_dist * 0.4
            dist_max = self.init_dist * 1.0

            dist_min = self.init_dist * 0.5
            dist_max = self.init_dist * 0.5

            # compute init postion and manipulation position in world frame based on random sampling

            # init_move_pos = self.sample_move_pos(dist_min, dist_max, self.num_envs * sample_scale)
            # init_pos = init_move_pos + interact_pos

            # TODO use premitive initial pose
            dist_norm = torch.rand(self.num_envs * sample_scale).to(self.device)
            init_grasp_dist = (dist_norm * (dist_max - dist_min) + dist_min).unsqueeze(-1)
            dist = init_grasp_dist.repeat((1, 3))
            init_pos = interact_pos + quat_apply(candidate_rotate, z_axis * dist)

            # TODO use discrete moving direction
            unit_force = torch.Tensor([
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
                [0, 0, 1],
                [0, 0, -1]]).to(self.device)

            num_unit_force = unit_force.shape[0]

            unit_force = unit_force.repeat(self.num_envs * sample_scale // num_unit_force, 1) * dist_min * 2.0
            manipulate_move_noise = self.sample_move_pos(dist_min * 0.01, dist_min * 0.05, self.num_envs * sample_scale)

            manipulate_move_pos = unit_force + manipulate_move_noise
            # manipulate_move_pos = self.sample_move_pos(dist_min, dist_max, self.num_envs * sample_scale)

            manipulate_pos = manipulate_move_pos + interact_pos

            # rejection sampling to filtering action
            mask = self.rejection_sampling_mask(candidate_rotate, init_pos, interact_pos, manipulate_pos)

            valid_rotate = candidate_rotate[mask]
            valid_init_pos = init_pos[mask]
            valid_interact_pos = interact_pos[mask]
            valid_manipulate_pos = manipulate_pos[mask]
            valid_state = torch.cat((valid_init_pos, valid_rotate), 1)
            valid_init_dist = init_grasp_dist[mask]
            if valid_action_num == 0:
                self.valid_init_state = valid_state
                self.interact_pos = valid_interact_pos
                self.rotation = valid_rotate
                self.init_pos = valid_init_pos
                self.manipulate_pos = valid_manipulate_pos
                self.init_grasp_dist = valid_init_dist
            else:
                self.valid_init_state = torch.cat((self.valid_init_state, valid_state), 0)
                self.interact_pos = torch.cat((self.interact_pos, valid_interact_pos), 0)
                self.rotation = torch.cat((self.rotation, valid_rotate), 0)
                self.init_pos = torch.cat((self.init_pos, valid_init_pos), 0)
                self.manipulate_pos = torch.cat((self.manipulate_pos, valid_manipulate_pos), 0)
                self.init_grasp_dist = torch.cat((self.init_grasp_dist, valid_init_dist), 0)

            interact_points = torch.cat((interact_points, selected_point[mask]))
            valid_action_num += mask.sum().cpu().numpy()

            valid_selected_pc_index = selected_pc_index[mask]
            valid_selected_pc_index_list = valid_selected_pc_index.detach().cpu().numpy().tolist()
            self.valid_index.extend(valid_selected_pc_index_list)

        # TODO save the valid interaction
        self.select_points = interact_points[:num_sample, :]
        self.valid_init_state = self.valid_init_state[:num_sample, :]
        self.interact_pos = self.interact_pos[:num_sample, :]
        self.valid_index = self.valid_index[:num_sample]
        self.rotation = self.rotation[:num_sample, :]
        self.init_pos = self.init_pos[:num_sample, :]
        self.manipulate_pos = self.manipulate_pos[:num_sample, :]
        self.init_grasp_dist = self.init_grasp_dist[:num_sample, :]
        self.valid_index = [int(x) for x in self.valid_index]

        init_vel = torch.zeros((num_sample, 6), device=self.device)
        self.valid_init_state = torch.cat((self.valid_init_state, init_vel), 1)

        self.pc_hit = np.zeros(self.candidate_points.shape[0])

    def get_current_action_primitives(self):
        return self.rotation, self.init_pos, self.interact_pos, self.manipulate_pos

    def get_action_seq(self):
        gripper_close = torch.ones((self.num_envs, 1)).to(self.device)
        gripper_open = torch.zeros((self.num_envs, 1)).to(self.device)

        init_pose = torch.cat((self.init_pos, self.rotation, gripper_open), dim=1)
        grasp_pose = torch.cat((self.interact_pos, self.rotation, gripper_open), dim=1)
        grasp_close_pose = torch.cat((self.interact_pos, self.rotation, gripper_close), dim=1)
        move_pose = torch.cat((self.manipulate_pos, self.rotation, gripper_close), dim=1)

        return init_pose, grasp_pose, grasp_close_pose, move_pose

    def test_action(self, points, rotation, force, init_poses=None):
        # TODO pre-define the num_envs as the number of action to be visualized
        self.select_points = to_torch(points, device=self.device)
        rotation = to_torch(rotation, device=self.device)
        self.norm_force = to_torch(force, device=self.device)
        force_norm = torch.norm(self.norm_force - 0.5, dim=-1).unsqueeze(-1)
        self.force = ((self.norm_force - 0.5) / force_norm) * 0.5 * self.init_dist

        # TODO what if we only has push action
        # self.force = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1)) * 0.5 * self.init_dist
        self.force[:, 2] += self.init_dist

        z_axis = to_torch([0, 0, -1], device=self.device).repeat((self.num_envs, 1))
        init_pos = self.select_points + quat_apply(rotation, z_axis * (self.init_dist + self.eef_hand_dist))

        valid_state = torch.cat((init_pos, rotation), 1)
        init_vel = torch.zeros((self.num_envs, 6), device=self.device)
        self.valid_init_state = torch.cat((valid_state, init_vel), 1)

        # init starting state of the object
        if init_poses is not None:
            for i in range(self.num_envs):
                for j in range(self.object_dof_num):
                    ij_dof_scale = int(init_poses[i][j]) / (self.init_pose_num - 1)
                    self.object_init_dof_pos[i, j] = ij_dof_scale

    def rgba2rgb(self, rgba, background=(255, 255, 255)):
        row, col, ch = rgba.shape
        if ch == 3:
            return rgba
        assert ch == 4, "RGBA image has 4 channels."
        rgb = np.zeros((row, col, 3), dtype="float32")
        r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]
        a = np.asarray(a, dtype="float32") / 255.0
        R, G, B = background
        rgb[:, :, 0] = r * a + (1.0 - a) * R
        rgb[:, :, 1] = g * a + (1.0 - a) * G
        rgb[:, :, 2] = b * a + (1.0 - a) * B
        return np.asarray(rgb, dtype="uint8")

    def save_proprioception(self):
        # save franka dof state for furthur RL
        # assert self.num_envs == 1
        franka_dof_pos = self.franka_dof_state[:, :, 0]

        # TODO instead of using joint dof, use pos and rot
        franka_state = torch.cat((self.hand_pos, self.hand_rot), 1)
        gripper_state = franka_dof_pos[:, -2:]
        franka_dof_pos = torch.cat((franka_state, gripper_state), 1)

        self.franka_proprio = torch.cat((self.franka_proprio, franka_dof_pos), dim=0)
        return franka_dof_pos

    def save_image_tensor_test(self, rgba_tensor, file_name="./test.png"):
        rgba_np = rgba_tensor.cpu().numpy()
        rgb = self.rgba2rgb(rgba_np)
        plt.imsave(file_name, rgb)

    def test_vgg_model(self, rgba_tensor):
        # This method test using model vgg19 to compute the rgb tensor feature
        model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19_bn', pretrained=True)
        model.eval()
        rgba_np = rgba_tensor.cpu().numpy()
        rgb = self.rgba2rgb(rgba_np)

        from PIL import Image
        rgb = Image.fromarray(rgb)
        # preprocess image size to (1, 3, 224, 224)
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        rgb = preprocess(rgb)
        rgb = rgb.unsqueeze(0)

        # Remove the final fully connected layers of the network
        model.classifier = model.classifier[:-3]

        # Freeze the parameters of the feature extractor
        for param in model.parameters():
            param.requires_grad = False

        feature = model(rgb)
        return feature

    def save_obs(self, is_rvt=False):
        # Save current state multi-camera observations

        # compute cam target based on avg of the candidate point cloud
        if is_rvt:
            vol_bnd_center = self.tsdf_vol_bnds.mean(1)
            cam_target = to_torch(vol_bnd_center, device=self.device)
        else:
            cam_target = torch.mean(self.candidate_points, dim=0)
            # add noise to generalize
            cam_target_noise = torch.randn(3).to(self.device) * 0.2
            cam_target += cam_target_noise

        cam_target_height = float(cam_target[2].cpu())
        cam_target_vec = gymapi.Vec3(cam_target[0], cam_target[1], cam_target[2])

        # init camera
        camera_properties = gymapi.CameraProperties()
        cam_width = self.image_width
        cam_height = self.image_height
        camera_properties.width = cam_width
        camera_properties.height = cam_height
        camera_properties.enable_tensors = True
        pi = math.pi

        camera_radius = self.camera_radius
        if is_rvt:
            camera_radius *= 1.5
            circle_point = self.PointsInCircumEven(-camera_radius, self.num_camera)
        else:
            circle_point = self.PointsInCircum(-camera_radius, self.num_camera)

        for camera_xy in circle_point:
            camera_xy.append(self.camera_height + cam_target_height)
        top_camera_pos = [0.1, 0.0, cam_target_height + self.camera_height + camera_radius * 0.7]
        # circle_point.insert(0, top_camera_pos)
        circle_point.append(top_camera_pos)

        if self.fu_list is None:
            self.fu_list = []
            self.fv_list = []
            if is_rvt:
                self.view_matrices_rvt = []
            else:
                self.view_matrices = []

        state_rgba_tensor = torch.Tensor().to(self.device)
        state_depth_tensor = torch.Tensor().to(self.device)
        state_seg_tensor = torch.Tensor().to(self.device)

        for env_index in range(self.num_envs):
            # create camera handle for env_index
            # get current obs camera handle

            need_generate_camera = len(self.save_rvt_obs_camera_handles)  < self.num_envs \
                if is_rvt else len(self.save_obs_camera_handles) < self.num_envs

            if need_generate_camera:
                curr_obs_camera_handles_env = []
                for i in range(self.num_camera + 1):
                    camera_handle = self.gym.create_camera_sensor(self.envs[env_index], camera_properties)
                    # Look at the env
                    cam_pos = gymapi.Vec3(circle_point[i][0], circle_point[i][1], circle_point[i][2])
                    # cam_target = gymapi.Vec3(0.0, 0.0, 1.0)
                    self.gym.set_camera_location(camera_handle, self.envs[env_index], cam_pos, cam_target_vec)
                    self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target_vec)
                    curr_obs_camera_handles_env.append(camera_handle)
                if is_rvt:
                    self.save_rvt_obs_camera_handles.append(curr_obs_camera_handles_env)
                else:
                    self.save_obs_camera_handles.append(curr_obs_camera_handles_env)

            # initialize camera
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)

            env_rgba_tensor = torch.Tensor().to(self.device)
            env_depth_tensor = torch.Tensor().to(self.device)
            env_seg_tensor = torch.Tensor().to(self.device)
            curr_view_matrices = []

            for i in range(self.num_camera + 1):
                # Retrieve depth and segmentation buffer
                # save tensor images
                # However, this is incorrect
                '''
                rgba_camera_tensor = self.gym.get_camera_image_gpu_tensor(
                    self.sim, self.envs[env_index], curr_obs_camera_handles[i], gymapi.IMAGE_COLOR
                )
                rgba_camera_tensor = gymtorch.wrap_tensor(rgba_camera_tensor)

                depth_camera_tensor = self.gym.get_camera_image_gpu_tensor(
                    self.sim, self.envs[env_index], curr_obs_camera_handles[i], gymapi.IMAGE_DEPTH
                )

                # IMAGE_SEGMENTATION
                seg_camera_tensor = self.gym.get_camera_image_gpu_tensor(
                    self.sim, self.envs[env_index], curr_obs_camera_handles[i], gymapi.IMAGE_SEGMENTATION
                )

                seg_camera_tensor = gymtorch.wrap_tensor(seg_camera_tensor)
                depth_camera_tensor = gymtorch.wrap_tensor(depth_camera_tensor)
                '''

                if is_rvt:
                    curr_obs_camera_handles = self.save_rvt_obs_camera_handles[env_index]
                else:
                    curr_obs_camera_handles = self.save_obs_camera_handles[env_index]

                # DEBUG
                depth_camera_tensor = self.gym.get_camera_image(self.sim, self.envs[env_index], curr_obs_camera_handles[i], gymapi.IMAGE_DEPTH)
                rgba_camera_tensor = self.gym.get_camera_image(
                    self.sim, self.envs[env_index], curr_obs_camera_handles[i], gymapi.IMAGE_COLOR
                ).reshape(cam_height, cam_width, -1)
                seg_camera_tensor = self.gym.get_camera_image(
                    self.sim, self.envs[env_index], curr_obs_camera_handles[i], gymapi.IMAGE_SEGMENTATION
                )

                # if not is_rvt:
                #     # do not mask out robot in rvt data collection
                #     rgba_camera_tensor[seg_camera_tensor == 9] = 0  # mask out the robot

                rgba_camera_tensor[seg_camera_tensor == 0] = 0  # mask out background and plane
                # self.save_image_tensor_test(rgba_camera_tensor, f"img_{i}_{self.task_state}.png")
                # self.save_image_tensor_test(rgba_camera_tensor, "after.png")

                depth_camera_tensor[seg_camera_tensor == 0] = 0  # mask out background and plane

                # to tensor
                depth_camera_tensor = to_torch(depth_camera_tensor, device=self.device)
                rgba_camera_tensor = to_torch(rgba_camera_tensor, device=self.device)
                seg_camera_tensor = to_torch(seg_camera_tensor.astype(np.int_), device=self.device)

                # feature = self.test_vgg_model(rgba_camera_tensor)

                # save camera params for later use
                need_save_camera_params = len(self.view_matrices_rvt) < self.num_camera + 1 \
                                                if is_rvt else len(self.view_matrices) < self.num_camera + 1
                if need_save_camera_params:
                    view_matrix = np.matrix(self.gym.get_camera_view_matrix(self.sim, self.envs[env_index],
                                                                  curr_obs_camera_handles[i]))

                    if is_rvt:
                        self.view_matrices_rvt.append(view_matrix)
                    else:
                        self.view_matrices.append(view_matrix)
                    # Get the camera projection matrix and get the necessary scaling
                    # coefficients for deprojection
                    proj = self.gym.get_camera_proj_matrix(self.sim, self.envs[env_index],
                                                           curr_obs_camera_handles[i])
                    fu = 2 / proj[0, 0]
                    fv = 2 / proj[1, 1]

                    self.fu_list.append(fu)
                    self.fv_list.append(fv)

                env_rgba_tensor = torch.cat((env_rgba_tensor, rgba_camera_tensor.unsqueeze(0)), dim=0)
                env_depth_tensor = torch.cat((env_depth_tensor, depth_camera_tensor.unsqueeze(0)), dim=0)
                env_seg_tensor = torch.cat((env_seg_tensor, seg_camera_tensor.unsqueeze(0)), dim=0)


            state_rgba_tensor = torch.cat((state_rgba_tensor, env_rgba_tensor.unsqueeze(0)), dim=0)
            state_depth_tensor = torch.cat((state_depth_tensor, env_depth_tensor.unsqueeze(0)), dim=0)
            state_seg_tensor = torch.cat((state_seg_tensor, env_seg_tensor.unsqueeze(0)), dim=0)

        # shape of rgba tensor: (num_state, num_env, num_camera, image_width, image_height, channel)

        if is_rvt:
            self.rgba_rvt_tensor = torch.cat((self.rgba_rvt_tensor, state_rgba_tensor.unsqueeze(0)), dim=0)
            self.depth_rvt_tensor = torch.cat((self.depth_rvt_tensor, state_depth_tensor.unsqueeze(0)), dim=0)
        else:
            self.rgba_tensor = torch.cat((self.rgba_tensor, state_rgba_tensor.unsqueeze(0)), dim=0)
            self.depth_tensor = torch.cat((self.depth_tensor, state_depth_tensor.unsqueeze(0)), dim=0)
            self.seg_tensor = torch.cat((self.seg_tensor, state_seg_tensor.unsqueeze(0)), dim=0)


        # print("save image list")

    def clear_state_obs_buf(self):
        # Clear observation buffer for previous state
        # Used for state-level testing

        self.rgba_tensor = torch.Tensor().to(self.device)
        self.depth_tensor = torch.Tensor().to(self.device)
        self.seg_tensor = torch.Tensor().to(self.device)

        self.rgba_rvt_tensor = torch.Tensor().to(self.device)
        self.depth_rvt_tensor = torch.Tensor().to(self.device)

    def get_state_obs_buf(self, is_rvt=False):
        if is_rvt:
            rgba_tensor = self.rgba_rvt_tensor.squeeze()
            depth_tensor = self.depth_rvt_tensor.squeeze()
        else:
            rgba_tensor = self.rgba_tensor.squeeze()
            depth_tensor = self.depth_tensor.squeeze()

        rgba_tensor = rgba_tensor.squeeze()
        batch_shape = rgba_tensor.shape[:-3]
        rgba_img_shape = rgba_tensor.shape[-3:]
        img_shape = depth_tensor.shape[-2:]

        rgba_tensor = rgba_tensor.view(-1, *rgba_img_shape)
        depth_tensor = depth_tensor.view(-1, *img_shape)
        num_camera = rgba_tensor.shape[0]

        rgb_imgs_list = []
        depth_imgs_list = []

        for camera_id in range(num_camera):
            rgba_i = rgba_tensor.squeeze()[camera_id]
            depth_i = depth_tensor.squeeze()[camera_id]

            rgba_np = rgba_i.cpu().numpy()
            depth_np = depth_i.cpu().numpy()

            # filter depth_np to avoid -inf
            depth_np[depth_np < -10001] = 0
            rgb = self.rgba2rgb(rgba_np)

            rgb_imgs_list.append(rgb)
            depth_imgs_list.append(depth_np)

        rgb_imgs = np.stack(rgb_imgs_list, axis=0)
        depth_imgs = np.stack(depth_imgs_list, axis=0)

        if is_rvt:
            view_matrices = self.view_matrices_rvt
        else:
            view_matrices = self.view_matrices

        # rgb_imgs = rgb_imgs.reshape(*batch_shape, *img_shape, 3)
        # depth_imgs = depth_imgs.reshape(*batch_shape, *img_shape)
        return rgb_imgs, depth_imgs, view_matrices

    def save_obs_voxel_finetune(self, dataset_root, traj_id=None, is_saving=True):
        # This function saves a list of voxel from

        if not (self.rgba_tensor.nelement() and self.depth_tensor.nelement()):
            print("Need to collect image list first!")
            exit()

        if not (self.rgba_rvt_tensor.nelement() and self.depth_rvt_tensor.nelement()):
            print("Need to collect rvt image list first!")
            exit()

        if len(self.view_matrices) == 0 or len(self.fu_list) == 0 or len(self.fv_list) == 0 or len(self.view_matrices_rvt) == 0:
            print("Need to collect camera instrinsic first")
            exit()

        object_dir = dataset_root + "/" + str(self.object_id)
        Path(object_dir).mkdir(parents=True, exist_ok=True)

        color_grid, voxel_grid = None, None

        if self.num_envs == 1:
            self.rgba_tensor = self.rgba_tensor.squeeze()
            self.depth_tensor = self.depth_tensor.squeeze()
            self.seg_tensor = self.seg_tensor.squeeze()
            self.rgba_rvt_tensor = self.rgba_rvt_tensor.squeeze()
            self.depth_rvt_tensor = self.depth_rvt_tensor.squeeze()
            # shape of rgba tensor: (num_state, num_camera, image_width, image_height, channel)

            if len(self.rgba_tensor.shape) < 5:
                self.rgba_tensor = self.rgba_tensor.unsqueeze(0)
                self.depth_tensor = self.depth_tensor.unsqueeze(0)
                self.seg_tensor = self.seg_tensor.unsqueeze(0)
                self.rgba_rvt_tensor = self.rgba_rvt_tensor.unsqueeze(0)
                self.depth_rvt_tensor = self.depth_rvt_tensor.unsqueeze(0)

            num_state = self.rgba_tensor.shape[0]

            # tsdf_volume_list = []
            for state_i in range(num_state):
                state_rgba_tensor = self.rgba_tensor[state_i, ...]
                state_depth_tensor = self.depth_tensor[state_i, ...]
                state_seg_tensor = self.seg_tensor[state_i, ...]

                tsdf_volume = self.compute_voxel_from_img_list(state_rgba_tensor, state_depth_tensor)
                # tsdf_volume_list.append(tsdf_volume)

                '''
                # Visualize voxel to debug
                verts, pc = tsdf_volume.get_point_cloud()

                points = pc[:, :3]
                pc_rgb = pc[:, 3:] / 255
                pcd = o3d.open3d.geometry.PointCloud()
                pcd.points = o3d.open3d.utility.Vector3dVector(points)
                pcd.colors = o3d.open3d.utility.Vector3dVector(pc_rgb)
                o3d.visualization.draw_geometries([pcd])
                '''

                self.save_image_tensor(state_rgba_tensor, state_depth_tensor, dir=object_dir, state_id=state_i, seg_tensor=state_seg_tensor,
                                       traj_id=traj_id, is_saving=is_saving)

                state_rgba_rvt_tensor = self.rgba_rvt_tensor[state_i, ...]
                state_depth_rvt_tensor = self.depth_rvt_tensor[state_i, ...]

                self.save_image_tensor(state_rgba_rvt_tensor, state_depth_rvt_tensor, dir=object_dir, state_id=state_i,
                                       traj_id=traj_id, is_rvt=True, is_saving=is_saving)

                voxel_grid, color_grid = tsdf_volume.get_volume()
                voxel_file_name = f"{str(self.object_id)}_{self.multi_pose_str}_{state_i}_{traj_id}"

                color_grid_file = object_dir + "/" + "color_" + voxel_file_name + ".npz"
                voxel_grid_file = object_dir + "/" + "voxel_" + voxel_file_name + ".npz"

                color_grid = color_grid.cpu().numpy()
                voxel_grid = voxel_grid.cpu().numpy()

                color_grid_file_path = Path(color_grid_file)
                voxel_grid_file_path = Path(voxel_grid_file)

                if (not color_grid_file_path.exists()) and is_saving:
                    np.savez_compressed(color_grid_file, voxel=color_grid)

                if (not voxel_grid_file_path.exists()) and is_saving:
                    np.savez_compressed(voxel_grid_file, voxel=voxel_grid)

            # save view matrices
            view_matrices_file_name = f"{str(self.object_id)}_{self.multi_pose_str}_{traj_id}"
            view_matrices_file = object_dir + "/" + "matrices_" + view_matrices_file_name + ".npz"

            view_matrices_file_path = Path(view_matrices_file)
            if (not view_matrices_file_path.exists()) and is_saving:
                np.savez_compressed(view_matrices_file_path, matrix=np.array(self.view_matrices))

            # save view matrices for rvt
            view_matrices_rvt_file = object_dir + "/" + "matrices_rvt_" + view_matrices_file_name + ".npz"

            view_matrices_rvt_file_path = Path(view_matrices_rvt_file)
            if (not view_matrices_rvt_file_path.exists()) and is_saving:
                np.savez_compressed(view_matrices_rvt_file_path, matrix=np.array(self.view_matrices_rvt))

        return color_grid, voxel_grid

    def save_image_tensor(self, rgba_tensor, depth_tensor, dir, state_id, seg_tensor=None, traj_id=None, is_rvt=False, is_saving=True):
        num_camera = rgba_tensor.shape[0]
        rgb, depth_np = None, None

        for camera_id in range(num_camera):
            rgba_i = rgba_tensor[camera_id]
            depth_i = depth_tensor[camera_id]

            rgba_np = rgba_i.cpu().numpy()
            depth_np = depth_i.cpu().numpy()

            # filter depth_np to avoid -inf
            depth_np[depth_np < -10001] = 0

            rgb = self.rgba2rgb(rgba_np)

            img_file_name = f"{str(self.object_id)}_{self.multi_pose_str}_{state_id}_{camera_id}_{traj_id}"

            if is_rvt:
                file_name_offset = "rvt_"
            else:
                file_name_offset = ""

            rgb_file = dir + "/" + file_name_offset + img_file_name + ".png"
            depth_file = dir + "/" + file_name_offset + img_file_name + ".npz"

            rgb_file_path = Path(rgb_file)
            depth_file_path = Path(depth_file)

            if (not rgb_file_path.exists()) and is_saving:
                plt.imsave(rgb_file, rgb)

            if (not depth_file_path.exists()) and is_saving:
                np.savez_compressed(depth_file, depth=depth_np)

            if seg_tensor is not None:
                seg_i = seg_tensor[camera_id]
                seg_np = seg_i.cpu().numpy()
                seg_file = dir + "/" + "seg_" + img_file_name + ".npz"
                seg_file_path = Path(seg_file)

                if (not seg_file_path.exists()) and is_saving:
                    np.savez_compressed(seg_file, seg=seg_np)

        return rgb, depth_np

    def compute_voxel_from_img_list(self, img_rgba, img_depth):
        # returns the voxel of current state
        # shape of rgba tensor: (num_camera, image_width, image_height, channel)
        num_camera = img_rgba.shape[0]
        voxel_size = (self.tsdf_vol_bnds[0, 1] - self.tsdf_vol_bnds[0, 0]) / self.num_voxel_per_len
        tsdf_volume = TSDFVolume(self.tsdf_vol_bnds, voxel_size, self.device)

        for i in range(num_camera):
            # print("Fusing frame %d/%d" % (i + 1, self.num_camera))
            # Read RGB-D image and camera pose
            # Integrate observation into voxel volume (assume color aligned with depth)
            color_img = img_rgba[i]
            depth_img = img_depth[i]
            color_img_np = color_img.cpu().numpy()
            depth_img_np = depth_img.cpu().numpy()
    
            tsdf_volume.integrate(color_img_np, depth_img_np, self.fu_list[i], self.fv_list[i], self.view_matrices[i],
                                  obs_weight=1.0)

        # TODO test generate novel view using rvt view matrices
        '''
        _, pc = tsdf_volume.get_point_cloud()
        pc_rgb = pc[:, 3:]
        pc_xyz = pc[:, :3]

        pc_rgb = to_torch(pc_rgb, device=self.device)
        pc_xyz = to_torch(pc_xyz, device=self.device)

        for j in range(num_camera):
            novel_img_depth, novel_img_rgb = tsdf_volume.pc_generate_novel_view(pc_xyz, pc_rgb, self.fu_list[j], self.fv_list[j], self.view_matrices_rvt[j])
            image_name = "test_" + str(j) + ".png"
            self.save_image_tensor_test(novel_img_rgb, image_name)
        '''

        return tsdf_volume

    def PointsInCircum(self, r, n=4, view_angle=2 * math.pi):
        offset = 0 if n % 2 else -0.5
        return [[math.cos(view_angle / n * (x + offset)) * r, math.sin(view_angle / n * (x + offset)) * r] for x in
                range((-n // 2) + 1, n // 2 + 1)]

    def PointsInCircumEven(self, r, n=4):
        return [[math.cos(2 * math.pi / n * (x + 0.5)) * r, math.sin(2 * math.pi / n * (x + 0.5)) * r] for x in range(0, n)]

    def generate_graspnet_pc(self):
        # compute cam target based on avg of the candidate point cloud
        cam_target = torch.mean(self.candidate_points, dim=0)
        cam_target_height = float(cam_target[2].cpu())
        cam_target = gymapi.Vec3(cam_target[0], cam_target[1], cam_target[2])

        camera_properties = gymapi.CameraProperties()
        cam_width = self.graspnet_image_width
        cam_height = self.graspne_image_height
        camera_properties.width = cam_width
        camera_properties.height = cam_height
        camera_properties.enable_tensors = True
        pi = math.pi
        circle_point = self.PointsInCircum(-self.graspnet_camera_radius, self.num_graspnet_camera,
                                           self.graspnet_view_angle)
        for camera_xy in circle_point:
            camera_xy.append(self.graspnet_camera_height + cam_target_height)
        # self.num_camera += 1
        cam_handles = []

        # pick env_index = 0 by default
        env_index = 0
        for i in range(self.num_graspnet_camera):
            camera_handle = self.gym.create_camera_sensor(self.envs[env_index], camera_properties)
            # Look at the env
            cam_pos = gymapi.Vec3(circle_point[i][0], circle_point[i][1], circle_point[i][2])

            self.gym.set_camera_location(camera_handle, self.envs[env_index], cam_pos, cam_target)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
            cam_handles.append(camera_handle)

        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.render_all_camera_sensors(self.sim)
        u_range = -torch.arange(-0.5, 0.5, 1 / cam_width).view(1, -1).cuda()
        v_range = torch.arange(-0.5, 0.5, 1 / cam_height).view(-1, 1).cuda()

        self.graspnet_pc = []
        self.graspnet_pc_rgb = []

        # save grsapnet pc and graspnet pc rgb
        for i in range(self.num_graspnet_camera):
            # Retrieve depth and segmentation buffer
            vinv = np.linalg.inv(
                np.matrix(self.gym.get_camera_view_matrix(self.sim, self.envs[env_index], cam_handles[i]))
            )
            depth_buffer = self.gym.get_camera_image(self.sim, self.envs[env_index], cam_handles[i], gymapi.IMAGE_DEPTH)
            color_buffer = self.gym.get_camera_image(
                self.sim, self.envs[env_index], cam_handles[i], gymapi.IMAGE_COLOR
            ).reshape(cam_height, cam_width, -1)
            seg_buffer = self.gym.get_camera_image(
                self.sim, self.envs[env_index], cam_handles[i], gymapi.IMAGE_SEGMENTATION
            )
            # self.gym.write_camera_image_to_file(self.sim, self.envs[env_index], cam_handles[i], gymapi.IMAGE_DEPTH, "frame-%06d.depth.png" % (i))
            # self.gym.write_camera_image_to_file(self.sim, self.envs[env_index], cam_handles[i], gymapi.IMAGE_COLOR, "frame-%06d.color.png" % (i))

            # Get the camera projection matrix and get the necessary scaling
            # coefficients for deprojection
            proj = self.gym.get_camera_proj_matrix(self.sim, self.envs[env_index], cam_handles[i])

            fu = 2 / proj[0, 0]
            fv = 2 / proj[1, 1]

            # Ignore any points which originate from ground plane or empty space
            depth_buffer[seg_buffer == 0] = -10001

            depth_buffer[seg_buffer == 9] = -10001

            depth_img = to_torch(depth_buffer, device=self.device)

            color_img = to_torch(color_buffer, device=self.device)
            # self.save_image_tensor_test(color_img, "test_graspnet_"+str(i)+".png")

            proj_u = fu * torch.mul(depth_img, u_range)
            proj_v = fv * torch.mul(depth_img, v_range)
            ones = torch.ones_like(depth_img)
            big_mat = torch.cat(
                (proj_u.unsqueeze(-1), proj_v.unsqueeze(-1), depth_img.unsqueeze(-1), ones.unsqueeze(-1)), dim=2
            )
            vinv = torch.from_numpy(vinv).to(self.device)
            max_depth = 20
            mask = depth_img > -max_depth

            pc_img = torch.matmul(big_mat, vinv)

            pc = pc_img[mask]
            pc_rgb_np = self.rgba2rgb(color_buffer)
            pc_rgb = to_torch(pc_rgb_np, device=self.device)[mask]
            pc_rgb = pc_rgb.float() / 255.0
            pc = pc[:, :3]

            self.graspnet_pc.append(pc.cpu().numpy())
            self.graspnet_pc_rgb.append(pc_rgb.cpu().numpy())

        # destory camera sensor after each generation
        for cam_handle in cam_handles:
            self.gym.destroy_camera_sensor(self.sim, self.envs[env_index], cam_handle)

    def viewer_camera(self, env_index, is_first, is_metric=False):

        print("render images")
        start_time = time.time()
        camera_properties = gymapi.CameraProperties()
        cam_width = self.image_width
        cam_height = self.image_height
        camera_properties.width = cam_width
        camera_properties.height = cam_height
        camera_properties.enable_tensors = True
        if is_first:
            self.num_camera = 1
        pi = math.pi
        circle_point = self.PointsInCircum(-self.camera_radius, self.num_camera, self.view_angle)

        for camera_xy in circle_point:
            camera_xy.append(self.camera_height)
            camera_xy.append(self.camera_height)
        top_camera_pos = [-0.5, -0.0, self.camera_height + self.camera_radius]
        circle_point.insert(0, top_camera_pos)
        # self.num_camera += 1
        cam_handles = []

        # Set a fixed position and look-target for the first camera
        # position and target location are in the coordinate frame of the environment

        if env_index not in self.camera_handles.keys():
            for i in range(self.num_camera + 1):
                camera_handle = self.gym.create_camera_sensor(self.envs[env_index], camera_properties)
                # Look at the env
                cam_pos = gymapi.Vec3(circle_point[i][0], circle_point[i][1], circle_point[i][2])
                cam_target = gymapi.Vec3(0.0, 0.0, 0.6)
                self.gym.set_camera_location(camera_handle, self.envs[env_index], cam_pos, cam_target)
                self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
                cam_handles.append(camera_handle)
            self.camera_handles[env_index] = cam_handles
        else:
            cam_handles = self.camera_handles[env_index]

        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.render_all_camera_sensors(self.sim)

        u_range = -torch.arange(-0.5, 0.5, 1 / cam_width).view(1, -1).cuda()
        v_range = torch.arange(-0.5, 0.5, 1 / cam_height).view(-1, 1).cuda()
        env_points = torch.Tensor().to(self.device)
        env_points_rgb = torch.Tensor().to(self.device)
        candidate_points = torch.Tensor().to(self.device)

        depth_imgs = []
        color_imgs = []
        view_matrices = []

        fu_s = []
        fv_s = []

        # TODO convert to GPU version
        print("Converting Depth images to point clouds. Have patience...")
        for i in range(self.num_camera + 1):
            # Retrieve depth and segmentation buffer
            vinv = np.linalg.inv(
                np.matrix(self.gym.get_camera_view_matrix(self.sim, self.envs[env_index], cam_handles[i]))
            )
            depth_buffer = self.gym.get_camera_image(self.sim, self.envs[env_index], cam_handles[i], gymapi.IMAGE_DEPTH)
            color_buffer = self.gym.get_camera_image(
                self.sim, self.envs[env_index], cam_handles[i], gymapi.IMAGE_COLOR
            ).reshape(cam_height, cam_width, -1)
            seg_buffer = self.gym.get_camera_image(
                self.sim, self.envs[env_index], cam_handles[i], gymapi.IMAGE_SEGMENTATION
            )
            # TODO check images
            # self.gym.write_camera_image_to_file(self.sim, self.envs[env_index], cam_handles[i], gymapi.IMAGE_DEPTH, "frame-%06d.depth.png" % (i))
            # self.gym.write_camera_image_to_file(self.sim, self.envs[env_index], cam_handles[i], gymapi.IMAGE_COLOR, "frame-%06d.color.png" % (i))

            if i == self.camera_view_id:
                rgbd = np.zeros((cam_width, cam_height, 4))
                rgbd[:, :, :3] = (color_buffer[:, :, :3] / 255).astype("float32")
                save_depth = depth_buffer.copy()
                save_depth[save_depth < -5] = 0
                save_depth = -save_depth
                rgbd[:, :, -1] = save_depth
                self.rgbd = np.transpose(rgbd, (2, 0, 1))

            depth_imgs.append(depth_buffer)
            color_imgs.append(color_buffer)
            view_matrices.append(
                np.matrix(self.gym.get_camera_view_matrix(self.sim, self.envs[env_index], cam_handles[i]))
            )

            # Get the camera projection matrix and get the necessary scaling
            # coefficients for deprojection
            proj = self.gym.get_camera_proj_matrix(self.sim, self.envs[env_index], cam_handles[i])
            # proj matrix
            # array([[ 1.,  0.,  0.,  0.],
            #        [ 0.,  1.,  0.,  0.],
            #        [ 0.,  0.,  0., -1.],
            #        [ 0.,  0.,  0.,  0.]], dtype=float32)

            # camera view matrix
            # matrix([[ 0.  ,  0.24, -0.97,  0.  ],
            #         [-1.  ,  0.  , -0.  ,  0.  ],
            #         [ 0.  ,  0.97,  0.24,  0.  ],
            #         [-0.  , -0.97, -2.3 ,  1.  ]], dtype=float32)

            fu = 2 / proj[0, 0]
            fv = 2 / proj[1, 1]

            fu_s.append(fu)
            fv_s.append(fv)

            # Ignore any points which originate from ground plane or empty space
            depth_buffer[seg_buffer == 0] = -10001

            # TODO Ignore points from robot
            depth_buffer[seg_buffer == 9] = -10001

            depth_img = to_torch(depth_buffer, device=self.device)

            # color_img = to_torch(color_buffer, device=self.device)
            # self.save_image_tensor_test(color_img, "test"+str(i)+".png")

            proj_u = fu * torch.mul(depth_img, u_range)
            proj_v = fv * torch.mul(depth_img, v_range)
            ones = torch.ones_like(depth_img)
            big_mat = torch.cat(
                (proj_u.unsqueeze(-1), proj_v.unsqueeze(-1), depth_img.unsqueeze(-1), ones.unsqueeze(-1)), dim=2
            )
            vinv = torch.from_numpy(vinv).to(self.device)
            max_depth = 20
            mask = depth_img > -max_depth

            pc_img = torch.matmul(big_mat, vinv)

            # TODO add color point cloud

            if i == self.camera_view_id:
                mask_img = mask.long().unsqueeze(-1).repeat(1, 1, 3)
                pc_coord = pc_img[:, :, :3]

                pc_img_valid = mask_img.long() * pc_coord
                self.coord = pc_img_valid.cpu().numpy()

                # tsdf_vol_bnds = np.array([[-1.5, 0.5], [-1., 1.], [0, 2]])
                # normalize the coordinate layer
                for j in range(3):
                    self.coord[:, :, j] = (self.coord[:, :, j] - self.tsdf_vol_bnds[j, 0]) / (
                            self.tsdf_vol_bnds[j, 1] - self.tsdf_vol_bnds[j, 0]
                    )

                self.coord = mask_img.long().cpu().numpy() * self.coord

            pc = pc_img[mask]
            pc_rgb_np = self.rgba2rgb(color_buffer)
            pc_rgb = to_torch(pc_rgb_np, device=self.device)[mask]
            pc_rgb = pc_rgb.float() / 255.0
            pc = pc[:, :3]
            env_points = torch.cat((env_points, pc))
            env_points_rgb = torch.cat((env_points_rgb, pc_rgb))

            """
            # inverse calculation
            vin = np.matrix(self.gym.get_camera_view_matrix(self.sim, self.envs[env_index], cam_handles[i]))
            vin = torch.from_numpy(vin).cuda()

            add_ones = torch.ones(env_points.shape[0]).unsqueeze(1).to(self.device)
            point_ones = torch.cat((env_points, add_ones), dim=1)
            big_mat_inv = torch.matmul(point_ones, vin)
            proj_u_inv = big_mat_inv[:, 0]
            proj_v_inv = big_mat_inv[:, 1]
            depth_img_inv = big_mat_inv[:, 2]

            u_range_inv = proj_u_inv / (fu * depth_img_inv)
            v_range_inv = proj_v_inv / (fv * depth_img_inv)

            u_inv = torch.round(cam_width * (-u_range_inv)) + cam_width * 0.5
            v_inv = torch.round(cam_height * v_range_inv) + cam_height * 0.5

            # inverse inverse calculation
            _proj_u = fu * depth_img_inv * u_range_inv
            _proj_v = fv * depth_img_inv * v_range_inv
            _ones = torch.ones_like(depth_img_inv)
            _big_mat = torch.cat(
                (_proj_u.unsqueeze(-1), _proj_v.unsqueeze(-1), depth_img_inv.unsqueeze(-1), _ones.unsqueeze(-1)), dim=1)
            max_depth = 20
            _mask = depth_img_inv > -max_depth
            _pc = torch.matmul(_big_mat, vinv)[_mask]
            _pc = _pc[:, :3]

            """

            if i == self.camera_view_id:
                rgb_filename = "rgb_env_cam%d.png" % (self.camera_id)
                # self.gym.write_camera_image_to_file(self.sim, self.envs[env_index], cam_handles[i], gymapi.IMAGE_COLOR, rgb_filename)
                candidate_points = torch.cat((candidate_points, pc))

        points = env_points.cpu().numpy()
        points_rgb = env_points_rgb.cpu().numpy()

        # TODO vis grasp
        # print("num_point: ", points.shape[0])
        # gg, cloud = predict_grasp(points, points_rgb, self.device)
        # vis_grasps(gg, cloud)
        # exit()

        pcd = o3d.open3d.geometry.PointCloud()
        pcd.points = o3d.open3d.utility.Vector3dVector(points)
        pcd.colors = o3d.open3d.utility.Vector3dVector(points_rgb)
        # o3d.visualization.draw_geometries([pcd])
        downpcd = pcd.voxel_down_sample(voxel_size=0.02)

        # bounding_box = downpcd.get_axis_aligned_bounding_box()
        self.pointcloud = np.asarray(downpcd.points)
        # o3d.visualization.draw_geometries([downpcd])

        # create sparse pointcloud used for checking to_pointcloud distacne
        sparse_voxel_size = 0.06
        if self.object_cate == "Switch":
            sparse_voxel_size = 0.02
        downpcd_sparse = pcd.voxel_down_sample(voxel_size=0.06)
        self.pointcloud_sparse = np.asarray(downpcd_sparse.points)

        # recompute candidate point with intersection of point cloud and workspace
        # candidate_points = candidate_points.cpu().numpy()
        # cpcd = o3d.open3d.geometry.PointCloud()
        # cpcd.points = o3d.open3d.utility.Vector3dVector(candidate_points)
        # downcpcd = cpcd.voxel_down_sample(voxel_size=0.02)
        # self.candidate_points = np.asarray(downcpcd.points)

        # franka base computation to set franka workspace
        franka_base_pos = self.get_franka_base_pos()

        # TODO using dense points instead of downsampling pointcloud

        # pointcloud_tensor = env_points
        # pointcloud_rgb_tensor = env_points_rgb

        franka_base_pos_tensor = torch.tensor(franka_base_pos).to(self.device)
        # pointcloud_franka_base_dist = torch.norm(pointcloud_tensor - franka_base_pos_tensor, dim=-1)

        # pc_not_too_far = pointcloud_franka_base_dist < self.workspace_sphere_R
        # pc_not_too_close = pointcloud_franka_base_dist > self.workspace_sphere_r
        # pc_inside_workspace = torch.logical_and(pc_not_too_close, pc_not_too_far)

        # self.graspnet_pc = pointcloud_tensor[pc_inside_workspace]
        # self.graspnet_pc_rgb = pointcloud_rgb_tensor[pc_inside_workspace]

        candidate_points_tensor = torch.tensor(self.pointcloud).to(self.device)
        pointcloud_franka_base_dist = torch.norm(candidate_points_tensor - franka_base_pos_tensor, dim=-1)

        pc_not_too_far = pointcloud_franka_base_dist < self.workspace_sphere_R
        pc_not_too_close = pointcloud_franka_base_dist > self.workspace_sphere_r
        pc_inside_workspace = torch.logical_and(pc_not_too_close, pc_not_too_far)

        self.candidate_points = candidate_points_tensor[pc_inside_workspace]

        # visualize graspnet points
        # gpcd = o3d.open3d.geometry.PointCloud()
        # gpcd.points = o3d.open3d.utility.Vector3dVector(self.graspnet_pc.to('cpu').numpy())
        # gpcd.colors = o3d.open3d.utility.Vector3dVector(self.graspnet_pc_rgb.to('cpu').numpy())
        # print("Recompute the normal of the downsampled point cloud")
        # computing the normals here does not make sense
        # pcd.estimate_normals(search_param=o3d.open3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50))
        # gdownpcd = pcd.voxel_down_sample(voxel_size=0.001)
        # print(np.asarray(gdownpcd.points).shape[0])
        # o3d.open3d.visualization.draw_geometries([gdownpcd])

        # visualize recomputed candidate points
        # cpcd = o3d.open3d.geometry.PointCloud()
        # cpcd.points = o3d.open3d.utility.Vector3dVector(self.candidate_points.to('cpu'))
        # cpcd.colors = o3d.open3d.utility.Vector3dVector(self.candidate_points_rgb.to('cpu'))
        # print("Recompute the normal of the downsampled point cloud")
        # computing the normals here does not make sense
        # cpcd.estimate_normals(search_param=o3d.open3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50))
        # o3d.open3d.visualization.draw_geometries([cpcd])

        # assume that the object is in valid position where robot can reach
        assert self.candidate_points.shape[0] > 0
        assert self.pointcloud.shape[0] > 0

        x_min = float(torch.min(self.candidate_points[:, 0]).cpu())
        x_max = float(torch.max(self.candidate_points[:, 0]).cpu())
        y_min = float(torch.min(self.candidate_points[:, 1]).cpu())
        y_max = float(torch.max(self.candidate_points[:, 1]).cpu())
        z_min = float(torch.min(self.candidate_points[:, 2]).cpu())
        z_max = float(torch.max(self.candidate_points[:, 2]).cpu())
        self.candidate_points_bounding_box = np.array([[x_min, x_max], [y_min, y_max], [z_min, z_max]])

        # print("Recompute the normal of the downsampled point cloud")
        # estimate_normals( downpcd, search_param=o3d.open3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50))

        x_min = np.min(self.pointcloud[:, 0])
        x_max = np.max(self.pointcloud[:, 0])
        y_min = np.min(self.pointcloud[:, 1])
        y_max = np.max(self.pointcloud[:, 1])
        z_min = np.min(self.pointcloud[:, 2])
        z_max = np.max(self.pointcloud[:, 2])

        # give 1.1 times to loose the bound
        self.action_bounding_box = np.array([[x_min, x_max], [y_min, y_max], [z_min, z_max]])
        print("Point Cloud Complete")
        end_time = time.time()
        print("Generating the point cloud takes: ", end_time - start_time)
        print("action_bounding box: ", self.action_bounding_box)

        npz_path = "./dataset/" + str(self.object_id) + ".npz"
        npz_path = Path(npz_path)

        if self.is_multi_pose:
            # TODO dof dimension too high, so the file name is too long
            npz_path = "./dataset/" + "/" + str(self.object_id) + "_" + self.multi_pose_str + ".npz"
        else:
            npz_path = "./dataset/" + "/" + str(self.object_id) + ".npz"
        npz_path = Path(npz_path)

        if is_metric or not npz_path.exists():
        # if True:
            print("Initializing voxel volume...")
            voxel_size = 0.01
            self.tsdf = TSDFVolume(self.tsdf_vol_bnds, voxel_size, self.device)
            for i in range(self.num_camera):
                # print("Fusing frame %d/%d" % (i + 1, self.num_camera))
                # Read RGB-D image and camera pose
                # Integrate observation into voxel volume (assume color aligned with depth)
                self.tsdf.integrate(color_imgs[i], depth_imgs[i], fu_s[i], fv_s[i], view_matrices[i], obs_weight=1.0)
            end_time2 = time.time()
            print("construct tsdf takes: ", end_time2 - end_time)


            # TODO visualize mesh to see reconstruction
            # verts, faces, norms, colors = self.tsdf.get_mesh()
            # meshwrite("mesh.ply", verts, faces, norms, colors)

        # destory camera sensor after each generation
        for cam_handle in cam_handles:
            self.gym.destroy_camera_sensor(self.sim, self.envs[env_index], cam_handle)

    def get_candidate_points_bounding_box(self):
        return self.candidate_points_bounding_box

    def get_candidate_points(self):
        return self.candidate_points

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_franka_reward(
            self.reset_buf,
            self.progress_buf,
            self.actions,
            self.franka_lfinger_pos,
            self.franka_lfinger_rot,
            self.franka_rfinger_pos,
            self.franka_rfinger_rot,
            self.hand_pos,
            self.hand_rot,
            self.down_dir,
            self.num_envs,
            self.reward_scale,
            self.action_penalty_scale,
            self.distX_offset,
            self.max_episode_length,
        )

    def compute_observations(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # self.gym.render_all_camera_sensors(self.sim)
        # self.gym.start_access_image_tensors(self.sim)

        # configuration space control
        self.franka_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, : self.num_franka_dofs]
        self.franka_dof_pos = self.franka_dof_state[..., 0]
        self.franka_dof_vel = self.franka_dof_state[..., 1]

        # object dof state
        self.object_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_franka_dofs:]
        self.object_dof_pos = self.object_dof_state[..., 0]
        self.object_dof_vel = self.object_dof_state[..., 1]

        # EEF control
        self.franka_lfinger_pos = self.rigid_body_states[:, self.lfinger_handle][:, 0:3]
        self.franka_rfinger_pos = self.rigid_body_states[:, self.rfinger_handle][:, 0:3]
        self.franka_lfinger_rot = self.rigid_body_states[:, self.lfinger_handle][:, 3:7]
        self.franka_rfinger_rot = self.rigid_body_states[:, self.rfinger_handle][:, 3:7]

        self.hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        self.hand_vel = self.rigid_body_states[:, self.hand_handle][:, 7:]

        # grasp_pos = (self.franka_lfinger_pos + self.franka_rfinger_pos) / 2.0
        # grasp_hand_dist = torch.norm(grasp_pos - self.hand_pos, dim=1)

        self.hand_rot = self.rigid_body_states[:, self.hand_handle][:, 3:7]
        dof_pos_scaled = (
                2.0
                * (self.franka_dof_pos - self.franka_dof_lower_limits)
                / (self.franka_dof_upper_limits - self.franka_dof_lower_limits)
                - 1.0
        )

        # TODO object observation compute the cornor
        # TODO current obs: (7dof+2gripper)*2 (pos+vel)(franka) + (3pos+4quat)(box) = 25
        self.obs_buf = torch.cat((dof_pos_scaled, self.franka_dof_vel * self.dof_vel_scale), dim=-1)

        return self.obs_buf

    def save_video_file(self, model_name, video_name):
        video_file = video_name + ".mp4"
        # frame = self.video_frames[0]
        # height, width, layers = frame.shape
        # plt.imsave("test.png", frame)
        self.video_frames.pop(0)

        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(self.video_frames, fps=20.0)
        clip.write_videofile(video_file)
        return self.video_frames

    def reset(self, env_ids):
        self.task_state = -1
        self.pre_task_state = -2
        self.reach_reward = None

        # reset object
        # reset object dof
        self.object_dof_state[env_ids, :, 1] = torch.zeros_like(self.object_dof_state[env_ids, :, 1])
        self.object_dof_state[env_ids, :, 0] = self.object_init_dof_pos.clone()

        # reset franka
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0), self.franka_dof_lower_limits, self.franka_dof_upper_limits
        )

        self.franka_dof_pos[env_ids, :] = pos
        self.franka_dof_vel[env_ids, :] = torch.zeros_like(self.franka_dof_vel[env_ids])
        self.franka_dof_targets[env_ids, : self.num_franka_dofs] = pos

        # self.root_state_tensor[self.franka_actor_idxs[env_ids]] = self.valid_init_state[env_ids].clone()
        self.root_state_tensor[self.object_actor_idxs[env_ids]] = self.object_actor_state[env_ids].clone()

        """
        # reset object actor
        object_indices = self.object_actor_idxs[env_ids].to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(object_indices), len(object_indices))
        """
        # reset franka actor
        franka_indices = self.franka_actor_idxs[env_ids].to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(franka_indices),
            len(franka_indices),
        )

        # reset franka dof
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))
        franka_indices = self.franka_actor_idxs.to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.franka_dof_targets),
            gymtorch.unwrap_tensor(franka_indices),
            len(franka_indices),
        )
        """
        # reset object dof
        self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.franka_dof_targets),
                                                        gymtorch.unwrap_tensor(object_indices), len(object_indices))
        """
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.obj_dof_change_str = []
        self.video_frames = []

        obs = self.compute_observations()

        # clear previous saved data
        self.rgba_tensor = torch.Tensor().to(self.device)
        self.depth_tensor = torch.Tensor().to(self.device)
        self.unreachable_penalty = None  # set unreachable penalty to be None

    def filter_action(self):

        hand_point_dist = torch.norm(self.hand_pos - self.select_points, dim=-1)
        threshold = 0.25
        self.reset_buf = torch.where(
            hand_point_dist > threshold, torch.ones_like(self.reset_buf), torch.zeros_like(self.reset_buf)
        )

        """
        # TODO earlier idea of using sensor to detect
        vec_sensor_force_sum = torch.sum(self.vec_sensor_tensor[:, :, :3], dim=(1, 2), keepdim=True).squeeze()
        force_threshold = 200
        mask = vec_sensor_force_sum > force_threshold
        self.reset_buf = torch.logical_or(torch.gt(self.reset_buf, 0), mask).long()
        """

    def draw_interaction_points(self):
        # if self.interaction_points is None:
        #     return

        for i in range(self.num_envs):
            interaction_pose = gymapi.Transform()
            interaction_pose.p.x = self.select_points[i, 0].cpu().float()
            interaction_pose.p.y = self.select_points[i, 1].cpu().float()
            interaction_pose.p.z = self.select_points[i, 2].cpu().float()
            interaction_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0)

            gymutil.draw_lines(self.sphere_interaction_point, self.gym, self.viewer, self.envs[i], interaction_pose)

        for i in range(self.num_envs):
            grasp_pose = gymapi.Transform()
            grasp_pose.p.x = self.interact_pos[i, 0].cpu().float()
            grasp_pose.p.y = self.interact_pos[i, 1].cpu().float()
            grasp_pose.p.z = self.interact_pos[i, 2].cpu().float()
            grasp_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0)

            gymutil.draw_lines(self.sphere_grasp_point, self.gym, self.viewer, self.envs[i], grasp_pose)

        for i in range(self.num_envs):
            grasp_pose = gymapi.Transform()
            grasp_pose.p.x = self.init_pos[i, 0].cpu().float()
            grasp_pose.p.y = self.init_pos[i, 1].cpu().float()
            grasp_pose.p.z = self.init_pos[i, 2].cpu().float()
            grasp_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0)

            gymutil.draw_lines(self.sphere_init_point, self.gym, self.viewer, self.envs[i], grasp_pose)

        for i in range(self.num_envs):
            grasp_pose = gymapi.Transform()
            grasp_pose.p.x = self.manipulate_pos[i, 0].cpu().float()
            grasp_pose.p.y = self.manipulate_pos[i, 1].cpu().float()
            grasp_pose.p.z = self.manipulate_pos[i, 2].cpu().float()
            grasp_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0)

            gymutil.draw_lines(self.sphere_manipulate_point, self.gym, self.viewer, self.envs[i], grasp_pose)

    def draw_workspace(self):
        for i in range(self.num_envs):
            # Draw sphere to visualize working space
            workspace_pose = gymapi.Transform()
            # Franka robot on the table
            workspace_pose.p = gymapi.Vec3(-1.8, 0.0, 0.0)
            workspace_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0)

            workspace_pose.p = self.franka_start_pose.p
            workspace_pose.r = self.franka_start_pose.r

            gymutil.draw_lines(self.sphere_work_geom_R, self.gym, self.viewer, self.envs[i], workspace_pose)
            gymutil.draw_lines(self.sphere_work_geom_r, self.gym, self.viewer, self.envs[i], workspace_pose)

            # tsdf_vol_bnds = np.array([[-1.2, 0.4], [-0.8, 0.8], [0, 1.4]])

            for j in range(8):
                binary_j = f'{j:03b}'
                x_bin = int(binary_j[0])
                y_bin = int(binary_j[1])
                z_bin = int(binary_j[2])

                voxel_bnd_pose = gymapi.Transform()
                voxel_bnd_pose.p.x = self.tsdf_vol_bnds[0, x_bin]
                voxel_bnd_pose.p.y = self.tsdf_vol_bnds[1, y_bin]
                voxel_bnd_pose.p.z = self.tsdf_vol_bnds[2, z_bin]
                voxel_bnd_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0)

                gymutil.draw_lines(self.sphere_voxel_bnd, self.gym, self.viewer, self.envs[i], voxel_bnd_pose)

    def update_attractor(self, next_attractor_state):
        # TODO first iteration, attractor is not working! and I don't know why!

        for i in range(self.num_envs):
            attractor_properties = self.gym.get_attractor_properties(self.envs[i], self.attractor_handles[i])
            pose = attractor_properties.target

            current_pos = self.hand_pos[i]
            next_attractor_pos = next_attractor_state[i, :3]
            dist = (torch.norm(next_attractor_pos - current_pos))
            direction = (next_attractor_pos - current_pos) / dist

            if self.task_state == 0:
                attractor_step_dist = 0.03
            else:
                attractor_step_dist = 0.012

            smooth_next_attractor = current_pos + direction * attractor_step_dist

            if dist.item() <= attractor_step_dist:
                pose.p.x = next_attractor_state[i, 0]
                pose.p.y = next_attractor_state[i, 1]
                pose.p.z = next_attractor_state[i, 2]

            else:
                pose.p.x = smooth_next_attractor[0]
                pose.p.y = smooth_next_attractor[1]
                pose.p.z = smooth_next_attractor[2]

            if next_attractor_state.shape[-1] > 3:
                pose.r.x = next_attractor_state[i, 3]
                pose.r.y = next_attractor_state[i, 4]
                pose.r.z = next_attractor_state[i, 5]
                pose.r.w = next_attractor_state[i, 6]

            self.gym.set_attractor_target(self.envs[i], self.attractor_handles[i], pose)

            # Draw axes and sphere at attractor location
            # gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, self.envs[i], pose)
            # gymutil.draw_lines(self.sphere_geom, self.gym, self.viewer, self.envs[i], pose)

    def compute_franka_unreachable_penalty(self):
        pre_target_pos = self.pre_target_pose[:, :3]
        pre_target_quat = self.pre_target_pose[:, 3:]

        reach_pos_dist = torch.norm(self.hand_pos - pre_target_pos, dim=-1)
        # reach_quat_dist = 2 * (1 - torch.dot(self.hand_rot, pre_target_quat))

        if self.unreachable_penalty is None:
            self.unreachable_penalty = torch.zeros_like(reach_pos_dist).to(self.device)

        if self.task_state < 3:
            reach_threshold = self.eef_hand_dist / 8
        else:
            reach_threshold = self.eef_hand_dist / 2
        # self.unreachable_penalty -= reach_pos_dist
        self.unreachable_penalty -= torch.where(reach_pos_dist > reach_threshold,
                                                torch.ones_like(reach_pos_dist) * 10,
                                                torch.zeros_like(reach_pos_dist)).float()
        print("current valid num: ", int((self.unreachable_penalty >= 0).sum().cpu()))

    def compute_franka_reach_reward(self):
        franka_eef_pos = (self.franka_lfinger_pos + self.franka_rfinger_pos) * 0.5

        num_pc = self.candidate_points.shape[0]
        franka_eef_pos = franka_eef_pos.unsqueeze(1).repeat(1, num_pc, 1)
        eef_pointcloud_dist = torch.norm(franka_eef_pos - self.candidate_points, dim=-1)
        eef_pointcloud_min_dist = torch.min(eef_pointcloud_dist, dim=-1)[0]
        # self.reach_reward = 1 / (eef_pointcloud_min_dist + 1)
        self.reach_reward = -eef_pointcloud_min_dist.float()

        print("compute franka reach reward")

    def draw_attractor(self, attractor_pos, attractor_quat):
        for i in range(self.num_envs):
            # Draw axes and sphere at attractor location
            # gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, self.envs[i], attractor_pose)

            vis_attractor_pose = gymapi.Transform()
            # Franka robot on the table
            vis_attractor_pose.p.x = attractor_pos[i, 0].cpu().float()
            vis_attractor_pose.p.y = attractor_pos[i, 1].cpu().float()
            vis_attractor_pose.p.z = attractor_pos[i, 2].cpu().float()

            vis_attractor_pose.r.x = attractor_quat[i, 0].cpu().float()
            vis_attractor_pose.r.y = attractor_quat[i, 1].cpu().float()
            vis_attractor_pose.r.z = attractor_quat[i, 2].cpu().float()
            vis_attractor_pose.r.w = attractor_quat[i, 3].cpu().float()

            gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, self.envs[i], vis_attractor_pose)
            gymutil.draw_lines(self.sphere_geom, self.gym, self.viewer, self.envs[i], vis_attractor_pose)

    def pre_physics_step(self, actions):
        # actions with size (num_envs, action_dim)
        # action_dim = 6dof + 1 gripper indicator

        self.gym.clear_lines(self.viewer)

        # actions could either be euler angle or quaternion
        assert (torch.is_tensor(actions)
                and ((actions.shape == torch.zeros(self.num_envs, 7).shape) or actions.shape == torch.zeros(
                    self.num_envs, 8).shape)) \
               or actions == -1

        # use to initialize environment for vision
        if isinstance(actions, int):
            return

        # Compute reach reward here
        # TODO this is used for CEM
        # if self.reach_reward is None and self.task_state == 2 and self.pre_task_state == 1:
        #     self.compute_franka_reach_reward()

        # Compute not reachable error
        if self.pre_task_state >= 0 and self.pre_task_state != 1 and self.local_step == 0:
            # if self.pre_task_state >= 0 and self.local_step == 0:
            print("compute unreachable penalty at current stage: ", self.task_state)
            self.compute_franka_unreachable_penalty()

        # Assume actions is inputed as tensor with shape (num_envs, action_dim)
        actions = actions.to(self.device)
        gripper_indicator = actions[:, -1]

        # TODO gripper 1 for close, 0 for open
        # franka_gripper_joint = torch.where(grippper_indicator > 0)
        gripper_open_close = torch.where(gripper_indicator > 0, 0, 1)
        gripper_open_close = gripper_open_close[:, None].repeat((1, 2)).to(self.device)

        self.franka_dof_targets[:, 7:9] = gripper_open_close * 0.04

        attractor_pos = actions[:, :3]

        if actions.shape[1] == 7:
            # Euler agnle
            attractor_rot = actions[:, 3:6]
            # TODO figure out how axis aligned
            # attractor_rot = torch.zeros_like(attractor_rot).to(self.device)
            # attractor_rot[:, 1] = np.pi / 2.0
            # attractor_rot[:, 0] = np.pi / 2.0
            # attractor_quat = tgm.angle_axis_to_quaternion(attractor_rot)
            attractor_quat = pytorch3d.transforms.matrix_to_quaternion(
                pytorch3d.transforms.euler_angles_to_matrix(attractor_rot, "XYZ"))

        else:
            # Quanternion
            attractor_quat = actions[:, 3:7]

        # TODO if testing, transfer to hand pos
        # init_pos_y = (
        #                 init_pos + quat_apply(candidate_rotate, y_axis * (self.eef_hand_dist)) + z_bias
        #         )


        franka_eef_pos = (self.franka_lfinger_pos + self.franka_rfinger_pos) * 0.5
        # franka hand eef dist
        eef_dist = torch.norm(self.hand_pos - franka_eef_pos, dim=-1)
        eef_dist = eef_dist[0].item()

        # if attractor_pos.shape[0] == 1:
            # Testing if only one env, transfer eef pos to hand pos
        z_axis = to_torch([0., 0., -1.], device=self.device).repeat((self.num_envs, 1))
        # z_axis = z_axis.type(torch.float32)

        attractor_pos = attractor_pos + quat_apply(attractor_quat.float(), z_axis * eef_dist)

        attractor_pose = torch.cat((attractor_pos, attractor_quat), dim=1)

        # self.draw_attractor(attractor_pos, attractor_quat)

        attractor_velocity = torch.zeros((self.num_envs, 6)).to(self.device)
        attractor_pose_vel = torch.cat((attractor_pose, attractor_velocity), dim=1)

        '''
        if self.task_state is not None and self.task_state == 0:
            print((attractor_pose_vel[:, :3] == self.init_pos).sum())

        if self.task_state is not None and self.task_state == 1:
            print((attractor_pose_vel[:, :3] == self.interact_pos).sum())

        if self.task_state is not None and self.task_state == 2:
            print((attractor_pose_vel[:, :3] == self.interact_pos).sum())

        if self.task_state is not None and self.task_state == 3:
            print((attractor_pose_vel[:, :3] == self.manipulate_pos).sum())
        
        '''
        # use ik to update franka
        if self.controller == "ik":
            control_u = self.control_ik(attractor_pose)
            self.franka_dof_targets[:, :7] = self.franka_dof_pos[:, :7] + control_u
        else:
            control_u = self.control_osc(attractor_pose)
            effort_action = torch.zeros_like(self.dof_state.view(self.num_envs, -1, 2)[..., 0])
            effort_action[:, :7] = control_u
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(effort_action))

        # TODO attractor does not work on new isaacgym gpu pipeline
        # update attractor
        # self.update_attractor(attractor_pose_vel)


        # visualize point and space
        # self.draw_workspace()
        # self.draw_interaction_points()

        # print(self.franka_dof_targets)
        # print(self.franka_dof_pos[:, :7])

        # TODO setup the 6dof control for robot
        # self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.franka_dof_targets))

        franka_indices = self.franka_actor_idxs.to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.franka_dof_targets),
            gymtorch.unwrap_tensor(franka_indices),
            len(franka_indices),
        )


        if self.task_state is not None and self.task_state < 3:
            self.lock_object_dof()

        # save previous step target position
        self.pre_target_pose = attractor_pose

        return

        # TODO need to rethink about the filter
        """
        self.actions[torch.gt(self.reset_buf, 0)] = torch.tensor([0, 0, 0, -1.6, 0, 1.9, 0, 0.0, 0.0]).to(self.device)

        action_mask = self.actions - self.franka_dof_targets[:, :self.num_franka_dofs]
        action_mask[action_mask > 0] = 1
        action_mask[action_mask < 0] = -1

        action_origin = self.actions - self.franka_dof_targets[:, :self.num_franka_dofs]
        action_unit = self.franka_dof_speed_scales * self.dt * action_mask * 0.5

        action_step = torch.where(torch.abs(action_origin) < torch.abs(action_unit), action_origin, action_unit)
        targets = self.franka_dof_targets[:, :self.num_franka_dofs] + action_step
        """

        targets = self.actions
        self.franka_dof_targets[:, : self.num_franka_dofs] = tensor_clamp(
            targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits
        )
        franka_indices = self.franka_actor_idxs.to(torch.int32)

        # self.franka_dof_targets[:, 7:9] = 0.0

        # TODO do not reset
        if actions < 3:
            self.object_dof_state[:, :, 1] = torch.zeros_like(self.object_dof_state[:, :, 1])
            self.object_dof_state[:, :, 0] = self.object_init_dof_pos
            object_indices = self.object_actor_idxs.to(torch.int32)
            # self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),gymtorch.unwrap_tensor(object_indices), len(object_indices))
            self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))

        """
        # self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))

        t = self.gym.get_sim_time(self.sim)
        for i in range(self.num_envs):
            attractor_properties = self.gym.get_attractor_properties(self.envs[i], self.attractor_handles[i])
            pose = attractor_properties.target

            pose.p.x = 0.3 * math.sin(1.5 * t - math.pi * float(i)) - 1.4
            pose.p.y = -0.1 * math.cos(1.5 * t - math.pi * float(i))
            pose.p.z = 0.55 + 0.1 * math.cos(2.5 * t - math.pi * float(i))

            self.gym.set_attractor_target(self.envs[i], self.attractor_handles[i], pose)

            # Draw axes and sphere at attractor location
            gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, self.envs[i], pose)
            gymutil.draw_lines(self.sphere_geom, self.gym, self.viewer, self.envs[i], pose)
        """

        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.franka_dof_targets),
            gymtorch.unwrap_tensor(franka_indices),
            len(franka_indices),
        )

    def control_ik(self, goal_pose):
        goal_pos = goal_pose[:, :3]
        goal_rot = goal_pose[:, 3:]

        # compute position and orientation error
        pos_err = goal_pos - self.hand_pos
        orn_err = orientation_error(goal_rot, self.hand_rot)
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
        # print(dpose[0])

        damping = 0.1
        # global damping, j_eef, num_envs
        # solve damped least squares
        j_eef_T = torch.transpose(self.j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (damping ** 2)
        u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 7)
        return u

    def control_osc(self, goal_pose):
        goal_pos = goal_pose[:, :3]
        goal_rot = goal_pose[:, 3:]

        # compute position and orientation error
        pos_err = goal_pos - self.hand_pos
        orn_err = orientation_error(goal_rot, self.hand_rot)
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)

        # OSC params
        kp = 150.
        kd = 2.0 * np.sqrt(kp)
        kp_null = 10.
        kd_null = 2.0 * np.sqrt(kp_null)

        # global default_dof_pos_tensor

        dof_pos = self.franka_dof_pos.unsqueeze(-1)
        dof_vel = self.franka_dof_vel.unsqueeze(-1)
        mm = self.mm[:, :7, :7]  # only need elements corresponding to the franka arm
        mm_inv = torch.inverse(mm)
        m_eef_inv = self.j_eef @ mm_inv @ torch.transpose(self.j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)
        u = torch.transpose(self.j_eef, 1, 2) @ m_eef @ (kp * dpose - kd * self.hand_vel.unsqueeze(-1)).float()

        # Nullspace control torques `u_null` prevents large changes in joint configuration
        # They are added into the nullspace of OSC so that the end effector orientation remains constant
        # roboticsproceedings.org/rss07/p31.pdf
        j_eef_inv = m_eef @ self.j_eef @ mm_inv
        u_null = kd_null * -dof_vel + kp_null * (
                (self.default_franka_dof_pos_tensor.view(1, -1, 1) - dof_pos + np.pi) % (2 * np.pi) - np.pi)
        u_null = u_null[:, :7]
        u_null = mm @ u_null

        u += (torch.eye(7, device=self.device).unsqueeze(0) - torch.transpose(self.j_eef, 1, 2) @ j_eef_inv) @ u_null
        return u.squeeze(-1)

    def lock_object_dof(self):
        self.object_dof_state[:, :, 1] = torch.zeros_like(self.object_dof_state[:, :, 1])
        self.object_dof_state[:, :, 0] = self.object_init_dof_pos
        object_indices = self.object_actor_idxs.to(torch.int32)
        # self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),gymtorch.unwrap_tensor(object_indices), len(object_indices))
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))

        # TODO should fix the object dof here
        for env_idx in range(self.num_envs):
            object_actor_handle = self.gym.get_actor_handle(self.envs[env_idx], self.object_actor_idxs[env_idx])
            actor_dof_props = self.gym.get_actor_dof_properties(self.envs[env_idx], object_actor_handle)

    def quat_axis(self, q, axis=0):
        basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
        basis_vec[:, axis] = 1
        return quat_rotate(q, basis_vec)

    def orientation_error(self, desired, current):
        cc = quat_conjugate(current)
        q_r = quat_mul(desired, cc)
        return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

    def post_physics_step(self):
        self.progress_buf += 1

        # env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # if len(env_ids) > 0:
        #     self.reset(env_ids)

        self.compute_observations()
        # TODO do not compute reward

        if self.task_state is not None and self.task_state < 3:
            self.lock_object_dof()

        if self.save_video:
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)

            color_frame = self.gym.get_camera_image(
                self.sim, self.envs[0], self.video_camera_handle, gymapi.IMAGE_COLOR
            ).reshape(self.resolution_height, self.resolution_width, -1)
            self.video_frames.append(color_frame)

        # self.compute_reward(self.actions)

        # if self.dof_force[:, self.num_franka_dofs:].sum().cpu().numpy() > 300:
        #     print("contact___________", self.dof_force)

        # print(self.vec_sensor_tensor[:, :, :3].sum().cpu().numpy())
        # print("force sensor object: ", self.vec_sensor_tensor.shape)

        if True:
            franka_rot = self.root_state_tensor[self.franka_actor_idxs, 3:7]

        # debug viz
        self.debug_viz = False
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            for i in range(self.num_envs):
                px = self.hand_pos[i, :3].cpu().numpy()
                p0 = self.select_points[i, :3].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 4, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])

                px = np.array([0, 0, 2])
                p0 = np.array([0, 0, 2.2])
                self.gym.add_lines(self.viewer, self.envs[i], 4, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 1, 0])

                """
                # px = self.root_state_tensor[self.franka_actor_idxs[:]][i].cpu().numpy()
                # p0 = self.interact_pos[i].cpu().numpy()
                # self.gym.add_lines(self.viewer, self.envs[i], 4, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])

                px = self.valid_init_state[i, :3].cpu().numpy()
                p0 = self.select_points[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 4, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0, 1, 0])

                y_axis = to_torch([0, 1, 0], device=self.device)
                init_pos_y = px + quat_apply(self.hand_rot[i], y_axis * 0.2).cpu().numpy()
                init_pos_y_ = px + quat_apply(self.hand_rot[i], y_axis * (-0.2)).cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 9,
                                   [px[0], px[1], px[2], init_pos_y[0], init_pos_y[1], init_pos_y[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 9,
                                   [px[0], px[1], px[2], init_pos_y_[0], init_pos_y_[1], init_pos_y_[2]], [1, 0, 0])

                x_axis = to_torch([1, 0, 0], device=self.device)
                init_pos_x = px + quat_apply(self.hand_rot[i], x_axis * 0.2).cpu().numpy()
                init_pos_x_ = px + quat_apply(self.hand_rot[i], x_axis * -0.2).cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 9,
                                   [px[0], px[1], px[2], init_pos_x[0], init_pos_x[1], init_pos_x[2]], [1, 0, 1])
                self.gym.add_lines(self.viewer, self.envs[i], 9,
                                   [px[0], px[1], px[2], init_pos_x_[0], init_pos_x_[1], init_pos_x_[2]], [1, 0, 1])

                z_axis = to_torch([0, 0, 1], device=self.device)
                init_pos_z = px + quat_apply(self.hand_rot[i], z_axis * 0.2).cpu().numpy()
                init_pos_z_ = px + quat_apply(self.hand_rot[i], z_axis * -0.2).cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 9,
                                   [px[0], px[1], px[2], init_pos_z[0], init_pos_z[1], init_pos_z[2]], [0, 1, 1])
                self.gym.add_lines(self.viewer, self.envs[i], 9,
                                   [px[0], px[1], px[2], init_pos_z_[0], init_pos_z_[1], init_pos_z_[2]], [0, 1, 1])

                px = (self.hand_pos[i] + quat_apply(self.hand_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.hand_pos[i] + quat_apply(self.hand_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.hand_pos[i] + quat_apply(self.hand_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.hand_pos[i].cpu().numpy()
                # self.gym.add_lines(self.viewer, self.envs[i], 4, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 4, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                # self.gym.add_lines(self.viewer, self.envs[i], 4, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])



                px = (self.hammer_pos[i] + quat_apply(self.hammer_rot[i],
                                                      to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.hammer_pos[i] + quat_apply(self.hammer_rot[i],
                                                      to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.hammer_pos[i] + quat_apply(self.hammer_rot[i],
                                                      to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.hammer_pos[i].cpu().numpy()

                p_new = px + py + pz

                self.gym.add_lines(self.viewer, self.envs[i], 9, [p0[0], p0[1], p0[2], p_new[0], p_new[1], p_new[2]],
                                   [0.85, 0.1, 0.1])

                px = (self.peg_pos[i] + quat_apply(self.peg_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.peg_pos[i] + quat_apply(self.peg_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.peg_pos[i] + quat_apply(self.peg_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.peg_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 4, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 4, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 4, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])


                px = (self.hammer_pos[i] + quat_apply(self.hammer_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.hammer_pos[i] + quat_apply(self.hammer_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.hammer_pos[i] + quat_apply(self.hammer_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.hammer_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 4, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 4, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 4, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

                px = (self.franka_lfinger_pos[i] + quat_apply(self.franka_lfinger_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.franka_lfinger_pos[i] + quat_apply(self.franka_lfinger_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.franka_lfinger_pos[i] + quat_apply(self.franka_lfinger_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.franka_lfinger_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 4, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 4, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 4, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                px = (self.franka_rfinger_pos[i] + quat_apply(self.franka_rfinger_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.franka_rfinger_pos[i] + quat_apply(self.franka_rfinger_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.franka_rfinger_pos[i] + quat_apply(self.franka_rfinger_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.franka_rfinger_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 4, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 4, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 4, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])
                """

    # add reward function for goal conditional sampling
    def compute_state_reward(self, goal_obs=None, metric_model=None):
        self.compute_observations()
        # if self.curr_obs is None:
        #     print("need current depth obs!")
        #     exit()
        # goal_dist = -torch.norm(metric_model.encode(goal_obs) - metric_model.encode(self.curr_obs), dim=-1)
        # goal_dist = torch.norm(self.object_dof_pos - self.object_init_dof_pos, dim=-1)
        goal_dist = -self.object_dof_pos[:, 0] + self.object_init_dof_pos[:, 0]
        goal_dist = torch.abs(-self.object_dof_pos + self.object_init_dof_pos)
        goal_dist, index = torch.min(goal_dist, dim=1)

        # goal_dist = torch.where(goal_dist > 0.03, torch.ones_like(goal_dist), torch.zeros_like(goal_dist))

        # TODO adding rotation loss
        # TODO select from pointcloud only
        # TODO 2D filter to weight edge or cornors

        if self.reach_reward is not None:
            total_reward = goal_dist * 200 + self.reach_reward
        else:
            total_reward = goal_dist * 200

        # unreachable_penalty_scale = torch.where(self.unreachable_penalty < -0.0,
        #                                         torch.ones_like(self.unreachable_penalty) * -100,
        #                                         torch.zeros_like(self.unreachable_penalty))

        # total_reward += unreachable_penalty_scale
        total_reward = torch.where(self.unreachable_penalty.float() < -0.0,
                                   total_reward, torch.ones_like(self.unreachable_penalty.float()) * -1)
        # return self.reach_reward
        return total_reward


#####################################################################
###=========================jit functions=========================###
#####################################################################

def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def compute_franka_reward(
        reset_buf,
        progress_buf,
        actions,
        franka_lfinger_pos,
        franka_lfinger_rot,
        franka_rfinger_pos,
        franka_rfinger_rot,
        hand_pos,
        hand_rot,
        down_dir,
        num_envs,
        reward_scale,
        action_penalty_scale,
        distX_offset,
        max_episode_length,
):
    # regularization on the actions (summed for each environment)
    # action_penalty = torch.sum(actions ** 2, dim=-1)
    # rewards = -action_penalty_scale * action_penalty
    rewards = 0

    """
    reset_buf = torch.where(franka_lfinger_pos[:, 0] < box_pose[:, 0] - distX_offset,
                            torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(franka_rfinger_pos[:, 0] < box_pose[:, 0] - distX_offset,
                            torch.ones_like(reset_buf), reset_buf)
    """
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))


def plot_gmm(gmm, X, gt_labels, label=True, ax=None):
    ax = ax or plt.gca()
    X = umap.UMAP(n_neighbors=3, min_dist=0.1, metric="cosine").fit_transform(X)

    gt_color = []
    non_color = []
    for i in range(gt_labels.shape[0]):
        if gt_labels[i] == 0:
            gt_color.append("yellow")
        elif gt_labels[i] == 1:
            gt_color.append("blue")
        elif gt_labels[i] == 2:
            gt_color.append("red")
        elif gt_labels[i] == 10:
            gt_color.append("green")
        elif gt_labels[i] == 20:
            gt_color.append("black")
        elif gt_labels[i] == 11:
            gt_color.append("magenta")
        non_color.append("red")

    labels = gmm.fit(X).predict(X)

    if label:
        ax.scatter(X[:, 0], X[:, 1], c=gt_color, s=20, cmap="viridis", zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis("equal")

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)
    plt.show()


def plot_kmeans(kmeans, X, n_clusters=4, rseed=0, ax=None):
    labels = kmeans.fit_predict(X)

    # plot the input data
    ax = ax or plt.gca()
    ax.axis("equal")
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap="viridis", zorder=2)

    # plot the representation of the KMeans model
    centers = kmeans.cluster_centers_
    radii = [cdist(X[labels == i], [center]).max() for i, center in enumerate(centers)]
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle(c, r, fc="#CCCCCC", lw=3, alpha=0.5, zorder=1))
    plt.show()


def random_string(string_length=6):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4())  # Convert UUID format to a Python string.
    random = random.upper()  # Make all characters uppercase.
    random = random.replace("-", "")  # Remove the UUID '-'.
    return random[0:string_length]  # Return the random string.


def compute_entropy(obj_dof_change_str):
    num_success = 0
    dof_dict = {}
    for dof in obj_dof_change_str:
        if '1' in dof or '2' in dof:
            num_success += 1
            if dof in dof_dict.keys():
                dof_dict[dof] += 1
            else:
                dof_dict[dof] = 1
    prob = []
    for key in dof_dict.keys():
        prob.append(dof_dict[key] / num_success)
    print("entropy: ", entropy(prob))
    return entropy(prob)


def compute_total_entropy(obj_dof_change_str):
    num_success = 0
    dof_dict = {}
    for dof in obj_dof_change_str:
        num_success += 1
        if dof in dof_dict.keys():
            dof_dict[dof] += 1
        else:
            dof_dict[dof] = 1
    prob = []
    for key in dof_dict.keys():
        prob.append(dof_dict[key] / num_success)
    print("total entropy: ", entropy(prob))


def compute_max_mode(mode_str):
    total_mode = 0
    for i in range(len(mode_str)):
        index = mode_str[i]
        if index == '0' or index == '2':
            total_mode += 1
        else:
            total_mode += 2
    return total_mode


def compute_max_entropy(max_mode):
    prob = []
    for i in range(max_mode):
        prob.append(1 / max_mode)
    print("max entropy: ", entropy(prob))
    return entropy(prob)


def compute_largest_xyz_bnd(pc):
    x_min = pc[:, 0].min()
    x_max = pc[:, 0].max()
    y_min = pc[:, 1].min()
    y_max = pc[:, 1].max()
    z_min = pc[:, 2].min()
    z_max = pc[:, 2].max()

    x_len = x_max - x_min
    y_len = y_max - y_min
    z_len = z_max - z_min

    return max(x_len, y_len, z_len)


def display_inlier_outlier(cloud, mask):
    inlier_cloud = cloud.select_by_index(mask)
    outlier_cloud = cloud.select_by_index(mask, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud = outlier_cloud.paint_uniform_color([1.0, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    inlier_cloud = o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                                     zoom=0.3412,
                                                     front=[0.4257, -0.2125, -0.8795],
                                                     lookat=[2.6172, 2.0475, 1.532],
                                                     up=[-0.0694, -0.9768, 0.2024])
