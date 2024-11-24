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
import math
import open3d as o3d
import math
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import collections
import time
import pickle
import uuid
import moviepy.video.io.ImageSequenceClip
import random

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

# from affordance.vision.perception import *
from affordance.vision.fusion import *
import umap

import pdb


class FrankaRobotObject(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, graphics_device, device):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.graphics_device = graphics_device
        self.device = device

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
        self.is_multi_pose = False
        self.multi_pose_str = ""

        # distance from the eef point to gripper center point
        # depends on the gripper, 0.0584 is standard for franka gripper
        self.eef_hand_dist = 0.0584
        # distance from interaction point to initial position
        self.init_dist = 0.3

        # prop dimensions
        self.prop_width = 0.08
        self.prop_height = 0.08
        self.prop_length = 0.08
        self.prop_spacing = 0.09
        self.video_camera_handle = None


        num_obs = 18
        num_acts = 9

        # the bound of object
        self.tsdf_vol_bnds = np.array([[-1.5, 0.5], [-1.0, 1.0], [0, 2]])

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

        self.select_points = torch.Tensor().to(self.device)
        self.mask = None
        self.tsdf = None
        self.local_step = None
        self.video_frames = []

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

        # create some wrapper tensors for different slices
        # self.franka_default_dof_pos = to_torch([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.04], device=self.device)
        self.franka_default_dof_pos = to_torch([0, 0, 0, -0.5, 0, 1.0, 0, 0.04, 0.04], device=self.device)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.franka_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, : self.num_franka_dofs]
        self.franka_dof_pos = self.franka_dof_state[..., 0]
        self.franka_dof_vel = self.franka_dof_state[..., 1]

        self.dof_force = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, -1)
        self.contact = gymtorch.wrap_tensor(contact_tensor).view(self.num_envs, -1, 3)

        self.object_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_franka_dofs :]
        self.object_dof_pos = self.object_dof_state[..., 0]
        self.object_dof_vel = self.object_dof_state[..., 1]

        # TODO for visualization
        self.interact_pos = torch.zeros_like(self.franka_dof_pos[:, :3])
        self.select_points = torch.zeros_like(self.franka_dof_pos[:, :3])

        self.global_indices = torch.arange(self.num_envs * 2, dtype=torch.int32, device=self.device).view(
            self.num_envs, -1
        )
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.object_init_dof_pos = (
            to_torch(self.object_dof_lower_limits, device=self.device) * 0.5
            + to_torch(self.object_dof_upper_limits, device=self.device) * 0.5
        ).repeat((self.num_envs, 1))

        self.object_dof_num = self.object_init_dof_pos.shape[-1]

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
        print("actor_root_state_tensor shape: ", self.root_state_tensor.shape)
        print("rigid_body_tensor shape: ", self.rigid_body_states.shape)
        print("dof_state_tensor shape: ", self.dof_state.shape)
        print("root_state_tensor shape: ", self.root_state_tensor.shape)
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

    def set_local_step(self, step):
        self.local_step = step

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


    def multi_pose_init(self, is_multi_pose):
        self.is_multi_pose = is_multi_pose
        if self.is_multi_pose:
            object_init_dof_pos_rand = to_torch(self.object_dof_lower_limits, device=self.device).clone()
            multi_pose_index = to_torch(
                np.random.choice(self.init_pose_num, self.object_dof_num, replace=True), device=self.device
            )
            for i in range(multi_pose_index.shape[0]):
                self.multi_pose_str += str(int(multi_pose_index[i].detach().cpu().numpy()))

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

    def init_vision(self, is_first, is_metric=False, obs="tsdf", modes=False):
        # call viewer camera to construct point cloud for sampling
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
        print("--------------------------Finish Initialize")

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
        # asset_options.collapse_fixed_joints = True
        # asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        # get link index of panda hand, which we will use as end effector
        franka_link_dict = self.gym.get_asset_rigid_body_dict(franka_asset)
        self.franka_hand_index = franka_link_dict["panda_hand"]

        franka_dof_stiffness = to_torch(
            [4000, 4000, 4000, 4000, 4000, 4000, 4000, 1.0e6, 1.0e6], dtype=torch.float, device=self.device
        )
        franka_dof_damping = to_torch([80, 80, 80, 80, 80, 80, 80, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)

        # set franka dof properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        for i in range(self.num_franka_dofs):
            franka_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props["stiffness"][i] = 1000
                franka_dof_props["damping"][i] = 150
            else:
                franka_dof_props["stiffness"][i] = 7000.0
                franka_dof_props["damping"][i] = 50.0

            self.franka_dof_lower_limits.append(franka_dof_props["lower"][i])
            self.franka_dof_upper_limits.append(franka_dof_props["upper"][i])
        print("franka_dof_lower_limits: ", self.franka_dof_lower_limits)
        print("franka_dof_upper_limits: ", self.franka_dof_upper_limits)

        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[[6, 7]] = 0.4
        franka_dof_props["effort"][6] = 2000
        franka_dof_props["effort"][7] = 2000

        franka_start_pose = gymapi.Transform()
        # Franka robot on the table
        franka_start_pose.p = gymapi.Vec3(-1.2, 0.0, 0.0)
        franka_start_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0)

        self.franka_start_pose = franka_start_pose

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

        # create object asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        # asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.armature = 0.005
        asset_options.thickness = 0.001
        object_asset = self.gym.load_asset(
            self.sim, os.path.join(self.dataset_path, str(self.object_id)), "mobility_vhacd.urdf", asset_options
        )

        self.num_object_bodies = self.gym.get_asset_rigid_body_count(object_asset)
        self.num_object_dofs = self.gym.get_asset_dof_count(object_asset)

        # set object dof properties
        object_dof_props = self.gym.get_asset_dof_properties(object_asset)
        self.object_dof_lower_limits = []
        self.object_dof_upper_limits = []
        for i in range(self.num_object_dofs):
            object_dof_props["driveMode"][i] = gymapi.DOF_MODE_NONE
            object_dof_props["stiffness"][i] = 0
            object_dof_props["damping"][i] = 100

            self.object_dof_lower_limits.append(object_dof_props["lower"][i])
            self.object_dof_upper_limits.append(object_dof_props["upper"][i])

        print("object_dof_lower_limits: ", self.object_dof_lower_limits)
        print("object_dof_upper_limits: ", self.object_dof_upper_limits)
        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)
        print("num object bodies: ", self.num_object_bodies)
        print("num object dofs: ", self.num_object_dofs)

        object_pose = gymapi.Transform()
        object_pose.p = gymapi.Vec3(0.0, 0.0, 1.0)

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
        cam_width = 320
        cam_height = 320
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

        # Create an wireframe sphere to visualize working space
        self.sphere_work_geom = gymutil.WireframeSphereGeometry(0.9, 24, 24, sphere_pose, color=(0, 0, 1))

        # Sensor camera properties
        cam_pos = gymapi.Vec3(-1.4, -1.4, 1.6)
        cam_target = gymapi.Vec3(-1.0, -0.0, 0.8)
        cam_props = gymapi.CameraProperties()

        self.resolution_width = 2400
        self.resolution_height = 2000

        cam_props.width = self.resolution_width
        cam_props.height = self.resolution_height

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            franka_actor = self.gym.create_actor(
                env_ptr, franka_asset, franka_start_pose, "franka", i, 0, segmentationId=0
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
            hand_handle = body = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_hand")
            props = self.gym.get_actor_rigid_body_states(env_ptr, franka_actor, gymapi.STATE_POS)

            attractor_properties.target = props["pose"][:][body_dict["panda_hand"]]
            attractor_properties.target.p.y -= 0.1
            attractor_properties.target.p.z = 0.1
            attractor_properties.rigid_handle = hand_handle

            attractor_handle = self.gym.create_rigid_body_attractor(env_ptr, attractor_properties)
            self.attractor_handles.append(attractor_handle)

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
                rigid_prop.friction = 100.0
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

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        #
        # User code to digest tensors
        #
        metric_obs_tensor = torch.Tensor().to(self.device)
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

            obs_tensor = torch.Tensor().to(self.device)
            if metric_obs == "depth":
                obs_tensor = depth_camera_tensor
            elif metric_obs == "rgbd":
                obs_tensor = torch.zeros_like(rgba_camera_tensor)
                obs_tensor[:3, :, :] = rgba_camera_tensor[:3, :, :]
                obs_tensor[-1, :, :] = depth_camera_tensor
            obs_tensor = obs_tensor.unsqueeze(0)
            metric_obs_tensor = torch.cat((metric_obs_tensor, obs_tensor), 0)

        init_obs = torch.Tensor().to(self.device)
        rgbd_tensor = to_torch(self.rgbd, device=self.device)
        if metric_obs == "depth":
            init_obs = rgbd_tensor[-1, :, :]
        elif metric_obs == "rgbd":
            init_obs = rgbd_tensor

        init_obs = init_obs.unsqueeze(0)
        init_feature = metric_model.encode(init_obs).detach()

        batch_size = 5 if (self.num_envs % 5 == 0) else 1

        feature_z = torch.Tensor().to(self.device)
        for i in range(self.num_envs // batch_size):
            feature_z_batch = metric_model.encode(
                metric_obs_tensor[i * batch_size : (i + 1) * batch_size, :, :]
            ).detach()
            feature_z = torch.cat((feature_z, feature_z_batch), 0)

        # save feature_change vector to determine whether dof change happens
        self.feature_change = feature_z - init_feature
        self.gym.end_access_image_tensors(self.sim)

    def check_hit_rate(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        object_init_dof_pos = self.object_init_dof_pos
        object_dof_pos_move = object_init_dof_pos - self.object_dof_pos

        self.object_dof_pos_move_scale = object_dof_pos_move / (
            to_torch(self.object_dof_upper_limits, device=self.device)
            - to_torch(self.object_dof_lower_limits, device=self.device)
        )
        mask = to_torch([False], device=self.device).repeat(self.num_envs)

        # if dof move 10% then it is successful
        success_threshold = 0.15

        for i in range(self.object_dof_num):
            mask = torch.logical_or(mask, (torch.abs(self.object_dof_pos_move_scale[:, i]) > success_threshold))

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
        self.mask = mask.long()

        # reset mask based on reset buffer
        self.mask = torch.where(self.reset_buf > 0, torch.zeros_like(self.mask), self.mask)

        hit_rate = mask.sum().cpu().numpy() / self.num_envs
        print("hit rate: ", hit_rate)

        self.success = mask.long().unsqueeze(-1)

        for i in range(self.num_envs):
            dof_str = ""
            for j in range(self.object_dof_num):
                dof_str += str(int(self.obj_dof_change[i, j].detach().cpu().numpy()))
            self.obj_dof_change_str.append(dof_str)

        # Check output dof change/modes
        print(self.obj_dof_change)

    def save_replay_buf(self, data, use_metric, cluster_method):
        # TODO use KNN first, assume that we know the k cluster
        key_list = self.obj_dof_change_str
        # self.object_dof_pos_move_scale
        if cluster_method == "gmm" or cluster_method == "kmean":
            if use_metric:
                dof_change_data = self.feature_change
            else:
                dof_change_data = self.object_dof_pos_move_scale

            n_components = 9 if self.num_envs > 9 else (self.num_envs // 2)

            if cluster_method == "gmm":
                self.cluster_model = GMM(n_components=n_components)
            elif cluster_method == "kmean":
                self.cluster_model = KMeans(n_clusters=n_components)

            gt_label = data[:, -1]
            gt_label = gt_label.cpu().numpy()

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
                    self.success_data = torch.cat((self.success_data, data[i, :].unsqueeze(0)), 0)
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

            self.fit_replay_buf()

        """
        for i in range(self.num_envs):
            if key_list[i] not in self.replay_buf.keys():
                self.replay_buf[key_list[i]] = torch.Tensor().to(self.device)
            save_data = data[i, :].unsqueeze(0)
            self.replay_buf[key_list[i]] = torch.cat((self.replay_buf[key_list[i]], save_data), 0)
        """

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

    def save_data(self, dataset_root, is_rgb, is_clustering, use_metric, cluster_method="dof"):
        # Save cvs file of different interactions
        # Save tsdf file for convOnet
        # Save point cloud for pc sampling
        # Save depth for metric learning and encoding

        Path(dataset_root).mkdir(parents=True, exist_ok=True)

        rotation = self.valid_init_state[:, 3:7]
        if len(self.valid_index) < self.num_envs:
            self.valid_index = [0 for i in range(self.num_envs)]

        index_tensor = to_torch(self.valid_index).unsqueeze(-1)

        # TODO check whether our clusters have meaningful representation
        dof_change = torch.zeros_like(self.success)
        for i in range(len(self.obj_dof_change_str)):
            dof_change[i] = int(self.obj_dof_change_str[i])

        data = torch.cat((rotation, self.select_points, self.norm_force, index_tensor, self.success, dof_change), 1)

        if is_clustering:
            self.save_replay_buf(data, use_metric, cluster_method)
        else:
            if len(self.replay_buf) == 0:
                self.replay_buf[0] = torch.Tensor().to(self.device)
            self.replay_buf[0] = torch.cat((self.replay_buf[0], data), 0)

        data = data.cpu().numpy()

        csv_path = dataset_root + "/" + str(self.object_id) + "_inter.csv"
        # csv_path = dataset_root + "/" + "interact.csv"
        csv_path = Path(csv_path)

        # save csv file
        if not csv_path.exists():
            with csv_path.open("w") as f:
                f.write(
                    ",".join(
                        ["object_id", "qx", "qy", "qz", "qw", "x", "y", "z", "fx", "fy", "fz", "id", "label", "dof"]
                    )
                )
                f.write("\n")

        with csv_path.open("a") as f:
            for i in range(data.shape[0]):
                line = data[i].tolist()[:-1]
                # TODO CVAE testing, we want positive data only
                if line[-1] > 0:
                    row = ",".join([str(data) for data in line])
                    # TODO add object id for training
                    if self.is_multi_pose:
                        row = str(self.object_id) + "_" + self.multi_pose_str + "," + row
                    else:
                        row = str(self.object_id) + "," + row
                    row += "," + self.obj_dof_change_str[i]
                    f.write(row)
                    f.write("\n")

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

        print("success num: ", self.success.sum().cpu())

        if self.success.sum().cpu() > 10000:
            hit_rotation = rotation[self.success.squeeze() > 0]
            hit_select_point = self.select_points[self.success.squeeze() > 0]
            hit_force = self.norm_force[self.success.squeeze() > 0]

            print("interact point: ", hit_select_point)
            print("rotation: ", hit_rotation)
            print("norm_force: ", hit_force)


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
        selected_point = to_torch([-0.6098,  -0.1710,  1.0607], device=self.device).repeat((self.num_envs * sample_scale, 1))
        print(selected_point)
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

        init_vel = torch.zeros((1, 6), device=self.device)
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



    def sample_action(self, is_clustering, cluster_method="dof"):
        # Remember to clear some tensor to avoid duplicate sampling
        self.force = torch.Tensor().to(self.device)
        self.norm_force = torch.Tensor().to(self.device)

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
                epsilon = 0.5
            else:
                epsilon = 1.0
            random_sample_num = int(self.num_envs * epsilon)
            self.uniform_action(random_sample_num)
            self.sample_cluster(self.num_envs - random_sample_num)
        else:
            self.uniform_action(self.num_envs)

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
        self.replay_buf_models = {}
        for key in self.replay_buf.keys():
            replay_buf_data = self.replay_buf[key]
            replay_buf_data_fit = replay_buf_data.clone()[:, :10]
            n_cluster = 6 if replay_buf_data_fit.shape[0] > 6 else replay_buf_data_fit.shape[0]

            self.replay_buf_models[key] = GMM(n_components=n_cluster)
            if replay_buf_data_fit.shape[0] > 1:
                self.replay_buf_models[key].fit(replay_buf_data_fit.cpu().numpy())
            else:
                replay_buf_data_fit_duplicate = replay_buf_data_fit.repeat((5, 1))
                replay_buf_data_fit_duplicate = replay_buf_data_fit_duplicate + (0.1**0.5) * torch.randn(5, 10).to(
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
        sample_model = self.replay_buf_models[key]
        if sample_model.weights_[0] > 1:
            sample_model.weights_[0] = 1
        sample_data_x, sample_data_y = sample_model.sample(num_sample)
        # rotation, self.select_points, self.norm_force

        # rotation, self.select_points, self.norm_force, index_tensor
        valid_action_num = 0
        interact_points = torch.Tensor().to(self.device)
        valid_init_state_pos = torch.Tensor().to(self.device)
        valid_force = torch.Tensor().to(self.device)

        while valid_action_num <= num_sample:
            sample_rotation = to_torch(sample_data_x[:, :4], device=self.device)
            sample_point = to_torch(sample_data_x[:, 4:7], device=self.device)
            sample_force = to_torch(sample_data_x[:, 7:], device=self.device)

            force = tensor_clamp(sample_force, torch.zeros_like(sample_force), torch.ones_like(sample_force))

            selected_point = sample_point
            candidate_rotate = sample_rotation

            z_axis = to_torch([0, 0, -1], device=self.device).repeat((num_sample, 1))
            y_axis = to_torch([0, 1, 0], device=self.device).repeat((num_sample, 1))

            self.interact_pos = selected_point + quat_apply(candidate_rotate, z_axis * self.eef_hand_dist)
            init_pos = selected_point + quat_apply(candidate_rotate, z_axis * (self.init_dist + self.eef_hand_dist))
            inter_pos = selected_point + quat_apply(
                candidate_rotate, z_axis * (self.init_dist * 0.5 + self.eef_hand_dist)
            )

            z_bias = quat_apply(candidate_rotate, z_axis * self.eef_hand_dist * 0.5)
            init_pos_y = (
                selected_point + quat_apply(candidate_rotate, y_axis * (self.init_dist + self.eef_hand_dist)) + z_bias
            )
            init_pos_y_ = (
                selected_point - quat_apply(candidate_rotate, y_axis * (self.init_dist + self.eef_hand_dist)) + z_bias
            )

            constrain_point = torch.stack([init_pos, inter_pos, init_pos_y, init_pos_y_], dim=2)
            constrain_point = torch.stack([init_pos, init_pos_y, init_pos_y_], dim=2)
            constrain_point = torch.stack([init_pos], dim=2)

            mask = to_torch([False], device=self.device).repeat(num_sample)

            for j in range(constrain_point.shape[-1]):
                gripper_point_inside_bound = to_torch([True], device=self.device).repeat(num_sample)
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

            # mask = to_torch([True], device=self.device).repeat(num_sample * sample_scale)
            mask = torch.gt(mask, 0)

            valid_init_rotate = candidate_rotate[mask]
            valid_init_pos = init_pos[mask]
            valid_state = torch.cat((valid_init_pos, valid_init_rotate), 1)
            valid_init_state_pos = torch.cat((valid_init_state_pos, valid_state), 0)
            valid_force = torch.cat((valid_force, force), 0)

            interact_points = torch.cat((interact_points, selected_point[mask]))
            valid_action_num += mask.sum().cpu().numpy()

        valid_interact_points = interact_points[:(num_sample), :]
        self.select_points = torch.cat((self.select_points, valid_interact_points))
        valid_init_state_pos = valid_init_state_pos[:num_sample, :]

        init_vel = torch.zeros((num_sample, 6), device=self.device)
        _valid_init_state = torch.cat((valid_init_state_pos, init_vel), 1)
        self.valid_init_state = torch.cat((self.valid_init_state, _valid_init_state), 0)

        # sample random force vector
        # force_sample = to_torch(np.random.rand(num_sample, 3), device=self.device) - 0.5
        norm_force = valid_force[:num_sample, :]
        self.norm_force = torch.cat((self.norm_force, norm_force), 0)
        force_norm_devide = torch.norm(norm_force - 0.5, dim=-1).unsqueeze(-1)

        force_exe = ((norm_force - 0.5) / force_norm_devide) * 0.5 * self.init_dist
        force_exe[:, :2] += self.init_dist

        self.force = torch.cat((self.force, force_exe), 0)

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
                selected_point + quat_apply(candidate_rotate, y_axis * (self.init_dist + self.eef_hand_dist)) + z_bias
            )
            init_pos_y_ = (
                selected_point - quat_apply(candidate_rotate, y_axis * (self.init_dist + self.eef_hand_dist)) + z_bias
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

    def uniform_action(self, num_sample):
        """
        self.franka_init_state.append([franka_start_pose.p.x, franka_start_pose.p.y, franka_start_pose.p.z,
                                       franka_start_pose.r.x, franka_start_pose.r.y, franka_start_pose.r.z,
                                       franka_start_pose.r.w,
                                       0, 0, 0, 0, 0, 0])
        """

        valid_action_num = 0
        sample_scale = 10
        self.valid_init_state = None

        interact_points = torch.Tensor().to(self.device)
        self.valid_index = []

        while valid_action_num <= num_sample:
            print("sample")
            # randomly sample rotation matrix
            candidate_rotate = to_torch(R.random(self.num_envs * sample_scale).as_quat(), device=self.device)

            # TODO test pull
            qx = gymapi.Quat.from_euler_zyx(0.5 * math.pi + random.uniform(0, 1) * 0.1, 0.0 * math.pi+ random.uniform(0, 1) * 0.1, 0.5 * math.pi+ random.uniform(0, 1) * 0.1)
            candidate_rotate = to_torch([qx.x, qx.y, qx.z, qx.w], device=self.device).repeat((self.num_envs * sample_scale, 1))

            # randomly sample point to interact
            index = np.random.choice(self.pointcloud.shape[0], self.num_envs * sample_scale, replace=True)
            index = np.random.choice(self.candidate_points.shape[0], self.num_envs * sample_scale, replace=True)

            selected_pc_index = to_torch(index, device=self.device)

            selected_point = to_torch(self.pointcloud[index], device=self.device)
            selected_point = to_torch(self.candidate_points[index], device=self.device)
            selected_point = to_torch([-1.2, 0, 1.18], device=self.device).repeat((self.num_envs * sample_scale, 1))
            selected_point = to_torch([-1.2, 0.24, 1.13], device=self.device).repeat((self.num_envs * sample_scale, 1))

            z_axis = to_torch([0, 0, -1], device=self.device).repeat((self.num_envs * sample_scale, 1))
            y_axis = to_torch([0, 1, 0], device=self.device).repeat((self.num_envs * sample_scale, 1))

            interact_pos = selected_point + quat_apply(candidate_rotate, z_axis * self.eef_hand_dist)
            init_pos = selected_point + quat_apply(candidate_rotate, z_axis * (self.init_dist + self.eef_hand_dist))
            inter_pos = selected_point + quat_apply(
                candidate_rotate, z_axis * (self.init_dist * 0.5 + self.eef_hand_dist)
            )

            z_bias = quat_apply(candidate_rotate, z_axis * self.eef_hand_dist * 0.5)
            init_pos_y = (
                selected_point + quat_apply(candidate_rotate, y_axis * (self.init_dist + self.eef_hand_dist)) + z_bias
            )
            init_pos_y_ = (
                selected_point - quat_apply(candidate_rotate, y_axis * (self.init_dist + self.eef_hand_dist)) + z_bias
            )

            constrain_point = torch.stack([init_pos, inter_pos, init_pos_y, init_pos_y_], dim=2)
            constrain_point = torch.stack([init_pos, init_pos_y, init_pos_y_], dim=2)
            constrain_point = torch.stack([init_pos], dim=2)

            mask = to_torch([False], device=self.device).repeat(self.num_envs * sample_scale)

            for j in range(constrain_point.shape[-1]):
                gripper_point_inside_bound = to_torch([True], device=self.device).repeat(self.num_envs * sample_scale)
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

            # TODO operational space must be inside the workspace
            # workspace_pose.p = gymapi.Vec3(-1.8, 0.0, 0.3)
            # self.sphere_work_geom = gymutil.WireframeSphereGeometry(0.9, 24, 24, sphere_pose, color=(0, 0, 1))

            constrain_point = torch.stack([init_pos, interact_pos], dim=2)
            robot_center = torch.Tensor([-1.8, 0, 0.3]).to(self.device)

            init_to_robot_center = init_pos - robot_center
            init_to_robot_dist = torch.norm(init_to_robot_center, dim=-1)
            workspace_bool = init_to_robot_dist < 0.9
            mask = torch.logical_and(mask, workspace_bool)

            inter_to_robot_center = interact_pos - robot_center
            inter_to_robot_dist = torch.norm(inter_to_robot_center, dim=-1)
            workspace_bool = inter_to_robot_dist < 0.9
            mask = torch.logical_and(mask, workspace_bool)

            # mask = to_torch([True], device=self.device).repeat(self.num_envs * sample_scale)
            mask = torch.gt(mask, 0)

            valid_init_rotate = candidate_rotate[mask]
            valid_init_pos = init_pos[mask]
            valid_interact_pos = interact_pos[mask]
            valid_state = torch.cat((valid_init_pos, valid_init_rotate), 1)
            if valid_action_num == 0:
                self.valid_init_state = valid_state
                self.interact_pos = valid_interact_pos
            else:
                self.valid_init_state = torch.cat((self.valid_init_state, valid_state), 0)
                self.interact_pos = torch.cat((self.interact_pos, valid_interact_pos), 0)
            interact_points = torch.cat((interact_points, selected_point[mask]))
            valid_action_num += mask.sum().cpu().numpy()

            valid_selected_pc_index = selected_pc_index[mask]
            valid_selected_pc_index_list = valid_selected_pc_index.detach().cpu().numpy().tolist()
            self.valid_index.extend(valid_selected_pc_index_list)
        self.select_points = interact_points[:num_sample, :]
        self.valid_init_state = self.valid_init_state[:num_sample, :]
        self.interact_pos = self.interact_pos[:num_sample, :]
        self.valid_index = self.valid_index[:num_sample]
        self.valid_index = [int(x) for x in self.valid_index]

        init_vel = torch.zeros((num_sample, 6), device=self.device)
        self.valid_init_state = torch.cat((self.valid_init_state, init_vel), 1)

        # sample random force vector
        # force_sample = to_torch(np.random.rand(num_sample, 3), device=self.device) - 0.5
        self.norm_force = torch.cat((self.norm_force, to_torch(np.random.rand(num_sample, 3), device=self.device)))
        force_norm = torch.norm(self.norm_force - 0.5, dim=-1).unsqueeze(-1)

        self.force = torch.cat((self.force, ((self.norm_force - 0.5) / force_norm) * 0.5 * self.init_dist))

        # TODO what if we only has push action
        self.norm_force = to_torch([0.5, 0.5, 1.0], device=self.device).repeat(num_sample, 1)
        self.force = to_torch([1, 0, 0], device=self.device).repeat((num_sample, 1)) * self.init_dist
        self.pc_hit = np.zeros(self.candidate_points.shape[0])

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
        row, col, ch = rgba.shap10
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

    def viewer_camera(self, env_index, is_first, is_metric=False):
        print("render images")
        start_time = time.time()
        camera_properties = gymapi.CameraProperties()
        cam_width = 320
        cam_height = 320
        camera_properties.width = cam_width
        camera_properties.height = cam_height
        camera_properties.enable_tensors = True
        num_camera = 16
        if is_first:
            num_camera = 1
        pi = math.pi

        def PointsInCircum(r, n=4):
            return [(math.cos(2 * pi / n * x) * r, math.sin(2 * pi / n * x) * r) for x in range(0, n + 1)]

        circle_point = PointsInCircum(-2.0, num_camera)
        cam_handles = []
        # Set a fixed position and look-target for the first camera
        # position and target location are in the coordinate frame of the environment

        if env_index not in self.camera_handles.keys():
            for i in range(num_camera):
                camera_handle = self.gym.create_camera_sensor(self.envs[env_index], camera_properties)
                # Look at the env
                cam_pos = gymapi.Vec3(circle_point[i][0], circle_point[i][1], 1.5)
                cam_target = gymapi.Vec3(0.0, 0.0, 1.0)
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
        candidate_points = torch.Tensor().to(self.device)

        depth_imgs = []
        color_imgs = []
        view_matrices = []

        fu_s = []
        fv_s = []

        # TODO convert to GPU version
        print("Converting Depth images to point clouds. Have patience...")
        for i in range(num_camera):
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

            if i == 0:
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
            fu = 2 / proj[0, 0]
            fv = 2 / proj[1, 1]

            fu_s.append(fu)
            fv_s.append(fv)

            # Ignore any points which originate from ground plane or empty space
            depth_buffer[seg_buffer == 0] = -10001

            depth_img = to_torch(depth_buffer, device=self.device)

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

            if i == 0:
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
            pc = pc[:, :3]
            env_points = torch.cat((env_points, pc))
            if i == 0:
                rgb_filename = "robot_env_cam%d.png" % (i)
                self.gym.write_camera_image_to_file(
                    self.sim, self.envs[i], cam_handles[i], gymapi.IMAGE_COLOR, rgb_filename
                )
                candidate_points = torch.cat((candidate_points, pc))

        points = env_points.cpu().numpy()
        pcd = o3d.open3d.geometry.PointCloud()
        pcd.points = o3d.open3d.utility.Vector3dVector(points)
        # o3d.visualization.draw_geometries([pcd])

        print("Downsample the point cloud with a voxel of 0.05")
        downpcd = pcd.voxel_down_sample(voxel_size=0.02)

        candidate_points = candidate_points.cpu().numpy()
        cpcd = o3d.open3d.geometry.PointCloud()
        cpcd.points = o3d.open3d.utility.Vector3dVector(candidate_points)
        downcpcd = cpcd.voxel_down_sample(voxel_size=0.02)
        self.candidate_points = np.asarray(downcpcd.points)

        # print("Recompute the normal of the downsampled point cloud")
        # estimate_normals( downpcd, search_param=o3d.open3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50))

        # visualize 3d point cloud
        # o3d.open3d.visualization.draw_geometries([downpcd])

        # bounding_box = downpcd.get_axis_aligned_bounding_box()
        self.pointcloud = np.asarray(downpcd.points)

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

        if not npz_path.exists() or is_metric:
            print("Initializing voxel volume...")
            voxel_size = 0.05
            self.tsdf = TSDFVolume(self.tsdf_vol_bnds, voxel_size, self.device)
            for i in range(num_camera):
                print("Fusing frame %d/%d" % (i + 1, num_camera))
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

        # self.gym.render_all_camera_sensors(self.sim)
        # self.gym.start_access_image_tensors(self.sim)

        # configuration space control
        self.franka_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, : self.num_franka_dofs]
        self.franka_dof_pos = self.franka_dof_state[..., 0]
        self.franka_dof_vel = self.franka_dof_state[..., 1]

        # object dof state
        self.object_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_franka_dofs :]
        self.object_dof_pos = self.object_dof_state[..., 0]
        self.object_dof_vel = self.object_dof_state[..., 1]

        # EEF control
        self.franka_lfinger_pos = self.rigid_body_states[:, self.lfinger_handle][:, 0:3]
        self.franka_rfinger_pos = self.rigid_body_states[:, self.rfinger_handle][:, 0:3]
        self.franka_lfinger_rot = self.rigid_body_states[:, self.lfinger_handle][:, 3:7]
        self.franka_rfinger_rot = self.rigid_body_states[:, self.rfinger_handle][:, 3:7]

        self.hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
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

    def save_video(self, video_name):
        if self.success.item() == 0:
            return
        rand_str = random_string(4)
        video_name = rand_str + "_" + video_name

        video_dir = "./video_actual_robot/" + str(self.object_id)
        Path(video_dir).mkdir(parents=True, exist_ok=True)
        video_name = video_dir + "/" + video_name
        frame = self.video_frames[0]
        # height, width, layers = frame.shape
        # plt.imsave("test.png", frame)
        self.video_frames.pop(0)

        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(self.video_frames, fps=20.0)
        clip.write_videofile(video_name)

    def reset(self, env_ids):
        print("------------------------------------reset")
        self.task_state = -1

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

    def draw_workspace(self):
        for i in range(self.num_envs):
            # Draw sphere to visualize working space
            workspace_pose = gymapi.Transform()
            # Franka robot on the table
            workspace_pose.p = gymapi.Vec3(-1.8, 0.0, 0.3)
            workspace_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0)

            gymutil.draw_lines(self.sphere_work_geom, self.gym, self.viewer, self.envs[i], workspace_pose)

    def update_attractor(self, next_attractor_state):
        for i in range(self.num_envs):
            attractor_properties = self.gym.get_attractor_properties(self.envs[i], self.attractor_handles[i])
            pose = attractor_properties.target

            current_pos = self.hand_pos[i]
            next_attractor_pos = next_attractor_state[i, :3]
            dist = (torch.norm(next_attractor_pos - current_pos))
            direction = (next_attractor_pos - current_pos) / dist

            smooth_next_attractor = current_pos + direction * 0.015

            if dist.item() <= 0.015:
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
            gymutil.draw_lines(self.sphere_geom, self.gym, self.viewer, self.envs[i], pose)

    def pre_physics_step(self, actions):
        self.gym.clear_lines(self.viewer)
        self.task_state = actions
        # torch.tensor([0, 0, env.init_dist, 0, 0, 0, 0.04, 0.04])
        if actions == 0:
            self.actions = (
                torch.tensor([0, 0, 0, -1.6, 0, 1.9, 0, 0.04, 0.04]).repeat((self.num_envs, 1)).to(self.device)
            )
        elif actions == 1:
            self.actions = (
                torch.tensor([0, 0, 0, -1.6, 0, 1.9, 0, 0.04, 0.04]).repeat((self.num_envs, 1)).to(self.device)
            )
        elif actions == 2:
            self.actions[:, :3] = self.force
        elif actions == -1:
            self.actions = -1
            return
        elif actions == 3:
            self.actions = torch.tensor([0, 0, 0, -1.6, 0, 1.9, 0, 0.0, 0.0]).repeat((self.num_envs, 1)).to(self.device)
        else:
            print("invalid actions primitive! exit...")
            exit()

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

        if actions == 0:
            self.franka_dof_targets[:, 7:9] = 0.04
            self.update_attractor(self.valid_init_state)


        if actions == 1:
            self.franka_dof_targets[:, 7:9] = 0.04
            self.update_attractor(self.interact_pos)

        if actions == 2:
            self.franka_dof_targets[:, 7:9] = 0.0

        if actions == 3:
            self.franka_dof_targets[:, 7:9] = 0.0
            self.update_attractor(self.interact_pos + self.force)


        # self.draw_workspace()
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


        self.gym.render_all_camera_sensors(self.sim)
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


#####################################################################
###=========================jit functions=========================###
#####################################################################


def random_string(string_length=6):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4())  # Convert UUID format to a Python string.
    random = random.upper()  # Make all characters uppercase.
    random = random.replace("-", "")  # Remove the UUID '-'.
    return random[0:string_length]  # Return the random string.


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

    pdb.set_trace()

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
