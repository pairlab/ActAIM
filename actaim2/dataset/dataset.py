import numpy as np
from scipy import ndimage
import torch.utils.data
from pathlib import Path
import pandas as pd
import os
import math
from PIL import Image, ImageEnhance
import pdb
import random
import matplotlib.pyplot as plt


# from vgn.io import *
# from vgn.perception import *
from vgn.utils.transform import Rotation, Transform
# from vgn.utils.implicit import get_scene_from_mesh_pose_list

# domain randomization
def adjust_brightness(image, factor):
    """Adjust the brightness of an image. A factor > 1.0 makes the image brighter."""
    enhancer = ImageEnhance.Brightness(Image.fromarray(image))
    enhanced_im = enhancer.enhance(factor)
    return np.array(enhanced_im)

def add_noise(image, noise_type='gaussian', amount=0.02):
    """Add noise to an image. Supports Gaussian noise."""
    if noise_type == 'gaussian':
        row, col, ch = image.shape
        mean = 0
        sigma = amount**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)
    else:
        return image

def simulate_depth_inaccuracy(depth_image, max_deviation=5):
    """Simulate depth sensor inaccuracies by adding random deviations to depth values."""
    deviation = np.random.uniform(-max_deviation, max_deviation, depth_image.shape)
    simulated_depth = depth_image + deviation
    return np.clip(simulated_depth, -4, 0)



def downsample_pc(pc, voxel_size=0.008, pc_num = 1024):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    # compute random down sample rate
    current_pc_num = pc.shape[0]
    ds_ratio = pc_num / current_pc_num
    ds_ratio += 0.05

    downpcd = pcd.random_down_sample(ds_ratio)
    downpcd_np = np.asarray(downpcd.points)
    # downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return downpcd_np[:pc_num, :]

class DatasetSeq(torch.utils.data.Dataset):
    # Dataset with action sequence
    def __init__(self, is_rvt=False, root="./dataset", without_init=False):
        self.without_init = without_init
        self.root = root
        self.is_rvt = is_rvt
        self.num_th = 32
        # self.df = pd.read_csv(self.root + "/19179_inter.csv")
        dataset_csv = "/action_seq_finetune.csv"
        if self.without_init:
            dataset_csv = "/action_seq_finetune_new.csv"

        self.df = pd.read_csv(self.root + dataset_csv)
        self.df.fillna('', inplace=True)
        # self.tsdf_vol_bnds = np.array([[-1.2, 0.4], [-0.8, 0.8], [0, 1.6]])
        self.tsdf_vol_bnds = np.array([[-0.9, 0.3], [-0.6, 0.6], [0.0, 1.2]])
        if self.without_init:
            self.traj_step = 2 # TODO simplified real-robot trajectory step
        else:
            self.traj_step = 4  # how many steps to complete the task
        self.cam_num = 5  # how many multi-view

        self.img_width = 256
        self.img_height = 256
        self.voxel_length = 40

        self.domain_rand = True


    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, i):
        object_id = self.df.loc[i, "object_id"]
        dof_change = self.df.loc[i, "dof"]
        obj_state = self.df.loc[i, "init_state"]
        lang_prompt = self.df.loc[i, "lang_prompt"]
        traj_id = self.df.loc[i, "traj_id"]
        object_cate = self.df.loc[i, "object_cate"]

        curr_state = 0
        begin_state = 0
        final_state = 4  # only get the final state to train contrastive learning

        # init rotation
        rotations = np.empty((self.traj_step, 2, 4), dtype=np.float64)
        R = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])

        # init position
        positions = np.empty((self.traj_step, 3), dtype=np.float64)

        # get action seq
        for step_i in range(self.traj_step):
            pos_i_str_min = str(step_i) + "_" + "px"
            pos_i_str_max = str(step_i) + "_" + "pz"
            rot_i_str_min = str(step_i) + "_" + "qx"
            rot_i_str_max = str(step_i) + "_" + "qw"

            ori_i = Rotation.from_quat(self.df.loc[i, rot_i_str_min:rot_i_str_max].to_numpy(np.float64))
            pos_i = self.df.loc[i, pos_i_str_min:pos_i_str_max].to_numpy(np.float64)

            rotations[step_i, 0] = ori_i.as_quat()
            rotations[step_i, 1] = (ori_i * R).as_quat()

            positions[step_i] = pos_i

        # image observation
        # f"{str(self.object_id)}_{self.multi_pose_str}_{state_id}_{camera_id}_{traj_id}"

        # used for obs input
        curr_color_voxel, curr_depth_voxel, curr_color_img, curr_depth_img = self.get_state_vision(object_id,
                                                                                                   obj_state,
                                                                                                   curr_state,
                                                                                                   traj_id,
                                                                                                   is_rvt=False)

        # used for codebook inference
        final_color_voxel, final_depth_voxel, final_color_img, final_depth_img = self.get_state_vision(object_id,
                                                                                                       obj_state,
                                                                                                       final_state,
                                                                                                       traj_id,
                                                                                                       is_rvt=False)

        # used for codebook inference
        begin_color_voxel, begin_depth_voxel, begin_color_img, begin_depth_img = self.get_state_vision(object_id,
                                                                                                       obj_state,
                                                                                                       begin_state,
                                                                                                       traj_id,
                                                                                                       is_rvt=False)

        # manually create gripper open close and franka_proprio is useless
        gripper_open_close = np.array([0.0, 0.0, 1.0, 1.0]) if self.traj_step == 4 else np.array([0.0, 1.0])
        franka_proprio = np.zeros(9)


        return object_cate, object_id, traj_id, obj_state, dof_change, lang_prompt, positions, rotations, gripper_open_close, \
               curr_color_voxel, curr_depth_voxel, curr_color_img, curr_depth_img, begin_color_img, begin_depth_img, \
               final_color_voxel, final_depth_voxel, final_color_img, final_depth_img, franka_proprio, curr_state


    def get_state_vision(self, object_id, obj_state, curr_state, traj_id, is_rvt=False):
        # image observation
        # f"{str(self.object_id)}_{self.multi_pose_str}_{state_id}_{camera_id}_{traj_id}"

        color_img = np.empty((self.cam_num, self.img_width, self.img_height, 4), dtype=np.uint8)
        depth_img = np.empty((self.cam_num, self.img_width, self.img_height), dtype=np.float64)

        voxel_file_name_list = [object_id, obj_state, str(curr_state), traj_id]
        voxel_file_name = "_".join([str(data) for data in voxel_file_name_list])

        color_voxel_path = self.root + "/" + str(object_id) + "/color_" + voxel_file_name + ".npz"
        voxel_path = self.root + "/" + str(object_id) + "/voxel_" + voxel_file_name + ".npz"

        color_voxel = np.load(color_voxel_path)["voxel"]
        depth_voxel = np.load(voxel_path)["voxel"]

        is_seg = self.without_init

        for cam_i in range(self.cam_num):
            img_file_name_list = [object_id, obj_state, str(curr_state), str(cam_i), traj_id]
            img_file_name = "_".join([str(data) for data in img_file_name_list])

            img_type = "rvt_" if is_rvt else ""

            color_path = self.root + "/" + str(object_id) + "/" + img_type + img_file_name + ".png"
            depth_path = self.root + "/" + str(object_id) + "/" + img_type + img_file_name + ".npz"

            color_img_i = np.array(Image.open(color_path))
            depth_img_i = np.load(depth_path)["depth"]

            if is_seg:
                seg_path = self.root + "/" + str(object_id) + "/" + "seg_" + img_file_name + ".npz"
                seg_img_i = np.load(seg_path)["seg"]
                color_img_i[seg_img_i == 9] = 0
                depth_img_i[seg_img_i == 9] = 0

                if self.domain_rand:
                    brightness_factor = random.uniform(0.9, 1.3)
                    brightened_rgb = adjust_brightness(color_img_i, factor=brightness_factor)
                    color_img_i = add_noise(brightened_rgb, 'gaussian', 0.02)
                    depth_img_i = simulate_depth_inaccuracy(depth_img_i, max_deviation=0.025)

            color_img[cam_i] = color_img_i
            depth_img[cam_i] = depth_img_i

        return color_voxel, depth_voxel, color_img, depth_img


class DatasetTuple(torch.utils.data.Dataset):
    def __init__(self, is_rvt=False, root="./dataset", without_init=False):
        self.root = root
        self.is_rvt = is_rvt
        self.num_th = 32
        self.without_init = without_init
        # self.df = pd.read_csv(self.root + "/19179_inter.csv")

        if self.without_init:
            self.df = pd.read_csv(self.root + "/action_tuple_finetune_new.csv")
        else:
            self.df = pd.read_csv(self.root + "/action_tuple_finetune.csv")

        self.df.fillna('', inplace=True)
        # self.tsdf_vol_bnds = np.array([[-1.2, 0.4], [-0.8, 0.8], [0, 1.6]])
        self.tsdf_vol_bnds = np.array([[-0.9, 0.3], [-0.6, 0.6], [0.0, 1.2]])
        self.traj_step = 4 # how many steps to complete the task
        self.cam_num = 5 # how many multi-view

        self.img_width = 256
        self.img_height = 256
        self.voxel_length = 40

        # whetehr to perform domain randomization here
        self.domain_rand = False


    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, i):
        object_id = self.df.loc[i, "object_id"]
        dof_change = self.df.loc[i, "dof"]
        obj_state = self.df.loc[i, "init_state"]
        lang_prompt = self.df.loc[i, "lang_prompt"]
        traj_id = self.df.loc[i, "traj_id"]
        object_cate = self.df.loc[i, "object_cate"]
        curr_state = int(self.df.loc[i, "state_i"])
        final_state = 4 # only get the final state to train contrastive learning
        gripper_open_close = int(self.df.loc[i, "jaw"])
        franka_proprio = self.df.loc[i, "robot_dof_0":"robot_dof_8"].to_numpy(np.float64)

        # init rotation
        rotations = np.empty((2, 4), dtype=np.float64)
        R = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])

        # get action seq

        ori = Rotation.from_quat(self.df.loc[i, "qx":"qw"].to_numpy(np.float64))
        pos = self.df.loc[i, "px":"pz"].to_numpy(np.float64)

        rotations[0] = ori.as_quat()
        rotations[1] = (ori * R).as_quat()

        curr_state_vis = curr_state
        if self.without_init:
            if curr_state == 1:
                curr_state_vis = 0

        curr_color_voxel, curr_depth_voxel, curr_color_img, curr_depth_img = self.get_state_vision(object_id,
                                                                                                   obj_state,
                                                                                                   curr_state_vis, traj_id)
        final_color_voxel, final_depth_voxel, final_color_img, final_depth_img = self.get_state_vision(object_id,
                                                                                                   obj_state,
                                                                                                   final_state, traj_id)

        # TODO load the begin state to compute the task embedding
        begin_state = 0
        begin_color_voxel, begin_depth_voxel, begin_color_img, begin_depth_img = self.get_state_vision(object_id,
                                                                                                       obj_state,
                                                                                                       begin_state,
                                                                                                       traj_id)

        return object_cate, object_id, traj_id, obj_state, dof_change, lang_prompt, pos, rotations, gripper_open_close, \
               curr_color_voxel, curr_depth_voxel, curr_color_img, curr_depth_img, begin_color_img, begin_depth_img, \
               final_color_voxel, final_depth_voxel, final_color_img, final_depth_img, franka_proprio, curr_state


    def get_state_vision(self, object_id, obj_state, curr_state, traj_id):
        # image observation
        # f"{str(self.object_id)}_{self.multi_pose_str}_{state_id}_{camera_id}_{traj_id}"

        color_img = np.empty((self.cam_num, self.img_width, self.img_height, 4), dtype=np.uint8)
        depth_img = np.empty((self.cam_num, self.img_width, self.img_height), dtype=np.float64)

        voxel_file_name_list = [object_id, obj_state, str(curr_state), traj_id]
        voxel_file_name = "_".join([str(data) for data in voxel_file_name_list])

        color_voxel_path = self.root + "/" + str(object_id) + "/color_" + voxel_file_name + ".npz"
        voxel_path = self.root + "/" + str(object_id) + "/voxel_" + voxel_file_name + ".npz"

        color_voxel = np.load(color_voxel_path)["voxel"]
        depth_voxel = np.load(voxel_path)["voxel"]


        for cam_i in range(self.cam_num):
            img_file_name_list = [object_id, obj_state, str(curr_state), str(cam_i), traj_id]
            img_file_name = "_".join([str(data) for data in img_file_name_list])

            img_type = "rvt_" if self.is_rvt else ""

            color_path = self.root + "/" + str(object_id) + "/" + img_type + img_file_name + ".png"
            depth_path = self.root + "/" + str(object_id) + "/" + img_type + img_file_name + ".npz"

            color_img_i = np.array(Image.open(color_path))
            depth_img_i = np.load(depth_path)["depth"]
            if self.domain_rand:
                brightness_factor = random.uniform(0.9, 1.3)
                brightened_rgb = adjust_brightness(color_img_i, factor=brightness_factor)
                color_img_i = add_noise(brightened_rgb, 'gaussian', 0.02)
                depth_img_i = simulate_depth_inaccuracy(depth_img_i, max_deviation=0.035)


            color_img[cam_i] = color_img_i
            depth_img[cam_i] = depth_img_i

        return color_voxel, depth_voxel, color_img, depth_img


class DatasetSeq_Memory(torch.utils.data.Dataset):
    # Dataset with action sequence
    def __init__(self, is_rvt=False, root="./dataset", num_point=2048):
        self.root = root
        self.is_rvt = is_rvt
        self.num_point = num_point
        self.num_th = 32
        # self.df = pd.read_csv(self.root + "/19179_inter.csv")
        self.df = pd.read_csv(self.root + "/action_seq_finetune.csv")
        self.df.fillna('', inplace=True)
        # self.tsdf_vol_bnds = np.array([[-1.5, 1.5], [-1., 1.], [0, 2]])
        self.tsdf_vol_bnds = np.array([[-1.2, 0.4], [-0.8, 0.8], [0, 1.6]])
        self.traj_step = 4 # how many steps to complete the task
        self.cam_num = 5 # how many multi-view

        self.img_width = 256
        self.img_height = 256
        self.voxel_length = 40


    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, i):
        object_id = self.df.loc[i, "object_id"]
        dof_change = self.df.loc[i, "dof"]
        obj_state = self.df.loc[i, "init_state"]
        lang_prompt = self.df.loc[i, "lang_prompt"]
        traj_id = self.df.loc[i, "traj_id"]
        object_cate = self.df.loc[i, "object_cate"]

        # init rotation
        rotations = np.empty((self.traj_step, 2, 4), dtype=np.float64)
        R = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])

        # init position
        positions = np.empty((self.traj_step, 3), dtype=np.float64)

        # get action seq
        for step_i in range(self.traj_step):
            pos_i_str_min = str(step_i) + "_" + "px"
            pos_i_str_max = str(step_i) + "_" + "pz"
            rot_i_str_min = str(step_i) + "_" + "qx"
            rot_i_str_max = str(step_i) + "_" + "qw"

            ori_i = Rotation.from_quat(self.df.loc[i, rot_i_str_min:rot_i_str_max].to_numpy(np.float64))
            pos_i = self.df.loc[i, pos_i_str_min:pos_i_str_max].to_numpy(np.float64)

            rotations[step_i, 0] = ori_i.as_quat()
            rotations[step_i, 1] = (ori_i * R).as_quat()

            positions[step_i] = pos_i

        # image observation
        # f"{str(self.object_id)}_{self.multi_pose_str}_{state_id}_{camera_id}_{traj_id}"

        color_img = np.empty((self.traj_step, self.cam_num, self.img_width, self.img_height, 4), dtype=np.uint8)
        depth_img = np.empty((self.traj_step, self.cam_num, self.img_width, self.img_height), dtype=np.float64)

        color_voxel = np.empty((self.traj_step, self.voxel_length, self.voxel_length, self.voxel_length, 3), dtype=np.float64)
        depth_voxel = np.empty((self.traj_step, self.voxel_length, self.voxel_length, self.voxel_length), dtype=np.float64)

        for step_i in range(self.traj_step):
            voxel_file_name_list = [object_id, obj_state, str(step_i), traj_id]
            voxel_file_name = "_".join([str(data) for data in voxel_file_name_list])

            color_voxel_path = self.root + "/" + str(object_id) + "/color_" + voxel_file_name + ".npz"
            voxel_path = self.root + "/" + str(object_id) + "/voxel_" + voxel_file_name + ".npz"

            color_voxel_i = np.load(color_voxel_path)["voxel"]
            voxel_i = np.load(voxel_path)["voxel"]

            color_voxel[step_i] = color_voxel_i
            depth_voxel[step_i] = voxel_i

            for cam_i in range(self.cam_num):
                img_file_name_list = [object_id, obj_state, str(step_i), str(cam_i), traj_id]
                img_file_name = "_".join([str(data) for data in img_file_name_list])

                color_path = self.root + "/" + str(object_id) + "/" + img_file_name + ".png"
                depth_path = self.root + "/" + str(object_id) + "/" + img_file_name + ".npz"

                color_img_i = np.array(Image.open(color_path))
                depth_img_i = np.load(depth_path)["depth"]

                color_img[step_i, cam_i] = color_img_i
                depth_img[step_i, cam_i] = depth_img_i

        return object_cate, object_id, obj_state, dof_change, lang_prompt, positions, rotations, color_img, depth_img, color_voxel, depth_voxel




# TODO wrote apply transform function
def apply_transform(voxel_grid, orientation, position):
    angle = np.pi / 2.0 * np.random.choice(4)
    R_augment = Rotation.from_rotvec(np.r_[0.0, 0.0, angle])

    z_offset = np.random.uniform(6, 34) - position[2]

    t_augment = np.r_[0.0, 0.0, z_offset]
    T_augment = Transform(R_augment, t_augment)

    T_center = Transform(Rotation.identity(), np.r_[20.0, 20.0, 20.0])
    T = T_center * T_augment * T_center.inverse()

    # transform voxel grid
    T_inv = T.inverse()
    matrix, offset = T_inv.rotation.as_matrix(), T_inv.translation
    voxel_grid[0] = ndimage.affine_transform(voxel_grid[0], matrix, offset, order=0)

    # transform grasp pose
    position = T.transform_point(position)
    orientation = T.rotation * orientation

    return voxel_grid, orientation, position


def sample_point_cloud(pc, num_point, return_idx=False):
    num_point_all = pc.shape[0]
    idxs = np.random.choice(np.arange(num_point_all), size=(num_point,), replace=num_point > num_point_all)
    if return_idx:
        return pc[idxs], idxs
    else:
        return pc[idxs]
