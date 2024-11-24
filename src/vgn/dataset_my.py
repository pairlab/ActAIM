import numpy as np
from scipy import ndimage
import torch.utils.data
from pathlib import Path
import pandas as pd
import os
import math
import open3d as o3d

from vgn.io import *
from vgn.perception import *
from vgn.utils.transform import Rotation, Transform
from vgn.utils.implicit import get_scene_from_mesh_pose_list

def downsample_pc(pc, voxel_size=0.008, pc_num = 1024):
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

class DatasetVoxel(torch.utils.data.Dataset):
    def __init__(self, pair, num_point=2048, augment=False, cvae=False, root="./dataset", pc=False):
        self.root = root
        self.num_point = num_point
        self.num_th = 32
        self.is_pair = pair
        self.is_cvae = cvae
        # self.df = pd.read_csv(self.root + "/19179_inter.csv")
        self.df = pd.read_csv(self.root + "/interact.csv")
        self.df.fillna('', inplace=True)
        # self.tsdf_vol_bnds = np.array([[-1.5, 1.5], [-1., 1.], [0, 2]])
        self.tsdf_vol_bnds = np.array([[-1.5, 0.5], [-1.0, 1.0], [0, 2]])
        self.pc = pc


    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, i):
        object_id = self.df.loc[i, "object_id"]
        ori = Rotation.from_quat(self.df.loc[i, "qx":"qw"].to_numpy(np.float64))
        pos = self.df.loc[i, "x":"z"].to_numpy(np.float64)
        force = self.df.loc[i, "fx":"fz"].to_numpy(np.float64)
        label = self.df.loc[i, "label"].astype(np.long)
        augment = self.df.loc[i, "augment"].astype(np.long)
        camera_id = self.df.loc[i, "camera_id"]

        # if self.is_cvae:
        #     if label == 0:
        #         pass
        if "_" in str(object_id):
            object = object_id.split("_")[0]
        else:
            object = str(object_id)

        voxel_path = self.root + "/" + object + "/" + str(object_id) + ".npz"
        voxel_grid = np.load(voxel_path)["grid"]

        tsdf_voxel_size = self.tsdf_vol_bnds[:, 1] - self.tsdf_vol_bnds[:, 0]
        pos = (pos - self.tsdf_vol_bnds[:, 0]) / tsdf_voxel_size

        # TODO should normalize it to [-0.5,0.5]
        pos -= 0.5

        rotations = np.empty((2, 4), dtype=np.float64)
        R = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])
        rotations[0] = ori.as_quat()
        rotations[1] = (ori * R).as_quat()

        # TODO save dof change for visualization
        dof = self.df.loc[i, "dof"]

        if self.is_pair:
            x, y = (voxel_grid, rotations, force), label
        else:
            x, y = voxel_grid, (label, rotations, force)

        depth_img = None
        if self.is_cvae:
            depth_path = self.root + "/" + object + "/" + str(object_id) + "_" + str(camera_id) + "_depth.npz"
            depth_img = np.load(depth_path)["grid"]
            y = (y, depth_img)

        obs_file = self.df.loc[i, "obs_file"]


        if len(obs_file) > 1:
            obs_file_path = self.root + "/" + object + "/" + str(obs_file)
            obs_depth = np.load(obs_file_path)["grid"]
        else:
            obs_depth = depth_img[0, :, :]

        if self.pc:
            num_point = 4096
            pc_path = self.root + "/" + object + "/" + str(object_id) + "_" + str(camera_id) + "_pc.npz"
            pc = np.load(pc_path)["grid"]

            # dist = np.linalg.norm(pc - pos, axis=1)
            # ind = np.argpartition(dist, -num_point)[-num_point:]
            pc = downsample_pc(pc)
            # pc = pc[ind]

            return pc, y, pos, dof, obs_depth, object_id, camera_id, augment

        return x, y, pos, dof, obs_depth, object_id, camera_id, augment

    def get_mesh(self, idx):
        scene_id = self.df.loc[idx, "scene_id"]
        mesh_pose_list_path = self.raw_root / "mesh_pose_list" / (scene_id + ".npz")
        mesh_pose_list = np.load(mesh_pose_list_path, allow_pickle=True)["pc"]
        scene = get_scene_from_mesh_pose_list(mesh_pose_list, return_list=False)
        return scene


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
