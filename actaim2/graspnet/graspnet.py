""" Demo to show prediction results.
    Author: chenxi-wang
"""

import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image

import torch
from graspnetAPI import GraspGroup

# graspnet dir
ROOT_DIR = "/home/licho/workspace/graspnet-baseline"
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
# from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

### config for graspnet ###
checkpoint_path = "/home/licho/workspace/graspnet-baseline/logs/checkpoint-rs.tar"
num_point_input = 20000
num_view = 300
collision_thresh = 0.001
voxel_size = 0.01

### ###

def get_net(device):
    # Init the model
    net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    # print("-> loaded checkpoint %s (epoch: %d)"%(checkpoint_path, start_epoch))
    # set model to eval mode
    net.eval()
    return net

def preprocess_data(pc, pc_rgb, device):
    # sample points
    assert pc.shape[0] == pc_rgb.shape[0]
    num_point = pc.shape[0]

    if num_point >= num_point_input:
        idxs = np.random.choice(num_point, num_point_input, replace=False)
    else:
        idxs1 = np.arange(num_point)
        idxs2 = np.random.choice(num_point, num_point_input-num_point, replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    pc_sampled = pc[idxs]
    pc_rgb_sampled = pc_rgb[idxs]
    # convert data
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pc_sampled.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(pc_rgb_sampled.astype(np.float32))
    end_points = dict()
    cloud_sampled = torch.from_numpy(pc_sampled[np.newaxis].astype(np.float32))
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = pc_rgb

    return end_points, cloud

def predict_grasp(pc, pc_rgb, device):
    graspnet = get_net(device)
    end_points, cloud = preprocess_data(pc, pc_rgb, device)

    # Forward pass
    with torch.no_grad():
        end_points = graspnet(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    if collision_thresh > 0:
        gg = collision_detection(gg, np.array(cloud.points))
    return gg, cloud


def get_and_process_data(data_dir):
    # load data
    color = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
    depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
    workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))
    meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
    intrinsic = meta['intrinsic_matrix']
    factor_depth = meta['factor_depth']

    # generate cloud
    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # get valid points
    mask = (workspace_mask & (depth > 0))
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    # sample points
    if len(cloud_masked) >= num_point_input:
        idxs = np.random.choice(len(cloud_masked), num_point_input, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), num_point_input-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # convert data
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud

def get_grasps(net, end_points):
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg

def collision_detection(gg, cloud):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
    gg = gg[~collision_mask]
    return gg

def vis_grasps(gg, cloud):
    gg.nms()
    gg.sort_by_score()
    gg = gg[:50]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])

def demo(data_dir):
    net = get_net()
    end_points, cloud = get_and_process_data(data_dir)
    gg = get_grasps(net, end_points)
    if collision_thresh > 0:
        gg = collision_detection(gg, np.array(cloud.points))

    vis_grasps(gg, cloud)
