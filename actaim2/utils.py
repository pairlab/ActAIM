import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils import tensorboard
import torch.nn as nn
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
# import open3d as o3d
from sklearn.manifold import TSNE

from affordance.vision.fusion import *
from new_scripts.model.rvt.renderer import BoxRenderer

import pdb

###############################################
# dataset root directory
dataset_root = "./dataset_all"

# Set the camera extrinsic here

fu_list = [2.0, 2.0, 2.0, 2.0, 2.0]
fv_list = [2.0, 2.0, 2.0, 2.0, 2.0]

franka_start_pos = np.array([-1.2, 0.0, 0.0])
workspace_r = 0.5
workspace_R = 1.4

tsdf_vol_bnds = np.array([[-0.9, 0.3], [-0.6, 0.6], [0.0, 1.2]])
num_voxel_per_len = 100

'''
view_matrices = [np.matrix([[-0.624447, -0.186822,  0.758395,  0.      ],
                        [ 0.781067, -0.149361,  0.606322,  0.      ],
                        [ 0.      ,  0.97097pr3,  0.239188,  0.      ],
                        [-0.121821, -0.493237, -1.247419,  1.      ]], dtype=np.float32),
                 np.matrix([[-0.782851,  0.183599, -0.594505,  0.      ],
                        [-0.62221 , -0.231001,  0.747993,  0.      ],
                        [ 0.      ,  0.955474,  0.295077,  0.      ],
                        [-0.124949, -0.420195, -1.273578,  1.      ]], dtype=np.float32),
                 np.matrix([[ 0.796113,  0.174079, -0.579569,  0.      ],
                        [-0.605148,  0.229013, -0.762462,  0.      ],
                        [ 0.      ,  0.957731,  0.287663,  0.      ],
                        [ 0.148536, -0.430902, -1.267453,  1.      ]], dtype=np.float32),
                 np.matrix([[ 0.641367, -0.180445,  0.745713,  0.      ],
                        [ 0.767234,  0.150842, -0.623376,  0.      ],
                        [-0.      ,  0.97195 ,  0.235189,  0.      ],
                        [ 0.097902, -0.497804, -1.247711,  1.      ]], dtype=np.float32),
                 np.matrix([[ 0.063312, -0.966811,  0.247524,  0.      ],
                        [ 0.997994,  0.061334, -0.015703,  0.      ],
                        [-0.      ,  0.248021,  0.968755,  0.      ],
                        [-0.006331, -0.28708 , -1.523695,  1.      ]], dtype=np.float32)]
'''
###############################################

def visualize_vec(image_epoch_root, task_embed, task_ind, name=None):
    if name is not None:
        save_image_name = image_epoch_root + "/{}_{}.png".format(name, task_ind)
    else:
        save_image_name = image_epoch_root + "/vec_{}.png".format(task_ind)

    # Reshape the vector into a 2D grid (e.g., 1 row, 640 columns)
    grid_size = (20, -1)  # -1 infers the size based on the length of the tensor
    grid_tensor = task_embed.view(*grid_size)

    # Convert the tensor to a NumPy array
    grid_array = grid_tensor.detach().cpu().numpy()

    # Set up the figure and axis
    fig, ax = plt.subplots()

    # Visualize the grid as an image with color bar
    img = ax.imshow(grid_array, cmap='viridis', aspect='auto', interpolation='nearest')

    # Add a color bar
    # cbar = plt.colorbar(img, ax=ax, orientation='horizontal', pad=0.1)
    # cbar.set_label('Value')

    # Save the image
    plt.savefig(save_image_name, bbox_inches='tight')


def draw_epoch_img_tensor(rgb, q_trans, image_epoch_root, traj_step, epoch=0, mode="unsup"):
    rgb = rgb.permute((0, 2, 3, 1)) * 255

    softmax_layer = nn.Softmax()

    # draw image with distribution keypointdraw
    num_cam = q_trans.shape[0]
    for i in range(num_cam):
        epoch_save_image_name = image_epoch_root + "/image_epoch_{}_cam_{}_{}_task_{}.png".format(epoch, i, mode, traj_step)
        epoch_save_raw_name = image_epoch_root + "/raw_epoch_{}_cam_{}_{}_task_{}.png".format(epoch, i, mode,
                                                                                                  traj_step)

        epoch_save_heatmap_name = image_epoch_root + "/heatmap_epoch_{}_cam_{}_{}_task_{}.png".format(epoch, i, mode,
                                                                                              traj_step)
        q_trans_i = q_trans[i]

        min_val = q_trans_i.min()
        max_val = q_trans_i.max()
        # q_trans_i_norm = ((q_trans_i - min_val) / (max_val - min_val)).squeeze()
        # q_trans_i_norm = softmax_layer(q_trans_i_norm).detach()

        q_trans_i_norm = F.softmax(q_trans_i.view(320 * 320)).view(320, 320) * 320 * 320

        # TODO
        # softmax first and linear norm

        # also save depth image
        depth_epoch_save_image_name = image_epoch_root + "/depth_epoch_{}_cam_{}_{}_task_{}.png".format(epoch, i, mode, traj_step)
        depth_image_np = q_trans_i_norm.detach().cpu().numpy() * 255
        colored_heatmap = plt.get_cmap('viridis')(depth_image_np)
        plt.imsave(epoch_save_heatmap_name, colored_heatmap)

        depth_image_pil = Image.fromarray((q_trans_i_norm.detach().cpu().numpy() * 255).astype('uint8'))
        depth_image_pil.save(depth_epoch_save_image_name)

        rgb_i = (rgb[i]).cpu().numpy()
        q_trans_i_plot = (1 - q_trans_i_norm).detach().cpu().numpy()

        # Normalize the heatmap values to the range [0, 255]
        heatmap = (q_trans_i_plot * 255).astype(np.uint8)

        h, w, c = rgb_i.shape

        # Create an RGBA image by combining RGB and heatmap as alpha channel
        depth_heatmap = np.zeros((h, w, 3), dtype=np.uint8)

        depth_heatmap[..., 0] = heatmap  # Red channel
        depth_heatmap[..., 1] = heatmap  # Green channel
        depth_heatmap[..., 2] = heatmap  # Blue channel

        rgb_i = rgb_i.astype(np.uint8)

        # make the background brighter
        import cv2
        hsv = cv2.cvtColor(rgb_i, cv2.COLOR_BGR2HSV)
        dark_background_mask = hsv[:, :, 2] < 5
        brightness_increase = 50  # Adjust this value to control the brightness increase
        hsv[:, :, 2] = np.where(dark_background_mask, hsv[:, :, 2] + brightness_increase, hsv[:, :, 2])

        new_background_color = (204,204,255)
        rgb_i = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        rgb_i[dark_background_mask] = new_background_color

        final_image = (0.5 * rgb_i + 0.5 * depth_heatmap).astype(np.uint8)
        plt.imsave(epoch_save_image_name, final_image)
        plt.imsave(epoch_save_raw_name, rgb_i)


        # q_trans_i_plot = q_trans_i_plot[..., np.newaxis]
        # ones_img = np.ones_like(q_trans_i_plot)
        # heatmap_i = np.concatenate([q_trans_i_plot, ones_img, ones_img], axis=-1)
        # rgb_i_coords = (rgb_i * heatmap_i).astype(np.uint8)
        # plt.imsave(epoch_save_image_name, heatmap_i)


def visualize_tsne_plot(x, y, name=None, dir=None):
    X_embedded = TSNE(n_components=2).fit_transform(x)
    colors = ['red' if label == 0 else 'blue' for label in y]
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=colors)
    plt.title("Latent Space Representation Estimated by cVAE")
    if name is None:
        file_name = "tsne_vis.png"
    else:
        file_name = "tsne_vis_" + name + ".png"

    if dir is None:
        plt.savefig(file_name)
    else:
        save_name = dir + "/" + file_name
        plt.savefig(save_name)
    plt.close()


def generate_grid_coords(size, bnd, device):
    # Generate voxel coordinates
    x_coords, y_coords, z_coords = torch.meshgrid(
        torch.arange(size),
        torch.arange(size),
        torch.arange(size)
    )

    # Flatten the coordinates into 1D tensors
    x_coords = x_coords.flatten()
    y_coords = y_coords.flatten()
    z_coords = z_coords.flatten()

    # Stack the coordinates to create a tensor with (x, y, z) for each voxel
    voxel_coords = torch.stack((x_coords, y_coords, z_coords), dim=1).float().to(device)
    voxel_coords /= size
    bnd = bnd.to(device)

    bnd_min = bnd[:, 0]
    bnd_max = bnd[:, 1]
    bnd_size = bnd_max - bnd_min

    voxel_coords = voxel_coords * bnd_size + bnd_min
    mask = points_inside_workspace(voxel_coords, device)
    valid_voxel_coords = voxel_coords[mask]
    return valid_voxel_coords

def points_inside_workspace(points, device):
    num_points = points.shape[0]
    franka_base_tensor = torch.Tensor(franka_start_pos).to(device)

    points_to_base_dist = torch.norm(points - franka_base_tensor, dim=-1)
    points_not_too_close_mask = points_to_base_dist > workspace_r
    points_not_too_far_mask = points_to_base_dist < workspace_R

    mask_workspace = torch.logical_and(points_not_too_close_mask, points_not_too_far_mask)
    return mask_workspace

def normalize_tensor(tensor):
    tensor_max = tensor.max()
    tensor_min = tensor.min()
    tensor_norm = (tensor - tensor_min) / (tensor_max - tensor_min)

    return tensor_norm

def compute_trans(trans, view_matrices, bnd, size, img_width, img_height, device):
    # compute the action point
    voxel_coords = generate_grid_coords(size, bnd, device)
    num_cam = len(view_matrices)
    num_points = voxel_coords.shape[0]
    trans = trans.squeeze()

    points_score = torch.zeros(num_points).to(device)
    mask = torch.ones(num_points).to(device)

    for i in range(num_cam):
        view_matrices_i = view_matrices[i]
        view_matrices_i = torch.Tensor(view_matrices_i).to(device)
        trans_i = trans[i]
        trans_norm_i = normalize_tensor(trans_i)

        x_inv, y_inv, mask_i, depth_weight_i = compute_2d_coords(voxel_coords, view_matrices_i, img_width, img_height, device)

        img_weight_i = trans_norm_i[x_inv, y_inv]

        # consider how to add img weight
        # points_score_i = img_weight_i * depth_weight_i
        points_score_i = img_weight_i

        points_score += points_score_i

        mask = torch.logical_and(mask, mask_i)

    valid_points = voxel_coords[mask]
    valid_points_score = points_score[mask]

    return valid_points, valid_points_score

def get_wpt(renderer, trans, dyn_cam_info=None, y_q=None):
    """
    Estimate the q-values given output from mvt
    :param out: output from mvt
    """

    bs, nc, h, w = trans.shape

    q_trans = trans.view(bs, nc, h * w)
    hm = torch.nn.functional.softmax(q_trans, 2)
    hm = hm.view(bs, nc, h, w)

    if dyn_cam_info is None:
        dyn_cam_info_itr = (None,) * bs
    else:
        dyn_cam_info_itr = dyn_cam_info
    pred_wpt = [
        renderer.get_max_3d_frm_hm_cube(
            hm[i: i + 1],
            fix_cam=True,
            dyn_cam_info=dyn_cam_info_itr[i: i + 1]
            if not (dyn_cam_info_itr[i] is None)
            else None,
        )
        for i in range(bs)
    ]
    pred_wpt = torch.cat(pred_wpt, 0)

    assert y_q is None

    return pred_wpt


def generate_hm_from_pt(pt, res, sigma, thres_sigma_times=3):
    """
    Pytorch code to generate heatmaps from point. Points with values less than
    thres are made 0
    :type pt: torch.FloatTensor of size (num_pt, 2)
    :type res: int or (int, int)
    :param sigma: the std of the gaussian distribition. if it is -1, we
        generate a hm with one hot vector
    :type sigma: float
    :type thres: float
    """

    num_pt, x = pt.shape
    assert x == 2

    if isinstance(res, int):
        resx = resy = res
    else:
        resx, resy = res

    _hmx = torch.arange(0, resy).to(pt.device)
    _hmx = _hmx.view([1, resy]).repeat(resx, 1).view([resx, resy, 1])
    _hmy = torch.arange(0, resx).to(pt.device)
    _hmy = _hmy.view([resx, 1]).repeat(1, resy).view([resx, resy, 1])
    hm = torch.cat([_hmx, _hmy], dim=-1)
    hm = hm.view([1, resx, resy, 2]).repeat(num_pt, 1, 1, 1)

    pt = pt.view([num_pt, 1, 1, 2])
    hm = torch.exp(-1 * torch.sum((hm - pt) ** 2, -1) / (2 * (sigma**2)))
    thres = np.exp(-1 * (thres_sigma_times**2) / 2)
    hm[hm < thres] = 0.0

    hm /= torch.sum(hm, (1, 2), keepdim=True) + 1e-6

    # TODO: make a more efficient version
    if sigma == -1:
        _hm = hm.view(num_pt, resx * resy)
        hm = torch.zeros((num_pt, resx * resy), device=hm.device)
        temp = torch.arange(num_pt).to(hm.device)
        hm[temp, _hm.argmax(-1)] = 1

    return hm


def compute_2d_coords(points, view_matrix, img_width, img_height, device):
    fu = fu_list[0]
    fv = fv_list[0]

    num_points = points.shape[0]
    vin = view_matrix
    add_one = torch.ones(num_points).to(device).unsqueeze(1)

    pos_one = torch.cat((points, add_one), 1)
    big_mat_inv = torch.matmul(pos_one, vin)

    proj_u_inv = big_mat_inv[:, 0]
    proj_v_inv = big_mat_inv[:, 1]
    depth_img_inv = big_mat_inv[:, 2]

    u_range_inv = proj_u_inv / (fu * depth_img_inv)
    v_range_inv = proj_v_inv / (fv * depth_img_inv)

    x_inv = torch.round(img_width * (-u_range_inv)) + img_width * 0.5
    y_inv = torch.round(img_height * v_range_inv) + img_height * 0.5

    x_mask = torch.logical_and(x_inv >= 0, x_inv < img_width)
    y_mask = torch.logical_and(y_inv >= 0, y_inv < img_height)

    mask = torch.logical_and(x_mask, y_mask)
    mask_inv = ~mask
    x_inv[mask_inv] = 0
    y_inv[mask_inv] = 0

    return x_inv.long(), y_inv.long(), mask, depth_img_inv


def load_view_matrices_batch(object_id, init_state, traj_id, is_rvt=True, dataset="dataset_all"):
    bs = init_state.shape[0]
    view_matrices_list = []
    for b in range(bs):
        b_object_id = object_id[b]
        b_init_state = init_state[b]
        b_traj_id = traj_id[b]
        matrices_type_name = "matrices_rvt" if is_rvt else "matrices"
        b_view_matrices_name_list = [matrices_type_name, b_object_id.item(), b_init_state.item(), b_traj_id]
        b_view_matrices_name = "_".join([str(data) for data in b_view_matrices_name_list])
        b_view_matrices_file = dataset + "/" + str(b_object_id.item()) + "/" + b_view_matrices_name + ".npz"
        b_matrices = np.load(b_view_matrices_file)["matrix"]

        view_matrices_list.append(b_matrices)

    return view_matrices_list

def compute_pcd_list(curr_depth_img, curr_color_img, view_matrices, device):

    bs = curr_depth_img.shape[0]
    num_camera = curr_depth_img.shape[1]
    assert num_camera == 5

    cam_width = curr_depth_img.shape[2]
    cam_height = curr_depth_img.shape[3]

    u_range = -torch.arange(-0.5, 0.5, 1 / cam_width).view(1, -1).to(device)
    v_range = torch.arange(-0.5, 0.5, 1 / cam_height).view(-1, 1).to(device)

    img_input_tensor = torch.Tensor().to(device)

    # save pcd and pcd_feat for novel view generation
    pcd_tensor = torch.Tensor().to(device)
    pcd_feat_tensor = torch.Tensor().to(device)

    for b in range(bs):
        depth_img_batch = curr_depth_img[b, ...]
        color_img_batch = curr_color_img[b, ...]
        img_batch_tensor = torch.Tensor().to(device)

        pcd_tensor_batch = torch.Tensor().to(device)
        pcd_feat_tensor_batch = torch.Tensor().to(device)

        for i in range(num_camera):
            depth_img = depth_img_batch[i, ...]
            color_img = color_img_batch[i, ...]

            color_img = color_img[:, :, :3]

            color_img = color_img / 255.0
            # color_img shape torch.Size([320, 320, 3])
            # depth_img shape torch.Size([320, 320])

            ones = torch.ones_like(depth_img)
            proj_u = fu_list[i] * torch.mul(depth_img, u_range)
            proj_v = fv_list[i] * torch.mul(depth_img, v_range)

            big_mat = torch.cat(
                (proj_u.unsqueeze(-1), proj_v.unsqueeze(-1), depth_img.unsqueeze(-1), ones.unsqueeze(-1)), dim=2
            )

            vin = np.linalg.inv(view_matrices[b][i])
            vinv = torch.from_numpy(vin).to(device)
            max_depth = 20
            mask = depth_img > -max_depth
            pc_img = torch.matmul(big_mat, vinv)

            # mask out outlier
            pc_img[~mask] = 0
            color_img[~mask] = 0
            depth_img[~mask] = 0
            depth_img = depth_img.unsqueeze(-1)

            pc_xyz = pc_img[:, :, :3]

            valid_pcd = pc_xyz[mask].view(-1, 3)
            valid_pcd_feat = color_img[mask].view(-1, 3)
            pcd_tensor_batch = torch.cat((pcd_tensor_batch, valid_pcd), 0)
            pcd_feat_tensor_batch = torch.cat((pcd_feat_tensor_batch, valid_pcd_feat), 0)

            '''
            pc_xyz_vis = pc_xyz.view(-1, 3)
            pc_xyz_rgb = color_img.view(-1, 3)
            points = pc_xyz_vis.cpu().numpy()
            points_rgb = pc_xyz_rgb.cpu().numpy()
            pcd = o3d.open3d.geometry.PointCloud()
            pcd.points = o3d.open3d.utility.Vector3dVector(points)
            pcd.colors = o3d.open3d.utility.Vector3dVector(points_rgb)
            o3d.visualization.draw_geometries([pcd])
            '''

            img_input = torch.cat((color_img, depth_img, pc_xyz), dim=-1).unsqueeze(0)
            img_batch_tensor = torch.cat((img_batch_tensor, img_input), dim=0)

        pcd_tensor = torch.cat((pcd_tensor, pcd_tensor_batch.unsqueeze(0)), 0)
        pcd_feat_tensor = torch.cat((pcd_feat_tensor, pcd_feat_tensor_batch.unsqueeze(0)), 0)

        img_input_tensor = torch.cat((img_input_tensor, img_batch_tensor.unsqueeze(0)), dim=0)

    img_input_tensor = img_input_tensor.permute(0, 1, 4, 2, 3)

    return img_input_tensor, pcd_tensor, pcd_feat_tensor


def place_pc_in_cube(
    pc, app_pc=None, with_mean_or_bounds=True, scene_bounds=None, no_op=False
):
    """
    calculate the transformation that would place the point cloud (pc) inside a
        cube of size (2, 2, 2). The pc is centered at mean if with_mean_or_bounds
        is True. If with_mean_or_bounds is False, pc is centered around the mid
        point of the bounds. The transformation is applied to point cloud app_pc if
        it is not None. If app_pc is None, the transformation is applied on pc.
    :param pc: pc of shape (num_points_1, 3)
    :param app_pc:
        Either
        - pc of shape (num_points_2, 3)
        - None
    :param with_mean_or_bounds:
        Either:
            True: pc is centered around its mean
            False: pc is centered around the center of the scene bounds
    :param scene_bounds: [x_min, y_min, z_min, x_max, y_max, z_max]
    :param no_op: if no_op, then this function does not do any operation
    """
    if no_op:
        if app_pc is None:
            app_pc = torch.clone(pc)

        return app_pc, lambda x: x

    if with_mean_or_bounds:
        assert scene_bounds is None
    else:
        assert not (scene_bounds is None)
    if with_mean_or_bounds:
        pc_mid = (torch.max(pc, 0)[0] + torch.min(pc, 0)[0]) / 2
        x_len, y_len, z_len = torch.max(pc, 0)[0] - torch.min(pc, 0)[0]
    else:
        x_min, y_min, z_min, x_max, y_max, z_max = scene_bounds
        pc_mid = torch.tensor(
            [
                (x_min + x_max) / 2,
                (y_min + y_max) / 2,
                (z_min + z_max) / 2,
            ]
        ).to(pc.device)
        x_len, y_len, z_len = x_max - x_min, y_max - y_min, z_max - z_min

    scale = 2 / max(x_len, y_len, z_len)
    if app_pc is None:
        app_pc = torch.clone(pc)
    app_pc = (app_pc - pc_mid) * scale

    # reverse transformation to obtain app_pc in original frame
    def rev_trans(x):
        return (x / scale) + pc_mid

    return app_pc, rev_trans


def render(renderer, pc, img_feat, img_aug, device, dyn_cam_info=None,):
    add_corr = True
    _place_with_mean = True
    scene_bounds = None

    # preprcess pc
    pc = [
        place_pc_in_cube(
            _pc,
            with_mean_or_bounds=_place_with_mean,
            scene_bounds=None if _place_with_mean else scene_bounds,
        )[0]
        for _pc in pc
    ]

    with torch.no_grad():
        if dyn_cam_info is None:
            dyn_cam_info_itr = (None,) * len(pc)
        else:
            dyn_cam_info_itr = dyn_cam_info

        if add_corr:
            img = [
                renderer(
                    _pc,
                    torch.cat((_pc, _img_feat), dim=-1),
                    fix_cam=True,
                    dyn_cam_info=(_dyn_cam_info,)
                    if not (_dyn_cam_info is None)
                    else None,
                ).unsqueeze(0)
                for (_pc, _img_feat, _dyn_cam_info) in zip(
                    pc, img_feat, dyn_cam_info_itr
                )
            ]
        else:
            img = [
                renderer(
                    _pc,
                    _img_feat,
                    fix_cam=True,
                    dyn_cam_info=(_dyn_cam_info,)
                    if not (_dyn_cam_info is None)
                    else None,
                ).unsqueeze(0)
                for (_pc, _img_feat, _dyn_cam_info) in zip(
                    pc, img_feat, dyn_cam_info_itr
                )
            ]

        img = torch.cat(img, 0)
        img = img.permute(0, 1, 4, 2, 3)

        # for visualization purposes
        if add_corr:
            vis_img = img[:, :, 3:-1, ...].clone().detach()
        else:
            vis_img = img.clone().detach()

        # image augmentation
        if img_aug != 0:
            stdv = img_aug * torch.rand(1, device=img.device)
            # values in [-stdv, stdv]
            noise = stdv * ((2 * torch.rand(*img.shape, device=img.device)) - 1)
            img = torch.clamp(img + noise, -1, 1)

        '''
        if mvt.add_pixel_loc:
            bs = img.shape[0]
            pixel_loc = mvt.pixel_loc.to(img.device)
            img = torch.cat(
                (img, pixel_loc.unsqueeze(0).repeat(bs, 1, 1, 1, 1)), dim=2
            )
        '''

    '''
    # visualize generated image
    vis_img = vis_img.permute(0, 1, 3, 4, 2)
    vis_img = vis_img[0].cpu().numpy()
    num_img = vis_img.shape[0]
    for i in range(num_img):
        img_i = vis_img[i]
        img_name = "0test{0}.png".format(str(i))
        plt.imsave(img_name, img_i)
    '''
    return img


def transfer_novel_view(renderer, pc_list, pc_feat_list, device):
    img_aug = 0
    obs_input = render(
        renderer,
        pc_list,
        pc_feat_list,
        img_aug,
        device,
        dyn_cam_info=None,
    )

    return obs_input


def mask_pc_robot_body_list(pc, pc_feat):
    franka_base_x = -1.2
    cut_offset = 0.2
    franka_cut_offset = franka_base_x + cut_offset
    bs = pc.shape[0]

    pc_list, pc_feat_list = [], []
    for i in range(bs):
        pc_i = pc[i]
        pc_feat_i = pc_feat[i]
        pc_robot_body_mask = pc_i[:, 0] > franka_cut_offset
        pc_masked_i = pc_i[pc_robot_body_mask, :]
        pc_feat_masked_i = pc_feat_i[pc_robot_body_mask, :]

        pc_list.append(pc_masked_i)
        pc_feat_list.append(pc_feat_masked_i)

    return pc_list, pc_feat_list


def preprocess_data_rvt(data, renderer, device, dataset="dataset_all"):
    # Testing dataloader
    object_cate, object_id, traj_id, init_state, dof_change, lang_prompt, pos, rotations, gripper_open_close, \
        curr_color_voxel, curr_depth_voxel, curr_color_img, curr_depth_img, begin_color_img, begin_depth_img, \
        final_color_voxel, final_depth_voxel, final_color_img, final_depth_img, franka_proprio, step = data

    curr_depth_img = curr_depth_img.to(device).float()
    curr_color_img = curr_color_img.to(device).float()
    begin_color_img = begin_color_img.to(device).float()

    curr_view_matrices = load_view_matrices_batch(object_id, init_state, traj_id, True, dataset)
    img_input, pc, pc_feat = compute_pcd_list(curr_depth_img, curr_color_img, curr_view_matrices, device)

    # TODO filter robot body so image looks clean
    pc_list, pc_feat_list = mask_pc_robot_body_list(pc, pc_feat)


    # saved_pc = pc_list[1]
    # saved_pc_rgb = pc_feat_list[1]
    # import open3d as o3d
    # Create an Open3D point cloud object
    # point_cloud = o3d.geometry.PointCloud()

    # Assign points and colors to the point cloud
    # point_cloud.points = o3d.utility.Vector3dVector(saved_pc.cpu().numpy())
    # point_cloud.colors = o3d.utility.Vector3dVector(saved_pc_rgb.cpu().numpy())

    # Save the point cloud to a file
    # output_filename = "color_point_cloud.ply"  # PLY format supports color information
    # o3d.io.write_point_cloud(output_filename, point_cloud)


    franka_proprio = franka_proprio.to(device).float()
    step = step.to(device).float()

    pos = pos.to(device).float()
    rotations = rotations.to(device).float()
    gripper_open_close = gripper_open_close.to(device).float()

    # TODO not converting dof_change
    dof_change = dof_change.to(device).int().unsqueeze(-1)
    dof_change = process_dof(dof_change, device)

    pos_shape = pos.shape
    pos = pos.reshape(-1, 3)
    pos_list = [pos[i, :] for i in range(pos.shape[0])]

    num_pc = len(pc_list)
    num_pos = len(pos_list)
    if num_pc != num_pos:
        seq_horizon = num_pos // num_pc
        assert seq_horizon == 4
        pc_list = [item for item in pc_list for _ in range(seq_horizon)]

    # Compute local pos
    _place_with_mean = True
    wpt_local = []
    rev_trans = []
    for _pc, _wpt in zip(pc_list, pos_list):

        a, b = place_pc_in_cube(
            _pc,
            _wpt,
            with_mean_or_bounds=_place_with_mean,
            scene_bounds=None,
        )
        wpt_local.append(a.unsqueeze(0))
        rev_trans.append(b)

    wpt_local = torch.cat(wpt_local, axis=0)

    # TODO not converting frame
    # pos = wpt_local.view(pos_shape)

    pc_list = [
        place_pc_in_cube(
            _pc,
            with_mean_or_bounds=_place_with_mean,
            scene_bounds=None,
        )[0]
        for _pc in pc_list
    ]

    # get novel view
    img_input = transfer_novel_view(renderer, pc_list, pc_feat_list, device)

    # determine input and output
    x_batch = (img_input, franka_proprio, lang_prompt)
    y_batch = (pos, rotations, gripper_open_close)

    begin_color_img = preprocess_image(begin_color_img, device)
    # curr_color_img = preprocess_image(curr_color_img, device)
    final_color_img = preprocess_image(final_color_img, device)

    curr_view_matrices = [matrices[np.newaxis, ...] for matrices in curr_view_matrices]
    curr_view_matrices = torch.Tensor(np.concatenate(curr_view_matrices, axis=0)).float().to(device)

    return x_batch, y_batch, begin_color_img, final_color_img, dof_change, step, curr_view_matrices


def preprocess_data_rvt_robot(data, renderer, device, dataset="dataset_all"):
    # Testing dataloader
    object_cate, object_id, traj_id, init_state, dof_change, lang_prompt, pos, rotations, gripper_open_close, \
        curr_color_voxel, curr_depth_voxel, curr_color_img, curr_depth_img, begin_color_img, begin_depth_img, \
        final_color_voxel, final_depth_voxel, final_color_img, final_depth_img, franka_proprio, step = data

    curr_depth_img = curr_depth_img.to(device).float()
    curr_color_img = curr_color_img.to(device).float()
    final_depth_img = final_depth_img.to(device).float()
    final_color_img = final_color_img.to(device).float()

    begin_color_img = begin_color_img.to(device).float()

    curr_view_matrices = load_view_matrices_batch(object_id, init_state, traj_id, False, dataset)

    franka_proprio = franka_proprio.to(device).float()
    step = step.to(device).float()

    pos = pos.to(device).float()
    rotations = rotations.to(device).float()
    gripper_open_close = gripper_open_close.to(device).float()

    # TODO not converting dof_change
    dof_change = dof_change.to(device).int().unsqueeze(-1)
    dof_change = process_dof(dof_change, device)

    img_input = generate_rvt_novel_view(renderer, curr_depth_img, curr_color_img, curr_view_matrices, device)
    final_img_input = generate_rvt_novel_view(renderer, final_depth_img, final_color_img, curr_view_matrices, device)


    # determine input and output
    x_batch = (img_input, franka_proprio, lang_prompt)
    y_batch = (pos, rotations, gripper_open_close)

    # use rvt image as the training data
    begin_color_img = img_input[:, :, 3:-1, ...].permute(0, 1, 3, 4 ,2)
    final_color_img = final_img_input[:, :, 3:-1, ...].permute(0, 1, 3, 4, 2)

    bs = img_input.shape[0]

    ones_tensor = torch.ones(bs, 5, 320, 320, 1).to(device)

    begin_color_img = torch.cat((begin_color_img, ones_tensor), dim=-1) * 255.
    final_color_img = torch.cat((final_color_img, ones_tensor), dim=-1) * 255.

    begin_color_img = preprocess_image(begin_color_img, device) # torch.Size([1, 5, 256, 256, 4])
    final_color_img = preprocess_image(final_color_img, device)

    curr_view_matrices = [matrices[np.newaxis, ...] for matrices in curr_view_matrices]
    curr_view_matrices = torch.Tensor(np.concatenate(curr_view_matrices, axis=0)).float().to(device)

    return x_batch, y_batch, begin_color_img, final_color_img, dof_change, step, curr_view_matrices


def generate_rvt_novel_view(renderer, depth_img, color_img, view_matrices, device):
    img_input, pc, pc_feat = compute_pcd_list(depth_img, color_img, view_matrices, device)
    # TODO filter robot body so image looks clean
    pc_list, pc_feat_list = mask_pc_robot_body_list(pc, pc_feat)

    npz_path = "./pcd.npz"
    pc_full = np.load(npz_path, allow_pickle=True)["pcd"]

    npz_path = "./pcd_color.npz"
    pc_colors = np.load(npz_path, allow_pickle=True)["colors"]


    saved_pc = pc_list[0]
    saved_pc_rgb = pc_feat_list[0]

    # TODO change to real robot color point cloud
    pc_list[0] = torch.Tensor(pc_full).to(device)
    pc_feat_list[0] = torch.Tensor(pc_colors).to(device)

    # import open3d as o3d
    # # Create an Open3D point cloud object
    # point_cloud = o3d.geometry.PointCloud()
    #
    # # Assign points and colors to the point cloud
    # point_cloud.points = o3d.utility.Vector3dVector(saved_pc.cpu().numpy())
    # point_cloud.colors = o3d.utility.Vector3dVector(saved_pc_rgb.cpu().numpy())
    #
    # point_cloud_new = o3d.geometry.PointCloud()
    # # Assign points and colors to the point cloud
    # point_cloud_new.points = o3d.utility.Vector3dVector(pc_full)
    # point_cloud_new.colors = o3d.utility.Vector3dVector(pc_colors)
    #
    # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    #     size=1.0, origin=[0, 0, 0])
    #
    # o3d.visualization.draw_geometries([point_cloud, point_cloud_new, coordinate_frame])
    # pdb.set_trace()


    seq_horizon = 2
    pc_list = [item for item in pc_list for _ in range(seq_horizon)]

    _place_with_mean = True
    pc_list = [
        place_pc_in_cube(
            _pc,
            with_mean_or_bounds=_place_with_mean,
            scene_bounds=None,
        )[0]
        for _pc in pc_list
    ]

    # get novel view
    img_input = transfer_novel_view(renderer, pc_list, pc_feat_list, device)

    # rgb = img_input[0, :, 3:-1, :, :]
    # rgb = rgb.permute((0, 2, 3, 1)) * 255
    # num_cam = rgb.shape[0]
    # for i in range(num_cam):
    #     rgb_i = (rgb[i]).cpu().numpy()
    #     rgb_i = rgb_i.astype(np.uint8)
    #     # make the background brighter
    #     import cv2
    #     hsv = cv2.cvtColor(rgb_i, cv2.COLOR_BGR2HSV)
    #     dark_background_mask = hsv[:, :, 2] < 5
    #     brightness_increase = 50  # Adjust this value to control the brightness increase
    #     hsv[:, :, 2] = np.where(dark_background_mask, hsv[:, :, 2] + brightness_increase, hsv[:, :, 2])
    #
    #     new_background_color = (204, 204, 255)
    #     rgb_i = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    #     rgb_i[dark_background_mask] = new_background_color
    #     raw_image_name = "image" + str(i) + ".png"
    #
    #     plt.imsave(raw_image_name, rgb_i)

    return img_input



def preprocess_data_task(data, device):
    # Testing dataloader
    object_cate, object_id, traj_id, init_state, dof_change, lang_prompt, pos, rotations, gripper_open_close, \
        curr_color_voxel, curr_depth_voxel, curr_color_img, curr_depth_img, begin_color_img, begin_depth_img, \
        final_color_voxel, final_depth_voxel, final_color_img, final_depth_img, franka_proprio, step = data

    begin_color_img = begin_color_img.to(device).float()
    curr_view_matrices = load_view_matrices_batch(object_id, init_state, traj_id)

    # determine input and output
    x_batch = (None, None, lang_prompt)
    y_batch = None

    begin_color_img = preprocess_image(begin_color_img, device)
    # curr_color_img = preprocess_image(curr_color_img, device)
    final_color_img = preprocess_image(final_color_img, device)

    curr_view_matrices = [matrices[np.newaxis, ...] for matrices in curr_view_matrices]
    curr_view_matrices = torch.Tensor(np.concatenate(curr_view_matrices, axis=0)).float().to(device)

    return x_batch, y_batch, begin_color_img, final_color_img, dof_change, step, curr_view_matrices

def process_image_rvt(depth_img, color_img, view_matrices, device):
    depth_img_tensor = torch.from_numpy(depth_img)
    color_img_tensor = torch.from_numpy(color_img)

    depth_img_tensor = depth_img_tensor.to(device).float().unsqueeze(0)
    color_img_tensor = color_img_tensor.to(device).float().unsqueeze(0)

    # add batch size 1
    img_input, pc, pc_feat = compute_pcd_list(depth_img_tensor, color_img_tensor, [view_matrices], device)
    return img_input

def preprocess_data_test(data, device):
    # Testing dataloader
    object_cate, object_id, traj_id, init_state, dof_change, lang_prompt, pos, rotations, gripper_open_close, \
    curr_color_voxel, curr_depth_voxel, curr_color_img, curr_depth_img, begin_color_img, begin_depth_img, \
    final_color_voxel, final_depth_voxel, final_color_img, final_depth_img, franka_proprio, step = data

    curr_color_voxel = curr_color_voxel.to(device).float()
    curr_depth_voxel = curr_depth_voxel.to(device).float()
    franka_proprio = franka_proprio.to(device).float()
    step = step.to(device).float()

    pos = pos.to(device).float()
    rotations = rotations.to(device).float()
    gripper_open_close = gripper_open_close.to(device).float()
    dof_change = dof_change.to(device).float()
    dof_change = dof_change.unsqueeze(-1).repeat(1, 640)

    # concat voxel input
    curr_depth_voxel = curr_depth_voxel.unsqueeze(-1).permute(0, 4, 1, 2, 3)
    curr_color_voxel = curr_color_voxel.permute(0, 4, 1, 2, 3)

    voxel_input = torch.cat((curr_color_voxel, curr_depth_voxel), dim=1)

    # determine input and output
    x_batch = (voxel_input, franka_proprio, lang_prompt)
    y_batch = (pos, rotations, gripper_open_close)

    curr_color_img = curr_color_img.to(device)
    final_color_img = final_color_img.to(device)

    # curr_color_img = preprocess_image(curr_color_img, device)
    # final_color_img = preprocess_image(final_color_img, device)

    return x_batch, y_batch, curr_color_img, final_color_img, dof_change, step

def preprocess_data(data, device):
    # Testing dataloader
    object_cate, object_id, traj_id, init_state, dof_change, lang_prompt, pos, rotations, gripper_open_close, \
    curr_color_voxel, curr_depth_voxel, curr_color_img, curr_depth_img, begin_color_img, begin_depth_img, \
    final_color_voxel, final_depth_voxel, final_color_img, final_depth_img, franka_proprio, step = data

    curr_color_voxel = curr_color_voxel.to(device).float()
    curr_depth_voxel = curr_depth_voxel.to(device).float()
    franka_proprio = franka_proprio.to(device).float()
    step = step.to(device).float()

    pos = pos.to(device).float()
    rotations = rotations.to(device).float()
    gripper_open_close = gripper_open_close.to(device).float()
    dof_change = dof_change.to(device).float()
    dof_change = dof_change.unsqueeze(-1).repeat(1, 640)

    # concat voxel input
    curr_depth_voxel = curr_depth_voxel.unsqueeze(-1).permute(0, 4, 1, 2, 3)
    curr_color_voxel = curr_color_voxel.permute(0, 4, 1, 2, 3)

    voxel_input = torch.cat((curr_color_voxel, curr_depth_voxel), dim=1)

    # determine input and output
    x_batch = (voxel_input, franka_proprio, lang_prompt)
    y_batch = (pos, rotations, gripper_open_close)

    begin_color_img = preprocess_image(begin_color_img, device)
    final_color_img = preprocess_image(final_color_img, device)

    return x_batch, y_batch, begin_color_img, final_color_img, dof_change, step, None


def preprocess_image(images, device):
    images = images / 255.0

    shp = images.shape # [B, v, w,h,c]
    images = images.view((shp[0] * shp[1], *shp[2:]))
    images = images.permute((0, 3, 1, 2))
    # preprocess image size to (1, 3, 224, 224)
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # TODO continue from here
    # need to iterate over to convert PIL
    image_tensor = torch.Tensor().to(device)

    for i in range(shp[0]*shp[1]):
        image_i = images[i]
        image_i = image_i[:3, :, :]
        image_i_tensor = preprocess(image_i)
        image_i_tensor = image_i_tensor.unsqueeze(0).to(device)
        image_tensor = torch.cat((image_tensor, image_i_tensor), dim=0)

    images_tensor = image_tensor.view(shp[0], shp[1], shp[-1] - 1, 224, 224)

    return images_tensor



def add_loss_dict(save_loss_dict, loss_dict, step, mode:str):
    for loss_name, loss_value in loss_dict.items():
        save_loss_name = mode + loss_name
        if save_loss_name not in save_loss_dict.keys():
            if isinstance(loss_value, torch.Tensor):
                loss_value = loss_value.item()
            save_loss_dict[save_loss_name] = loss_value
        else:
            current_value = save_loss_dict[save_loss_name]
            if isinstance(loss_value, torch.Tensor):
                loss_value = loss_value.item()
            save_loss_dict[save_loss_name] = ((current_value * step) + loss_value) / (step + 1)

def find_earliest_checkpoint(checkpoint_list, description):
    # checkpoint_form = "net={},time={},batch_size={},lr={:.0e},train_task={},gpu_num={},slurm_id={}"
    description_split_list = description.split(",")
    for f_name in checkpoint_list:
        f_name_split_list = f_name.split(",")
        checkpoint_match = True
        for i in range(len(description_split_list)):
            if len(f_name_split_list) == len(description_split_list):
                item_match = (f_name_split_list[i] == description_split_list[i])
                if i == 1:
                    item_match = not item_match
                checkpoint_match = item_match and checkpoint_match
            else:
                checkpoint_match = False
        if checkpoint_match:
            return f_name
    return None

def process_dof(dof_change, device):
    dof_digit = [[int(digit) for digit in str(number.item())] for numbers in dof_change for number in numbers]

    new_dof_change_list = []
    for b in range(len(dof_digit)):
        dof_digit_i = dof_digit[b]
        repeat_time = 640 // len(dof_digit_i)
        repeat_dof_digit_i = dof_digit_i * repeat_time
        while len(repeat_dof_digit_i) < 640:
            repeat_dof_digit_i.extend(dof_digit_i)
        repeat_dof_digit_i = repeat_dof_digit_i[:640]

        new_dof_change_list.append(repeat_dof_digit_i)

    dof_change = torch.Tensor(new_dof_change_list).to(device).float()
    return dof_change