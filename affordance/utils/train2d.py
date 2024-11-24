import torch
import numpy as np
import pdb

# Some parameters for P2G and G2P
cam_width, cam_height = 320, 320
vin0 = np.matrix(
    [
        [-0.7071, 0.2416, -0.6645, 0.0000],
        [-0.7071, -0.2416, 0.6645, 0.0000],
        [0.0000, 0.9398, 0.3417, 0.0000],
        [-0.0000, -0.9398, -2.6827, 1.0000],
    ],
    dtype=np.float32,
)

vin1 = np.matrix(
    [
        [-0.3827, 0.3157, -0.8683, 0.0000],
        [-0.9239, -0.1308, 0.3596, 0.0000],
        [0.0000, 0.9398, 0.3417, 0.0000],
        [-0.0000, -0.9398, -2.6827, 1.0000],
    ],
    dtype=np.float32,
)

vin2 = np.matrix(
    [
        [0.0000, 0.3417, -0.9398, 0.0000],
        [-1.0000, 0.0000, -0.0000, 0.0000],
        [0.0000, 0.9398, 0.3417, 0.0000],
        [-0.0000, -0.9398, -2.6827, 1.0000],
    ],
    dtype=np.float32,
)

vin3 = np.matrix(
    [
        [0.3827, 0.3157, -0.8683, 0.0000],
        [-0.9239, 0.1308, -0.3596, 0.0000],
        [0.0000, 0.9398, 0.3417, 0.0000],
        [-0.0000, -0.9398, -2.6827, 1.0000],
    ],
    dtype=np.float32,
)

vin4 = np.matrix(
    [
        [0.7071, 0.2416, -0.6645, 0.0000],
        [-0.7071, 0.2416, -0.6645, 0.0000],
        [0.0000, 0.9398, 0.3417, 0.0000],
        [-0.0000, -0.9398, -2.6827, 1.0000],
    ],
    dtype=np.float32,
)

proj = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32
)

vin = [vin0, vin1, vin2, vin3, vin4]

tsdf_vol_bnds = np.array([[-1.5, 0.5], [-1.0, 1.0], [0, 2]])


# Grid to Pixel
def G2P(pos, camera_id):
    fu = 2 / proj[0, 0]
    fv = 2 / proj[1, 1]

    current_vin = vin[camera_id]

    vin_matrix = torch.from_numpy(current_vin).cuda()
    add_ones = torch.ones(pos.shape[0]).unsqueeze(1).cuda()
    point_ones = torch.cat((pos, add_ones), dim=1)

    big_mat_inv = torch.matmul(point_ones, vin_matrix)

    proj_u_inv = big_mat_inv[:, 0]
    proj_v_inv = big_mat_inv[:, 1]
    depth_img_inv = big_mat_inv[:, 2]

    u_range_inv = proj_u_inv / (fu * depth_img_inv)
    v_range_inv = proj_v_inv / (fv * depth_img_inv)

    u_inv = torch.round(cam_width * (-u_range_inv)) + cam_width * 0.5
    v_inv = torch.round(cam_height * v_range_inv) + cam_height * 0.5

    xy_pixel = torch.stack((v_inv, u_inv), dim=-1)
    return xy_pixel


# Pixel to Grid
def P2G(pixel, depth_image, camera_id):
    current_vin = vin[camera_id]

    u_range = -torch.arange(-0.5, 0.5, 1 / cam_width).view(1, -1).cuda()
    v_range = torch.arange(-0.5, 0.5, 1 / cam_height).view(-1, 1).cuda()
    fu = 2 / proj[0, 0]
    fv = 2 / proj[1, 1]

    proj_u = fu * torch.mul(depth_image, u_range)
    proj_v = fv * torch.mul(depth_image, v_range)

    ones = torch.ones_like(depth_image)
    big_mat = torch.cat(
        (proj_u.unsqueeze(-1), proj_v.unsqueeze(-1), depth_image.unsqueeze(-1), ones.unsqueeze(-1)), dim=-1
    )
    vinv = torch.from_numpy(np.linalg.inv(current_vin)).cuda()
    # max_depth = 20
    # mask = depth_image > -max_depth

    pc = torch.matmul(big_mat, vinv)
    pc = pc[..., :3]

    point_batch = torch.Tensor().cuda()
    for i in range(pixel.shape[0]):
        point = pc[i, pixel[i, 0].long(), pixel[i, 1].long(), :]
        point_batch = torch.cat((point_batch, point.unsqueeze(0)), 0)

    return point_batch


def norm2world(pos):
    pos_01 = pos + 0.5
    tsdf_vol_bnds_tensor = torch.from_numpy(tsdf_vol_bnds).float().cuda()
    world_frame_pos = pos_01 * (tsdf_vol_bnds_tensor[:, 1] - tsdf_vol_bnds_tensor[:, 0]) + tsdf_vol_bnds_tensor[:, 0]
    return world_frame_pos.float()
