# Copyright (c) 2018 Andy Zeng

import numpy as np

from numba import njit, prange
from skimage import measure
import torch
import pdb

FUSION_GPU_MODE = 0


class TSDFVolume:
    """Volumetric TSDF Fusion of RGB-D Images."""

    def __init__(self, vol_bnds, voxel_size, device, is_save_images=False):
        """Constructor.

        Args:
          vol_bnds (ndarray): An ndarray of shape (3, 2). Specifies the
            xyz bounds (min/max) in meters.
          voxel_size (float): The volume discretization in meters.
        """
        vol_bnds = np.asarray(vol_bnds)
        assert vol_bnds.shape == (3, 2), "[!] `vol_bnds` should be of shape (3, 2)."

        self.device = device
        # Define voxel volume parameters
        self._vol_bnds = vol_bnds
        self._voxel_size = float(voxel_size)
        self._trunc_margin = 5 * self._voxel_size  # truncation on SDF
        self._color_const = 256 * 256

        # Adjust volume bounds and ensure C-order contiguous
        self._vol_dim = (
            np.ceil((self._vol_bnds[:, 1] - self._vol_bnds[:, 0]) / self._voxel_size).copy(order="C").astype(int)
        )
        self._vol_bnds[:, 1] = self._vol_bnds[:, 0] + self._vol_dim * self._voxel_size
        self._vol_origin = self._vol_bnds[:, 0].copy(order="C").astype(np.float32)

        print(
            "Voxel volume size: {} x {} x {} - # points: {:,}".format(
                self._vol_dim[0],
                self._vol_dim[1],
                self._vol_dim[2],
                self._vol_dim[0] * self._vol_dim[1] * self._vol_dim[2],
            )
        )

        # Initialize pointers to voxel volume in GPU memory
        self._tsdf_vol = torch.ones(tuple(self._vol_dim)).to(self.device)
        # for computing the cumulative moving average of observations per voxel
        self._weight_vol = torch.zeros(tuple(self._vol_dim)).to(self.device)
        self._color_vol = torch.zeros(tuple(self._vol_dim), dtype=torch.float64).to(self.device)
        self._rgb_vol = torch.zeros(tuple(np.insert(self._vol_dim, 3, 3)), dtype=torch.float64).to(self.device)

        # Get voxel grid coordinates
        xv, yv, zv = np.meshgrid(
            range(self._vol_dim[0]), range(self._vol_dim[1]), range(self._vol_dim[2]), indexing="ij"
        )

        self.vox_coords = (
            np.concatenate([xv.reshape(1, -1), yv.reshape(1, -1), zv.reshape(1, -1)], axis=0).astype(int).T
        )

        self.vox_coords = torch.from_numpy(self.vox_coords).to(self.device)
        self._vol_origin = torch.from_numpy(self._vol_origin).to(self.device)

        # whether to save images for each traj
        self.is_save_images = is_save_images
        self.depth_images = []
        self.rgb_imges = []

        self.img_width = 320
        self.img_height = 320


    def vox2world(self):
        """Convert voxel grid coordinates to world coordinates."""

        vox_pts = self._vol_origin + self._voxel_size * self.vox_coords
        return vox_pts

    def integrate_tsdf(self, tsdf_vol, dist, w_old, obs_weight):
        """Integrate the TSDF volume."""
        w_new = w_old + obs_weight
        tsdf_vol_int = (w_old * tsdf_vol + obs_weight * dist) / w_new
        return tsdf_vol_int, w_new

    def integrate(self, color_im, depth_im, fu, fv, view_matrix, obs_weight=1.0):
        """Integrate an RGB-D frame into the TSDF volume."""
        if self.is_save_images:
            self.rgb_imges.append(color_im)
            self.depth_images.append(depth_im)

        im_h, im_w = depth_im.shape
        self.img_height, self.img_width = im_h, im_w

        # Fold RGB color image into a single channel image
        color_im = color_im.astype(np.float32)
        color_im = np.floor(color_im[..., 2] * self._color_const + color_im[..., 1] * 256 + color_im[..., 0])

        # Convert voxel grid coordinates to pixel coordinates
        cam_pts = self.vox2world()

        view_matrix = torch.from_numpy(view_matrix).to(self.device)

        add_ones = torch.ones(cam_pts.shape[0]).unsqueeze(1).to(self.device)
        point_ones = torch.cat((cam_pts, add_ones), dim=1)
        big_mat = torch.matmul(point_ones, view_matrix)
        proj_u = big_mat[:, 0]
        proj_v = big_mat[:, 1]
        depth_img = big_mat[:, 2]

        u_range = proj_u / (fu * depth_img)
        v_range = proj_v / (fv * depth_img)

        pix_x = torch.round(im_w * (-u_range)) + im_w * 0.5
        pix_y = torch.round(im_h * v_range) + im_h * 0.5
        pix_z = -depth_img

        # Eliminate pixels outside view frustum
        valid_pix = torch.logical_and(
            pix_x >= 0,
            torch.logical_and(pix_x < im_w, torch.logical_and(pix_y >= 0, torch.logical_and(pix_y < im_h, pix_z > 0))),
        ).to(self.device)
        depth_val = torch.zeros(pix_x.shape).to(self.device)
        depth_im = torch.from_numpy(depth_im).to(self.device)
        depth_im = depth_im.float()
        depth_val[valid_pix] = -depth_im[
            (pix_y[valid_pix]).cpu().numpy().astype(int), (pix_x[valid_pix]).cpu().numpy().astype(int)
        ]
        # Integrate TSDF
        depth_diff = depth_val - pix_z
        valid_pts = torch.logical_and(
            depth_val > 0, torch.logical_and(depth_diff >= -self._trunc_margin, depth_diff < 1000)
        )
        dist = torch.clamp(depth_diff / self._trunc_margin, max=1)

        valid_vox_x = self.vox_coords[valid_pts, 0]
        valid_vox_y = self.vox_coords[valid_pts, 1]
        valid_vox_z = self.vox_coords[valid_pts, 2]

        w_old = self._weight_vol[valid_vox_x, valid_vox_y, valid_vox_z]
        tsdf_vals = self._tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z]
        valid_dist = dist[valid_pts]

        tsdf_vol_new, w_new = self.integrate_tsdf(tsdf_vals, valid_dist, w_old, obs_weight)
        self._weight_vol[valid_vox_x, valid_vox_y, valid_vox_z] = w_new
        self._tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z] = tsdf_vol_new

        # Integrate color
        color_im = torch.from_numpy(color_im).to(self.device)
        old_color = self._color_vol[valid_vox_x, valid_vox_y, valid_vox_z]
        old_b = torch.floor(old_color / self._color_const)
        old_g = torch.floor((old_color - old_b * self._color_const) / 256)
        old_r = old_color - old_b * self._color_const - old_g * 256
        new_color = color_im[(pix_y[valid_pts]).cpu().numpy().astype(int), (pix_x[valid_pts]).cpu().numpy().astype(int)]
        new_b = torch.floor(new_color / self._color_const)
        new_g = torch.floor((new_color - new_b * self._color_const) / 256)
        new_r = new_color - new_b * self._color_const - new_g * 256
        new_b = torch.clamp(torch.round((w_old * old_b + obs_weight * new_b) / w_new), max=255)
        new_g = torch.clamp(torch.round((w_old * old_g + obs_weight * new_g) / w_new), max=255)
        new_r = torch.clamp(torch.round((w_old * old_r + obs_weight * new_r) / w_new), max=255)
        self._rgb_vol[valid_vox_x, valid_vox_y, valid_vox_z, 0] = new_r / 255
        self._rgb_vol[valid_vox_x, valid_vox_y, valid_vox_z, 1] = new_g / 255
        self._rgb_vol[valid_vox_x, valid_vox_y, valid_vox_z, 2] = new_b / 255

        # self._color_vol[valid_vox_x, valid_vox_y, valid_vox_z] = (new_b * self._color_const + new_g * 256 + new_r)

    def get_volume(self):
        return self._tsdf_vol, self._rgb_vol

    def get_point_cloud(self):
        """Extract a point cloud from the voxel volume."""

        tsdf_vol, color_vol = self.get_volume()
        tsdf_vol = tsdf_vol.cpu().numpy()
        color_vol = color_vol.cpu().numpy()

        # Marching cubes
        verts, faces, norms, vals = measure.marching_cubes(tsdf_vol, level=0, method='lewiner')
        verts_ind = np.round(verts).astype(int)
        verts = verts * self._voxel_size + self._vol_origin.cpu().numpy()  # voxel grid coordinates to world coordinates

        # Get vertex colors
        rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        colors = rgb_vals * 255
        colors = np.floor(colors)
        colors = colors.astype(np.uint8)

        pc = np.hstack([verts, colors])
        return verts, pc

    def get_mesh(self):
        """Compute a mesh from the voxel volume using marching cubes."""
        tsdf_vol, color_vol = self.get_volume()
        tsdf_vol = tsdf_vol.cpu().numpy()
        color_vol = color_vol.cpu().numpy()

        # Marching cubes
        verts, faces, norms, vals = measure.marching_cubes(tsdf_vol, level=0, method='lewiner')
        verts_ind = np.round(verts).astype(int)
        verts = verts * self._voxel_size + self._vol_origin.cpu().numpy()  # voxel grid coordinates to world coordinates

        # Get vertex colors
        rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        colors = rgb_vals * 255
        colors = np.floor(colors)
        colors = colors.astype(np.uint8)
        return verts, faces, norms, colors

    def pc_generate_novel_view(self, pc, pc_rgb, fu, fv, view_matrix):
        num_points = pc.shape[0]

        vin = torch.from_numpy(view_matrix).to(self.device)

        add_ones = torch.ones(pc.shape[0]).unsqueeze(1).to(self.device)
        point_ones = torch.cat((pc, add_ones), dim=1)
        big_mat_inv = torch.matmul(point_ones, vin)
        proj_u_inv = big_mat_inv[:, 0]
        proj_v_inv = big_mat_inv[:, 1]
        depth_img_inv = big_mat_inv[:, 2]

        u_range_inv = proj_u_inv / (fu * depth_img_inv)
        v_range_inv = proj_v_inv / (fv * depth_img_inv)

        u_inv = (torch.round(self.img_width * (-u_range_inv)) + self.img_width * 0.5).long()
        v_inv = (torch.round(self.img_height * v_range_inv) + self.img_height * 0.5).long()

        generated_img_rgb = torch.zeros((self.img_width, self.img_height, 3)).to(self.device)
        generated_img_depth = torch.zeros((self.img_width, self.img_height)).to(self.device)

        mask = self.filter_depth_rgb(u_inv, v_inv, depth_img_inv)
        # mask = torch.tensor([True]).repeat(num_points).to(self.device)

        mask_u = torch.logical_and(u_inv < self.img_width, u_inv >= 0)
        mask_v = torch.logical_and(v_inv < self.img_height, v_inv >= 0)
        mask_ = torch.logical_and(mask_v, mask_u)
        mask = torch.logical_and(mask_, mask)
        
        u_inv_valid = u_inv[mask]
        v_inv_valid = v_inv[mask]

        depth_valid = depth_img_inv[mask]
        rgb_valid = pc_rgb[mask, :]

        generated_img_rgb[v_inv_valid, u_inv_valid] = rgb_valid
        generated_img_depth[v_inv_valid, u_inv_valid] = depth_valid
                
        generated_img_rgb /= 255.0
        
        return generated_img_depth, generated_img_rgb


    def filter_depth_rgb(self, x_ind, y_ind, depth):
        num_points = depth.shape[0]

        index_hash = x_ind * self.img_width + y_ind
        index_unique, inverse_indices = torch.unique(index_hash, sorted=False, return_inverse=True)
        depth_mask = torch.tensor([True]).repeat(num_points).to(self.device)
        num_unique_points = index_unique.shape[0]
        depth_max = torch.ones(num_unique_points).to(self.device) * -100
        depth_max_ind = torch.ones(num_unique_points).to(self.device) * -1

        for i in range(num_points):
            depth_i = depth[i].item()
            unique_ind_i = inverse_indices[i].item()
            if depth_i >= depth_max[unique_ind_i]:
                pre_max_id = depth_max_ind[unique_ind_i].item()
                if pre_max_id > 0:
                    depth_mask[int(pre_max_id)] = 0

                depth_max[unique_ind_i] = depth_i
                depth_max_ind[unique_ind_i] = i
            else:
                depth_mask[i] = False

        # TODO why always one more left
        print(depth_mask.sum())
        print(num_unique_points)

        return depth_mask



def rigid_transform(xyz, transform):
    """Applies a rigid transform to an (N, 3) pointcloud."""
    xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
    xyz_t_h = np.dot(transform, xyz_h.T).T
    return xyz_t_h[:, :3]


def meshwrite(filename, verts, faces, norms, colors):
    """Save a 3D mesh to a polygon .ply file."""
    # Write header
    ply_file = open(filename, "w")
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (verts.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property float nx\n")
    ply_file.write("property float ny\n")
    ply_file.write("property float nz\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("element face %d\n" % (faces.shape[0]))
    ply_file.write("property list uchar int vertex_index\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(verts.shape[0]):
        ply_file.write(
            "%f %f %f %f %f %f %d %d %d\n"
            % (
                verts[i, 0],
                verts[i, 1],
                verts[i, 2],
                norms[i, 0],
                norms[i, 1],
                norms[i, 2],
                colors[i, 0],
                colors[i, 1],
                colors[i, 2],
            )
        )

    # Write face list
    for i in range(faces.shape[0]):
        ply_file.write("3 %d %d %d\n" % (faces[i, 0], faces[i, 1], faces[i, 2]))

    ply_file.close()


def pcwrite(filename, xyzrgb):
    """Save a point cloud to a polygon .ply file."""
    xyz = xyzrgb[:, :3]
    rgb = xyzrgb[:, 3:].astype(np.uint8)

    # Write header
    ply_file = open(filename, "w")
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (xyz.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(xyz.shape[0]):
        ply_file.write(
            "%f %f %f %d %d %d\n"
            % (
                xyz[i, 0],
                xyz[i, 1],
                xyz[i, 2],
                rgb[i, 0],
                rgb[i, 1],
                rgb[i, 2],
            )
        )
