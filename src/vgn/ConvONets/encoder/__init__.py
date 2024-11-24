from vgn.ConvONets.encoder import pointnet, voxels, pointnetpp, images


encoder_dict = {
    "pointnet_local_pool": pointnet.LocalPoolPointnet,
    "pointnet_crop_local_pool": pointnet.PatchLocalPoolPointnet,
    "pointnet_plus_plus": pointnetpp.PointNetPlusPlus,
    "voxel_simple_local": voxels.LocalVoxelEncoder,
    "image_simple_local": images.LocalImageEncoder,
}
