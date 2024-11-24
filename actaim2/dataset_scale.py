import xml.etree.ElementTree as ET
import pdb
import shutil
import os
from pathlib import Path

def copy_folder(source_dir, target_dir):
    # Create the target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Iterate over the files and subdirectories in the source directory
    for item in os.listdir(source_dir):
        source_item = os.path.join(source_dir, item)
        target_item = os.path.join(target_dir, item)

        if os.path.isdir(source_item):
            # Recursively copy subdirectories
            copy_folder(source_item, target_item)
        else:
            # Copy files
            shutil.copy2(source_item, target_item)


def scale_urdf(urdf_path, scale_factor, output_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Scale joint positions and link dimensions
    for joint in root.findall(".//joint"):
        for origin in joint.findall("origin"):
            xyz = origin.get("xyz").split()
            xyz = [float(x)*scale_factor for x in xyz]
            origin.set("xyz", " ".join(str(x) for x in xyz))

        if joint.get("type") != "revolute":
            for limit in joint.findall("limit"):
                limit.set("lower", str(float(limit.get("lower"))*scale_factor))
                limit.set("upper", str(float(limit.get("upper"))*scale_factor))

    for link in root.findall(".//link"):
        visual = link.findall("visual")
        collision = link.findall("collision")
        obj_list = visual + collision

        for obj in obj_list:
            for origin in obj.findall("origin"):
                xyz = origin.get("xyz").split()
                xyz = [float(x)*scale_factor for x in xyz]
                origin.set("xyz", " ".join(str(x) for x in xyz))
            for geometry in obj.findall("geometry"):
                if geometry.tag == "box":
                    size = geometry.get("size").split()
                    size = [float(x)*scale_factor for x in size]
                    geometry.set("size", " ".join(str(x) for x in size))
                elif geometry.tag == "cylinder":
                    radius = float(geometry.get("radius"))*scale_factor
                    length = float(geometry.get("length"))*scale_factor
                    geometry.set("radius", str(radius))
                    geometry.set("length", str(length))
                elif geometry.tag == "sphere":
                    radius = float(geometry.get("radius"))*scale_factor
                    geometry.set("radius", str(radius))

                elif geometry.tag == "geometry":
                    meshes = geometry.findall("mesh")
                    for mesh in meshes:
                        scale = mesh.get("scale")
                        if scale is not None:
                            xyz = scale.split()
                            xyz = [float(x) * scale_factor for x in xyz]
                            mesh.set("scale", " ".join(str(x) for x in xyz))
                        else:
                            mesh.set("scale", f"{scale_factor} {scale_factor} {scale_factor}")

    # Write scaled URDF to file
    tree.write(output_path)

def read_object_id(file_path):
    # Open the file in read mode
    object_id_list = []

    with open(file_path, 'r') as file:
        # Read the file line by line
        for line in file:
            # Process each line
            object_id_list.append(line.strip())
    return object_id_list


if __name__ == "__main__":
    # Example usage: Scale myrobot.urdf by a factor of 0.5 and save as myrobot_scaled.urdf
    object_id_file = "./dataset_scale/safe_id.txt"

    object_id_list = read_object_id(object_id_file)

    for id in object_id_list:
        source_dir = f"./data_sample/where2act_original_sapien_dataset/{id}"
        target_dir = f"./dataset_scale/{id}"
        target_dir_path = Path(target_dir)
        # if not target_dir_path.exists():
        copy_folder(source_dir, target_dir)
        scale_urdf(source_dir + "/mobility_vhacd.urdf", 0.7, target_dir + "/mobility_vhacd.urdf")

# window 0.9
# box 0.7
# safe 0.7
# door 0.5