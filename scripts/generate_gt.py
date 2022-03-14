# make ground truth point clouds of victor with associated EEF poses
import numpy as np
from typing import Dict, Optional, List
from pathlib import Path
import gzip
import pybullet as p
import pybullet_data
import cv2 
from visual_servoing.camera import *
import open3d as o3d
from visual_servoing.utils import draw_pose, draw_sphere_marker, erase_pos

#from link_bot_data.coerce_types import coerce_types
import pickle
client = p.connect(p.GUI)
#p.setGravity(0, 0, -10)
urdf = p.loadURDF("models/victor_description/urdf/victor.urdf", [0,0,0], p.getQuaternionFromEuler([0,0,0]))
camera = PyBulletCamera(np.array([1.0, -1.0, 1.0]), np.array([1.0, 0.0, 1.0]))



# Organize joints into a dict from name->info
joints_by_name = {}
num_joints = p.getNumJoints(urdf)
for i in range(num_joints):
    info = p.getJointInfo(urdf, i)
    name = info[1].decode("ascii")
    joints_by_name[name] = info
    print(f"idx: {info[0]}, joint: {name}, type:{info[2]} ")
eef_idx = 16


def dump_gzipped_pickle(data, filename):
    while True:
        try:
            with gzip.open(filename, 'wb') as data_file:
                pickle.dump(data, data_file)
            return
        except KeyboardInterrupt:
            pass

def index_to_filename(file_extension, traj_idx):
    new_filename = f"example_{traj_idx:08d}{file_extension}"
    return new_filename

def pkl_write_example(full_output_directory, example, traj_idx, extra_metadata_keys: Optional[List[str]] = None):
    metadata_filename = index_to_filename('.pkl', traj_idx)
    full_metadata_filename = full_output_directory / metadata_filename
    example_filename = index_to_filename('.pkl.gz', traj_idx)
    full_example_filename = full_output_directory / example_filename

    if 'metadata' in example:
        metadata = example.pop('metadata')
    else:
        metadata = {}
    metadata['data'] = example_filename
    if extra_metadata_keys is not None:
        for k in extra_metadata_keys:
            metadata[k] = example.pop(k)

    metadata = coerce_types(metadata)
    with full_metadata_filename.open("wb") as metadata_file:
        pickle.dump(metadata, metadata_file)

    example = coerce_types(example)
    dump_gzipped_pickle(example, full_example_filename)

    return full_example_filename, full_metadata_filename

while(True):
    p.stepSimulation()
    # create point cloud from RGBD image
    rgb, depth = camera.get_image()
    rgb_edit = rgb[..., [2, 1, 0]].copy()
    cv2.imshow("RGB", rgb_edit)
    pcl_raw = camera.get_pointcloud(depth)
    pcl = o3d.geometry.PointCloud() 
    pcl.points = o3d.utility.Vector3dVector(pcl_raw.T)
    pcl.colors = o3d.utility.Vector3dVector(rgb_edit.reshape(-1, 3)/255.0)
    o3d.visualization.draw_geometries([pcl])
    cv2.waitKey(1)


    # draw tool ground truth
    tool_idx = eef_idx 
    result = p.getLinkState(urdf,
                            tool_idx,
                            computeLinkVelocity=1,
                            computeForwardKinematics=1)

    link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
    #draw_pose(link_trn, link_rot)
    # draw camera pose
    draw_pose(np.linalg.inv(camera.get_extrinsics())[0:3, 3], np.linalg.inv(camera.get_extrinsics())[0:3, 0:3], mat=True)
    Tcw = camera.get_extrinsics()
    Tew = np.zeros((4,4))
    Tew[0:3, 0:3] = np.array(p.getMatrixFromQuaternion(link_rot)).reshape(3,3) 
    Tew[0:3, 3] = link_trn
    Tew[3,3] = 1
    Tec = np.linalg.inv(Tcw) @ Tew

    # draw EEF pose as observed from camera gt
    Twe = Tcw @ Tec
    draw_pose(Twe[0:3, 3], Twe[0:3, 0:3], mat=True) 
# draw_pose(camera.camera_eye, (np.linalg.inv(camera.get_extrinsics())@Tc1c2 )[0:3, 0:3], mat=True, axis_len=0.1)
    #pkl_write_example(Path("/home/ashwin/Desktop/temp/"), {"a": "a"}, 1) 
    #break
