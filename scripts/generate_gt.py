# make ground truth point clouds of victor with associated EEF poses
import numpy as np
import pybullet as p
import pybullet_data
import cv2 
from visual_servoing.camera import *
import open3d as o3d
from visual_servoing.utils import draw_pose, draw_sphere_marker, erase_pos

client = p.connect(p.GUI)
#p.setGravity(0, 0, -10)
urdf = p.loadURDF("models/victor_description/urdf/victor.urdf", [0,0,0], p.getQuaternionFromEuler([0,0,0]))
camera = PyBulletCamera(np.array([1.0, 1.0, 1.0]), np.array([1.0, 0.0, 1.0]))



# Organize joints into a dict from name->info
joints_by_name = {}
num_joints = p.getNumJoints(urdf)
for i in range(num_joints):
    info = p.getJointInfo(urdf, i)
    name = info[1].decode("ascii")
    joints_by_name[name] = info
    print(f"idx: {info[0]}, joint: {name}, type:{info[2]} ")
eef_idx = 16

while(True):
    p.stepSimulation()
    rgb, depth = camera.get_image()
    rgb_edit = rgb[..., [2, 1, 0]].copy()
    cv2.imshow("RGB", rgb_edit)

    pcl_raw = camera.get_pointcloud(depth)
    # naive point cloud
    pcl = o3d.geometry.PointCloud() 
    pcl.points = o3d.utility.Vector3dVector(pcl_raw.T)
    pcl.colors = o3d.utility.Vector3dVector(rgb_edit.reshape(-1, 3)/255.0)
    #o3d.visualization.draw_geometries([pcl])
    cv2.waitKey(1)


    tool_idx = eef_idx 
    result = p.getLinkState(urdf,
                            tool_idx,
                            computeLinkVelocity=1,
                            computeForwardKinematics=1)

    link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
    draw_pose(link_trn, link_rot)
