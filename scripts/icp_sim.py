
# make ground truth point clouds of victor with associated EEF poses
from tkinter import W
import numpy as np
from typing import Dict, Optional, List
from pathlib import Path
import gzip
import pybullet as p
import pybullet_data
import cv2 
from visual_servoing.camera import *
import open3d as o3d
from visual_servoing.utils import get_link_tf, draw_pose, draw_sphere_marker, erase_pos
import pickle
from visual_servoing.victor import *

victor = Victor()
camera = PyBulletCamera(np.array([1.0, -1.0, 1.0]), np.array([1.0, 0.0, 1.0]))

# load pickle
gripper_pts = pickle.load(open("robot_points.pkl", 'rb'))
gripper_pcl = []
for link in gripper_pts ['points'].keys():
    if(link in victor.links_by_name):
        pass
    else:
        print("fail")

    pts = np.array(gripper_pts['points'][link]).T
    pts = np.vstack((pts, np.ones(pts.shape[1]))) 
    tf = get_link_tf(victor.urdf, victor.links_by_name[link][0])
    pts_tf = tf @ pts
    gripper_pcl.append(pts_tf.T)
gripper_pcl = np.vstack(gripper_pcl)
gripper_pcl = gripper_pcl[:, 0:3]
print(gripper_pcl)
gpcl = o3d.geometry.PointCloud()
gpcl.points = o3d.utility.Vector3dVector(gripper_pcl)
#o3d.visualization.draw_geometries([gpcl])
target = np.hstack(( np.array([0.5, 0.5, 0.5]), np.array(p.getQuaternionFromEuler((0, 0, 0))) ) )

uids_target_marker = None
uids_eef_marker = None
while(True):
    p.stepSimulation()
    # create point cloud from RGBD image
    rgb, depth = camera.get_image()
    rgb_edit = rgb[..., [2, 1, 0]].copy()
    #cv2.imshow("RGB", rgb_edit)
    pcl_raw = camera.get_pointcloud(depth)
    pcl = o3d.geometry.PointCloud() 
    pcl.points = o3d.utility.Vector3dVector(pcl_raw.T)
    pcl.colors = o3d.utility.Vector3dVector(rgb_edit.reshape(-1, 3)/255.0)
    #o3d.visualization.draw_geometries([pcl])
    #cv2.waitKey(1)
    #victor.set_velo([10.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])


    # draw tool ground truth
    tool_idx = victor.eef_idx 
    result = p.getLinkState(victor.urdf,
                            tool_idx,
                            computeLinkVelocity=1,
                            computeForwardKinematics=1)

    link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
    if (uids_eef_marker is not None):
        erase_pos(uids_eef_marker)
    uids_eef_marker = draw_pose(link_trn, link_rot)

    if (uids_target_marker is not None):
        erase_pos(uids_target_marker)
    uids_target_marker = draw_pose(target[0:3], target[3:7]) 
    victor.psuedoinv_ik_controller("left", np.hstack(((target[0:3] - link_trn)*10, target[4:7])))
   
    #draw_pose(np.linalg.inv(camera.get_extrinsics())[0:3, 3], np.linalg.inv(camera.get_extrinsics())[0:3, 0:3], mat=True)
    #Tcw = camera.get_extrinsics()
    #Tew = np.zeros((4,4))
    #Tew[0:3, 0:3] = np.array(p.getMatrixFromQuaternion(link_rot)).reshape(3,3) 
    #Tew[0:3, 3] = link_trn
    #Tew[3,3] = 1
    #Tec = np.linalg.inv(Tcw) @ Tew

    ## draw EEF pose as observed from camera gt
    #Twe = Tcw @ Tec
    #draw_pose(Twe[0:3, 3], Twe[0:3, 0:3], mat=True) 
