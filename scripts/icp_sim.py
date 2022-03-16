
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
from visual_servoing.utils import draw_pose, draw_sphere_marker, erase_pos
import pickle

def get_link_tf(urdf, idx):
    result = p.getLinkState(urdf,
                            idx,
                            computeLinkVelocity=1,
                            computeForwardKinematics=1)

    link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
    rot_mat = np.array(p.getMatrixFromQuaternion(link_rot)).reshape(3,3)
    tf = np.zeros((4,4))
    tf[0:3, 0:3] = rot_mat
    tf[0:3, 3] = link_trn
    tf[3, 3] = 1
    return tf

class Victor:
    def __init__(self):
        client = p.connect(p.GUI)
        self.urdf = p.loadURDF("models/victor_description/urdf/victor.urdf", [0,0,0], p.getQuaternionFromEuler([0,0,0]))
        self.camera = PyBulletCamera(np.array([1.0, -1.0, 1.0]), np.array([1.0, 0.0, 1.0]))


        # Organize joints into a dict from name->info
        self.joints_by_name = {}
        self.links_by_name = {}
        num_joints = p.getNumJoints(self.urdf)
        for i in range(num_joints):
            info = p.getJointInfo(self.urdf, i)
            joint_name = info[1].decode("ascii") 
            link_name = info[12].decode("ascii") 
            self.joints_by_name[joint_name] = info
            self.links_by_name[link_name] = info
            print(f"idx: {info[0]}, joint: {joint_name}, link: {link_name}, type:{info[2]} ")
        eef_idx = 16

    # transform helper, gets Tab b --> a 
    def get_tf(self, link_a, link_b):
        #joints_by_name[link_a]
        link_info_a = self.links_by_name[link_a]
        link_info_b = self.links_by_name[link_b]
        Twa = get_link_tf(self.urdf, link_info_a[0])
        Twb = get_link_tf(self.urdf, link_info_b[0])
        Tab = Twb @ np.linalg.inv(Twa)
        return Tab

victor = Victor()

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
o3d.visualization.draw_geometries([gpcl])

while(True):
    p.stepSimulation()
    # create point cloud from RGBD image
    rgb, depth = victor.camera.get_image()
    rgb_edit = rgb[..., [2, 1, 0]].copy()
    cv2.imshow("RGB", rgb_edit)
    pcl_raw = victor.camera.get_pointcloud(depth)
    pcl = o3d.geometry.PointCloud() 
    pcl.points = o3d.utility.Vector3dVector(pcl_raw.T)
    pcl.colors = o3d.utility.Vector3dVector(rgb_edit.reshape(-1, 3)/255.0)
    o3d.visualization.draw_geometries([pcl])
    cv2.waitKey(1)


    # draw tool ground truth
    #tool_idx = eef_idx 
    #result = p.getLinkState(urdf,
    #                        tool_idx,
    #                        computeLinkVelocity=1,
    #                        computeForwardKinematics=1)

    #link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
    ##draw_pose(link_trn, link_rot)
    ## draw camera pose
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
