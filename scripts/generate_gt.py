# make ground truth point clouds of victor with associated EEF poses
import numpy as np
import pybullet as p
import pybullet_data
import cv2 
from visual_servoing.camera import *
import open3d as o3d

client = p.connect(p.GUI)
#p.setGravity(0, 0, -10)
urdf = p.loadURDF("models/victor_description/urdf/victor.urdf", [0,0,0], p.getQuaternionFromEuler([0,0,0]))
camera = PyBulletCamera(np.array([1.0, 1.0, 1.0]), np.array([1.0, 0.0, 1.0]))

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
    o3d.visualization.draw_geometries([pcl])
    cv2.waitKey(1)


