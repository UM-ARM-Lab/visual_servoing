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
    #o3d.visualization.draw_geometries([pcl])
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
