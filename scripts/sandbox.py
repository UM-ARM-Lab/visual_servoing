import cv2
from visual_servoing.marker_pbvs import MarkerPBVS
from visual_servoing.val import Val
from visual_servoing.utils import *
from visual_servoing.camera import PyBulletCamera
import numpy as np
import pybullet as p
from qpsolvers import solve_qp

val = Val([0.0, 0.0, -0.5])
camera = PyBulletCamera(camera_eye=np.array([0.7, -0.8, 0.5]), camera_look=np.array([0.7, 0.0, 0.2]))
p.setGravity(0, 0, -10)

def get_interaction_mat(u, v, z):
    return np.array([
        [-1/z, 0, u/z, u*v, -1(1+u**2), v],
        [0, -1/z, v/z, (1+v**2), -u*v, -u]
    ])
uids_camera_marker = None

pbvs = MarkerPBVS(camera, 1, 1, 1.1)

while(True):

    # Step sim
    for _ in range(24):
        link_pos, link_rot = val.get_eef_pos("camera")
        camera_rot = np.array(p.getMatrixFromQuaternion(link_rot)).reshape(3,3)
        camera.upate_from_pose(link_pos, camera_rot)
        #draw_sphere_marker(link_pos, 0.01, (1, 0, 0, 1))

        p.stepSimulation()
    
    #ctrl = np.array([0, 1, 0, 0, 0, 0])
    #jac = val.get_camera_jacobian()
    #Q = np.eye(6)
    #P = 2 * jac.T @ Q @ jac
    #num_joints = jac.shape[1]
    #q = (-ctrl @ Q @ jac - ctrl @ Q.T @ jac)
    #G = np.vstack((np.eye(num_joints), -np.eye(num_joints)))
    #h = np.ones(num_joints * 2) * 1.5
    #num_joints = jac.shape[1]
    #ctrl = solve_qp(P, q, G, h, None, None, solver="cvxopt")
    #val.torso_control(ctrl)

    # Visualize camera pose 
    if (uids_camera_marker is not None):
        erase_pos(uids_camera_marker)
    Twc = np.linalg.inv(camera.get_extrinsics())
    uids_camera_marker = draw_pose(Twc[0:3, 3], Twc[0:3, 0:3], mat=True)

    rgb, depth = camera.get_image()
    cv2.imshow("Im", rgb)
    cv2.waitKey(1)
