# make ground truth point clouds of victor with associated EEF poses
import time
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
from visual_servoing.icp_pbvs import ICPPBVS
from visual_servoing.utils import get_link_tf, draw_pose, draw_sphere_marker, erase_pos
import pickle
from visual_servoing.victor import *
import copy
import matplotlib.pyplot as plt

KEY_U = 117
KEY_N = 110

victor = Victor()
camera = PyBulletCamera(np.array([1.0, -1.0, 1.0]), np.array([1.0, 0.0, 1.0]))

# Get EEF Link GT 
tool_idx = victor.eef_idx
result = p.getLinkState(victor.urdf,
                        tool_idx,
                        computeLinkVelocity=1,
                        computeForwardKinematics=1)
link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
Tcw = camera.get_view()
Twe = np.zeros((4, 4))
Twe[0:3, 0:3] = np.array(p.getMatrixFromQuaternion(frame_rot)).reshape(3, 3)
Twe[0:3, 3] = frame_pos
Twe[3, 3] = 1
Tce = Tcw @ Twe

pbvs = ICPPBVS(camera, 1, 1, victor.get_gripper_pcl(np.eye(4)), Tce, 1.5)

target = np.hstack((np.array([0.5, 0.5, 0.5]), np.array(p.getQuaternionFromEuler((0, 0, 0)))))


# target = np.hstack(( np.array([0.8, 1.3, 0.4]), np.array(p.getQuaternionFromEuler((0, 0, 0))) ) )

def to_homogenous(vec):
    quat = vec[3:7]
    pos = vec[0:3]
    H = np.eye(4)
    H[0:3, 0:3] = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    H[0:3, 3] = pos
    return H


uids_target_marker = None
uids_eef_marker = None

pos_error = []
rot_error = []
fig, (ax1, ax2) = plt.subplots(2, 1)
i = 0
start = False

sim_hz = 240
pbvs_hz = 10
sim_steps_per_pbvs = int(sim_hz / pbvs_hz)
p.setTimeStep(1 / sim_hz)

while (True):
    # stop condition 
    events = p.getKeyboardEvents()
    for _ in range(sim_steps_per_pbvs):
        p.stepSimulation()
    if (KEY_U in events):
        start = True
    if KEY_N in events:
        break
    if (not start):
        pbvs.draw_registration_result()
        plt.pause(0.01)
        cv2.imshow("Camera", np.zeros((800 // 5, 1280 // 5)))
        continue
    i += 1
    # create point cloud from RGBD image
    rgb, depth, seg = camera.get_image(True)
    rgb_edit = rgb[..., [2, 1, 0]].copy()
    cv2.imshow("Camera", cv2.resize(rgb_edit, (1280 // 5, 800 // 5)))

    # draw tool ground truth
    tool_idx = victor.eef_idx
    result = p.getLinkState(victor.urdf,
                            tool_idx,
                            computeLinkVelocity=1,
                            computeForwardKinematics=1)

    link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
    # if (uids_eef_marker is not None):
    #    erase_pos(uids_eef_marker)
    # uids_eef_marker = draw_pose(link_trn, link_rot)

    # Draw target marker
    if (uids_target_marker is not None):
        erase_pos(uids_target_marker)
    uids_target_marker = draw_pose(target[0:3], target[3:7])

    print("doing pbvs")
    prev_time = time.time()
    ctrl, Twe = pbvs.do_pbvs(depth, seg, to_homogenous(target), victor.get_arm_jacobian('left'),
                             victor.get_jacobian_pinv('left'))
    print(time.time() - prev_time)
    print("finished pbvs")
    # victor.psuedoinv_ik_controller("left", np.hstack(((target[0:3] - link_trn)*10, target[4:7])))
    victor.psuedoinv_ik_controller("left", ctrl)

    # draw EEF pose as observed from camera gt
    if (uids_eef_marker is not None):
        erase_pos(uids_eef_marker)
    uids_eef_marker = draw_pose(Twe[0:3, 3], Twe[0:3, 0:3], mat=True)

    # Twc = np.linalg.inv(Tcw)
    # draw_pose(Twc[0:3, 3], Twc[0:3, 0:3], mat=True)
    pos_error.append(np.linalg.norm(Twe[0:3, 3] - frame_pos))
    link_rod, _ = cv2.Rodrigues(np.array(p.getMatrixFromQuaternion(frame_rot)).reshape(3, 3) @ Twe[0:3, 0:3].T)
    rot_error.append(np.linalg.norm(link_rod))
    ax1.scatter(i, pos_error[-1], c='g')
    ax2.scatter(i, rot_error[-1], c='r')
    plt.pause(0.01)

plt.figure()
plt.plot(pos_error)
plt.xlabel("iteration")
plt.ylabel("meters")
plt.title("EEF Position error (m)")
plt.figure()
plt.xlabel("iteration")
plt.ylabel("Rodrigues norm")
plt.title("EEF Rotation error")
plt.plot(rot_error)
plt.show()
