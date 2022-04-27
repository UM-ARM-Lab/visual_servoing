##########################################################
# This demo detects an end effector via its AR tag and   #
# does PBVS to various predetermined points in the world #
##########################################################

from visual_servoing.utils import draw_pose, draw_sphere_marker, erase_pos
from visual_servoing.val import *
from visual_servoing.pbvs import *
from visual_servoing.camera import *
from visual_servoing.marker_pbvs import *
import time
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pybullet as p

# Key bindings
KEY_U = 117
KEY_I = 105
KEY_J = 106
KEY_K = 107
KEY_N = 110
KEY_M = 109

# Val robot and PVBS controller
val = Val([0.0, 0.0, -0.5])
# y = -1.3
camera = PyBulletCamera(camera_eye=np.array([0.7, -0.8, 0.5]), camera_look=np.array([0.7, 0.0, 0.2]))

# draw the PBVS camera pose
Tc1c2 = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])

# draw the camera
# draw_pose(camera.camera_eye, (np.linalg.inv(camera.get_extrinsics())@Tc1c2 )[0:3, 0:3], mat=True, axis_len=0.1)

# AR tag on a box for debugging AR tag detection, commented out
box_pos = (0.8, 0.3 + 0.2, 0.4)
#box_orn = [0, 0, np.pi/8]
box_orn = [np.pi/2, np.pi + np.pi/4, 3*np.pi/2]

#box_vis = p.createVisualShape(p.GEOM_MESH,fileName="models/AR Tag Cuff 2/PINCER_HOUSING2_EDIT.obj", meshScale=[1.0,1.0, 1.0])
#box_multi = p.createMultiBody(baseCollisionShapeIndex = 0, baseVisualShapeIndex=box_vis, basePosition=box_pos, baseOrientation=p.getQuaternionFromEuler(box_orn))


# Specify the 3D geometry of the end effector marker board 
tag_len = 0.0305
gap_len = 0.0051
angle = np.pi / 4
# center tag
tag0_tl = np.array([-tag_len / 2, tag_len / 2, 0.0], dtype=np.float32)
tag0_tr = np.array([tag_len / 2, tag_len / 2, 0.0], dtype=np.float32)
tag0_br = np.array([tag_len / 2, -tag_len / 2, 0.0], dtype=np.float32)
tag0_bl = np.array([-tag_len / 2, -tag_len / 2, 0.0], dtype=np.float32)
z1 = -np.cos(angle) * gap_len
z2 = -np.cos(angle) * (gap_len + tag_len)
y1 = tag_len / 2 + gap_len + gap_len * np.sin(angle)
y2 = tag_len / 2 + gap_len + (gap_len + tag_len) * np.sin(angle)
# lower tag
tag1_tl = np.array([-tag_len / 2, -y1, z1], dtype=np.float32)
tag1_tr = np.array([tag_len / 2, -y1, z1], dtype=np.float32)
tag1_br = np.array([tag_len / 2, -y2, z2], dtype=np.float32)
tag1_bl = np.array([-tag_len / 2, -y2, z2], dtype=np.float32)
# upper tag
tag2_tl = np.array([-tag_len / 2, y2, z2], dtype=np.float32)
tag2_tr = np.array([tag_len / 2, y2, z2], dtype=np.float32)
tag2_br = np.array([tag_len / 2, y1, z1], dtype=np.float32)
tag2_bl = np.array([-tag_len / 2, y1, z1], dtype=np.float32)

tag0 = np.array([tag0_tl, tag0_tr, tag0_br, tag0_bl])
tag1 = np.array([tag1_tl, tag1_tr, tag1_br, tag1_bl])
tag2 = np.array([tag2_tl, tag2_tr, tag2_br, tag2_bl])
tag_geometry = [tag0, tag1, tag2]
ids = np.array([1, 2, 3])
ids2 = np.array([4,5,6])

pbvs = MarkerPBVS(camera, 1, 1, 1.5, np.eye(4), ids, tag_geometry, ids2, tag_geometry)
sim_dt = 1/240
p.setTimeStep(sim_dt)
sim_steps_per_pbvs = 24
#p.setRealTimeSimulation(1)

Two = None
Twa = None

# UIDS for ar tag pose marker 
uids_eef_marker = None
uids_target_marker = None
uids_eef_gt = None

initial_arm = val.get_eef_pos("left")[0:3]

Two = np.eye(4) 
Two[0:3, 3] = np.array([0.8, 0.24, 0.3])
#Two[0:3, 0:3] = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler((np.pi/4, np.pi/4, 0)))).reshape(3, 3)
Two[0:3, 0:3] = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler((np.pi/2, np.pi/4, 0)))).reshape(3, 3)

pos_errors = []
rot_errors = []

armed = False

def get_eef_gt():
    tool_idx = val.left_tag[0]
    result = p.getLinkState(val.urdf,
                            tool_idx,
                            computeLinkVelocity=1,
                            computeForwardKinematics=1)

    link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
    Twe = np.eye(4)
    Twe[0:3, 0:3] = np.array(p.getMatrixFromQuaternion(frame_rot)).reshape(3, 3)
    Twe[0:3, 3] = frame_pos
    return Twe

cv2.imshow("Camera", np.zeros((1280//3, 720//3)))  

start = False
while True:
    events = p.getKeyboardEvents()
    if(KEY_I in events):
        start = True
    if(not start):
        continue

    t0 = time.time()


    # Get camera feed and detect markers
    rgb, depth = camera.get_image()
    rgb_edit = rgb[..., [2, 1, 0]].copy()

    # Do PBVS if there is a target 
    ctrl = np.zeros(6)
    #cv2.imshow("image", rgb_edit)
    ctrl, Twe = pbvs.do_pbvs(rgb_edit, depth, Two, np.eye(4), val.get_arm_jacobian("left"), val.get_jacobian_pinv("left"), sim_dt)

    target_pos_error = np.linalg.norm(Twe[0:3, 3] -  Two[0:3, 3])
    target_rot_error = np.linalg.norm(cv2.Rodrigues(Twe[0:3, 0:3].T @ Two[0:3, 0:3])[0])
    if(target_pos_error < 0.03 and target_rot_error < 0.02):
        break
    
    truth = get_eef_gt()
    pos_errors.append(np.linalg.norm(Twe[0:3, 3] - truth[0:3, 3]))
    link_rod, _ = cv2.Rodrigues(Twe[0:3, 0:3] @ truth[0:3, 0:3].T)
    rot_errors.append(np.linalg.norm(link_rod))

    val.set_velo(val.get_jacobian_pinv("left") @ ctrl)

    # Visualize estimated end effector pose 
    if (uids_eef_marker is not None):
        erase_pos(uids_eef_marker)
    uids_eef_marker = draw_pose(Twe[0:3, 3], Twe[0:3, 0:3], mat=True)

    #  Visualize target pose 
    if (uids_target_marker is not None):
        erase_pos(uids_target_marker)
    uids_target_marker = draw_pose(Two[0:3, 3], Two[0:3, 0:3], mat=True)
    
    cv2.waitKey(1)

    # step simulation
    for _ in range(sim_steps_per_pbvs):
        p.stepSimulation()

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.set_xlabel("iteration")
ax2.set_xlabel("iteration")
ax1.set_ylabel("error (m)")
ax2.set_ylabel("error (rad)")
ax1.plot(pos_errors, "r")
ax2.plot(rot_errors, "b")
plt.show()