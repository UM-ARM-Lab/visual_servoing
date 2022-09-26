from dis import get_instructions
from tkinter import W
import cv2
from visual_servoing.pbvs import CheaterPBVS
from visual_servoing.marker_pbvs import MarkerPBVS
from visual_servoing.val import Val
from visual_servoing.utils import *
from visual_servoing.camera import PyBulletCamera
import numpy as np
import pybullet as p
from qpsolvers import solve_qp
from pytorch_mppi import mppi

val = Val([0.0, 0.0, -0.5])
camera = PyBulletCamera(camera_eye=np.array([0.7, -0.8, 0.5]), camera_look=np.array([0.7, 0.0, 0.2]))
p.setGravity(0, 0, -10)

def get_interaction_mat(u, v, z):
    return np.array([
        [-1/z, 0, u/z, u*v, -1* (1+u**2), v],
        [0, -1/z, v/z, (1+v**2), -u*v, -u]
    ])
uids_camera_marker = None
uids_eef_marker = None
uids_target_marker = None
uids_pred_eef_marker = None

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

#pbvs = MarkerPBVS(camera, 1, 1, 1.5, np.eye(4), ids, tag_geometry, ids2, tag_geometry)
def get_eef_gt(robot):
    '''
    Gets the ground truth pose of the end effector from the simulator
    '''
    tool_idx = robot.left_tag[0]
    result = p.getLinkState(robot.urdf,
                            tool_idx,
                            computeLinkVelocity=1,
                            computeForwardKinematics=1)

    link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
    Twe = np.eye(4)
    Twe[0:3, 0:3] = np.array(p.getMatrixFromQuaternion(link_rot)).reshape(3, 3)
    Twe[0:3, 3] = link_trn
    return Twe
pbvs = CheaterPBVS(camera, 1, 1, 1.5, lambda : get_eef_gt(val))

def add_global_twist(twist, pose, dt):
    r3 = dt * twist[:3]
    so3, _ = cv2.Rodrigues(dt * twist[3:])
    tf = np.eye(4)
    tf[0:3, 0:3] = so3 @ pose[0:3, 0:3]
    tf[0:3, 3] = r3 + pose[0:3, 3]
    return tf

def reproject(Tcm, point_marker, camera):
    # Get another pt
    pt2_cam = Tcm @ point_marker
    pt2_im = camera.get_intrinsics() @ pt2_cam[0:3]
    pt2_im /= pt2_im[2]
    px2 = pt2_im[0:2].astype(int)
    return px2, pt2_cam

# Target
Two = np.eye(4) 
Two[0:3, 3] = np.array([0.8, 0.0, 0.2])
#Two[0:3, 0:3] = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler((np.pi/2, np.pi/4, 0)))).reshape(3, 3) # really hard
Two[0:3, 3] = np.array([0.75, 0.2, 0.2])
Two[0:3, 0:3] = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler((np.pi/2, 0, np.pi/10)))).reshape(3, 3) # hard
#Two[0:3, 0:3] = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler((np.pi/2, 0, -np.pi/4)))).reshape(3, 3) # really easy

rgb, depth = camera.get_image()
cv2.imshow("Im", rgb)
cv2.waitKey(1)


def batchable_dynamics_arm():
    """
    Arm dynamics function
    x_(t+1) = x_t + dt J q'
    """

while(True):

    # Step sim
    for _ in range(24):
        link_pos, link_rot = val.get_eef_pos("camera")
        camera_rot = np.array(p.getMatrixFromQuaternion(link_rot)).reshape(3,3)
        camera.upate_from_pose(link_pos, camera_rot)
        p.stepSimulation()
    
    rgb, depth = camera.get_image()

    mppi.MPPI()

    # Send command to val
    val.velocity_control("left", ctrl, True)

    # Visualize camera pose  
    if (uids_camera_marker is not None):
        erase_pos(uids_camera_marker)
    Twc = np.linalg.inv(camera.get_extrinsics())
    uids_camera_marker = draw_pose(Twc[0:3, 3], Twc[0:3, 0:3], mat=True)

    cv2.imshow("Im", rgb)
    cv2.waitKey(1)

    # Visualize target pose 
    if (uids_target_marker is not None):
        erase_pos(uids_target_marker)
    uids_target_marker = draw_pose(Two[0:3, 3], Two[0:3, 0:3], mat=True)