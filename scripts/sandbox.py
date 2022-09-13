from dis import get_instructions
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
        [-1/z, 0, u/z, u*v, -1* (1+u**2), v],
        [0, -1/z, v/z, (1+v**2), -u*v, -u]
    ])
uids_camera_marker = None
uids_eef_marker = None
uids_target_marker = None

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

while(True):

    # Step sim
    for _ in range(24):
        link_pos, link_rot = val.get_eef_pos("camera")
        camera_rot = np.array(p.getMatrixFromQuaternion(link_rot)).reshape(3,3)
        camera.upate_from_pose(link_pos, camera_rot)
        #draw_sphere_marker(link_pos, 0.01, (1, 0, 0, 1))

        p.stepSimulation()
    
    rgb, depth = camera.get_image()
    #markers = pbvs.detect_markers(rgb)
    #ref_marker = pbvs.get_board_pose(markers, pbvs.eef_board, rgb)
    #Tcm = ref_marker.Tcm
    #L = get_interaction_mat(ref_marker.c_x, ref_marker.c_y, Tcm[2, 3])
    #J = val.get_camera_jacobian() 

    # Servo
    Two = np.eye(4) 
    Two[0:3, 3] = np.array([0.8, 0.0, 0.2])
    Two[0:3, 0:3] = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler((np.pi/2, np.pi/4, 0)))).reshape(3, 3)
    twist, Twe = pbvs.do_pbvs(rgb, depth, Two, np.eye(4), val.get_arm_jacobian("left"), val.get_jacobian_pinv("left"), 24)

    # Torso control
    ref_marker = pbvs.ref_marker
    Tcm = ref_marker.Tcm

    # Get another pt
    pt2_cam = Tcm @ np.array([0, 0, 0.1, 1])
    pt2_im = camera.get_intrinsics() @ pt2_cam[0:3]
    pt2_im /= pt2_im[2]
    px2 = pt2_im[0:2].astype(int)
    cv2.circle(rgb, tuple(px2), 4, (255, 0, 255), -1)
    
    L1 = get_interaction_mat(ref_marker.c_x, ref_marker.c_y, Tcm[2, 3])
    L2 = get_interaction_mat(px2[0], px2[1], pt2_cam[2])
    L = np.vstack((L1, L2))
    ctrl = np.array([
        -(ref_marker.c_x - camera.image_dim[0]/2), -(ref_marker.c_y - camera.image_dim[1]/2),
        -(px2[0] - camera.image_dim[0]/2), -(px2[1] - camera.image_dim[1]/2),
    ]
    )

    jac = val.get_camera_jacobian()
    Q = np.eye(4) * 100
    P = jac.T @ L.T @ Q @ L @ jac
    q = (-ctrl @ Q @ L @ jac)
    num_joints = jac.shape[1]
    G = np.vstack((np.eye(num_joints), -np.eye(num_joints)))
    h = np.ones(num_joints * 2) * 3.5
    num_joints = jac.shape[1]
    ctrl = solve_qp(P, q, G, h, None, None, solver="cvxopt")
    val.torso_control(ctrl)

    # Servo
    J = val.get_arm_jacobian("left")
    lmda = 0.0000001
    J_pinv = np.linalg.inv(J.T @ J + lmda * np.eye(7)) @ J.T
    q_dot = J_pinv @ twist
    val.velocity_control("left", q_dot)

    # Visualize camera pose 
    if (uids_camera_marker is not None):
        erase_pos(uids_camera_marker)
    Twc = np.linalg.inv(camera.get_extrinsics())
    uids_camera_marker = draw_pose(Twc[0:3, 3], Twc[0:3, 0:3], mat=True)

    # Visualize estimated end effector pose 
    if (uids_eef_marker is not None):
        erase_pos(uids_eef_marker)
    uids_eef_marker = draw_pose(Twe[0:3, 3], Twe[0:3, 0:3], mat=True)

    # Visualize target pose 
    if (uids_target_marker is not None):
        erase_pos(uids_target_marker)
    uids_target_marker = draw_pose(Two[0:3, 3], Two[0:3, 0:3], mat=True)
    cv2.imshow("Im", rgb)
    cv2.waitKey(1)

    #cv2.waitKey(1)
