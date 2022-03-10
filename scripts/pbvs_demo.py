##########################################################
# This demo detects an end effector via its AR tag and   #
# does PBVS to various predetermined points in the world #
##########################################################

from visual_servoing.utils import draw_pose, draw_sphere_marker, erase_pos
from visual_servoing.val import *
from visual_servoing.pbvs import *
from visual_servoing.camera import *
import time
import matplotlib.pyplot as plt

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

#camera = PyBulletCamera(camera_eye=np.array([0.7, -0.3, 0.5]), camera_look=np.array([0.7, 0.0, 0.2]))
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
box_pos = (0.8, 0.3, 0.4)
#box_orn = [0, 0, np.pi/8]
box_orn = [np.pi/2, np.pi, 3*np.pi/2]

box_vis = p.createVisualShape(p.GEOM_MESH,fileName="models/AR Tag Cuff 2/PINCER_HOUSING2_EDIT.obj", meshScale=[1.0,1.0, 1.0])
box_multi = p.createMultiBody(baseCollisionShapeIndex = 0, baseVisualShapeIndex=box_vis, basePosition=box_pos, baseOrientation=p.getQuaternionFromEuler(box_orn))


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

pbvs = MarkerPBVS(camera, 0.9, 0.9, ids, tag_geometry, ids2, tag_geometry, False, 0.55)
sim_dt = 0.1#1/240
p.setTimeStep(sim_dt)
#p.setRealTimeSimulation(1)

Two = None
Twa = None

# UIDS for ar tag pose marker 
uids_eef_marker = None
uids_target_marker = None
uids_eef_gt = None
# Transform from AR tag EEF frame to EEF frame
rigid_rotation = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler((0, 3*np.pi/2, np.pi)))).reshape(3, 3)
Tae = np.zeros((4, 4))
Tae[0:3, 0:3] = rigid_rotation
Tae[0:3, 3] = np.array([0.14265 - 0.002, 0.03797 -0.0288 - 0.012, -0.008620875 -0.0376])
Tae[3, 3] = 1
Tae = np.eye(4)

# Transform from AR tag to target frame
Tao = np.zeros((4, 4))
Tao[0:3, 0:3] = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler((0, 0, 0)))).reshape(3, 3)
Tao[0:3, 3] = np.array([0.0, 0.0, 0.1])
Tao[3, 3] = 1

initial_arm = val.get_eef_pos("left")[0:3]


#delete me
Two = np.eye(4) 
Two[0:3, 3] = np.array([0.8, 0.5, 0.5])
Two[0:3, 0:3] = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler((np.pi/4, np.pi/4, 0)))).reshape(3, 3)

position_error = []
rotation_error = []

marker_pos_error = []
marker_rot_error = []

armed = False

while True:
    t0 = time.time()

    p.stepSimulation()
    # Visualization ground truth AR link [delete me]
    tool_idx = val.left_tag[0]
    result = p.getLinkState(val.urdf,
                            tool_idx,
                            computeLinkVelocity=1,
                            computeForwardKinematics=1)

    link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
#    draw_pose(link_trn, link_rot)
    #  Visualize eef gripper ground truth 
    #    if (uids_eef_gt is not None):
    #        erase_pos(uids_eef_gt)
    #    gt_t, gt_o = val.get_eef_pos("left") 
    #    uids_eef_gt = draw_pose(gt_t, gt_o)
    #
    # Get camera feed and detect markers
    rgb, depth = camera.get_image()
    rgb_edit = rgb[..., [2, 1, 0]].copy()

    # Do PBVS if there is a target 
    ctrl = np.zeros(6)
    #cv2.imshow("image", rgb_edit)
    if armed and Two is not None:
        ctrl, Twe = pbvs.do_pbvs(rgb_edit, depth, Two, Tae, val.get_arm_jacobian("left"), val.get_jacobian_pinv("left"), sim_dt)

        val.set_velo(val.get_jacobian_pinv("left") @ ctrl)
        # Visualize estimated end effector pose 
        if (uids_eef_marker is not None):
            erase_pos(uids_eef_marker)
        if(Twe is not None):
            uids_eef_marker = draw_pose(Twe[0:3, 3], Twe[0:3, 0:3], mat=True)

        #  Visualize target pose 
        if (uids_target_marker is not None):
            erase_pos(uids_target_marker)
        uids_target_marker = draw_pose(Two[0:3, 3], Two[0:3, 0:3], mat=True)
        #print(gt_t - Twe[0:3, 3])

        if(Twe is not None):
            position_error.append( np.linalg.norm(Twe[0:3, 3] - Two[0:3, 3]))
            r, _ = cv2.Rodrigues( (Twe[0:3, 0:3] @ Two[0:3, 0:3].T ).T)
            rotation_error.append(np.linalg.norm(r))

            marker_pos_error.append(np.linalg.norm(Twe[0:3, 3] - link_trn))
            link_rod, _ = cv2.Rodrigues(np.array(p.getMatrixFromQuaternion(link_rot)).reshape(3,3) @ Twe[0:3, 0:3].T)
            marker_rot_error.append(np.linalg.norm(link_rod))


    # Execute control on Val
    #val.set_velo(val.get_jacobian_pinv("left") @ ctrl)

    # Do target pose estimation while not armed, but stop once servoing starts so occlusions don't get in the way
    if(not armed):
        Two = pbvs.get_target_pose(rgb_edit, depth, Tao)
        if Two is not None:
            if (uids_target_marker is not None):
                erase_pos(uids_target_marker)
            uids_target_marker = draw_pose(Two[0:3, 3], Two[0:3, 0:3], mat=True)
    
    cv2.waitKey(1)

    # Process keyboard to change target position
    events = p.getKeyboardEvents()
    if KEY_N in events:
        armed = True

    if KEY_J in events:
        armed = True
        initial_arm = val.get_eef_pos("left")[0:3]
        perturb = np.zeros((3))
        perturb[0] = -0.05
        perturb[1] = 0.0
        perturb[2] = 0.15
        target = initial_arm + perturb
        Rwo = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler((np.pi/2, np.pi/4, np.pi)))).reshape(3,3)
        #Rwo = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler((np.pi / 4, 0, -np.pi / 2)))).reshape(3, 3)
        Two = np.zeros((4, 4))
        Two[0:3, 0:3] = Rwo
        Two[0:3, 3] = target
        Two[3, 3] = 1
    if (KEY_I in events):
        plt.plot(position_error)
        plt.xlabel("iteration")
        plt.ylabel("meters")
        plt.title("EEF Position error (m)")
        plt.figure()
        plt.xlabel("iteration")
        plt.ylabel("Rodrigues norm")
        plt.title("EEF Rotation error")
        plt.plot(rotation_error)
        plt.show()

    # Set the orientation of our static AR tag object
    # p.resetBasePositionAndOrientation(box_multi, posObj=box_pos, ornObj =p.getQuaternionFromEuler(box_orn) )
    # p.stepSimulation()
    print(time.time() - t0)
