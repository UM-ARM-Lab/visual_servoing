##########################################################
# This demo detects an end effector via its AR tag and   #
# does PBVS to various predetermined points in the world #
##########################################################

from src.utils import draw_pose, erase_pos
from src.val import *
from src.pbvs import *
from src.camera import *
import time

# Key bindings
KEY_U = 117
KEY_I = 105
KEY_J = 106
KEY_K = 107
KEY_N = 110
KEY_M = 109

# Val robot and PVBS controller
val = Val([0.0, 0.0, 0.0])
camera = PyBulletCamera(camera_eye=np.array([-1.0, 0.5, 0.5]), camera_look=np.array([0, 0.5, 0]))

# create an initial target to do IK too based on the start position of the EEF
perturb = np.zeros((3)) 
perturb[0] = -0.05
perturb[1] = 0.0
perturb[2] = 0.15
initial_arm = val.get_eef_pos("left")[0:3]
target = initial_arm + perturb
Rwo = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler((np.pi/4, 0, -np.pi/2)))).reshape(3,3)

# draw the PBVS camera pose
Tc1c2 = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0 ,0.0, 0.0, 1.0]
    ]) 

#(camera.get_extrinsics()@Tc1c2)[0:3, 0:3]).T
draw_pose(camera.camera_eye, (np.linalg.inv(camera.get_extrinsics())@Tc1c2 )[0:3, 0:3], mat=True, axis_len=0.1)

# AR tag on a box for debugging AR tag detection, commented out
box_pos = (0.0, 2.0, 0.0)
box_orn = [0,0, -np.pi/2 ]
box_vis = p.createVisualShape(p.GEOM_MESH,fileName="AR Tag Static/box.obj", meshScale=[0.1,0.1, 0.1])
#box_multi = p.createMultiBody(baseCollisionShapeIndex = 0, baseVisualShapeIndex=box_vis, basePosition=box_pos, baseOrientation=p.getQuaternionFromEuler(box_orn))


# Specify the 3D geometry of the end effector marker board 
tag_len = 0.0305
gap_len = 0.0051
angle = np.pi/4
# center tag
tag0_tl = np.array([-tag_len/2, tag_len/2, 0.0], dtype=np.float32)
tag0_tr = np.array([tag_len/2, tag_len/2, 0.0], dtype=np.float32)
tag0_br = np.array([tag_len/2, -tag_len/2, 0.0], dtype=np.float32)
tag0_bl = np.array([-tag_len/2, -tag_len/2, 0.0], dtype=np.float32)
z1 = -np.cos(angle) * gap_len
z2 = -np.cos(angle) * (gap_len+tag_len)
y1 = tag_len/2 + gap_len + gap_len*np.sin(angle)
y2 = tag_len/2 + gap_len + (gap_len + tag_len)*np.sin(angle)
# lower tag
tag1_tl = np.array([-tag_len/2, -y1, z1], dtype=np.float32)
tag1_tr = np.array([tag_len/2,  -y1, z1], dtype=np.float32)
tag1_br = np.array([tag_len/2, -y2, z2], dtype=np.float32)
tag1_bl = np.array([-tag_len/2, -y2, z2], dtype=np.float32)
# upper tag
tag2_tl = np.array([-tag_len/2, y2, z2], dtype=np.float32)
tag2_tr = np.array([tag_len/2,  y2, z2], dtype=np.float32)
tag2_br = np.array([tag_len/2, y1, z1], dtype=np.float32)
tag2_bl = np.array([-tag_len/2, y1, z1], dtype=np.float32)

tag0 = np.array([tag0_tl, tag0_tr, tag0_br, tag0_bl])
tag1 = np.array([tag1_tl, tag1_tr, tag1_br, tag1_bl])
tag2 = np.array([tag2_tl, tag2_tr, tag2_br, tag2_bl])
tag_geometry = [tag0, tag1, tag2]
ids = np.array([1,2,3])

pbvs = MarkerPBVS(camera, 1.1, 1.1, ids, tag_geometry)
p.setRealTimeSimulation(1)


# UIDS for ar tag pose marker 
uids_eef_marker = None
uids_target_marker = None

while(True):
    t0 = time.time()    

    # Move target marker based on updated target position
    # Draw the pose estimate of the AR tag
    if(uids_target_marker is not None):
        erase_pos(uids_target_marker)
    uids_target_marker = draw_pose(target[0:3], Rwo, mat=True)

    # Get camera feed and detect markers
    rgb, depth = camera.get_image()
    rgb_edit = np.copy(rgb)
    print(f"image time {time.time()-t0}")
    
    # Use ArUco PNP result for orientation estimate
    markers = pbvs.detect_markers(rgb_edit)
    ref_marker = pbvs.get_board_pose(markers, pbvs.eef_board, rgb_edit)
    cv2.imshow("image", rgb_edit)

    pos = None
    if(ref_marker is not None):
        Tcm =  ref_marker.Tcm 
        Rwa = (np.linalg.inv(camera.get_extrinsics()) @ Tcm)[0:3, 0:3]
        pos_unstable = (np.linalg.inv(camera.get_extrinsics()) @ Tcm)[0:3, 3]
        pos = camera.get_xyz(ref_marker.c_x, ref_marker.c_y, depth)
        #print(np.linalg.norm(pos-pos_unstable))
        #draw_sphere_marker(pos_unstable, 0.01, (1.0, 0.0, 0.0, 1.0))
        if(uids_eef_marker is not None):
            erase_pos(uids_eef_marker)
        uids_eef_marker = draw_pose(pos[0:3], Rwa[0:3, 0:3], mat=True)
        pass
    
    cv2.waitKey(1)


    # Process keyboard to adjust target positions
    events = p.getKeyboardEvents()
    if(KEY_U in events):
        target = initial_arm + np.array([-0.1, -0.2, 0.1])
        Rwo = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler((np.pi/7, np.pi/4, -np.pi/2)))).reshape(3,3)
    if(KEY_I in events):
        continue
    
    # Set the orientation of our static AR tag object
    #p.resetBasePositionAndOrientation(box_multi, posObj=box_pos, ornObj =p.getQuaternionFromEuler(box_orn) )
    if(pos is not None):
        ctrl = pbvs.get_control(target, pos, Rwa, Rwo)
        val.psuedoinv_ik_controller("left", ctrl)
    print(time.time()-t0)