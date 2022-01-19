# Control end effector position with keyboard

from src.utils import draw_pose, erase_pos
from src.val import *
from src.pbvs import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Key bindings
KEY_U = 117
KEY_I = 105
KEY_UP = 65297
KEY_RIGHT = 65296
KEY_DOWN = 65298
KEY_LEFT = 65295

# Val robot and PVBS controller
val = Val()
pbvs = PBVS()
p.setRealTimeSimulation(1)

# create an initial target to do IK too
perturb = np.zeros((6)) 
perturb[0:3] = 0.1
target = val.get_eef_pos("left") + perturb
marker = draw_pose(target[0:3], p.getQuaternionFromEuler(target[3:6]))

draw_pose(val.get_eef_pos("left")[0:3], p.getQuaternionFromEuler(val.get_eef_pos("left")[3:6]))

while(True):
    # Move target marker based on updated target position
    marker_new = draw_pose(target[0:3], p.getQuaternionFromEuler(target[3:6]))
    erase_pos(marker)
    marker = marker_new

    # Get camera feed and detect markers
    rgb, depth = pbvs.get_static_camera_img()
    # Rotation of camera in AR tag relative frame
    Rac, t, tag_center = pbvs.detect_markers(np.copy(rgb))
    # Rotation of ar tag in camera relative frame
    Rca = Rac.T
    # note that viewMatrix of the camera is rotation of the world relative to the camera
    Rcw = np.array(pbvs.viewMatrix)[0:3, 0:3].reshape(3, 3)
    Rwc = Rcw.T
    # Rotation of ar tag in world relative frame by adding known camera rotation
    Rwa = np.matmul(Rca, Rwc)
    #draw_pose()

    #print(f"[Stereo] dist: {0}, depth: {depth[tag_center[1]][tag_center[0]] }")
    cv2.waitKey(10)

    # Process keyboard to adjust target position
    events = p.getKeyboardEvents()
    if(KEY_LEFT in events):
        target += np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    elif(KEY_RIGHT in events):
        target -= np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])

    if(KEY_UP in events):
        target += np.array([0.00, 0.00, 0.01, 0.0, 0.0, 0.0])
    elif(KEY_DOWN in events):
        target -= np.array([0.00, 0.00, 0.01, 0.0, 0.0, 0.0])

    if(KEY_U in events):
        target += np.array([0.00, 0.00, 0.00, 0.01, 0.0, 0.0])

    # IK controller to target
    #val.psuedoinv_ik("left", target, val.get_eef_pos("left"))
