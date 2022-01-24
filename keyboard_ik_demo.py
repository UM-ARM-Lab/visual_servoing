# Control end effector position with keyboard
from src.utils import draw_pose, erase_pos, get_true_depth
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
#marker = draw_pose(target[0:3], p.getQuaternionFromEuler(target[3:6]))

#draw_pose(val.get_eef_pos("left")[0:3], p.getQuaternionFromEuler(val.get_eef_pos("left")[3:6]))

#draw_sphere_marker(pbvs.camera_eye, 0.07, (1.0, 0.0, 0.0, 1.0))
Q = np.array([
    [0.0, -1.0, 0.0], 
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
])
draw_pose(pbvs.camera_eye, pbvs.get_view(), mat=True)
#draw_pose(pbvs.camera_eye, Q, mat=True)

uids = None

while(True):
    # Move target marker based on updated target position
    #marker_new = draw_pose(target[0:3], p.getQuaternionFromEuler(target[3:6]))
    #erase_pos(marker)
    #marker = marker_new

    # Get camera feed and detect markers
    rgb, depth = pbvs.get_static_camera_img()
    #cv2.imshow("depth", depth)

    # Rotation from ar tag frame to camera 
    Rca, t, tag_center = pbvs.detect_markers(np.copy(rgb))
    if(tag_center[0] != -1):
        print(Rca)
        # Rotation of ar tag in camera relative frame
        # note that viewMatrix of the camera is rotation of the world relative to the camera
        Rwc = pbvs.get_view()
        #Rwc = Rcw.T
        # Rotation of ar tag in world relative frame by adding known camera rotation
        Rwa =np.matmul(Rca, Rwc)
        #print(np.matmul(Rwa, Rwa.T))
        
        # source: https://stackoverflow.com/questions/59128880/getting-world-coordinates-from-opengl-depth-buffer
        projectionMatrix = np.asarray(pbvs.projectionMatrix).reshape([4,4],order='F')
        viewMatrix = np.asarray(pbvs.viewMatrix).reshape([4,4],order='F')
        T = np.linalg.inv(np.matmul(projectionMatrix, viewMatrix))
        x = (2*tag_center[0]- pbvs.image_width)/pbvs.image_width
        y = -(2*tag_center[1]- pbvs.image_height)/pbvs.image_height
        z = 2 * depth[tag_center[1], tag_center[0]] - 1
        pix = np.asarray([x, y, z, 1])
        pos = np.matmul(T, pix)
        pos /= pos[3]
        #print(pos)
        #print(Rwa)
        if(uids is not None):
            erase_pos(uids)
        uids = draw_pose(pos[0:3], Rwa, mat=True)

        #draw_sphere_marker(pos[0:3], 0.03, (1.0, 0.0, 1.0, 1.0)) 
        
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
