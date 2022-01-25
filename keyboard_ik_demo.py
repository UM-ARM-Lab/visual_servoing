# Control end effector position with keyboard
from fileinput import filename
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
val = Val([-5.0, 0, 0])
pbvs = PBVS()
p.setRealTimeSimulation(1)

# create an initial target to do IK too
perturb = np.zeros((6)) 
perturb[0:3] = 0.1
target = val.get_eef_pos("left") + perturb

#draw_pose(pbvs.camera_eye, pbvs.get_view(), mat=True, axis_len=0.1)

'''
draw_pose(pbvs.camera_eye, np.matmul(np.matmul(np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0]
        ]), pbvs.get_view()), ), mat=True, axis_len=0.1)
'''

# visual shape
box_pos = (0.0, 2.0, 0.0)
box_orn = [np.pi/8,-np.pi/8, -np.pi/2 ]
box_vis = p.createVisualShape(p.GEOM_MESH,fileName="AR Tag Static/box.obj", meshScale=[0.1,0.1, 0.1])
box_multi = p.createMultiBody(baseCollisionShapeIndex = 0, baseVisualShapeIndex=box_vis, basePosition=box_pos, baseOrientation=p.getQuaternionFromEuler(box_orn))
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
    Rc1a, t, tag_center = pbvs.detect_markers(np.copy(rgb))
    if(tag_center[0] != -1):
        Rc2c1 = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0]
        ])

        Rwc2 = pbvs.get_view().T
        
        # Rotation of ar tag in world relative frame by adding known camera rotation
        Rwa =  np.matmul(np.matmul(Rwc2, Rc2c1), Rc1a)#
        Raw = np.matmul ( Rc1a.T,np.matmul(np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0]
        ]), pbvs.get_view()))
        #print(Rc1a)
        print(pbvs.get_intrinsics())
        
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
  
        if(uids is not None):
            erase_pos(uids)
        uids = draw_pose(pos[0:3], Raw, mat=True)
        #uids = draw_pose(pbvs.camera_eye, Raw, mat=True)

        #draw_sphere_marker(pos[0:3], 0.03, (1.0, 0.0, 1.0, 1.0)) 
        
    cv2.waitKey(10)


    # Process keyboard to adjust target position
    events = p.getKeyboardEvents()
    if(KEY_LEFT in events):
        target += np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
        box_orn[0]+=0.1
    elif(KEY_RIGHT in events):
        target -= np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
        box_orn[0]-=0.1

    if(KEY_UP in events):
        target += np.array([0.00, 0.00, 0.01, 0.0, 0.0, 0.0])
        box_orn[1]+=0.1
    elif(KEY_DOWN in events):
        target -= np.array([0.00, 0.00, 0.01, 0.0, 0.0, 0.0])
        box_orn[1]-=0.1

    if(KEY_U in events):
        target += np.array([0.00, 0.00, 0.00, 0.01, 0.0, 0.0])

    p.resetBasePositionAndOrientation(box_multi, posObj=box_pos, ornObj =p.getQuaternionFromEuler(box_orn) )
    # IK controller to target
    #val.psuedoinv_ik("left", target, val.get_eef_pos("left"))
