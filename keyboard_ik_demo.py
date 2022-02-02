##################################################################################
# This demo uses IK to bring the end effector to a position set by the keyboard  #
##################################################################################

from src.utils import draw_pose, erase_pos
from src.val import *
from src.pbvs import *
import math

# Key bindings
KEY_U = 117
KEY_I = 105
KEY_UP = 65297
KEY_RIGHT = 65296
KEY_DOWN = 65298
KEY_LEFT = 65295

# Val robot and PVBS controller
val = Val([0.0, 0, 0])
pbvs = PBVS()
p.setRealTimeSimulation(1)

# create an initial target to do IK too based on the start position of the EEF
perturb = np.zeros((6)) 
perturb[0:3] = 0.1
target = val.get_eef_pos("left") + perturb

# draw the PBVS camera pose
#draw_pose(pbvs.camera_eye, pbvs.get_view(), mat=True, axis_len=0.1)

# AR tag on a box for debugging AR tag detection, commented out
box_pos = (0.0, 2.0, 0.0)
box_orn = [0,0, -np.pi/2 ]
box_vis = p.createVisualShape(p.GEOM_MESH,fileName="AR Tag Static/box.obj", meshScale=[0.1,0.1, 0.1])
#box_multi = p.createMultiBody(baseCollisionShapeIndex = 0, baseVisualShapeIndex=box_vis, basePosition=box_pos, baseOrientation=p.getQuaternionFromEuler(box_orn))

# UIDS for ar tag pose marker 
uids_eef_marker = None
uids_target_marker = None

while(True):
    # Move target marker based on updated target position
     # Draw the pose estimate of the AR tag
    if(uids_target_marker is not None):
        erase_pos(uids_target_marker)
    uids_target_marker = draw_pose(target[0:3], p.getQuaternionFromEuler(target[3:6]))

    # Get camera feed and detect markers
    rgb, depth = pbvs.get_static_camera_img()
    
    # Detect markers and visualize estimated pose in 3D

    # Use ArUco PNP result for orientation estimate
    Rc1a, t, tag_center = pbvs.detect_markers(np.copy(rgb))

    if(tag_center[0] != -1):
        # There are a couple of frames of interest:
        # a is the AR Tag frame, it's z axis is out of the plane of the tag, its y axis is up 
        # c1 is the OpenCV camera frame with +z in the look direction, +y down (google this for image)
        # c2 is the OpenGL/PyBullet camera frame, with -z in the look direction, and +y up 

        # The transformations below take us between these frames
        # Rc2w aka the view matrix is the rotation of the c2 (OpenGL) camera in the world frame
        # Rc2c1 is the rotation of c1 (OpenGL) camera frame in c2 (OpenCV) frame
        # Rc1a from ArUco is the rotation of the AR tag in the c1 (OpenCV) camera frame        
        
        Rc2c1 = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0]
        ])
        Rc2w = pbvs.get_view()

        # Note, we start with the transform from the world to the c2 (OpenGL) camera
        # and right multiply the remaining transforms to transform with respect to the
        # current axis each time and build up the rotations all the way to the AR tag
        # orientation
        Rwa = np.matmul (np.matmul(Rc2w, Rc2c1), Rc1a)

        # This code querys the depth buffer returned from the simulated camera and gets the <x,y,z> 
        # point of the AR tag in world space 
        # Source: https://stackoverflow.com/questions/59128880/getting-world-coordinates-from-opengl-depth-buffer

        # First we compose the projection and view matrix using OpenGL convention (note column major order)
        projectionMatrix = np.asarray(pbvs.projectionMatrix).reshape([4,4],order='F')
        viewMatrix = np.asarray(pbvs.viewMatrix).reshape([4,4],order='F')
        # We now invert the transform that was used to bring points into the normalized image coordinates
        # This maps us from normalized image coordinates back into world coordinates
        T = np.linalg.inv(np.matmul(projectionMatrix, viewMatrix))
        # The AR tag center in u,v pixel coordinates is converted into OpenGL normalized image coordinates 
        x = (2*tag_center[0]- pbvs.image_width)/pbvs.image_width
        y = -(2*tag_center[1]- pbvs.image_height)/pbvs.image_height
        # Z, a depth buffer reading from 0->1 is mapped from -1 to 1
        z = 2 * depth[tag_center[1], tag_center[0]] - 1
        pix = np.asarray([x, y, z, 1])
        # Map points from normalized image coordinates into world
        pos = np.matmul(T, pix)
        # Divide by the w component of the homogenous vector to get cartesian vector
        pos /= pos[3]
  

        # Draw the pose estimate of the AR tag
        if(uids_eef_marker is not None):
            erase_pos(uids_eef_marker)
        uids_eef_marker = draw_pose(pos[0:3], Rwa, mat=True)
        
    cv2.waitKey(10)


    # Process keyboard to adjust target positions
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

    # Set the orientation of our static AR tag object
    #p.resetBasePositionAndOrientation(box_multi, posObj=box_pos, ornObj =p.getQuaternionFromEuler(box_orn) )

    # IK controller to EEF 
    rotx = math.atan2(Rwa[2, 1], Rwa[2,2])
    roty = math.atan2(-Rwa[2,0], (Rwa[2,1]**2 + Rwa[2,2]**2)**0.5)
    rotz = math.atan2(Rwa[1,0], Rwa[0,0])
    euler = np.array([rotx, roty, rotz])
    cur_est = np.hstack((pos[0:3], euler))
    val.psuedoinv_ik("left", target, cur_est)#val.get_eef_pos("left"))
