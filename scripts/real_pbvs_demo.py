##########################################################
# This demo detects an end effector via its AR tag and   #
# does PBVS to various predetermined points in the world #
##########################################################
import time

import cv2
import numpy as np
import pybullet as p

from arc_utilities import ros_init
from arc_utilities.reliable_tf import ReliableTF
from visual_servoing.camera import PyBulletCamera, RealsenseCamera
from visual_servoing.pbvs import MarkerPBVS
from visual_servoing.utils import draw_pose, erase_pos
# Key bindings
from arm_robots.hdt_michigan import Val
import rospy
from sensor_msgs.msg import JointState
KEY_U = 117


@ros_init.with_ros("real_pbvs_servoing")
def main():
    # Create a Val
    #val = Val(raise_on_failure=True)
    #val.connect()

    # Create a camera 
    camera = RealsenseCamera(camera_eye=np.array([0.0, 0.0, 0.0]), camera_look=np.array([1.0, 0.0, 0.0]))
    tf_obj = ReliableTF()
    
    # Create a publisher to Val's joints
    command_pub = rospy.Publisher("/hdt_adroit_coms/joint_cmd", JointState, queue_size=10)
    latest_cmd = JointState()

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

    pbvs = MarkerPBVS(camera, 1.1, 1.1, ids, tag_geometry, ids2, tag_geometry)

    Two = None
    Twa = None

    # Transform from AR tag EEF frame to EEF frame
    #rigid_rotation = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler((0, 0, 0)))).reshape(3, 3)
    #Tae = np.zeros((4, 4))
    #Tae[0:3, 0:3] = rigid_rotation
    #Tae[0:3, 3] = np.array([-0.1, 0.0, 0.0])
    #Tae[3, 3] = 1
    Tae = np.eye(4)

    Two = np.eye(4)
    Tce = None
    armed = False

    # Visualization stuff
    client = p.connect(p.GUI)
    uids_target_marker = None
    uids_eef_marker = None
    while True:
        t0 = time.time()

        # Get camera feed and detect markers
        rgb, depth = camera.get_image()
        rgb_edit = rgb[..., [2, 1, 0]].copy()

        if (not armed):
            Two = pbvs.get_target_pose(rgb_edit, depth, np.eye(4))

        # Visualization stuff with PyBullet
        if (uids_target_marker is not None):
            erase_pos(uids_target_marker)
        if(Two is not None):
            uids_target_marker = draw_pose(Two[0:3, 3], Two[0:3, 0:3], mat=True)

        if (uids_eef_marker is not None):
            erase_pos(uids_eef_marker)
        if(Tce is not None):
            uids_eef_marker = draw_pose(Tce[0:3, 3], Tce[0:3, 0:3], mat=True)

        events = p.getKeyboardEvents()
        if KEY_U in events:
            armed=True


        # Do PBVS if there is a target
        ctrl = np.zeros(6)
        if (armed and Two is not None):
            # Compute the control to the end effector in camera space as well as the position
            ctrl, Tce = pbvs.do_pbvs(rgb_edit, depth, Two, Tae, debug=True)
            # Send this transform to TF
            # tf_obj.start_send_transform_matrix(Tce, "camera_color_optical_frame", "eef_ar_tag")

            # From forward kinematics, get position of eef to robot base
            # Teb = tf_obj.get_transform("left_hand", "base_link")
            # We now can get matrix that lets us take eef velocities in camera frame to robot frame
            # Estimate transform from camera to robot base link since we go camera->eef->base
            # Tcb = Teb @ Tce

            #  Compute eef target
            # v = Tcb @ ctrl[0:3]
            # omega = Tcb @ ctrl[3:6]
            # Compute a jacobian

            # Create jacobian
            # command_pub.publish(latest_cmd)

        # Visualization with OCV
        #cv2.imshow("real", rgb_edit)
        cv2.waitKey(10)
        print(time.time() - t0)


if __name__ == '__main__':
    main()
