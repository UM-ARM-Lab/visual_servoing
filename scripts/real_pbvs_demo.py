##########################################################
# This demo detects an end effector via its AR tag and   #
# does PBVS to various predetermined points in the world #
##########################################################
import time

import cv2
import numpy as np
import pybullet as p

from arc_utilities import ros_init
from visual_servoing.camera import PyBulletCamera, RealsenseCamera
from visual_servoing.pbvs import MarkerPBVS
from visual_servoing.utils import draw_pose, erase_pos
# Key bindings
from visual_servoing.val import Val


@ros_init.with_ros("real_pbvs_servoing")
def main():
    camera = RealsenseCamera(camera_eye=np.array([0.0, 0.0, 0.0]), camera_look=np.array([1.0, 0.0, 0.0]))

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

    pbvs = MarkerPBVS(camera, 1.1, 1.1, ids, tag_geometry)

    Two = None
    Twa = None

    # Transform from AR tag EEF frame to EEF frame
    rigid_rotation = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler((0, 0, 0)))).reshape(3, 3)
    Tae = np.zeros((4, 4))
    Tae[0:3, 0:3] = rigid_rotation
    Tae[0:3, 3] = np.array([-0.1, 0.0, 0.0])
    Tae[3, 3] = 1
    initial_arm = None

    Two = np.eye(4)
    while True:
        t0 = time.time()
        #time.sleep(5)

        # Get camera feed and detect markers
        rgb, depth = camera.get_image()
        rgb_edit = rgb[..., [2, 1, 0]].copy()

        # Do PBVS if there is a target
        ctrl = np.zeros(6)
        if (Two is not None):
            ctrl, Twe = pbvs.do_pbvs(rgb_edit, depth, Two, Tae)

        # Execute control on Val
        cv2.imshow("real", rgb)
        cv2.waitKey(3)

        #print(time.time() - t0)


if __name__ == '__main__':
    main()
