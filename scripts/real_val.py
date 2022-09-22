from arm_robots.hdt_michigan import Val
from arc_utilities import ros_init
from visual_servoing.marker_pbvs import MarkerPBVS, MarkerBoardDetector
from visual_servoing.camera import RealsenseCamera
import rospy
import numpy as np
import cv2

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


@ros_init.with_ros("real_pbvs_servoing")
def main():
    detector = MarkerBoardDetector(ids, tag_geometry)
    target_detector = MarkerBoardDetector(ids2, tag_geometry)
    camera = RealsenseCamera(np.zeros(3), np.array([0, 0, 1]), ())
    pbvs = MarkerPBVS(camera, 1, 1, 1.5, detector)

    # Create a Val
    #val = Val(raise_on_failure=True)
    #val.connect()

    while(True):
        rgb = camera.get_image()[:, :, :3]
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)

        detector.update(rgb, camera.get_intrinsics())
        target_detector.update(rgb, camera.get_intrinsics())
        #pbvs.do_pbvs(rgb, None, np.eye(4), np.eye(4))
        #pbvs.do_pbvs(rgb, None, )
        cv2.imshow("image", rgb)
        cv2.waitKey(1)

    #r = rospy.Rate(10) 
    #for i in range(10):
    #    r.sleep()
    #    #val.send_velocity_joint_command(val.get_joint_names("right_arm"), np.array([1, 0, 0, 0, 0, 0, 0]))

main()