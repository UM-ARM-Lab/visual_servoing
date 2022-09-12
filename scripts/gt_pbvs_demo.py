##########################################################
# This demo detects an end effector via its AR tag and   #
# does PBVS to various predetermined points in the world #
##########################################################

from visual_servoing.utils import draw_pose, draw_sphere_marker, erase_pos
from visual_servoing.val import *
from visual_servoing.pbvs import CheaterPBVS, PBVS
from visual_servoing.camera import *
from visual_servoing.marker_pbvs import *
from visual_servoing.pbvs_loop import PybulletPBVSLoop, AbstractPBVSLoop
import time
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pybullet as p

# Target 
Two = np.eye(4) 
#Two[0:3, 3] = np.array([0.8, 0.24, 0.3])
Two[0:3, 3] = np.array([0.8, 0.0, 0.2])
Two[0:3, 0:3] = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler((np.pi/2, np.pi/4, 0)))).reshape(3, 3)

def get_eef_gt(robot):
    '''
    Gets the ground truth pose of the end effector from the simulator
    '''
    tool_idx = robot.left_tag[0]
    result = p.getLinkState(robot.urdf,
                            tool_idx,
                            computeLinkVelocity=1,
                            computeForwardKinematics=1)

    link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
    Twe = np.eye(4)
    Twe[0:3, 0:3] = np.array(p.getMatrixFromQuaternion(link_rot)).reshape(3, 3)
    Twe[0:3, 3] = link_trn
    return Twe

class GtValLoop(PybulletPBVSLoop):

    def __init__(self, pbvs: PBVS, camera: PyBulletCamera, robot: Val, side: str, pbvs_hz: float, sim_hz: float,
                 config ):
        super().__init__(pbvs, camera, robot, side, pbvs_hz, sim_hz, config)
        self.uids_eef_marker = None
        self.uids_target_marker = None
        self.uids_camera_marker = None
        self.pos_errors = []
        self.rot_errors = []
    
    def on_after_step_pbvs(self, Twe):

        # Visualize estimated end effector pose 
        if (self.uids_eef_marker is not None):
            erase_pos(self.uids_eef_marker)
        self.uids_eef_marker = draw_pose(Twe[0:3, 3], Twe[0:3, 0:3], mat=True)

        # Visualize target pose 
        if (self.uids_target_marker is not None):
            erase_pos(self.uids_target_marker)
        self.uids_target_marker = draw_pose(Two[0:3, 3], Two[0:3, 0:3], mat=True)

        # Visualize camera pose 
        if (self.uids_camera_marker is not None):
            erase_pos(self.uids_camera_marker)
        Twc = np.linalg.inv(self.camera.get_extrinsics())

        self.uids_camera_marker = draw_pose(Twc[0:3, 3], Twc[0:3, 0:3], mat=True)
        #draw_sphere_marker(Twe[0:3, 3], 0.01, (1, 0, 0, 1))

        # Errors 
        truth = get_eef_gt(self.robot)
        #draw_pose(truth[0:3, 3], truth[0:3, 0:3], mat=True)
        self.pos_errors.append(np.linalg.norm(Twe[0:3, 3] - truth[0:3, 3]))
        link_rod, _ = cv2.Rodrigues(Twe[0:3, 0:3] @ truth[0:3, 0:3].T)
        self.rot_errors.append(np.linalg.norm(link_rod))

        super().on_after_step_pbvs(Twe)

    def servoing_controller(self, twist):
        # Note, this is a hack since we are working in joint
        # space directly
        return twist

    def on_before_step_pbvs(self, rgb, depth):
        cv2.imshow("test", rgb)
        cv2.waitKey(1)

        # Update camera GT
        link_pos, link_rot = self.robot.get_eef_pos("camera")
        camera_rot = np.array(p.getMatrixFromQuaternion(link_rot)).reshape(3,3)
        self.camera.upate_from_pose(link_pos, camera_rot)

        return super().on_before_step_pbvs(rgb, depth)

def main():
    # Objects needed to do PBVS
    camera = PyBulletCamera(camera_eye=np.array([0.7, -0.8, 0.5]), camera_look=np.array([0.7, 0.0, 0.2]))
    #camera = PyBulletCamera(camera_eye=np.array([0.4, 0.0, 0.4]), camera_look=np.array([0.7, 0.0, 0.3]))

    val = Val([0.0, 0.0, -0.5])
    pbvs = CheaterPBVS(camera, 1, 1, 1.5, lambda : get_eef_gt(val))
    loop = GtValLoop(pbvs, camera, val, "left", 10, 240, {
        "timeout": 60,
        "max_pos_error": 0.03, 
        "max_rot_error": 0.1, 
    }) 
    loop.run(Two)

    # Plot ground truth vs predicted poses at each iteration of the loop
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_xlabel("iteration")
    ax2.set_xlabel("iteration")
    ax1.set_ylabel("error (m)")
    ax2.set_ylabel("error (rad)")
    ax1.plot(loop.pos_errors, "r")
    ax2.plot(loop.rot_errors, "b")
    plt.show()

if __name__ == "__main__":
    main()