import hjson
import time
import pybullet as p
import cv2
from visual_servoing.camera import *
import numpy as np
from visual_servoing.icp_pbvs import *
from visual_servoing.victor import *
from visual_servoing.utils import *

def create_target_tf(target_pos, target_rot):
    H = np.eye(4)
    H[0:3, 0:3] = np.array(p.getMatrixFromQuaternion(target_rot)).reshape(3, 3)
    H[0:3, 3] = target_pos
    return H

def get_eef_gt_tf(victor, camera, world_relative=False):
    # Get EEF Link GT 
    tool_idx = victor.eef_idx
    result = p.getLinkState(victor.urdf,
                            tool_idx,
                            computeLinkVelocity=1,
                            computeForwardKinematics=1)
    link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
    Tcw = camera.get_view()
    Twe = np.eye(4)
    Twe[0:3, 0:3] = np.array(p.getMatrixFromQuaternion(frame_rot)).reshape(3, 3)
    Twe[0:3, 3] = frame_pos
    if(world_relative):
        return Twe
    else:
        Tce = Tcw @ Twe
        return Tce

def run_servoing(pbvs, camera, victor, target, config):
    pose_est_uids = None
    target_uids = None

    p.setTimeStep(1/config['sim_hz'])
    sim_steps_per_pbvs = int(config['sim_hz']/config['pbvs_hz']) 
    start_time = time.time()

    if(config['vis']):
        draw_pose(target[0:3, 3], target[0:3, 0:3], mat=True, uids=target_uids)

    while(True):
        # check if timeout exceeded
        if(time.time() - start_time > config['timeout']):
            return 0

        # check if error is low enough to terminate
        

        # get camera image
        rgb, depth, seg = camera.get_image(True)
        rgb_edit = rgb[..., [2, 1, 0]].copy()
        
        # do visual servo
        ctrl, Twe = pbvs.do_pbvs(depth, target, victor.get_arm_jacobian('left'),
                                victor.get_jacobian_pinv('left'), 1/config['pbvs_hz'])
        victor.psuedoinv_ik_controller("left", ctrl)

        # draw debug stuff
        if(config['vis']):
            erase_pos(pose_est_uids)
            draw_pose(Twe[0:3, 3], Twe[0:3, 0:3], mat=True, uids=pose_est_uids)
            cv2.imshow("Camera", cv2.resize(rgb_edit, (1280 // 5, 800 // 5))) 

        
        # step simulation
        for _ in range(sim_steps_per_pbvs):
            p.stepSimulation()

def main():
    # Loads hjson config to do visual servoing with
    config_file = open('config.hjson', 'r')
    config_text = config_file.read()
    config = hjson.loads(config_text)

    servo_configs = config['servo_configs']
    for servo_config in servo_configs:
        camera = PyBulletCamera(np.array(servo_config['camera_pos']), np.array(servo_config['camera_look']))
        victor = Victor()
        target = create_target_tf(np.array(servo_config['target_pos']), np.array(servo_config['target_rot'])) 
        pbvs = ICPPBVS(camera, 1, 1, get_eef_gt_tf(victor, camera), 
            config['pbvs_settings']['max_joint_velo'], config['pbvs_settings']['seg_range']) 
        run_servoing(pbvs, camera, victor, target, config)
        

main()