import hjson
import time
import pybullet as p
import cv2
from visual_servoing.camera import *
import numpy as np
from visual_servoing.icp_pbvs import *
from visual_servoing.victor import *
from visual_servoing.utils import *
from datetime import datetime
import pickle
import os
import shutil

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

def run_servoing(pbvs, camera, victor, target, config, result_dict):
    pose_est_uids = None
    target_uids = None

    p.setTimeStep(1/config['sim_hz'])
    sim_steps_per_pbvs = int(config['sim_hz']/config['pbvs_hz']) 
    start_time = time.time()

    if(config['vis']):
        cam_inv = np.linalg.inv(camera.get_view())
        draw_pose(target[0:3, 3], target[0:3, 0:3], mat=True, uids=target_uids)
        draw_pose(cam_inv[0:3, 3], cam_inv[0:3, 0:3], mat=True)

    while(True):
        # check if timeout exceeded
        if(time.time() - start_time > config['timeout']):
            return 1

        # check if error is low enough to terminate
        eef_gt = get_eef_gt_tf(victor, camera, True)
        pos_error = np.linalg.norm(eef_gt[0:3, 3] -  target[0:3, 3])
        rot_error = np.linalg.norm(cv2.Rodrigues(eef_gt[0:3, 0:3].T @ target[0:3, 0:3])[0])
        #print(f"{pos_error} ---- {rot_error}")
        if(pos_error < config['max_pos_error'] and rot_error < config['max_rot_error']):
            return 0
        

        # get camera image
        rgb, depth, seg = camera.get_image(True)
        rgb_edit = rgb[..., [2, 1, 0]].copy()
        
        # do visual servo
        #pbvs.cheat(get_eef_gt_tf(victor, camera, False))
        ctrl, Twe = pbvs.do_pbvs(depth, target, victor.get_arm_jacobian('left'),
                                victor.get_jacobian_pinv('left'), 1/config['pbvs_hz'])
        victor.psuedoinv_ik_controller("left", ctrl)

        #print(ctrl)
        #_, _, _, _, _, _, lin, ang = p.getLinkState(victor.urdf, victor.eef_idx, 1)
        #print(lin + ang)
        #print(" ")

        # draw debug stuff
        if(config['vis']):
            erase_pos(pose_est_uids)
            pose_est_uids = draw_pose(Twe[0:3, 3], Twe[0:3, 0:3], mat=True) 
            #cv2.imshow("Camera", cv2.resize(rgb_edit, (1280 // 5, 800 // 5)))  

        result_dict["est_eef_pose"].append(Twe)
        result_dict["gt_eef_pose"].append(eef_gt)
        result_dict["joint_config"].append(victor.get_arm_joint_configs())

        
        # step simulation
        for _ in range(sim_steps_per_pbvs):
            p.stepSimulation()

        _, _, _, _, _, _, lin, ang = p.getLinkState(victor.urdf, victor.eef_idx, 1)
        print(np.linalg.norm(ctrl[0:3] - lin)/np.linalg.norm(ctrl[0:3]))
        print(np.linalg.norm(ctrl[3:6] - ang)/np.linalg.norm(ctrl[3:6]))
        print(" ")

def main():
    # Loads hjson config to do visual servoing with
    config_file = open('config.hjson', 'r')
    config_text = config_file.read()
    config = hjson.loads(config_text)

    result_dict = {}

    # Executes servoing for all the servo configs provided
    servo_configs = config['servo_configs']
    for i, servo_config in enumerate(servo_configs):
        client = p.connect(p.GUI)
        victor = Victor(servo_config["arm_states"])
        camera = PyBulletCamera(np.array(servo_config['camera_pos']), np.array(servo_config['camera_look']))
        target = create_target_tf(np.array(servo_config['target_pos']), np.array(servo_config['target_rot'])) 
        pbvs = ICPPBVS(camera, 1, 1, get_eef_gt_tf(victor, camera), 
            config['pbvs_settings']['max_joint_velo'], config['pbvs_settings']['seg_range'], debug=True) 
        result_dict[f"traj{i}"] = {"joint_config": [], "est_eef_pose": [], "gt_eef_pose": [],
             "camera_to_world" : np.linalg.inv(camera.get_view()), "victor_to_world": np.eye(4)}
        run_servoing(pbvs, camera, victor, target, config, result_dict[f'traj{i}'])
        p.disconnect()
    
    now = datetime.now()
    dirname = now.strftime("test-results/%Y%m%d-%H%M%S")
    if not os.path.exists(dirname):
        os.makedirs(dirname)


    result_file = open(f'{dirname}/result.pkl', 'wb')
    pickle.dump(result_dict, result_file)
    result_file.close()

    # Copy config to result
    shutil.copyfile('config.hjson', f'{dirname}/config.hjson')
        

main()