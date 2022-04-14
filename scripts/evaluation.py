import hjson
import time
import pybullet as p
import cv2
from visual_servoing.camera import Camera, PyBulletCamera
import numpy as np
from visual_servoing.icp_pbvs import ICPPBVS
from visual_servoing.victor import Victor
from visual_servoing.utils import *
from datetime import datetime
import pickle
import os
import shutil
import open3d as o3d
from visual_servoing.pbvs_loop import PBVSLoop
from visual_servoing.arm_robot import ArmRobot
from visual_servoing.pbvs import PBVS

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

def image_augmentation(numpy_depth):
    # Get to btween 0 and 255
    #depth = numpy_depth + np.min(numpy_depth)
    synthetic_data = True
    if synthetic_data:
        #depth /= np.max(depth)
        depth = 255.0 * numpy_depth
        depth = depth.astype(np.uint8)
        edges = cv2.Canny(depth, 100, 200)
        ex, ey = np.where(edges == 255)
        num_edge = len(ex)
        min_edge = int(0.5 * num_edge)
        num_disturbances = np.random.randint(min_edge, num_edge)
        perm = np.random.permutation(np.arange(0, num_edge))
        pixels_to_disturb = perm[:num_disturbances]
        idx = (ex[pixels_to_disturb], ey[pixels_to_disturb])
        eones = np.ones(depth.shape)
        eones[idx] = 0.0

        # Add edge noise
        numpy_depth *= eones

    # Get area where depth beyond threshold (background)
    background_idx = np.where(np.logical_or(numpy_depth > 2/2.6, numpy_depth == 0))
    bones = np.ones(numpy_depth.shape)
    bones[background_idx] = 0.0

    # Salt and pepper noise to background
    bx, by = background_idx
    max_disturbances = int(0.2 * len(bx))
    min_disturbances = int(0.1 * len(bx))
    num_snp = np.random.randint(min_disturbances, max_disturbances)
    b_permutation = np.random.permutation(len(bx))
    snp_idx = (bx[b_permutation[:num_snp]], by[b_permutation[:num_snp]])
    snp_noise_depth = 1.0 + 0.1 * np.random.randn(num_snp)

    # Subtract background
    numpy_depth *= bones

    # Add snp noise
    if synthetic_data:
        numpy_depth[snp_idx] = snp_noise_depth

    numpy_depth += 0.05 * np.random.normal(size=numpy_depth.shape)
    return numpy_depth 

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
        if(pos_error < config['max_pos_error'] and rot_error < config['max_rot_error']):
            result_dict['finished'] = True
            return 0
        
        # get camera image
        rgb, depth, seg = camera.get_image(True)
        rgb_edit = rgb[..., [2, 1, 0]].copy()
        true_depth = camera.get_true_depth(depth).reshape(depth.shape)
        if(config['use_depth_noise']):
            min_val = np.min(true_depth)
            max_val = np.max(true_depth)
            rng = max_val - min_val
            noisy_depth = image_augmentation((true_depth - min_val)/rng)
            noisy_depth = image_augmentation(true_depth)
            noisy_depth = (noisy_depth * rng + min_val)
        else:
            noisy_depth = true_depth
        noisy_depth_buffer = camera.get_depth_buffer(noisy_depth).reshape(depth.shape)
        #pcl_raw = camera.get_pointcloud(noisy_depth_buffer)
        #pcl = o3d.geometry.PointCloud() 
        #pcl.points = o3d.utility.Vector3dVector(pcl_raw.T)
        #pcl.colors = o3d.utility.Vector3dVector(rgb_edit.reshape(-1, 3)/255.0)
        #o3d.visualization.draw_geometries([pcl])
        
        # do visual servo
        ctrl, Twe = pbvs.do_pbvs(rgb, noisy_depth_buffer, target, np.eye(4), victor.get_arm_jacobian('left'),
                                victor.get_jacobian_pinv('left'), 1/config['pbvs_hz'])
        # noise injection
        ctrl[0:3] += np.random.normal(scale=config['twist_execution_noise_linear'], size=(3))
        ctrl[3:6] += np.random.normal(scale=config['twist_execution_noise_angular'], size=(3))
        victor.psuedoinv_ik_controller("left", ctrl)

        # draw debug stuff
        if(config['vis']):
            erase_pos(pose_est_uids)
            pose_est_uids = draw_pose(Twe[0:3, 3], Twe[0:3, 0:3], mat=True) 
            cv2.imshow("Camera", cv2.resize(rgb_edit, (1280 // 5, 800 // 5)))  
            cv2.waitKey(1)
        
        # step simulation
        for _ in range(sim_steps_per_pbvs):
            p.stepSimulation()

        # populate results
        result_dict["seg_cloud"].append(np.asarray(pbvs.pcl.points))
        result_dict["est_eef_pose"].append(Twe)
        result_dict["gt_eef_pose"].append(eef_gt)
        result_dict["joint_config"].append(victor.get_arm_joint_configs())

def main():
    # Loads hjson config to do visual servoing with
    config_file = open('config.hjson', 'r')
    config_text = config_file.read()
    config = hjson.loads(config_text)

    result_dict = {"traj" : []}

    # Executes servoing for all the servo configs provided
    servo_configs = config['servo_configs']
    for i, servo_config in enumerate(servo_configs):
        # Create objects for visual servoing
        client = p.connect(p.GUI)
        victor = Victor(servo_config["arm_states"])
        camera = PyBulletCamera(np.array(servo_config['camera_pos']), np.array(servo_config['camera_look']))
        target = create_target_tf(np.array(servo_config['target_pos']), np.array(servo_config['target_rot'])) 
        pbvs = ICPPBVS(camera, 1, 1,  
            config['pbvs_settings']['max_joint_velo'], get_eef_gt_tf(victor, camera), config['pbvs_settings']['seg_range'], debug=True) 
        
        # Create entry for this trajectory in result
        result_dict[f"traj"].append(
            {
                "joint_config": [], 
                "est_eef_pose": [],
                "gt_eef_pose": [],
                "seg_cloud": [],
                "camera_to_world" : np.linalg.inv(camera.get_view()), 
                "victor_to_world": np.eye(4),
                "target_pose" : target,
                "finished" : False
            }
        )
        
        # Do visual servoing and record results
        run_servoing(pbvs, camera, victor, target, config, result_dict[f'traj'][-1])

        # Destroy GUI when done
        p.disconnect()
    
    # Create folder for storing result
    now = datetime.now()
    dirname = now.strftime("test-results/%Y%m%d-%H%M%S")
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # Dump result pkl into folder
    result_file = open(f'{dirname}/result.pkl', 'wb')
    pickle.dump(result_dict, result_file)
    result_file.close()

    # Copy config to result folder
    shutil.copyfile('config.hjson', f'{dirname}/config.hjson')
        

if __name__ == "__main__":
    main()
'''

class EvalPBVSLoop(PBVSLoop):
    def __init__(self, pbvs: PBVS, camera : Camera, robot : ArmRobot, side : str, pbvs_hz : float, sim_hz : float, config, result_dict):
        self.config = config
        self.result_dict = result_dict

    def terminating_condition(self):
        # check if timeout exceeded
        if(time.time() - self.start_time > self.config['timeout']):
            return True

        # check if error is low enough to terminate
        eef_gt = get_eef_gt_tf(self.robot, self.camera, True)
        pos_error = np.linalg.norm(eef_gt[0:3, 3] -  self.target[0:3, 3])
        rot_error = np.linalg.norm(cv2.Rodrigues(eef_gt[0:3, 0:3].T @ self.target[0:3, 0:3])[0])
        if(pos_error < self.config['max_pos_error'] and rot_error < self.config['max_rot_error']):
            self.result_dict['finished'] = True
            return True


    def on_before_step_pbvs(self):
        rgb, depth, seg = self.camera.get_image(True)
        rgb_edit = rgb[..., [2, 1, 0]].copy()
        true_depth = self.camera.get_true_depth(depth).reshape(depth.shape)
        if(self.config['use_depth_noise']):
            min_val = np.min(true_depth)
            max_val = np.max(true_depth)
            rng = max_val - min_val
            #noisy_depth = image_augmentation((true_depth - min_val)/rng)
            noisy_depth = image_augmentation(true_depth)
            #noisy_depth = (noisy_depth * rng + min_val)
        else:
            noisy_depth = true_depth
        noisy_depth_buffer = self.camera.get_depth_buffer(noisy_depth).reshape(depth.shape)

    def on_after_step_pbvs(self):
        # noise injection
        self.ctrl[0:3] += np.random.normal(scale=self.config['twist_execution_noise_linear'], size=(3))
        self.ctrl[3:6] += np.random.normal(scale=self.config['twist_execution_noise_angular'], size=(3))
        self.robot.psuedoinv_ik_controller("left", self.ctrl)

        # draw debug stuff
        if(self.config['vis']):
            erase_pos(pose_est_uids)
            pose_est_uids = draw_pose(Twe[0:3, 3], Twe[0:3, 0:3], mat=True) 
            cv2.imshow("Camera", cv2.resize(rgb_edit, (1280 // 5, 800 // 5)))  
            cv2.waitKey(1)
            '''