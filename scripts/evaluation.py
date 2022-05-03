import hjson
import time
import pybullet as p
import cv2
import numpy as np
from datetime import datetime
import pickle
import os
import pickle
import shutil

import open3d as o3d

from visual_servoing.arm_robot import ArmRobot
from visual_servoing.camera import Camera, PyBulletCamera
from visual_servoing.icp_pbvs import ICPPBVS
from visual_servoing.pbvs import PBVS
from visual_servoing.pbvs_loop import PybulletPBVSLoop
from visual_servoing.utils import *
from visual_servoing.victor import Victor


def create_target_tf(target_pos, target_rot):
    '''
    Creates a 4x4 homogenous TF given an input position and quaternion 
    '''
    H = np.eye(4)
    H[0:3, 0:3] = np.array(p.getMatrixFromQuaternion(target_rot)).reshape(3, 3)
    H[0:3, 3] = target_pos
    return H


def get_eef_gt_tf(victor, camera, world_relative=False):
    '''
    Gets the ground truth position of the eef link
    '''
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
    if (world_relative):
        return Twe
    else:
        Tce = Tcw @ Twe
        return Tce


def image_augmentation(numpy_depth):
    '''
    Adds depth noise (TODO needs work!!!!)
    '''
    # Get to btween 0 and 255
    # depth = numpy_depth + np.min(numpy_depth)
    synthetic_data = True
    if synthetic_data:
        # depth /= np.max(depth)
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
    background_idx = np.where(np.logical_or(numpy_depth > 2 / 2.6, numpy_depth == 0))
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

class EvalPBVSLoop(PybulletPBVSLoop):
    def __init__(self,
                 pbvs: PBVS,
                 camera: Camera,
                 robot: ArmRobot,
                 side: str,
                 pbvs_hz: float,
                 sim_hz: float,
                 config,
                 result_dict):
        super().__init__(pbvs, camera, robot, side, pbvs_hz, sim_hz, config)
        self.result_dict = result_dict
        self.pose_est_uids = None
        self.target_uids = None

    def on_check_is_done(self, is_timed_out, target_reached):
        if(self.is_done(is_timed_out, target_reached)):
            self.result_dict['finished'] = True

    def get_camera_image(self):
        rgb, depth = super().get_camera_image()
        true_depth = self.camera.get_true_depth(depth).reshape(depth.shape)
        if (self.config['use_depth_noise']):
            min_val = np.min(true_depth)
            max_val = np.max(true_depth)
            rng = max_val - min_val
            # noisy_depth = image_augmentation((true_depth - min_val)/rng)
            noisy_depth = image_augmentation(true_depth)
            # noisy_depth = (noisy_depth * rng + min_val)
        else:
            noisy_depth = true_depth
        noisy_depth_buffer = self.camera.get_depth_buffer(noisy_depth).reshape(depth.shape)

        if self.config['vis']:
            cv2.imshow("Camera", cv2.resize(rgb, (1280 // 5, 800 // 5)))
            cv2.waitKey(1)

        return rgb, noisy_depth_buffer

    def on_after_step_pbvs(self, Twe):
        super().on_after_step_pbvs(Twe)

        # draw debug stuff
        if self.config['vis']:
            erase_pos(self.pose_est_uids)
            self.pose_est_uids = draw_pose(Twe[0:3, 3], Twe[0:3, 0:3], mat=True)

        # populate results
        self.result_dict["seg_cloud"].append(np.asarray(self.pbvs.pcl.points))
        self.result_dict["est_eef_pose"].append(Twe)
        eef_gt = get_eef_gt_tf(self.robot, self.camera, True)
        self.result_dict["gt_eef_pose"].append(eef_gt)
        self.result_dict["joint_config"].append(self.robot.get_arm_joint_configs())


def run_servoing(pbvs, camera, victor, target, config, result_dict):
    loop = EvalPBVSLoop(pbvs, camera, victor, "left", config['pbvs_hz'], config['sim_hz'], config, result_dict) 
    if (config['vis']):
        cam_inv = np.linalg.inv(camera.get_view())
        draw_pose(target[0:3, 3], target[0:3, 0:3], mat=True)
        draw_pose(cam_inv[0:3, 3], cam_inv[0:3, 0:3], mat=True)
    loop.run(target)


def main():
    # Loads hjson config to do visual servoing with
    config_file = open('config.hjson', 'r')
    config_text = config_file.read()
    config = hjson.loads(config_text)

    result_dict = {"traj": []}

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    use_aruco = True if config['eef_perception_type'] == "aruco" else False

    # Executes servoing for all the servo configs provided
    servo_configs = config['servo_configs']
    for i, servo_config in enumerate(servo_configs):
        # Create objects for visual servoing
        client = p.connect(p.GUI)
        victor = Victor(servo_config, config['twist_execution_noise'])
        camera = PyBulletCamera(np.array(servo_config['camera_pos']), np.array(servo_config['camera_look']), p.ER_BULLET_HARDWARE_OPENGL)
        target = create_target_tf(np.array(servo_config['target_pos']), np.array(servo_config['target_rot'])) 
        pbvs = None

        if(use_aruco):
            tag_ids, tag_geometry = victor.get_tag_geometry()
            pbvs = MarkerPBVS(camera, 1, 1,
                config['pbvs_settings']['max_joint_velo'], 
                get_eef_gt_tf(victor, camera),
                tag_ids,
                tag_geometry,
                None,
                None)
        else:
            pbvs = ICPPBVS(camera, 1, 1,  
                config['pbvs_settings']['max_joint_velo'], get_eef_gt_tf(victor, camera), config['pbvs_settings']['seg_range'], debug=True, vis=vis) 
        
        # Create entry for this trajectory in result
        result_dict[f"traj"].append(
            {
                "joint_config":    [],
                "est_eef_pose":    [],
                "gt_eef_pose":     [],
                "seg_cloud":       [],
                "camera_to_world": np.linalg.inv(camera.get_view()),
                "victor_to_world": np.eye(4),
                "target_pose":     target,
                "finished":        False
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