from tkinter import W
import cv2
import numpy as np
import open3d as o3d
from visual_servoing.camera import Camera
import pickle as pkl
import time
import copy
from scipy.spatial.distance import cdist
import tensorflow as tf
from visual_servoing.utils import draw_pose, erase_pos


def SE3(se3):
    """
    Switches between SE3 representations of the same twist command
    ​
    Args:
        se3: either a [vx,vy,vz,omegax,omegay,omegeaz] or homogenous tf
    ​
    Returns:
        se3: the alternative representation of the input
    ​
    """
    if(se3.shape[0] == 6):
        rot_3x3, _ = cv2.Rodrigues(se3[3:6]) 
        T = np.zeros((4,4)) 
        T[0:3, 0:3] = rot_3x3 
        T[0:3, 3] = se3[0:3] 
        T[3, 3] = 1
        return T
    else:
        rvec, _ = cv2.Rodrigues(se3[0:3, 0:3])
        tvec = se3[0:3, 3].squeeze() 
        return np.hstack((tvec, rvec.squeeze()))


class ICPPBVS:
    """
    ​
    Args:
        camera: instance of a camera following generic camera interface, must have OpenGL intrinsics 
        k_v: scaling constant for linear velocity control
        k_omega: scaling constant for angular velocity control
        start_eef_pose: the starting pose of the end effector link in camera frame (Tcl)
        max_joint_velo: maximum joint velocity 
        seg_range: segmentation range in meters
        debug: do debugging visualizations or not
    ​
    """
    def __init__(self, camera, k_v, k_omega, start_eef_pose, max_joint_velo=0, seg_range=0.04, debug=False):
        self.seg_range = seg_range
        self.k_v = k_v
        self.k_omega = k_omega
        self.camera = camera

        self.model_sdf = pkl.load(open("points_and_sdf.pkl", "rb"))
        self.model = o3d.geometry.PointCloud()
        self.model_raw = np.array(self.model_sdf['points'])
        self.model.points = o3d.utility.Vector3dVector(self.model_raw)
        self.model.paint_uniform_color([0, 0.651, 0.929])

        self.seg_range = seg_range

        self.prev_pose = start_eef_pose
        self.prev_twist = np.zeros(6)
        self.prev_time = time.time()
        self.max_joint_velo = max_joint_velo

        self.pcl = o3d.geometry.PointCloud()
        
        if(debug):
            pass
            #self.vis = o3d.visualization.Visualizer()
            #self.vis.create_window()
            #self.vis.add_geometry(self.pcl)
            #self.vis.add_geometry(self.model)

        self.pose_predict_uids = None
        self.cheat_pose_uids = None
        self.debug = debug
        self.cheat_pose = None
    
    # REMOVE ME
    def cheat(self, gt_pose):
        self.cheat_pose = gt_pose

    def draw_registration_result(self):
        #o3d.visualization.draw_geometries([self.pcl, self.model])
        self.vis.update_geometry(self.pcl)
        self.vis.update_geometry(self.model)
        self.vis.poll_events()
        self.vis.update_renderer()

    def round_to_res(self, x, res):
        # helps with stupid numerics issues
        return tf.cast(tf.round(x / res), tf.int64)

    def batch_point_to_idx(self, points, res, origin_point):
        """
    ​
        Args:
            points: [b,3] points in a frame, call it world
            res: [b] meters
            origin_point: [b,3] the position [x,y,z] of the center of the voxel (0,0,0) in the same frame as points
    ​
        Returns:
    ​
        """
        return self.round_to_res((points - origin_point), tf.expand_dims(res, -1))

    def segment(self, pc, sdf, origin_point, res, threshold):
        """
    ​
        Args:
            pc: [n, 3], as set of n (x,y,z) points in the same frame as the voxel grid
            sdf: [h, w, c], signed distance field
            origin_point: [3], the (x,y,z) position of voxel [0,0,0]
            res: scalar, size of one voxel in meters
            threshold: the distance threshold determining what's segmented
    ​
        Returns:
            [m, 3] the segmented points
    ​
        """
        pc = tf.convert_to_tensor(pc, dtype=tf.float32)
        indices = self.batch_point_to_idx(pc, res, origin_point)
        in_bounds = tf.logical_not(tf.logical_or(tf.reduce_any(indices <= 0, -1), tf.reduce_any(indices >= sdf.shape, -1)))
        in_bounds_indices = tf.boolean_mask(indices, in_bounds, axis=0)
        in_bounds_pc = tf.boolean_mask(pc, in_bounds, axis=0)
        distances = tf.gather_nd(sdf, in_bounds_indices)
        close = distances < threshold
        segmented_points = tf.boolean_mask(in_bounds_pc, close, axis=0)
        return segmented_points

    def get_eef_state_estimate(self, depth, dt):
        """
        get eef state estimate relative to camera

        Args:
            depth: [h, w], depth image to create point cloud with 
            dt: the time since the last command was executed, needed for state prediction
    ​
        Returns:
            Tcl: homogenous transform from camera to eef link
    ​
        """
        pcl_raw = self.camera.get_pointcloud(depth)

        # compute predicted pose using previous pose estimate, previously commanded twist and ellapsed time
        action = self.prev_twist * dt
        action_tf = SE3(action)
        pose_predict = self.prev_pose @ np.linalg.inv(action_tf)
        if(self.debug):
            cheat_pose_vis = np.linalg.inv(self.camera.get_view()) @ self.cheat_pose  
            erase_pos(self.cheat_pose_uids)
            self.cheat_pose_uids = draw_pose(cheat_pose_vis[0:3, 3], cheat_pose_vis[0:3, 0:3], mat=True)
            pose_predict_vis = np.linalg.inv(self.camera.get_view()) @ pose_predict  
            erase_pos(self.pose_predict_uids)
            self.pose_predict_uids = draw_pose(pose_predict_vis[0:3, 3], pose_predict_vis[0:3, 0:3], axis_len=0.2, alpha=0.5, mat=True)
            predict_dist = np.linalg.norm(self.prev_pose[0:3, 3] - pose_predict[0:3, 3])
            actual_dist = np.linalg.norm(self.prev_pose[0:3, 3] - self.cheat_pose[0:3, 3])
            print(predict_dist/actual_dist)

        # transform point cloud into eef link frame using predicted pose
        pcl_raw_linkfrm = np.linalg.inv(pose_predict)@np.vstack((pcl_raw,np.ones( (1, pcl_raw.shape[1] ) )))
        pcl_raw_linkfrm = (pcl_raw_linkfrm.T)[:, 0:3]
     
        # segment the point cloud and convert back to camera frame 
        pcl_seg = self.segment(pcl_raw_linkfrm, self.model_sdf['sdf'], self.model_sdf['origin_point'], self.model_sdf['res'], self.seg_range)
        pcl_raw = pose_predict@np.hstack((pcl_seg,np.ones( (pcl_seg.shape[0], 1) ))).T
        pcl_raw = pcl_raw[ 0:3, :]

        # store the segmented point cloud from the camera as a class member for vis
        self.pcl.points = o3d.utility.Vector3dVector(pcl_raw.T)
        self.pcl.paint_uniform_color([1, 0.706, 0])
        
        # run ICP: note we want Tcl, transform of eef link (l) in camera frame, but we do ICP with
        # segmented camera cloud as source and the model in link frame as target, so we estimate Tlc instead
        reg = o3d.pipelines.registration.registration_icp(
            self.pcl, self.model, 0.5, np.linalg.inv(self.prev_pose), o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        # for visualization purposes, PCL can be translated into link frame
        self.pcl.transform(reg.transformation)

        # compute the thing we care about, the transform of the end effector in camera frame
        Tcl = np.linalg.inv(reg.transformation)
        Tcl = self.cheat_pose#pose_predict

        # visualize 
        #if(self.debug):
        #    self.draw_registration_result()
        
        # store eef pose in camera frame
        self.prev_pose = Tcl
        return Tcl
        

    def do_pbvs(self, depth, Two, jac, jac_inv, dt=1/240, Tle=np.eye(4), debug=True):
        """
        Computes a twist command for one iteration of PBVS control
    ​
        Args:
            depth: [h, w], depth image to create point cloud with 
            Two: homogenous transform of target object in world
            jac: eef jacobian
            jac_inv: eef jacobian inverse, probably will be jac psuedoinverse or transpose
            dt: the time since the last command was executed, needed for state prediction
            Tle: transform from end effector link to another point to servo relative to 
            debug: do debug visualizations or not
    ​
        Returns:
            [6,] twist command that is [vx, vy, vz, omegax,omegay,omegaz]
    ​
        """
        # get the end effector state estimate relative to world frame using camera extrinsics + intrinsics
        Tcw = self.camera.get_view()
        Tcl = self.get_eef_state_estimate(depth, dt) 
        Twc = np.linalg.inv(Tcw)
        ctrl = np.zeros(6)
        Twe = Twc @ Tcl @ Tle 

        # compute control using state estimate + target position
        ctrl = self.get_control(Twe, Two)
        
        # compute the joint velocities this eef velocity would result in 
        q_prime = jac_inv @ ctrl 

        # rescale joint velocities to self.max_joint_velo if the largest exceeds the limit 
        if(np.max(np.abs(q_prime)) > self.max_joint_velo):
            q_prime /= np.max(np.abs(q_prime))
            q_prime *= self.max_joint_velo

        # compute the actual end effector velocity given the limit
        ctrl = jac @ q_prime

        # store results and return eef twist command + pose estimate
        self.prev_time = time.time()
        self.prev_twist = ctrl
        return ctrl, Twe

    def get_v(self, object_pos, eef_pos):
        """
        Get PBVS linear velocity command 
    ​
        Args:
            object_pos: position of the target object in PBVS world
            eef_pos: position of end effector in PBVS world
    ​
        Returns:
            v: end effector linear velocity in arbitrary unit
    ​
        """
        return (object_pos - eef_pos) * self.k_v

    def get_omega(self, Rwa, Rwo):
        """
        Get PBVS angular velocity command 
    ​
        Args:
            Rwa: rotation of end effector in PBVS world
            Rwo: rotation of the target object in PBVS world
    ​
        Returns:
            omega: end effector angular velocity rodrigues vector with arbitrary scale
    ​
        """
        Rao = np.matmul(Rwa, Rwo.T).T
        Rao_rod, _ = cv2.Rodrigues(Rao)
        return Rao_rod * self.k_omega

    def get_control(self, Twe, Two):
        """
        Get PBVS twist command
    ​
        Args:
            Twe: pose of end effector in PBVS world
            Two: pose of the target object in PBVS world
    ​
        Returns:
            ctrl: end effector twist command
    ​
        """
        ctrl = np.zeros(6)
        ctrl[0:3] = self.get_v(Two[0:3, 3], Twe[0:3, 3])
        ctrl[3:6] = np.squeeze(self.get_omega(Twe[0:3, 0:3], Two[0:3, 0:3]))
        return ctrl
