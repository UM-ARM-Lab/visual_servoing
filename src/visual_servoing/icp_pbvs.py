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
from visual_servoing.pbvs import PBVS

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


class ICPPBVS(PBVS):
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
    def __init__(self, camera : Camera, k_v : float, k_omega : float, max_joint_velo : float, start_eef_pose, seg_range=0.04, debug=True, vis=None):
        super().__init__( camera, k_v, k_omega, max_joint_velo, debug)
        self.seg_range = seg_range

        self.model_sdf = pkl.load(open("points_and_sdf.pkl", "rb"))
        self.model = o3d.geometry.PointCloud()
        self.model_raw = np.array(self.model_sdf['points'])
        self.model.points = o3d.utility.Vector3dVector(self.model_raw)
        self.model.paint_uniform_color([0, 0.651, 0.929])

        self.prev_pose = start_eef_pose
        self.prev_twist = np.zeros(6)
        self.max_joint_velo = max_joint_velo

        self.pcl = o3d.geometry.PointCloud()

        if(debug):
            self.vis = vis
            self.vis.add_geometry(self.pcl)
            self.vis.add_geometry(self.model)

        self.pose_predict_uids = None
        self.prev_pose_predict_uids = None
        self.cheat_pose_uids = None
        self.debug = debug
        self.cheat_pose = None
    
    def __del__(self):
        if(self.debug):
            self.vis.remove_geometry(self.pcl)
            self.vis.remove_geometry(self.model)
        print('destroyed')

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

    # segment point cloud via radius around predicted pose in camera frame
    def radius_segment(self,pcl, pose_predict):
        c = (pose_predict[0:3, 3]).reshape(-1, 1)
        dist = np.sqrt(np.sum((pcl - c)**2, axis=0))
        return pcl[:, dist < self.seg_range]

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
        action_trans = np.hstack((action[0:3], 0))
        action_rot = np.hstack((action[3:6], 0))
        view = self.camera.get_view()
        action_trans = view @ action_trans
        action_rot = view @ action_rot
        action_tf = SE3(np.hstack((action_trans[0:3],action_rot[0:3])))
        pose_predict = self.prev_pose#action_tf @ self.prev_pose#np.linalg.inv(action_tf) @ self.prev_pose #action_tf @ self.prev_pose  
        pose_predict[0:3, 3] = pose_predict[0:3, 3] + action_trans[0:3]
        pose_predict[0:3, 0:3] = action_tf[0:3, 0:3] @ pose_predict[0:3, 0:3]

        # transform point cloud into eef link frame using predicted pose
        pcl_raw_linkfrm = np.linalg.inv(pose_predict)@np.vstack((pcl_raw,np.ones( (1, pcl_raw.shape[1] ) )))
        pcl_raw_linkfrm = (pcl_raw_linkfrm.T)[:, 0:3]
     
        # segment the point cloud and convert back to camera frame 
        pcl_seg = self.segment(pcl_raw_linkfrm, self.model_sdf['sdf'], self.model_sdf['origin_point'], self.model_sdf['res'], self.seg_range)
        pcl_raw = pose_predict@np.hstack((pcl_seg,np.ones( (pcl_seg.shape[0], 1) ))).T
        pcl_raw = pcl_raw[ 0:3, :]
        #pcl_raw = self.radius_segment(pcl_raw, pose_predict) 

        # store the segmented point cloud from the camera as a class member for vis
        self.pcl.points = o3d.utility.Vector3dVector(pcl_raw.T)
        self.pcl.paint_uniform_color([1, 0.706, 0])
        
        # run ICP: note we want Tcl, transform of eef link (l) in camera frame, but we do ICP with
        # segmented camera cloud as source and the model in link frame as target, so we estimate Tlc instead
        reg = o3d.pipelines.registration.registration_icp(
            self.pcl, self.model, 0.1, np.linalg.inv(pose_predict), o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        # for visualization purposes, PCL can be translated into link frame
        self.pcl.transform(reg.transformation)

        # compute the thing we care about, the transform of the end effector in camera frame
        Tcl = np.linalg.inv(reg.transformation)

        # visualize 
        #o3d.visualization.draw_geometries([self.pcl, self.model])
        if(self.debug):
            self.draw_registration_result()
        
        # store eef pose in camera frame
        self.prev_pose = Tcl
        return Tcl
        

    def do_pbvs(self, rgb, depth, Two, Tle, jac, jac_inv, dt):
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
        
        ctrl = self.limit_twist(jac, jac_inv, ctrl)

        # store results and return eef twist command + pose estimate
        self.prev_twist = ctrl
        return ctrl, Twe