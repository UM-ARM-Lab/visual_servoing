import cv2
import numpy as np
import open3d as o3d
from visual_servoing.camera import Camera
import time


class ICPPBVS:
    # camera: (Instance of a camera following the Camera interface)
    # k_v: (Scaling constant for linear velocity control)
    # k_omega: (Scaling constant for angular velocity control) 
    # start_eef_pose: (Starting pose of end effector in camera frame (Tce))
    def __init__(self, camera, k_v, k_omega, model, start_eef_pose, max_joint_velo=0):
        self.k_v = k_v
        self.k_omega = k_omega
        self.camera = camera
        self.model = o3d.geometry.PointCloud()
        self.model.points = o3d.utility.Vector3dVector(model)
        self.model.paint_uniform_color([0, 0.651, 0.929])

        self.prev_pose = start_eef_pose
        self.prev_twist = np.zeros(6)
        self.prev_time = time.time()
        self.max_joint_velo = max_joint_velo

    def get_segmented_pcl_sim(self):
        pass

    # Will only work in sim
    # get EEF state estimate relative to camera
    def get_eef_state_estimate(self, depth, seg):
        # Compute segmented point cloud of eef from depth/seg img
        pcl_raw = self.camera.get_pointcloud(depth)
        pcl_raw = self.camera.segmented_pointcloud(pcl_raw, (np.arange(16, 30) + 1) << 24, seg)
        pcl = o3d.geometry.PointCloud() 
        pcl.points = o3d.utility.Vector3dVector(pcl_raw.T)
        pcl.paint_uniform_color([1, 0.706, 0])

        # Run ICP from previous est 
        # we want Tcl, transform of eef link (l) in camera frame, but we do ICP the other way so we estimate Tlc instead
        reg = o3d.pipelines.registration.registration_icp(
            pcl, self.model, 0.1, self.prev_pose, o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        Tcl = np.linalg.inv(reg.transformation)
        return Tcl
        

    # Executes an iteration of PBVS control and returns a twist command
    # Two: (Pose of target object w.r.t world)
    # Returns the twist command [v, omega] and pose of end effector in world
    def do_pbvs(self, depth, seg, Two, jac, jac_inv, Tle=np.eye(4), debug=True):
        Tcw = self.camera.get_view()
        Tcl = self.get_eef_state_estimate(depth, seg) 

        Twc = np.linalg.inv(Tcw)
        ctrl = np.zeros(6)
        Twe = Twc @ Tcl @ Tle 

        ctrl = self.get_control(Twe, Two)
        
        # compute the joint velocities using jac_inv and PBVS twist cmd
        q_prime = jac_inv @ ctrl 
        # rescale joint velocities to self.max_joint_velo if the largest exceeds the limit 
        if(np.max(np.abs(q_prime)) > self.max_joint_velo):
            q_prime /= np.max(np.abs(q_prime))
            q_prime *= self.max_joint_velo
        # compute the actual end effector velocity given the limit
        ctrl = jac @ q_prime

        # store results
        self.prev_time = time.time()
        self.prev_twist = ctrl
        return ctrl, Twe
    
    ####################
    # PBVS control law #
    #################### 

    def get_v(self, object_pos, eef_pos):
        return (object_pos - eef_pos) * self.k_v

    def get_omega(self, Rwa, Rwo):
        Rao = np.matmul(Rwa, Rwo.T).T
        Rao_rod, _ = cv2.Rodrigues(Rao)
        return Rao_rod * self.k_omega

    def get_control(self, Twe, Two):
        ctrl = np.zeros(6)
        ctrl[0:3] = self.get_v(Two[0:3, 3], Twe[0:3, 3])
        ctrl[3:6] = np.squeeze(self.get_omega(Twe[0:3, 0:3], Two[0:3, 0:3]))
        return ctrl