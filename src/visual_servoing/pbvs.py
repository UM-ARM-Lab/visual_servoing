from xmlrpc.client import Boolean
from visual_servoing.camera import Camera
import numpy as np
import cv2
import pybullet

class PBVS:
    def __init__(self, camera : Camera, k_v : float, k_omega : float, max_joint_velo : float, debug : bool=True):
        """
        Args:
            camera: instance of a camera following generic camera interface, must have OpenGL and OpenCV matricies defined
            k_v: scaling constant for linear velocity control
            k_omega: scaling constant for angular velocity control
            start_eef_pose: the starting pose of the end effector link in camera frame (Tcl)
            max_joint_velo: maximum joint velocity 
            seg_range: segmentation range in meters
            debug: do debugging visualizations or not
        ​
        """
        self.k_v = k_v
        self.k_omega = k_omega
        self.camera = camera
        self.max_joint_velo = max_joint_velo
        self.debug = debug

    def do_pbvs(self, rgb, depth, Two, Tle, jac, jac_inv, dt):
        """
        Does one iteration of PBVS and returns a twist

        Args:
            rgb: current camera image
            depth: current depth image
            Two: transform from world to target object
            Tle: transform from the link we are estimating the pose of to the end effector 
            jac: jacobian of the end effector
            jac_inv: inverse jacobian of the end effector

        Returns:
            twist: end effector linear and angular velocity

        """
        raise NotImplementedError()

    def limit_twist(self, jac, jac_inv, ctrl):
        # compute the joint velocities using jac_inv and PBVS twist cmd
        q_prime = jac_inv @ ctrl
        # rescale joint velocities to self.max_joint_velo if the largest exceeds the limit 
        if(np.max(np.abs(q_prime)) > self.max_joint_velo):
            q_prime /= np.max(np.abs(q_prime))
            q_prime *= self.max_joint_velo
        # compute the actual end effector velocity given the limit
        ctrl = jac @ q_prime
        return ctrl

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

    def get_omega(self, Rwe, Rwo):
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
        Reo = np.matmul(Rwe, Rwo.T).T
        Reo_rod, _ = cv2.Rodrigues(Reo)
        return Reo_rod * self.k_omega

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