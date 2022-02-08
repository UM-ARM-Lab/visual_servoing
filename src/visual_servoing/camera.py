import numpy as np
import pybullet as p

import ros_numpy
import rospy

#############################################
# Interface for OpenCV style pinhole camera #
#############################################
from arc_utilities.listener import Listener
from sensor_msgs.msg import Image


class Camera:
    def __init__(self, camera_eye, camera_look, image_dim):
        self.camera_eye = camera_eye
        self.camera_look = camera_look
        self.image_dim = image_dim

    def get_intrinsics(self):
        """Return OpenCV style intrinsics 3x3"""
        pass

    def get_extrinsics(self):
        """Get homogenous extrisnic transform from world to camera Tcw 4x4"""
        pass

    def get_image(self):
        """Return RGB image and depth image"""
        raise NotImplementedError()

    def get_xyz(self, u, v, depth):
        """ Retrieve pointcloud point from depth and image pt """
        pass


#####################################
# PyBullet implementation of Camera #
#####################################
class PyBulletCamera(Camera):

    def __init__(self, camera_eye, camera_look, image_dim=(1280, 800)):
        super().__init__(camera_eye, camera_look, image_dim)

        # This is a column major order of the projection
        self.ogl_projection_matrix = p.computeProjectionMatrixFOV(
            fov=45.0,
            aspect=self.image_dim[0] / self.image_dim[1],
            nearVal=0.1,
            farVal=3.1,
        )

        # This is a column major order of the extrinsics
        self.ogl_view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.camera_eye,
            cameraTargetPosition=self.camera_look,
            cameraUpVector=[0, 0, 1])

    def get_intrinsics(self):
        proj_4x4 = np.array(self.ogl_projection_matrix).reshape(4, 4)
        proj_3x3 = np.array(self.ogl_projection_matrix).reshape(4, 4)[:3, :3]
        proj_3x3[0, 0] = proj_3x3[0, 0] * self.image_dim[0] / 2
        proj_3x3[1, 1] = proj_3x3[1, 1] * self.image_dim[1] / 2
        proj_3x3[0, 2] = self.image_dim[0] / 2
        proj_3x3[1, 2] = self.image_dim[1] / 2
        proj_3x3[2, 2] = 1
        return proj_3x3

    def get_extrinsics(self):
        # Note that the OpenGL [C2] and OpenCV [C/C1] camera systems are different
        # An additional transform is needed
        Tc2w = np.array(self.ogl_view_matrix).reshape([4, 4], order='F')
        Tc1c2 = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        Tcw = Tc1c2 @ Tc2w
        return Tcw

    def get_image(self):
        width, height, rgbImg, depthImg, segImg = p.getCameraImage(
            width=self.image_dim[0],
            height=self.image_dim[1],
            viewMatrix=self.ogl_view_matrix,
            projectionMatrix=self.ogl_projection_matrix,
            lightDirection=-(self.camera_look - self.camera_eye),
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        rgb_img = np.array(rgbImg)[:, :, :3]
        depth_img = np.array(depthImg)
        return rgb_img, depth_img

    def get_xyz(self, u, v, depth):
        # This code querys the depth buffer returned from the simulated camera and gets the <x,y,z> 
        # point of the AR tag in world space 
        # Source: https://stackoverflow.com/questions/59128880/getting-world-coordinates-from-opengl-depth-buffer
        # First we compose the projection and view matrix using OpenGL convention (note column major order)
        projectionMatrix = np.asarray(self.ogl_projection_matrix).reshape([4, 4], order='F')
        viewMatrix = np.asarray(self.ogl_view_matrix).reshape([4, 4], order='F')
        # We now invert the transform that was used to bring points into the normalized image coordinates
        # This maps us from normalized image coordinates back into world coordinates
        T = np.linalg.inv(np.matmul(projectionMatrix, viewMatrix))
        # The AR tag center in u,v pixel coordinates is converted into OpenGL normalized image coordinates 
        x = (2 * u - self.image_dim[0]) / self.image_dim[0]
        y = -(2 * v - self.image_dim[1]) / self.image_dim[1]
        # Z, a depth buffer reading from 0->1 is mapped from -1 to 1
        z = 2 * depth[v, u] - 1
        pix = np.asarray([x, y, z, 1])
        # Map points from normalized image coordinates into world
        pos = np.matmul(T, pix)
        # Divide by the w component of the homogenous vector to get cartesian vector
        pos /= pos[3]
        return pos[0:3]


############################################
# Realsense D455 ROS Camera implementation #
############################################
class RealsenseCamera(Camera):
    def __init__(self, camera_eye, camera_look, image_dim=(1280, 800)):
        super().__init__(camera_eye, camera_look, image_dim)
        self.depth = Listener("/camera/depth/image_rect_raw", Image)
        self.color = Listener("/camera/color/image_raw", Image)

    def get_intrinsics(self):
        """Return OpenCV style intrinsics 3x3"""
        pass

    def get_extrinsics(self):
        """Get homogenous extrisnic transform from world to camera Tcw 4x4"""
        pass

    def get_image(self):
        """Return RGB image and depth image"""
        color_img: Image = self.color.get()
        depth_img: Image = self.depth.get()

        color_np = ros_numpy.numpify(color_img)
        depth_np = ros_numpy.numpify(depth_img)
        return color_np

    def get_xyz(self, u, v, depth):
        """ Retrieve pointcloud point from depth and image pt """
        pass
