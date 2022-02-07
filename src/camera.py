import numpy as np
import pybullet as p

# An OpenCV style camera 
class Camera:
    def __init__(self, camera_eye,  camera_look, image_dim):
        self.camera_eye = camera_eye
        self.camera_look = camera_look
        self.image_dim = image_dim

    # Return OpenCV style intrinsics 3x3
    def get_projection():
        pass

    # Get transform from camera to world, aka pose of camera in world 4x4 Twc
    def get_view():
        pass 
    
    # Return RGB image and depth image 
    def get_image():
        pass 

class PyBulletCamera(Camera):
    def __init__(self, camera_eye,  camera_look, image_dim=(1280,800)):
        super().__init__(camera_eye, camera_look, image_dim)

        # This is a column major order of the projection
        self.ogl_projection_matrix = p.computeProjectionMatrixFOV(
            fov=45.0,
            aspect=self.image_width/self.image_height,
            nearVal=0.1,
            farVal=3.1,
        )

        # This is actually column major order of the transform from the world to the camera aka Tcw
        # If we parse it as row majro it is the transform from camera to world instead Twc
        self.ogl_view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.camera_eye,
            cameraTargetPosition=self.camera_look,
            cameraUpVector=[0, 0, 1]) 
    
    def get_image():
         width, height, rgbImg, depthImg, segImg = p.getCameraImage(
            width= self.image_width,
            height= self.image_height,
            viewMatrix=self.viewMatrix,
            projectionMatrix=self.projectionMatrix,
            lightDirection=-(self.camera_look-self.camera_eye)
        )
        rgb_img = np.array(rgbImg)[:, :, :3]
        depth_img = np.array(depthImg)
        return rgb_img, depth_img

class RealsenseCamera(Camera):
    def __init__(self, camera_eye,  camera_look, image_dim=(1280,800)):
        pass 