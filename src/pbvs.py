import cv2
import pybullet as p
import numpy as np

class PBVS:

    def __init__(self):
        self.image_width = 2000
        self.image_height = 2000
         #-1.9
        self.camera_eye = np.array([-1.0, 0.5, 0.5])
        self.target_pos = np.array([0, 0.5, 0])

        self.camera_eye = np.array([0.0, 0.0, 0.0])
        self.target_pos = np.array([0, 3.0, 0])


        self.projectionMatrix = p.computeProjectionMatrixFOV(
            fov=45.0,
            aspect=self.image_width/self.image_height,
            nearVal=0.1,
            farVal=3.1,
            )
        self.viewMatrix = p.computeViewMatrix(
            cameraEyePosition=self.camera_eye,
            cameraTargetPosition=self.target_pos,
            cameraUpVector=[0, 0, 1])

        

    def get_static_camera_img(self): 
        # TODO replace magic numbers with class params
        
        width, height, rgbImg, depthImg, segImg = p.getCameraImage(
            width= self.image_width,
            height= self.image_height,
            viewMatrix=self.viewMatrix,
            projectionMatrix=self.projectionMatrix,
            lightDirection=[0,-1,0]
            )
        rgb_img = np.array(rgbImg)[:, :, :3]
        depth_img = np.array(depthImg)
        return rgb_img, depth_img
    
    def get_intrinsics(self):
        proj_4x4 = np.array(self.projectionMatrix).reshape(4,4)
        proj_3x3 = np.array(self.projectionMatrix).reshape(4,4)[:3, :3]
        
        proj_3x3[0, 0] = proj_3x3[0, 0] * self.image_width/2
        proj_3x3[1, 1] = proj_3x3[1, 1] * self.image_height/2
        proj_3x3[0, 2] = self.image_width/2
        proj_3x3[1, 2] = self.image_height/2
        proj_3x3[2, 2] = 1
        #print(proj_3x3)
        return proj_3x3
    
    def get_view(self):
        view_4x4 = np.array(self.viewMatrix).reshape(4,4)
        #print(view_4x4)
        view_3x3 = np.array(self.viewMatrix).reshape(4,4)[:3, :3]
        #print(view_3x3)
        return view_3x3

    def detect_markers(self, frame):
        center_x = -1
        center_y = -1
        dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        aruco_params = cv2.aruco.DetectorParameters_create()

        (corners, ids, rejected) = cv2.aruco.detectMarkers(
        frame, dict, parameters=aruco_params)
        R = np.zeros((3,3))
        tvec = np.zeros((3, 1))

        if(len(corners) > 0):
            # Flatten the ArUco IDs list
            ids = ids.flatten()
            #print(ids)
            # Loop over the detected ArUco corners
            for (marker_corner, marker_id) in zip(corners, ids):
                
                # Extract the marker corners
                corners = marker_corner.reshape((4, 2))
                (top_left, top_right, bottom_right, bottom_left) = corners
                    
                # Convert the (x,y) coordinate pairs to integers
                top_right = (int(top_right[0]), int(top_right[1]))
                bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
                bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
                top_left = (int(top_left[0]), int(top_left[1]))
                    
                # Draw the bounding box of the ArUco detection
                cv2.line(frame, top_left, top_right, (0, 255, 0), 2)
                cv2.line(frame, top_right, bottom_right, (0, 255, 0), 2)
                cv2.line(frame, bottom_right, bottom_left, (0, 255, 0), 2)
                cv2.line(frame, bottom_left, top_left, (0, 255, 0), 2)
                    
                # Calculate and draw the center of the ArUco marker
                center_x = int((top_left[0] + bottom_right[0]) / 2.0)
                center_y = int((top_left[1] + bottom_right[1]) / 2.0)
                #cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

                
                # Marker pose estimation with PnP (no depth)
                proj_3x3 = self.get_intrinsics()
                #print(proj_3x3)


                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(marker_corner, 0.242, proj_3x3, 0)
                cv2.aruco.drawAxis(frame, proj_3x3, 0, rvec[0], tvec[0], 0.8) #tvec[0]

                # compute end effector rotation from Rordigues 
                R, _ = cv2.Rodrigues(rvec[0])

            # board stuff
            #board = cv2.aruco.GridBoard_create(2, 2, 0.01045, 0.00417, dict, firstMarker = 1)
            #_, rvec, tvec = cv2.aruco.estimatePoseBoard(corners, ids, board, self.get_intrinsics(), 0, None, None)
            #cv2.aruco.drawAxis(frame, self.get_intrinsics(), 0, rvec, tvec, 0.4)

        # Display the resulting frame with annotations
        cv2.imshow('frame',frame)

        # return the location of the tag and pose
        return R, tvec[0].T, (center_x, center_y)
