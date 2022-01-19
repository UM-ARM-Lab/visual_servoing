import cv2
import pybullet as p
import numpy as np

class PVBS:

    def get_static_camera_img(self): 
        # TODO replace magic numbers with class params
        projectionMatrix = p.computeProjectionMatrixFOV(
            fov=45.0,
            aspect=1.0,
            nearVal=0.1,
            farVal=3.1)
        viewMatrix = p.computeViewMatrix(
            cameraEyePosition=[-2, 0, 0],
            cameraTargetPosition=[0, 0, 0],
            cameraUpVector=[0, 0, 1])
        width, height, rgbImg, depthImg, segImg = p.getCameraImage(
            width=1000,
            height=1000,
            viewMatrix=viewMatrix,
            projectionMatrix=projectionMatrix)
        rgb_img = np.array(rgbImg)[:, :, :3]
        depth_img = np.array(depthImg)
        return rgb_img, depth_img

    def detect_markers(self, frame):
        dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        aruco_params = cv2.aruco.DetectorParameters_create()

        (corners, ids, rejected) = cv2.aruco.detectMarkers(
        frame, dict, parameters=aruco_params)
        
        if(len(corners) > 0):
            # Flatten the ArUco IDs list
            ids = ids.flatten()
            
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
                cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
                
    
        # Display the resulting frame
        cv2.imshow('frame',frame)

