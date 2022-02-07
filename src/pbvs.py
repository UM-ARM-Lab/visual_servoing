from camera import Camera
import cv2
import pybullet as p
import numpy as np


class Marker:
    def __init__(self, u, v, Rot):
        self.u = u
        self.v = v
        self.R


class ArUcoPBVS:

    def __init__(self, camera, k_v, k_omega):
        self.k_v = k_v
        self.k_omega = k_omega
        self.camera = camera

        # AR Tag detection parameters
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
    
        # Specify the 3D geometry of the end effector marker board 
        tag_len = 0.0305
        gap_len = 0.0051
        angle = np.pi/4
        # center tag
        tag0_tl = np.array([-tag_len/2, tag_len/2, 0.0], dtype=np.float32)
        tag0_tr = np.array([tag_len/2, tag_len/2, 0.0], dtype=np.float32)
        tag0_br = np.array([tag_len/2, -tag_len/2, 0.0], dtype=np.float32)
        tag0_bl = np.array([-tag_len/2, -tag_len/2, 0.0], dtype=np.float32)
        z1 = -np.cos(angle) * gap_len
        z2 = -np.cos(angle) * (gap_len+tag_len)
        y1 = tag_len/2 + gap_len + gap_len*np.sin(angle)
        y2 = tag_len/2 + gap_len + (gap_len + tag_len)*np.sin(angle)
        # lower tag
        tag1_tl = np.array([-tag_len/2, -y1, z1], dtype=np.float32)
        tag1_tr = np.array([tag_len/2,  -y1, z1], dtype=np.float32)
        tag1_br = np.array([tag_len/2, -y2, z2], dtype=np.float32)
        tag1_bl = np.array([-tag_len/2, -y2, z2], dtype=np.float32)
        # upper tag
        tag2_tl = np.array([-tag_len/2, y2, z2], dtype=np.float32)
        tag2_tr = np.array([tag_len/2,  y2, z2], dtype=np.float32)
        tag2_br = np.array([tag_len/2, y1, z1], dtype=np.float32)
        tag2_bl = np.array([-tag_len/2, y1, z1], dtype=np.float32)
        self.board = cv2.aruco.Board_create([ 
                np.array([tag0_tl, tag0_tr, tag0_br, tag0_bl]), 
                np.array([tag1_tl, tag1_tr, tag1_br, tag1_bl]),
            np.array([tag2_tl, tag2_tr, tag2_br, tag2_bl])
            ], self.aruco_dict, np.array([1, 2, 3]))
       
        
    # Retrieve an image from the simulation camera
    def get_static_camera_img(self):         
       
    
    # Retrieve 3x3 OpenCV style intrinsic matrix for the camera
    def get_intrinsics(self):
        proj_4x4 = np.array(self.projectionMatrix).reshape(4,4)
        proj_3x3 = np.array(self.projectionMatrix).reshape(4,4)[:3, :3]
        proj_3x3[0, 0] = proj_3x3[0, 0] * self.image_width/2
        proj_3x3[1, 1] = proj_3x3[1, 1] * self.image_height/2
        proj_3x3[0, 2] = self.image_width/2
        proj_3x3[1, 2] = self.image_height/2
        proj_3x3[2, 2] = 1
        return proj_3x3
    
    # Return 3x3 view matrix specifying camera rotation
    def get_view(self):
        view_4x4 = np.array(self.viewMatrix).reshape(4,4)
        view_3x3 = np.array(self.viewMatrix).reshape(4,4)[:3, :3]
        return view_3x3

    # Detect ArUco tags in a given camera frame
    # Returns list of detections (u, v, id, )
    def detect_markers(self, frame):
        #print(self.viewMatrix)
        #print(self.get_view())
        center_x = -1
        center_y = -1
        

        (corners_all, ids_all, rejected) = cv2.aruco.detectMarkers(
        frame, self.aruco_dict, parameters=self.aruco_params)
        
        Rot = np.zeros((3,3))
        tvec = np.zeros((3, 1))

        if(len(corners_all) > 0):
            # Flatten the ArUco IDs list
            ids = ids_all.flatten()
            #print(ids)
            # Loop over the detected ArUco corners
            for (marker_corner, marker_id) in zip(corners_all, ids):
                
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
                c_x = int((top_left[0] + bottom_right[0]) / 2.0)
                c_y = int((top_left[1] + bottom_right[1]) / 2.0)

                if(marker_id == 1):
                    center_x = c_x
                    center_y = c_y
                #cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

                
                # Marker pose estimation with PnP (no depth)
                proj_3x3 = self.get_intrinsics()
                #print(proj_3x3)


                #rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(marker_corner, 0.242, proj_3x3, 0)
                #cv2.aruco.drawAxis(frame, proj_3x3, 0, rvec[0], tvec[0], 0.8) #tvec[0]

                # compute end effector rotation from Rordigues 
                #Rot, _ = cv2.Rodrigues(rvec[0])

            
            _, rvec, tvec = cv2.aruco.estimatePoseBoard(corners_all, ids_all, board, self.get_intrinsics(), 0, None, None)
            cv2.aruco.drawAxis(frame, self.get_intrinsics(), 0, rvec, tvec, 0.4)
            Rot, _ = cv2.Rodrigues(rvec)

        # Display the resulting frame with annotations
        cv2.imshow('frame',frame)
        #0.282
#0.02813
        # return the location of the tag and pose
        return Rot, tvec, (center_x, center_y)

    # return (v, w) velocity in world frame at this timestep
    # that will try to drive the EEF to have the target position 
    # Twa - translation of eef (a) in world (w) 
    # Rwa - rotation of eef (a) in world (w)
    # Two - translation of target (o) in world (w)
    # Rwo - rotation of target (o) in world (w)
    def get_control(self, Twa, Rwa, Two, Rwo):
        lmbda = 0.1 

        # translation of target (o) in end effector frame (a)
        Tao = np.matmul(Rwa.T, Two) - np.matmul(Rwa.T, Twa)
        # translation of target (o) in desired end effector frame (d)
        Tdo = np.zeros(3)

        # Rotation of target (o) in current end effector frame
        Rao = np.matmul(Rwa.T, Rwo) 
        Rao_rod, _ = cv2.Rodrigues(Rao)

        # Desired rotation of target (o) in desired end effector frame (d)
        Rdo = np.zeros(3)

        v_a = -lmbda*(Tdo-Tao) + np.cross(np.squeeze(Tao), np.squeeze(Rao_rod))
        omega_a = np.squeeze(-lmbda * Rao_rod)
        
        v_w = np.matmul(Rwa, v_a)
        omega_w = np.matmul(Rwa, omega_a)
        #print(v_c)
        #print(omega_c)
        return np.hstack((v_w, np.squeeze(omega_w)))

    # get angular
    def get_omega(self, Rwa, Rwo):
        Roa = np.matmul(Rwa, Rwo.T).T
        Roa_rod, _ = cv2.Rodrigues(Roa)
        return Roa_rod * 1.1