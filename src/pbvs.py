from src.camera import *
import cv2
import pybullet as p
import numpy as np


class Marker:
    def __init__(self, corners, id, rvec=None, tvec=None):
        self.corners = corners.reshape((4, 2))
        self.id = id
        (self.top_left, self.top_right, self.bottom_right, self.bottom_left) = corners

        # Compute centers
        self.c_x = int((self.top_left[0] + self.bottom_right[0]) / 2.0)
        self.c_y = int((self.top_left[1] + self.bottom_right[1]) / 2.0)

        # Build homogenous transform if possible
        if(rvec is not None):
            self.build_transform(rvec, tvec)

    # Create homogenous transform from marker to camera
    def build_transform(self, rvec, tvec):
        Rcm, _ = cv2.Rodrigues(rvec)
        self.Tcm = np.vstack((np.hstack((Rcm, tvec)), np.array([0.0, 0, 0, 1])))


class MarkerPBVS:
    # camera: Instance of a camera following the Camera interface
    # k_v: scaling constant for linear velocity control
    # k_omega: scaling constant for angular velocity control 
    # eef_tag_ids: IDs of tags on a board, length of N for a board of N many tags
    # eef_tag_geometry: list of 4x3 numpy, each numpy mat is the 3d coordinates of
    #                   the 4 tag corners, tl, tr, br, bl in that order in the eef_tag 
    #                   coordinate system the list is length N for a board of N many tags
    def __init__(self, camera, k_v, k_omega, eef_tag_ids, eef_tag_geometry):
        self.k_v = k_v
        self.k_omega = k_omega
        self.camera = camera

        # AR Tag detection parameters
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        # EEF board
        self.eef_board = cv2.aruco.Board_create(eef_tag_geometry, self.aruco_dict, eef_tag_ids)
       
    # Detect ArUco tags in a given RGB frame
    # Return a list of Marker objects
    def detect_markers(self, frame, draw_debug=True):
        out = []
        (corners_all, ids_all, rejected) = cv2.aruco.detectMarkers(frame, self.aruco_dict, parameters=self.aruco_params)
        if(len(corners_all) == 0):
            return out
        ids = ids_all.flatten()
    
        # Loop over the detected ArUco corners
        for (marker_corner, marker_id) in zip(corners_all, ids):
            # Extract the marker corners
            marker_corner = marker_corner.reshape((4, 2))
            marker = Marker(marker_corner, marker_id)
                
            # Convert the (x,y) coordinate pairs to integers
            top_right = (int(marker.top_right[0]), int(marker.top_right[1]))
            bottom_right = (int(marker.bottom_right[0]), int(marker.bottom_right[1]))
            bottom_left = (int(marker.bottom_left[0]), int(marker.bottom_left[1]))
            top_left = (int(marker.top_left[0]), int(marker.top_left[1]))
                
            # Draw the bounding box of the ArUco detection
            if(draw_debug):
                cv2.line(frame, top_left, top_right, (0, 255, 0), 2)
                cv2.line(frame, top_right, bottom_right, (0, 255, 0), 2)
                cv2.line(frame, bottom_right, bottom_left, (0, 255, 0), 2)
                cv2.line(frame, bottom_left, top_left, (0, 255, 0), 2)

            out.append(marker)

        return out

    # Estimate the pose of a predefined marker board given a set of candidate markers that may be in the board
    def get_board_pose(self, markers, board, frame=None):
        # Setup stuff we need for pose estimate
        intrinsics = self.camera.get_intrinsics()    
        corners_all = []
        ids_all = []
        if(len(markers) == 0):
            return None
        
        # Build up corner and id set for the board 
        ref_marker = None
        for marker in markers:
            corners_all.append(marker.corners.reshape(-1))  
            ids_all.append(marker.id)
            if(marker.id == board.ids[0]):
                ref_marker = marker

        if(ref_marker is None):
            return ref_marker
        # Marker pose estimation with PnP
        _, rvec, tvec = cv2.aruco.estimatePoseBoard(corners_all, np.array(ids_all), board, intrinsics, 0, None, None)

        # The first marker of the board is considered the reference marker and will contain the transform
        ref_marker.build_transform(rvec, tvec)

        # Draw debug pose visualization if a frame is passed in
        if(frame is not None):
            cv2.aruco.drawAxis(frame, self.camera.get_intrinsics(), 0, rvec, tvec, 0.4)

        return ref_marker

    ####################
    # PBVS control law #
    #################### 
    
    def get_v(self, object_pos, eef_pos):
        return (object_pos - eef_pos) * self.k_v

    def get_omega(self, Rwa, Rwo):
        Rao = np.matmul(Rwa, Rwo.T).T
        Rao_rod, _ = cv2.Rodrigues(Rao)
        return Rao_rod * self.k_omega

    def get_control(self, object_pos, eef_pos, Rwa, Rwo):
        ctrl = np.zeros(6)
        ctrl[0:3] = self.get_v(object_pos, eef_pos)
        ctrl[3:6] = np.squeeze(self.get_omega(Rwa, Rwo))
        return ctrl