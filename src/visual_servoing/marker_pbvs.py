from visual_servoing.camera import Camera
from visual_servoing.pf import ParticleFilter
from visual_servoing.pbvs import PBVS

import cv2
import time
import numpy as np

from typing import List, Optional


class Marker:
    """
    Represents a detected marker
    """
    def __init__(self, id : int, corners : np.ndarray):
        """
        Args:
            corners: 4 by 2 numpy mat for the 4 (u,v) corners of this marker 
            id: id of this marker
        """
        self.id = id
        self.corners = corners
        (self.top_left, self.top_right, self.bottom_right, self.bottom_left) = [tuple(corner) for corner in corners.astype(np.int32) ]

        # Compute centers
        self.c_x = int((self.top_left[0] + self.bottom_right[0]) / 2.0)
        self.c_y = int((self.top_left[1] + self.bottom_right[1]) / 2.0)


class MarkerBoardDetector:
    """
    Used to detect and track a marker board 
    """
    def __init__(self, ids : List[int], geometry : List[np.ndarray], dictionary : int = cv2.aruco.DICT_6X6_250, initial_rvec : Optional[np.ndarray] = None,
        initial_tvec : Optional[np.ndarray] = None):
        """
        Args: 
            ids: ids of the aruco tags on the board 
            geometry: a list of 4x3 numpy arrays, where each array describes the 3D location 
            of the marker corners in the board frame. The row dim is for each corner, which
            are top left, top right, bottom right, then bottom left. The col dim is [x, y, z]
            initial_rvec: initial Rodrigues board in camera frame
            initial_tvec: initial translation board in camera frame
        """
        # AR Tag detection parameters and dictionary
        self.aruco_dict = cv2.aruco.Dictionary_get(dictionary)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.board = cv2.aruco.Board_create(geometry, self.aruco_dict, ids)
        self.rvec = initial_rvec
        self.tvec = initial_tvec
        self.ids = ids
    
    def detect_markers(self, frame : np.ndarray, draw_debug : bool = True) -> Optional[List[Marker]]:
        """
        Detect ARUCO tags in a given RGB frame and return a list of Marker objects, assumes
        undistorted image frame
        ​
        Args:
            frame: undistorted rgb camera frame 
        """
        out = []
        (corners_all, ids_all, rejected) = cv2.aruco.detectMarkers(frame, self.aruco_dict, parameters=self.aruco_params)
        if len(corners_all) == 0:
            return None
        ids = ids_all.flatten()

        # Loop over the detected ArUco corners
        for (marker_corner, marker_id) in zip(corners_all, ids):
            if not (marker_id in self.ids):
                continue
            # Extract the marker corners
            marker_corner = marker_corner.reshape((4, 2))
            marker = Marker(marker_id, marker_corner)
            out.append(marker)

            # Draw the bounding box of the ArUco detection
            if draw_debug:
                col_green = (0, 255, 0)
                cv2.line(frame, marker.top_left, marker.top_right, col_green, 2)
                cv2.line(frame, marker.top_right, marker.bottom_right, col_green, 2)
                cv2.line(frame, marker.bottom_right, marker.bottom_left, col_green, 2)
                cv2.line(frame, marker.bottom_left, marker.top_left, col_green, 2)
                cv2.circle(frame, (marker.c_x, marker.c_y), 5, col_green, -1)
        return out
    
    def get_board_pose(self, intrinsics : np.ndarray, markers : List[Marker], frame : Optional[np.ndarray]=None) -> np.ndarray:
        """
        Estimate the pose of a marker board, assumes the 
        """
        # Setup stuff we need for pose estimate
        corners_all = [marker.corners.reshape(-1) for marker in markers if marker.id in self.ids]
        ids_all = [marker.id for marker in markers if marker.id in self.ids]

        # Marker pose estimation with PnP
        use_guess = True
        if(self.rvec is None or self.tvec is None):
            use_guess = False
        _, self.rvec, self.tvec = cv2.aruco.estimatePoseBoard(corners_all, np.array(ids_all), self.board, intrinsics, 0, self.rvec, self.tvec, use_guess)

        Rcm, _ = cv2.Rodrigues(self.rvec)
        Tcb = np.vstack((np.hstack((Rcm, self.tvec)), np.array([0.0, 0.0, 0.0, 1.0])))

        # Draw debug pose visualization if a frame is passed in
        if frame is not None:
            cv2.drawFrameAxes(frame, intrinsics, 0, self.rvec, self.tvec, 0.4)

        return Tcb

    def update(self, frame : np.ndarray, intrinsics : np.ndarray, ) -> Optional[np.ndarray]:
        """
        Attempts to detect markers and estimate the board pose with your frame if possible 
        """
        markers = self.detect_markers(frame)
        if(not markers):
            return None
        return self.get_board_pose(intrinsics, markers, frame)

    @staticmethod
    def generate_marker(self, id):
        """
        Generate a particular marker ID from the dictionary

        """
        return cv2.aruco.drawMarker(self.aruco_dict, id, 600)

class MarkerPBVS(PBVS):

    def __init__(self, camera : Camera, k_v : float, k_omega : float, max_joint_velo : float,
        detector : MarkerBoardDetector):
        """
        Args:
            camera: instance of a camera following generic camera interface, must have OpenGL and OpenCV matricies defined
            k_v: scaling constant for linear velocity control
            k_omega: scaling constant for angular velocity control
            max_joint_velo: maximum joint velocity 
            detector: marker board detector instance

        """
        super().__init__(camera, k_v, k_omega, max_joint_velo)

        # AR Tag detection parameters
        self.detector = detector


    def compute_board_to_world(self, Tcb):
        """
        Get the transform from the world to the marker board given the board pose in camera frame
        """
        # world -> camera -> board
        Twb = (np.linalg.inv(self.camera.get_extrinsics()) @ Tcb)
        return Twb

    def do_pbvs(self, rgb, depth, Two, Tbe, jac, jac_inv, dt, rescale=True):
        # Find the EEF ar tag board and estimate its pose in camera frame
        Tcb = self.detector.update(rgb, self.camera.get_intrinsics())

        # Return dummy vals when pose can't be estimated
        if(Tcb is None):
            return np.zeros(6), np.eye(4)

        # If it was found, compute its world pose using camera extrinsics
        Twb = (np.linalg.inv(self.camera.get_extrinsics()) @ Tcb)
        Twe = Twb @ Tbe

        # compute twist command based on state estimate and target
        ctrl = self.get_control(Twe, Two)
        if(rescale):
            ctrl = self.limit_twist(jac, jac_inv, ctrl)

        return ctrl, Twe
