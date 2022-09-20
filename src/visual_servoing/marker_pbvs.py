import cv2
import numpy as np

from visual_servoing.camera import Camera
from visual_servoing.pf import ParticleFilter
from visual_servoing.pbvs import PBVS

from typing import List, Optional
import time

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
        (self.top_left, self.top_right, self.bottom_right, self.bottom_left) = corners.astype(np.int32)

        # Compute centers
        self.c_x = int((self.top_left[0] + self.bottom_right[0]) / 2.0)
        self.c_y = int((self.top_left[1] + self.bottom_right[1]) / 2.0)


class MarkerBoardDetector:
    """
    Used to detect and track a marker board 
    """
    def __init__(self, ids : List[int], geometry : List[np.ndarray], initial_rvec : Optional[np.ndarray] = None,
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
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.board = cv2.aruco.Board_create(geometry, self.aruco_dict, ids)
        self.rvec = initial_rvec
        self.tvec = initial_tvec
    
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
            # Extract the marker corners
            marker_corner = marker_corner.reshape((4, 2))
            marker = Marker(marker_corner, marker_id)
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
        corners_all = [marker.corners.reshape(-1) for marker in markers]
        ids_all = [marker.id for marker in markers]

        # Marker pose estimation with PnP
        _, self.rvec, self.tvec = cv2.aruco.estimatePoseBoard(corners_all, np.array(ids_all), self.board, intrinsics, 0, self.rvec, self.tvec, True)

        Rcm, _ = cv2.Rodrigues(self.rvec)
        Tcm = np.vstack((np.hstack((Rcm, self.tvec)), np.array([0.0, 0.0, 0.0, 1.0])))

        # Draw debug pose visualization if a frame is passed in
        if frame is not None:
            cv2.drawFrameAxes(frame, intrinsics, 0, self.rvec, self.tvec, 0.4)

        return Tcm

    def update(self, frame : np.ndarray, intrinsics : np.ndarray, ) -> Optional[np.ndarray]:
        """
        Attempts to detect markers and estimate the board pose with your frame if possible 
        """
        markers = self.detect_markers(frame)
        if(not markers):
            return None
        return self.get_board_pose(intrinsics, markers, frame)

class MarkerPBVS(PBVS):

    def __init__(self, camera : Camera, k_v : float, k_omega : float, max_joint_velo : float, 
        start_eef_pose, eef_tag_ids, eef_tag_geometry, target_tag_ids, target_tag_geometry, use_pf=False, debug=True):
        """
        Args:
            camera: instance of a camera following generic camera interface, must have OpenGL and OpenCV matricies defined
            k_v: scaling constant for linear velocity control
            k_omega: scaling constant for angular velocity control
            max_joint_velo: maximum joint velocity 
            eef_tag_ids: IDs of tags on a board, length of N for a board of N many tags
            eef_tag_geometry: (list of 4x3 numpy, each numpy mat is the 3d coordinates of the 4 tag corners, tl, tr, br, bl in that order in the eef_tag 
                coordinate system the list is length N for a board of N many tags)

        """
        super().__init__( camera, k_v, k_omega, max_joint_velo, debug)

        # AR Tag detection parameters
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        # EEF board
        self.eef_board = cv2.aruco.Board_create(eef_tag_geometry, self.aruco_dict, eef_tag_ids)
        self.target_board = cv2.aruco.Board_create(target_tag_geometry, self.aruco_dict, target_tag_ids)
        self.prev_pose = start_eef_pose

        # PF
        self.use_pf = use_pf
        self.prev_twist = np.zeros(6)
        self.prev_time = time.time()
        self.pf = ParticleFilter()
        self.max_joint_velo = max_joint_velo

    def generate_marker(self, output, id):
        """
        Generate a particular marker ID from the dictionary

        """
        cv2.imwrite(output, cv2.aruco.drawMarker(self.aruco_dict, id, 600))

    def compute_board_to_world(self, Tcm):
        """
        Get the transform from the world to the marker board

        """
        Tcm = ref_marker.Tcm
        # world -> camera -> ar tag
        Twm = (np.linalg.inv(self.camera.get_extrinsics()) @ Tcm)
        pos_unstable = (np.linalg.inv(self.camera.get_extrinsics()) @ Tcm)[0:3, 3]
        # query point cloud at center of tag for the position of the tag in the world frame
        pos = self.camera.get_xyz(ref_marker.c_x, ref_marker.c_y, depth)
        Twm[0:3, 3] = pos_unstable #pos[0:3]
        return Twm

    def do_pbvs(self, rgb, depth, Two, Tle, jac, jac_inv, dt):
        # Find the EEF ar tag board
        markers = self.detect_markers(rgb)
        ref_marker = self.get_board_pose(markers, self.eef_board, rgb)
        self.ref_marker = ref_marker # TODO delete
        #cv2.imshow("Camera", cv2.resize(rgb, (1280 // 3, 800 // 3)))  

        # If it was found, compute its pose estimate
        ctrl = np.zeros(6)
        Twe = self.prev_pose 
        if ref_marker is not None:
            Twa_sensor = self.compute_board_to_world(ref_marker, depth)
            # compute transform from world to end effector by including rigid transform from
            # eef ar tag to end effector frame
            Twe = Twa_sensor @ Tle

        # compute twist command based on state estimate and target
        ctrl = self.get_control(Twe, Two)
        ctrl = self.limit_twist(jac, jac_inv, ctrl)

        # store results
        self.prev_twist = ctrl
        self.prev_pose = Twe

        return ctrl, Twe
    
    # Find pose of target board
    def get_target_pose(self, rgb, depth, Tao, debug=True):
        markers = self.detect_markers(rgb)
        ref_marker = self.get_board_pose(markers, self.target_board, rgb)

        if ref_marker is not None:
            Twa = self.compute_board_to_world(ref_marker, depth) 
            return Twa @ Tao
        else:
            return None