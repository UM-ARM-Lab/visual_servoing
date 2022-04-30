import cv2
import numpy as np

from visual_servoing.camera import Camera
from visual_servoing.pf import ParticleFilter
from visual_servoing.pbvs import PBVS
import time

class Marker:
    def __init__(self, corners, id, rvec=None, tvec=None):
        """
        Args:
            corners: 4 by 2 numpy mat for the 4 (u,v) corners of this marker 
            id: id of this marker
            rvec: Rodrigues of this marker
            tvec: transform of this marker

        """
        self.corners = corners.reshape((4, 2))
        self.id = id
        (self.top_left, self.top_right, self.bottom_right, self.bottom_left) = corners

        # Compute centers
        self.c_x = int((self.top_left[0] + self.bottom_right[0]) / 2.0)
        self.c_y = int((self.top_left[1] + self.bottom_right[1]) / 2.0)

        # Build homogenous transform if possible
        if rvec is not None:
            self.build_transform(rvec, tvec)

    def build_transform(self, rvec, tvec):
        """
        Create homogenous transform from marker to camera

        """
        Rcm, _ = cv2.Rodrigues(rvec)
        self.Tcm = np.vstack((np.hstack((Rcm, tvec)), np.array([0.0, 0, 0, 1])))


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
        #self.aruco_params.adaptiveThreshWinSizeMax = 10
        #self.aruco_params.adaptiveThreshWinSizeMin = 3
        #self.aruco_params.adaptiveThreshConstant = 10

        # EEF board
        self.eef_board = cv2.aruco.Board_create(eef_tag_geometry, self.aruco_dict, eef_tag_ids)
        #self.target_board = cv2.aruco.Board_create(target_tag_geometry, self.aruco_dict, target_tag_ids)
        self.prev_pose = start_eef_pose

        # PF
        self.use_pf = use_pf
        self.prev_twist = np.zeros(6)
        self.prev_time = time.time()
        self.pf = ParticleFilter()
        self.max_joint_velo = max_joint_velo

    def detect_markers(self, frame, draw_debug=True):
        """
        Detect ARUCO tags in a given RGB frame and return a list of Marker objects
        ​
        Args:
            frame: rgb camera frame
        ​
        Returns:
            out: python list of detected Marker objects
        ​
        """
        out = []
        (corners_all, ids_all, rejected) = cv2.aruco.detectMarkers(frame, self.aruco_dict, parameters=self.aruco_params)
        if len(corners_all) == 0:
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
            if draw_debug:
                cv2.line(frame, top_left, top_right, (0, 255, 0), 2)
                cv2.line(frame, top_right, bottom_right, (0, 255, 0), 2)
                cv2.line(frame, bottom_right, bottom_left, (0, 255, 0), 2)
                cv2.line(frame, bottom_left, top_left, (0, 255, 0), 2)
                cv2.circle(frame, (marker.c_x, marker.c_y), 5, (255, 0, 0), -1)

            out.append(marker)

        return out

    def get_board_pose(self, markers, board, frame=None):
        """
        Estimate the pose of a predefined marker board given a set of candidate markers that may be in the board

        """
        # Setup stuff we need for pose estimate
        intrinsics = self.camera.get_intrinsics()
        corners_all = []
        ids_all = []
        if len(markers) == 0:
            return None

        # Build up corner and id set for the board 
        ref_marker = None
        for marker in markers:
            corners_all.append(marker.corners.reshape(-1))
            ids_all.append(marker.id)
            if marker.id == board.ids[0]:
                ref_marker = marker

        if ref_marker is None:
            return ref_marker
        # Marker pose estimation with PnP
        _, rvec, tvec = cv2.aruco.estimatePoseBoard(corners_all, np.array(ids_all), board, intrinsics, 0, None, None)

        # The first marker of the board is considered the reference marker and will contain the transform
        ref_marker.build_transform(rvec, tvec)

        # Draw debug pose visualization if a frame is passed in
        if frame is not None:
            cv2.aruco.drawAxis(frame, self.camera.get_intrinsics(), 0, rvec, tvec, 0.4)

        return ref_marker

    def generate_marker(self, output, id):
        """
        Generate a particular marker ID from the dictionary

        """
        cv2.imwrite(output, cv2.aruco.drawMarker(self.aruco_dict, id, 600))

    def compute_board_to_world(self, ref_marker, depth):
        """
        Get the transform from the world to the ar tag

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
        cv2.imshow("Camera", cv2.resize(rgb, (1280 // 3, 800 // 3)))  

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