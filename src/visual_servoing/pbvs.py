import cv2
import numpy as np

from visual_servoing.camera import Camera


class Marker:
    def __init__(self, corners, id, rvec=None, tvec=None):
        self.corners = corners.reshape((4, 2))
        self.id = id
        (self.top_left, self.top_right, self.bottom_right, self.bottom_left) = corners

        # Compute centers
        self.c_x = int((self.top_left[0] + self.bottom_right[0]) / 2.0)
        self.c_y = int((self.top_left[1] + self.bottom_right[1]) / 2.0)

        # Build homogenous transform if possible
        if rvec is not None:
            self.build_transform(rvec, tvec)

    # Create homogenous transform from marker to camera
    def build_transform(self, rvec, tvec):
        Rcm, _ = cv2.Rodrigues(rvec)
        self.Tcm = np.vstack((np.hstack((Rcm, tvec)), np.array([0.0, 0, 0, 1])))


class MarkerPBVS:
    # camera: (Instance of a camera following the Camera interface)
    # k_v: (Scaling constant for linear velocity control)
    # k_omega: (Scaling constant for angular velocity control) 
    # eef_tag_ids: (IDs of tags on a board, length of N for a board of N many tags)
    # eef_tag_geometry: (list of 4x3 numpy, each numpy mat is the 3d coordinates of
    # the 4 tag corners, tl, tr, br, bl in that order in the eef_tag 
    # coordinate system the list is length N for a board of N many tags)
    def __init__(self, camera: Camera, k_v, k_omega, eef_tag_ids, eef_tag_geometry, target_tag_ids, target_tag_geometry):
        self.k_v = k_v
        self.k_omega = k_omega
        self.camera = camera

        # AR Tag detection parameters
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        # EEF board
        self.eef_board = cv2.aruco.Board_create(eef_tag_geometry, self.aruco_dict, eef_tag_ids)
        self.target_board = cv2.aruco.Board_create(target_tag_geometry, self.aruco_dict, target_tag_ids)

    # Detect ArUco tags in a given RGB frame
    # Return a list of Marker objects
    def detect_markers(self, frame, draw_debug=True):
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

    # Estimate the pose of a predefined marker board given a set of candidate markers that may be in the board
    def get_board_pose(self, markers, board, frame=None):
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

    # Generate a particular marker ID from the dictionary
    def generate_marker(self, output, id):
        cv2.imwrite(output, cv2.aruco.drawMarker(self.aruco_dict, id, 600))

    # Get the transform from the world to the ar tag
    def compute_board_to_world(self, ref_marker, depth):
        Tcm = ref_marker.Tcm
        # world -> camera -> ar tag
        Twm = (np.linalg.inv(self.camera.get_extrinsics()) @ Tcm)
        pos_unstable = (np.linalg.inv(self.camera.get_extrinsics()) @ Tcm)[0:3, 3]
        # query point cloud at center of tag for the position of the tag in the world frame
        pos = self.camera.get_xyz(ref_marker.c_x, ref_marker.c_y, depth)
        Twm[0:3, 3] = pos_unstable#pos[0:3]
        return Twm

    # Executes an iteration of PBVS control and returns a twist command
    # Two: (Pose of target object w.r.t world)
    # Tae: (Pose of end effector w.r.t. ar tag coordinates)
    # Returns the twist command [v, omega] and pose of end effector in world
    def do_pbvs(self, rgb, depth, Two, Tae, debug=True):
        # Find the EEF ar tag board
        markers = self.detect_markers(rgb)
        ref_marker = self.get_board_pose(markers, self.eef_board, rgb)
        if debug:
            cv2.imshow("image", rgb)

        if ref_marker is not None:
            Twa = self.compute_board_to_world(ref_marker, depth)
            # compute transform from world to end effector by including rigid transform from
            # eef ar tag to end effector frame
            Twe = Twa @ Tae
            # compute twist command
            ctrl = self.get_control(Twe, Two)
            return ctrl, Twe
        return np.zeros(6), np.zeros((4, 4))
    
    # Find pose of target board
    def get_target_pose(self, rgb, depth, Tao, debug=True):
        markers = self.detect_markers(rgb)
        ref_marker = self.get_board_pose(markers, self.target_board, rgb)
        if debug:
            cv2.imshow("image", rgb)

        if ref_marker is not None:
            Twa = self.compute_board_to_world(ref_marker, depth) 
            return Twa @ Tao
        else:
            return None

    ####################
    # PBVS control law #
    #################### 

    def get_v(self, object_pos, eef_pos):
        return (object_pos - eef_pos) * self.k_v

    def get_omega(self, Rwa, Rwo):
        Rao = np.matmul(Rwa, Rwo.T).T
        Rao_rod, _ = cv2.Rodrigues(Rao)
        return Rao_rod * self.k_omega

    def get_control(self, Twe, Two):
        ctrl = np.zeros(6)
        ctrl[0:3] = self.get_v(Two[0:3, 3], Twe[0:3, 3])
        ctrl[3:6] = np.squeeze(self.get_omega(Twe[0:3, 0:3], Two[0:3, 0:3]))
        return ctrl
