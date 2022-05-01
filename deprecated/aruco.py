import cv2
import numpy as np

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

aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters_create()

frame = cv2.imread("current.png")

out = []
(corners_all, ids_all, rejected) = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
print(rejected)
ids = []
if(ids_all is not None):
    ids = ids_all.flatten()

for marker in rejected:
    marker = marker[0]
    for corner in marker:
        cv2.circle(frame, ( int(corner[0]), int(corner[1]) ), 5, (0, 255, 0), -1)

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
    cv2.line(frame, top_left, top_right, (0, 255, 0), 2)
    cv2.line(frame, top_right, bottom_right, (0, 255, 0), 2)
    cv2.line(frame, bottom_right, bottom_left, (0, 255, 0), 2)
    cv2.line(frame, bottom_left, top_left, (0, 255, 0), 2)
    cv2.circle(frame, (marker.c_x, marker.c_y), 5, (255, 0, 0), -1)

    out.append(marker)

cv2.imshow("frame", frame)
cv2.waitKey(0)