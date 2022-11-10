from visual_servoing.camera import RealsenseCamera, ZEDCamera
import numpy as np
import cv2


camera = ZEDCamera()

selection = None
while(selection != 32):
    rgb = camera.get_image()[:, :, :3]
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    cv2.imshow("image", rgb)
    selection = cv2.waitKey(1)