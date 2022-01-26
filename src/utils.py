import pybullet as p
import numpy as np

def draw_sphere_marker(position, radius, color):
   vs_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
   marker_id = p.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)
   return marker_id

def draw_pose(trans, rot, uids=None, width=5, axis_len=0.1, alpha = 1.0, mat=False):
    unique_ids = []
    if(mat):
        coords = rot * axis_len + np.array(trans).reshape(3, 1)
    else:
        coords = np.array(p.getMatrixFromQuaternion(rot)).reshape(3, 3) * axis_len + np.array(trans).reshape(3, 1)
    colors = np.eye(3) * alpha + np.ones((3,3)) * (1 - alpha)
    if uids is None:
        for i in range(3):
            unique_ids.append(p.addUserDebugLine(trans,
                                                 np.asarray(coords[:, i]),
                                                 lineColorRGB=np.asarray(colors[:, i]),
                                                 lineWidth=width))
        return unique_ids
    else:
        for i in range(3):
            unique_ids.append(p.addUserDebugLine(trans,
                                                 np.asarray(coords[:, i]),
                                                 lineColorRGB=np.asarray(colors[:, i]),
                                                 lineWidth=width,
                                                 replaceItemUniqueId=uids[i]))
        return None


def erase_pos(line_ids):
    if(line_ids is not None):
        for line_id in line_ids:
            p.removeUserDebugItem(line_id)
