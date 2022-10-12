import pybullet as p
import numpy as np


def get_link_tf(urdf, idx):
    result = p.getLinkState(urdf,
                            idx,
                            computeLinkVelocity=1,
                            computeForwardKinematics=1)

    link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
    rot_mat = np.array(p.getMatrixFromQuaternion(frame_rot)).reshape(3,3)
    tf = np.zeros((4,4))
    tf[0:3, 0:3] = rot_mat
    tf[0:3, 3] = frame_pos
    tf[3, 3] = 1
    return tf

def draw_sphere_marker(position, radius, color):
    vs_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
    marker_id = p.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)
    return marker_id


def draw_pose(trans, rot, uids=None, width=5, axis_len=0.1, alpha=1.0, mat=False):
    unique_ids = []
    if mat:
        coords = rot * axis_len + np.array(trans).reshape(3, 1)
    else:
        coords = np.array(p.getMatrixFromQuaternion(rot)).reshape(3, 3) * axis_len + np.array(trans).reshape(3, 1)
    colors = np.eye(3) * alpha + np.ones((3, 3)) * (1 - alpha)
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
    if line_ids is not None:
        for line_id in line_ids:
            p.removeUserDebugItem(line_id)

class PoseVisualizer:
    def __init__(self):
        self.uids = None

    def update(self, Two): 
        if self.uids is not None:
            erase_pos(self.uids)
        self.uids = draw_pose(Two[0:3, 3], Two[0:3, 0:3], mat=True)