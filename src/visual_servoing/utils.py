import pybullet as p
import numpy as np
import torch

def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)



def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)



def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions of shape (..., 4).
    """
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)

def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles

def quaternion_invert(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    scaling = torch.tensor([1, -1, -1, -1], device=quaternion.device)
    return quaternion * scaling

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