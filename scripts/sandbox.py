from dataclasses import dataclass
import cv2
from visual_servoing.pbvs import CheaterPBVS
from visual_servoing.marker_pbvs import MarkerPBVS
from visual_servoing.val import Val
from visual_servoing.utils import *
from visual_servoing.camera import PyBulletCamera
import numpy as np
import pybullet as p
from qpsolvers import solve_qp
from pytorch_mppi import mppi
import torch
import tf

val = Val([0.0, 0.0, -0.5])
camera = PyBulletCamera(camera_eye=np.array([0.7, -0.8, 0.5]), camera_look=np.array([0.7, 0.0, 0.2]))
p.setGravity(0, 0, -10)

def get_interaction_mat(u, v, z):
    return np.array([
        [-1/z, 0, u/z, u*v, -1* (1+u**2), v],
        [0, -1/z, v/z, (1+v**2), -u*v, -u]
    ])
uids_camera_marker = None
uids_eef_marker = None
uids_target_marker = None
uids_pred_eef_marker = None

# Specify the 3D geometry of the end effector marker board 
tag_len = 0.0305
gap_len = 0.0051
angle = np.pi / 4
# center tag
tag0_tl = np.array([-tag_len / 2, tag_len / 2, 0.0], dtype=np.float32)
tag0_tr = np.array([tag_len / 2, tag_len / 2, 0.0], dtype=np.float32)
tag0_br = np.array([tag_len / 2, -tag_len / 2, 0.0], dtype=np.float32)
tag0_bl = np.array([-tag_len / 2, -tag_len / 2, 0.0], dtype=np.float32)
z1 = -np.cos(angle) * gap_len
z2 = -np.cos(angle) * (gap_len + tag_len)
y1 = tag_len / 2 + gap_len + gap_len * np.sin(angle)
y2 = tag_len / 2 + gap_len + (gap_len + tag_len) * np.sin(angle)
# lower tag
tag1_tl = np.array([-tag_len / 2, -y1, z1], dtype=np.float32)
tag1_tr = np.array([tag_len / 2, -y1, z1], dtype=np.float32)
tag1_br = np.array([tag_len / 2, -y2, z2], dtype=np.float32)
tag1_bl = np.array([-tag_len / 2, -y2, z2], dtype=np.float32)
# upper tag
tag2_tl = np.array([-tag_len / 2, y2, z2], dtype=np.float32)
tag2_tr = np.array([tag_len / 2, y2, z2], dtype=np.float32)
tag2_br = np.array([tag_len / 2, y1, z1], dtype=np.float32)
tag2_bl = np.array([-tag_len / 2, y1, z1], dtype=np.float32)

tag0 = np.array([tag0_tl, tag0_tr, tag0_br, tag0_bl])
tag1 = np.array([tag1_tl, tag1_tr, tag1_br, tag1_bl])
tag2 = np.array([tag2_tl, tag2_tr, tag2_br, tag2_bl])
tag_geometry = [tag0, tag1, tag2]
ids = np.array([1, 2, 3])
ids2 = np.array([4,5,6])

#pbvs = MarkerPBVS(camera, 1, 1, 1.5, np.eye(4), ids, tag_geometry, ids2, tag_geometry)
def get_eef_gt(robot, quat=False):
    '''
    Gets the ground truth pose of the end effector from the simulator
    '''
    tool_idx = robot.left_tag[0]
    result = p.getLinkState(robot.urdf,
                            tool_idx,
                            computeLinkVelocity=1,
                            computeForwardKinematics=1)

    link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
    Twe = np.eye(4)
    Twe[0:3, 0:3] = np.array(p.getMatrixFromQuaternion(link_rot)).reshape(3, 3)
    Twe[0:3, 3] = link_trn
    if(quat):
        return link_trn, link_rot
    return Twe
#pbvs = CheaterPBVS(camera, 1, 1, 1.5, lambda : get_eef_gt(val))

def add_global_twist(twist, pose, dt):
    r3 = dt * twist[:3]
    so3, _ = cv2.Rodrigues(dt * twist[3:])
    tf = np.eye(4)
    tf[0:3, 0:3] = so3 @ pose[0:3, 0:3]
    tf[0:3, 3] = r3 + pose[0:3, 3]
    return tf

def reproject(Tcm, point_marker, camera):
    # Get another pt
    pt2_cam = Tcm @ point_marker
    pt2_im = camera.get_intrinsics() @ pt2_cam[0:3]
    pt2_im /= pt2_im[2]
    px2 = pt2_im[0:2].astype(int)
    return px2, pt2_cam

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


# Target
Two = np.eye(4) 
Two[0:3, 3] = np.array([0.8, 0.0, 0.2])
#Two[0:3, 0:3] = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler((np.pi/2, np.pi/4, 0)))).reshape(3, 3) # really hard
Two[0:3, 3] = np.array([0.75, 0.2, 0.2])
Two[0:3, 0:3] = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler((np.pi/2, 0, np.pi/10)))).reshape(3, 3) # hard
#Two[0:3, 0:3] = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler((np.pi/2, 0, -np.pi/4)))).reshape(3, 3) # really easy

#rgb, depth = camera.get_image()
#cv2.imshow("Im", rgb)
#cv2.waitKey(1)

@dataclass
class ArmDynamics:
    target_pos : np.ndarray # r3
    target_rot : np.ndarray # quat 
    J : np.ndarray
    dt : float

    def running_cost(self, x : torch.Tensor, u : torch.Tensor):
        pos_error = torch.norm((x[:, 0:3] - self.target_pos), 2, dim=1)
        ctrl_error = torch.norm(u, 2, dim=1)
        target_rot = torch.tensor(self.target_rot, device='cuda', dtype=torch.float32)
        rot_delta = quaternion_multiply(quaternion_invert(x[:, 3:7]), target_rot.reshape((1, -1)))
        rot_error = torch.norm(rot_delta ,2, dim=1)
        return pos_error # + rot_error #+ ctrl_error +

    def batchable_dynamics_arm(self, x : torch.Tensor, u : torch.Tensor):
        """
        Arm dynamics function
        x_(t+1) = x_t + dt J q'
        x : K x nx
        u : K x nu

        x is a vector [p r]' which has an R3 position and quaternion
        u is a vector of joint vleocities q'
        """
        # Compute velocity in state vector, p' and omega
        x_dot = (self.J @ u.T).T
        x_next = torch.zeros(x.shape, device="cuda", dtype=torch.float32)

        # Update position 
        x_next[:, 0:3] = x[:, 0:3] + x_dot[:, 0:3] * self.dt

        # Get the action delta as a quaternion
        # Note that action TFs are in global frame
        action_quat = axis_angle_to_quaternion(self.dt * x_dot[:, 3:6]).reshape(1, -1)
        
        # Update rot
        #x_next[:, 3:7] = quaternion_multiply(action_quat, x[:, 3:7])

        return x_next

J = val.get_arm_jacobian("left", True)
rot_target = torch.tensor(tf.transformations.quaternion_from_matrix(Two), device='cuda', dtype=torch.float32)
pos_target = torch.tensor(Two[0:3, 3], device="cuda", dtype=torch.float32)
dynamics = ArmDynamics(pos_target, rot_target, torch.tensor(J, device='cuda', dtype=torch.float32), 0.1)

controller = mppi.MPPI(dynamics.batchable_dynamics_arm, dynamics.running_cost, 
    7, 1.5 * torch.eye(9), 1000, 100, device="cuda",
    u_min=-1.5 * torch.ones(9, dtype=torch.float32, device='cuda'),
    u_max=1.5 * torch.ones(9, dtype=torch.float32, device='cuda') 
    )

pe, re = get_eef_gt(val, True)
pose = torch.tensor(np.hstack((pe, re)), device="cuda", dtype=torch.float32).reshape(1, -1)

def get_homogenous(se3):
    Twe = np.eye(4)
    Twe[0:3, 0:3] = np.array(p.getMatrixFromQuaternion(se3[3:7])).reshape(3, 3)
    Twe[0:3, 3] = se3[0:3]
    return Twe

pbvs = CheaterPBVS(camera, 1, 1, 1.5, lambda : get_homogenous(pose[0, :].cpu()))
while(True):

    # Step sim
    for _ in range(24):
        link_pos, link_rot = val.get_eef_pos("camera")
        camera_rot = np.array(p.getMatrixFromQuaternion(link_rot)).reshape(3,3)
        camera.upate_from_pose(link_pos, camera_rot)
        p.stepSimulation()
    
    #rgb, depth = camera.get_image()

    #pe, re = get_eef_gt(val, True)
    #pose = torch.tensor(np.hstack((pe, re)), device="cuda", dtype=torch.float32)

    jac = val.get_arm_jacobian("left", True)
    jac_pinv = val.get_jacobian_pinv("left", True)

    rgb = np.zeros(3)
    depth = np.zeros(3)
    ctrl, Twe = pbvs.do_pbvs(rgb, depth, Two, np.eye(4), jac, jac_pinv, 24) 
    #ctrl = controller.command(pose)
    # Send command to val
    ctrl_limit = pbvs.limit_twist(jac, jac_pinv, ctrl)
    pred = dynamics.batchable_dynamics_arm(pose.reshape(1, -1), torch.tensor(jac_pinv @ ctrl_limit, device="cuda", dtype=torch.float32).reshape(1, -1))
    if uids_pred_eef_marker is not None:
        erase_pos(uids_pred_eef_marker)
    uids_pred_eef_marker = draw_pose(pred[0, 0:3].cpu(), pred[0, 3:7].cpu())
    pose = pred
    #val.velocity_control("left", jac_pinv @ ctrl_limit, True)

    # Visualize camera pose  
    if (uids_camera_marker is not None):
        erase_pos(uids_camera_marker)
    Twc = np.linalg.inv(camera.get_extrinsics())
    uids_camera_marker = draw_pose(Twc[0:3, 3], Twc[0:3, 0:3], mat=True)

    #cv2.imshow("Im", rgb)
    #cv2.waitKey(1)

    # Visualize target pose 
    if (uids_target_marker is not None):
        erase_pos(uids_target_marker)
    uids_target_marker = draw_pose(Two[0:3, 3], Two[0:3, 0:3], mat=True)