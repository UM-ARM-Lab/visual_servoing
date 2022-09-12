from visual_servoing.val import Val
import numpy as np
import pybullet as p
from qpsolvers import solve_qp

val = Val([0.0, 0.0, -0.5])

while(True):

    ctrl = np.array([0, 0, 0, 0, 0, 0])
    jac = val.get_camera_jacobian()
    Q = np.eye(6)
    P = 2 * jac.T @ Q @ jac
    num_joints = jac.shape[1]
    q = (-ctrl @ Q @ jac - ctrl @ Q.T @ jac)
    G = np.vstack((np.eye(num_joints), -np.eye(num_joints)))
    h = np.ones(num_joints * 2) * 1.5
    num_joints = jac.shape[1]
    ctrl = solve_qp(P, q, G, h, None, None, solver="cvxopt")
    val.torso_control(ctrl)

    # Step sim
    for _ in range(24):
        p.stepSimulation()