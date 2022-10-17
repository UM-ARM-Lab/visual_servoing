from dataclasses import dataclass
import torch
import numpy as np
from pytorch_mppi import mppi
import pytorch_kinematics as pk

from visual_servoing.utils import axis_angle_to_quaternion, quaternion_multiply

class VisualServoMPPI:

    def __init__(self, dt : float, eef_target_pos : np.ndarray):
        self.dt = dt    
        self.eef_target_pos = eef_target_pos

        #self.controller = mppi.MPPI(self.arm_dynamics, self.cost, 
        #    9, 1.5 * torch.eye(9), 1000, 100, device="cuda",
        #    u_min=-1.5 * torch.ones(9, dtype=torch.float32, device='cuda'),
        #    u_max=1.5 * torch.ones(9, dtype=torch.float32, device='cuda') 
        #    )

        self.chain = pk.build_serial_chain_from_urdf(
            open("/home/ashwin/source/lab/catkin_ws/src/hdt_robot/hdt_michigan_description/urdf/hdt_michigan.urdf").read(), "bracelet")
        self.chain.to(device="cuda")

    def arm_dynamics_tester(self, T : np.ndarray, q : np.ndarray, u : np.ndarray):
        """
        Interface to arm dynamics for debugging

        x := homogenous transform to EEF (4, 4) in base link frame
        q := joint angles (9, )
        u := joint vel (9, )
        """
        pos = torch.tensor(x[:3])

        #rot = torch.tensor()

    def arm_dynamics(self, x : torch.Tensor, u : torch.Tensor):
        """
        x := eef pose + joint angles (k, 9) 
        u := joint vel (k, 9)
        """
        pos = x[:3]
        rot = x[3:7]
        q = x[7:]

        # get jacobians from current joint configs and compute eef twist
        J = self.chain.jacobian(q)
        eef_twist = (J @ u.T).T

        # Update EEF position
        pos_next = pos + eef_twist[:, :3] * self.dt

        # EEF rotation
        rot_delta = axis_angle_to_quaternion(self.dt * eef_twist[:, 3:])
        rot_next = rot #quaternion_multiply(rot, rot_delta)

        # Update joint config
        q_next = q + u * self.dt

        # New state vector
        x_next = torch.cat((pos_next, rot_next, q_next), dim=1)
        return x_next
    
    def cost(self, q : torch.Tensor, u : torch.Tensor):
        """
        q := eef_pose + joint angles (k, 9) 
        u := joint vel (k, 9)
        
        c(x, u) = (x - x_r)'Q(x - x_r)  
        """ 
        # compute the forward kinematic configuration from q
        x = self.chain.forward_kinematics(q) 


    def get_control(self, q : np.ndarray):
        """
        x := current joint angles
        """

        ctrl = self.controller.command(x)