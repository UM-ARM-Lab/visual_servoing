from dataclasses import dataclass
import torch
import numpy as np
from pytorch_mppi import mppi
import pytorch_kinematics as pk

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
            open("models/val/husky_custom_description/urdf/mic09_description.urdf").read(), "ar_link", "pedestal_link")
        self.chain.to(device="cuda")


    def arm_dynamics(self, q : torch.Tensor, u : torch.Tensor):
        """
        q := joint angles (k, 9) 
        u := joint vel (k, 9)
        """
        return q + u * self.dt
    
    def cost(self, q : torch.Tensor, u : torch.Tensor):
        """
        q := joint angles (k, 9) 
        u := joint vel (k, 9)
        
        c(x, u) = (x - x_r)'Q(x - x_r)  
        """ 
        # compute the forward kinematic configuration from q
        x = self.chain.forward_kinematics(q) 


    def get_control(self, x : np.ndarray):
        """
        x := current joint angles
        """
        ctrl = self.controller.command(x)