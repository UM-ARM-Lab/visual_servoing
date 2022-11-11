from dataclasses import dataclass
import torch
import numpy as np
from pytorch_mppi import mppi
import pytorch_kinematics as pk

import tf.transformations
import pybullet as p

from visual_servoing.utils import axis_angle_to_quaternion, quaternion_invert, quaternion_multiply, quaternion_to_axis_angle, quaternion_to_matrix, matrix_to_quaternion
from visual_servoing.se3 import se3_exp_map

class VisualServoMPPI:

    def __init__(self, dt : float, eef_target_baselink : np.ndarray):
        """
        Targets in base link frame
        """
        self.dt = dt    
        self.eef_target_pos = torch.tensor(eef_target_baselink[0:3, 3], device="cuda", dtype=torch.float32)
        self.eef_target_rot = torch.tensor(tf.transformations.quaternion_from_matrix(eef_target_baselink), 
            device="cuda", dtype=torch.float32)

        self.controller = mppi.MPPI(self.arm_dynamics, self.cost, 
            9, 5 * torch.eye(9), num_samples=1000, horizon=15, device="cuda",
            u_min=-1.5 * torch.ones(9, dtype=torch.float32, device='cuda'),
            u_max=1.5 * torch.ones(9, dtype=torch.float32, device='cuda') 
            )

        self.chain = pk.build_serial_chain_from_urdf(
            open("/home/ashwin/source/lab/catkin_ws/src/hdt_robot/hdt_michigan_description/urdf/hdt_michigan.urdf").read(), "bracelet")
        self.chain.to(device="cuda")
        self.line_id = None

    def arm_dynamics_tester(self, Twe : np.ndarray, q : np.ndarray, q_dot : np.ndarray, Twb : np.ndarray, n_steps : int = 1):
        """
        Interface to arm dynamics for debugging

        Twe := homogenous transform to EEF in world frame (4, 4)
        q := joint angles (9, )
        q_dot := joint vel (9, )
        Twb := homogenous transform to base link in world frame (4, 4)
        """
        device = "cuda"
        dtype = torch.float32

        # Compute frame transforms to get from world relative to base relative
        self.Twb = Twb
        Tbw = np.linalg.inv(Twb)
        Tbe = Tbw @ Twe

        # Get starting pos, rot, joint angles as GPU tensors in base frame
        pos = torch.tensor(Tbe[:3, 3], device=device, dtype=dtype)
        rot = torch.tensor(tf.transformations.quaternion_from_matrix(Tbe), device=device, dtype=dtype)
        joint_angle = torch.tensor(q, device=device, dtype=dtype)
        x = torch.unsqueeze(torch.cat((pos, rot, joint_angle)), dim=0)
        
        # Rotate command into the base link frame
        q_dot_base_frame = q_dot
        u = torch.unsqueeze(torch.tensor(q_dot_base_frame, device=device, dtype=dtype), dim=0) 

        # Compute a step of dynamics
        for _ in range(n_steps):
            x = self.arm_dynamics(x, u)
            #x_pred = self.arm_dynamics(x, u)[0].cpu().numpy()
        x_pred = x[0].cpu().numpy()

        # Convert predited pose back to homogenous TF in world frame
        Tbe_pred = np.eye(4)
        Tbe_pred[0:3, 0:3] = tf.transformations.quaternion_matrix(x_pred[3:7])[0:3, 0:3]
        Tbe_pred[0:3, 3] = x_pred[:3]
        Twe_pred = Twb @ Tbe_pred
        return Twe_pred, x_pred[7:]

    def arm_dynamics(self, x : torch.Tensor, u : torch.Tensor):
        """
        x := eef pose + joint angles (k, 9) 
        u := joint vel (k, 9)
        """
        pos = x[:, :3]
        rot = x[:, 3:7]
        q = x[:, 7:]

        # get jacobians from current joint configs and compute eef twist
        J = self.chain.jacobian(q)
        eef_twist = torch.squeeze((J @ u.unsqueeze(dim=2)), dim=2)

        delta = se3_exp_map(eef_twist * self.dt)
        Tbe = torch.zeros((x.shape[0], 4, 4), device='cuda', dtype=torch.float32)
        Tbe[:, :3, :3] = quaternion_to_matrix(rot) 
        Tbe[:, :3, 3] = pos
        Tbe[:, 3, 3] = 1
        #Tbe_new = delta @ Tbe
        #Tbe_new = delta @ Tbe
        Tbe_new = Tbe @ torch.linalg.inv(delta)

        pos_next = Tbe_new[:, :3, 3]

        # Update EEF position
        rot_next = matrix_to_quaternion(Tbe_new[:, :3, :3])

        # Update joint config
        q_next = q + u * self.dt

        # New state vector
        x_next = torch.cat((pos_next, rot_next, q_next), dim=1)
        return x_next
    
    def cost(self, x : torch.Tensor, u : torch.Tensor):
        """
        q := eef_pose + joint angles (k, 9) 
        u := joint vel (k, 9)
        
        c(x, u) = (x - x_r)'Q(x - x_r)  
        """ 
        cost_pos = torch.linalg.norm(self.eef_target_pos - x[:, :3], dim=1)
        delta_rot = quaternion_multiply(x[:, 3:7], quaternion_invert(self.eef_target_rot))
        cost_rot = torch.linalg.norm(quaternion_to_axis_angle(delta_rot), dim=1)

        return cost_pos + 0.1*cost_rot

    def get_control(self, Twe : np.ndarray, Twb : np.ndarray, q : np.ndarray):
        """
        x := current EEF pose estimate
        """
        device = "cuda"
        dtype = torch.float32

        # Get starting pos, rot, joint angles as GPU tensors in base frame
        Tbw = np.linalg.inv(Twb)
        Tbe = Tbw @ Twe
        pos = torch.tensor(Tbe[:3, 3], device=device, dtype=dtype)
        rot = torch.tensor(tf.transformations.quaternion_from_matrix(Tbe), device=device, dtype=dtype)
        joint_angle = torch.tensor(q, device=device, dtype=dtype)
        x = torch.unsqueeze(torch.cat((pos, rot, joint_angle)), dim=0)
        

        ctrl = self.controller.command(x)
        

        return ctrl.cpu().numpy()