import numpy as np
import pybullet as p
import pybullet_data
from visual_servoing.utils import get_link_tf
# Joint names
right_arm_joints = [
    'victor_right_arm_joint_1',
    'victor_right_arm_joint_2',
    'victor_right_arm_joint_3',
    'victor_right_arm_joint_4',
    'victor_right_arm_joint_5',
    'victor_right_arm_joint_6',
    'victor_right_arm_joint_7',
]

left_arm_joints = [
    'victor_left_arm_joint_1',
    'victor_left_arm_joint_2',
    'victor_left_arm_joint_3',
    'victor_left_arm_joint_4',
    'victor_left_arm_joint_5',
    'victor_left_arm_joint_6',
    'victor_left_arm_joint_7',
]

class Victor:
    def __init__(self, start_pos=None, start_orientation=None):
        # Set up simulation 
        # Load Victor URDF
        client = p.connect(p.GUI)
        self.urdf = p.loadURDF("models/victor_description/urdf/victor.urdf", [0,0,0], p.getQuaternionFromEuler([0,0,0]))

        # Organize joints into a dict from name->info
        self.joints_by_name = {}
        self.links_by_name = {}
        num_joints = p.getNumJoints(self.urdf)
        for i in range(num_joints):
            info = p.getJointInfo(self.urdf, i)
            joint_name = info[1].decode("ascii") 
            link_name = info[12].decode("ascii") 
            self.joints_by_name[joint_name] = info
            self.links_by_name[link_name] = info
            print(f"idx: {info[0]}, joint: {joint_name}, link: {link_name}, type:{info[2]} ")
        self.eef_idx = 16

        # Get arm and end effector joint indicies
        self.right_tool = p.getJointInfo(self.urdf, self.eef_idx)
        self.left_tool = p.getJointInfo(self.urdf, self.eef_idx)

        self.right_arm_joints = []
        self.left_arm_joints = []
        for joint_name in left_arm_joints: 
            self.left_arm_joints.append(self.joints_by_name[joint_name][0])

    # transform helper, gets Tab b --> a 
    def get_tf(self, link_a, link_b):
        #joints_by_name[link_a]
        link_info_a = self.links_by_name[link_a]
        link_info_b = self.links_by_name[link_b]
        Twa = get_link_tf(self.urdf, link_info_a[0])
        Twb = get_link_tf(self.urdf, link_info_b[0])
        Tab = Twb @ np.linalg.inv(Twa)
        return Tab

    def get_eef_pos(self, side):
        """
        Returns ground truth end effector position in world frame
        """
        tool_idx = self.left_tool[0] if side == "left" else self.right_tool[0]
        result = p.getLinkState(self.urdf,
                                tool_idx,
                                computeLinkVelocity=1,
                                computeForwardKinematics=1)

        link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result
        return link_trn, link_rot 

    def get_arm_jacobian(self, side):
        """
        return 6 by 7 jacobian of the 7 dof left or right arm
        """

        if side == "left":
            tool = self.left_tool[0]
        else:
            tool = self.right_tool[0]

        # query joint positions
        joint_states = p.getJointStates(self.urdf, range(p.getNumJoints(self.urdf)))
        joint_infos = [p.getJointInfo(self.urdf, i) for i in range(p.getNumJoints(self.urdf))]
        joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
        joint_positions = [state[0] for state in joint_states]

        zero_vec = [0.0] * len(joint_positions)
        # offset from the CoM of the end effector to get the Jacobian relative to 
        loc_pos = [0.0] * 3

        jac_t, jac_r = p.calculateJacobian(self.urdf, tool, loc_pos, joint_positions, zero_vec, zero_vec)
        jac_t = np.array(jac_t)
        jac_r = np.array(jac_r)
        
        if side == "left": 
            return np.vstack((jac_t[:, 0:7], jac_r[:, 0:7]))  # Jacobian is 6 (end effector dof) x 7 (joints)
        else:
            return np.vstack((jac_t[:, 11:18], jac_r[:, 11:18]))

    def get_jacobian_pinv(self,side):
        J = self.get_arm_jacobian(side)
        lmda = 0.0000001
        J_pinv = np.dot(np.linalg.inv(np.dot(J.T, J) + lmda * np.eye(7)), J.T)
        return J_pinv

    def psuedoinv_ik_controller(self, side, target, current=None):
        x_prime = target
        if current is not None:
            x_prime = target - current

        J = self.get_arm_jacobian(side)
        lmda = 0.0000001

        J_pinv = np.dot(np.linalg.inv(np.dot(J.T, J) + lmda * np.eye(7)), J.T)

        q_prime = np.dot(J_pinv, x_prime)
        if np.linalg.norm(q_prime) > 0.55:
            q_prime = 0.55 * q_prime / np.linalg.norm(q_prime)  # * np.linalg.norm(x_prime)

        # joint limits 

        # control
        joint_list = self.left_arm_joints if (side == "left") else right_arm_joints

        p.setJointMotorControlArray(self.urdf, joint_list, p.VELOCITY_CONTROL, targetVelocities=q_prime)

    def set_velo(self, targetVelo):
        joint_list = self.left_arm_joints
        p.setJointMotorControlArray(self.urdf, joint_list, p.VELOCITY_CONTROL, targetVelocities=targetVelo)
    
