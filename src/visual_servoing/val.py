import numpy as np
import pybullet as p
import pybullet_data
from visual_servoing.arm_robot import ArmRobot

# Joint names
right_arm_joints = [
    'joint1',
    'joint2',
    'joint3',
    'joint4',
    'joint5',
    'joint6',
    'joint7',
]

left_arm_joints = [
    'joint41',
    'joint42',
    'joint43',
    'joint44',
    'joint45',
    'joint46',
    'joint47',
]


class Val(ArmRobot):
    def __init__(self, start_pos=None, start_orientation=None):
        # Set up simulation 
        if start_orientation is None:
            start_orientation = [0, 0, 0]
        if start_pos is None:
            start_pos = [0, 0, -0.0]
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        #p.setGravity(0, 0, -10)

        # Load Val URDF
        self.urdf =  p.loadURDF("models/husky_custom_description/urdf/mic09_description.urdf", start_pos, p.getQuaternionFromEuler(start_orientation))
        #self.urdf = p.loadURDF("models/hdt_michigan_description_orig/urdf/hdt_michigan_generated.urdf", start_pos,
        #                       p.getQuaternionFromEuler(start_orientation))
        planeId = p.loadURDF("models/short_floor.urdf", [start_pos[0], start_pos[1], start_pos[2]-0.15], useFixedBase=1)

        # Organize joints into a dict from name->info
        self.joints_by_name = {}
        num_joints = p.getNumJoints(self.urdf)
        for i in range(num_joints):
            info = p.getJointInfo(self.urdf, i)
            name = info[1].decode("ascii")
            self.joints_by_name[name] = info
            print(f"idx: {info[0]}, joint: {name}, type:{info[2]} ")

        # Get arm and end effector joint indicies
        self.left_tool = self.joints_by_name["left_tool_joint"]
        self.right_tool = self.joints_by_name["right_tool_joint"]

        self.left_tag = self.joints_by_name["ar_joint"]

        self.left_arm_joints = []
        self.right_arm_joints = []
        for i in range(1, 8):
            #print(self.joints_by_name["joint4" + str(i)][0])
            self.left_arm_joints.append(self.joints_by_name["joint4" + str(i)][0])
            self.right_arm_joints.append(self.joints_by_name["joint" + str(i)][0])

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

            return np.vstack((jac_t[:, 6+6:13+6], jac_r[:, 6+6:13+6]))  # Jacobian is 6 (end effector dof) x 7 (joints)
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

        # joint limits 

        # control
        joint_list = self.left_arm_joints if (side == "left") else right_arm_joints

        p.setJointMotorControlArray(self.urdf, joint_list, p.VELOCITY_CONTROL, targetVelocities=q_prime)

    def set_velo(self, targetVelo):
        joint_list = self.left_arm_joints
        p.setJointMotorControlArray(self.urdf, joint_list, p.VELOCITY_CONTROL, targetVelocities=targetVelo)
    
