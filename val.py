import pybullet as p
import time
import pybullet_data
import numpy as np

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

class Val: 
    def __init__(self, start_pos=[0,0,0], start_orientation=[0,0,0]):
        # Set up simulation 
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) 
        p.setGravity(0,0,-10)

        # Load Val URDF
        self.urdf =  p.loadURDF("hdt_michigan_description/urdf/hdt_michigan_generated.urdf", start_pos, p.getQuaternionFromEuler(start_orientation))

        # Organize joints into a dict from name->info
        self.joints_by_name = {}
        num_joints = p.getNumJoints(self.urdf)
        for i in range(num_joints):
            info = p.getJointInfo(self.urdf, i)
            name = info[1].decode("ascii")
            self.joints_by_name[name] = info

        # Get arm and end effector joint indicies
        self.left_tool = self.joints_by_name["left_tool_joint"]
        self.right_tool = self.joints_by_name["right_tool_joint"]

        self.left_arm_joints = []
        self.right_arm_joints = []
        for i in range(1, 8):
            self.left_arm_joints.append(self.joints_by_name["joint4"+str(i)][0])
            self.right_arm_joints.append(self.joints_by_name["joint"+str(i)][0])

        

    def get_arm_jacobian(self, side):
        """
        return 6 by 7 jacobian of the 7 dof left or right arm
        """

        if(side == "left"):
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
        loc_pos=[0.0] * 3

        jac_t, jac_r = p.calculateJacobian(self.urdf, tool, loc_pos, joint_positions, zero_vec, zero_vec)

        if side == "left":
            return np.vstack((jac_t[:, 2:9], jac_r[:, 2:9])) # Jacobian is 6 (end effector dof) x 7 (joints)
        else:
            return np.vstack((jac_t[:, 11:18], jac_r[:, 11:18]))

val = Val()
p.setRealTimeSimulation(1)
print(val.get_jacobian(val.left_tool[0]))
input()