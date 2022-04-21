import numpy as np
import pybullet as p
import pybullet_data
from visual_servoing.utils import get_link_tf, draw_pose
from visual_servoing.arm_robot import ArmRobot
import pickle
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

class Victor(ArmRobot):
    def __init__(self, arm_states=None, use_aruco=False, start_pos=None, start_orientation=None):
        # Set up simulation 
        # Load Victor URDF
        p.setAdditionalSearchPath("models/")
        #box_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[10, 10, 10], rgbaColor=[1, 0, 0, 1])
        #box_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[10, 10, 10])
        #box_multi = p.createMultiBody(baseCollisionShapeIndex = 0, baseVisualShapeIndex=box_vis, basePosition=[0, 0, 0])
        if(use_aruco):
            self.urdf = p.loadURDF("models/victor_description/urdf/victor-with-cuff.urdf", [0,0,0], p.getQuaternionFromEuler([0,0,0]))
        else:
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

        # load pickle
        self.gripper_pkl = pickle.load(open("robot_points.pkl", 'rb'))

        # set arm states
        if(arm_states is not None):
            for joint_name, state in zip(left_arm_joints, arm_states):
                id = self.joints_by_name[joint_name][0]
                p.resetJointState(self.urdf, id, state)

    def get_arm_joint_configs(self):
        joint_states = {}
        for joint_name in left_arm_joints:
            idx = self.joints_by_name[joint_name][0]
            pos, vel, force, torque = p.getJointState(self.urdf, idx)
            joint_states[joint_name] = pos
        return joint_states

    # transform helper, gets Tab 
    def get_tf(self, link_a, link_b):
        #joints_by_name[link_a]
        link_info_a = self.links_by_name[link_a]
        link_info_b = self.links_by_name[link_b]
        Twa = get_link_tf(self.urdf, link_info_a[0])
        Twb = get_link_tf(self.urdf, link_info_b[0])
        Tab = np.linalg.inv(Twa) @ Twb
        return Tab

    def get_gripper_pcl(self, tf_to_palm): 
        gripper_pcl = []

        link_tf = {}
        # statically initialize TF from world -> palm
        link_tf["l_palm"] = tf_to_palm 
        # apply transforms for downstream joints
        links = list(self.gripper_pkl['points'].keys())[10:22]

        for link in links:
            link = str(link)
            link_info = self.links_by_name[link]
            parent_idx = link_info[16]
            parent_link = p.getJointInfo(self.urdf, parent_idx)[12].decode("ascii")
            if(parent_link not in link_tf):
                print("ERROR")
            parent_tf = link_tf[parent_link]
            tf = parent_tf @ self.get_tf(parent_link, link )
            link_tf[link] = tf

        links.insert(0, "l_palm")
        for link in links: 
            pts = np.array(self.gripper_pkl['points'][link]).T
            pts = np.vstack((pts, np.ones(pts.shape[1]))) 
            tf = link_tf[link] #get_link_tf(victor.urdf, victor.links_by_name[link][0])
            pts_tf = tf @ pts
            gripper_pcl.append(pts_tf.T)
        gripper_pcl = np.vstack(gripper_pcl)
        gripper_pcl = gripper_pcl[:, 0:3]
        return gripper_pcl

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

    def psuedoinv_ik_controller(self, side, x_prime, noise=0):
        J = self.get_arm_jacobian(side)
        lmda = 0.0000001
        J_pinv = np.linalg.inv(J.T @ J + lmda * np.eye(7)) @ J.T
        q_prime = J_pinv @ x_prime
        q_prime += np.random.normal(scale=noise, size=q_prime.shape)

        # control
        joint_list = self.left_arm_joints if (side == "left") else right_arm_joints
        p.setJointMotorControlArray(self.urdf, joint_list, p.VELOCITY_CONTROL, targetVelocities=q_prime)

    def set_velo(self, targetVelo):
        joint_list = self.left_arm_joints
        p.setJointMotorControlArray(self.urdf, joint_list, p.VELOCITY_CONTROL, targetVelocities=targetVelo)

    def get_tag_geometry(self):
        tag_ids = None
        # corners go tl -> tr -> br -> bl
        tag_geometry = [
            # Tag 1
            np.array([
                [0.02559, 0.09058, 0.02154],
                [-0.02493, 0.09058, 0.02154],
                [-0.02493, 0.09058, -0.02897],
                [0.02559, 0.09058, -0.02897],
            ], dtype=np.float32),
            # Tag 2
            np.array([
                [0.082, 0.04531, 0.02145],
                [0.0463, 0.08103, 0.02145],
                [0.0463, 0.08103, -0.02906],
                [0.082, 0.04531, -0.02906],
            ], dtype=np.float32),
            # Tag 3
            np.array([
                [0.08985, -0.02565, 0.02123], 
                [0.08985, 0.02486, 0.02123], 
                [0.08985, 0.02486, -0.02928], 
                [0.08985, -0.02565, -0.02928], 
            ], dtype=np.float32),
            # Tag 4 
            np.array([
                [0.04547, -0.082, 0.02158], 
                [0.08114, -0.04624, 0.02158], 
                [0.08114, -0.04624, -0.02894], 
                [0.04547, -0.082, -0.02894], 
            ], dtype=np.float32),
            np.array([
                [-0.02653, -0.09164, 0.0224],
                [0.02394, -0.09164, 0.0224],
                [0.02394, -0.09164, -0.02827],
                [-0.02653, -0.09164, -0.02827],
            ], dtype=np.float32),
            np.array([
                [-0.08257, -0.04674, 0.02168],
                [-0.04685, -0.08245, 0.02168],
                [-0.04685, -0.08245, -0.02883],
                [-0.08257, -0.04674, -0.02883],
            ], dtype=np.float32),
            np.array([
                [-0.09103, 0.02497, 0.02202], 
                [-0.09103, -0.02554, 0.02202],
                [-0.09103, -0.02554, -0.02849],
                [-0.09103, 0.02497, -0.02849], 
            ], dtype=np.float32),
            np.array([
                [-0.04628, 0.08114, 0.02149], 
                [-0.08201, 0.04541, 0.02149],
                [-0.08201, 0.04541, -0.02902],
                [-0.04628, 0.08114, -0.02902], 
            ], dtype=np.float32)
        ]
        Twm = get_link_tf(self.urdf, 17)
        for t_idx, tag in enumerate(tag_geometry):
            for p_idx, pt in enumerate(tag):
                x = pt[0] 
                y = pt[1] 
                z = pt[2]
                tag_geometry[t_idx][p_idx] = np.array([x, z, -y])

        #for tag in tag_geometry:
        #    for pt in tag:
        #        pt_aug = np.array([pt[0], pt[1], pt[2], 1])
        #        Q = Twm @ pt_aug
        #        draw_pose(Q[0:3], np.eye(3), mat=True, axis_len=0.02)
        return np.arange(1,9), tag_geometry
    
