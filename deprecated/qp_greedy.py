# Get the PBVS twist for the eef
eef_twist, Twe = pbvs.do_pbvs(rgb, depth, Two, np.eye(4), val.get_arm_jacobian("left", True), val.get_jacobian_pinv("left", True), 24)
# Compute the rotation to algin tag normal to cam look
Twc = np.linalg.inv(camera.get_extrinsics())
camera_look = -Twc[0:3, 2]
tag_normal = Twe[0:3, 2]
camera_rot_cmd =  np.cross(camera_look, tag_normal)
#twist[3:] = dir
# Visualize estimated end effector pose 
if (uids_eef_marker is not None):
    erase_pos(uids_eef_marker)
uids_eef_marker = draw_pose(Twe[0:3, 3], Twe[0:3, 0:3], mat=True)
k = 40 # control gain on target for PBVS
full_jac = val.get_arm_jacobian("left", True)
num_joints = full_jac.shape[1]
# augmented torso jacobian ignoring the arm only joints, 6 x 9, but sparse
torso_twist = np.zeros(6)
torso_twist[3:] = camera_rot_cmd
torso_jac = np.hstack((val.get_camera_jacobian(), np.zeros((6, 7))))
Q = np.eye(6)
# Weigh accurate position more
Q[np.arange(3), np.arange(3)] = 50
Q[np.arange(3, 6), np.arange(3, 6)] = 1
# Don't care about position control of camrea
R = np.eye(6)
R[np.arange(3), np.arange(3)] = 0
R[np.arange(3, 6), np.arange(3, 6)] = 1
wp = 0.999999999
ws = 1 - wp
# Use a low weight objective on the arm for normal alignment?
# Use another objective that accounts for reachability?
P = wp * (full_jac.T @ Q @ full_jac) + ws*(torso_jac.T @ R @ torso_jac)
q = wp*(-k*eef_twist @ Q @ full_jac) + ws*(-k * torso_twist @ R @ torso_jac)
# Inequality for joint vel limits
max_joint_vel = 1.5 # max joint for limit
G = np.vstack((np.eye(num_joints), -np.eye(num_joints)))
h = np.ones(num_joints * 2) * max_joint_vel
ctrl = solve_qp(P, q, G, h, None, None, solver="cvxopt")
print(ctrl)