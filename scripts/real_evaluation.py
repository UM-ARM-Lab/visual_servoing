import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import cv2

result = pkl.load(open("test-results/20221020-142233/result.pkl", "rb"))

# Get the mocap camera -> eef pose estimates

#data = {
#        "T[mocap_zed_base]_[mocap_val_braclet]" : [],
#        "T[zed2i_left_optical]_[left_tool]" : [],
#        "T[mocap_zed_base]_[mocap_tag]" : np.array([]),
#        "T[zed2i_left_optical]_[target]" : np.array([]),
#        "T[zed_base]_[zed2i_left_optical]" : Tzedbase_leftoptical,
#        "T[target]_[target_adj]" : T_offset,
#        "T[bracelet]_[left_tool]" : Tbe,
#        # [mocap_tag] = [target]
#} 
Tbt = result["T[bracelet]_[left_tool]"]
Tzedbase_zedoptical = result["T[zed_base]_[zed2i_left_optical]"]
Ttarget_targetadj = result["T[target]_[target_adj]"]

# Get data into the same frames
gt_cam_to_tool = [np.linalg.inv(Tzedbase_zedoptical) @ Tcb @ Tbt for Tcb in result["T[mocap_zed_base]_[mocap_val_braclet]"] ]
aruco_cam_to_tool = result["T[zed2i_left_optical]_[left_tool]"]

gt_cam_to_target_adj = np.linalg.inv(Tzedbase_zedoptical) @ result["T[mocap_zed_base]_[mocap_tag]"] @ Ttarget_targetadj 
aruco_cam_to_target_adj = result["T[zed2i_left_optical]_[target]"] @ Ttarget_targetadj
print(f"target measured vs actual pos error {np.linalg.norm(gt_cam_to_target_adj[0:3, 3] - aruco_cam_to_target_adj[0:3, 3])}")

# Plot tool pose estimation error
tool_pos_error = []
tool_rot_error = []
for gt, measured in zip(gt_cam_to_tool, aruco_cam_to_tool):
    tool_pos_error.append( np.linalg.norm(gt[0:3, 3] - measured[0:3, 3]))
    rot_error, _ = cv2.Rodrigues(gt[0:3, 0:3] @ np.linalg.inv(measured)[0:3, 0:3])
    tool_rot_error.append(180/np.pi * np.linalg.norm(rot_error))
tool_error_fig, (tool_pos_error_ax, tool_rot_error_ax) = plt.subplots(2, 1)
tool_pos_error_ax.set_xlabel("iteration")
tool_pos_error_ax.set_ylabel("error (m)")
tool_pos_error_ax.plot(tool_pos_error)
tool_pos_error_ax.set_title("Tool position estimate error (mocap vs aruco)")
tool_rot_error_ax.set_xlabel("iteration")
tool_rot_error_ax.set_ylabel("error (deg)")
tool_rot_error_ax.plot(tool_rot_error)
tool_rot_error_ax.set_title("Tool rotation estimate error (mocap vs aruco)")

# Plot gripper vs target over iter
gt_tool_to_target_pos_error = []
gt_tool_to_target_rot_error = []
for tool in gt_cam_to_tool:
    gt_tool_to_target = tool @ np.linalg.inv(gt_cam_to_target_adj)
    gt_tool_to_target_pos_error.append( np.linalg.norm(gt_tool_to_target[0:3, 3]))
    rvec, _ = cv2.Rodrigues(gt_tool_to_target[0:3, 0:3])
    gt_tool_to_target_rot_error.append( 180/np.pi * np.linalg.norm(rvec))
tool_target_error_fig, (gt_tool_to_target_pos_error_ax, gt_tool_to_target_rot_error_ax) = plt.subplots(2, 1)
gt_tool_to_target_pos_error_ax.set_xlabel("iteration")
gt_tool_to_target_pos_error_ax.set_ylabel("error (m)")
gt_tool_to_target_pos_error_ax.set_title("position error to target (mocap)")
gt_tool_to_target_pos_error_ax.plot(gt_tool_to_target_pos_error)
gt_tool_to_target_rot_error_ax.set_xlabel("iteration")
gt_tool_to_target_rot_error_ax.set_ylabel("error (deg)")
gt_tool_to_target_rot_error_ax.set_title("rotation error to target (mocap)")
gt_tool_to_target_rot_error_ax.plot(gt_tool_to_target_rot_error)

#
#aruco_tool_target_error = []
#for tool in aruco_cam_to_tool:
#    tool_to_target = tool @ np.linalg.inv(aruco_cam_to_target_adj)
#    aruco_tool_target_error.append( np.linalg.norm(tool_to_target[0:3, 3]))
#aruco_tool_target_error_ax.plot(aruco_tool_target_error)
#aruco_tool_target_error_ax.set_xlabel("iteration")
#aruco_tool_target_error_ax.set_ylabel("error (m)")
#gt_tool_target_error_ax.set_title("position error to target (aruco)")

plt.show()