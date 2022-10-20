import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from pathlib import Path
from typing import Tuple, List

class Trial:
    """
    Parse results of a single servoing trial
    """
    def __init__(self, dir : Path):
        result = pkl.load(open(dir, "rb"))

        # Get mocap and aruco data into the same frames
        Tbt = result["T[bracelet]_[left_tool]"]
        Tzedbase_zedoptical = result["T[zed_base]_[zed2i_left_optical]"]
        Ttarget_targetadj = result["T[target]_[target_adj]"]

        self.gt_cam_to_tool = [np.linalg.inv(Tzedbase_zedoptical) @ Tcb @ Tbt for Tcb in result["T[mocap_zed_base]_[mocap_val_braclet]"] ]
        self.aruco_cam_to_tool = result["T[zed2i_left_optical]_[left_tool]"]

        self.gt_cam_to_target_adj = np.linalg.inv(Tzedbase_zedoptical) @ result["T[mocap_zed_base]_[mocap_tag]"] @ Ttarget_targetadj 
        self.aruco_cam_to_target_adj = result["T[zed2i_left_optical]_[target]"] @ Ttarget_targetadj
    
    def get_target_pose_error(self) -> Tuple[float, float]:
        """
        Returns the position and rotation error of the target
        error is defined as difference between aruco and mocap
        """
        pos_error = np.linalg.norm(self.gt_cam_to_target_adj[0:3, 3] - self.aruco_cam_to_target_adj[0:3, 3])
        rvec, _ = cv2.Rodrigues(self.gt_cam_to_target_adj[0:3, 0:3] @ np.linalg.inv(self.aruco_cam_to_target_adj)[0:3, 0:3])
        rot_error = 180/np.pi * np.linalg.norm(rvec)
        return pos_error, rot_error
    
    def get_tool_pose_error(self) -> Tuple[List[float], List[float]]:
        """
        Returns the position and rotation error of the tool
        error is defined as difference between aruco and mocap
        """
        tool_pos_error = []
        tool_rot_error = []
        for gt, measured in zip(self.gt_cam_to_tool, self.aruco_cam_to_tool):
            tool_pos_error.append( np.linalg.norm(gt[0:3, 3] - measured[0:3, 3]))
            rot_error, _ = cv2.Rodrigues(gt[0:3, 0:3] @ np.linalg.inv(measured)[0:3, 0:3])
            tool_rot_error.append(180/np.pi * np.linalg.norm(rot_error))
        return tool_pos_error, tool_rot_error
    
    def get_gripper_vs_target_error(self) -> Tuple[List[float], List[float]]:
        """
        Returns the position and rotation error of the tool relative to the target
        according to mocap
        """
        gt_tool_to_target_pos_error = []
        gt_tool_to_target_rot_error = []
        for tool in self.gt_cam_to_tool:
            gt_tool_to_target = tool @ np.linalg.inv(self.gt_cam_to_target_adj)
            gt_tool_to_target_pos_error.append( np.linalg.norm(gt_tool_to_target[0:3, 3]))
            rvec, _ = cv2.Rodrigues(gt_tool_to_target[0:3, 0:3])
            gt_tool_to_target_rot_error.append( 180/np.pi * np.linalg.norm(rvec))
        return gt_tool_to_target_pos_error, gt_tool_to_target_rot_error


class ResultGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)

        # Log file select button
        self.select_log_button = QPushButton("Select Log")
        self.select_log_button.clicked.connect(self.select_log_click)
        self.layout.addWidget(self.select_log_button)
        
        # File select dialog
        self.file_selector = QFileDialog()
        self.file_selector.setFileMode(QFileDialog.AnyFile)
        self.file_selector.setDirectory("/home/ashwin/catkin_ws/src/Val-Visual-Servo/test-results")

        # Trajectory chooser combo box
        self.traj_choice = QComboBox()
        self.traj_choice.currentIndexChanged.connect(self.traj_select)
        self.layout.addWidget(self.traj_choice)


    # Callback for file open button press
    def select_log_click(self):
        self.file_selector.exec_()
        filename = self.file_selector.selectedFiles()
        self.load_result_file(str(filename[0]))
    
    # Callback for trajectory selection
    def traj_select(self, i):
        traj_len = len(self.result['traj'][i]['joint_config'])
        self.traj_slider.setMinimum(0)
        self.traj_slider.setMaximum(traj_len-1)
        self.traj_slider.setValue(0)
        self.compute_traj_metrics(i)
        self.publish_playback_state(i, 0)
    
    # Callback for slider change
    def slider_change(self):
        self.publish_playback_state(self.traj_choice.currentIndex(), self.traj_slider.value())

    # Loads result file
    def load_result_file(self, filename):
        self.result = pkl.load(open(filename, "rb"))
        self.num_traj = len(self.result['traj'])
        for i in range(self.num_traj):
            self.traj_choice.addItem(f"traj {i}")
        self.compute_global_metrics()

if __name__ == "__main__":
    #app = QApplication([])
    #gui = ResultGUI()
    #gui.show()
    #app.exec()

    # Plot tool pose estimation error
    t = Trial("test-results/20221020-142233/result.pkl")

    print(t.get_target_pose_error())
    tool_error_fig, (tool_pos_error_ax, tool_rot_error_ax) = plt.subplots(2, 1)
    tool_pos_error, tool_rot_error = t.get_tool_pose_error()
    tool_pos_error_ax.set_xlabel("iteration")
    tool_pos_error_ax.set_ylabel("error (m)")
    tool_pos_error_ax.plot(tool_pos_error)
    tool_pos_error_ax.set_title("Tool position estimate error (mocap vs aruco)")
    tool_rot_error_ax.set_xlabel("iteration")
    tool_rot_error_ax.set_ylabel("error (deg)")
    tool_rot_error_ax.plot(tool_rot_error)
    tool_rot_error_ax.set_title("Tool rotation estimate error (mocap vs aruco)")

    # Plot gripper vs target over iter
    gt_tool_to_target_pos_error, gt_tool_to_target_rot_error = t.get_gripper_vs_target_error()
    tool_target_error_fig, (gt_tool_to_target_pos_error_ax, gt_tool_to_target_rot_error_ax) = plt.subplots(2, 1)
    gt_tool_to_target_pos_error_ax.set_xlabel("iteration")
    gt_tool_to_target_pos_error_ax.set_ylabel("error (m)")
    gt_tool_to_target_pos_error_ax.set_title("position error to target (mocap)")
    gt_tool_to_target_pos_error_ax.plot(gt_tool_to_target_pos_error)
    gt_tool_to_target_rot_error_ax.set_xlabel("iteration")
    gt_tool_to_target_rot_error_ax.set_ylabel("error (deg)")
    gt_tool_to_target_rot_error_ax.set_title("rotation error to target (mocap)")
    gt_tool_to_target_rot_error_ax.plot(gt_tool_to_target_rot_error)

    plt.show()
