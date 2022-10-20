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
        self.datestr = dir.stem

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
        self.file_selector.setFileMode(QFileDialog.ExistingFiles)
        self.file_selector.setDirectory("/home/ashwin/source/lab/catkin_ws/src/visual_servoing/test-results")

        # Trial chooser combo box
        self.trial_choice = QComboBox()
        self.trial_choice.currentIndexChanged.connect(self.trial_select)
        self.layout.addWidget(self.trial_choice)


    # Callback for file open button press
    def select_log_click(self):
        self.file_selector.exec_()
        filenames = self.file_selector.selectedFiles()
        self.trials = [Trial(Path(filename)) for filename in filenames]

        self.compute_metrics()

        for trial in self.trials:
            self.trial_choice.addItem(trial.datestr)
    
    def compute_metrics(self):
        # Error to target final iteration across trajectories
        gripper_v_target_pos_final = []
        gripper_v_target_rot_final = []
        for trial in self.trials:
            pos, rot = trial.get_gripper_vs_target_error()
            gripper_v_target_pos_final.append(pos[-1])
            gripper_v_target_rot_final.append(rot[-1])
        error_to_target_fig, (pos_error_to_target_ax, rot_error_to_target_ax) = plt.subplots(2, 1)
        pos_error_to_target_ax.boxplot(gripper_v_target_pos_final) 
        pos_error_to_target_ax.set_xlabel("iteration")
        pos_error_to_target_ax.set_ylabel("error (m)")
        pos_error_to_target_ax.set_title("Tool position error to target after servoing")

        rot_error_to_target_ax.boxplot(gripper_v_target_rot_final) 
        rot_error_to_target_ax.set_xlabel("iteration")
        rot_error_to_target_ax.set_ylabel("error (deg)")
        rot_error_to_target_ax.set_title("Tool rotation error to target after servoing")
        plt.show()
    
    # Callback for trial selection
    def trial_select(self, i):
        t = self.trials[i]
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

    

if __name__ == "__main__":
    app = QApplication([])
    gui = ResultGUI()
    gui.show()
    app.exec()