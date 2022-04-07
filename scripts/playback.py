import pickle as pkl
import rospy
from sensor_msgs.msg import JointState
import tf_conversions
import tf2_ros
import geometry_msgs.msg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


# publishes a homogenous TF to the TF2 tree
def publish_tf(tf, ref_frame, frame, static=False):
    if(not static):
        br = tf2_ros.TransformBroadcaster()
    else:
        br = tf2_ros.StaticTransformBroadcaster()
    t = geometry_msgs.msg.TransformStamped()
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = ref_frame
    t.child_frame_id = frame
    t.transform.translation.x = tf[0, 3] 
    t.transform.translation.y = tf[1, 3]
    t.transform.translation.z = tf[2, 3]
    quat = tf_conversions.transformations.quaternion_from_matrix(tf)
    t.transform.rotation.x = quat[0]
    t.transform.rotation.y = quat[1]
    t.transform.rotation.z = quat[2]
    t.transform.rotation.w = quat[3]
    br.sendTransform(t)

class GUI(QWidget):
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

        # Idx slider
        self.traj_slider = QSlider(Qt.Horizontal)
        self.traj_slider.setValue(0)
        self.traj_slider.setTickPosition(QSlider.TicksBelow)
        self.traj_slider.setTickInterval(1)
        self.traj_slider.valueChanged.connect(self.slider_change)
        self.layout.addWidget(self.traj_slider)

        # ROS stuff
        rospy.init_node("playback", anonymous=True)
        self.joint_state_pub = rospy.Publisher('arm_joint_states', JointState, queue_size=10 )
        self.model_sdf = pkl.load(open("points_and_sdf.pkl", "rb"))

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
    
    def compute_traj_metrics(self, traj):
        plt.close("all")
        fig, (self.ax1, self.ax2) = plt.subplots(2, 1)
        self.ax1.set_xlabel("iteration")
        self.ax2.set_xlabel("iteration")
        self.ax1.set_ylabel("error (m)")
        self.ax2.set_ylabel("error (rad)")
        self.pos_error = []
        self.rot_error = []
        res = self.result['traj'][traj]
        for idx in range(len(res['est_eef_pose'])):
            # Publish TF from world to camera 
            world_to_camera = res['camera_to_world']

            # Publish TF from world to estimated EEF
            est_eef_pose = res['est_eef_pose'][idx]
            gt_eef_pose = res['gt_eef_pose'][idx]

            # Compute error and plot
            self.pos_error.append(np.linalg.norm(est_eef_pose[0:3, 3] - gt_eef_pose[0:3, 3]))
            link_rod, _ = cv2.Rodrigues(est_eef_pose[0:3, 0:3] @ gt_eef_pose[0:3, 0:3].T)
            self.rot_error.append(np.linalg.norm(link_rod))

        #self.ax1.plot(self.pos_error)
        #self.ax2.plot(self.rot_error)
        #plt.show()

    def publish_playback_state(self, traj, idx):
        res = self.result['traj'][traj]
        # Publish joint configs
        for key in res['joint_config'][idx]:
            msg = JointState()
            msg.name.append(key)
            msg.position.append(res['joint_config'][idx][key])
            self.joint_state_pub.publish(msg)

        # Publish TF from world to camera 
        world_to_camera = res['camera_to_world']
        publish_tf(world_to_camera, "world", "camera", True)

        # Publish TF from world to target 
        world_to_target = res['target_pose']
        publish_tf(world_to_target, "world", "target", True)

        # Publish TF from world to estimated EEF
        est_eef_pose = res['est_eef_pose'][idx]
        gt_eef_pose = res['gt_eef_pose'][idx]
        publish_tf(est_eef_pose, "world", "est_eef_pose", True)

        #self.ax1.clear()
        #self.ax2.clear()
        self.ax1.plot(self.pos_error[0:idx], "r")
        self.ax2.plot(self.rot_error[0:idx], "b")
        plt.pause(0.01)


if __name__ == "__main__":
    app = QApplication([])
    gui = GUI()
    gui.show()
    app.exec()
