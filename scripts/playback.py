import pickle as pkl
import rospy
from sensor_msgs.msg import JointState
import tf_conversions
import tf2_ros
import geometry_msgs.msg
import matplotlib.pyplot as plt
import numpy as np
import cv2

def publish_tf(tf, ref_frame, frame, static=False):
    if(static):
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

rospy.init_node("playback", anonymous=True)
pub = rospy.Publisher('arm_joint_states', JointState, queue_size=10 )
rate = rospy.Rate(10)
result = pkl.load(open("test-results/20220404-122502/result.pkl", "rb"))

fig, (ax1, ax2) = plt.subplots(2, 1)
iterations = len(result["traj0"]["joint_config"])
pos_error = []
rot_error = []

for i, joint_config in enumerate(result["traj0"]["joint_config"]):
    #plt.clf()
    # Publish joint configs
    for key in joint_config:
        msg = JointState()
        msg.name.append(key)
        msg.position.append(joint_config[key])
        pub.publish(msg)

    # Publish TF from world to camera 
    world_to_camera = result["traj0"]['camera_to_world']
    publish_tf(world_to_camera, "world", "camera", True)

    # Publish TF from world to estimated EEF
    est_eef_pose = result["traj0"]['est_eef_pose'][i]
    gt_eef_pose = result["traj0"]['gt_eef_pose'][i]
    publish_tf(est_eef_pose, "world", "est_eef_pose")

    # Compute error and plot
    pos_error.append(np.linalg.norm(est_eef_pose[0:3, 3] - gt_eef_pose[0:3, 3]))
    link_rod, _ = cv2.Rodrigues(est_eef_pose[0:3, 0:3] @ gt_eef_pose[0:3, 0:3].T)
    rot_error.append(np.linalg.norm(link_rod))
    #ax1.plot(pos_error)
    #ax2.plot(rot_error)
    ax1.scatter(i, pos_error[-1], c='g')
    ax2.scatter(i, rot_error[-1], c='r')
    #plt.show()
    plt.pause(0.01)

    # Publish estimated EEF TF 
    rate.sleep()
