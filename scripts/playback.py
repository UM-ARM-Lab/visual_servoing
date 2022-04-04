import pickle as pkl
import rospy
from sensor_msgs.msg import JointState
import tf_conversions
import tf2_ros
import geometry_msgs.msg

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

for i, joint_config in enumerate(result["traj0"]["joint_config"]):
    # Publish joint configs
    for key in joint_config:
        msg = JointState()
        msg.name.append(key)
        msg.position.append(joint_config[key])
        pub.publish(msg)

    # Publish TF from world to camera 
    world_to_camera = result["traj0"]['camera_to_world']
    publish_tf(world_to_camera, "world", "camera", True)

    # Publish TF from camera to estimated EEF
    est_eef_pose = result["traj0"]['est_eef_pose'][i]
    publish_tf(est_eef_pose, "world", "est_eef_pose")

    # Publish estimated EEF TF 
    rate.sleep()
