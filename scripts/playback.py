import pickle as pkl
import rospy
from sensor_msgs.msg import JointState

rospy.init_node("playback", anonymous=True)
pub = rospy.Publisher('arm_joint_states', JointState, queue_size=10 )
rate = rospy.Rate(10)
result = pkl.load(open("test-results/result20220331-182346.pkl", "rb"))

for joint_config in result["traj0"]["joint_config"]:
    for key in joint_config:
        msg = JointState()
        msg.name.append(key)
        msg.position.append(joint_config[key])
        pub.publish(msg)
        print(msg)
    rate.sleep()
