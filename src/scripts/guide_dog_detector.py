import rospy
import actionlib
from guide_dog.msg import ObjectDetectorAction, ObjectDetectorGoal
from std_msgs.msg import String
import sys

def feedback_cb(feedback):
    print(f"Progress: {feedback.progress}")

def callback(msg):
    # Initializes the action client node

    # Creates the client, specifying the action type
    client = actionlib.SimpleActionClient('visual_locator_action', ObjectDetectorAction)

    # Waits until the action server has started up and started listening for goals
    client.wait_for_server()

    # Creates a goal to send to the action server
    goal = ObjectDetectorGoal()
    #goal.task="find washroom sign"
    goal.task=msg.data
    #goal.type="audio"
    #goal.file_path="/root/catkin_ws/src/guide_dog/src/tmp/recording.wav"
    goal.type="text"
    print("goal created")
    # Sends the goal to the action server, specifying which feedback function to call when feedback received
    client.send_goal(goal, feedback_cb=feedback_cb)
    print("goal sent")

    # Waits for the server to finish performing the action
    client.wait_for_result()
    
    # Prints out the result of executing the action
    result= client.get_result()  # A VideoCaptureResult
    

if __name__ == '__main__':
    if not rospy.is_shutdown():
        rospy.init_node('visual_locator_client')
        subscriber=rospy.Subscriber('/find_object', String,callback)
        rospy.spin()
