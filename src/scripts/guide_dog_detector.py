import rospy
import actionlib
from guide_dog.msg import ObjectDetectorAction, ObjectDetectorGoal
import sys

def feedback_cb(feedback):
    print(f"Progress: {feedback.progress}")

def visual_locator_client():
    # Initializes the action client node
    rospy.init_node('visual_locator_client')

    # Creates the client, specifying the action type
    client = actionlib.SimpleActionClient('visual_locator_action', ObjectDetectorAction)

    # Waits until the action server has started up and started listening for goals
    client.wait_for_server()

    # Creates a goal to send to the action server
    goal = ObjectDetectorGoal()
    goal.task="find the closest place for me to sit down"
    goal.type="text"
    #goal.file_path="/root/catkin_ws/src/guide_dog/src/tmp/recording.wav"
    print("goal created")
    # Sends the goal to the action server, specifying which feedback function to call when feedback received
    client.send_goal(goal, feedback_cb=feedback_cb)
    print("goal sent")

    # Waits for the server to finish performing the action
    client.wait_for_result()
    
    # Prints out the result of executing the action
    return client.get_result()  # A VideoCaptureResult

if __name__ == '__main__':
    try:
        result = visual_locator_client()
        print(f"Result: {result.success}, {result.message}")
    except rospy.ROSInterruptException:
        print("Program interrupted before completion", file=sys.stderr)