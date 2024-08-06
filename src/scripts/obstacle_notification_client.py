import rospy
import actionlib
from actionlib.msg import TestAction, TestGoal
import subprocess

def feedback_cb(feedback):
    rospy.loginfo(f"Feedback received: {feedback.feedback}")

def send_goal():
    # Initialize the ROS node
    rospy.init_node('obstacle_notification_client')

    # Create an action client
    client = actionlib.SimpleActionClient('obstacle_notification', TestAction)

    # Wait for the action server to start
    rospy.loginfo("Waiting for action server to start...")
    client.wait_for_server()

    # Create a goal to send to the action server
    goal = TestGoal()
    goal.goal = 1

    # Send the goal to the action server
    rospy.loginfo("Sending goal to action server...")
    client.send_goal(goal, feedback_cb=feedback_cb)

    # Wait for the result
    client.wait_for_result()

    # Print the result
    result = client.get_result()

    rospy.loginfo(f"Result received: {result.result}")

if __name__ == '__main__':
    try:
        send_goal()
    except rospy.ROSInterruptException:
        pass