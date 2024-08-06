import rospy
import actionlib
from actionlib.msg import TestAction, TestGoal, TestResult, TestFeedback
from std_srvs.srv import Empty, EmptyResponse  # Import the Empty service type and its response
import os
import sys
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PoseStamped
import re
import json
script_dir = os.path.dirname(__file__)  # Get the directory where the script is located
include_dir = os.path.join(script_dir, '../include')  # Path to the 'include' directory
sys.path.append(include_dir)  # Add 'include' directory to sys.path
from gpt_caller import GPTCaller
from PIL import Image

class ObstacleNotificationServer(object):
    def __init__(self):
        self.action_server = actionlib.SimpleActionServer('obstacle_notification', TestAction, self.execute, False)
        self.action_server.start()
        print("Action server started")
        print("Server started")
        self.caller=GPTCaller()

    def execute(self, goal):
        # Create feedback and result messages
        feedback = TestFeedback()
        result = TestResult()

        # Read image from file
        image_path = './tmp/color/0.png'
        # Open the image using PIL
        image = Image.open(image_path)
        
        with open('./prompts/obstacle_notification/system_prompt', 'r') as file:
            system_prompt = file.read()
        
        self.caller.create_prompt(system_prompt_list=[system_prompt],user_prompt_list=[image])
        response=self.caller.call()
        
        match = re.search(r'\{.*?\}', response)
        if match:
            json_str = match.group(0)
            response_dict = json.loads(json_str)
            rospy.loginfo(f"Extracted dictionary: {response_dict}")
        else:
            print(response)
            rospy.logwarn("No JSON object found in the response")

        print(response_dict)
        
        # Example feedback and result
        feedback.feedback = "Processing goal"
        result.result = "Goal processed"

        # Publish feedback
        self.action_server.publish_feedback(feedback)


        # Set the action as succeeded
        self.action_server.set_succeeded(result)

if __name__ == '__main__':
    rospy.init_node('obstacle_notification_server')
    server = ObstacleNotificationServer()
    rospy.spin()