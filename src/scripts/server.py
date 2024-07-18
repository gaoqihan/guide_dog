import rospy
import actionlib
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
from sensor_msgs.msg import Image
from guide_dog.msg import ObjectDetectorAction, ObjDetectorFeedback, ObjDetectorResult  # Update with your actual action package and message names
from std_srvs.srv import Empty

import subprocess  # Import subprocess module
import sys  # Import sys module

# Add the '../include' directory to sys.path to import modules from there
script_dir = os.path.dirname(__file__)  # Get the directory where the script is located
include_dir = os.path.join(script_dir, '../include')  # Path to the 'include' directory
sys.path.append(include_dir)  # Add 'include' directory to sys.path

# Now you can import modules from the 'include' directory
from user_input_manager import UserInputManager, UserInput, UserAudio, UserVideo, UserRGBDSet
from gpt_caller import GPTCaller
from seg_any import SegAny


class UserInputManagerServer(object):
    def __init__(self):
        self.action_server = actionlib.SimpleActionServer('video_capture_action', ObjectDetectorAction, self.execute, False)
        self.action_server.start()
        self.manager=UserInputManager()
        self.seg_any=SegAny()
        print("Server started")
        
    def execute(self, goal):
        print(goal.task)
        #rospy.loginfo("Video capture request received, processing...")
        feedback = ObjDetectorFeedback()
        result = ObjDetectorResult()
        #pause video capture
        #TBD call python script pause_resume_video.py p 
        subprocess.call([sys.executable, 'scripts/pause_resume_video.py', 'p'])  # Use sys.executable
        #create RGBDSet input
        rgbd_set=UserRGBDSet("./tmp")
        self.manager.add_new_input(rgbd_set)


        print(self.manager.get_most_recent_input().id)

        #distill the task thruough gpt
        #assume done
        owl_keyword=["chair"]
        gpt_keyword="which chair is empty?"

        #get bounding boxes through owl
        self.manager.detect_objects(rgbd_set.id,owl_keyword)

        #choose the bounding box through gpt
        caller=GPTCaller()
        system_prompt="You are an AI assistant that can help with identifiying requested item in an image. The options will be included in bounding boxes with an index. Pick the bounding box that contains requested item.return nothing but the number. Return -1 if not found."
        user_prompt="Task is : which chair is occupied by a person."
        for i in range(len(rgbd_set.request["chair"])):
            print(i)
            if len(rgbd_set.request["chair"][i]["boxes"])==0:
                print(f"no bounding box in frame {i}")
                continue
            caller.create_prompt([user_prompt,rgbd_set.request["chair"][0]["image"]],system_prompt_list=[system_prompt])
            response=caller.call()
            print(f"gpt selected bounding box is {response}")
            if response==-1:
                print(f"target not found in frame {i}")
                continue
            x1, y1, x2, y2 = tuple(rgbd_set.request["chair"][i]["boxes"][int(response)])
            #print(x1,y1,x2,y2)
            cropped_image=rgbd_set.data[i][0].crop((x1,y1,x2,y2))
            #get segmentation mask
            self.seg_any.encode(cropped_image)
            mask=self.seg_any.get_mask()
            print(type(mask))
            print(mask.shape)

        #get depth point

        #resume video capture
        
        
        result.success = True
        result.message = "Completed"
        self.action_server.set_succeeded(result)
    
if __name__ == '__main__':
    rospy.init_node('detector_server')
    server = UserInputManagerServer()
    rospy.spin()