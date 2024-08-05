from sklearn.cluster import KMeans
import rospy
import actionlib
import cv2
from cv_bridge import CvBridge
import numpy as np
import os
from sensor_msgs.msg import Image,CameraInfo
from guide_dog.msg import ObjectDetectorAction, ObjDetectorFeedback, ObjDetectorResult  # Update with your actual action package and message names
from std_srvs.srv import Empty, EmptyResponse  # Import the Empty service type and its response
import torch
import subprocess  # Import subprocess module
import sys  # Import sys module
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from scipy import stats
import json
from time import time


# Add the '../include' directory to sys.path to import modules from there
script_dir = os.path.dirname(__file__)  # Get the directory where the script is located
include_dir = os.path.join(script_dir, '../include')  # Path to the 'include' directory
sys.path.append(include_dir)  # Add 'include' directory to sys.path

# Now you can import modules from the 'include' directory
from user_input_manager import UserInputManager, UserInput, UserAudio, UserVideo, UserRGBDSet
from gpt_caller import GPTCaller
from seg_any import SegAny
from depth_to_3d import get3d
from utils import extract_number_from_brackets

class UserInputManagerServer(object):
    def __init__(self):
        self.action_server = actionlib.SimpleActionServer('visual_locator_action', ObjectDetectorAction, self.execute, False)
        self.action_server.start()
        print("action server started")
        rospy.Subscriber("/camera/aligned_depth_to_color/camera_info", CameraInfo, self.get_info)

        
        self.manager=UserInputManager(model="nano")
        print("UserInputManager Initialized")
        self.seg_any=SegAny(model="nano")
        print("Server started")
    def get_info(self,data):
        global info
        info = data
    def execute(self, goal):
        print(goal.type)
        if goal.type=="audio":
            audio_input=UserAudio(goal.file_path)
            self.manager.add_new_input(audio_input)
            goal_text=self.manager.transcribe_audio(audio_input.id)
            self.task=goal_text
            self.excute_text()
        elif goal.type=="text":
            self.task=goal.task
            self.excute_text()
        
        
    def excute_text(self):
        print(self.task)
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
        caller=GPTCaller()

        #distill the task thruough gpt

        with open('./prompts/distillation/system_prompt', 'r') as file:
            system_prompt = file.read()
        user_prompt=self.task
        caller.create_prompt([user_prompt],system_prompt_list=[system_prompt])
        response=caller.call()
        result_dict = json.loads(response)

        #assume done
        owl_keyword=result_dict["keyword"]
        gpt_keyword=result_dict["key instruction"]

        print(owl_keyword,gpt_keyword)
        #get bounding boxes through owl
        self.manager.detect_objects(rgbd_set.id,owl_keyword)
        #choose the bounding box through gpt
        system_prompt="You are an AI assistant that can help with identifiying requested item in an image. The options will be included in bounding boxes with a number on the top left corner. Pick the bounding box that contains requested item by anwsering the number. reason about each bounding box on why or why not it is selected, then return your selected index of bounding box in []. if none of the bounding box is desirable, answer shoueld be [-1]" #return nothing but the number. Return -1 if not found."
        user_prompt=f"Task is : {gpt_keyword}"

        stacked_world_points=[]
        if os.path.exists("./tmp/cropped_depth"):
            for item in os.listdir("./tmp/cropped_depth"):
                item_path = os.path.join("./tmp/cropped_depth", item)
                try:
                    os.remove(item_path)  # Remove files and links
                    print(f"Deleted {item_path}")
                except Exception as e:
                    print(f"Failed to delete {item_path}. Reason: {e}")
        x1,x2,y1,y2=None,None,None,None
        for i in range(len(rgbd_set.request[owl_keyword[0]])):
            print(i)
            if x1==None:
                start_time=time()
                if len(rgbd_set.request[owl_keyword[0]][i]["boxes"])==0:
                    print(f"no bounding box in frame {i}")
                    continue
                selection_range="choose from following numbers"+str(range(len(rgbd_set.request[owl_keyword[0]][i]["boxes"])))
                caller.create_prompt([user_prompt,selection_range,rgbd_set.request[owl_keyword[0]][i]["image"]],system_prompt_list=[system_prompt])
    
                response=caller.call()
                print(f"gpt response is {response}")
                response=extract_number_from_brackets(response)
                
                
                print("time for gpt call is: ", time()-start_time)
                #response=0
                print(f"gpt selected bounding box is {response}")
                if int(response)==-1:
                    print(f"target not found in frame {i}")
                    continue

                try:
                    x1, y1, x2, y2 = tuple(rgbd_set.request[owl_keyword[0]][i]["boxes"][int(response)])
                except:
                    print(f"gpt error in frame {i}")
                    continue
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                # Cap the x2 and y2 values at the image's width and height
                x1 = max(x1, 0)
                y1 = max(y1, 0)
                x2 = min(x2, rgbd_set.data[i][0].width)
                y2 = min(y2, rgbd_set.data[i][0].height)
            cropped_image=rgbd_set.data[i][0].crop((x1,y1,x2,y2))
            print(f"cropped image size is {cropped_image.size}") 
            #get segmentation mask
            self.seg_any.encode(cropped_image)

            mask=self.seg_any.get_mask()
            #save the masked cropped image in /tem/masked
            os.makedirs("./tmp/masked", exist_ok=True)
            self.seg_any.get_mask_image(f"./tmp/masked/{str(i)}.png")
            number_of_true = np.sum(mask)
 
            #get depth point
            image=rgbd_set.data[i][1]

            sum_check=np.sum(image[y1:y2,x1:x2], axis=(0, 1))
            world_points=get3d(image,(x1,y1,x2,y2),info,i)
            masked_world_points=mask[:,:,np.newaxis]*world_points
            sum_result=np.sum(masked_world_points, axis=(0, 1))
            world_point_mean=sum_result/number_of_true
            print(f"the {owl_keyword} is at {world_point_mean}")
            stacked_world_points.append(world_point_mean)
            #combine with the dog's pose to get world coordinante
            print(f"frame {i} took {time()-start_time} seconds")
        if len(stacked_world_points)>1:    
            stacked_world_points=np.array(stacked_world_points)
            #remove outliers
            # Calculate Z-scores
            z_scores = np.abs(stats.zscore(stacked_world_points, axis=0))

            # Set a threshold (e.g., 3) for identifying outliers
            threshold = 2

            # Remove outliers
            non_outliers = (z_scores < threshold).all(axis=1)
            stacked_world_points = stacked_world_points[non_outliers]



        if len(stacked_world_points)>1:
            kmeans = KMeans(n_clusters=2)
            kmeans.fit(stacked_world_points)
            labels = kmeans.labels_

            # Step 4: Compute the mean point for each cluster
            means = np.array([stacked_world_points[labels == i].mean(axis=0) for i in range(2)])
            print(f"more than one point detected")
            for i in range(2):
                print(f"Number of points in cluster {i}: {np.sum(labels == i)}")
                print(f"Mean of cluster {i}: {means[i]}")
        elif len(stacked_world_points)==1:
            print(f"one point detected: {stacked_world_points[0]}")
        else:
            print("no point detected")

        #resume video capture
        subprocess.call([sys.executable, 'scripts/pause_resume_video.py', 'r'])
        #print("finish",torch.cuda.mem_get_info())

        result.success = True
        result.message = "Completed"
        self.action_server.set_succeeded(result)
    
if __name__ == '__main__':
    rospy.init_node('detector_server')
    server = UserInputManagerServer()
    rospy.spin()