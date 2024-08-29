from sklearn.cluster import KMeans
import rospy
import actionlib
import cv2
from cv_bridge import CvBridge
import numpy as np
import os
from sensor_msgs.msg import Image,CameraInfo
from geometry_msgs.msg import PoseStamped, Pose
from guide_dog.msg import ObjectDetectorAction, ObjDetectorFeedback, ObjDetectorResult  # Update with your actual action package and message names
from std_srvs.srv import Empty, EmptyResponse  # Import the Empty service type and its response
import torch
import subprocess  # Import subprocess module
import sys  # Import sys module
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from scipy import stats
import json
from time import time
from std_msgs.msg import Float32, String


# Add the '../include' directory to sys.path to import modules from there
script_dir = os.path.dirname(__file__)  # Get the directory where the script is located
include_dir = os.path.join(script_dir, '../include')  # Path to the 'include' directory
sys.path.append(include_dir)  # Add 'include' directory to sys.path

# Now you can import modules from the 'include' directory
from user_input_manager import UserInputManager, UserInput, UserAudio, UserVideo, UserRGBDSet
from gpt_caller import GPTCaller
from seg_any import SegAny
from depth_to_3d import get3d,MapBridge
from utils import extract_number_from_brackets
from owl_detector import Detector

class UserInputManagerServer(object):
    def __init__(self):
        #self.action_server = actionlib.SimpleActionServer('visual_locator_action', ObjectDetectorAction, self.object_finder, False)
        #self.action_server.start()
        print("action server started")
        rospy.Subscriber("/camera/aligned_depth_to_color/camera_info", CameraInfo, self.get_info)
        rospy.Subscriber('/find_object', String,self.object_finder_text)

        self.goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)
        self.rel_pos_publisher=rospy.Publisher("/rel_pos",Float32,queue_size=10)

        self.map_bridge=MapBridge()
        
        #self.manager=UserInputManager(model="nano")
        #print("UserInputManager Initialized")
        self.detector = Detector(model="nano")
        print("Detector Initialized")
        self.seg_any=SegAny(model="nano")
        print("Server started")
    def get_info(self,data):
        global info
        info = data
        
    def condition_checker(self,prompt):
        caller=GPTCaller()
        

        
    
    #def object_finder(self, goal):

    #    self.task=goal.task
    #   self.object_finder_text()
        
        
    def object_finder_text(self,msg):
        self.task=msg.data
        print(self.task)
        #rospy.loginfo("Video capture request received, processing...")
        #feedback = ObjDetectorFeedback()
        #result = ObjDetectorResult()
        #pause video capture
        subprocess.call([sys.executable, 'scripts/pause_resume_video.py', 'p'])  # Use sys.executable
        #create RGBDSet input
        rgbd_set=UserRGBDSet("./tmp")
        #self.manager.add_new_input(rgbd_set)
        

        caller=GPTCaller()

        #distill the task thruough gpt

        with open('./prompts/distillation/prompt', 'r') as file:
            prompt_json = json.loads(file.read())
        system_prompt=prompt_json["system_prompt"]
        response_format=prompt_json["response_format"]
        user_prompt=self.task
        caller.create_prompt([user_prompt],system_prompt_list=[system_prompt],response_format=response_format)
        response=caller.call(model="gpt-4o-2024-08-06")
        result_dict = json.loads(response)

        #assume done
        owl_keyword=result_dict["keyword"]
        
        gpt_keyword=result_dict["key_instruction"]

        print(owl_keyword,gpt_keyword)
        

        #get bounding boxes through owl
        bbox_list_list,labeled_image_list=rgbd_set.detect_objects(self.detector,owl_keyword)
        
        #image=Image.open("./tmp/color/0.png")

       
        #choose the bounding box through gpt
        #system_prompt="You are an AI assistant that can help with identifiying requested item in an image. The options will be included in at most three \
        #    bounding boxes with different color. You will be provided with a whole image containing bounding boxes, and cropped images corresponding to the bounding boxes. An index number is attached to each bounding box: the bounding box 0 is red, the bounding box 1 is green, and the bounding box 2 is blue. \
        #        Pick the bounding box that contains requested item by anwsering the index number. reason about each bounding box on why or why not it is selected, \
        #            describe the location of the bounding box in the picture, the color of the bounding box, the color of the item inside the box, and the index of the bounding box, then you MUST give a confidence score for picking this bounding box range from 0 to 10 (0 being lowest). return your \
        #                selected index of bounding box in []. if none of the bounding box is desirable, answer shoueld be [-1]" #return nothing but the number. Return -1 if not found."
        
        with open('./prompts/visual_selector/given_points', 'r') as file:
            prompt_json = json.loads(file.read())
        system_prompt=prompt_json["system_prompt"]
        response_format=prompt_json["response_format"]
        user_prompt=f"Task is : {gpt_keyword}"
        prompt_image=[]

        
        stacked_rel_points=[]
        if os.path.exists("./tmp/cropped_depth"):
            for item in os.listdir("./tmp/cropped_depth"):
                item_path = os.path.join("./tmp/cropped_depth", item)
                try:
                    os.remove(item_path)  # Remove files and links
                    print(f"Deleted {item_path}")
                except Exception as e:
                    print(f"Failed to delete {item_path}. Reason: {e}")
        x1,x2,y1,y2=None,None,None,None
        for i in range(rgbd_set.length):
            print(i)
            start_time=time()
            if len(bbox_list_list[i])==0:
                print(f"no bounding box in frame {i}")
                continue
            selection_range="choose from following numbers"+str(range(len(bbox_list_list[i])))
            user_prompt_list=[user_prompt,selection_range,labeled_image_list[i]]
            #test_pic_path=f"./tmp/bbox_image.png"
            #test_pic = Image.open(test_pic_path)
            #test_pic=caller.encode_image(test_pic_path)
            #user_prompt=f"Task is : find an empty chair to sit down"
            #user_prompt_list=[user_prompt,selection_range,test_pic]
            #for j in range(len(prompt_image[i])):
            #    user_prompt_list.append(f"{j}: ")
            #    user_prompt_list.append(prompt_image[i][j])
            caller.create_prompt(user_prompt_list=user_prompt_list,system_prompt_list=[system_prompt],response_format=response_format)
            response=caller.call(model="gpt-4o-2024-08-06")
            print(f"gpt response is {response}")

            response = json.loads(response)["final_decision"]  

            
            print("time for gpt call is: ", time()-start_time)
            #response=0
            print(f"gpt selected bounding box is {response}")
            if int(response)==-1:
                print(f"target not found in frame {i}")
                continue

            try:
                x1, y1, x2, y2 = tuple(bbox_list_list[i][int(response)])
                break
            except:
                print(f"gpt error in frame {i}")
                continue
        
        #TBD Splatter points
        if x1 is None:
            x1,y1,x2,y2=rgbd_set.points_array_detector()
               
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        print(f"bounding box is {x1,y1,x2,y2}")
        # Cap the x2 and y2 values at the image's width and height
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, rgbd_set.data[i][0].width)
        y2 = min(y2, rgbd_set.data[i][0].height)
        cropped_image=rgbd_set.data[i][0].crop((x1,y1,x2,y2))
        #print(f"cropped image size is {cropped_image.size}") 
        
        
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
        rel_points=get3d(image,(x1,y1,x2,y2),info,i)
        masked_rel_points=mask[:,:,np.newaxis]*rel_points
        sum_result=np.sum(masked_rel_points, axis=(0, 1))
        world_point_mean=sum_result/number_of_true
        print(f"the {owl_keyword} is at {world_point_mean}")
        stacked_rel_points.append(world_point_mean)
        print(f"frame {i} took {time()-start_time} seconds")
        
            
        #remove outliers
    
        if len(stacked_rel_points)>1:    
            
            stacked_rel_points=np.array(stacked_rel_points)
            #remove outliers
            # Calculate Z-scores
            z_scores = np.abs(stats.zscore(stacked_rel_points, axis=0))

            # Set a threshold (e.g., 3) for identifying outliers
            threshold = 2

            # Remove outliers
            non_outliers = (z_scores < threshold).all(axis=1)
            stacked_rel_points = stacked_rel_points[non_outliers]

        #get clusters and pick top 1 best relative point
        if len(stacked_rel_points)>1:
            kmeans = KMeans(n_clusters=2)
            kmeans.fit(stacked_rel_points)
            labels = kmeans.labels_

            means = np.array([stacked_rel_points[labels == i].mean(axis=0) for i in range(2)])
            print(f"more than one point detected")
            for i in range(2):
                print(f"Number of points in cluster {i}: {np.sum(labels == i)}")
                print(f"Mean of cluster {i}: {means[i]}")
                
            if np.linalg.norm(means[0] - means[1]) > 0.3:
                if np.sum(labels == 0) > np.sum(labels == 1):
                    best_rel_point=means[0]
                else:
                    best_rel_point=means[1]
            else:
                best_rel_point=means[0]
                    
        elif len(stacked_rel_points)==1:
            print(f"one point detected: {stacked_rel_points[0]}")
            best_rel_point=stacked_rel_points[0]
        else:
            print("no point detected")
            best_rel_point=None
            
        if best_rel_point is not None:
            self.rel_pos_publisher.publish(Float32(best_rel_point[2]))
            
            '''
            object_position_in_map=self.map_bridge.get_object_position_in_map(best_rel_point[0],best_rel_point[1],best_rel_point[2],1)
            print(f"best point is {object_position_in_map}")
            

            self.map_bridge.publish_markers([object_position_in_map])
            pose=PoseStamped()
            pose.header.frame_id = "map"

            pose.pose.position.x = object_position_in_map[0]
            pose.pose.position.y = object_position_in_map[1]
            pose.pose.position.z = object_position_in_map[2]

            pose.pose.orientation.x = 0
            pose.pose.orientation.y = 0
            pose.pose.orientation.z = 0
            pose.pose.orientation.w = 1
            pose.header.stamp = rospy.Time.now()
            for i in range(5):
                
                self.goal_pub.publish(pose)
            print("Done publishing goal")
            result.success = True
            result.message = "Completed"
            '''
        else:
            self.rel_pos_publisher.publish(int(-1))

            #result.success = False
            #result.message = "No object detected"
            
        

        #resume video capture
        subprocess.call([sys.executable, 'scripts/pause_resume_video.py', 'r'])
        #print("finish",torch.cuda.mem_get_info())

        #self.action_server.set_succeeded(result)
    
if __name__ == '__main__':
    rospy.init_node('detector_server')
    server = UserInputManagerServer()
    rospy.spin()