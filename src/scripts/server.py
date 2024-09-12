from sklearn.cluster import KMeans
import rospy
import actionlib
import cv2
from cv_bridge import CvBridge
import numpy as np
import os
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image as rImage
from geometry_msgs.msg import PoseStamped, Pose
from guide_dog.msg import ObjectDetectorAction, ObjDetectorFeedback, ObjDetectorResult  # Update with your actual action package and message names
from std_srvs.srv import Empty, EmptyResponse  # Import the Empty service type and its response
import torch
import subprocess  # Import subprocess module
import sys  # Import sys module
from scipy import stats
import json
from time import time
from std_msgs.msg import Float32, String
from dotenv import load_dotenv
import os
from PIL import ImageDraw
import argparse
# Load environment variables from .env file
load_dotenv()
import shutil


    
import datetime
from PIL import Image

# Add the '../include' directory to sys.path to import modules from there
script_dir = os.path.dirname(__file__)  # Get the directory where the script is located
include_dir = os.path.join(script_dir, '../include')  # Path to the 'include' directory
sys.path.append(include_dir)  # Add 'include' directory to sys.path

# Now you can import modules from the 'include' directory
from user_input_manager import UserInputManager, UserInput, UserAudio, UserVideo, UserRGBDSet
from gpt_caller import GPTCaller
from seg_any import SegAny
from depth_to_3d import get3d_mask,get3d_point,MapBridge
from utils import extract_number_from_brackets
from owl_detector import Detector
from nanosam.mobile_sam.automatic_mask_generator import SamAutomaticMaskGenerator
from nanosam.mobile_sam.build_sam import sam_model_registry
#from nanosam.mobile_sam.predictor import SamPredictor
import torch
class UserInputManagerServer(object):
    def __init__(self,args):
        #self.action_server = actionlib.SimpleActionServer('visual_locator_action', ObjectDetectorAction, self.object_finder, False)
        #self.action_server.start()
        self.args=args
        print("action server started")
        rospy.Subscriber("/camera/aligned_depth_to_color/camera_info", CameraInfo, self.get_info)
        rospy.Subscriber('/find_object', String,self.object_finder)
        rospy.Subscriber('/describe_environment', String,self.describe_environment)
        rospy.Subscriber('/describe_object', String,self.describe_object)
        rospy.Subscriber('/read', String,self.read)
        self.seg_mask_publisher=rospy.Publisher("/seg_mask",rImage,queue_size=10)
        self.describe_environment_complete_publisher=rospy.Publisher("/describe_environment_complete",String,queue_size=10)
        self.describe_object_complete_publisher=rospy.Publisher("/describe_object_complete",String,queue_size=10)
        self.read_complete_publisher=rospy.Publisher("/read_complete",String,queue_size=10)

        self.add_to_semantic_map_publisher=rospy.Publisher("/add_to_semantic_map", PoseStamped, queue_size=10)
        self.goal_pub = rospy.Publisher("/best_goal", PoseStamped, queue_size=10)
        #self.goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)
        #self.rel_pos_publisher=rospy.Publisher("/rel_pos",Float32,queue_size=10)
        self.speak_to_user_publisher=rospy.Publisher('/speak_to_user', String, queue_size=10)
        self.map_bridge=MapBridge()
        
        #self.manager=UserInputManager(model="nano")
        #print("UserInputManager Initialized")
        self.detector = Detector(model="nano")
        print("Detector Initialized")
        self.seg_any=SegAny(model="default")
        self.caller=GPTCaller()
        
        print("Server started")
    def get_info(self,data):
        global info
        info = data
        
    def condition_checker(self,prompt):
        print("checking condition")
        
    def describe_environment(self,msg):
        direction=msg.data
        #update this after install 360 camera
        direction="front"
        system_prompt="Your task is to provide a description of the environment in front of you based on the image provided. Please provide detaild and clear description in less than 150 words"
        subprocess.call([sys.executable, 'scripts/pause_resume_video.py', 'p'])  # Use sys.executable

        image=Image.open("./tmp/color/0.png")
        subprocess.call([sys.executable, 'scripts/pause_resume_video.py', 'r'])  # Use sys.executable

        self.caller.create_prompt([image],system_prompt_list=[system_prompt])
        response=self.caller.call(model="chatgpt-4o-latest")
        self.describe_environment_complete_publisher.publish(response)
        return
    
    def describe_object(self,msg):
        object_name=msg.data
        #update this after install 360 camera
        system_prompt="Your task is to provide a description of an object in the image based on the image provided. Please provide detaild and clear description in less than 150 words"
        subprocess.call([sys.executable, 'scripts/pause_resume_video.py', 'p'])  # Use sys.executable

        image=Image.open("./tmp/color/0.png")
        subprocess.call([sys.executable, 'scripts/pause_resume_video.py', 'r'])  # Use sys.executable
        user_promt_list=['Please describe'+object_name,image]
        self.caller.create_prompt(user_promt_list=user_promt_list,system_prompt_list=[system_prompt])
        response=self.caller.call(model="chatgpt-4o-latest")
        self.describe_object_complete_publisher.publish(response)
        return
    def read(self,msg):
        object_name=msg.data
        #update this after install 360 camera
        system_prompt="Your task is to read the text on given object in the image based on the image provided. If you are not able to identify the text, please tell the user that the text is not clear or you can't see it"
        subprocess.call([sys.executable, 'scripts/pause_resume_video.py', 'p'])  # Use sys.executable

        image=Image.open("./tmp/color/0.png")
        subprocess.call([sys.executable, 'scripts/pause_resume_video.py', 'r'])  # Use sys.executable

        self.caller.create_prompt(['Please describe'+object_name,image],system_prompt_list=[system_prompt])
        response=self.caller.call(model="chatgpt-4o-latest")
        self.read_complete_publisher.publish(response)
        return
    #def object_finder(self, goal):

    #    self.task=goal.task
    #   self.object_finder_text()
        
        
    def object_finder(self,msg):
        self.task=msg.data
        print(self.task)
        #rospy.loginfo("Video capture request received, processing...")
        #feedback = ObjDetectorFeedback()
        #result = ObjDetectorResult()
        #pause video capture
        subprocess.call([sys.executable, 'scripts/pause_resume_video.py', 'p'])  # Use sys.executable
        #create RGBDSet input
        rgbd_set=UserRGBDSet("./tmp")
        
        gpt_answer_log=""
        bbox_list_list=[]
        labeled_image_list=[]
        alter_labeled_image_lists=[]
        point_grid_image_list=[]
        sam_labeled_image_list=[]
        look_around_image_list=[]
        x1,x2,y1,y2=None,None,None,None
        best_rel_point=None
        onboard=self.args.onboard
        detector_enabled=not self.args.disable_owl
        #sam_enabled=True
        sam_enabled=not self.args.disable_sam
        point_grid_enabled=not self.args.disable_point_grid
        look_around=True
        if look_around:
            if onboard:
                with open('./prompts/distillation_onboard/rotation_and_distillation', 'r') as file:
                    prompt_json = json.loads(file.read())           
            else:
                with open('./prompts/distillation/rotation_and_distillation', 'r') as file:
                    prompt_json = json.loads(file.read())
            system_prompt=prompt_json["system_prompt"]
            response_format=prompt_json["response_format"]
            user_prompt=self.task
            full_view_image=self.get_360_image()
            #for now
            full_view_image=rgbd_set.data[0][0] #remove this once we have 360 running
            #end
            labeled_360_image=rgbd_set.draw_vertical_lines_with_numbers(full_view_image)
            look_around_image_list.append(labeled_360_image)
            self.caller.create_prompt([user_prompt,labeled_360_image],system_prompt_list=[system_prompt],response_format=response_format)
            response=self.caller.call(model="gpt-4o-2024-08-06")
            gpt_answer_log+=response+"\n"
            result_dict = json.loads(response)

            #assume done
            owl_keyword=result_dict["keyword"]
            
            key_instruction=result_dict["key_instruction"]
            direction=result_dict["direction"]
            existence_check=result_dict["existence_check"]

            self.rotate_to_direction(int(direction))         
            
        else:
        #distill the task thruough gpt
            if onboard:
                with open('./prompts/distillation_onboard/prompt', 'r') as file:
                    prompt_json = json.loads(file.read())           
            else:
                with open('./prompts/distillation/prompt', 'r') as file:
                    prompt_json = json.loads(file.read())
            system_prompt=prompt_json["system_prompt"]
            response_format=prompt_json["response_format"]
            user_prompt=self.task
            raw_image=rgbd_set.data[0][0]
            self.caller.create_prompt([user_prompt,raw_image],system_prompt_list=[system_prompt],response_format=response_format)
            response=self.caller.call(model="gpt-4o-2024-08-06")
            gpt_answer_log+=response+"\n"
            result_dict = json.loads(response)

            #assume done
            owl_keyword=result_dict["keyword"]
            
            key_instruction=result_dict["key_instruction"]
            existence_check=result_dict["existence_check"]
        
        
        if existence_check=="False":
            detector_enabled=False
            sam_enabled=False
            point_grid_enabled=False
            speech="I am sorry, I am not able to find the object you are looking for"
            speech=String(speech)
            self.speak_to_user_publisher.publish(speech)
            rospy.sleep(7)
        else:
            speech="I found what you want infront of us, I will now try to get its location"
            speech=String(speech)
            self.speak_to_user_publisher.publish(speech)
        

        print(owl_keyword,key_instruction,existence_check)
        

        #Approach 1: Owl detector
        if detector_enabled:

            #get bounding boxes through owl
            bbox_list_list,labeled_image_list,alter_labeled_image_lists=rgbd_set.detect_objects(self.detector,owl_keyword)
            if onboard:
                with open('./prompts/visual_selector_onboard/given_points', 'r') as file:
                    prompt_json = json.loads(file.read())
            else:
                with open('./prompts/visual_selector/given_points', 'r') as file:
                    prompt_json = json.loads(file.read())
            system_prompt=prompt_json["system_prompt"]
            response_format=prompt_json["response_format"]
            user_prompt=f"Task is : {key_instruction}"

            
            if os.path.exists("./tmp/cropped_depth"):
                for item in os.listdir("./tmp/cropped_depth"):
                    item_path = os.path.join("./tmp/cropped_depth", item)
                    try:
                        os.remove(item_path)  # Remove files and links
                        print(f"Deleted {item_path}")
                    except Exception as e:
                        print(f"Failed to delete {item_path}. Reason: {e}")           
            for i in range(rgbd_set.length):
                print(i)
                start_time=time()
                if len(bbox_list_list[i])==0:
                    print(f"no bounding box in frame {i}")
                    continue
                selection_range="choose from following numbers"+str(range(len(bbox_list_list[i])))
                user_prompt_list=[user_prompt,selection_range,labeled_image_list[i]]

                self.caller.create_prompt(user_prompt_list=user_prompt_list,system_prompt_list=[system_prompt],response_format=response_format)
                gpt_response=self.caller.call(model="gpt-4o-2024-08-06")
                print(f"gpt response is {gpt_response}")
                gpt_answer_log+=gpt_response+"\n"
                result_dict = json.loads(gpt_response)
                response = result_dict["final_decision"]  
                
                print("time for gpt call is: ", time()-start_time)
                #response=0
                print(f"gpt selected bounding box is {response}")
                if int(response)==-1:
                    print(f"target not found in frame {i}")
                    print(f"OWL frame {i} took {time()-start_time} seconds")

                    continue

                x1, y1, x2, y2 = tuple(bbox_list_list[i][int(response)])
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                print(f"bounding box is {x1,y1,x2,y2}")
                # Cap the x2 and y2 values at the image's width and height
                x1 = max(x1, 0)
                y1 = max(y1, 0)
                x2 = min(x2, rgbd_set.data[i][0].width)
                y2 = min(y2, rgbd_set.data[i][0].height)
                        #get segmentation mask
                self.seg_any.encode(rgbd_set.data[i][0])

                mask=self.seg_any.get_mask(bbox=(x1,y1,x2,y2))
                #save the masked cropped image in /tem/masked
                os.makedirs("./tmp/masked", exist_ok=True)
                self.seg_any.get_mask_image(f"./tmp/masked/{str(i)}.png")            
                #get depth point
                seg_mask=cv2.imread(f"./tmp/masked/0.png")
                bridge=CvBridge()
                self.seg_mask_publisher.publish(bridge.cv2_to_imgmsg(seg_mask))
                best_rel_point=get3d_mask(rgbd_set.data[i][1],mask,info,i)
                print("best rel point = ", best_rel_point)
                print(f"the {owl_keyword} is at {best_rel_point}")
                

                print(f"frame {i} took {time()-start_time} seconds")

                
                break
                
        #Method2 SAM detector:
        if sam_enabled:
            start_time=time()
            if best_rel_point is None:# or best_rel_point is not None:
                print("no object detected by owl detector, trying SAM detector")
                
                sam_labeled_image_list,point_list=rgbd_set.seg_any_label(self.seg_any)
                if onboard:
                    with open('./prompts/visual_selector_onboard/sam_points', 'r') as file:
                        prompt_json = json.loads(file.read())
                else:
                    with open('./prompts/visual_selector/sam_points', 'r') as file:
                        prompt_json = json.loads(file.read())
                system_prompt=prompt_json["system_prompt"]
                response_format=prompt_json["response_format"]
                user_prompt=key_instruction
                for i in range(len(sam_labeled_image_list)):
                    image=sam_labeled_image_list[i]
                    self.caller.create_prompt([user_prompt,image],system_prompt_list=[system_prompt],response_format=response_format)
                    response=self.caller.call(model="gpt-4o-2024-08-06")
                    gpt_answer_log+=response+"\n"
                    print(f"gpt response is {response}")
                    result_dict = json.loads(response)
                    if result_dict["final_decision"]!="-1":
                        best_pixel=point_list[int(result_dict["final_decision"])]
                        # Get a square bounding box around the best pixel
                        bbox_size = 50  # Adjust the size of the bounding box as needed
                        x = best_pixel[0]
                        y = best_pixel[1]
                        x1 = max(x - bbox_size // 2, 0)
                        y1 = max(y - bbox_size // 2, 0)
                        x2 = min(x + bbox_size // 2, rgbd_set.data[i][0].width)
                        y2 = min(y + bbox_size // 2, rgbd_set.data[i][0].height)
                        self.seg_any.encode(rgbd_set.data[i][0])

                        mask=self.seg_any.get_mask(bbox=(x1,y1,x2,y2))
                        
                        
                        #best_rel_point=get3d_point(rgbd_set.data[i][1],best_pixel,info,i)
                        best_rel_point=get3d_mask(rgbd_set.data[i][1],mask,info,i)

                        break
                    else:
                        print(f"target not found in frame {i} with point sam")   
            print("time for sam is: ", time()-start_time)       
        # Grid points
        if point_grid_enabled:
            start_time=time()
            if best_rel_point is None:# or best_rel_point is not None:
                print("no object detected by SAM detector, trying point grid")
                #self.manager.add_new_input(rgbd_set)
                os.makedirs("./tmp/point_grid", exist_ok=True)

                point_grid_image_list,points_coord_list=rgbd_set.point_grid_label(points_num=100)
                #point_grid_image_list,points_coord_list=rgbd_set.point_grid_label(points_num=400)
                if onboard:
                    with open('./prompts/visual_selector_onboard/point_grid', 'r') as file:
                        prompt_json = json.loads(file.read())
                else:
                    with open('./prompts/visual_selector/point_grid', 'r') as file:
                        prompt_json = json.loads(file.read())
                system_prompt=prompt_json["system_prompt"]
                response_format=prompt_json["response_format"]
                user_prompt=key_instruction
                for i in range(len(point_grid_image_list)):
                    image=point_grid_image_list[i]
                    self.caller.create_prompt([user_prompt,image],system_prompt_list=[system_prompt],response_format=response_format)
                    response=self.caller.call(model="gpt-4o-2024-08-06")
                    gpt_answer_log+=response+"\n"
                    print(f"gpt response is {response}")
                    result_dict = json.loads(response)
                    if result_dict["final_decision"]!="-1":
                        best_pixel=points_coord_list[int(result_dict["final_decision"])]
                        #best_rel_point=get3d_point(rgbd_set.data[i][1],best_pixel,info,i)
                        bbox_size = 50  # Adjust the size of the bounding box as needed
                        x = best_pixel[0]
                        y = best_pixel[1]
                        x1 = max(x - bbox_size // 2, 0)
                        y1 = max(y - bbox_size // 2, 0)
                        x2 = min(x + bbox_size // 2, rgbd_set.data[i][0].width)
                        y2 = min(y + bbox_size // 2, rgbd_set.data[i][0].height)
                        self.seg_any.encode(rgbd_set.data[i][0])

                        mask=self.seg_any.get_mask(bbox=(x1,y1,x2,y2))
                        self.seg_any.get_mask_image(f"./tmp/masked/0.png")            
                        #get depth point
                        seg_mask=cv2.imread(f"./tmp/masked/0.png")
                        bridge=CvBridge()
                        self.seg_mask_publisher.publish(bridge.cv2_to_imgmsg(seg_mask))
                        best_rel_point=get3d_mask(rgbd_set.data[i][1],mask,info,i)
                        break
                    else:
                        print(f"target not found in frame {i} with point grid")   
            print("time for point grid is: ", time()-start_time)
        #save the package
        save_package(rgbd_set, labeled_image_list=labeled_image_list, alter_labeled_image_lists=alter_labeled_image_lists, bbox_list_list=bbox_list_list, point_grid_image_list=point_grid_image_list, sam_labeled_image_list=sam_labeled_image_list, look_around_image_list=look_around_image_list, gpt_log=gpt_answer_log)
    
        if best_rel_point is not None:
            best_rel_point[2]=best_rel_point[2]
            print(best_rel_point)
            #self.rel_pos_publisher.publish(Float32(best_rel_point[2]))
            
            
            object_position_in_map=self.map_bridge.get_object_position_in_map(best_rel_point[0],best_rel_point[1],best_rel_point[2],1)
            print(f"best point is {object_position_in_map}")
            
            #remove this when onboard
            #object_position_in_map=[27,15,0]
            self.map_bridge.publish_markers([object_position_in_map])
            pose=PoseStamped()
            pose.header.frame_id = "map"

            pose.pose.position.x = object_position_in_map[0]
            pose.pose.position.y = object_position_in_map[1]
            pose.pose.position.z = 0

            pose.pose.orientation.x = 0
            pose.pose.orientation.y = 0
            pose.pose.orientation.z = 0
            pose.pose.orientation.w = 1
            pose.header.stamp = rospy.Time.now()
            for i in range(5):
                
                self.goal_pub.publish(pose)
            #this needs to be changed, its not always owl_keypoint[0]. instead ask gpt to clarify what object it selected
            #self.add_to_permanant_map(result_dict["object_name"],pose)
            print("Done publishing goal")
            #result.success = True
            #result.message = "Completed"
            
        
        else:
            pose=PoseStamped()
            pose.header.frame_id = "map"

            pose.pose.position.x = -1
            pose.pose.position.y = -1
            pose.pose.position.z = 0

            pose.pose.orientation.x = 0
            pose.pose.orientation.y = 0
            pose.pose.orientation.z = 0
            pose.pose.orientation.w = 1
            pose.header.stamp = rospy.Time.now()
            for i in range(5):
               
                self.goal_pub.publish(pose)
            print('No object detected')
            #self.rel_pos_publisher.publish(int(-1))

            #result.success = False
            #result.message = "No object detected"
        #resume video capture
        subprocess.call([sys.executable, 'scripts/pause_resume_video.py', 'r'])
    
    def add_to_permanant_map(self,object,pose_stamped):
        print(f"adding {object} to permanant map")
        with open('./prompts/map_processing/add_semantic_point_to_map', 'r') as file:
            prompt_json = json.loads(file.read())
        system_prompt=prompt_json["system_prompt"]
        response_format=prompt_json["response_format"]
        user_prompt="the object is a "+object+"is the object a permanent object or a moveable object?"

        self.caller.create_prompt([user_prompt],system_prompt_list=[system_prompt],response_format=response_format)
        response=self.caller.call(model="gpt-4o-2024-08-06")
        self.add_to_semantic_map_publisher.publish(pose_stamped)
        print(f"gpt response for permanant map is {response}")
    
        
    def rotate_to_direction(self,direction):
        print(f"rotating to {direction}")    
        print("TBD")
        pass
    def get_360_image(self):
        print("getting 360 image")
        print("TBD")
        pass
    
    

def save_package(rgbd_set, bbox_list_list=[], labeled_image_list=[], alter_labeled_image_lists=[],point_grid_image_list=[],sam_labeled_image_list=[],look_around_image_list=[], gpt_log=""):
    # Create a folder with the current time as folder_name
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_folder = os.path.join('./log', current_time)
    os.makedirs(base_folder, exist_ok=True)
    
    # Create a raw_image folder inside the base_folder
    raw_image_folder = os.path.join(base_folder, 'raw_image')
    os.makedirs(raw_image_folder, exist_ok=True)
    
    # Iterate over the rgbd_set.data list
    for idx, (pil_image, npy_array) in enumerate(rgbd_set.data):
        # Create a separate subfolder for each tuple
        tuple_folder = os.path.join(raw_image_folder, f'image_{idx}')
        os.makedirs(tuple_folder, exist_ok=True)
        
        # Save the PIL image
        image_path = os.path.join(tuple_folder, 'image.png')
        pil_image.save(image_path)
        
        # Save the .npy array
        npy_path = os.path.join(tuple_folder, 'data.npy')
        np.save(npy_path, npy_array)
    
    # Create a labeled_image folder inside the base_folder
    labeled_image_folder = os.path.join(base_folder, 'labeled_image')
    os.makedirs(labeled_image_folder, exist_ok=True)
    point_grid_image_folder=os.path.join(base_folder, 'point_grid_image')
    os.makedirs(point_grid_image_folder, exist_ok=True)
    sam_image_folder=os.path.join(base_folder, 'sam_image')
    os.makedirs(sam_image_folder, exist_ok=True)
    look_around_image_folder=os.path.join(base_folder, 'look_around_image')
    os.makedirs(look_around_image_folder, exist_ok=True)
    # Save each PIL image in labeled_image_list
    for idx, pil_image in enumerate(labeled_image_list):
        image_path = os.path.join(labeled_image_folder, f'labeled_image_{idx}.png')
        pil_image.save(image_path)
    
    # Create an alter_labeled_image folder inside the base_folder
    alter_labeled_image_folder = os.path.join(base_folder, 'alter_labeled_image')
    os.makedirs(alter_labeled_image_folder, exist_ok=True)
    seg_mask_image_path="./tmp/masked/0.png"
    new_seg_mask_path=os.path.join(base_folder, 'seg_mask.png')
    shutil.copy(seg_mask_image_path,new_seg_mask_path)
    
    # Save each PIL image in alter_labeled_image_lists
    for idx, pil_image in enumerate(alter_labeled_image_lists):
        image_path = os.path.join(alter_labeled_image_folder, f'alter_labeled_image_{idx}.png')
        pil_image.save(image_path)
    for idx, pil_image in enumerate(point_grid_image_list):
        image_path = os.path.join(point_grid_image_folder, f'point_grid_image_{idx}.png')
        pil_image.save(image_path)
    for idx, pil_image in enumerate(sam_labeled_image_list):
        image_path = os.path.join(sam_image_folder, f'sam_labeled_image_{idx}.png')
        pil_image.save(image_path)
    for idx, pil_image in enumerate(look_around_image_list):
        image_path = os.path.join(look_around_image_folder, f'look_around_image_{idx}.png')
        pil_image.save(image_path)
    # Save bbox_list_list as a text file
    bbox_list_path = os.path.join(base_folder, 'bbox_list.txt')
    with open(bbox_list_path, 'w') as f:
        for bbox_list in bbox_list_list:
            f.write(f"{bbox_list}\n")
    
    # Save gpt_log as a text file
    gpt_log_path = os.path.join(base_folder, 'gpt_log.txt')
    with open(gpt_log_path, 'w') as f:
        f.write(gpt_log)
    
    print(f"Data saved in {base_folder}")


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Description of your program')
    
    # Add arguments
    parser.add_argument('--onboard', action='store_true', help='if set to True, enables onboard mode')
    parser.add_argument('--disable_owl', action='store_true', help='if set to True, disables owl detector')
    parser.add_argument('--disable_sam', action='store_true', help='if set to True, disables sam detector')
    parser.add_argument('--disable_point_grid', action='store_true', help='if set to True, disables point grid detector')
    
    # Parse the arguments
    args = parser.parse_args()
    


    rospy.init_node('detector_server')
    server = UserInputManagerServer(args)
    rospy.spin()
