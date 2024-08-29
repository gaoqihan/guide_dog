from datetime import datetime
import whisper
import random

from moviepy.editor import VideoFileClip
import os
from PIL import Image
import numpy as np
import sys
import rospy
import torch
# Add the '../include' directory to sys.path to import modules from there
script_dir = os.path.dirname(__file__)  # Get the directory where the script is located
include_dir = os.path.join(script_dir, '../include')  # Path to the 'include' directory
sys.path.append(include_dir)  # Add 'include' directory to sys.path


from owl_detector import Detector

class UserInputManager:
    def __init__(self,model="default"):
        self.model=model
        self.content=[]
        self.audio_model = whisper.load_model("base")
        self.detector = Detector(model=self.model)
    def add_new_input(self,user_input):
        self.assign_id(user_input)
        self.content.append(user_input)
        rospy.loginfo(f"New input added: {user_input.id}")

        return user_input.id
        
    def assign_id(self,user_input):
        user_input.id = f"{user_input.type}_{abs(hash(int(user_input.created_time.timestamp())))}"
        collision = True
        while collision:
            collision = False
            for content in self.content:
                if content.id == user_input.id:
                    collision = True
                    user_input.assign_id()
                    break
    def get_most_recent_input(self):
        if self.content:
            most_recent_input = self.content[0]
            for user_input in self.content:
                if user_input.created_time.timestamp() > most_recent_input.created_time.timestamp():
                    most_recent_input = user_input
            return most_recent_input
        else:
            return None
        
        
    def get_input_by_id(self,id):
        for user_input in self.content:
            if user_input.id == id:
                return user_input
        return None

    def search_input_by_keyword(self,keyword):
        results = []
        for user_input in self.content:
            if keyword in user_input.data:
                results.append(user_input.id)
        return results
          
    def transcribe_audio(self,id):
        
        audio_input = self.get_input_by_id(id)
        if audio_input.type != "audio":
            raise TypeError("Invalid input type")
        result = self.audio_model.transcribe(audio_input.audio_file)["text"]
        audio_input.data = result
        return result
    
    def downsample_video(self,id,fps=1):
        video_input = self.get_input_by_id(id)
        image_list=[]
        if video_input.type != "video":
            raise TypeError("Invalid input type")

        # Load the video
        video_clip = VideoFileClip(video_input.video_file)

        # Set the fps to 1
        downsampled_video = video_clip.set_fps(fps)
        

        # Save each frame as an image file
        for i, frame in enumerate(downsampled_video.iter_frames()):
            img = Image.fromarray(frame)
            #img = img.resize((640, 480))
            image_list.append(img)        
        video_input.data = image_list
        return video_input.data
    '''
    def detect_objects(self,id,texts):
        user_input = self.get_input_by_id(id)
        user_input.request[texts[0]] = []
        if user_input.type == "video":
            for image in user_input.data:
                results = self.detector.detect(image, texts)
                #print(results)
                labeled_image=self.detector.displayBoundingBox(image,results,texts)
                #image.show()
                user_input.request[texts[0]].append({"boxes":results[0]["boxes"].tolist(),"image":labeled_image})
        
        elif user_input.type == "RGBD_set":
            user_input.request[texts[0]] = []
            for image, depth in user_input.data:
                results = self.detector.detect(image, texts)
                #print(results)
                labeled_image=self.detector.displayBoundingBox(image,results,texts)
                if self.model=="default":
                    user_input.request[texts[0]].append({"boxes":results[0]["boxes"].tolist(),"image":labeled_image})
                elif self.model=="nano":
                    user_input.request[texts[0]].append({"boxes":results.boxes.tolist(),"image":labeled_image})

        else:
            raise TypeError("Invalid input type")
    '''
    


class UserInput:
    def __init__(self):
        self.id=None
        self.created_time = datetime.now()
        self.type = None
        self.data=None
        self.request={}
class UserAudio(UserInput):
    def __init__(self,file_path):
        super().__init__()
        self.type="audio"
        self.audio_file = file_path
        
class UserVideo(UserInput):
    def __init__(self,file):
        super().__init__()
        self.type = "video"
        self.video_file = file
        


class UserRGBDSet(UserInput):
    #self.data is the list of tuples of (color_image, depth_image)
    def __init__(self,path_to_images):
        super().__init__()
        self.type = "RGBD_set"
        self.image_set_dir = path_to_images
        self.data=[]
        self.length=0
        color_path = os.path.join(self.image_set_dir,'color')
        depth_path = os.path.join(self.image_set_dir,'depth')

        # Get the list of png files in the color directory
        color_files = [f for f in os.listdir(color_path) if f.endswith('.png')]

        # Sort the files to ensure they match with the depth files
        color_files.sort()

        # Load the images and .npy files into tuple pairs
        for filename in color_files:
            # Load the PIL image
            image = Image.open(os.path.join(color_path, filename))

            # Load the corresponding .npy file
            depth_filename = os.path.splitext(filename)[0] + '.npy'
            depth = np.load(os.path.join(depth_path, depth_filename))

            # Add the pair to the list
            self.data.append((image, depth)) 
            self.length+=1
    def detect_objects(self,detector,texts):
        self.bbox_list_list=[]
        self.labeled_image_list=[]
        for image, depth in self.data:
            results = detector.detect(image, texts)
            self.bbox_list=results.boxes.tolist()
            self.labeled_image=detector.displayBoundingBox(image,results,texts)
            self.bbox_list_list.append(self.bbox_list)
            self.labeled_image_list.append(self.labeled_image)
            
            
        return self.bbox_list_list,self.labeled_image_list

        