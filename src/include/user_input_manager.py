from datetime import datetime
import whisper
import random

from moviepy.editor import VideoFileClip
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import sys
import rospy
#import torch
import cv2
# Add the '../include' directory to sys.path to import modules from there
script_dir = os.path.dirname(__file__)  # Get the directory where the script is located
include_dir = os.path.join(script_dir, '../include')  # Path to the 'include' directory
sys.path.append(include_dir)  # Add 'include' directory to sys.path


from owl_detector import Detector

class UserInputManager:
    def __init__(self,model="default"):
        self.model=model
        self.content=[]
        #self.audio_model = whisper.load_model("base")
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
          
    #def transcribe_audio(self,id):
        
    #    audio_input = self.get_input_by_id(id)
    #    if audio_input.type != "audio":
    #        raise TypeError("Invalid input type")
    #    result = self.audio_model.transcribe(audio_input.audio_file)["text"]
    #    audio_input.data = result
    #s    return result
    
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
        alt_labeled_image_list=[]
        for image, depth in self.data:
            results = detector.detect(image, texts)
            self.bbox_list=results.boxes.tolist()
            self.labeled_image=detector.displayBoundingBox(image,results,texts,mode="mid")
            alt_labeled_image=[detector.displayBoundingBox(image,results,texts,mode="bbox"),
                               detector.displayBoundingBox(image,results,texts,mode="small"),
                               detector.displayBoundingBox(image,results,texts,mode="mid"),
                               detector.displayBoundingBox(image,results,texts,mode="large")]
            self.bbox_list_list.append(self.bbox_list)
            self.labeled_image_list.append(self.labeled_image)
            alt_labeled_image_list+=alt_labeled_image
            
            
        return self.bbox_list_list,self.labeled_image_list,alt_labeled_image_list
    

    def seg_any_label(self,seg_any_object):
        labeled_image_list=[]
        for image, depth in self.data:
            image = np.array(image)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            labeled_image,point_list=seg_any_object.get_auto_mask(image)
            
            labeled_image = Image.fromarray(labeled_image)
            labeled_image_list.append(labeled_image)
            break #takes too long, only do once
        return labeled_image_list,point_list
        
        
    def point_grid_label(self, points_num=32):
        is_pil = not isinstance(self.data[0][0], np.ndarray)

        point_labeled_image_list = []
        points_coord_list = []
        
        # Determine the number of rows and columns for the grid
        grid_size = int(points_num ** 0.5)
        
        for image, _ in self.data:
            if is_pil:
                image = np.asarray(image)
            height, width, _ = image.shape

            # Determine the number of rows and columns for the grid based on the aspect ratio
            aspect_ratio = width / height
            grid_cols = int((points_num * aspect_ratio) ** 0.5)
            grid_rows = int(points_num / grid_cols)
            
            # Load a larger font
            font_scale = 0.5  # Adjust this value as needed
            font_thickness = 1  # Adjust this value as needed
            font = cv2.FONT_HERSHEY_TRIPLEX
            
            # Calculate coordinates for grid points
            for i in range(grid_rows):
                for j in range(grid_cols):
                    x = int((j + 0.5) * width / grid_cols)
                    y = int((i + 0.5) * height / grid_rows * 0.6 + height * 0.4)  # Adjust y to be in the lower 60%
                    points_coord_list.append((x, y))
                    text = str(i * grid_cols + j)
                    (text_width, text_height), baseline = cv2.getTextSize(
                                text,
                                font,
                                font_scale,
                                2  # thickness
                            )

                    # Draw a solid black circle
                    radius = max(text_width, text_height) // 2  # Add some padding
                    cv2.circle(image, (x, y), radius, (0, 0, 0), -1)
                            
                    # Calculate text size and position to center it
                    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                    text_x = x - text_size[0] // 2
                    text_y = y + text_size[1] // 2
                            
                    # Draw the number in white
                    cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
            if is_pil:
                image = Image.fromarray(image)
            point_labeled_image_list.append(image)
        
        return point_labeled_image_list, points_coord_list
        
    def draw_vertical_lines_with_numbers(self,image, sections=8):
        is_pil = not isinstance(image, np.ndarray)
        if is_pil:
            image = np.asarray(image)
        
        # Downsample the image to 720p resolution
        image = cv2.resize(image, (1280, 720))
        
        height, width, _ = image.shape
        
        # Calculate the width of each section
        section_width = width // sections
        
        # Load a larger font
        font_scale = 5.0  # Adjust this value as needed
        font_thickness = 3  # Adjust this value as needed
        font = cv2.FONT_HERSHEY_TRIPLEX
        
        for i in range(sections):
            x = i * section_width
            
            # Draw thicker red vertical line
            cv2.line(image, (x, 0), (x, height), (0, 0, 255), 4)
            
            # Calculate the center of the section for the number
            center_x = x + section_width // 2
            
            # Prepare the number text
            text = str(i + 1)
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            
            # Draw black background for the number at the top
            top_rect_start = (center_x - text_width // 2 - 10, 10)
            top_rect_end = (center_x + text_width // 2 + 10, text_height + 20)
            cv2.rectangle(image, top_rect_start, top_rect_end, (0, 0, 0), -1)
            
            # Draw the number in white at the top
            top_text_x = center_x - text_width // 2
            top_text_y = text_height + 10
            cv2.putText(image, text, (top_text_x, top_text_y), font, font_scale, (255, 255, 255), font_thickness)
            
            # Draw black background for the number at the bottom
            bottom_rect_start = (center_x - text_width // 2 - 10, height - text_height - 20)
            bottom_rect_end = (center_x + text_width // 2 + 10, height - 10)
            cv2.rectangle(image, bottom_rect_start, bottom_rect_end, (0, 0, 0), -1)
            
            # Draw the number in white at the bottom
            bottom_text_x = center_x - text_width // 2
            bottom_text_y = height - 10
            cv2.putText(image, text, (bottom_text_x, bottom_text_y), font, font_scale, (255, 255, 255), font_thickness)
        
        if is_pil:
            image = Image.fromarray(image)
        
        return image

