import sys
sys.path.append("../model/")
import cv2
import numpy as np
#from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
import torch
from nanosam.utils.predictor import Predictor as NanoPredictor
import copy
from scipy.ndimage import measurements
from nanosam.mobile_sam.automatic_mask_generator import SamAutomaticMaskGenerator,SamPredictor
from nanosam.mobile_sam.build_sam import sam_model_registry

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

class SegAny:
    def __init__(self,model="defualt"):
        self.model=model
        if self.model=="default":
            sam_checkpoint = "./model/sam_vit_h_4b8939.pth"
            model_type = "vit_h"
            self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            self.sam.to(device="cuda")
            self.mask_generator = SamAutomaticMaskGenerator(self.sam)
            self.image = None
            print("SegAny Model loaded")
            self.predictor = SamPredictor(self.sam)
        if self.model=="nano":
            self.predictor = NanoPredictor(
                image_encoder_engine="/opt/nanosam/data/resnet18_image_encoder.engine",
                mask_decoder_engine="/opt/nanosam/data/mobile_sam_mask_decoder.engine"
            )
            print("Nano Model loaded")
        
        sam_checkpoint = "./model/mobile_sam.pt"
        model_type = "vit_t"
        self.auto_mask_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.auto_mask_sam.to(device="cuda")
        self.auto_mask_generator = SamAutomaticMaskGenerator(self.auto_mask_sam)
        print("SegAny auto mask Model loaded")

    def encode(self, input_image):
        if self.model=="default":
            self.image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR) 
        elif self.model=="nano":
            self.image = input_image

        return self.predictor.set_image(self.image)
    
    def get_mask(self,bbox=None):
        if self.model=="default":
            #self.input_point = np.array([[self.image.shape[1] // 2, self.image.shape[0] // 2]])
            
            self.input_point = np.array([
                [0, 0],
                [self.image.shape[1]-1, self.image.shape[0]-1]
            ])   
            
            if bbox is None:
                bbox=np.array([0, 0, self.image.shape[1]-1, self.image.shape[0]-1])
            else:
                bbox = np.array(bbox)
            self.input_label = np.array([2,3])
            #self.masks, self.scores, self.logit = self.predictor.predict(
            #    point_coords=self.input_point,
            #    point_labels=self.input_label,
            #    multimask_output=True,
            #)
            self.masks, self.scores, self.logit =  self.predictor.predict(
                box=bbox,
                multimask_output=False
            )

            #highest_score_mask = self.masks[np.argmax(self.scores)]
            highest_score_mask = self.masks[0]
            #torch.cuda.empty_cache()
            return highest_score_mask
        elif self.model=="nano":
            self.bbox = [0, 0, 850, 759]
            points = np.array([
                [self.image.width//2, self.image.height//4*3],
            ])

            point_labels = np.array([1])
            #print("before predict")  
            mask, _, _ = self.predictor.predict(points, point_labels)
            #print("after predict")
            self.mask = (mask[0, 0] > 0).detach().cpu().numpy()
            return self.mask
    
    
    
    def delete_predictor(self):
        del self.predictor
        torch.cuda.empty_cache()

    def get_mask_image(self,filename=None):
        if self.model=="default":
            for i, (mask, score) in enumerate(zip(self.masks, self.scores)):
                plt.figure(figsize=(10,10))
                plt.imshow(self.image)
                show_mask(mask, plt.gca())
                show_points(self.input_point, self.input_label, plt.gca())
                plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
                plt.axis('off')
                #plt.show() 
                plt.savefig(filename)
        elif self.model=="nano":
            # Draw resykts
            plt.imshow(self.image)
            plt.imshow(self.mask, alpha=0.5)
            #x = [self.bbox[0], self.bbox[2], self.bbox[2], self.bbox[0], self.bbox[0]]
            #y = [self.bbox[1], self.bbox[1], self.bbox[3], self.bbox[3], self.bbox[1]]
            #plt.plot(x, y, 'g-')
            plt.savefig(filename)

    def get_auto_mask(self,image):
        
        masks = self.auto_mask_generator.generate(image)
        point_list = []
        for mask in masks:
            binary_mask = mask['segmentation']
            # Calculate the center of mass
            centroid = measurements.center_of_mass(binary_mask)

            centroid_int = (int(round(centroid[1])), int(round(centroid[0])))
            point_list.append(centroid_int)

        labeled_image = point_placer(image, point_list)
        # Swap (y,x) to (x,y) in point_list
        #point_list = [(y, x) for x, y in point_list]
        return labeled_image,point_list
        
def point_placer(image, point_list):
    image = copy.deepcopy(image)
    is_pil = not isinstance(image, np.ndarray)
    if is_pil:
        image = np.asarray(image)
    height, width, _ = image.shape
    
    # Load a larger font
    font_scale = 0.5  # Adjust this value as needed
    font_thickness = 1  # Adjust this value as needed
    font = cv2.FONT_HERSHEY_TRIPLEX
    cnt=0
    # Calculate coordinates for grid points
    for x,y in point_list:
        text=str(cnt)
        cnt+=1
        (text_width, text_height), baseline = cv2.getTextSize(
                text,
                font,
                font_scale,
                2  # thickness
            )

        # Draw a solid black circle
        radius = max(text_width, text_height) //2  # Add some padding
        cv2.circle(image, (x, y), radius, (0, 0, 0), -1)
            
        # Calculate text size and position to center it
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = x - text_size[0] // 2
        text_y = y + text_size[1] // 2
            
        # Draw the number in white
        cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
    if is_pil:
        image = Image.fromarray(image)
    
    
    return image