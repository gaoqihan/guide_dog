import sys
sys.path.append("../model/")
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
import cv2
import torch
from nanosam.utils.predictor import Predictor as NanoPredictor


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
        elif self.model=="nano":
            self.predictor = NanoPredictor(
                image_encoder_engine="/opt/nanosam/data/resnet18_image_encoder.engine",
                mask_decoder_engine="/opt/nanosam/data/mobile_sam_mask_decoder.engine"
            )
            print("Nano Model loaded")

    def encode(self, input_image):
        if self.model=="default":
            self.image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR) 
        elif self.model=="nano":
            self.image = input_image

        return self.predictor.set_image(self.image)
    
    def get_mask(self):
        if self.model=="default":
            #self.input_point = np.array([[self.image.shape[1] // 2, self.image.shape[0] // 2]])
            
            self.input_point = np.array([
                [0, 0],
                [self.image.shape[1]-1, self.image.shape[0]-1]
            ])            
            self.input_label = np.array([2,3])
            self.masks, self.scores, self.logit = self.predictor.predict(
                point_coords=self.input_point,
                point_labels=self.input_label,
                multimask_output=True,
            )

            highest_score_mask = self.masks[np.argmax(self.scores)]
            torch.cuda.empty_cache()
            return highest_score_mask
        elif self.model=="nano":
            self.bbox = [0, 0, 850, 759]
            points = np.array([
                [self.image.width//2, self.image.height//2],
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
