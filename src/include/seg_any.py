import sys
sys.path.append("../model/")
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
import cv2
import torch

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
    def __init__(self):
        sam_checkpoint = "./model/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device="cuda")
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        self.image = None
        print("SegAny Model loaded")

    def encode(self, input_image):
        self.predictor = SamPredictor(self.sam)

        self.image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
        

        return self.predictor.set_image(self.image)
    
    def get_mask(self):
        self.input_point = np.array([[self.image.shape[1] // 2, self.image.shape[0] // 2]])

        self.input_label = np.array([1])
        self.masks, self.scores, self.logit = self.predictor.predict(
            point_coords=self.input_point,
            point_labels=self.input_label,
            multimask_output=True,
        )

        highest_score_mask = self.masks[np.argmax(self.scores)]
        torch.cuda.empty_cache()
        #del self.masks
        #del self.scores
        #del self.logit
        #self.predictor.reset_image()
        #self.delete_predictor()
        return highest_score_mask
    
    def delete_predictor(self):
        del self.predictor
        torch.cuda.empty_cache()

    def get_mask_image(self,filename=None):
        for i, (mask, score) in enumerate(zip(self.masks, self.scores)):
            plt.figure(figsize=(10,10))
            plt.imshow(self.image)
            show_mask(mask, plt.gca())
            show_points(self.input_point, self.input_label, plt.gca())
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            #plt.show() 
            plt.savefig(filename)