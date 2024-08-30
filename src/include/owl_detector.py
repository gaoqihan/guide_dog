

from transformers import Owlv2Processor, Owlv2ForObjectDetection
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw,ImageFont
import subprocess
import torch
from nanoowl.owl_predictor import OwlPredictor
from nanoowl.owl_drawing import draw_owl_output
import os

class Detector:
    """Class that implements Owl v2 for object detection."""
    def __init__(self,model="default"):
        #print("cuda is available:",torch.cuda.is_available())
        self.model_type=model
        if self.model_type=="default":
            print("Loading Default model")
            self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
            self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to("cuda")
        elif self.model_type=="nano":
            print("Loading Nano model")
            self.predictor = OwlPredictor(
                "google/owlvit-base-patch32",
                image_encoder_engine="/opt/nanoowl/data/owl_image_encoder_patch32.engine"
            )


    def detect(self, image, texts):
        #print("start detect objects",torch.cuda.mem_get_info())
        if self.model_type=="default":
            inputs = self.processor(texts, image, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Convert model outputs to COCO API format.
            #target_sizes = torch.Tensor([image.size[::-1]])
            target_sizes=[(image.size[1],image.size[0])]
            results = self.processor.post_process_object_detection(outputs=outputs, threshold=0.25)

            box=results[0]["boxes"]*960

            a=image.size[0]/960
            box *= a 

            results[0]["boxes"]=box

            boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
            topk=min(3,scores.size(0))
            # Sort scores in descending order and get the top 3 indices
            _, top_indices = torch.topk(scores, topk, largest=True, sorted=True)

            # Filter boxes, scores, and labels to keep only the top 3
            boxes = boxes[top_indices]
            scores = scores[top_indices]
            labels = labels[top_indices]

            results[0]["boxes"] = boxes
            results[0]["scores"] = scores
            results[0]["labels"] = labels

            del outputs
            del inputs
            del target_sizes
            torch.cuda.empty_cache()
            #print("finish detect objects",torch.cuda.mem_get_info())
            #print(type(results))
            #print(results)
            return results
        elif self.model_type=="nano":
            text_encodings = self.predictor.encode_text(texts)
            results = self.predictor.predict(image=image, text=texts, text_encodings=text_encodings, threshold=0.01)
            boxes, scores, labels = results.boxes, results.scores, results.labels

            # Group results by label
            label_groups = {}
            for box, score, label in zip(boxes, scores, labels):
                if label.item() not in label_groups:
                    label_groups[label.item()] = {'boxes': [], 'scores': [], 'labels': []}
                label_groups[label.item()]['boxes'].append(box)
                label_groups[label.item()]['scores'].append(score)
                label_groups[label.item()]['labels'].append(label)
            # Filter top 4 for each label
            filtered_boxes = []
            filtered_scores = []
            filtered_labels = []
            for label, group in label_groups.items():
                group_scores = torch.tensor(group['scores'])
                topk = min(4, group_scores.size(0))
                _, top_indices = torch.topk(group_scores, topk, largest=True, sorted=True)
                for idx in top_indices:
                    filtered_boxes.append(group['boxes'][idx])
                    filtered_scores.append(group['scores'][idx])
                    filtered_labels.append(group['labels'][idx])

            # Convert lists back to tensors
            if filtered_boxes:
                results.boxes = torch.stack(filtered_boxes)
            else:
                results.boxes = torch.tensor([])
            results.scores = torch.tensor(filtered_scores)
            results.labels = torch.tensor(filtered_labels)

            return results


    
    def displayBoundingBox(self,image,results,text,mode="small"):
        # Ensure the directory exists
        output_dir = './tmp/bbox'
        os.makedirs(output_dir, exist_ok=True)
        if self.model_type=="default":

            # Retrieve predictions for the first image.
            i = 0
            #text = texts[i]
            boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

            # Sort indices based on scores in descending order
            sorted_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)

            # Select the top 3 indices
            top_indices = sorted_indices[:3]

            # Filter boxes, scores, and labels to keep only the top 3
            boxes = [boxes[idx] for idx in top_indices]
            scores = [scores[idx] for idx in top_indices]
            labels = [labels[idx] for idx in top_indices]
            # Draw bounding boxes and labels on the image.
            copied_image = image.copy()
            draw = ImageDraw.Draw(copied_image)

            for box, score, label in zip(boxes, scores, labels):
                box = [round(i, 2) for i in box.tolist()]
                x1, y1, x2, y2 = tuple(box)
                draw.rectangle(xy=((x1, y1), (x2, y2)), outline="red")
                font_size = 40
                font = ImageFont.load_default(size=font_size)
                draw.text(xy=(x1, y1), text=str(i), font=font, fill="green")
                i += 1
            return copied_image

            #copied_image.show()
        elif self.model_type=="nano":
            # Draw the bounding boxes on the image
            #copied_image = draw_owl_output(image, results, text=text, draw_text=True)
            #copied_image.show()``
            if mode=="small":
                copied_image = annotate(image, results, size=0.5)
            elif mode=="mid":
                copied_image = annotate(image, results,size=1)
            elif mode=="large":
                copied_image = annotate(image, results,size=2)
            elif mode=="bbox":
                copied_image = draw_owl_output(image, results, text=text, draw_text=True)
            else:
                copied_image = annotate(image, results,size=1)

            output_path = os.path.join(output_dir, 'bbox_image.png')

            copied_image.save(output_path)
            return copied_image

import PIL.Image
import PIL.ImageDraw
import cv2
from nanoowl.owl_predictor import OwlDecodeOutput
import matplotlib.pyplot as plt
import numpy as np
from typing import List

def annotate(image, output: OwlDecodeOutput,size=1):
    is_pil = not isinstance(image, np.ndarray)
    if is_pil:
        image = np.asarray(image)
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = size
    num_detections = len(output.labels)
    #num_detections=min(num_detections,3)

    for i in range(num_detections):
        box = output.boxes[i]
        label_index = int(output.labels[i])
        box = [int(x) for x in box]
        pt0 = (box[0], box[1])
        pt1 = (box[2], box[3])

        offset_y = (box[3] - box[1]) // 2
        offset_x = (box[2] - box[0]) // 2
        center = (box[0] + offset_x, box[1] + offset_y)

        # Calculate the size of the text box
        text = str(i)
        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            font,
            font_scale,
            2  # thickness
        )

        # Determine the radius of the circle
        radius = max(text_width, text_height) // 2  # Add some padding

        cv2.circle(
            image,
            center,
            radius,
            (0, 0, 0),  # black color
            cv2.FILLED
        )
        text_x = center[0] - text_width // 2
        text_y = center[1] + text_height // 2 #- baseline

        # Draw the text on top of the circle
        cv2.putText(
            image,
            str(i),
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),  # white color for text
            1,  # thickness
            cv2.LINE_AA
        )
    if is_pil:
        image = PIL.Image.fromarray(image)
    return image