

from transformers import Owlv2Processor, Owlv2ForObjectDetection
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw,ImageFont
import subprocess
import torch
from nanoowl.owl_predictor import OwlPredictor
from nanoowl.owl_drawing import draw_owl_output

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
            results = self.predictor.predict(image=image, text=texts,text_encodings=text_encodings, threshold=0.1)
            #print(results)
            boxes, scores, labels = results.boxes, results.scores, results.labels
            topk=min(3,scores.size(0))
            # Sort scores in descending order and get the top 3 indices
            _, top_indices = torch.topk(scores, topk, largest=True, sorted=True)

            # Filter boxes, scores, and labels to keep only the top 3
            boxes = boxes[top_indices]
            scores = scores[top_indices]
            labels = labels[top_indices]
            results.boxes = boxes
            results.scores = scores
            results.labels = labels
            return results


    
    def displayBoundingBox(self,image,results,text):
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
            copied_image = draw_owl_output(image, results, text=text, draw_text=True)
            #copied_image.show()
            return copied_image
