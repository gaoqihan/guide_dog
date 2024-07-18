

from transformers import Owlv2Processor, Owlv2ForObjectDetection
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw,ImageFont
import subprocess
import torch
class Detector:
    """Class that implements Owl v2 for object detection."""
    def __init__(self):
        print("cuda is available:",torch.cuda.is_available())
        self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to("cuda")
    
    def detect(self, image, texts):
        inputs = self.processor(texts, image, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Convert model outputs to COCO API format.
        target_sizes = torch.Tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.25)

        return results
    
    def displayBoundingBox(self,image,results,text):

        # Retrieve predictions for the first image.
        i = 0
        #text = texts[i]
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
        # Draw bounding boxes and labels on the image.
        copied_image = image.copy()
        draw = ImageDraw.Draw(copied_image)

        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            x1, y1, x2, y2 = tuple(box)
            draw.rectangle(xy=((x1, y1), (x2, y2)), outline="red")
            font_size = 40
            font = ImageFont.load_default(size=font_size)
            draw.text(xy=(x1, y1), text=str(i), font=font)
            i += 1
        return copied_image

        #copied_image.show()