

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
        self.model=model
        if self.model=="default":
            self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
            self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to("cuda")
        elif self.model=="nano":
            self.predictor = OwlPredictor(
                "google/owlvit-base-patch32",
                image_encoder_engine="/opt/nanoowl/data/owl_image_encoder_patch32.engine"
            )


    def detect(self, image, texts):
        #print("start detect objects",torch.cuda.mem_get_info())
        if self.model=="default":
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

            results=box

        
            del outputs
            del inputs
            del target_sizes
            torch.cuda.empty_cache()
            #print("finish detect objects",torch.cuda.mem_get_info())
            #print(type(results))
            return results
        elif self.model=="nano":
            text_encodings = self.predictor.encode_text(texts)
            results = self.predictor.predict(image=image, text=texts,text_encodings=text_encodings, threshold=0.1)

            #print(output)
            return results


    
    def displayBoundingBox(self,image,results,text):
        if self.model=="default":

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
                draw.text(xy=(x1, y1), text=str(i), font=font, fill="green")
                i += 1
            return copied_image

            #copied_image.show()
        elif self.model=="nano":
            copied_image = draw_owl_output(image, results, text=text, draw_text=True)
            #copied_image.show()
            return copied_image
