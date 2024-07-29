

import base64
import requests
from openai import OpenAI
import os
import re
import shutil

class GPTCaller:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        directory = "./tmp/temp"

        # Remove the directory if it exists
        if os.path.exists(directory):
            shutil.rmtree(directory)

        # Create a new empty directory
        os.makedirs(directory, exist_ok=True)

    def encode_image(self,image):
        if type(image) == str:
            with open(image, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        else:
            directory="./tmp/temp"
            extension = ".png"

            files = os.listdir(directory)
            
            # Filter files based on the pattern 'temp<number>.png'
            pattern = re.compile(rf"^(\d+){extension}$")
            indices = [int(pattern.match(f).group(1)) for f in files if pattern.match(f)]
            
            # Determine the next index
            if indices:
                next_index = max(indices) + 1
            else:
                next_index = 0
            
            # Construct the new file name
            new_file_name = f"{next_index}{extension}"
            new_file_path = os.path.join(directory, new_file_name)
            
            # Save the image
            image.save(new_file_path)
            
            # Return the encoded image or the path, as needed
            return self.encode_image(new_file_path)

    def create_prompt(self, user_prompt_list=[],system_prompt_list=[]):
        messages = []
        content=[]
        for item in system_prompt_list:
            content.append({
                    "type": "text",
                    "text": item
                })
            

        messages.append({
            "role": "system",
            "content": content
        })
        
        content=[]

        for item in user_prompt_list:

            if type(item) == str:
                content.append({
                    "type": "text",
                    "text": item
                })
            
            else:
                base64_image=self.encode_image(item)

                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "low"

                    }
                })


        messages.append({
            "role": "user",
            "content": content
        })
        self.messages=messages

    def call(self,temperature=0):

        completion = self.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=self.messages,temperature=temperature
        )

        #print(completion.choices[0].message)

        return completion.choices[0].message.content