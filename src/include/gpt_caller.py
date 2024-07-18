

import base64
import requests
from openai import OpenAI
class GPTCaller:
    def __init__(self):
        self.client = OpenAI()
    
    def encode_image(self,image):
        if type(image) == str:
            with open(image, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        else:
            temp_image_path = "./temp.png"
            image.save(temp_image_path)
            return self.encode_image(temp_image_path)

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

    def call(self):

        completion = self.client.chat.completions.create(
        model="gpt-4o",
        messages=self.messages,temperature=0
        )

        print(completion.choices[0].message)

        return completion.choices[0].message.content