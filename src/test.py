import os
import sys
from PIL import Image
from time import time
# Add the '../include' directory to sys.path to import modules from there
script_dir = os.path.dirname(__file__)  # Get the directory where the script is located
include_dir = os.path.join(script_dir, './include')  # Path to the 'include' directory
sys.path.append(include_dir)  # Add 'include' directory to sys.path

from gpt_caller import GPTCaller

caller=GPTCaller()
system_prompt="""You are an AI assistant, your task is to help me extract key word and instruction from a long instruction for guiding blind person. 
"""
        
        

test_pic_path=f"./tmp/bbox_image_4.png"
test_pic = Image.open(test_pic_path)
test_pic.show()
user_prompt=f"Task is :"

user_prompt_list=[user_prompt]
                
response_format= {
    "type": "json_schema",
    "json_schema": {
      "name": "reasoning_schema",
      "strict": True,
      "schema": {
        "type": "object",
        "properties": {
          "reasoning_steps": {
            "type": "array",
            "items": {
              "type": "string"
            },
            "description": "The reasoning steps leading to the final conclusion."
          },
          "keyword": {
            "type": "array",
            "items": {
                "type": "string"
              },
                "description": "The key object of needed to be found baed on instruction. The keyword MUST be simple non-abstract object,\
                                it must be well-defined and common (e.g. car, mouse, apple, keyboard, \
                                traffic light)!!! It MUST be easy to detect with computer vision algorithms.\
                                You can infer what the keyword is based on the intention of the instruction\
                                    The list can contain multiple key objects, as long as they are related to the instruction.",
 

            },


            "key_instruction": {
            "type": "string",
            "description": "The instruction is for guiding a blind person. You must use common sense and social rule to build this instruction.\
                            The instruction contains a summrized instruction for the AI to find object based on original instruction given, and some conditions for the object."
          }
        },
        "required": ["reasoning_steps", "keyword","key_instruction"],
        "additionalProperties": False
      }
    }
  }
caller.create_prompt(user_prompt_list=user_prompt_list,system_prompt_list=[system_prompt],response_format=response_format)

start_time=time()
#response=caller.call(model="gpt-4o-2024-08-06")
response=""
end_time=time()
print(f"Time taken to get response is {end_time-start_time}")
print(f"gpt response is {response}")

import rospy
from std_msgs.msg import String


if __name__ == '__main__':
    try:
        rospy.init_node('find_object_talker', anonymous=True)
        rospy.sleep(1)
        pub = rospy.Publisher('/find_object', String, queue_size=10)
        rospy.sleep(1)

        print("Publisher initialized")
        while not rospy.is_shutdown():
          message=input("Press Enter to publish a message")
          if message=="":
            message=last_message
          last_message=message
          pub.publish(message)
          print(f"Message published: {message}")
        
    except rospy.ROSInterruptException:
        pass