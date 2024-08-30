
import rospy
from std_msgs.msg import String

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
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