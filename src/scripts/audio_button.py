import rospy
from std_msgs.msg import Bool,Int8  # Assuming the message type is Bool
import subprocess
from time import time



import rospy
from std_srvs.srv import Trigger, TriggerRequest
import subprocess
def call_service():
    # Initialize the ROS node
    
    # Wait for the service to be available
    rospy.wait_for_service('audio_record_service')
    
    try:
        # Create a service proxy
        service_proxy = rospy.ServiceProxy('audio_record_service', Trigger)
        
        # Create a request object
        request = TriggerRequest()
        
        # Call the service to start recording
        response = service_proxy(request)
        if response.message=="Recording stopped and guide_dog_detector.py executed":
            subprocess.run(["cp", "/world/recording.wav", "/root/catkin_ws/src/guide_dog/src/tmp/recording.wav"])
            detector_command = ['/usr/bin/python3', '/root/catkin_ws/src/guide_dog/src/scripts/guide_dog_detector.py']
            subprocess.run(detector_command)
        rospy.loginfo(f"Service response: {response.message}")
        
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

def button_pressed_callback(msg):
    global last_time
    if time()-last_time<=1:
        return
    last_time = time()
    
    if msg.data:  # Check if the button is pressed
        try:
            # Run the command when the button is pressed
            print("Button Press")
            call_service()
            last_time = time()
            #subprocess.run(["/usr/bin/python3", "/root/catkin_ws/src/guide_dog/src/scripts/audio_recorder_client.py"], check=True)
        except subprocess.CalledProcessError as e:
            rospy.logerr(f"Failed to run the command: {e}")

def listener():
    # Initialize the ROS node
    rospy.init_node('audio_button_listener', anonymous=True)
    
    # Create a subscriber to the /button_pressed topic
    rospy.Subscriber('/ButtonPress', Int8, button_pressed_callback,queue_size=1)
    
    # Keep the node running
    rospy.spin()

if __name__ == '__main__':
    last_time = time()

    listener()