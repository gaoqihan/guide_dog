import os
import rospy
from std_msgs.msg import Int32
from std_srvs.srv import Trigger, TriggerResponse  # Import the Trigger service type
import subprocess

recording = False
process = None

def callback(req):
    global recording, process
    output_file = '/mnt/T7/world_volume/recording.wav'

    if not recording:        
        # Command to record audio using audio_capture
        command = ['arecord', '-f', 'cd', '-t', 'wav', '-r', '44100', '-c', '1', output_file]
        
        # Start recording
        rospy.loginfo("Recording started...")
        process = subprocess.Popen(command)
        recording = True
        return TriggerResponse(success=True, message="Recording started")
    else:
        # Stop recording
        rospy.loginfo("Recording stopped...")
        process.terminate()
        process.wait()
        recording = False
        rospy.loginfo(f"Recording saved to {output_file}")

        # Call the guide_dog_detector.py script
        #detector_command = ['/usr/bin/python3', '/root/catkin_ws/src/guide_dog/src/scripts/guide_dog_detector.py']
        #subprocess.run(detector_command)
        #rospy.loginfo("guide_dog_detector.py script executed.")
        return TriggerResponse(success=True, message="Recording stopped and guide_dog_detector.py executed")

def audio_record_service():
    # Initialize the ROS node
    rospy.init_node('audio_recorder', anonymous=True)
    
    # Define the service server
    service = rospy.Service('audio_record_service', Trigger, callback)
    
    # Keep the node running
    rospy.spin()

if __name__ == '__main__':
    audio_record_service()