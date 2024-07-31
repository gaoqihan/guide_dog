import os
import rospy
from std_msgs.msg import Int32
import subprocess

recording = False
process = None

def callback(data):
    global recording, process
    if data.data == 1:
        if not recording:
            # Ensure the tmp directory exists
            if not os.path.exists('./tmp'):
                os.makedirs('./tmp')
            
            # Define the path for the output file
            output_file = './tmp/recording.wav'
            
            # Command to record audio using audio_capture
            command = ['arecord', '-D', 'plughw:1,0', '-f', 'cd', '-t', 'wav', '-r', '44100', '-c', '1', output_file]
            
            # Start recording
            rospy.loginfo("Recording started...")
            process = subprocess.Popen(command)
            recording = True
        else:
            # Stop recording
            rospy.loginfo("Recording stopped...")
            process.terminate()
            process.wait()
            recording = False
            rospy.loginfo(f"Recording saved to {output_file}")

def listener():
    # Initialize the ROS node
    rospy.init_node('audio_recorder', anonymous=True)
    
    # Subscribe to the ROS topic
    rospy.Subscriber('audio_record_topic', Int32, callback)
    
    # Keep the node running
    rospy.spin()

if __name__ == '__main__':
    listener()