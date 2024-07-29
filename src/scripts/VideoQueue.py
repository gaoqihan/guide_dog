import rospy
import actionlib
import cv2
from cv_bridge import CvBridge

import numpy as np
import os
from sensor_msgs.msg import Image
from std_srvs.srv import Empty, EmptyResponse  # Import the Empty service type and its response
from time import time

class VideoQueue(object):
    def __init__(self):
        self.bridge = CvBridge()
        self.color_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.color_callback)
        self.depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)
        self.pause_service = rospy.Service('pause_video_capture', Empty, self.pause_callback)
        self.resume_service = rospy.Service('resume_video_capture', Empty, self.resume_callback)  # Add a service to resume video capture
        self.window = 5
        self.pause = False  # Initialize the pause attribute
        # Create directories if they don't exist
        self.color_dir = os.path.expanduser('~/catkin_ws/src/guide_dog/src/tmp/color')
        self.depth_dir = os.path.expanduser('~/catkin_ws/src/guide_dog/src/tmp/depth')
        os.makedirs(self.color_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)

    def color_callback(self, data):
        self.color_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')

    def depth_callback(self, data):
        if self.pause:
            return

        rate = rospy.Rate(1)  # 1 fps

        number_of_files = len(os.listdir(self.depth_dir))
        if number_of_files < self.window:
            cv2.imwrite(f'{self.color_dir}/{number_of_files}.png', self.color_image)
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="16UC1")
            np.save(f'{self.depth_dir}/{number_of_files}.npy', self.depth_image)
            print(f"Saved image {number_of_files}.npy' at {time()}")
            rate.sleep()
            return

        self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="16UC1")
        os.remove(f'{self.color_dir}/0.png')
        os.remove(f'{self.depth_dir}/0.npy')
        for j in range(1, self.window):
            os.rename(f'{self.color_dir}/{j}.png', f'{self.color_dir}/{j-1}.png')
            os.rename(f'{self.depth_dir}/{j}.npy', f'{self.depth_dir}/{j-1}.npy')

        cv2.imwrite(f'{self.color_dir}/{self.window-1}.png', self.color_image)
        np.save(f'{self.depth_dir}/{self.window-1}.npy', self.depth_image)
        print(f"Saved image at {time()}")

        rate.sleep()
    
    def pause_callback(self, req):
        self.pause = True
        return EmptyResponse()

    def resume_callback(self, req):  # Method to resume video capture
        self.pause = False
        return EmptyResponse()

if __name__ == '__main__':
    rospy.init_node('video_queue_recorder_node')
    server = VideoQueue()
    rospy.spin()