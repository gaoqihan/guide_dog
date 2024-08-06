import rospy
import sys  # Import the sys module to access command-line arguments
from std_srvs.srv import Empty

def pause_video_capture():
    rospy.init_node('pause_video_capture_client', anonymous=True)  # Add anonymous=True to allow multiple instances
    rospy.wait_for_service('pause_video_capture',timeout=1)
    try:
        pause_service = rospy.ServiceProxy('pause_video_capture', Empty)
        pause_service()
        print("Video capture paused successfully.")
        
        
    except rospy.ServiceException as e:
        print(f"Service call failed: {e}")

def resume_video_capture():
    rospy.init_node('resume_video_capture_client', anonymous=True)  # Add anonymous=True to allow multiple instances
    rospy.wait_for_service('resume_video_capture',timeout=1)
    try:
        resume_service = rospy.ServiceProxy('resume_video_capture', Empty)
        resume_service()
        print("Video capture resumed successfully.")
    except rospy.ServiceException as e:
        print(f"Service call failed: {e}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python pause_resume_video.py [p|r]")
    else:
        action = sys.argv[1]
        if action == 'p':
            pause_video_capture()
        elif action == 'r':
            resume_video_capture()
        else:
            print("Invalid argument. Use 'p(pause)' or 'r(resume)'.")