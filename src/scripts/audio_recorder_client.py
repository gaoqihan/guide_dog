#!/usr/bin/env python

import rospy
from std_srvs.srv import Trigger, TriggerRequest

def call_service():
    # Initialize the ROS node
    rospy.init_node('trigger_client', anonymous=True)
    
    # Wait for the service to be available
    rospy.wait_for_service('audio_record_service')
    
    try:
        # Create a service proxy
        service_proxy = rospy.ServiceProxy('audio_record_service', Trigger)
        
        # Create a request object
        request = TriggerRequest()
        
        # Call the service to start recording
        rospy.loginfo("Calling service to start recording...")
        response = service_proxy(request)
        rospy.loginfo(f"Service response: {response.message}")
        
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

if __name__ == '__main__':
    try:
        call_service()
    except rospy.ROSInterruptException:
        pass