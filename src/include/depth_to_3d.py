import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
import image_geometry, numpy
from cv_bridge import CvBridge,CvBridgeError
import math
import os
import matplotlib.pyplot as plt
import shutil


def get3d_mask(image,mask,info,index):
    bridge = CvBridge()
    cam_model = image_geometry.PinholeCameraModel()
    #d_ref=image[y2,x2]    

    plt.savefig(f"./tmp/cropped_depth/{str(index)}_original.png")    

    cam_model.fromCameraInfo(info)

    height, width = image.shape[:2]

    # Create meshgrid
    u_mesh, v_mesh = np.meshgrid(np.arange(width), np.arange(height))

    real_z = image * 0.001
    x_mesh=(u_mesh-cam_model.cx())/cam_model.fx()*real_z
    y_mesh=(v_mesh-cam_model.cy())/cam_model.fy()*real_z
    point_cloud=numpy.stack((x_mesh, y_mesh, real_z), axis=-1)
    
    
    masked_rel_points=mask[:,:,np.newaxis]*point_cloud
    sum_result=np.sum(masked_rel_points, axis=(0, 1))
    best_rel_point=sum_result/np.sum(mask)    

    return best_rel_point



def get3d_point(image,point,info,index):
    bridge = CvBridge()
    cam_model = image_geometry.PinholeCameraModel()
    x, y= point
    #d_ref=image[y2,x2]
    
    d_ref=image[y,x]

    real_z = image[y,x] * 0.001
    cam_model.fromCameraInfo(info)


    real_x=(x-cam_model.cx())/cam_model.fx()*real_z
    real_y=(y-cam_model.cy())/cam_model.fy()*real_z
    return np.array((real_x,real_y,real_z))



import cv2
import rospy
from sensor_msgs.msg import Image
import PIL
from torchvision.ops import box_convert
import torch
import tf
import tf2_ros
import tf.transformations as tft
import geometry_msgs.msg
from geometry_msgs.msg import PoseStamped, Pose
import numpy as np
import time
from std_msgs.msg import Int8, Empty

from tf.transformations import quaternion_from_euler
import math

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


bridge = CvBridge()

global_frame = "map"

class MapBridge:
    def __init__(self):
        self.listener = tf.TransformListener()
        self.marker_pub=   rospy.Publisher('/markers', MarkerArray, queue_size=10)
    #def transform_point(dx, dy, translation, rotation):
    def transform_point(self,dx, dy, dz, translation, rotation):
        # Create a 4x4 transformation matrix from translation and rotation (quaternion)
        transform_matrix = tft.quaternion_matrix(rotation)
        transform_matrix[0:3, 3] = translation

        # Object position in the body frame (assuming z = 0)
        #point_body = np.array([dx, dy, 0, 1])
        point_body = np.array([dx, dy, dz, 1])

        # Transform the point to the map frame
        point_map = np.dot(transform_matrix, point_body)
        return point_map[0:3]

    #def get_object_position_in_map(dx_body, dy_body, cam_id):
    def get_object_position_in_map(self,dx_body, dy_body, dz_body, cam_id):


        try:
            # Get the transform from /map to /body
            #transform = tf_buffer.lookup_transform('map', 'body', rospy.Time(0))
            if cam_id == 1:
                (trans,rot) = self.listener.lookupTransform(global_frame, '/cam2', rospy.Time(0))
            else:
                (trans,rot) = self.listener.lookupTransform(global_frame, '/cam2', rospy.Time(0))


            print(trans, rot)

            # Object's position in the /body frame
            object_position_body = geometry_msgs.msg.PoseStamped()
            object_position_body.header.frame_id = "body"
            object_position_body.header.stamp = rospy.Time.now()
            object_position_body.pose.position.x = dx_body
            object_position_body.pose.position.y = dy_body
            object_position_body.pose.position.z = dz_body

            object_position_in_map = self.transform_point(dx_body, dy_body, dz_body, trans, rot)
            print("Object position in /map frame:", object_position_in_map)

            #p_in_base = listener.transformPose("/map", object_position_body)
            #print("p in base" , p_in_base)
            #object_position_body.point.z = 0  # Assuming the object is on the same plane

            # Transform this position to the /map frame
            #object_position_map = tf_buffer.transform(object_position_body, "map", rospy.Duration(1))

            return object_position_in_map
            #return None

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr(f"Transform error: {e}")
            return None

    def publish_markers(self,points_to_pub):

        marker_array = MarkerArray()
        color = [(1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0), (1.0, 0.0, 1.0, 1.0)]
        for idx, point in enumerate(points_to_pub):
            marker = Marker()
            #marker.header.frame_id = "map"
            marker.header.frame_id = global_frame
            marker.header.stamp = rospy.Time.now()
            marker.ns = "points_array"
            marker.id = idx
            marker.type = marker.SPHERE  # Use SPHERE shape for better visibility
            marker.action = marker.ADD

            marker.pose.position.x = point[0]
            marker.pose.position.y = point[1]
            marker.pose.position.z = point[2]
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2  # size of the sphere
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = color[idx]

            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

    