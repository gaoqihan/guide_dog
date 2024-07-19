import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
import image_geometry, numpy
from cv_bridge import CvBridge
import math

def get3d(image,bounding_box,info):
    bridge = CvBridge()
    cam_model = image_geometry.PinholeCameraModel()
    x1, y1, x2, y2= bounding_box
    #d_ref=image[y2,x2]
    
    d = image[y1:y2, x1:x2]
    cam_model.fromCameraInfo(info)
    uv1 = (x1, y1)
    uv2 = (x2, y2)

    u_range = numpy.arange(uv1[0], uv2[0])
    v_range = numpy.arange(uv1[1], uv2[1])
    u_mesh, v_mesh = numpy.meshgrid(u_range, v_range)

    x_mesh=(v_mesh-cam_model.cx())/cam_model.fx()
    y_mesh=(u_mesh-cam_model.cy())/cam_model.fy()
    
    norm = numpy.sqrt(x_mesh*x_mesh + y_mesh*y_mesh + 1)

    
    x_mesh /= norm
    y_mesh /= norm
    z_mesh = 1.0 / norm
    
    #xyz_ref=cam_model.projectPixelTo3dRay((y2,x2))
    

    xyz_mesh= numpy.stack((x_mesh, y_mesh, z_mesh), axis=-1)
    point_cloud=d[:, :, numpy.newaxis]*xyz_mesh
    return point_cloud