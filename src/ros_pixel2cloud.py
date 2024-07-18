import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
import image_geometry, numpy
from cv_bridge import CvBridge
import math
'''    
    def projectPixelRegionTo3d(self, uv1, uv2):
        h = uv2[1] - uv1[1]
        w = uv2[0] - uv1[0]
        result = numpy.empty((h, w, 3))
        u_range = numpy.arange(uv1[0], uv2[0])
        v_range = numpy.arange(uv1[1], uv2[1])
        u_mesh, v_mesh = numpy.meshgrid(u_range, v_range)
        uv_mesh = numpy.stack((u_mesh, v_mesh), axis=-1)
        xyz_mesh = self.projectPixelTo3d(uv_mesh)
        result[:, :, 0] = xyz_mesh[:, :, 0]
        result[:, :, 1] = xyz_mesh[:, :, 1]
        result[:, :, 2] = xyz_mesh[:, :, 2]
        return result
'''


def callback(data):
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
    print(type(cv_image))
    cam_model = image_geometry.PinholeCameraModel()
    x1, y1, x2, y2= 0, 0, 360, 640
    d_ref=cv_image[y2,x2]
    
    d = cv_image[y1:y2, x1:x2]
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
    
    xyz_ref=cam_model.projectPixelTo3dRay((y2,x2))
    

    xyz_mesh= numpy.stack((x_mesh, y_mesh, z_mesh), axis=-1)
    point_cloud=d[:, :, numpy.newaxis]*xyz_mesh
    print(point_cloud[-1,-1],d_ref*numpy.array(xyz_ref))
    

def get_info(data):
    global info
    info = data

def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/camera/aligned_depth_to_color/camera_info", CameraInfo, get_info)
    rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, callback)
    rospy.spin()

if __name__ == '__main__':
    bridge = CvBridge()

    listener()
