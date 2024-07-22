import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
import image_geometry, numpy
from cv_bridge import CvBridge
import math
import os
import matplotlib.pyplot as plt
import shutil

def get3d(image,bounding_box,info):
    bridge = CvBridge()
    cam_model = image_geometry.PinholeCameraModel()
    x1, y1, x2, y2= bounding_box
    #d_ref=image[y2,x2]
    
    d = image[y1:y2, x1:x2]
    os.makedirs("./tmp/cropped_depth", exist_ok=True)
    plt.figure(figsize=(10,10))
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    #plt.show() 
    index=len(os.listdir("./tmp/cropped_depth"))
    if index==5:
        for item in os.listdir("./tmp/cropped_depth"):
            item_path = os.path.join("./tmp/cropped_depth", item)
            try:
                os.remove(item_path)  # Remove files and links
                print(f"Deleted {item_path}")
            except Exception as e:
                print(f"Failed to delete {item_path}. Reason: {e}")

    plt.savefig(f"./tmp/cropped_depth/{str(index)}.png")
    
    d_ref=image[int((y2-y1)/2+y1), int((x2-x1)/2+x1)]

    real_z = image[int((y2-y1)/2+y1), int((x2-x1)/2+x1)] * 0.001
    cam_model.fromCameraInfo(info)
    uv1 = (x1, y1)
    uv2 = (x2, y2)

    u_range = numpy.arange(uv1[0], uv2[0])
    v_range = numpy.arange(uv1[1], uv2[1])
    u_mesh, v_mesh = numpy.meshgrid(u_range, v_range)

    x_mesh=(u_mesh-cam_model.cx())/cam_model.fx()
    y_mesh=(v_mesh-cam_model.cy())/cam_model.fy()
    
    norm = numpy.sqrt(x_mesh*x_mesh + y_mesh*y_mesh + 1)

    #print(x_mesh.shape,y_mesh.shape,norm.shape,(x_mesh/norm).shape)
    x_mesh /= norm
    y_mesh /= norm
    z_mesh = 1.0 / norm
    #z_mesh = numpy.ones_like(x_mesh) * 1.0
    #print(z_mesh.shape,)
    center_point=(int((x2-x1)/2+x1), int((y2-y1)/2+y1))
    print("center point is",center_point)
    xyz_ref=d_ref*numpy.array(cam_model.projectPixelTo3dRay(center_point))
    print("real_z is", real_z,"ref_z",xyz_ref[2],"image_value",d_ref)


    xyz_mesh= numpy.stack((x_mesh, y_mesh, z_mesh), axis=-1)
    point_cloud=d[:, :, numpy.newaxis]*xyz_mesh
    return point_cloud