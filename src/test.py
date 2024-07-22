
#from groundingdino.util.inference import load_model, load_image, predict, annotate
from groundingdino.util.inference import load_model, load_image, annotate, preprocess_caption, get_phrases_from_posmap
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import groundingdino.datasets.transforms as T
import message_filters
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

#TEXT_PROMPT = "chair . person . dog ."
#TEXT_PROMPT = "chair"
#TEXT_PROMPT = "door"
TEXT_PROMPT = "head"
#TEXT_PROMPT = "door handle"
#TEXT_PROMPT = "handle"
#TEXT_PROMPT = "knob"
BOX_TRESHOLD = 0.35
#BOX_TRESHOLD = 0.45
#BOX_TRESHOLD = 0.3
TEXT_TRESHOLD = 0.25
bridge = CvBridge()
object_seg_enabled = True

global_frame = "camera_init"

def micCallback(msg):
    global object_seg_enabled
    print("mic callback = ", msg.data)
    if msg.data == 2:
        object_seg_enabled = True


#def transform_point(dx, dy, translation, rotation):
def transform_point(dx, dy, dz, translation, rotation):
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
def get_object_position_in_map(dx_body, dy_body, dz_body, cam_id):


    try:
        # Get the transform from /map to /body
        #transform = tf_buffer.lookup_transform('map', 'body', rospy.Time(0))
        if cam_id == 1:
            (trans,rot) = listener.lookupTransform(global_frame, '/cam1', rospy.Time(0))
        else:
            (trans,rot) = listener.lookupTransform(global_frame, '/cam2', rospy.Time(0))


        print(trans, rot)

        # Object's position in the /body frame
        object_position_body = geometry_msgs.msg.PoseStamped()
        object_position_body.header.frame_id = "body"
        object_position_body.header.stamp = rospy.Time.now()
        object_position_body.pose.position.x = dx_body
        object_position_body.pose.position.y = dy_body
        object_position_body.pose.position.z = dz_body

        object_position_in_map = transform_point(dx_body, dy_body, dz_body, trans, rot)
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

def predict(
        model,
        image: torch.Tensor,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        device: str = "cuda",
        remove_combined: bool = False):

    caption = preprocess_caption(caption=caption)

    #model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)

    mask = prediction_logits.max(dim=1)[0] > box_threshold
    logits = prediction_logits[mask]  # logits.shape = (n, 256)
    boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
   
    phrases = [
        get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
        for logit
        in logits
    ]

    return boxes, logits.max(dim=1)[0], phrases

#def image_callback(msg):
#def image_callback(msg, depth_image):

def publish_markers(points_to_pub):

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

    marker_pub.publish(marker_array)

def image_callback(msg, depth_image, color2, depth2):

    #time_diff = msg.header.stamp - rospy.Time.now()
    print("here")

    global object_seg_enabled
    if object_seg_enabled == False:
        return

    curr_time = msg.header.stamp
    delayed_secs = (rospy.Time.now() - msg.header.stamp).to_sec()
    print("ros tim = ", rospy.Time.now())
    #print("msg header stamp = ", msg.header.stamp)
    print("delayed secs = ", delayed_secs)
    if delayed_secs > 0.5:
    #if delayed_secs > 0.1:
        return

    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        depth_img = bridge.imgmsg_to_cv2(depth_image, "16UC1")
        cv2_img2 = bridge.imgmsg_to_cv2(color2, "bgr8")
        depth_img2 = bridge.imgmsg_to_cv2(depth2, "16UC1")
        print(cv2_img.shape, depth_img.shape, cv2_img2.shape, depth_img2.shape)
    except CvBridgeError as e:
        print(e)
        return
    else:
        # Save your OpenCV2 image as a jpeg
        #resized = cv2.resize(cv2_img, (464, 400), interpolation = cv2.INTER_AREA)
        #image_source, image = load_image(img)
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        cv2_img = cv2.hconcat([cv2_img, cv2_img2])
        #1273 1 1665 587
        #cv2_img = cv2_img[1:587, 1273:1665, :]
        #969 1 1723 661
        #cv2_img = cv2_img[1:661, 969:1723, :]
        pil_img = PIL.Image.fromarray(cv2_img)
        image_transformed, _ = transform(pil_img, None)
       
        global model
        start_ts = time.time()
        boxes, logits, phrases = predict(
            model=model,
            image=image_transformed,
            caption=TEXT_PROMPT,
            #box_threshold=BOX_TRESHOLD,
            box_threshold=0.45,
            text_threshold=TEXT_TRESHOLD
        )
        end_ts = time.time()
        print("prediction time = ", end_ts - start_ts)

       
        annotated_frame = annotate(image_source=cv2_img, boxes=boxes, logits=logits, phrases=phrases)
        #print("box confidence")
        confidence = logits.numpy()
        print("confidence = ", confidence)

        def get_depth_value(x1, y1):
            if x1 <  1280:
                real_z = depth_img[y1, x1] * 0.001
                real_x = -(x1 - cx) / fx * real_z
                real_y = (y1 - cy) / fy * real_z
                #object_pos = get_object_position_in_map(real_z, real_x, 1)
                object_pos = get_object_position_in_map(real_z, real_x, real_y, 1)
            else:
                real_z = depth_img2[y1, x1-img_width] * 0.001
                real_x = -(x1 - cx - img_width) / fx * real_z
                real_y = (y1 - cy) / fy * real_z
                #object_pos = get_object_position_in_map(real_z, real_x, 2)
                object_pos = get_object_position_in_map(real_z, real_x, real_y, 2)

            return object_pos

        if boxes.shape[0] > 0:
            h, w, _ = cv2_img.shape
            boxes_origin = boxes * torch.Tensor([w, h, w, h])
            xyxy = box_convert(boxes=boxes_origin, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            #detections = sv.Detections(xyxy=xyxy)
            #import pdb
            #pdb.set_trace()

            x1 = int(xyxy[0][0])
            y1 = int(xyxy[0][1])
            x2 = int(xyxy[0][2])
            y2 = int(xyxy[0][3])
            print("bounding box = ", x1, y1, x2, y2)

            object_pos1 = get_depth_value(x1, y1)
            object_pos2 = get_depth_value(x2, y1)
            object_pos3 = get_depth_value(x2, y2)

            if object_pos1 is None or object_pos2 is None:
                return

            x_comp = object_pos1[0] - object_pos2[0]
            y_comp = object_pos1[1] - object_pos2[1]


           
            # Calculate vectors in the plane
            AB = object_pos2 - object_pos1
            AC = object_pos3 - object_pos1
           
            # Compute the normal vector by cross product
            print("AB = ", AB)
            print("AC = ", AC)
            #normal_vector = np.cross(AB, AC)
            normal_vector = np.cross(AC, AB)

           
            # Normalize the vector (optional)
            normal_vector = normal_vector / np.linalg.norm(normal_vector)

            door_center_x = ((object_pos1[0] + object_pos3[0]) * 0.5)
            door_center_y = ((object_pos1[1] + object_pos3[1]) * 0.5)
            door_center_z = ((object_pos1[2] + object_pos3[2]) * 0.5)

            print("door center x , y , z = ", door_center_x, door_center_y, door_center_z)

            publish_markers([object_pos1, object_pos2, object_pos3, [door_center_x, door_center_y, door_center_z]])
           
            print(object_pos1)
            print(object_pos2)
            print(object_pos3)
            print("Normal Vector:", normal_vector)

            theta = math.atan2(y_comp, x_comp)

            #theta2 = math.atan2(normal_vector[0], -normal_vector[1])
            theta2 = math.atan2(normal_vector[1], normal_vector[0])


            ### add an additional delta to the normal vector

            print("object pos1 ", object_pos1)
            print("object pos2 ", object_pos2)
            print("object pos3 ", object_pos3)

            print("theta = ", theta)
            print("theta2 = ", theta2)

            print("x comp , y comp", x_comp, y_comp)

            quaternion = quaternion_from_euler(0, 0, theta)
            quaternion2 = quaternion_from_euler(0, 0, theta2)
            print("quaternion = ", quaternion)


            ## compute the door orientation vector here



            center_x = int((x1 + x2) * 0.5)
            center_y = int((y1 + y2) * 0.5)

            radius = 5
            color = (0, 255, 0)  # Green color in BGR
            thickness = 2  # in pixels, -1
            cv2.circle(annotated_frame, (center_x, center_y), radius, color, thickness)
            image_pub.publish(bridge.cv2_to_imgmsg(annotated_frame))

            print("publish original frame");

            cv2_img = cv2_img[:, x1:x2, :]
            pil_img = PIL.Image.fromarray(cv2_img)
            image_transformed, _ = transform(pil_img, None)
            handle_boxes, logits, phrases = predict(
                model=model,
                image=image_transformed,
                #caption="door knob",
                caption="door knob",
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD
            )

            print("handle boxes shape = ", handle_boxes.shape[0])

            if handle_boxes.shape[0] > 0:

                annotated_frame_crop = annotate(image_source=cv2_img, boxes=handle_boxes, logits=logits, phrases=phrases)

                h, w, _ = cv2_img.shape
                boxes_origin = handle_boxes * torch.Tensor([w, h, w, h])
                xyxy = box_convert(boxes=boxes_origin, in_fmt="cxcywh", out_fmt="xyxy").numpy()

                x1_handle = int(xyxy[0][0])
                y1_handle = int(xyxy[0][1])
                x2_handle = int(xyxy[0][2])
                y2_handle = int(xyxy[0][3])


                center_x_handle = int((x1_handle + x2_handle) * 0.5)
                center_y_handle = int((y1_handle + y2_handle) * 0.5)

                center_x_handle += x1


                if center_x_handle < 1280:
                    ## cam1
                   
                    real_z = depth_img[center_y_handle, center_x_handle] * 0.001
                    real_x = -(center_x_handle - cx) / fx * real_z
                    real_y = (center_y_handle - cy) / fy * real_z
                    object_pos = get_object_position_in_map(real_z, real_x, real_y, 1)

                else:
                    ## cam2
                    real_z = depth_img2[center_y_handle, center_x_handle - img_width] * 0.001
                    real_x = -(center_x_handle - img_width - cx) / fx * real_z
                    real_y = (center_y_handle - cy) / fy * real_z
                    object_pos = get_object_position_in_map(real_z, real_x, real_y, 2)




                if object_pos is not None:


                    pose = PoseStamped()
                    pose.header.seq = 0;
                    #pose.header.frame_id =  "map";
                    pose.header.frame_id =  global_frame;
                   

                    door_knob_pos = object_pos
                    door_knob_pos[0] -= normal_vector[0]
                    door_knob_pos[1] -= normal_vector[1]
                    #pose.pose.position.x = object_pos[0];
                    #pose.pose.position.y = object_pos[1];
                    pose.pose.position.x = door_knob_pos[0];
                    pose.pose.position.y = door_knob_pos[1];
                    pose.pose.position.z = 0;

                    #pose.pose.orientation.x = 0;
                    #pose.pose.orientation.y = 0;
                    #pose.pose.orientation.z = 0;
                    #pose.pose.orientation.w = 1;

                    #pose.pose.orientation.x = quaternion[0];
                    #pose.pose.orientation.y = quaternion[1];

                    #pose.pose.orientation.z = quaternion[2];
                    #pose.pose.orientation.w = quaternion[3];

                    pose.pose.orientation.x = quaternion2[0];
                    pose.pose.orientation.y = quaternion2[1];

                    pose.pose.orientation.z = quaternion2[2];
                    pose.pose.orientation.w = quaternion2[3];

                    for i in range(5):
                        goal_pub.publish(pose)

                    pose = PoseStamped()
                    pose.header.seq = 0;
                    #pose.header.frame_id =  "map";
                    pose.header.frame_id =  global_frame;
                    print("center x = ", door_center_x)
                    print("center y = ", door_center_y)

                    normal_vector_2d = normal_vector[:2]
                    #norm = np.linalg.norm(normal_vector_2d)
                    #normal_vector_2d = normal_vector_2d / norm

                    door_center_x += normal_vector_2d[0]
                    door_center_y += normal_vector_2d[1]  

                    pose.pose.position.x = door_center_x;
                    pose.pose.position.y = door_center_y;

                    ### add delta based on normal direction here

                    #theta2 = math.atan2(normal_vector[0], -normal_vector[1])

                    pose.pose.position.z = 0;

                    #pose.pose.orientation.x = 0;
                    #pose.pose.orientation.y = 0;
                    #pose.pose.orientation.z = 0;
                    #pose.pose.orientation.w = 1;

                    pose.pose.orientation.x = quaternion2[0];
                    pose.pose.orientation.y = quaternion2[1];

                    pose.pose.orientation.z = quaternion2[2];
                    pose.pose.orientation.w = quaternion2[3];

                    for i in range(5):
                        goal2_pub.publish(pose)
                        #goal_pub.publish(pose)



                radius = 5
                color = (0, 255, 0)  # Green color in BGR
                thickness = 2  # in pixels, -1
                cv2.circle(annotated_frame_crop, (center_x_handle - x1, center_y_handle), radius, color, thickness)
                image_crop_pub.publish(bridge.cv2_to_imgmsg(annotated_frame_crop))


            #position_in_map = get_object_position_in_map(real_z, real_x)
            #print("position in map ", position_in_map)
           
            #object_seg_enabled = False

            ### need to get the current robot pose as well, from the transform maybe
            ### then multiple with the pose to get the object pose in the map


        #cv2.imwrite(img+"_pred.jpg", annotated_frame)
        #cv2.imwrite('camera_image_' + str(image_count) + '.jpg', resized)
        #cv2.imshow("annotated_frame", annotated_frame)
        #cv2.imshow("annotated_frame", cv2_img)
        #cv2.waitKey(1)



if __name__ == '__main__':


    print("is cuda available ", torch.cuda.is_available())
    rospy.init_node('object_finder')

    model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
    model = model.to('cuda')


    #tf_buffer = tf2_ros.Buffer()
    #listener = tf2_ros.TransformListener(tf_buffer)
    listener = tf.TransformListener()

    goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)
    goal2_pub = rospy.Publisher("/move_base_simple/goal2", PoseStamped, queue_size=10)
    image_pub = rospy.Publisher("/camera/color/image_seg", Image, queue_size=10)
    image_crop_pub = rospy.Publisher("/camera/color/image_seg_crop", Image, queue_size=10)

    rospy.Subscriber("/VoiceCommand", Int8, micCallback)

    # Wait for the transform to become available
    rospy.sleep(1)

    print("finish model loading")
   
    #image_topic = "/camera/color/image_raw"
    image_topic = "/usb_cam/image_raw"
    #rospy.Subscriber(image_topic, Image, image_callback)

    color_sub = message_filters.Subscriber('/cam_1/color/image_raw', Image)
    depth_sub = message_filters.Subscriber('/cam_1/aligned_depth_to_color/image_raw', Image)

    color2_sub = message_filters.Subscriber('/cam_2/color/image_raw', Image)
    depth2_sub = message_filters.Subscriber('/cam_2/aligned_depth_to_color/image_raw', Image)

    marker_pub = rospy.Publisher('/door_corners', MarkerArray, queue_size=10)

    #fx = 606.0199584960938
    #fy = 605.2523193359375
    #cx = 327.2325439453125
    #cy = 248.458648681640
    #K: [606.0199584960938, 0.0, 327.2325439453125, 0.0, 605.2523193359375, 248.45864868164062, 0.0, 0.0, 1.0]

    #K: [909.0299072265625, 0.0, 650.848876953125, 0.0, 907.8784790039062, 372.6879577636719, 0.0, 0.0, 1.0]
    img_width = 1280
    img_height = 720
    fx = 909.0299072265625
    fy = 907.8784790039062
    cx = 650.848876953125
    cy = 372.6879577636719

    # TimeSynchronizer to synchronize the topics
    #ts = message_filters.TimeSynchronizer([color_sub, depth_sub], 10)
    ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub, color2_sub, depth2_sub], queue_size=10, slop=0.5)
    #ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], queue_size=10, slop=0.5)
    #ts = message_filters.TimeSynchronizer([color_sub, depth_sub, color2_sub, depth2_sub], 10)
    ts.registerCallback(image_callback)
   
    rospy.spin()

