import math
import numpy as np
import os
import pycuda.driver as cuda
import tensorrt as trt
from PIL import Image
import cv2
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image as Image_type
from sensor_msgs.msg import CompressedImage
from ros_numpy.image import image_to_numpy, numpy_to_image
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
import std_msgs
import threading
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header, Float32MultiArray

mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)

def postprocess(data):
    num_classes = 2
    palette = np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = np.array([palette * i % 255 for i in range(num_classes)]).astype("uint8")
    img = Image.fromarray(data.astype('uint8'), mode='P')
    img.putpalette(colors)
    return img

def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def infer(engine, img):
    input_image = img
    image_width, image_height = img.shape[1], img.shape[2]
    with engine.create_execution_context() as context:
        ctx = cuda.Context.attach()
        context.set_binding_shape(engine.get_binding_index("input"), (1, 3, image_height, image_width))
        bindings = []

        for binding in engine:
            binding_idx = engine.get_binding_index(binding)
            size = trt.volume(context.get_binding_shape(binding_idx))
            dtype = trt.nptype(engine.get_binding_dtype(binding))

            if engine.binding_is_input(binding):
                input_buffer = np.ascontiguousarray(input_image)
                input_memory = cuda.mem_alloc(input_buffer.nbytes)
                bindings.append(int(input_memory))
                cuda.memcpy_htod(input_memory, input_buffer)

            else:
                output_buffer = np.empty(size, dtype)
                output_memory = cuda.mem_alloc(output_buffer.nbytes)
                bindings.append(int(output_memory))

        stream = cuda.Stream()

        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
        stream.synchronize()
        ctx.pop()
        # ctx.detach()
        # Release GPU memory
        input_memory.free()
        output_memory.free()
        # Return the output as needed
        return output_buffer.copy()

def extract_exact_border_and_adjacent(mask):

    kernel = np.ones((5,5),np.uint8)
    dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)
    edges = np.zeros_like(mask)
    edges[1:-1, 1:-1] = dilated_mask[1:-1, 1:-1] & (~dilated_mask[:-2, 1:-1] | ~dilated_mask[1:-1, :-2] | ~dilated_mask[2:, 1:-1] | ~dilated_mask[1:-1, 2:])

    return edges.astype(int)


def set_outer_border_to_zero(array):
    array[[0, -1], :] = 0
    array[:, [0, -1]] = 0
    return array


def generate_laserscan(result_mask, depth_image, x, y, laserscan_pub, image_header):
    laser_scan_msg = LaserScan()
    laser_scan_msg.header = image_header
    laser_scan_msg.header.frame_id = 'camera_link'
    laser_scan_msg.angle_min = -3.142  # Minimum angle [rad]
    laser_scan_msg.angle_max = 3.142   # Maximum angle [rad]
    laser_scan_msg.angle_increment = 0.01  # Angular distance between measurements [rad]
    laser_scan_msg.time_increment = 0.000001  # Time between measurements [seconds]
    laser_scan_msg.range_min = 0.0  # Minimum range value [meters]
    laser_scan_msg.range_max = 15.0  # Maximum range value [meters]
    num_of_scan_point = int((laser_scan_msg.angle_max - laser_scan_msg.angle_min) / laser_scan_msg.angle_increment) + 1
    laser_scan_msg.scan_time = num_of_scan_point * laser_scan_msg.time_increment
    Z = set_outer_border_to_zero(extract_exact_border_and_adjacent(result_mask)) * np.nan_to_num(depth_image)/1000
    # Calculate X
    X = x * Z
    # Calculate angles for all points
    angles = np.arctan2(-X, Z)
    # Calculate indices for all points
    index = ((angles - (-3.142)) / 0.01).astype(int)
    # Filter out invalid indices
    valid_indices_mask = (index >= 0) & (index < num_of_scan_point)
    # Calculate distances for all points
    distances = np.sqrt(X ** 2 + Z ** 2)
    # Filter out distances outside of range
    valid_distances_mask = distances <= 15.0
    # Create a mask for closer points
    closer_points_mask = (distances < np.inf) & valid_indices_mask & valid_distances_mask
    # Initialize ranges with infinity
    laser_scan_msg.ranges = np.full(num_of_scan_point, np.inf)
    # Update laser scan where distances are closer
    laser_scan_msg.ranges[index[closer_points_mask]] = distances[closer_points_mask]
    # Publish the updated laser scan message
    laserscan_pub.publish(laser_scan_msg)


class SegRos:
    def __init__(self, predictor):
        self.predictor = predictor
        self.bridge = CvBridge()
        self.box_coords = []
        self.yolo_image = None
        # Retrieve parameters directly from ROS parameter server
        self.size = rospy.get_param("~model_size", 480)
        # self.person_location = rospy.get_param("~person_location", False)

        self.imgsz_h = rospy.get_param("~imgsz_h")
        self.imgsz_w = rospy.get_param("~imgsz_w")
        self.fx,self.fy = rospy.get_param("~fx"),rospy.get_param("~fy")
        self.cx,self.cy = rospy.get_param("~cx"),rospy.get_param("~cy")
        u = np.arange(self.imgsz_w)
        v = np.arange(self.imgsz_h )
        u_grid, v_grid = np.meshgrid(u, v)
        x_transform= (u_grid - self.cx) / self.fx
        y_transform  = (v_grid - self.cy) / self.fy
        self.x = x_transform
        self.y = y_transform    
        # Initialize publishers and subscribers with topic names from parameter server
        self.laserscan_pub = rospy.Publisher(rospy.get_param("~output_topic_laser_camera"), LaserScan, queue_size=1)
        self.laserscan_person = rospy.Publisher(rospy.get_param("~output_topic_laser_person"), LaserScan, queue_size=1)
        self.pointcloud_pub = rospy.Publisher(rospy.get_param("~output_topic_pointcloud"), PointCloud2, queue_size=1)
        self.image_publisher = rospy.Publisher(rospy.get_param("~output_topic_image"), Image_type, queue_size=1)
        
        self.depth_sub = rospy.Subscriber(rospy.get_param("~input_topic_depth"), Image_type, self.depth_callback)  
        self.image_subscriber_yolo = rospy.Subscriber(rospy.get_param("~input_topic_yolo"), Image_type, self.yolo_callback, queue_size=1)
        self.data_subscriber_yolo = rospy.Subscriber(rospy.get_param("~input_topic_person"), Float32MultiArray, self.person_callback, queue_size=1)
        self.is_compressed = rospy.get_param("~is_compressed")  
        if self.is_compressed:
            self.image_subscriber_zed = rospy.Subscriber(rospy.get_param("~input_topic_zed"), CompressedImage, self.zed_callback, queue_size=1)
        else:
            self.image_subscriber_zed = rospy.Subscriber(rospy.get_param("~input_topic_zed"), Image_type, self.zed_callback, queue_size=1)
        
        self.depth_image = None

    def yolo_callback(self, msg):
        self.yolo_image = image_to_numpy(msg)
        # Process yolo_image as needed

    def person_callback(self, msg):
        data = np.array(msg.data)
        self.box_coords=data

    def depth_callback(self, msg):
        self.depth_image = image_to_numpy(msg)

    def zed_callback(self, msg):

        if self.is_compressed:
            compressed_data = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(compressed_data, cv2.IMREAD_COLOR)
        else:
            image = image_to_numpy(msg)[:,:,:3]

        image_header = msg.header
        original_image = image.astype(np.uint8)
        image = cv2.resize(image, (self.size, self.size))
        image = np.array(image, dtype=np.float32).transpose(2, 0, 1)
        image /= 255
        image -= mean
        image /= std
        result_image = infer(self.predictor, image)
        result_image = result_image.reshape((2, self.size, self.size))
        result_image = np.argmax(result_image, 0)
        result_mask = cv2.resize(result_image.astype(np.uint8), (self.imgsz_w, self.imgsz_h))
        result_image = result_image.astype(np.uint8)
        result_image = postprocess(np.reshape(result_image, (self.size, self.size))).convert('RGB')
        result_image = np.array(result_image)
        result_image = cv2.resize(result_image, (self.imgsz_w, self.imgsz_h))
        blended_image = self.blend_images(original_image, result_image)
        self.image_publisher.publish(numpy_to_image(blended_image, encoding='bgr8'))

        if self.depth_image is not None:
            threading.Thread(target=generate_laserscan, args=(result_mask, self.depth_image, self.x, self.y, self.laserscan_pub,image_header)).start()
            # if self.person_location:
            #     threading.Thread(target=generate_laserscan_person, args=(self.box_coords,self.depth_image, self.laserscan_person, image_header)).start()
    
    def blend_images(self, img1, img2):
        if self.yolo_image is not None:
            blended_image = cv2.addWeighted(self.yolo_image, 0.7, img2, 0.3, 0)
        else:
            blended_image = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
        return blended_image

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    rospy.init_node("seg_node")

    # Load the engine file directly from ROS parameter server
    engine_file = rospy.get_param("~engine_file")
    TRT_LOGGER = trt.Logger()
    engine = load_engine(engine_file)
    
    seg_ros = SegRos(predictor=engine)
    seg_ros.spin()