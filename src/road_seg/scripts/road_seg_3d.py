#!/home/delta/archiconda3/envs/project/bin/python3

import sys
print(sys.executable)
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
    kernel = np.ones((8,8),np.uint8)
    dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)

    edges = np.zeros_like(mask)
    edges[1:-1, 1:-1] = dilated_mask[1:-1, 1:-1] & (~dilated_mask[:-2, 1:-1] | ~dilated_mask[1:-1, :-2] | ~dilated_mask[2:, 1:-1] | ~dilated_mask[1:-1, 2:])

    return edges.astype(int)

def set_outer_border_to_zero(array):
    array[[0, -1], :] = 0
    array[:, [0, -1]] = 0
    return array

def generate_point_cloud(result_mask, depth_image, x, y, pointcloud_pub,image_header):
    Z = set_outer_border_to_zero(extract_exact_border_and_adjacent(result_mask)) * np.nan_to_num(depth_image)/1000
    # Z = set_outer_border_to_zero(extract_exact_border_and_adjacent(result_mask)) * np.nan_to_num(depth_image)
    X = x * Z
    Y = y * Z
    mask = np.logical_and.reduce((X >= -5, X <= 5, Y >= -1, Y <= 1, Z > 0.1, Z <= 5))
    X = X[mask]
    Y = Y[mask]
    Z = Z[mask]
    points = np.stack((Z.flatten(), -X.flatten(), Y.flatten()), axis=-1)
    header = image_header
    header.frame_id = "camera_link"
    if len(points) < 100:
        rospy.logwarn("Not enough valid points to create point cloud.")
        return
    # print(len(points))
    pc_msg = pc2.create_cloud_xyz32(header, points)
    pointcloud_pub.publish(pc_msg)

class SegRos(object):
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
        self.pointcloud_pub = rospy.Publisher(rospy.get_param("~output_topic_pointcloud"), PointCloud2, queue_size=1)
        self.image_publisher = rospy.Publisher(rospy.get_param("~output_topic_image"), Image_type, queue_size=1)
        
        self.depth_sub = rospy.Subscriber(rospy.get_param("~input_topic_depth"), Image_type, self.depth_callback)  
        self.image_subscriber_yolo = rospy.Subscriber(rospy.get_param("~input_topic_yolo"), Image_type, self.yolo_callback, queue_size=1)
        self.is_compressed = rospy.get_param("~is_compressed")  
        if self.is_compressed:
            self.image_subscriber_zed = rospy.Subscriber(rospy.get_param("~input_topic_zed"), CompressedImage, self.zed_callback, queue_size=1)
        else:
            self.image_subscriber_zed = rospy.Subscriber(rospy.get_param("~input_topic_zed"), Image_type, self.zed_callback, queue_size=1)
        
        self.depth_image = None


    def yolo_callback(self, msg):
        self.yolo_image = image_to_numpy(msg)
        # Process yolo_image as needed

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
        result_mask = cv2.resize(result_image.astype(np.uint8), (640, 360))
        result_image = result_image.astype(np.uint8)
        result_image = postprocess(np.reshape(result_image, (self.size, self.size))).convert('RGB')
        result_image = np.array(result_image)
        result_image = cv2.resize(result_image, (640, 360))

        blended_image = self.blend_images(original_image, result_image)
        self.image_publisher.publish(numpy_to_image(blended_image, encoding='bgr8'))

        if self.depth_image is not None:
            threading.Thread(target=generate_point_cloud, args=(result_mask, self.depth_image, self.x, self.y, self.pointcloud_pub,image_header)).start()

    def blend_images(self, img1, img2):
        if self.yolo_image is not None:
            blended_image = cv2.addWeighted(self.yolo_image, 0.8, img2, 0.2, 0)
        else:
            blended_image = cv2.addWeighted(img1, 0.6, img2, 0.4, 0)
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

