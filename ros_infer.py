import rospy
from sensor_msgs.msg import Image
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from ros_numpy.image import image_to_numpy, numpy_to_image
from sklearn.linear_model import RANSACRegressor
import time
from ultralytics import FastSAM

class ImageProcessor:
    def __init__(self, fastsam_path, depth_engine_path):
        self.fastsam_path = fastsam_path
        self.depth_engine_path = depth_engine_path
        self.model = FastSAM(self.fastsam_path)
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(self.depth_engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

    def select_two_points(self, mask):
        points = np.argwhere(mask == 1)
        if points.shape[0] < 2:
            selected_points = np.array([[400, 300], [400, 200]])
        else:
            selected_points = points[np.random.choice(points.shape[0], 2, replace=False)]
        swapped_points = selected_points[:, [1, 0]]
        return swapped_points.tolist()

    def depth_infer(self, img):
        with self.engine.create_execution_context() as trt_context:
            ctx = cuda.Context.attach()
            input_image = img
            start = time.time()
            input_shape = trt_context.get_binding_shape(0)
            output_shape = trt_context.get_binding_shape(1)
            h_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)
            h_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
            d_input = cuda.mem_alloc(h_input.nbytes)
            d_output = cuda.mem_alloc(h_output.nbytes)
            stream = cuda.Stream()
            np.copyto(h_input, input_image.ravel())
            cuda.memcpy_htod_async(d_input, h_input, stream)
            trt_context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(h_output, d_output, stream)
            stream.synchronize()
            ctx.pop()
            end = time.time()
            depth = h_output
            depth = np.reshape(depth, output_shape[1:])
            depth = depth[0]
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 0.01) * 255.0
            depth = depth.astype(np.uint8)
            depth = cv2.resize(depth, (420, 420))
            print(f"Inference time: {end - start} seconds")
            d_input.free()
            d_output.free()
            return depth

    def obtain_road(self, depth):
        disparity_map = depth
        height, width = disparity_map.shape
        max_disparity = int(np.max(disparity_map))
        v_disparity = np.zeros((height, max_disparity + 1), dtype=np.int32)
        rows = np.arange(height).reshape(-1, 1)
        np.add.at(v_disparity, (rows, disparity_map), 1)
        disparity_map = v_disparity
        disparity_map[:250, :30] = 0
        ground_threshold = 40
        ground_points_mask = disparity_map > ground_threshold
        y_coords, x_coords = np.where(ground_points_mask)
        ransac = RANSACRegressor()
        ransac.fit(x_coords.reshape(-1, 1), y_coords)
        line_x = np.arange(disparity_map.shape[1])
        line_y = ransac.predict(line_x.reshape(-1, 1))
        filtered_indices = (line_x > 30) & (line_y < 360)
        filtered_line_x = line_x[filtered_indices]
        filtered_line_y = line_y[filtered_indices]
        valid_disparity_mask = (0 <= filtered_line_x) & (filtered_line_x <= max_disparity)
        filtered_y_pred = filtered_line_y[valid_disparity_mask].astype(int)
        filtered_disparity_value = filtered_line_x[valid_disparity_mask]
        depth_rows = depth[filtered_y_pred, :]
        disparity_matrix = filtered_disparity_value[:, np.newaxis]
        ground_pixel_coords_mask = (depth_rows >= (disparity_matrix - 10)) & (depth_rows <= (disparity_matrix + 10))
        ground_pixel_coords_rows = np.repeat(filtered_y_pred, ground_pixel_coords_mask.shape[1])[ground_pixel_coords_mask.ravel()]
        ground_pixel_coords_cols = np.tile(np.arange(depth.shape[1]), ground_pixel_coords_mask.shape[0])[ground_pixel_coords_mask.ravel()]
        ground_image = np.zeros((height, width))
        ground_image[ground_pixel_coords_rows, ground_pixel_coords_cols] = 1
        return ground_image, ground_pixel_coords_rows, ground_pixel_coords_cols

    def sam_infer(self, image, ref_points):
        results = self.model(image, points=ref_points, labels=[1, 1], imgsz=448)
        return results

    def vis_mask(self, image_raw, mask):
        colored_mask = np.zeros_like(image_raw)
        colored_mask[..., 0] = mask * 255  # Red channel
        colored_mask[..., 1] = mask * 0    # Green channel
        colored_mask[..., 2] = mask * 0    # Blue channel
        alpha = 0.5
        overlay_image = image_raw.copy()
        overlay_image = (overlay_image * (1 - alpha) + colored_mask * alpha).astype(np.uint8)
        return overlay_image

    def image_callback(self, ros_image, pub):
        try:
            cv_image = image_to_numpy(ros_image)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            cv_image = cv2.resize(cv_image, (420, 420))
        except Exception as e:
            rospy.logerr(f"Error converting ROS Image to numpy array: {e}")
            return

        depth = self.depth_infer(cv_image)
        road_points, _, _ = self.obtain_road(depth)
        selected_points = self.select_two_points(road_points)
        cv_image = cv2.resize(cv_image, (448, 448))
        results = self.sam_infer(cv_image, selected_points)
        result_image = self.vis_mask(cv_image, results[0].masks.data.cpu().numpy()[0])
        result_image = numpy_to_image(result_image, encoding='bgr8')
        pub.publish(result_image)

def main():
    rospy.init_node('image_listener', anonymous=True)
    image_pub = rospy.Publisher("/processed/image", Image, queue_size=10)
    processor = ImageProcessor("/home/Disk/Depth-Anything/jupyter/FastSAM-s.engine",
                               "/home/Disk/Depth-Anything/depth_anything_v2/depth_anything_v2_vits.engine")
    rospy.Subscriber("/zed2i/zed_node/left/image_rect_color", Image, processor.image_callback, image_pub)
    rospy.spin()

if __name__ == '__main__':
    main()
