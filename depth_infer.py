import sys
import argparse
import os
import cv2
import numpy as np
# import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import time
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor

import os
sys.path.append('/home/Disk/Depth-Anything/tensorrt/depth-anything-tensorrt/python/')
from depth_anything.util.transform import load_image

cuda.init()
def depth_infer(engine,img):
    with engine.create_execution_context() as context:
        input_image, (orig_h, orig_w) = load_image(img)
        start = time.time()
        input_shape = context.get_tensor_shape('input')
        output_shape = context.get_tensor_shape('output')
        h_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)
        h_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
        d_input = cuda.mem_alloc(h_input.nbytes)
        d_output = cuda.mem_alloc(h_output.nbytes)
        stream = cuda.Stream()
        np.copyto(h_input, input_image.ravel())
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
        end = time.time()
        depth = h_output
        depth = np.reshape(depth, output_shape[2:])
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        depth = cv2.resize(depth, (448,448))
        print(end-start)
        d_input.free()
        d_output.free()
        stream = None
        return depth


def obtain_road(depth):
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

    ground_pixel_coords_mask = (depth_rows >= (disparity_matrix - 10)) & \
                               (depth_rows <= (disparity_matrix + 10))

    ground_pixel_coords_rows = np.repeat(filtered_y_pred, ground_pixel_coords_mask.shape[1])[ground_pixel_coords_mask.ravel()]
    ground_pixel_coords_cols = np.tile(np.arange(depth.shape[1]), ground_pixel_coords_mask.shape[0])[ground_pixel_coords_mask.ravel()]

    ground_image = np.zeros((height,width))
    ground_image[ground_pixel_coords_rows, ground_pixel_coords_cols] =1
    return ground_image,ground_pixel_coords_rows,ground_pixel_coords_cols


