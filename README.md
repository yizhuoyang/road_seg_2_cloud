
# Road Segmentation Network ROS Integration

This repository provides the integration of a road segmentation neural network with the ROS (Robot Operating System) framework, enabling real-time road segmentation in robotic applications.

The codes will segment the drivable region and generate correspoding 3D pointcloud based on the depth input.

The networks used in this repo are accelerated by transform the pytorch model into tensorrt model.

The pytorch model (.pt) needs to transfered into .onnx using torch.onnx.export(), then the onnx is required to be transferred into .engine file using tenserrt: 
```bash
./trtexec --onnx='save_path/ddrnet.onnx' --saveEngine='save_path/ddrnet.engine' --fp16
```
This repo provides an onnx file with input shape 480*480, which can be directly used to transfer into engine file.

---

## Setup and Usage

### Prerequisites
Ensure you have the following ready:
- A ROS workspace configured (`ws_road`)
- Python interpreter installed with the required machine learning dependencies (e.g., PyTorch, OpenCV, ROS Python bindings)
- Trained road segmentation model file

### Instructions

1. **Navigate to the ROS Workspace**  
   Use the following command to move into the ROS workspace:
   ```bash
   cd ws_road
   ```

2. **Build the Workspace**  
   Compile the workspace using `catkin_make`, specifying the Python interpreter installed with the necessary dependencies:
   ```bash
   catkin_make -DPYTHON_EXECUTABLE='path_to_your_python_interpreter'
   ```
   Replace `'path_to_your_python_interpreter'` with the actual path to your Python executable.

3. **Source the Setup File**  
   Activate the workspace by sourcing the setup file:
   ```bash
   source devel/setup.bash
   ```

4. **Configuration**  
   Go to the launch file and modify the launch files accordingly.

   Edit the configuration:
   - Set the path to your trained road segmentation model.
   - Adjust input rostopics and other settings as needed.

5. **Run the Road Segmentation Node**  
   Start the road segmentation node with the following command:
   ```bash
   roslaunch road_seg road_seg_3d 
   ```
   Above is the codes to generate the segmentation results with the boundary of the road in pointcloud2 format (3D)
   
   ```bash
   roslaunch road_seg road_seg_2d 
   ```
   Above is the codes to generate the segmentation results with the boundary of the road in laserscan format (2D)
---

## Additional Notes

- **Environment Sourcing**: Ensure you source the workspace in each new terminal where you intend to run ROS nodes. This activates the necessary ROS paths and dependencies.
- **Troubleshooting**: If you encounter issues, verify that all required dependencies are installed and compatible with your ROS setup. Additionally, ensure that your Python environment includes all libraries needed for machine learning and ROS.
- **Yolo with ROS**: Can be found in this repo: https://github.com/yizhuoyang/yolov5_ros.git
- **Pretrain weights and real time inference**: a pretrained model in onnx format has been uploaded and can be found in this page.
If this RosNode also run in the same time, the running script will also privide the 3D location of the pedestrians in the pointcloud format.

For further details, consult the documentation or reach out via the repository’s issue tracker.


